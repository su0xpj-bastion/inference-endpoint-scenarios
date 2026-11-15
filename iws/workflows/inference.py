"""Inference: the API contract a client depends on.

Things that break silently while the endpoint keeps returning 200: usage that
doesn't add up (billing), a stream that drops usage (metering), max tokens not
enforced, stop sequences leaking, schema-invalid structured output, tool calls
with unparseable arguments, and error codes that drift.
"""
from __future__ import annotations

import json
import random
import time

import cerebras.cloud.sdk as cerebras
import jsonschema

from ..client import MAX_ATTEMPTS, parse_args
from ..core import Check, Ctx, guarded

W = "inference"

PERSON_SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                 "required": ["name", "age"], "additionalProperties": False}
WEATHER_TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.", "strict": True,
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"], "additionalProperties": False}}}


def models_list(ctx: Ctx) -> Check:
    ids = [m.id for m in ctx.client.models.list().data]
    ok = ctx.model in ids
    return Check(W, "models_list", "pass" if ok else "fail", f"{ctx.model} {'listed' if ok else 'MISSING'} "
                 f"({len(ids)} models)", metrics={"models": ids})


def usage_accounting(ctx: Ctx) -> Check:
    c = ctx.chat("Explain in two sentences why low latency matters for agents.", max_completion_tokens=64)
    problems = []
    if None in (c.prompt_tokens, c.completion_tokens, c.total_tokens):
        problems.append("usage missing")
    elif c.prompt_tokens + c.completion_tokens != c.total_tokens:
        problems.append(f"prompt {c.prompt_tokens} + completion {c.completion_tokens} != total {c.total_tokens}")
    if c.server_total is None:
        problems.append("time_info missing")
    else:
        parts = (c.queue_time or 0) + (c.prompt_time or 0) + (c.completion_time or 0)
        if parts > c.server_total * 1.05 + 1e-3:
            problems.append(f"time_info parts {parts:.4f}s exceed total {c.server_total:.4f}s")
    if not c.fingerprint:
        problems.append("system_fingerprint empty")
    return Check(W, "usage_accounting", "fail" if problems else "pass",
                 "; ".join(problems) or f"usage sums; time_info consistent; fp={c.fingerprint}",
                 metrics={"prompt": c.prompt_tokens, "completion": c.completion_tokens, "total": c.total_tokens})


def stream_usage_parity(ctx: Ctx) -> Check:
    msg = "Write a detailed paragraph explaining what a wafer-scale processor is."
    s = ctx.chat(msg, stream=True, temperature=0, seed=7, max_completion_tokens=200)
    n = ctx.chat(msg, temperature=0, seed=7, max_completion_tokens=200)
    problems = []
    if not s.usage_seen:
        problems.append("stream ended without a usage chunk")
    elif s.prompt_tokens != n.prompt_tokens:
        problems.append(f"prompt_tokens stream {s.prompt_tokens} != non-stream {n.prompt_tokens}")
    if s.finish_reason is None:
        problems.append("stream has no finish_reason")
    if s.usage_seen and s.completion_tokens and s.completion_tokens < 1:
        problems.append("completion_tokens 0 with content")
    status = "fail" if problems else "pass"
    detail = "; ".join(problems) or f"stream usage present, prompt_tokens match ({s.prompt_tokens})"
    if not problems and s.content != n.content:
        status, detail = "warn", detail + "; content differs stream vs non-stream at T=0 (seed is best-effort)"
    return Check(W, "stream_usage_parity", status, detail)


def max_tokens_length(ctx: Ctx) -> Check:
    c = ctx.chat("Write a detailed story about a GPU cluster.", max_completion_tokens=16)
    ok = c.finish_reason == "length" and (c.completion_tokens or 0) <= 16
    return Check(W, "max_tokens_length", "pass" if ok else "fail",
                 f"finish_reason={c.finish_reason} completion_tokens={c.completion_tokens} (limit 16)")


def stop_sequence(ctx: Ctx) -> Check:
    c = ctx.chat("Count from 1 to 10, separated by commas. Output only the numbers.",
                 temperature=0, stop=["6"], max_completion_tokens=64)
    ok = "6" not in c.content and c.finish_reason == "stop" and "5" in c.content
    return Check(W, "stop_sequence", "pass" if ok else "fail",
                 f"content={c.content!r} finish={c.finish_reason}")


def structured_output(ctx: Ctx) -> Check:
    c = ctx.chat("Extract the person: 'Maria is 34 years old and lives in Lisbon.'", temperature=0,
                 response_format={"type": "json_schema", "json_schema": {
                     "name": "person", "strict": True, "schema": PERSON_SCHEMA}})
    obj = json.loads(c.content)
    jsonschema.validate(obj, PERSON_SCHEMA)
    ok = obj == {"name": "Maria", "age": 34}
    return Check(W, "structured_output", "pass" if ok else "fail", f"schema-valid; value={obj}")


def tool_call(ctx: Ctx) -> Check:
    c = ctx.chat("What's the weather in Paris right now?", tools=[WEATHER_TOOL], tool_choice="required",
                 temperature=0)
    args = parse_args(c)
    ok = c.finish_reason == "tool_calls" and c.tool_calls[0]["name"] == "get_weather" and \
        str(args.get("city", "")).lower() == "paris"
    return Check(W, "tool_call", "pass" if ok else "fail",
                 f"finish={c.finish_reason} calls={c.tool_calls}")


def tool_call_streamed(ctx: Ctx) -> Check:
    c = ctx.chat("What's the weather in Tokyo?", stream=True, tools=[WEATHER_TOOL], tool_choice="required",
                 temperature=0)
    args = parse_args(c)   # raises if streamed argument fragments don't reassemble into JSON
    ok = str(args.get("city", "")).lower() == "tokyo"
    return Check(W, "tool_call_streamed", "pass" if ok else "fail", f"reassembled args={args}")


def reasoning_off(ctx: Ctx) -> Check:
    if ctx.base_params.get("reasoning_effort") != "none":
        return Check(W, "reasoning_off", "skip", "model has no reasoning_effort=none", gated=False)
    c = ctx.chat("Reply with only the word: ready", max_completion_tokens=16)
    ok = c.reasoning_tokens == 0 and not c.reasoning
    return Check(W, "reasoning_off", "pass" if ok else "fail",
                 f"reasoning_tokens={c.reasoning_tokens} reasoning_text={len(c.reasoning)} chars")


def error_contract(ctx: Ctx) -> Check:
    got = {}
    for label, fn, want in (
        ("unknown model", lambda: ctx.client.chat.completions.create(
            model="no-such-model-xyz", messages=[{"role": "user", "content": "hi"}]), {404}),
        ("temperature=3", lambda: ctx.client.chat.completions.create(
            model=ctx.model, messages=[{"role": "user", "content": "hi"}], temperature=3), {400, 422}),
        ("bad api key", lambda: ctx.bad_key_client().chat.completions.create(
            model=ctx.model, messages=[{"role": "user", "content": "hi"}]), {401}),
    ):
        for attempt in range(MAX_ATTEMPTS):
            ctx.budget.take()
            try:
                fn()
                got[label] = "200 (expected error)"
                break
            except cerebras.RateLimitError:
                got[label] = "429"
                time.sleep(2 ** attempt * 0.05 + random.random() * 0.05)
                continue
            except cerebras.APIStatusError as exc:
                got[label] = exc.status_code if exc.status_code in want else f"{exc.status_code} (want {sorted(want)})"
                break
    bad = {k: v for k, v in got.items() if not isinstance(v, int)}
    return Check(W, "error_contract", "fail" if bad else "pass", json.dumps(got))


CHECKS = (models_list, usage_accounting, stream_usage_parity, max_tokens_length, stop_sequence,
          structured_output, tool_call, tool_call_streamed, reasoning_off, error_contract)


def run(ctx: Ctx) -> list[Check]:
    return [guarded(W, fn.__name__, lambda fn=fn: fn(ctx)) for fn in CHECKS]
