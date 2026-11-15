"""One Chat Completions call, timed and fully captured.

Every workflow goes through `chat()`, so retry policy, timing, and usage capture
are identical across scenarios. The SDK's own retries are disabled
(max_retries=0) so the retry loop here is the one being measured and counted.
"""
from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass, field

import cerebras.cloud.sdk as cerebras

MAX_ATTEMPTS = 3
MAX_RETRY_AFTER = 30.0


class Budget:
    """Hard cap on requests per run, so a bug in a sweep can't burn quota."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0
        self._lock = threading.Lock()

    def take(self) -> None:
        with self._lock:
            if self.used >= self.limit:
                raise RuntimeError(f"request budget exhausted ({self.limit}); raise --max-requests")
            self.used += 1


@dataclass
class Call:
    content: str = ""
    reasoning: str = ""
    finish_reason: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    queue_time: float | None = None
    prompt_time: float | None = None
    completion_time: float | None = None
    server_total: float | None = None
    fingerprint: str = ""
    wall: float = 0.0
    ttft: float | None = None
    content_times: list[float] = field(default_factory=list)
    chunks: int = 0
    attempts: int = 1
    usage_seen: bool = False
    headers: dict[str, str] = field(default_factory=dict)
    logprobs: list[tuple[str, float]] = field(default_factory=list)


def make_client(api_key: str | None, base_url: str | None) -> cerebras.Cerebras:
    return cerebras.Cerebras(api_key=api_key, base_url=base_url, max_retries=0,
                             warm_tcp_connection=False, timeout=120.0)


def _backoff(exc: cerebras.RateLimitError, attempt: int) -> float:
    ra = exc.response.headers.get("retry-after") if exc.response is not None else None
    if ra:
        try:
            return min(float(ra), MAX_RETRY_AFTER)
        except ValueError:
            pass
    return 2 ** attempt + random.random()


def _take_usage(call: Call, usage: object, time_info: object) -> None:
    if usage is not None:
        call.usage_seen = True
        call.prompt_tokens = usage.prompt_tokens
        call.completion_tokens = usage.completion_tokens
        call.total_tokens = usage.total_tokens
        if usage.prompt_tokens_details is not None:
            call.cached_tokens = usage.prompt_tokens_details.cached_tokens or 0
        if usage.completion_tokens_details is not None:
            call.reasoning_tokens = usage.completion_tokens_details.reasoning_tokens or 0
    if time_info is not None:
        call.queue_time = time_info.queue_time
        call.prompt_time = time_info.prompt_time
        call.completion_time = time_info.completion_time
        call.server_total = time_info.total_time


def _once(client: cerebras.Cerebras, model: str, messages: list[dict], stream: bool,
          params: dict) -> Call:
    call = Call()
    t0 = time.perf_counter()
    if not stream:
        raw = client.chat.completions.with_raw_response.create(model=model, messages=messages, **params)
        resp = raw.parse()
        call.wall = time.perf_counter() - t0
        call.headers = {k.lower(): v for k, v in raw.headers.items() if k.lower().startswith("x-ratelimit")}
        call.fingerprint = resp.system_fingerprint
        choice = resp.choices[0]
        call.content = choice.message.content or ""
        call.reasoning = choice.message.reasoning or ""
        call.finish_reason = choice.finish_reason
        for tc in choice.message.tool_calls or []:
            call.tool_calls.append({"name": tc.function.name, "arguments": tc.function.arguments})
        if choice.logprobs is not None and choice.logprobs.content:
            call.logprobs = [(c.token, c.logprob) for c in choice.logprobs.content]
        _take_usage(call, resp.usage, resp.time_info)
        return call

    raw = client.chat.completions.with_raw_response.create(
        model=model, messages=messages, stream=True, stream_options={"include_usage": True}, **params)
    call.headers = {k.lower(): v for k, v in raw.headers.items() if k.lower().startswith("x-ratelimit")}
    parts: list[str] = []
    reasoning: list[str] = []
    tools: dict[int, dict] = {}
    with raw.parse() as stream:
        for chunk in stream:
            call.chunks += 1
            call.fingerprint = chunk.system_fingerprint or call.fingerprint
            if chunk.choices and chunk.choices[0].delta is not None:
                delta = chunk.choices[0].delta
                if delta.content:
                    now = time.perf_counter() - t0
                    call.ttft = now if call.ttft is None else call.ttft
                    call.content_times.append(now)
                    parts.append(delta.content)
                if delta.reasoning:
                    reasoning.append(delta.reasoning)
                for tc in delta.tool_calls or []:
                    slot = tools.setdefault(tc.index or 0, {"name": "", "arguments": ""})
                    if tc.function is not None:
                        slot["name"] += tc.function.name or ""
                        slot["arguments"] += tc.function.arguments or ""
                    if call.ttft is None:
                        call.ttft = time.perf_counter() - t0
            if chunk.choices and chunk.choices[0].logprobs is not None:
                call.logprobs += [(c.token, c.logprob) for c in chunk.choices[0].logprobs.content or []]
            if chunk.choices and chunk.choices[0].finish_reason:
                call.finish_reason = chunk.choices[0].finish_reason
            if chunk.usage or chunk.time_info:
                _take_usage(call, chunk.usage, chunk.time_info)
    call.wall = time.perf_counter() - t0
    call.content = "".join(parts)
    call.reasoning = "".join(reasoning)
    call.tool_calls = [tools[i] for i in sorted(tools)]
    return call


def chat(client: cerebras.Cerebras, model: str, messages: list[dict], *, stream: bool = False,
         budget: Budget | None = None, **params: object) -> Call:
    """Chat Completions with jittered / retry-after backoff on 429 only."""
    for attempt in range(MAX_ATTEMPTS):
        if budget is not None:
            budget.take()
        try:
            call = _once(client, model, messages, stream, params)
            call.attempts = attempt + 1
            return call
        except cerebras.RateLimitError as exc:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            time.sleep(_backoff(exc, attempt))
        except cerebras.APIStatusError:
            raise
        except cerebras.APIError:
            raise
    raise RuntimeError("retry loop exited without return or raise")


def parse_args(call: Call) -> dict:
    return json.loads(call.tool_calls[0]["arguments"]) if call.tool_calls else {}
