"""Cerebras-compatible mock inference server with fault injection.

Speaks the same wire format as api.cerebras.ai (/v1/chat/completions, /v1/models,
SSE streaming, usage + time_info, system_fingerprint, x-ratelimit-* headers,
error bodies), so the harness runs unchanged against it or the live API.

It exists so every workflow can be proven to CATCH the regression it claims to
guard against: each fault below is a real failure class, and tests/test_faults.py
pins which check goes red for which fault.

Faults (set on the server object, or MOCK_FAULTS=a,b for `python -m mockserve`):
  slow_ttft          +400 ms before the first token (scheduler / queue regression)
  slow_decode        decode at 200 tok/s instead of 2,000 (kernel regression)
  usage_drift        total_tokens != prompt + completion (billing bug)
  stream_no_usage    final stream chunk drops usage (metering bug)
  ignore_max_tokens  max_completion_tokens not enforced
  midstream_error    SSE error event after a few chunks on long outputs
  nondeterministic   temperature 0 + seed is not reproducible (different text)
  logprob_noise      same tokens, logprobs drift ~1e-2 between runs (non-associative
                     reductions / changed kernel: numerics move before argmax does)
  regress_math       arithmetic golden answers are off by one (bad model build)
  fingerprint_v2     system_fingerprint changes (new backend release)
  flaky_answers      one golden case alternates right / wrong
  contention         latency grows with in-flight requests (no batching headroom)
  crosstalk          concurrent requests can receive another request's answer
  rate_limit_light   every 3rd request gets 429 + retry-after; its retry succeeds
  rate_limit_storm   4 of 5 requests get 429 (retries exhaust)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

API_KEY = "mock-key"
MODELS = ("qwen-3.8-27b", "gpt-oss-120b", "gemma-4-31b")
NO_REASONING_OK = {"qwen-3.8-27b", "gemma-4-31b"}
DECODE_TPS = 2000.0
SLOW_DECODE_TPS = 200.0
PREFILL_TPS = 200_000.0
CHUNK_TOKENS = 4
CACHE_BLOCK = 128
RPM_LIMIT = 300
VOCAB = ("wafer", "scale", "token", "latency", "kernel", "memory", "fabric", "batch", "decode",
         "prefill", "model", "cluster", "router", "tensor", "stream", "cache", "sparse", "weight",
         "compiler", "runtime", "the", "a", "of", "and", "to", "in", "is", "that", "for", "with")
TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def ntokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def _stable(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def _is_prime(n: int) -> bool:
    return n > 1 and all(n % d for d in range(2, int(math.isqrt(n)) + 1))


class State:
    def __init__(self, faults: set[str] | None = None) -> None:
        self.faults: set[str] = set(faults or ())
        self.lock = threading.Lock()
        self.requests = 0
        self.inflight: dict[str, str] = {}
        self.cache: set[str] = set()
        self.limited: set[str] = set()
        self.flaky_flip = 0
        self.window: list[float] = []

    def next_request(self) -> int:
        with self.lock:
            self.requests += 1
            now = time.monotonic()
            self.window = [t for t in self.window if now - t < 60] + [now]
            return self.requests


def golden_answer(prompt: str, state: State) -> str | None:
    """A 'correct model' for checkable prompts. Faults can corrupt it."""
    if m := re.search(r"What is (\d+) \* (\d+)", prompt):
        v = int(m[1]) * int(m[2])
        return str(v + 1 if "regress_math" in state.faults else v)
    if m := re.search(r"What is (\d+) \+ (\d+)", prompt):
        v = int(m[1]) + int(m[2])
        return str(v + 1 if "regress_math" in state.faults else v)
    if m := re.search(r"number that comes after (\d+)", prompt):
        return str(int(m[1]) + 1)
    if m := re.search(r"Is (\d+) a prime number", prompt):
        return "yes" if _is_prime(int(m[1])) else "no"
    if m := re.search(r"Sort these numbers ascending[^:]*: ([\d, ]+)", prompt):
        return ", ".join(str(x) for x in sorted(int(t) for t in m[1].split(",")))
    if "capital of Australia" in prompt:
        if "flaky_answers" in state.faults:
            with state.lock:
                state.flaky_flip += 1
                if state.flaky_flip % 2 == 0:
                    return "Sydney"
        return "Canberra"
    if m := re.search(r"Repeat exactly: (.+)$", prompt):
        return m[1].strip()
    if m := re.search(r"Count from (\d+) to (\d+)", prompt):
        return ", ".join(str(i) for i in range(int(m[1]), int(m[2]) + 1))
    return None


def generate(body: dict, state: State, rid: str) -> dict:
    msgs = body["messages"]
    prompt = str(msgs[-1].get("content", ""))
    seed = body.get("seed")
    temp = body.get("temperature", 1.0)
    out: dict = {"content": "", "tool_calls": None, "reasoning": ""}

    # crosstalk: answer some OTHER in-flight request's prompt
    if "crosstalk" in state.faults:
        with state.lock:
            others = [p for k, p in state.inflight.items() if k != rid]
        if others:
            prompt = others[0]

    tools = body.get("tools") or []
    if tools and body.get("tool_choice", "auto") != "none" and (
            "weather" in prompt.lower() or body.get("tool_choice") == "required"):
        city = (re.search(r"in ([A-Z][a-zA-Z]+)", prompt) or [None, "Paris"])[1]
        out["tool_calls"] = [{"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                              "function": {"name": tools[0]["function"]["name"],
                                           "arguments": json.dumps({"city": city})}}]
        return out

    rf = body.get("response_format") or {}
    if rf.get("type") in ("json_schema", "json_object"):
        m = re.search(r"([A-Z][a-z]+) is (\d+) years old", prompt)
        out["content"] = json.dumps({"name": m[1], "age": int(m[2])} if m else {"result": "ok"})
        return out

    ans = golden_answer(prompt, state)
    if ans is not None:
        out["content"] = ans
    else:
        long_form = any(w in prompt.lower() for w in ("essay", "story", "detailed", "explain"))
        n = 400 if long_form else 30
        reproducible = (temp == 0 or seed is not None) and "nondeterministic" not in state.faults
        rng = random.Random(_stable(prompt, seed, temp) if reproducible else None)
        out["content"] = " ".join(rng.choice(VOCAB) for _ in range(n)) + "."

    effort = body.get("reasoning_effort")
    model = body.get("model")
    if not (effort == "none" and model in NO_REASONING_OK):
        out["reasoning"] = "Let me think about this step by step before answering."
    return out


def apply_limits(text: str, body: dict, state: State, reasoning_tokens: int = 0) -> tuple[str, str]:
    """max_completion_tokens covers reasoning + content, as the real API documents."""
    finish = "stop"
    for s in ([body["stop"]] if isinstance(body.get("stop"), str) else body.get("stop") or []):
        if s and s in text:
            text = text[: text.index(s)]
    limit = body.get("max_completion_tokens") or body.get("max_tokens")
    toks = TOKEN_RE.findall(text)
    room = None if not limit else max(0, limit - reasoning_tokens)
    if room is not None and len(toks) > room and "ignore_max_tokens" not in state.faults:
        text = " ".join(toks[:room])
        finish = "length"
    return text, finish


def logprob_entries(tokens: list[str], seed: object, state: State, top: int) -> list[dict]:
    out = []
    noise = random.Random() if "logprob_noise" in state.faults else None
    for i, t in enumerate(tokens):
        lp = -((_stable(t, i, seed) % 1000) / 1000.0) * 2.0
        if noise:
            lp += noise.uniform(-1e-2, 1e-2)
        alts = [{"token": t, "logprob": lp, "bytes": list(t.encode())}]
        for j in range(1, top):
            alt = VOCAB[_stable(t, i, j) % len(VOCAB)]
            alts.append({"token": alt, "logprob": lp - j, "bytes": list(alt.encode())})
        out.append({"token": t, "logprob": lp, "bytes": list(t.encode()), "top_logprobs": alts[:top]})
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "mock-cerebras/1.0"
    state: State  # injected

    def log_message(self, *args: object) -> None:  # quiet
        pass

    # ---------- helpers
    def _json(self, code: int, obj: dict, headers: dict | None = None) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _error(self, code: int, message: str, etype: str, param: str | None = None,
               headers: dict | None = None) -> None:
        self._json(code, {"error": {"message": message, "type": etype, "param": param,
                                    "code": etype}}, headers)

    def _authed(self) -> bool:
        if self.headers.get("authorization", "") != f"Bearer {API_KEY}":
            self._error(401, "Wrong API Key", "invalid_request_error")
            return False
        return True

    def _rl_headers(self) -> dict:
        st = self.state
        with st.lock:
            used = len(st.window)
        return {"x-ratelimit-limit-requests-minute": str(RPM_LIMIT),
                "x-ratelimit-remaining-requests-minute": str(max(0, RPM_LIMIT - used)),
                "x-ratelimit-limit-tokens-minute": "150000",
                "x-ratelimit-remaining-tokens-minute": str(max(0, 150000 - used * 200))}

    # ---------- routes
    def do_GET(self) -> None:
        if self.path.startswith("/v1/tcp_warming"):
            return self._json(200, {})
        if not self._authed():
            return
        if self.path.rstrip("/") == "/v1/models":
            return self._json(200, {"object": "list", "data": [
                {"id": m, "object": "model", "created": 0, "owned_by": "Cerebras"} for m in MODELS]})
        self._error(404, f"no route {self.path}", "not_found_error")

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            return self._error(404, f"no route {self.path}", "not_found_error")
        if not self._authed():
            return
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        st = self.state
        n = st.next_request()

        pkey = hashlib.sha256(json.dumps(body.get("messages"), sort_keys=True).encode()).hexdigest()
        with st.lock:
            retried = pkey in st.limited          # light: a retried request is let through
            st.limited.discard(pkey)
            light = "rate_limit_light" in st.faults and n % 3 == 0 and not retried
            if light:
                st.limited.add(pkey)
        if light or ("rate_limit_storm" in st.faults and n % 5 != 0):
            return self._error(429, "Tokens per minute limit exceeded - too many uncached tokens "
                               "processed.", "too_many_tokens_error",
                               headers={"retry-after": "0.05", **self._rl_headers()})

        model = body.get("model")
        if model not in MODELS:
            return self._error(404, f"Model {model} does not exist or you do not have access to it.",
                               "not_found_error", "model")
        t = body.get("temperature")
        if t is not None and not 0 <= t <= 2:
            return self._error(400, "temperature must be between 0 and 2", "invalid_request_error",
                               "temperature")
        if body.get("n", 1) != 1:
            return self._error(400, "n > 1 is not supported", "invalid_request_error", "n")
        if not body.get("messages"):
            return self._error(400, "messages must be non-empty", "invalid_request_error", "messages")

        rid = uuid.uuid4().hex
        with st.lock:
            st.inflight[rid] = str(body["messages"][-1].get("content", ""))
        try:
            self._complete(body, rid)
        finally:
            with st.lock:
                st.inflight.pop(rid, None)

    def _complete(self, body: dict, rid: str) -> None:
        st = self.state
        t0 = time.perf_counter()
        created = int(time.time())
        fp = "fp_mock_v2" if "fingerprint_v2" in st.faults else "fp_mock_v1"
        msgs = body["messages"]
        all_text = "\n".join(str(m.get("content") or "") for m in msgs)
        prompt_tokens = ntokens(all_text) + 3 * len(msgs)
        prefix = "\n".join(str(m.get("content") or "") for m in msgs[:-1])
        key = (body.get("prompt_cache_key") or "") + "|" + hashlib.sha256(prefix.encode()).hexdigest()
        cached = 0
        if prefix:
            with st.lock:
                if key in st.cache:
                    cached = (ntokens(prefix) // CACHE_BLOCK) * CACHE_BLOCK
                st.cache.add(key)

        gen = generate(body, st, rid)
        reasoning_tokens = ntokens(gen["reasoning"])
        text, finish = apply_limits(gen["content"], body, st, reasoning_tokens)
        tool_calls = gen["tool_calls"]
        if tool_calls:
            finish = "tool_calls"
        out_tokens = TOKEN_RE.findall(text)
        completion_tokens = len(out_tokens) + reasoning_tokens + (8 if tool_calls else 0)

        with st.lock:
            inflight = len(st.inflight)
        queue = 0.0005 + (0.08 * inflight if "contention" in st.faults else 0.0)
        prompt_time = (prompt_tokens - cached) / PREFILL_TPS
        pre = queue + prompt_time + (0.4 if "slow_ttft" in st.faults else 0.0)
        tps = SLOW_DECODE_TPS if "slow_decode" in st.faults else DECODE_TPS
        time.sleep(pre)

        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                 "total_tokens": prompt_tokens + completion_tokens + (1 if "usage_drift" in st.faults else 0),
                 "prompt_tokens_details": {"cached_tokens": cached},
                 "completion_tokens_details": {"reasoning_tokens": reasoning_tokens}}
        lp = None
        if body.get("logprobs"):
            lp = {"content": logprob_entries(out_tokens, body.get("seed"), st,
                                             int(body.get("top_logprobs") or 1))}
        base = {"id": f"chatcmpl-{rid}", "created": created, "model": body["model"],
                "system_fingerprint": fp}

        def time_info(decode: float) -> dict:
            total = time.perf_counter() - t0
            return {"queue_time": queue, "prompt_time": prompt_time, "completion_time": decode,
                    "total_time": total, "created": time.time()}

        if not body.get("stream"):
            d0 = time.perf_counter()
            time.sleep(completion_tokens / tps)
            msg = {"role": "assistant", "content": None if tool_calls else text, "tool_calls": tool_calls}
            if gen["reasoning"]:
                msg["reasoning"] = gen["reasoning"]
            return self._json(200, {**base, "object": "chat.completion", "choices": [
                {"index": 0, "finish_reason": finish, "message": msg, "logprobs": lp}],
                "usage": usage, "time_info": time_info(time.perf_counter() - d0)}, self._rl_headers())

        # ---- SSE stream
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        for k, v in self._rl_headers().items():
            self.send_header(k, v)
        self.end_headers()

        def send(obj: dict) -> None:
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()

        chunk = {**base, "object": "chat.completion.chunk"}
        d0 = time.perf_counter()
        send({**chunk, "choices": [{"index": 0, "delta": {"role": "assistant"}}]})
        if gen["reasoning"]:
            time.sleep(reasoning_tokens / tps)
            send({**chunk, "choices": [{"index": 0, "delta": {"reasoning": gen["reasoning"]}}]})
        if tool_calls:
            tc = tool_calls[0]
            send({**chunk, "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": tc["id"], "type": "function",
                 "function": {"name": tc["function"]["name"], "arguments": ""}}]}}]})
            send({**chunk, "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "type": "function", "function": {"arguments": tc["function"]["arguments"]}}]}}]})
        else:
            pieces = re.findall(r"\S+\s*", text) or [text]
            lp_items = (lp or {}).get("content") or []
            for i in range(0, len(pieces), CHUNK_TOKENS):
                if "midstream_error" in st.faults and i >= 3 * CHUNK_TOKENS and len(pieces) > 100:
                    send({"error": {"message": "internal error: decode worker lost",
                                    "type": "server_error", "code": "server_error"}})
                    return
                part = "".join(pieces[i:i + CHUNK_TOKENS])
                time.sleep(ntokens(part) / tps)
                c = {"index": 0, "delta": {"content": part}}
                if lp_items:
                    c["logprobs"] = {"content": lp_items[i:i + CHUNK_TOKENS]}
                send({**chunk, "choices": [c]})
        final: dict = {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                       "time_info": time_info(time.perf_counter() - d0)}
        if "stream_no_usage" not in st.faults:
            final["usage"] = usage
        send(final)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def serve(port: int = 0, faults: set[str] | None = None) -> tuple[ThreadingHTTPServer, State]:
    """Start in a daemon thread. Returns (server, state); base URL is http://127.0.0.1:<port>."""
    state = State(faults)
    handler = type("BoundHandler", (Handler,), {"state": state})
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, state


def main() -> None:
    port = int(os.environ.get("MOCK_PORT", "8089"))
    faults = {f for f in os.environ.get("MOCK_FAULTS", "").split(",") if f}
    srv, _ = serve(port, faults)
    print(f"mock cerebras on http://127.0.0.1:{srv.server_address[1]} faults={sorted(faults)}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
