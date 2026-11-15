# Agent notes

- Chat Completions only (`client.chat.completions.create` via `cerebras_cloud_sdk`). Every call goes through `ies.client.chat()`.
- Exception order everywhere: `RateLimitError` → `APIStatusError` → `APIError`; bare `raise`.
- Retries only on 429; SDK `max_retries=0` so retries are counted.
- Never force 429s or large load against the live API. Prove retry/limit behaviour on `mockserve`.
- Adding a check: add the fault that it exists to catch to `mockserve/server.py` and pin it in `tests/test_faults.py`.
- Cost: `(tokens / 1000) * rate`, prompt and completion separately (`ies/cli.py: RATES`).
- `pytest` must pass offline; live runs need `CEREBRAS_API_KEY` in `.env`.
