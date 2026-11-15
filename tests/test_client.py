"""Client behaviour (retry policy, capture) and the pure helpers."""
from __future__ import annotations

import math

import cerebras.cloud.sdk as cerebras
import pytest

from iws import cli
from iws.client import MAX_ATTEMPTS, Budget, chat, make_client
from iws.core import pct, tbt
from mockserve.server import API_KEY


def test_429_is_retried_honouring_retry_after(mock):
    base_url, state = mock
    state.faults = {"rate_limit_light"}
    client = make_client(API_KEY, base_url)
    attempts = [chat(client, "qwen-3.8-27b", [{"role": "user", "content": f"q{i}"}]).attempts for i in range(6)]
    assert max(attempts) == 2 and attempts.count(2) == 2      # every 3rd request retried once


def test_429_storm_raises_after_max_attempts(mock):
    base_url, state = mock
    state.faults = {"rate_limit_storm"}
    state.requests = 0
    with pytest.raises(cerebras.RateLimitError):
        chat(make_client(API_KEY, base_url), "qwen-3.8-27b", [{"role": "user", "content": "x"}])
    assert state.requests == MAX_ATTEMPTS


def test_non_429_errors_are_not_retried(mock):
    base_url, state = mock
    state.requests = 0
    with pytest.raises(cerebras.NotFoundError):
        chat(make_client(API_KEY, base_url), "nope", [{"role": "user", "content": "x"}])
    assert state.requests == 1


def test_midstream_error_raises_instead_of_truncating(mock):
    base_url, state = mock
    state.faults = {"midstream_error"}
    with pytest.raises(cerebras.APIStatusError):
        chat(make_client(API_KEY, base_url), "qwen-3.8-27b",
             [{"role": "user", "content": "Write a detailed essay."}], stream=True)


def test_stream_captures_usage_time_info_and_headers(mock):
    base_url, _ = mock
    c = chat(make_client(API_KEY, base_url), "qwen-3.8-27b",
             [{"role": "user", "content": "Write a detailed essay."}], stream=True, max_completion_tokens=50)
    assert c.usage_seen and c.completion_tokens == 50 and c.finish_reason == "length"
    assert c.completion_time and c.ttft is not None and c.fingerprint == "fp_mock_v1"
    assert "x-ratelimit-remaining-requests-minute" in c.headers


def test_budget_caps_requests():
    b = Budget(2)
    b.take()
    b.take()
    with pytest.raises(RuntimeError):
        b.take()


def test_cost_is_per_thousand_split_by_type():
    # 1,000 prompt + 2,000 completion tokens of qwen-3.8-27b at $0.99 / $1.49 per M
    assert cli.cost("qwen-3.8-27b", 1000, 2000) == pytest.approx(0.00099 + 2 * 0.00149)
    assert cli.cost("unknown", 1, 1) is None


def test_pct_nearest_rank_and_tbt():
    assert pct([5, 1, 3, 2, 4], 50) == 3 and pct([1, 2, 3, 4], 95) == 4
    assert math.isnan(pct([], 50))
    assert tbt([0.1, 0.2, 0.4]) == pytest.approx(0.15) and tbt([0.1]) == 0.0


def test_cli_run_mock_exit_codes(tmp_path):
    assert cli.main(["run", "--target", "mock", "-w", "inference", "--out", str(tmp_path)]) == 0
    assert cli.main(["run", "--target", "mock", "-w", "inference", "--faults", "usage_drift",
                     "--out", str(tmp_path)]) == 1
    assert "usage_accounting" in (tmp_path / "report.md").read_text()


def test_cli_live_without_key_is_a_clean_error(monkeypatch, tmp_path):
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    assert cli.main(["run", "--target", "live", "--out", str(tmp_path)]) == 2
