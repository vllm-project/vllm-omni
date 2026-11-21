# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for voice isolation request planning, fail-fast controls, and first-window gating."""

import json
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest

from tests.helpers.speaker_similarity import LABEL_CORRECT, LABEL_DISAGREE, LABEL_WRONG_BOTH_AGREE, OutputScore
from tests.helpers.voice_isolation import VoiceIsolationSuite, gate_first_window, server_mode

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _suite() -> VoiceIsolationSuite:
    return VoiceIsolationSuite(
        model="m",
        voices={"a": ("a.wav", "ref a"), "b": ("b.wav", "ref b"), "c": ("c.wav", "ref c")},
        texts=["t0", "t1", "t2", "t3"],
        n_requests=12,
        base_seed=2,
    )


def _score(idx: int, label: str) -> OutputScore:
    wrong = "b" if label == LABEL_WRONG_BOTH_AGREE else None
    return OutputScore(
        idx=idx,
        voice="a",
        duration_s=3.0,
        scored_s=2.0,
        window_short=False,
        voices=["a", "b"],
        sims={},
        argmax={},
        margin={},
        label=label,
        wrong_voice=wrong,
    )


def test_plan_requests_round_robin_with_decorrelated_texts() -> None:
    plan = _suite().plan_requests()
    assert [r["voice"] for r in plan] == ["a", "b", "c"] * 4
    assert [r["text_idx"] for r in plan] == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert [r["seed"] for r in plan] == [2000 + i for i in range(12)]
    assert all(r["text"] == f"t{r['text_idx']}" for r in plan)


def test_request_kwargs_per_source() -> None:
    suite = _suite()
    req = {"voice": "b", "seed": 7}
    urls = {"a": "data:a", "b": "data:b", "c": "data:c"}
    assert suite.request_kwargs(req, urls, "registered") == ("voice_isolation_b", {"seed": 7})
    assert suite.request_kwargs(req, urls, "inline") == (None, {"ref_audio": "data:b", "ref_text": "ref b", "seed": 7})
    assert suite.request_kwargs(req, urls, "inline_xvec") == (
        None,
        {"ref_audio": "data:b", "x_vector_only_mode": True, "seed": 7},
    )
    with pytest.raises(ValueError, match="unknown voice source"):
        suite.request_kwargs(req, urls, "speaker_embedding")


def test_server_mode() -> None:
    assert server_mode(["--trust-remote-code"]) == "async_chunk"
    assert server_mode(None) == "async_chunk"
    assert server_mode(["--trust-remote-code", "--no-async-chunk"]) == "no_async_chunk"


def test_gate_first_window_counts_only_openings_right_at_c1() -> None:
    mismatched = [_score(0, LABEL_WRONG_BOTH_AGREE), _score(2, LABEL_WRONG_BOTH_AGREE)]
    baseline = [_score(0, LABEL_CORRECT), _score(1, LABEL_CORRECT), _score(2, LABEL_DISAGREE)]
    leaks, excused = gate_first_window(mismatched, baseline)
    assert [r.idx for r in leaks] == [0]
    assert [(r.idx, base.label) for r, base in excused] == [(2, LABEL_DISAGREE)]


def test_partial_registration_fails_both_registered_batches_and_leaves_inline_running(monkeypatch, tmp_path) -> None:
    from tests.helpers import runtime

    suite = _suite()
    params = runtime.OmniServerParams(model="m", env_dict={"SPEAKER_SAMPLES_DIR": str(tmp_path), "OTHER": "keep"})
    request = SimpleNamespace(param=params)
    voice_dirs = []
    generated = []

    def server_fixture(request, run_level, lock):
        env = request.param.env_dict
        voice_dir = Path(env["SPEAKER_SAMPLES_DIR"])
        assert voice_dir.is_dir() and voice_dir != tmp_path and env["OTHER"] == "keep"
        voice_dirs.append(voice_dir)
        yield SimpleNamespace(host="127.0.0.1", port=1, model="m", log_stats=False)

    def fail_upload(url):
        (voice_dirs[0] / "partial-upload.wav").touch()
        raise RuntimeError("second upload failed")

    def generate(client, model, concurrency, urls, source):
        generated.append((source, concurrency))
        return {"audio": []}

    monkeypatch.setattr(runtime, "iter_omni_server", server_fixture)
    monkeypatch.setattr(suite, "reference_audio_urls", lambda: {})
    monkeypatch.setattr(suite, "register_voices", fail_upload)
    monkeypatch.setattr(suite, "_generate", generate)
    mode = suite.generate(request, "full_model", [("registered", 8), ("inline", 8), ("registered", 1)])

    for concurrency in (8, 1):
        assert suite.generated[mode, "registered", concurrency] == {"error": "RuntimeError: second upload failed"}
    assert generated == [("inline", 8)]
    assert suite.generated[mode, "inline", 8] == {"audio": []}
    assert request.param is params and not voice_dirs[0].exists()


def test_no_async_control_sends_once_on_timeout_and_closes_client(monkeypatch) -> None:
    import httpx
    from openai import OpenAI

    from tests.helpers import client, runtime

    suite = _suite()
    request = SimpleNamespace(param=runtime.OmniServerParams(model="m", server_args=["--no-async-chunk"]))
    received = []
    voice_dirs = []

    def time_out(http_request):
        received.append((http_request.url.path, json.loads(http_request.content), http_request.extensions["timeout"]))
        raise httpx.ReadTimeout("speech stalled", request=http_request)

    sdk = OpenAI(
        api_key="EMPTY",
        base_url="http://test/v1",
        max_retries=2,
        http_client=httpx.Client(transport=httpx.MockTransport(time_out)),
    )

    def server_fixture(request, run_level, lock):
        voice_dir = Path(request.param.env_dict["SPEAKER_SAMPLES_DIR"])
        voice_dirs.append(voice_dir)
        try:
            yield SimpleNamespace(host="127.0.0.1", port=1, model="m", log_stats=False)
        finally:
            assert sdk.is_closed()
            assert voice_dir.exists()

    monkeypatch.setattr(client, "OnlineOmniClient", lambda **kwargs: SimpleNamespace(client=sdk))
    monkeypatch.setattr(runtime, "iter_omni_server", server_fixture)
    monkeypatch.setattr(suite, "reference_audio_urls", lambda: {"a": "data:a", "b": "data:b", "c": "data:c"})
    mode = suite.generate(request, "full_model", [("inline", 1)])

    assert len(received) == 1
    route, body, deadlines = received[0]
    assert route == "/v1/audio/speech"
    assert body["seed"] == 2000 and body["ref_audio"] == "data:a" and body["ref_text"] == "ref a"
    assert deadlines["read"] == 600.0
    assert suite.generated[mode, "inline", 1]["error"].startswith("APITimeoutError:")
    assert sdk.is_closed() and not voice_dirs[0].exists()


def test_inline_c8_keeps_concurrent_requests_and_response_order(monkeypatch) -> None:
    suite = _suite()
    concurrency = 8
    barrier = Barrier(concurrency, timeout=5)

    def send(client, model, req, urls, source):
        assert source == "inline"
        if req["idx"] < concurrency:
            barrier.wait()
        return str(req["seed"]).encode()

    monkeypatch.setattr(suite, "_send", send)
    run = suite._generate(None, "m", concurrency, {}, "inline")
    assert run["requests"] == suite.plan_requests()
    assert run["audio"] == [str(req["seed"]).encode() for req in run["requests"]]
