# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for tests.helpers.voice_isolation request planning and first-window gating. No server or model."""

from pathlib import Path
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
