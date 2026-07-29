# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E offline inference tests for Vevo2.

Vevo2 needs a reference audio clip on every request. Upstream ships a
reference Arabic-male sample at
``Amphion/models/vc/vevo/wav/arabic_male.wav``; we re-host the same WAV
on raw.githubusercontent so the test does not require a local Amphion
clone. Set ``VEVO2_LOCAL_REF=/path/to/arabic_male.wav`` to skip the
download (useful in air-gapped CI).

Marked ``slow`` + ``tts`` so it is collected by the weekly "TTS · L4"
buildkite lane (``-m "slow and L4 and tts"``) and does not gate the basic
CI matrix.
"""

from __future__ import annotations

import os
import urllib.request

import pytest
import torch
from vllm import SamplingParams

from tests.helpers.mark import hardware_test
from vllm_omni import Omni


# The published RMSnow/Vevo2 repo ships its configs in sub-folders with no
# root config.json, so loading by the bare repo id fails model_type
# resolution. ``init_vevo2_checkpoint.py`` writes the root config for a local
# checkout; point VEVO2_MODEL_PATH at that init'd directory to run this suite.
# When it is absent we skip the whole suite with a clear message instead of
# hard-failing on an opaque loader error.
def _resolve_vevo2_model() -> str:
    model = os.environ.get("VEVO2_MODEL_PATH")
    if model and os.path.isdir(model) and os.path.exists(os.path.join(model, "config.json")):
        return model
    reason = (
        "Vevo2 e2e tests require an init_vevo2_checkpoint.py-prepared local "
        "checkpoint (the bare RMSnow/Vevo2 repo ships no root config.json). "
        "Download the repo, run init_vevo2_checkpoint.py on it, and set "
        "VEVO2_MODEL_PATH=/path/to/Vevo2."
    )
    if model:
        reason = (
            f"VEVO2_MODEL_PATH={model!r} is not an initialized checkpoint dir (missing root config.json). " + reason
        )
    pytest.skip(reason, allow_module_level=True)


MODEL_NAME = _resolve_vevo2_model()
SAMPLE_RATE = 24000
# A single short prompt yields a few seconds of speech. Anything longer than
# this points to a generation/streaming bug (e.g. the AR scheduler re-running
# inference per step and concatenating the waveform hundreds of times). Keep
# it generous so legitimately long sentences don't flake, but tight enough to
# catch a runaway (the regression this guards against produced ~8600 s).
MAX_REASONABLE_DURATION_S = 120
REF_AUDIO_URL = "https://raw.githubusercontent.com/open-mmlab/Amphion/main/models/vc/vevo/wav/arabic_male.wav"
REF_TEXT = "Philip stood undecided, his ears strained to catch the slightest sound."

DEFAULT_SAMPLING = SamplingParams(
    temperature=1.0,
    top_p=0.8,
    top_k=25,
    max_tokens=4096,
    seed=42,
    detokenize=False,
)


@pytest.fixture(scope="session")
def ref_audio_path(tmp_path_factory) -> str:
    """Resolve the upstream reference clip on disk.

    Tries ``VEVO2_LOCAL_REF`` first, falls back to fetching from the
    upstream Amphion repo. Failure escalates to a hard ``pytest.fail`` so
    a broken network path can't silently mask regressions; opt into
    skipping with ``VEVO2_SKIP_ON_NET_FAIL=1`` for air-gapped runners.
    """
    local = os.environ.get("VEVO2_LOCAL_REF")
    if local and os.path.exists(local):
        return local

    cache_dir = tmp_path_factory.mktemp("vevo2_ref")
    target = cache_dir / "arabic_male.wav"
    try:
        with urllib.request.urlopen(REF_AUDIO_URL, timeout=30) as resp:
            data = resp.read()
        target.write_bytes(data)
    except Exception as e:
        msg = f"Cannot fetch upstream reference clip {REF_AUDIO_URL}: {e}"
        if os.environ.get("VEVO2_SKIP_ON_NET_FAIL"):
            pytest.skip(msg)
        pytest.fail(msg)
    if not target.exists() or os.path.getsize(target) == 0:
        pytest.fail(f"Reference clip empty after download: {target}")
    return str(target)


def _build_request(
    text: str,
    prompt_audio_path: str,
    ref_text: str = REF_TEXT,
    seed: int = 42,
) -> dict:
    additional: dict = {
        "text": [text],
        "prompt_audio_path": [prompt_audio_path],
        "ref_text": [ref_text],
        "top_k": [25],
        "top_p": [0.8],
        "temperature": [1.0],
        "flow_matching_steps": [32],
        "seed": [seed],
    }
    return {
        "prompt": "<|im_start|>assistant\n",
        "additional_information": additional,
    }


def _as_request_outputs(stage_outputs):
    """Normalise ``stage_outputs.request_output`` to a list.

    Older vllm-omni returned a list of ``RequestOutput``; 0.20+ returns a
    single ``RequestOutput``. Iterating the latter raises ``TypeError:
    'RequestOutput' object is not iterable``, so coerce defensively (the
    offline example does the same).
    """
    req_outputs = stage_outputs.request_output
    if not isinstance(req_outputs, (list, tuple)):
        req_outputs = [req_outputs]
    return req_outputs


def _collect_audio(omni: Omni, request: dict) -> tuple[torch.Tensor, int]:
    for stage_outputs in omni.generate(request, DEFAULT_SAMPLING):
        for req_output in _as_request_outputs(stage_outputs):
            mm = req_output.outputs[0].multimodal_output
            assert mm is not None, "Expected multimodal_output to be non-None"
            audio = mm.get("audio")
            if audio is None:
                audio = mm.get("model_outputs")
            assert audio is not None, "Expected 'audio' / 'model_outputs' in multimodal_output"
            if isinstance(audio, list):
                non_empty = [c for c in audio if hasattr(c, "numel") and c.numel() > 0]
                assert non_empty, "Audio chunk list was empty"
                audio = torch.cat([c.reshape(-1) for c in non_empty], dim=0)
            assert isinstance(audio, torch.Tensor), f"audio should be Tensor, got {type(audio)}"
            sr = mm.get("sr")
            if isinstance(sr, list) and sr:
                sr = sr[-1]
            return audio.cpu(), int(sr.item()) if sr is not None and hasattr(sr, "item") else SAMPLE_RATE
    raise AssertionError("No stage outputs received")


@pytest.fixture(scope="module")
def omni_engine():
    return Omni(model=MODEL_NAME, stage_init_timeout=240)


@pytest.mark.slow
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"})
def test_vevo2_english(omni_engine, ref_audio_path):
    """English zero-shot TTS produces non-empty 24 kHz audio."""
    req = _build_request("Hello, this is a short Vevo2 voice cloning demo.", ref_audio_path)
    audio, sr = _collect_audio(omni_engine, req)

    assert sr == SAMPLE_RATE, f"Expected sample_rate={SAMPLE_RATE}, got {sr}"
    assert audio.numel() > 0, "Audio tensor should not be empty"
    assert not torch.all(audio == 0), "Audio should not be all-zeros (silence)"
    assert torch.isfinite(audio).all(), "Audio should not contain NaN / Inf"
    duration_s = audio.numel() / sr
    assert duration_s <= MAX_REASONABLE_DURATION_S, (
        f"Audio is {duration_s:.0f}s for a single short prompt "
        f"(> {MAX_REASONABLE_DURATION_S}s) — likely a runaway generation bug"
    )


@pytest.mark.slow
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"})
def test_vevo2_chinese(omni_engine, ref_audio_path):
    """Chinese zero-shot TTS produces non-empty audio."""
    req = _build_request("你好，这是一段Vevo2的语音合成测试。", ref_audio_path)
    audio, sr = _collect_audio(omni_engine, req)

    assert sr == SAMPLE_RATE
    assert audio.numel() > 0
    assert not torch.all(audio == 0)
    assert torch.isfinite(audio).all()
    assert audio.numel() / sr <= MAX_REASONABLE_DURATION_S, (
        f"Audio is {audio.numel() / sr:.0f}s for a single short prompt — likely a runaway generation bug"
    )


@pytest.mark.slow
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"})
def test_vevo2_batch(omni_engine, ref_audio_path):
    """Batch of two requests returns audio for each (per-request state isolation)."""
    requests = [
        _build_request("First request, just a quick test.", ref_audio_path, seed=11),
        _build_request("Second request, also short.", ref_audio_path, seed=22),
    ]
    results = []
    # Omni.generate expects one SamplingParams per *stage* (Vevo2 is single-stage),
    # not one per request; per-request differentiation (the seeds) is carried in
    # each request's additional_information.
    for stage_outputs in omni_engine.generate(requests, DEFAULT_SAMPLING):
        for req_output in _as_request_outputs(stage_outputs):
            mm = req_output.outputs[0].multimodal_output
            assert mm is not None
            audio = mm.get("audio")
            if audio is None:
                audio = mm.get("model_outputs")
            if isinstance(audio, list):
                non_empty = [c for c in audio if hasattr(c, "numel") and c.numel() > 0]
                audio = torch.cat([c.reshape(-1) for c in non_empty], dim=0) if non_empty else None
            assert audio is not None and audio.numel() > 0
            assert audio.numel() / SAMPLE_RATE <= MAX_REASONABLE_DURATION_S, (
                f"Audio is {audio.numel() / SAMPLE_RATE:.0f}s — likely a runaway generation bug"
            )
            results.append(audio.cpu())

    assert len(results) == 2, f"Expected 2 outputs, got {len(results)}"
    # Different seeds should produce non-identical outputs (modulo rounding).
    assert results[0].shape != results[1].shape or not torch.allclose(results[0], results[1])
