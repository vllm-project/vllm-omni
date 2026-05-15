"""
Offline inference regression tests: Qwen3-TTS _estimate_prompt_len.

Covers the four ref_audio input forms handled by the estimator in
examples/offline_inference/text_to_speech/qwen3_tts/end2end.py:
  - bare local path
  - file:// URI
  - http:// URL
  - data: URI
"""

import base64
import io
import sys
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

_examples_dir = str(
    Path(__file__).parent.parent.parent.parent / "examples" / "offline_inference" / "text_to_speech" / "qwen3_tts"
)
sys.path.insert(0, _examples_dir)
from end2end import _estimate_prompt_len  # noqa: E402

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


TEST_SR = 16000
TEST_DURATION_S = 3
CODEC_FRAME_RATE = 12
EXPECTED_REF_CODE_LEN = TEST_DURATION_S * CODEC_FRAME_RATE  # 36, distinct from fallback 2048


def _wav_bytes() -> bytes:
    buf = io.BytesIO()
    samples = np.zeros(TEST_SR * TEST_DURATION_S, dtype=np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TEST_SR)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


@pytest.fixture
def wav_file(tmp_path: Path) -> Path:
    path = tmp_path / "ref.wav"
    path.write_bytes(_wav_bytes())
    return path


@pytest.fixture
def estimator_cache(monkeypatch):
    """
    Bypass model loading by pre-populating the estimator cache, and shortcut
    Qwen3TTSTalker.estimate_prompt_len_from_additional_information so that
    the test directly observes the real _estimate_ref_code_len closure.
    """
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
        Qwen3TTSTalkerForConditionalGeneration,
    )

    def _shortcut(**kwargs):
        ref_audio = kwargs["additional_information"]["ref_audio"][0]
        return kwargs["estimate_ref_code_len"](ref_audio)

    monkeypatch.setattr(
        Qwen3TTSTalkerForConditionalGeneration,
        "estimate_prompt_len_from_additional_information",
        staticmethod(_shortcut),
    )

    mock_tcfg = SimpleNamespace(codec_frame_rate=CODEC_FRAME_RATE, codec_language_id=None, spk_is_dialect=None)
    return {"test_model": (MagicMock(), mock_tcfg, None)}


def _run(cache: dict, ref_audio: str) -> int | None:
    return _estimate_prompt_len(
        additional_information={"task_type": ["Base"], "ref_audio": [ref_audio]},
        model_name="test_model",
        _cache=cache,
    )


def test_bare_local_path(estimator_cache, monkeypatch, wav_file):
    """Bare local path must be loaded via load_audio, not MediaConnector."""
    from vllm.multimodal.media import MediaConnector

    def _must_not_be_called(self, path):
        pytest.fail(f"MediaConnector.fetch_audio should not be called for bare local path: {path}")

    monkeypatch.setattr(MediaConnector, "fetch_audio", _must_not_be_called)

    assert _run(estimator_cache, str(wav_file)) == EXPECTED_REF_CODE_LEN


def test_file_uri(estimator_cache, wav_file):
    """file:// URI is routed through MediaConnector, which handles it natively."""
    assert _run(estimator_cache, f"file://{wav_file}") == EXPECTED_REF_CODE_LEN


def test_http_url(estimator_cache, monkeypatch):
    """http:// URL is routed through MediaConnector.fetch_audio (mocked to avoid network)."""
    from vllm.multimodal.media import MediaConnector

    audio = np.zeros(TEST_SR * TEST_DURATION_S, dtype=np.float32)

    def _fake_fetch_audio(self, path):
        return audio, TEST_SR

    monkeypatch.setattr(MediaConnector, "fetch_audio", _fake_fetch_audio)

    assert _run(estimator_cache, "http://example.com/ref.wav") == EXPECTED_REF_CODE_LEN


def test_data_uri(estimator_cache):
    """data: URI is routed through MediaConnector, which decodes base64 natively."""
    data_uri = "data:audio/wav;base64," + base64.b64encode(_wav_bytes()).decode("ascii")
    assert _run(estimator_cache, data_uri) == EXPECTED_REF_CODE_LEN
