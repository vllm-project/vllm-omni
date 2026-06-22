# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prompt construction and request validation tests for StepAudioEditX."""

from __future__ import annotations

import base64
import io
import sys
import types
import wave
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from examples.offline_inference.text_to_speech.step_audio_editx import end2end

omni_module = types.ModuleType("vllm_omni.entrypoints.omni")
omni_module.Omni = object
sys.modules.setdefault("vllm_omni.entrypoints.omni", omni_module)


pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

MODEL = "stepfun-ai/Step-Audio-EditX"
AUDIO_TOKENIZER = "stepfun-ai/Step-Audio-Tokenizer"
STAGE_CONFIG = "vllm_omni/deploy/step_audio_editx.yaml"
REF_AUDIO = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
REF_TEXT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it!"
DEFAULT_TEXT = "Please review the document before we begin."
ONLINE_REF_TEXT = "Good one. Okay, fine, I'm just gonna leave this here. Goodbye."


def _offline_args(**overrides):
    base = dict(
        model=MODEL,
        audio_tokenizer=AUDIO_TOKENIZER,
        deploy_config=STAGE_CONFIG,
        edit_type="clone",
        edit_info=None,
        text=DEFAULT_TEXT,
        ref_text=REF_TEXT,
        ref_audio=REF_AUDIO,
        output=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _create_dummy_audio(sample_rate: int = 16000, duration_sec: float = 1.0) -> tuple[np.ndarray, int]:
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * 440 * t), sample_rate


def _create_dummy_audio_base64(duration_sec: float = 2.0, sample_rate: int = 16000) -> str:
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples, dtype=np.float32)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _create_ref_audio_data_url(duration_sec: float = 2.0) -> str:
    return f"data:audio/wav;base64,{_create_dummy_audio_base64(duration_sec=duration_sec)}"


def _create_speech_request(
    *,
    edit_type: str = "clone",
    edit_info: str | None = None,
    text: str | None = None,
    stream: bool = False,
) -> dict:
    if text is None:
        text = DEFAULT_TEXT if edit_type in {"clone", "paralinguistic"} else ""
    payload = {
        "model": MODEL,
        "input": text,
        "voice": "step_audio_editx",
        "response_format": "pcm" if stream else "wav",
        "stream": stream,
        "ref_audio": _create_ref_audio_data_url(),
        "ref_text": ONLINE_REF_TEXT,
        "max_new_tokens": 256,
        "extra_params": {"edit_type": edit_type},
    }
    if edit_info is not None:
        payload["extra_params"]["edit_info"] = edit_info
    return payload


def test_offline_build_inputs_clone_uses_duration_based_prompt_len() -> None:
    with patch.object(end2end, "estimate_step_audio_editx_prompt_len", return_value=123) as estimate:
        inputs = end2end._build_inputs(_offline_args())

    assert len(inputs) == 1
    assert inputs[0]["prompt_token_ids"] == [0] * 123
    additional_information = inputs[0]["additional_information"]
    assert additional_information == {
        "edit_type": "clone",
        "ref_audio": [REF_AUDIO],
        "ref_text": [REF_TEXT],
        "text": [DEFAULT_TEXT],
    }
    estimate.assert_called_once_with(additional_information, MODEL)


def test_offline_build_inputs_edit_includes_edit_info_and_allows_empty_text() -> None:
    with patch.object(end2end, "estimate_step_audio_editx_prompt_len", return_value=77):
        inputs = end2end._build_inputs(
            _offline_args(
                edit_type="emotion",
                edit_info="angry",
                text=None,
            )
        )

    assert inputs[0]["prompt_token_ids"] == [0] * 77
    assert inputs[0]["additional_information"]["edit_type"] == "emotion"
    assert inputs[0]["additional_information"]["edit_info"] == "angry"
    assert inputs[0]["additional_information"]["text"] == [""]


def test_offline_build_inputs_requires_ref_text_when_ref_audio_is_explicit() -> None:
    with pytest.raises(ValueError, match="ref_text must be provided"):
        end2end._build_inputs(_offline_args(ref_text=None))


def test_offline_dummy_audio_shape() -> None:
    audio, sr = _create_dummy_audio(duration_sec=0.25)

    assert sr == 16000
    assert audio.dtype == np.float32
    assert audio.shape == (4000,)


def test_online_dummy_audio_base64_is_wav() -> None:
    raw = base64.b64decode(_create_dummy_audio_base64(duration_sec=0.25))

    with wave.open(io.BytesIO(raw), "rb") as wav_file:
        assert wav_file.getframerate() == 16000
        assert wav_file.getnchannels() == 1
        assert wav_file.getnframes() == 4000


@pytest.mark.parametrize(
    ("edit_type", "edit_info"),
    [
        ("clone", None),
        ("emotion", "angry"),
        ("style", "sweet"),
        ("paralinguistic", "laughter"),
        ("denoise", None),
    ],
)
def test_online_create_speech_request_shape(edit_type: str, edit_info: str | None) -> None:
    payload = _create_speech_request(edit_type=edit_type, edit_info=edit_info)

    if edit_type in {"clone", "paralinguistic"}:
        assert payload["input"]
    else:
        assert payload["input"] == ""
    assert payload["ref_audio"].startswith("data:audio/wav;base64,")
    assert payload["ref_text"]
    assert payload["extra_params"]["edit_type"] == edit_type
    if edit_info is None:
        assert "edit_info" not in payload["extra_params"]
    else:
        assert payload["extra_params"]["edit_info"] == edit_info
