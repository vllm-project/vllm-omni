# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E offline inference tests for TADA TTS voice cloning.

TADA is a voice-cloning TTS that walks the input text and requires a reference audio +
transcript. The test synthesises an English sentence in the reference voice, in both the batch
(sync) and async-chunk (streaming) deployments, and checks the generated audio transcribes back
to the input text (Whisper).
"""

import os
import tempfile
from pathlib import Path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

import numpy as np
import pytest
import soundfile as sf

import vllm_omni
from tests.helpers.assertions import assert_audio_speech_response
from tests.helpers.mark import hardware_test
from tests.helpers.media import convert_audio_file_to_text, get_asset_path
from tests.helpers.runtime import OmniResponse
from vllm_omni.model_executor.models.tada_tts import prompt_utils

MODEL = os.environ.get("TADA_MODEL_PATH", "HumeAI/tada-1b")

# Reference voice: a real English speech clip plus its transcript.
REFERENCE_PROMPT_WAV_PATH = get_asset_path("tada_tts/ljspeech_en.wav")
REF_TEXT = (
    "The examination and testimony of the experts, enabled the commission to conclude "
    "that five shots may have been fired."
)

_STAGE_CONFIG_DIR = Path(vllm_omni.__file__).parent / "model_executor" / "stage_configs"

DEPLOY_MODES = [
    pytest.param(str(_STAGE_CONFIG_DIR / "tada_tts.yaml"), id="sync"),
    pytest.param(str(_STAGE_CONFIG_DIR / "tada_tts_async_chunk.yaml"), id="async_chunk"),
]


def _load_ref_audio() -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(REFERENCE_PROMPT_WAV_PATH), dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def _concat_audio(audio_val) -> np.ndarray:
    import torch

    if isinstance(audio_val, list):
        tensors = [torch.as_tensor(t).float().reshape(-1) for t in audio_val if t is not None]
        if not tensors:
            return np.zeros((0,), dtype=np.float32)
        return torch.cat(tensors, dim=-1).cpu().numpy().astype(np.float32, copy=False)
    if isinstance(audio_val, torch.Tensor):
        return audio_val.float().cpu().numpy().reshape(-1)
    return np.asarray(audio_val, dtype=np.float32).reshape(-1)


def _extract_sample_rate(audio_mm: dict) -> int:
    sr_raw = audio_mm.get("sr", 24000)
    if isinstance(sr_raw, list):
        sr_raw = sr_raw[-1] if sr_raw else 24000
    if hasattr(sr_raw, "item"):
        return int(sr_raw.item())
    return int(sr_raw)


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("stage_config", DEPLOY_MODES)
def test_offline_voice_clone_en(stage_config: str) -> None:
    """
    Voice-cloning offline inference, sync and async-chunk deployments.
    Deploy Setting: tada_tts.yaml / tada_tts_async_chunk.yaml
    Input Modal: text + ref_audio + ref_text
    Output Modal: audio
    """
    synth_text = "Please call Stella and ask her to bring these things from the store."
    prompt_audio = str(REFERENCE_PROMPT_WAV_PATH)
    from tests.helpers.runtime import OmniRunner

    with OmniRunner(MODEL, stage_configs_path=stage_config, stage_init_timeout=600) as omni_runner:
        prompt, walk_len = prompt_utils.build_voice_clone_prompt(synth_text, prompt_audio, REF_TEXT, MODEL)
        sampling_params_list = prompt_utils.apply_walk_sampling_params(
            omni_runner.get_default_sampling_params_list(), walk_len
        )
        outputs = omni_runner.omni.generate([prompt], sampling_params_list=sampling_params_list)

        assert outputs, "No outputs returned"
        audio_mm = outputs[0].multimodal_output
        assert "audio" in audio_mm, "No audio output found"
        audio = _concat_audio(audio_mm["audio"])
        assert audio.size > 0, "Generated audio is empty"

        with tempfile.NamedTemporaryFile(suffix=".wav") as out_file:
            sf.write(out_file.name, audio, samplerate=_extract_sample_rate(audio_mm), format="WAV")
            transcript = convert_audio_file_to_text(out_file.name)
            audio_bytes = Path(out_file.name).read_bytes()

        assert_audio_speech_response(
            OmniResponse(
                success=True,
                audio_bytes=audio_bytes,
                audio_content=transcript,
                audio_format="audio/wav",
            ),
            {"input": synth_text, "response_format": "wav"},
            run_level="advanced_model",
        )
