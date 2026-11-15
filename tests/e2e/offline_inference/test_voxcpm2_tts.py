# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""E2E test for VoxCPM2 native AR offline inference."""

import os
from collections.abc import Mapping

import pytest
import soundfile as sf
import torch

from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path

VOXCPM2_MODEL = os.environ.get("VOXCPM2_MODEL", "openbmb/VoxCPM2")
DEPLOY_CONFIG = get_deploy_config_path("ci/voxcpm2.yaml")
REF_AUDIO_PATH = get_asset_path("qwen3_tts/clone_2.wav")
SAMPLE_RATE = 48000

# VoxCPM2 ships a custom tokenizer, so remote code must be explicitly enabled.
_OMNI_RUNNER_PARAM = (VOXCPM2_MODEL, DEPLOY_CONFIG, {"trust_remote_code": True})

pytestmark = pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True)


def _extract_audio(multimodal_output: dict) -> torch.Tensor:
    """Extract the final complete audio tensor from multimodal output."""
    assert isinstance(multimodal_output, (dict, Mapping)), f"Expected dict/Mapping, got {type(multimodal_output)}"

    # Output processor accumulates per-step audio chunks under "audio".
    audio = multimodal_output.get("audio")
    if audio is None:
        audio = multimodal_output.get("model_outputs")
    assert audio is not None, f"No audio key, got {list(multimodal_output.keys())}"

    if isinstance(audio, list):
        valid = [torch.as_tensor(x).float().cpu().reshape(-1) for x in audio if x is not None]
        assert valid, "No valid audio tensors in output list"
        audio = torch.cat(valid, dim=0) if len(valid) > 1 else valid[0]

    assert isinstance(audio, torch.Tensor), f"Expected Tensor, got {type(audio)}"
    return audio


def _build_voice_clone_prompt(omni_runner: OmniRunner, text: str) -> dict:
    """Build the full VoxCPM2 prefill prompt for reference-audio inference."""
    from transformers import AutoTokenizer

    from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import (
        build_cjk_split_map,
        build_voxcpm2_prompt,
    )

    reference_audio, reference_sample_rate = sf.read(str(REF_AUDIO_PATH), dtype="float32", always_2d=True)
    reference_audio_bytes = reference_audio.mean(axis=1).astype("float32", copy=False).tobytes()

    tokenizer = AutoTokenizer.from_pretrained(omni_runner.model_name, trust_remote_code=True)
    split_map = build_cjk_split_map(tokenizer)
    hf_config = omni_runner.omni.engine.stage_vllm_configs[0].model_config.hf_config
    return build_voxcpm2_prompt(
        hf_config=hf_config,
        tokenizer=tokenizer,
        split_map=split_map,
        text=text,
        ref_audio=reference_audio_bytes,
        ref_sr=int(reference_sample_rate),
    )


@pytest.mark.slow
@pytest.mark.tts
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "npu": "A2"}, num_cards=1)
def test_voxcpm2_zero_shot_001(omni_runner: OmniRunner) -> None:
    """Test zero-shot TTS produces valid audio output."""
    outputs = omni_runner.omni.generate([{"prompt": "Hello, this is a test."}])
    assert len(outputs) == 1

    audio = _extract_audio(outputs[0].multimodal_output)
    duration_s = audio.shape[0] / SAMPLE_RATE
    assert 0.5 < duration_s < 30.0, f"Audio duration out of range: {duration_s:.2f}s"


@pytest.mark.slow
@pytest.mark.tts
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "npu": "A2"}, num_cards=1)
def test_voxcpm2_voice_clone_002(omni_runner: OmniRunner) -> None:
    """Test voice cloning with a reference audio file.

    Use the tracked CI asset so every hardware lane executes the voice-clone
    contract instead of depending on an optional example file in the package.
    """
    prompt = _build_voice_clone_prompt(omni_runner, "Hello, this is a voice clone demo.")
    outputs = omni_runner.omni.generate([prompt])
    assert len(outputs) == 1

    audio = _extract_audio(outputs[0].multimodal_output)
    duration_s = audio.shape[0] / SAMPLE_RATE
    assert 0.5 < duration_s < 30.0, f"Audio duration out of range: {duration_s:.2f}s"


@pytest.mark.slow
@pytest.mark.tts
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "npu": "A2"}, num_cards=1)
def test_voxcpm2_prefill_decode_mixed_batch_003(omni_runner: OmniRunner) -> None:
    """Regression: prefill+decode mixed batch must not crash (PR #2903)."""
    long_prompt = (
        "This is a deliberately long prompt that will stay in the decode "
        "phase for many steps so that subsequent shorter prompts keep "
        "entering prefill alongside it, reproducing the prefill plus "
        "decode mixed batch scheduling pattern."
    )
    short_prompts = [
        "Hello one.",
        "Hello two.",
        "Hello three.",
        "Hello four.",
    ]
    requests = [{"prompt": long_prompt}] + [{"prompt": p} for p in short_prompts]

    outputs = omni_runner.omni.generate(requests)
    assert len(outputs) == len(requests)

    for i, out in enumerate(outputs):
        audio = _extract_audio(out.multimodal_output)
        duration_s = audio.shape[0] / SAMPLE_RATE
        assert 0.1 < duration_s < 30.0, f"Request {i} audio duration out of range: {duration_s:.2f}s"
