# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E offline inference test for CSM-1B (Sesame).

CSM-1B is a plain text to speech model (speaker-id conditioning, no reference
audio on this path). Two-stage pipeline: backbone AR + inline 31-step depth
decoder, then the Mimi vocoder (code2wav) at 24 kHz.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

import numpy as np
import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

MODEL = os.environ.get("CSM_MODEL_PATH", "sesame/csm-1b")
DEPLOY_CONFIG = get_deploy_config_path("csm.yaml")
_DEFAULT_MAX_FRAMES = 64


def _concat_audio(audio_val) -> np.ndarray:
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


def _build_csm_input(text: str, speaker: str = "0", max_frames: int = _DEFAULT_MAX_FRAMES) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tagged = f"<|begin_of_text|>[{speaker}]{text}<|end_of_text|>"
    ids = list(tokenizer(tagged, add_special_tokens=False)["input_ids"])
    return {
        "prompt_token_ids": ids,
        "additional_information": {
            "prompt_token_ids": [ids],
            "temperature": [0.0],
            "top_k": [0],
            "max_new_frames": [max_frames],
        },
    }


def _get_deploy_config() -> str:
    """CSM deploy with both stages eager (matches the example + first-pass deploy)."""
    return modify_stage_config(
        DEPLOY_CONFIG,
        updates={"stages": {0: {"enforce_eager": True}, 1: {"enforce_eager": True}}},
    )


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_offline_text_to_speech() -> None:
    """Plain text to speech offline.

    Deploy: csm.yaml, both stages eager.
    Input modal: text (speaker-id conditioning).
    Output modal: 24 kHz audio, bounded by the per-request frame cap.
    """
    from tests.helpers.runtime import OmniRunner

    with OmniRunner(
        MODEL,
        stage_configs_path=_get_deploy_config(),
        stage_init_timeout=600,
    ) as omni_runner:
        outputs = omni_runner.omni.generate(
            [_build_csm_input("The quick brown fox jumps over the lazy dog.")],
            omni_runner.get_default_sampling_params_list(),
        )

    assert outputs, "No outputs returned"
    audio_mm = outputs[0].multimodal_output
    assert "audio" in audio_mm, "No audio output found"
    audio = _concat_audio(audio_mm["audio"])
    sr = _extract_sample_rate(audio_mm)
    assert sr == 24000, f"Unexpected sample rate {sr}"
    assert audio.size > 0, "Empty audio"
    assert np.isfinite(audio).all(), "Non-finite audio samples"
    # 64-frame cap at 80 ms/frame is up to ~5.12 s; expect a plausible clip.
    assert audio.size / sr > 0.3, f"Implausibly short audio: {audio.size / sr:.2f}s"
