# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline E2E smoke test for Audex text-to-audio (TTA).

Caption → RVQ-phase-masked <audiocodec_*> tokens → XCodec1 waveform through
the ``nemotron_labs_audex_tta`` pipeline. RVQ phase validity is enforced by
the mask logits processor (unit-tested against the official validator); the
hard gates here are decode-side: non-silent, finite audio at the documented
sample rate. CFG runs at the official TTA default (3.0).
"""

from __future__ import annotations

import copy
import os

import numpy as np
import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.model_executor.models.audex.prompt import build_tta_cond_prompt

MODEL = "nvidia/Nemotron-Labs-Audex-2B"
MODEL_DIR_ENV = "VLLM_OMNI_AUDEX_MODEL_DIR"
SAMPLE_RATE = 16_000
CAPTION = "Heavy rain falling on a tin roof."

_audex_deployment = get_deploy_config_path("nemotron_labs_audex_tta.yaml")
_audex_model = os.environ.get(MODEL_DIR_ENV) or MODEL
_OMNI_RUNNER_PARAMS = [
    pytest.param(
        (_audex_model, _audex_deployment, {"async_chunk": False}),
        id="tta_full_payload",
    ),
]
pytestmark = [
    pytest.mark.slow,
    pytest.mark.parametrize("omni_runner", _OMNI_RUNNER_PARAMS, indirect=True),
]


def _concat_audio(audio_val) -> np.ndarray:
    if isinstance(audio_val, list):
        tensors = [t.detach().cpu().float().reshape(-1) for t in audio_val if isinstance(t, torch.Tensor)]
        if not tensors:
            return np.zeros((0,), dtype=np.float32)
        return torch.cat(tensors, dim=-1).numpy().astype(np.float32, copy=False)
    if isinstance(audio_val, torch.Tensor):
        return audio_val.detach().cpu().float().reshape(-1).numpy()
    return np.asarray(audio_val, dtype=np.float32).reshape(-1)


def _tta_sampling_params(runner: OmniRunner, cond_prompt: str, cfg_scale: float = 3.0):
    from transformers import AutoTokenizer

    from vllm_omni.model_executor.models.audex.checkpoint import ensure_audex_snapshot
    from vllm_omni.model_executor.models.audex.prompt import build_tta_null_prompt
    from vllm_omni.model_executor.models.audex.tta import build_tta_phase_token_ids

    root = ensure_audex_snapshot(_audex_model, profile="tta")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(root, "checkpoint_folder_audiogen"))
    phase_token_ids, start_tid, end_tid = build_tta_phase_token_ids(tokenizer)

    params = copy.deepcopy(runner.omni.resolve_sampling_params_list(None))
    stage0 = params[0]
    if stage0.extra_args is None:
        stage0.extra_args = {}
    stage0.extra_args["tta_rvq"] = {
        "phase_token_ids": phase_token_ids,
        "start_tid": start_tid,
        "end_tid": end_tid,
        "codec_cap": 4000,
        "start_in_prompt": True,
    }
    if cfg_scale > 1.0:
        stage0.extra_args.update(
            {
                "cfg_scale": float(cfg_scale),
                "cfg_role": "cond",
                "cfg_pair_id": "e2e-tta-pair",
                "cfg_null_prompt": build_tta_null_prompt(cond_prompt, tokenizer),
            }
        )
    return params


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_audex_offline_tta_smoke(omni_runner: OmniRunner, run_level: str) -> None:
    """A caption decodes through XCodec1 to non-silent, finite 16 kHz audio."""
    prompt = build_tta_cond_prompt(CAPTION)
    params = _tta_sampling_params(omni_runner, prompt)
    outputs = omni_runner.omni.generate([prompt], params)

    assert len(outputs) == 1
    audio_mm = outputs[0].multimodal_output
    assert "audio" in audio_mm, f"No audio output found: {list(audio_mm.keys())}"
    audio = _concat_audio(audio_mm["audio"])
    assert audio.size > 0, "Generated audio is empty"
    assert np.isfinite(audio).all(), "Audio contains NaN/Inf"

    sr_val = audio_mm.get("sr", SAMPLE_RATE)
    if isinstance(sr_val, list) and sr_val:
        sr_val = sr_val[-1]
    if hasattr(sr_val, "item"):
        sr_val = sr_val.item()
    assert int(sr_val) == SAMPLE_RATE, f"Unexpected sample_rate={sr_val}"

    if run_level in {"advanced_model", "full_model"}:
        duration_s = audio.size / SAMPLE_RATE
        # The decode cap is 500 frames = 10 s at 50 fps.
        assert 0.1 <= duration_s <= 10.5, f"Unexpected duration={duration_s:.3f}s"
        rms = float(np.sqrt(np.mean(np.square(audio))))
        assert rms > 1e-4, f"Audio is near-silent (rms={rms})"
