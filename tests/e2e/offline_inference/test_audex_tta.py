# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline E2E smoke test for Audex text-to-audio (TTA).

Caption → RVQ-phase-masked <audiocodec_*> tokens → XCodec1 waveform through
the ``audex_tta`` pipeline. RVQ phase validity is enforced by
the mask logits processor (unit-tested against the official validator); the
hard gates here are decode-side: non-silent, finite audio at the documented
sample rate. CFG runs at the official TTA default (3.0).
"""

from __future__ import annotations

import copy
import os

import numpy as np
import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import concat_audio
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.model_executor.models.audex.prompt import build_tta_cond_prompt

MODEL = "nvidia/Nemotron-Labs-Audex-2B"
MODEL_DIR_ENV = "VLLM_OMNI_AUDEX_MODEL_DIR"
SAMPLE_RATE = 16_000

# The acceptance gate requires eight captions decoding to non-silent,
# finite audio through XCodec1.
CAPTIONS = (
    "Heavy rain falling on a tin roof.",
    "A dog barking in the distance while birds chirp.",
    "Ocean waves crashing on a rocky shore.",
    "A crackling campfire at night with crickets.",
    "Thunder rumbling during a storm.",
    "A steam train whistle blowing as it departs.",
    "Wind chimes ringing in a gentle breeze.",
    "Footsteps crunching through fresh snow.",
)

_audex_deployment = get_deploy_config_path("audex_tta.yaml")
_audex_model = os.environ.get(MODEL_DIR_ENV) or MODEL
_OMNI_RUNNER_PARAMS = [
    pytest.param(
        (_audex_model, _audex_deployment, {"async_chunk": False}),
        id="tta_full_payload",
    ),
]
# 30B-A3B variant: opt-in (pulls ~60 GB); enable with AUDEX_E2E_30B=1.
_AUDEX_30B_MODEL = os.environ.get("VLLM_OMNI_AUDEX_30B_MODEL_DIR") or "nvidia/Nemotron-Labs-Audex-30B-A3B"
if os.environ.get("AUDEX_E2E_30B") == "1":
    _OMNI_RUNNER_PARAMS.append(
        pytest.param(
            (_AUDEX_30B_MODEL, get_deploy_config_path("audex_tta_30b.yaml"), {"async_chunk": False}),
            id="tta_30b",
        )
    )
pytestmark = [
    pytest.mark.slow,
    pytest.mark.tts,
    pytest.mark.parametrize("omni_runner", _OMNI_RUNNER_PARAMS, indirect=True),
]


@pytest.fixture(scope="module")
def tta_tokenizer_bits():
    from transformers import AutoTokenizer

    from vllm_omni.model_executor.models.audex.checkpoint import ensure_audex_snapshot
    from vllm_omni.model_executor.models.audex.tta import build_tta_phase_token_ids

    root = ensure_audex_snapshot(_audex_model, profile="tta")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(root, "checkpoint_folder_audiogen"), trust_remote_code=True)
    phase_token_ids, start_tid, end_tid = build_tta_phase_token_ids(tokenizer)
    return tokenizer, phase_token_ids, start_tid, end_tid


def _tta_sampling_params(runner: OmniRunner, cond_prompt: str, tokenizer_bits, cfg_scale: float = 3.0):
    from vllm_omni.model_executor.models.audex.prompt import build_tta_null_prompt

    tokenizer, phase_token_ids, start_tid, end_tid = tokenizer_bits
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
                "cfg_pair_id": f"e2e-tta-{abs(hash(cond_prompt)) % 10_000_000}",
                "cfg_null_prompt": build_tta_null_prompt(cond_prompt, tokenizer),
            }
        )
    return params


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_audex_offline_tta_eight_captions(omni_runner: OmniRunner, run_level: str, tta_tokenizer_bits) -> None:
    """All eight captions decode through XCodec1 to non-silent, finite audio.

    RVQ phase validity is enforced by the mask logits processor by
    construction; the final user output carries only decoded audio (the
    audio-final pipeline emits no stage-0 token record), and disabling the
    mask in-engine as an e2e control would require a second engine boot
    with a different logits-processor config. The mask's sensitivity is
    therefore proven at unit level (see test_audex_tta.py's mask tests and
    the unmasked-sampling control) against the official validator.
    """
    real_weights = run_level in {"advanced_model", "full_model"}
    for caption in CAPTIONS:
        prompt = build_tta_cond_prompt(caption)
        params = _tta_sampling_params(omni_runner, prompt, tta_tokenizer_bits)
        outputs = omni_runner.omni.generate([prompt], params)

        assert len(outputs) == 1, f"{caption!r}: expected one output, got {len(outputs)}"
        audio_mm = outputs[0].multimodal_output
        assert "audio" in audio_mm, f"{caption!r}: no audio output: {list(audio_mm.keys())}"
        audio = concat_audio(audio_mm["audio"])
        assert audio.size > 0, f"{caption!r}: generated audio is empty"
        assert np.isfinite(audio).all(), f"{caption!r}: audio contains NaN/Inf"

        sr_val = audio_mm.get("sr", SAMPLE_RATE)
        if isinstance(sr_val, list) and sr_val:
            sr_val = sr_val[-1]
        if hasattr(sr_val, "item"):
            sr_val = sr_val.item()
        assert int(sr_val) == SAMPLE_RATE, f"{caption!r}: unexpected sample_rate={sr_val}"

        if real_weights:
            duration_s = audio.size / SAMPLE_RATE
            # The decode cap is 500 frames = 10 s at 50 fps.
            assert 0.1 <= duration_s <= 10.5, f"{caption!r}: unexpected duration={duration_s:.3f}s"
            rms = float(np.sqrt(np.mean(np.square(audio))))
            assert rms > 1e-4, f"{caption!r}: audio is near-silent (rms={rms})"
