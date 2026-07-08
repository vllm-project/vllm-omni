# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline E2E smoke test for Audex (Nemotron-Labs-Audex-2B) TTS.

Passes the HF repo ROOT and verifies the 2-stage pipeline (thinker →
streaming causal speech decoder): root-path/subfolder resolution, codec-token
extraction, per-request streaming sessions, and the lookahead flush at EOS.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.model_executor.models.audex.prompt import build_cond_prompt

MODEL = "nvidia/Nemotron-Labs-Audex-2B"
MODEL_DIR_ENV = "VLLM_OMNI_AUDEX_MODEL_DIR"
SAMPLE_RATE = 16_000

SYNTH_TEXTS = (
    "The weather is so good, and I want to enjoy the beautiful morning in the park.",
    "The quick brown fox jumps over the lazy dog.",
)


def _concat_audio(audio_val) -> np.ndarray:
    if isinstance(audio_val, list):
        tensors = [t.detach().cpu().float().reshape(-1) for t in audio_val if isinstance(t, torch.Tensor)]
        if not tensors:
            return np.zeros((0,), dtype=np.float32)
        return torch.cat(tensors, dim=-1).numpy().astype(np.float32, copy=False)
    if isinstance(audio_val, torch.Tensor):
        return audio_val.detach().cpu().float().reshape(-1).numpy()
    return np.asarray(audio_val, dtype=np.float32).reshape(-1)


_audex_deployment = get_deploy_config_path("nemotron_labs_audex.yaml")
# Collection must not download anything: pass the repo id (or a local
# override) and let the engine's stage-init path resolve/download the
# required snapshot subset at execution time (ensure_audex_snapshot).
_audex_model = os.environ.get(MODEL_DIR_ENV) or MODEL
_OMNI_RUNNER_PARAMS = [
    pytest.param(
        (_audex_model, _audex_deployment, {"async_chunk": True}),
        id="async_chunk",
    ),
]
pytestmark = [
    pytest.mark.slow,
    pytest.mark.tts,
    pytest.mark.parametrize("omni_runner", _OMNI_RUNNER_PARAMS, indirect=True),
]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_audex_offline_tts_smoke(omni_runner: OmniRunner, run_level: str) -> None:
    """Audex TTS from the repo root should produce sane 16 kHz audio per prompt.

    At ``core_model`` level the runner loads DUMMY weights (structural smoke
    only); real-speech assertions apply from ``advanced_model`` up.
    """
    prompts = [build_cond_prompt(text) for text in SYNTH_TEXTS]
    outputs = omni_runner.omni.generate(prompts)

    real_weights = run_level in {"advanced_model", "full_model"}
    assert len(outputs) == len(SYNTH_TEXTS), f"expected {len(SYNTH_TEXTS)} outputs, got {len(outputs)}"
    for output in outputs:
        audio_mm = output.multimodal_output
        assert "audio" in audio_mm, f"No audio output found: {list(audio_mm.keys())}"
        audio = _concat_audio(audio_mm["audio"])
        assert audio.size > 0, "Generated audio is empty"

        sr_val = audio_mm.get("sr", SAMPLE_RATE)
        if isinstance(sr_val, list) and sr_val:
            sr_val = sr_val[-1]
        if hasattr(sr_val, "item"):
            sr_val = sr_val.item()
        assert int(sr_val) == SAMPLE_RATE, f"Unexpected sample_rate={sr_val}"

        if real_weights:
            duration_s = audio.size / SAMPLE_RATE
            assert 0.5 <= duration_s <= 20.0, f"Unexpected duration={duration_s:.3f}s"
            # Real speech, not silence.
            rms = float(np.sqrt(np.mean(np.square(audio))))
            assert rms > 1e-3, f"Audio is near-silent (rms={rms})"
