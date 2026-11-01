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

from tests.helpers.mark import hardware_test
from tests.helpers.media import concat_audio
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


_audex_deployment = get_deploy_config_path("audex_tts.yaml")
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
# 30B-A3B variant: opt-in (pulls ~60 GB); enable with AUDEX_E2E_30B=1.
_AUDEX_30B_MODEL = os.environ.get("VLLM_OMNI_AUDEX_30B_MODEL_DIR") or "nvidia/Nemotron-Labs-Audex-30B-A3B"
if os.environ.get("AUDEX_E2E_30B") == "1":
    _OMNI_RUNNER_PARAMS.append(
        pytest.param(
            (_AUDEX_30B_MODEL, get_deploy_config_path("audex_tts_30b.yaml"), {"async_chunk": True}),
            id="tts_30b",
        )
    )
pytestmark = [
    pytest.mark.slow,
    pytest.mark.tts,
    pytest.mark.parametrize("omni_runner", _OMNI_RUNNER_PARAMS, indirect=True),
]


@pytest.mark.advanced_model
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
        audio = concat_audio(audio_mm["audio"])
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


def _cfg_sampling_params(runner: OmniRunner, cfg_scale: float, pair_id: str, cond_prompt: str):
    """Stage sampling params carrying the CFG pair contract for one request."""
    import copy

    from transformers import AutoTokenizer

    from vllm_omni.model_executor.models.audex.checkpoint import ensure_audex_snapshot
    from vllm_omni.model_executor.models.audex.prompt import build_null_prompt

    # Resolve the ACTIVE param's model (2B or 30B), not the module default:
    # the null prompt must be built with the same tokenizer the engine uses.
    root = ensure_audex_snapshot(runner.model_name)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(root, "checkpoint_folder_audiogen"), trust_remote_code=True)
    params = copy.deepcopy(runner.omni.resolve_sampling_params_list(None))
    stage0 = params[0]
    if stage0.extra_args is None:
        stage0.extra_args = {}
    stage0.extra_args.update(
        {
            "cfg_scale": float(cfg_scale),
            "cfg_role": "cond",
            "cfg_pair_id": pair_id,
            "cfg_null_prompt": build_null_prompt(cond_prompt, tokenizer),
        }
    )
    return params


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_audex_offline_cfg_guided_single_stream(omni_runner: OmniRunner, run_level: str) -> None:
    """A guided request must yield exactly ONE audio stream (companion suppressed)."""
    prompt = build_cond_prompt(SYNTH_TEXTS[0])
    params = _cfg_sampling_params(omni_runner, 1.5, "e2e-cfg-pair", prompt)
    outputs = omni_runner.omni.generate([prompt], params)

    assert len(outputs) == 1, f"expected exactly one output (no companion leak), got {len(outputs)}"
    audio = concat_audio(outputs[0].multimodal_output["audio"])
    assert audio.size > 0, "Guided generation produced empty audio"
    if run_level in {"advanced_model", "full_model"}:
        rms = float(np.sqrt(np.mean(np.square(audio))))
        assert rms > 1e-3, f"Guided audio is near-silent (rms={rms})"
        duration_s = audio.size / SAMPLE_RATE
        assert 0.5 <= duration_s <= 20.0, f"Unexpected duration={duration_s:.3f}s"


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_audex_offline_cfg_disabled_is_byte_identical(omni_runner: OmniRunner, run_level: str) -> None:
    """cfg_scale=1.0 must be indistinguishable from not passing cfg at all."""
    prompt = build_cond_prompt(SYNTH_TEXTS[1])

    baseline = omni_runner.omni.generate([prompt])
    base_audio = concat_audio(baseline[0].multimodal_output["audio"])

    import copy

    params = copy.deepcopy(omni_runner.omni.resolve_sampling_params_list(None))
    if params[0].extra_args is None:
        params[0].extra_args = {}
    params[0].extra_args["cfg_scale"] = 1.0
    disabled = omni_runner.omni.generate([prompt], params)
    disabled_audio = concat_audio(disabled[0].multimodal_output["audio"])

    assert base_audio.shape == disabled_audio.shape, (
        f"cfg_scale=1.0 changed output length: {base_audio.shape} vs {disabled_audio.shape}"
    )
    np.testing.assert_array_equal(base_audio, disabled_audio)
