# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Serving-side registration invariants for the Audio8 TTS adapter.

The engine never names ``audio8_tts`` directly: ``/v1/audio/speech`` resolves the
adapter by matching the deployed stage's ``model_stage`` key against the
``stage_keys`` each adapter declares. That makes the pipeline definition and the
adapter two halves of one contract, with nothing in the type system holding them
together -- rename either side and the endpoint silently stops recognising the
model (falling through to "unknown TTS model") instead of failing at startup.

These tests pin the contract from both ends.
"""

import pytest

from vllm_omni.entrypoints.openai.tts_adapters import (
    all_tts_model_types,
    all_tts_stage_keys,
    detect_tts_model_type,
    resolve_adapter,
)
from vllm_omni.entrypoints.openai.tts_adapters.audio8_tts import Audio8TTSAdapter
from vllm_omni.model_executor.models.audio8_tts.pipeline import AUDIO8_TTS_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SLOW_AR_STAGE_KEY = "audio8_tts_slow_ar"


def _pipeline_stage_keys() -> list[str]:
    return [stage.model_stage for stage in AUDIO8_TTS_PIPELINE.stages]


def test_pipeline_stage_key_is_the_one_the_adapter_claims():
    """The shipped pipeline must name the stage the adapter matches on."""
    assert SLOW_AR_STAGE_KEY in _pipeline_stage_keys()
    assert SLOW_AR_STAGE_KEY in Audio8TTSAdapter.stage_keys


def test_detection_resolves_slow_ar_stage_to_the_audio8_adapter():
    assert detect_tts_model_type(SLOW_AR_STAGE_KEY, "Audio8TTSSlowARForConditionalGeneration") == "audio8_tts"
    assert resolve_adapter("audio8_tts") is Audio8TTSAdapter


def test_adapter_is_registered_in_the_global_tables():
    """Importing the package must register the adapter, not just define it."""
    assert "audio8_tts" in all_tts_model_types()
    assert SLOW_AR_STAGE_KEY in all_tts_stage_keys()


def test_codec_decoder_stage_does_not_serve_speech():
    """Only stage 0 answers /v1/audio/speech; the codec stage must not match."""
    codec_stage_key = _pipeline_stage_keys()[1]
    assert codec_stage_key != SLOW_AR_STAGE_KEY
    assert detect_tts_model_type(codec_stage_key, "Audio8TTSCodecDecoder") != "audio8_tts"


def test_apply_sampling_overrides_defaults_tts_local_seed_for_fast_ar():
    """The deploy seed must reach the Fast AR through ``tts_local_seed``.

    Only the Slow AR is seeded via ``sampling_metadata.generators``; the residual
    codebooks read ``tts_local_seed`` and otherwise fall back to global RNG, which
    makes an unseeded request's audio depend on batch composition. The adapter
    defaults it from the stage-0 seed without clobbering an explicit value.
    """
    from types import SimpleNamespace

    from vllm.sampling_params import SamplingParams

    adapter = Audio8TTSAdapter.__new__(Audio8TTSAdapter)
    request = SimpleNamespace(max_new_tokens=None)

    # Deploy default seed propagates to tts_local_seed.
    out = adapter.apply_sampling_overrides([SamplingParams(seed=42)], request)
    assert out[0].extra_args["tts_local_seed"] == 42

    # An explicit tts_local_seed is left intact (setdefault, not overwrite).
    pinned = [SamplingParams(seed=42, extra_args={"tts_local_seed": 7})]
    assert adapter.apply_sampling_overrides(pinned, request)[0].extra_args["tts_local_seed"] == 7

    # No seed at all -> nothing injected (non-deterministic by request contract).
    out_unseeded = adapter.apply_sampling_overrides([SamplingParams(seed=None)], request)
    assert (out_unseeded[0].extra_args or {}).get("tts_local_seed") is None
