# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Which stages declare that recompute preemption cannot replay them.

A stage qualifies when its decode step feeds PagedAttention something the token
stream does not carry, so `_preempt_request()`'s prompt replay rebuilds a
different KV than the one that produced the audio already delivered
(vllm-project/vllm-omni#6179):

- a sampled RVQ codebook folded into `inputs_embeds` (Qwen3-Omni talker,
  VoxCPM2, audio8_tts, fish_speech, qwen3_tts, moss_tts),
- a per-frame temporal hidden plus codebook (personaplex),
- 31 side-channel FSQ codes per frame (Gepard).

Stages that embed the sampled token alone -- every thinker, Qwen2.5-Omni's
talker, which keeps all codebook layers in the token stream -- keep upstream's
recompute mode. Fish Omni also keeps recompute: its MRv2 model state saves and
restores the full generated codebooks across preemption.

This pins the declaration, not the failure mode: a stage dropped from `_FAIL`
silently goes back to splicing or corrupting audio under memory pressure.
"""

from __future__ import annotations

import pytest

from vllm_omni.model_executor.models.audio8_tts.pipeline import AUDIO8_TTS_PIPELINE
from vllm_omni.model_executor.models.fish_omni.pipeline import FISH_OMNI_PIPELINE
from vllm_omni.model_executor.models.fish_speech.pipeline import FISH_SPEECH_PIPELINE
from vllm_omni.model_executor.models.gepard.pipeline import GEPARD_PIPELINE
from vllm_omni.model_executor.models.moss_tts.pipeline import (
    MOSS_TTS_LOCAL_PIPELINE,
    MOSS_TTS_PIPELINE,
    MOSS_TTS_REALTIME_PIPELINE,
)
from vllm_omni.model_executor.models.personaplex.pipeline import PERSONAPLEX_PIPELINE
from vllm_omni.model_executor.models.qwen2_5_omni.pipeline import QWEN2_5_OMNI_PIPELINE
from vllm_omni.model_executor.models.qwen3_omni.pipeline import QWEN3_OMNI_PIPELINE
from vllm_omni.model_executor.models.qwen3_tts.pipeline import (
    QWEN3_TTS_FUSED_PIPELINE,
    QWEN3_TTS_PIPELINE,
)
from vllm_omni.model_executor.models.voxcpm2.pipeline import VOXCPM2_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# (pipeline, stage that cannot replay)
_FAIL = [
    (QWEN3_OMNI_PIPELINE, "talker"),
    (VOXCPM2_PIPELINE, "latent_generator"),
    (GEPARD_PIPELINE, "gepard"),
    (AUDIO8_TTS_PIPELINE, "audio8_tts_slow_ar"),
    (FISH_SPEECH_PIPELINE, "fish_speech_slow_ar"),
    (QWEN3_TTS_PIPELINE, "qwen3_tts"),
    (QWEN3_TTS_FUSED_PIPELINE, "qwen3_tts"),
    (PERSONAPLEX_PIPELINE, "personaplex"),
    (MOSS_TTS_PIPELINE, "moss_tts"),
    (MOSS_TTS_REALTIME_PIPELINE, "moss_tts"),
    (MOSS_TTS_LOCAL_PIPELINE, "moss_tts_local"),
]

_ALLOW = [
    # Fish Omni MRv2 retains all generated codebooks across slot reuse.
    (FISH_OMNI_PIPELINE, "fish_omni"),
    # Text AR: the token stream is the state.
    (QWEN3_OMNI_PIPELINE, "thinker"),
    (QWEN2_5_OMNI_PIPELINE, "thinker"),
    (QWEN2_5_OMNI_PIPELINE, "talker"),
]


def _stage(pipeline, model_stage):
    (stage,) = [s for s in pipeline.stages if s.model_stage == model_stage]
    return stage


@pytest.mark.parametrize(
    ("pipeline", "model_stage"),
    _FAIL,
    ids=[f"{p.model_type}:{m}" for p, m in _FAIL],
)
def test_unreplayable_stage_fails_instead_of_replaying(pipeline, model_stage):
    stage = _stage(pipeline, model_stage)
    assert stage.recompute_preemption == "fail", (
        f"{pipeline.model_type}/{model_stage} feeds decode embeddings that its "
        "token stream cannot rebuild; a recompute preemption would splice or "
        "corrupt its audio."
    )


@pytest.mark.parametrize(
    ("pipeline", "model_stage"),
    _ALLOW,
    ids=[f"{p.model_type}:{m}" for p, m in _ALLOW],
)
def test_replayable_stage_keeps_upstream_recompute(pipeline, model_stage):
    assert _stage(pipeline, model_stage).recompute_preemption == "allow"


def test_default_is_allow_so_untouched_pipelines_are_unaffected():
    from vllm_omni.config.stage_config import StagePipelineConfig

    assert StagePipelineConfig(stage_id=0, model_stage="x").recompute_preemption == "allow"
