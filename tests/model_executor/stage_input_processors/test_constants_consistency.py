# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Constant-consistency guards (RFC #4872).

Verifies that magic special-token ids that were previously duplicated across
processors / pipelines / model implementations now resolve to a single value:

* qwen3 codec ``4196/4197/4198``: processor ``_QWEN3_CODEC_*`` == model
  ``TALKER_CODEC_*`` == HF config defaults (``codec_pad_id`` /
  ``codec_bos_id`` / ``codec_eos_token_id``).
* qwen2_5 talker codec end ``8294``: processor ``TALKER_CODEC_END_TOKEN_ID``
  == model ``TALKER_CODEC_EOS_TOKEN_ID`` == pipeline ``stop_token_ids``.
* ``2150``: single-sourced in ``stage_input_processors._constants`` and shared
  by the qwen3_omni / qwen3_tts / aura_omni pipelines.

CPU-only: imports modules and reads constants; never loads model weights.
"""

import pytest

from vllm_omni.model_executor.models.aura_omni.pipeline import AURA_OMNI_PIPELINE
from vllm_omni.model_executor.models.qwen2_5_omni.pipeline import QWEN2_5_OMNI_PIPELINE
from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni import TALKER_CODEC_EOS_TOKEN_ID
from vllm_omni.model_executor.models.qwen3_omni.pipeline import QWEN3_OMNI_PIPELINE
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
    TALKER_CODEC_BOS_TOKEN_ID,
    TALKER_CODEC_PAD_TOKEN_ID,
)
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
    TALKER_CODEC_EOS_TOKEN_ID as QWEN3_MODEL_CODEC_EOS,
)
from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSTalkerConfig
from vllm_omni.model_executor.models.qwen3_tts.pipeline import QWEN3_TTS_PIPELINE
from vllm_omni.model_executor.stage_input_processors._constants import QWEN3_CODEC_EOS_TOKEN_ID
from vllm_omni.model_executor.stage_input_processors.qwen2_5_omni import TALKER_CODEC_END_TOKEN_ID
from vllm_omni.model_executor.stage_input_processors.qwen3_omni import (
    _QWEN3_CODEC_BOS_TOKEN_ID,
    _QWEN3_CODEC_EOS_TOKEN_ID,
    _QWEN3_CODEC_PAD_TOKEN_ID,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Expected values locked by the baseline commit `731a771`.
QWEN3_CODEC_EXPECTED = (4196, 4197, 4198)  # (pad, bos, eos)
QWEN2_5_CODEC_END_EXPECTED = 8294
QWEN3_CODEC_STOP_EXPECTED = 2150


def _pipeline_stop_token_ids(pipeline) -> list[int]:
    """Extract the (unique) ``stop_token_ids`` across a pipeline's stages."""
    ids: list[int] = []
    for stage in pipeline.stages:
        constraints = getattr(stage, "sampling_constraints", None) or {}
        stage_ids = constraints.get("stop_token_ids")
        if stage_ids:
            ids.extend(stage_ids)
    return ids


def test_qwen3_codec_token_ids_processor_matches_model():
    """4196/4197/4198: processor and model implementation must agree."""
    processor = (
        _QWEN3_CODEC_PAD_TOKEN_ID,
        _QWEN3_CODEC_BOS_TOKEN_ID,
        _QWEN3_CODEC_EOS_TOKEN_ID,
    )
    model = (TALKER_CODEC_PAD_TOKEN_ID, TALKER_CODEC_BOS_TOKEN_ID, QWEN3_MODEL_CODEC_EOS)
    assert processor == model == QWEN3_CODEC_EXPECTED


def test_qwen3_codec_token_ids_match_hf_config():
    """4196/4197/4198: processor constants must match the in-repo Qwen3-TTS
    HF config defaults (single source of truth)."""
    cfg = Qwen3TTSTalkerConfig()
    assert (cfg.codec_pad_id, cfg.codec_bos_id, cfg.codec_eos_token_id) == QWEN3_CODEC_EXPECTED
    assert (
        _QWEN3_CODEC_PAD_TOKEN_ID,
        _QWEN3_CODEC_BOS_TOKEN_ID,
        _QWEN3_CODEC_EOS_TOKEN_ID,
    ) == (cfg.codec_pad_id, cfg.codec_bos_id, cfg.codec_eos_token_id)


def test_qwen3_omni_runtime_talker_config_matches_processor():
    """The transformers ``Qwen3OmniMoeTalkerConfig`` (the config qwen3-omni
    actually runs with) must also carry 4196/4197/4198."""
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeTalkerConfig,
    )

    cfg = Qwen3OmniMoeTalkerConfig()
    assert (cfg.codec_pad_id, cfg.codec_bos_id, cfg.codec_eos_token_id) == QWEN3_CODEC_EXPECTED


def test_qwen2_5_stop_token_consistent():
    """8294: processor / model implementation / pipeline must agree."""
    assert TALKER_CODEC_END_TOKEN_ID == QWEN2_5_CODEC_END_EXPECTED
    assert TALKER_CODEC_EOS_TOKEN_ID == TALKER_CODEC_END_TOKEN_ID
    assert _pipeline_stop_token_ids(QWEN2_5_OMNI_PIPELINE) == [TALKER_CODEC_END_TOKEN_ID]


def test_qwen3_codec_stop_token_shared_across_pipelines():
    """2150: the three pipelines must share the single-sourced constant."""
    assert QWEN3_CODEC_EOS_TOKEN_ID == QWEN3_CODEC_STOP_EXPECTED
    for pipeline in (QWEN3_OMNI_PIPELINE, QWEN3_TTS_PIPELINE, AURA_OMNI_PIPELINE):
        assert _pipeline_stop_token_ids(pipeline) == [QWEN3_CODEC_EOS_TOKEN_ID]
