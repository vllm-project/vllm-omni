# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gepard-1.0 pipeline topology.

Single-stage AR TTS. The Qwen3.5 backbone runs under paged attention, the 32
codebook heads and the binary stop head are sampled inside the talker, and the
NeMo NanoCodec decodes the accumulated frames outside vLLM.

``compute_logits`` scatters head0 into vLLM's vocab-wide logits; head0's valid
range is 0..7, so the synthetic STOP sentinel is 8. ``stop_token_ids=[8]``
couples it to vLLM's stop machinery, with ``detokenize=False`` because head0 is
an audio code. Same idiom as VoxCPM2/MOSS.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

# head0 valid range is 0..7 (its FSQ cardinality is 8); STOP is the sentinel
# one past that range. Mirrors the reference config.py STOP_TOKEN = 8.
_GEPARD_STOP_TOKEN_ID = 8

GEPARD_PIPELINE = PipelineConfig(
    model_type="gepard",
    default_deploy_config_name="gepard.yaml",
    model_arch="GepardTalkerForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="gepard",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="audio",
            owns_tokenizer=True,
            engine_output_type="audio",
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [_GEPARD_STOP_TOKEN_ID],
            },
        ),
    ),
)
