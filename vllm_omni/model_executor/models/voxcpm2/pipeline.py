# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VoxCPM2 pipeline topology (frozen).

Single-stage AR TTS: text → speech waveform in one pass.
Uses the native MiniCPM4 base_lm with a per-request StaticKVCache that the
talker restores into the paged attention layer at step boundaries.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

VOXCPM2_PIPELINE = PipelineConfig(
    model_type="voxcpm2",
    model_arch="VoxCPM2TalkerForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="latent_generator",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="audio",
            owns_tokenizer=True,
            engine_output_type="audio",
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [1],
            },
        ),
    ),
)
