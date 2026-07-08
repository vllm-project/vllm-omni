# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OmniVoice pipeline topology (frozen).

Single-stage diffusion model for text-to-speech.
Stage 0: DiT — text → iterative unmasking → codebook tokens → DAC decode → 24kHz audio
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

OMNIVOICE_PIPELINE = PipelineConfig(
    model_type="omnivoice",
    model_arch="OmniVoicePipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="audio",
            owns_tokenizer=False,
            requires_multimodal_data=False,
            engine_output_type="audio",
        ),
    ),
)
