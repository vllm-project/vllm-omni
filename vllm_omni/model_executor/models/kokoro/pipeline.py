# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Kokoro-82M pipeline topology (frozen).

Single-stage TTS: text -> 24 kHz speech. The non-autoregressive network runs
inside ``KokoroForConditionalGeneration.forward()``, one phoneme chunk per AR
step (MOSS-TTS-Nano generator pattern), so long inputs stream chunk by chunk.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)
from vllm_omni.model_executor.models.kokoro.configuration_kokoro import KOKORO_STOP_TOKEN_ID

KOKORO_PIPELINE = PipelineConfig(
    model_type="kokoro",
    default_deploy_config_name="kokoro.yaml",
    model_arch="KokoroForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="kokoro",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="audio",
            owns_tokenizer=True,
            engine_output_type="audio",
            sampling_constraints={
                "detokenize": False,
                # compute_logits() forces this on the last chunk's step.
                "stop_token_ids": [KOKORO_STOP_TOKEN_ID],
            },
        ),
    ),
)
