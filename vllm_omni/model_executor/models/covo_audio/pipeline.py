# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Covo-Audio pipeline topology (frozen).

Stage 0: fused_thinker_talker — multimodal understanding + interleaved text/audio
         token generation (AR).
Stage 1: code2wav              — audio codes → waveform (generation).
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.covo_audio"

COVO_AUDIO_PIPELINE = PipelineConfig(
    model_type="covo_audio",
    model_arch="CovoAudioForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="fused_thinker_talker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="latent",
            sampling_constraints={
                "detokenize": True,
                "stop_token_ids": [151645],
                "ignore_eos": True,
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            custom_process_input_func=f"{_PROC}.llm2code2wav",
            sampling_constraints={"detokenize": False},
        ),
    ),
)
