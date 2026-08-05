# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IndexTTS2 pipeline: GPT AR talker (text → mel codes) → S2Mel + BigVGAN (mel → audio).

Two-stage non-streaming pipeline. S2Mel flow matching (25 Euler steps) requires
the full mel code sequence, so async_chunk is not applicable.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.indextts2"

INDEXTTS2_PIPELINE = PipelineConfig(
    model_type="indextts2",
    default_deploy_config_name="indextts2.yaml",
    model_arch="IndexTTS2TalkerForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="indextts2_talker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            extras={"skip_tokenizer_init": True, "tokenizer": "gpt2"},
            custom_process_next_stage_input_func=f"{_PROC}.talker2s2mel_full_payload",
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [8193],
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="indextts2_s2mel_decoder",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="IndexTTS2S2MelDecoder",
            sync_process_input_func=f"{_PROC}.talker2s2mel_token_only",
            extras={"skip_tokenizer_init": True, "tokenizer": "gpt2"},
            sampling_constraints={"detokenize": True},
        ),
    ),
)
