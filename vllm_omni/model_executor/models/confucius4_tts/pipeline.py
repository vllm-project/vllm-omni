# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Confucius4-TTS two-stage, non-streaming pipeline."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)
from vllm_omni.outputs.output_modality import (
    TensorAccumulationStrategy,
    register_key_accumulation_strategy,
)

from .configuration_confucius4_tts import SEMANTIC_EOS_TOKEN_ID

_PROCESSOR = "vllm_omni.model_executor.stage_input_processors.confucius4_tts"

# Stage 0 re-emits its decode counter as a 0-d tensor every step. It is a
# snapshot, not generated content, and torch.cat on a list of 0-d tensors
# raises when the output processor consolidates the request.
register_key_accumulation_strategy("meta.semantic_code_count", TensorAccumulationStrategy.REPLACE)

CONFUCIUS4_TTS_PIPELINE = PipelineConfig(
    model_type="confucius4_tts",
    default_deploy_config_name="confucius4_tts.yaml",
    model_arch="Confucius4TTSTalkerForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="confucius4_tts_talker",
            execution_type=StageExecutionType.LLM_AR,
            supports_native_mrv2_data_plane=True,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            extras={"skip_tokenizer_init": True},
            custom_process_next_stage_input_func=(f"{_PROCESSOR}.talker2decoder_full_payload"),
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [SEMANTIC_EOS_TOKEN_ID],
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="confucius4_tts_decoder",
            execution_type=StageExecutionType.LLM_GENERATION,
            supports_native_mrv2_data_plane=True,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="Confucius4TTSS2ADecoder",
            sync_process_input_func=(f"{_PROCESSOR}.talker2decoder_token_only"),
            extras={"skip_tokenizer_init": True},
            sampling_constraints={"detokenize": True},
            requires_full_payload_input=True,
        ),
    ),
)
