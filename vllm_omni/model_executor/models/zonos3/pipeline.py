# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Single native MRV2 scheduler; AR and incremental ZAC-1 share GPU slots."""

from vllm_omni.config.endpoint_policy import EndpointRestriction, OmniServingCapability
from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

ZONOS3_PIPELINE = PipelineConfig(
    model_type="zonos3",
    single_stage_async_chunk=True,
    model_arch="Zonos3ForCausalLM",
    default_deploy_config_name="zonos3_mini.yaml",
    endpoint_restrictions=tuple(
        EndpointRestriction(capability, "ZONOS3-Mini is served through the Speech API.")
        for capability in (
            OmniServingCapability.COMPLETIONS,
            OmniServingCapability.CHAT_COMPLETIONS,
            OmniServingCapability.CHAT_COMPLETIONS_BATCH,
            OmniServingCapability.RESPONSES,
            OmniServingCapability.MESSAGES,
            OmniServingCapability.INVOCATIONS,
            OmniServingCapability.GENERATE_TOKENS,
        )
    ),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="zonos3_ar",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="audio",
            final_output=True,
            final_output_type="audio",
            supports_native_mrv2_data_plane=True,
            recompute_preemption="fail",
            sampling_constraints={
                "stop_token_ids": [2],
                "detokenize": False,
            },
        ),
    ),
)
