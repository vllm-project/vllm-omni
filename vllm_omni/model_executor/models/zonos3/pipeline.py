# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native MRV2 AR → incremental ZAC-1 topology; exactly two schedulers."""

from vllm.sampling_params import RequestOutputKind

from vllm_omni.config.endpoint_policy import EndpointRestriction, OmniServingCapability
from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

ZONOS3_PIPELINE = PipelineConfig(
    model_type="zonos3",
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
            engine_output_type="latent",
            supports_native_mrv2_data_plane=True,
            recompute_preemption="fail",
            async_chunk_process_next_stage_input_func="vllm_omni.model_executor.stage_input_processors.zonos3.zonos3_rvq_async_chunk",
            sampling_constraints={
                "stop_token_ids": [2],
                "detokenize": False,
                "output_kind": RequestOutputKind.FINAL_ONLY,
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="zonos3_codec",
            model_arch="Zonos3CodecModel",
            hf_config_name="codec_config",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            supports_native_mrv2_data_plane=True,
            retains_state_across_chunks=True,
            sampling_constraints={"detokenize": False},
        ),
    ),
)
