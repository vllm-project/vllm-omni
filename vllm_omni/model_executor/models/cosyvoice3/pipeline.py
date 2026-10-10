# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CosyVoice3 pipeline topology (frozen).

Stage 0: Talker   — text prompt → speech tokens (LLM autoregressive).
Stage 1: Code2Wav — flow-matching decoder → acoustic features → waveform.
  * ``sync_process_input_func`` (``text2flow_token_only``) runs when
    ``deploy.async_chunk=false``: stage 1 allocates placeholder slots while
    bulk tensors arrive via ``text2flow_full_payload`` on the connector.
  * ``async_chunk_process_next_stage_input_func`` runs when
    ``deploy.async_chunk=true``: stage 0 streams codec chunks to stage 1
    through the shared-memory connector.
"""

import shutil
from dataclasses import replace

from transformers import PretrainedConfig

from vllm_omni.config.endpoint_policy import (
    EndpointRestriction,
    OmniServingCapability,
)
from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)
from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config

_PROC = "vllm_omni.model_executor.stage_input_processors.cosyvoice3"

COSYVOICE3_PIPELINE = PipelineConfig(
    model_type="cosyvoice3",
    default_deploy_config_name="cosyvoice3.yaml",
    model_arch="CosyVoice3Model",
    endpoint_restrictions=(
        EndpointRestriction(
            OmniServingCapability.COMPLETIONS,
            "CosyVoice3 does not support the Completions API.",
        ),
        EndpointRestriction(
            OmniServingCapability.CHAT_COMPLETIONS,
            "CosyVoice3 does not support the Chat Completions API.",
        ),
    ),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="cosyvoice3_talker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=(f"{_PROC}.talker2code2wav_async_chunk"),
            custom_process_next_stage_input_func=f"{_PROC}.text2flow_full_payload",
            sampling_constraints={
                # Standard sampling can emit any of the 200 control tokens.
                # RAS merges their logits into 6562, which is also in this range.
                "stop_token_ids": list(range(6561, 6761)),
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="cosyvoice3_code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="latent",
            sync_process_input_func=f"{_PROC}.text2flow_token_only",
            requires_full_payload_input=True,
        ),
    ),
)


def resolve_cosyvoice3_pipeline(hf_config: PretrainedConfig | None = None) -> PipelineConfig | None:
    """Select a streaming default compatible with the device and runtime.

    Query the platform through NVML without initializing CUDA in the parent.
    Explicit deploy configs still take precedence over this pipeline default.
    """
    from vllm.platforms import current_platform

    if hf_config is not None and not isinstance(hf_config, CosyVoice3Config):
        return None
    if current_platform.is_cuda():
        capability = current_platform.get_device_capability()
        if (
            capability is not None
            and capability.major == 9
            and current_platform.get_device_total_memory() >= 140 * 1024**3
            and shutil.which("nvidia-cuda-mps-control") is not None
        ):
            return replace(
                COSYVOICE3_PIPELINE,
                default_deploy_config_name="cosyvoice3_packed_streaming_optimized_standard.yaml",
            )
    return COSYVOICE3_PIPELINE
