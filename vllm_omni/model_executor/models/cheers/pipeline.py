# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cheers (UMM) pipeline topology (frozen).

Two-stage:
  Stage 0: AR — multimodal understanding + text generation (Qwen2-based LLM)
  Stage 1: DiT — diffusion image generation (SigLIP2 + flow-matching denoiser)

KV cache from the AR prefill stage is transferred to the diffusion stage
via SharedMemoryConnector; the diffusion loop uses the cached context to
condition each denoising step.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

UMM_PIPELINE = PipelineConfig(
    model_type="umm",
    model_arch="CheersForConditionalGeneration",
    hf_architectures=("Cheers",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            model_arch="CheersForConditionalGeneration",
            engine_output_type="text",
            omni_kv_config={
                "need_send_cache": True,
                "kv_transfer_criteria": {"type": "prefill_finished"},
            },
            sampling_constraints={"detokenize": True},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,),
            final_output=True,
            final_output_type="image",
            omni_kv_config={"need_recv_cache": True},
        ),
    ),
)
