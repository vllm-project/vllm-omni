# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SenseNova-U1 pipeline topologies.

Two-stage (default):
  Stage 0: Thinker — multimodal understanding + text generation (AR)
  Stage 1: DiT     — diffusion image generation

Single-stage:
  Stage 0: DiT — self-contained diffusion stage that handles all modalities
           (text2img, img2img, img2text, text2text, think) internally via its
           own LLM, ViT, VAE, and tokenizer.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.sensenova_u1"

SENSENOVA_U1_PIPELINE = PipelineConfig(
    model_type="sensenova_u1",
    model_arch="OmniSenseNovaU1ForCausalLM",
    hf_architectures=(),
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
            model_arch="OmniSenseNovaU1ForCausalLM",
            hf_config_name="llm_config",
            engine_output_type="text",
            prompt_expand_func=f"{_PROC}.expand_cfg_prompts",
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
            model_arch="SenseNovaU1Pipeline",
            requires_multimodal_data=True,
            cfg_kv_collect_func=f"{_PROC}.collect_cfg_kv_caches",
            omni_kv_config={"need_recv_cache": True},
        ),
    ),
)

SENSENOVA_U1_SINGLE_STAGE_PIPELINE = PipelineConfig(
    model_type="sensenova_u1_single_stage",
    model_arch="SenseNovaU1Pipeline",
    hf_architectures=("NEOChatModel",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="image",
            model_arch="SenseNovaU1Pipeline",
            requires_multimodal_data=True,
        ),
    ),
)
