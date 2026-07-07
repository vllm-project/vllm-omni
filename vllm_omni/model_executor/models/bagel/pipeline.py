# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BAGEL-7B-MoT pipeline topologies (frozen).

Two-stage (default):
  Stage 0: Thinker — multimodal understanding + text generation (AR)
  Stage 1: DiT     — diffusion image generation

Two-stage think:
  Same as two-stage but the Thinker decodes <think>...</think> tokens before
  KV transfer.  Uses expand_cfg_prompts_think (companion max_tokens=1) and
  omits kv_transfer_criteria so transfer happens after EOS, not after prefill.

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

_PROC = "vllm_omni.model_executor.stage_input_processors.bagel"

BAGEL_PIPELINE = PipelineConfig(
    model_type="bagel",
    model_arch="OmniBagelForConditionalGeneration",
    hf_architectures=("BagelForConditionalGeneration",),
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
            model_arch="OmniBagelForConditionalGeneration",
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
            cfg_kv_collect_func=f"{_PROC}.collect_cfg_kv_caches",
            omni_kv_config={"need_recv_cache": True},
        ),
    ),
)

BAGEL_THINK_PIPELINE = PipelineConfig(
    model_type="bagel_think",
    model_arch="OmniBagelForConditionalGeneration",
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
            model_arch="OmniBagelForConditionalGeneration",
            engine_output_type="text",
            prompt_expand_func=f"{_PROC}.expand_cfg_prompts_think",
            omni_kv_config={"need_send_cache": True},
            sampling_constraints={"detokenize": True},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,),
            final_output=True,
            final_output_type="image",
            cfg_kv_collect_func=f"{_PROC}.collect_cfg_kv_caches",
            omni_kv_config={"need_recv_cache": True},
        ),
    ),
)

BAGEL_SINGLE_STAGE_PIPELINE = PipelineConfig(
    model_type="bagel_single_stage",
    model_arch="BagelForConditionalGeneration",
    hf_architectures=(),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="image",
        ),
    ),
)
