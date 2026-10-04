# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
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

BAGEL_CHAT_TEMPLATE = (
    "{%- macro frame(text) -%}"
    "{%- set t = text.removeprefix('<|im_start|>').removesuffix('<|im_end|>') -%}"
    "{%- if t -%}{{- '<|im_start|>' + t + '<|im_end|>' -}}{%- endif -%}"
    "{%- endmacro -%}"
    "{%- for message in messages -%}"
    "{%- if message['content'] is string -%}"
    "{{- frame(message['content']) -}}"
    "{%- else -%}"
    "{%- for content in message['content'] -%}"
    "{%- if content['type'] in ('image', 'image_url') -%}"
    "{{- '<|vision_start|><|image_pad|><|vision_end|>' -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- for content in message['content'] -%}"
    "{%- if 'text' in content -%}{{- frame(content['text']) -}}{%- endif -%}"
    "{%- endfor -%}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>' -}}{%- endif -%}"
)

BAGEL_PIPELINE = PipelineConfig(
    model_type="bagel",
    default_deploy_config_name="bagel.yaml",
    chat_template=BAGEL_CHAT_TEMPLATE,
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
            prompt_transform_func=f"{_PROC}.frame_prompt",
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
    default_deploy_config_name="bagel_think.yaml",
    chat_template=BAGEL_CHAT_TEMPLATE,
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
            prompt_transform_func=f"{_PROC}.frame_prompt",
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
    default_deploy_config_name="bagel_single_stage.yaml",
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
