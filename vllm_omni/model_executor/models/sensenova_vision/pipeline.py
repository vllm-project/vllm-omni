# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SenseNova-Vision-7B-MoT pipeline topologies.

Two-stage (default):
  Stage 0: Thinker — multimodal understanding + text generation (AR)
  Stage 1: DiT     — diffusion image generation

Two-stage think:
  Same as two-stage but the Thinker decodes <thinking> tokens to EOS before
  the KV is transferred.  Uses expand_cfg_prompts_think (companion
  max_tokens=1) and omits kv_transfer_criteria so transfer happens after EOS.
  A stage-1 custom_process_input_func lifts the AR text into the DiT request's
  extra_args['text_output'] so it surfaces under the existing {image, text}
  output-modality contract.

This mirrors the BAGEL two-stage / two-stage-think pipelines but is registered
under ``model_type="sensenova_vision"`` / ``"sensenova_vision_think"`` and uses
the SenseNovaVision model architecture string.
The stage-transition functions are reused verbatim from the BAGEL processor
module (``expand_cfg_prompts`` / ``expand_cfg_prompts_think`` /
``collect_cfg_kv_caches``).
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.bagel"
_SNV_PROMPT = "vllm_omni.model_executor.models.sensenova_vision.prompt_utils"

SENSENOVA_VISION_PIPELINE = PipelineConfig(
    model_type="sensenova_vision",
    default_deploy_config_name="sensenova_vision.yaml",
    model_arch="OmniSenseNovaVisionForConditionalGeneration",
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
            model_arch="OmniSenseNovaVisionForConditionalGeneration",
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
SENSENOVA_VISION_THINK_PIPELINE = PipelineConfig(
    model_type="sensenova_vision_think",
    default_deploy_config_name="sensenova_vision_think.yaml",
    model_arch="OmniSenseNovaVisionForConditionalGeneration",
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
            model_arch="OmniSenseNovaVisionForConditionalGeneration",
            engine_output_type="text",
            prompt_expand_func=f"{_PROC}.expand_cfg_prompts_think",
            # The think topology does NOT transfer after prefill: the Thinker
            # decodes its <thinking> tokens to EOS first, so the KV sent to the
            # DiT includes the thought.  Hence no kv_transfer_criteria here
            # (mirrors BAGEL_THINK_PIPELINE).
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
            # Lift the AR stage's decoded think text into the DiT request's
            # extra_args['text_output'] so _merge_mixed_task_text surfaces it
            # under the existing {image, text} contract (no new payload keys).
            custom_process_input_func=(
                f"{_SNV_PROMPT}.bridge_think_text_to_image"
            ),
        ),
    ),
)
