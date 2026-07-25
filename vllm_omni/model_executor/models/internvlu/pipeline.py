# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""InternVL-U's native AR-to-diffusion pipeline topology."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.internvlu"


INTERNVLU_PIPELINE = PipelineConfig(
    model_type="internvlu_chat",
    default_deploy_config_name="internvlu_chat.yaml",
    model_arch="InternVLUChatModel",
    hf_architectures=("InternVLUChatModel",),
    diffusers_class_name="InternVLUPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="vlm",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            model_arch="InternVLUChatModel",
            engine_output_type="latent",
            model_subdir="vlm",
            tokenizer_subdir="processor",
            prompt_expand_func=f"{_PROC}.expand_cfg_prompts",
            # Keep Stage-0 detokenization on even though its engine output is
            # latent conditioning: think-mode CoT text reaches the bridge (and
            # the response's cot_output) via ``completion.cumulative_text``,
            # which only exists when the stage detokenizes.
            sampling_constraints={"detokenize": True},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,),
            final_output=True,
            final_output_type="image",
            requires_multimodal_data=True,
            model_arch="InternVLUPipeline",
            custom_process_input_func=f"{_PROC}.ar2diffusion",
            omni_kv_config={"need_recv_cache": False},
        ),
    ),
)
