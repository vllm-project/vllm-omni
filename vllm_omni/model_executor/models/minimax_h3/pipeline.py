# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Two-stage MiniMax H3 topology."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROCESSOR = "vllm_omni.model_executor.stage_input_processors.minimax_h3"
_CHECKPOINT = "vllm_omni.model_executor.models.minimax_h3.checkpoint"
_DIFFUSION_PIPELINE = "vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3"

MINIMAX_H3_PIPELINE = PipelineConfig(
    model_type="minimax_h3_disaggregated",
    default_deploy_config_name="minimax_h3_disaggregated.yaml",
    stage_cli_aliases={"text_encoder_tp_size": (0, "tensor_parallel_size")},
    model_arch="MiniMaxH3TextEncoder",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="text_encoder",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            requires_multimodal_data=True,
            model_arch="MiniMaxH3TextEncoder",
            engine_output_type="latent",
            prompt_transform_func=f"{_PROCESSOR}.prepare_text_encoder_prompt",
            sampling_constraints={
                "max_tokens": 1,
                "temperature": 0.0,
                "detokenize": False,
            },
            model_path_resolver=f"{_CHECKPOINT}.resolve_minimax_h3_model_root",
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,),
            final_output=True,
            final_output_type="video",
            requires_multimodal_data=True,
            model_arch="MiniMaxH3Pipeline",
            custom_process_input_func=f"{_PROCESSOR}.text_encoder2diffusion",
            omni_kv_config={"need_recv_cache": False},
            model_path_resolver=f"{_DIFFUSION_PIPELINE}.resolve_minimax_h3_diffusion_model_path",
            inline_diffusion=True,
        ),
    ),
)
