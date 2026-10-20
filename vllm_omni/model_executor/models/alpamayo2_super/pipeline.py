# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Alpamayo 2 Super policy topology: reasoning stage, then the trajectory expert."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

ALPAMAYO2_SUPER_PIPELINE = PipelineConfig(
    model_type="alpamayo2_super",
    default_deploy_config_name="alpamayo2_super.yaml",
    model_arch="Alpamayo2Super",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="reasoning",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="text",
            # Sent when the request finishes, after <|traj_future_start|>'s KV is written.
            omni_kv_config={"need_send_cache": True},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="expert",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,),
            final_output=True,
            final_output_type="actions",
            model_arch="Alpamayo2SuperPipeline",
            custom_process_input_func="vllm_omni.model_executor.stage_input_processors.alpamayo2_super.reasoning2expert",
            omni_kv_config={"need_recv_cache": True},
        ),
    ),
)
