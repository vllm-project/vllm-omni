# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek Janus single-stage omni pipeline topology.

Stage 0 runs :class:`JanusPipeline` in a diffusion worker. The pipeline keeps
Janus prompt formatting, CFG pairing, AR image-token generation, and VQ decode
inside one stage.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

DEEPSEEK_JANUS_SINGLE_STAGE_PIPELINE = PipelineConfig(
    model_type="deepseek_janus_single_stage",
    default_deploy_config_name="deepseek_janus_single_stage.yaml",
    model_arch="MultiModalityCausalLM",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            model_arch="JanusPipeline",
            input_sources=(),
            final_output=True,
            final_output_type="image",
        ),
    ),
)
