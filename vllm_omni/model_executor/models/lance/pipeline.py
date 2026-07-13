# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lance pipeline topology.

Single-stage DiT — self-contained diffusion stage that handles all modalities
(text2img, image_edit, x2t_image, video) internally via its own Qwen2-MoT
LLM, Qwen2.5-VL ViT, Wan2.2 VAE, and tokenizer. Mirrors ``bagel_single_stage``.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

LANCE_PIPELINE = PipelineConfig(
    model_type="lance",
    default_deploy_config_name="lance.yaml",
    model_arch="LancePipeline",
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
