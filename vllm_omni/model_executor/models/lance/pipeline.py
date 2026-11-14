# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
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
from vllm_omni.serving_capabilities import (
    IMAGE_RESPONSE,
    IMAGE_TO_IMAGE,
    TEXT_RESPONSE,
    TEXT_TO_IMAGE,
    TEXT_TO_VIDEO,
    VIDEO_TO_VIDEO,
    declare_capabilities,
)

# The video checkpoint uses a different VAE and the video API supplies source
# clips under multi_modal_data.video. Its first-frame I2V path instead expects
# multi_modal_data.first_frame, which the current video request adapter does
# not produce. Do not advertise that offline-only request mapping here.
LANCE_VIDEO_CAPABILITIES = declare_capabilities(
    TEXT_TO_VIDEO,
    VIDEO_TO_VIDEO,
    unknown_operations=("chat", "image_generation", "image_edit"),
    unknown_reason="video_checkpoint_image_contract_unverified",
    sources=(
        "vllm_omni/diffusion/models/lance/pipeline_lance.py",
        "vllm_omni/entrypoints/openai/serving_video.py",
    ),
)

LANCE_PIPELINE = PipelineConfig(
    serving_capabilities=declare_capabilities(
        TEXT_RESPONSE,
        IMAGE_RESPONSE,
        TEXT_TO_IMAGE,
        IMAGE_TO_IMAGE,
        sources=("vllm_omni/model_executor/models/lance/pipeline.py",),
    ),
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
