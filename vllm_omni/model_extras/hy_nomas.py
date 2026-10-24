# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections.abc import Mapping

from vllm_omni.model_extras.video_generation import VideoGenerationDefaults

HY_NOMAS_MODEL_ID = "tencent/Hy-Nomas"
HY_NOMAS_DIT_FILE = "DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors"
HY_NOMAS_DECODER_FILE = "VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors"
HY_NOMAS_EXTRA_BODY_PARAMS = frozenset(
    {"segment_frame_boundaries", "camera_trajectory", "segment_has_camera", "flow_shift"}
)


def get_hy_nomas_video_generation_defaults(
    extra_body: Mapping[str, object] | None = None,
) -> VideoGenerationDefaults:
    return VideoGenerationDefaults(
        width=1280,
        height=704,
        num_frames=121,
        num_inference_steps=8,
        guidance_scale=1.0,
        flow_shift=5.0,
        fps=24.0,
        dimension_multiple=32,
        output="hy_nomas_output.mp4",
    )
