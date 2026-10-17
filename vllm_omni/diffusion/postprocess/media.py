# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import replace

from vllm_omni.diffusion.media import (
    DiffusionMediaOutput,
    FloatVideoConsumer,
    VideoTensorEncoding,
    VideoTensorLayout,
    VideoTransportConstraints,
    VideoValueRange,
)
from vllm_omni.diffusion.postprocess.rife_interpolator import interpolate_video_tensor
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def finalize_diffusion_media(
    media: DiffusionMediaOutput,
    *,
    sampling_params: OmniDiffusionSamplingParams,
) -> dict[str, object]:
    """Finalize typed video media without invoking a model postprocessor."""
    media.validate()
    if not media.prepared_for_transport:
        raise ValueError("Diffusion media reached the engine before transport preparation")

    video = media.video
    metadata: dict[str, object] = {}
    consumers = video.constraints.pending_float_consumers
    if FloatVideoConsumer.FRAME_INTERPOLATION in consumers:
        if video.spec.encoding is not VideoTensorEncoding.NORMALIZED_FLOAT:
            raise ValueError("Frame interpolation requires NORMALIZED_FLOAT video")
        # RIFE guesses [0,1] vs [-1,1] from the tensor's min/max, so relying on
        # the sample content misclassifies both an all-nonnegative [-1,1] clip
        # (read as [0,1]) and a [0,1] clip whose single bf16 value overshoots 1
        # (read as [-1,1]). Always map to unit range per the declared spec and
        # clamp to [0,1] before RIFE, then restore the declared range so
        # downstream denormalization is correct regardless of the sample.
        to_negative_one = video.spec.value_range is VideoValueRange.NEGATIVE_ONE_TO_ONE
        if to_negative_one:
            interp_input = video.tensor.mul(0.5).add(0.5).clamp_(0.0, 1.0)
        else:
            interp_input = video.tensor.clamp(0.0, 1.0)
        interpolated, multiplier = interpolate_video_tensor(
            interp_input,
            exp=sampling_params.frame_interpolation_exp,
            scale=sampling_params.frame_interpolation_scale,
            model_path=sampling_params.frame_interpolation_model_path,
        )
        if to_negative_one:
            interpolated = interpolated.mul(2.0).sub(1.0)
        consumers = consumers - {FloatVideoConsumer.FRAME_INTERPOLATION}
        video = replace(
            video,
            tensor=interpolated,
            constraints=VideoTransportConstraints(pending_float_consumers=frozenset(consumers)),
        )
        metadata["video"] = {"video_fps_multiplier": multiplier}

    if consumers:
        names = sorted(consumer.value for consumer in consumers)
        raise ValueError(f"Video has pending float consumers at finalization: {names}")

    output_type = getattr(sampling_params, "output_type", None) or "np"
    if output_type == "latent":
        raise ValueError("Latent output cannot be finalized from VideoMediaOutput")

    if video.spec.encoding is VideoTensorEncoding.UINT8_FRAMES:
        if video.spec.layout is not VideoTensorLayout.BTHWC:
            raise ValueError("UINT8_FRAMES video must use BTHWC layout")
        if output_type != "np":
            raise ValueError(f"UINT8_FRAMES finalization does not support output_type={output_type!r}")
        frames = video.tensor.detach().cpu().numpy()
    else:
        if video.spec.layout is not VideoTensorLayout.BCTHW:
            raise ValueError("NORMALIZED_FLOAT video must use BCTHW layout")
        from diffusers.video_processor import VideoProcessor

        do_denormalize = video.spec.value_range is VideoValueRange.NEGATIVE_ONE_TO_ONE
        frames = VideoProcessor(vae_scale_factor=8).postprocess_video(
            video.tensor,
            output_type=output_type,
            do_denormalize=[do_denormalize] * video.tensor.shape[2],
        )

    return {
        "payload": {"video": frames},
        "metadata": metadata,
    }
