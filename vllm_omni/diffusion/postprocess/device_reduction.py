# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Device-side reduction of decoded video tensors to uint8 frames."""

from __future__ import annotations

from dataclasses import replace

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.media import (
    DiffusionMediaOutput,
    FloatVideoConsumer,
    VideoMediaOutput,
    VideoTensorEncoding,
    VideoTensorLayout,
    VideoTensorSpec,
    VideoTransportConstraints,
    VideoValueRange,
    ensure_request_owned_tensor,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

# VaeImageProcessor.denormalize: (x * 0.5 + 0.5).clamp(0, 1)
_DENORM_SCALE = 0.5
_DENORM_SHIFT = 0.5

logger = init_logger(__name__)


def _request_float_consumers(
    sampling_params: OmniDiffusionSamplingParams | None,
) -> frozenset[FloatVideoConsumer]:
    if sampling_params is not None and sampling_params.enable_frame_interpolation:
        return frozenset({FloatVideoConsumer.FRAME_INTERPOLATION})
    return frozenset()


def _request_supports_uint8_frames(sampling_params: OmniDiffusionSamplingParams | None) -> bool:
    return sampling_params is None or (sampling_params.output_type or "np") == "np"


def _prepare_float_media_for_transport(
    media: DiffusionMediaOutput,
    video: VideoMediaOutput,
) -> DiffusionMediaOutput:
    prepared = replace(
        media,
        video=video.with_tensor(ensure_request_owned_tensor(video.tensor)),
        prepared_for_transport=True,
    )
    prepared.validate()
    return prepared


def prepare_diffusion_media_for_transport(
    media: DiffusionMediaOutput,
    *,
    od_config: OmniDiffusionConfig,
    sampling_params: OmniDiffusionSamplingParams | None = None,
) -> DiffusionMediaOutput:
    """Validate and prepare one request-local media output before D2H."""
    media.validate()
    if media.prepared_for_transport:
        # Only the model runner prepares media, exactly once, on media the
        # pipeline emits unprepared. Accepting pre-prepared media here would let
        # a pipeline forge runner-owned state and skip request policy (e.g. a
        # forged uint8 payload silently bypasses frame interpolation).
        raise ValueError(
            "Media reached transport preparation already prepared; pipelines must "
            "emit unprepared media so request policy is applied by the runner"
        )
    if media.video.spec.encoding is not VideoTensorEncoding.NORMALIZED_FLOAT:
        raise ValueError("Pipelines must emit unprepared media as NORMALIZED_FLOAT video")

    video = media.video
    constraints = VideoTransportConstraints(
        pending_float_consumers=(video.constraints.pending_float_consumers | _request_float_consumers(sampling_params))
    )
    constrained_video = replace(video, constraints=constraints)

    enabled = od_config.video_output_transport.enable_device_postprocess
    output_supported = _request_supports_uint8_frames(sampling_params)
    if not enabled or constraints.pending_float_consumers or not output_supported:
        if not enabled:
            reason = "disabled"
        elif constraints.pending_float_consumers:
            reason = ",".join(sorted(consumer.value for consumer in constraints.pending_float_consumers))
        else:
            reason = "unsupported_presentation"
        logger.debug("Device video preparation kept normalized float: reason=%s", reason)
        return _prepare_float_media_for_transport(media, constrained_video)

    try:
        frames = reduce_video_to_uint8_frames(
            constrained_video.tensor,
            do_denormalize=constrained_video.spec.value_range is VideoValueRange.NEGATIVE_ONE_TO_ONE,
        )
    except torch.OutOfMemoryError:
        logger.warning("Device video preparation ran out of memory; using normalized float transport")
    else:
        prepared_video = replace(
            constrained_video,
            tensor=frames,
            spec=VideoTensorSpec(
                layout=VideoTensorLayout.BTHWC,
                encoding=VideoTensorEncoding.UINT8_FRAMES,
                value_range=VideoValueRange.ZERO_TO_255,
            ),
            constraints=VideoTransportConstraints(),
        )
        prepared = replace(media, video=prepared_video, prepared_for_transport=True)
        prepared.validate()
        logger.debug(
            "Device video preparation converted normalized float to uint8: shape=%s",
            tuple(frames.shape),
        )
        return prepared

    # Leave the exception handler before allocator cleanup and fallback
    # construction. A late conversion OOM otherwise keeps the helper traceback
    # (and its live float32 intermediates) reachable while the fallback may need
    # to clone a non-compact request view.
    current_omni_platform.empty_cache()
    current_omni_platform.synchronize()
    return _prepare_float_media_for_transport(media, constrained_video)


def reduce_video_to_uint8_frames(video: torch.Tensor, *, do_denormalize: bool = True) -> torch.Tensor:
    """Reduce a decoded ``[B, C, F, H, W]`` video to uint8 ``[B, F, H, W, C]`` frames.

    Runs denormalize/clamp/permute/round on the input's device so the following
    D2H copy carries uint8 instead of float. The result matches
    ``VideoProcessor.postprocess_video(output_type="np")`` then the ``*255``
    rounding done in the API server. Pass ``do_denormalize=False`` for VAEs that
    already emit ``[0, 1]``.
    """
    if video.dim() != 5:
        raise ValueError(f"expected a [B, C, F, H, W] video tensor, got shape {tuple(video.shape)}")

    # Match the numpy path, which promotes to float before scaling.
    frames = video.to(torch.float32)
    if do_denormalize:
        frames = frames.mul(_DENORM_SCALE).add(_DENORM_SHIFT).clamp_(0.0, 1.0)
    else:
        frames = frames.clamp(0.0, 1.0)
    frames = frames.permute(0, 2, 3, 4, 1)
    frames = frames.mul_(255.0).round_().clamp_(0.0, 255.0).to(torch.uint8)
    return frames.contiguous()
