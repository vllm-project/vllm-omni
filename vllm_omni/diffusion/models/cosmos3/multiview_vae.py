# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VAE encode and decode of the Cosmos3 multiview pipeline as owner-scheduled tasks.

Every encoded camera field, the LiDAR control and the LiDAR condition are one
task each; so are every decoded camera and the LiDAR. ``OwnerScheduler``
decides which rank runs each task.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from functools import cached_property, partial
from pathlib import Path
from typing import Any

import torch

from vllm_omni.model_extras.cosmos3_lidar import load_lidar_frames, required_lidar_sweeps

from .transfer import IMAGE_EXTENSIONS, _pil_to_uint8_rgb, media_to_uint8_cthw, resize_center_crop_uint8_cthw
from .vae_owner import OwnerScheduler, VaeTask

# Measured LiDAR sweeps conditioned when lidar.condition_path omits a count,
# as in the reference inference.
COSMOS3_MULTIVIEW_DEFAULT_LIDAR_CONDITION_SWEEPS = 1
# Relative VAE cost per input pixel-frame, normalized to one camera decode.
# Only used to place tasks; never affects numerics.
_CAMERA_ENCODE_COST = 0.9
_CAMERA_DECODE_COST = 1.0
_LIDAR_ENCODE_COST = 6.7
_LIDAR_DECODE_COST = 6.8


class MultiviewVaeMixin:
    """Camera and LiDAR VAE work of ``Cosmos3MultiviewPipeline``."""

    @cached_property
    def _vae_scheduler(self) -> OwnerScheduler:
        return OwnerScheduler.create(self.device, self.vae)

    def _encode_inputs(
        self,
        *,
        views: Sequence[Mapping[str, Any]],
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        vision_views: Sequence[int],
        keep_first: bool,
        require_complete: bool,
        encode_control: bool,
        lidar_request: Mapping[str, Any] | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Encode the request inputs; every rank returns the same latents.

        Returns camera-major vision latents, camera-major control latents, LiDAR
        control latents and LiDAR condition latents, each None when absent.
        Vision is encoded only for ``vision_views`` and is zero for other views.
        """
        latent_shape = (
            1,
            self.transformer.latent_channel_size,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        fields = [("vision", index) for index in vision_views]
        if encode_control:
            fields += [("control", index) for index in range(len(views))]
        tasks = [
            VaeTask(
                f"{field}/{index}",
                _CAMERA_ENCODE_COST * num_frames * height * width,
                latent_shape,
                self.sampling_dtype,
                run=self._encode_video_tensor,
                prepare=partial(
                    self._prepare_camera_major_pixels,
                    [views[index]],
                    field=field,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    keep_first=keep_first and field == "vision",
                    require_complete=require_complete and field == "vision",
                    load_frames=stream_media_to_uint8_cthw,
                ),
            )
            for field, index in fields
        ]
        if lidar_request is not None:
            config = self.lidar_encoder.config
            sweeps = required_lidar_sweeps(num_frames, frame_rate, config["fps"])
            h, w = config["network_config"]["resolution"]
            sh, sw = config["spatial_compression"]
            lidar_shape = (1, config["latent_channels"], sweeps, h // sh, w // sw)
            tasks.append(
                VaeTask(
                    "lidar/control",
                    _LIDAR_ENCODE_COST * sweeps * h * w,
                    lidar_shape,
                    self.dtype,
                    run=lambda frames: self.lidar_encoder(frames).to(device=self.device, dtype=self.dtype),
                    prepare=partial(load_lidar_frames, lidar_request["control_path"], num_sweeps=sweeps),
                )
            )
            if lidar_request.get("condition_path") is not None:
                # Validated on every rank before any rank starts encoding.
                count = _condition_sweeps(lidar_request, sweeps)
                chunk = config["streaming_chunk_frames"]
                padded = min(math.ceil(count / chunk) * chunk, sweeps)
                tasks.append(
                    VaeTask(
                        "lidar/condition",
                        _LIDAR_ENCODE_COST * padded * h * w,
                        (*lidar_shape[:2], count, *lidar_shape[3:]),
                        self.sampling_dtype,
                        run=lambda _: self._encode_lidar_condition(lidar_request, sweeps, lidar_shape),
                    )
                )

        results = self._vae_scheduler.run_on_all(tasks)
        vision = control = lidar_control = None
        if vision_views:
            blank = torch.zeros(latent_shape, dtype=self.sampling_dtype, device=self.device)
            vision = torch.cat([results.get(f"vision/{index}", blank) for index in range(len(views))], dim=2)
        if encode_control:
            control = torch.cat([results[f"control/{index}"] for index in range(len(views))], dim=2)
        if lidar_request is not None:
            # The LiDAR encoder returns channels-last storage, which transport does
            # not keep; the transformer's projection GEMM rounds differently without it.
            lidar_control = results["lidar/control"].contiguous(memory_format=torch.channels_last_3d)
        return vision, control, lidar_control, results.get("lidar/condition")

    def _decode_outputs(
        self,
        final_targets: Sequence[torch.Tensor],
        *,
        num_views: int,
        num_frames: int,
        height: int,
        width: int,
        return_lidar: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        """Decode every camera, and the LiDAR when ``return_lidar``, onto the output rank.

        The output rank gets the camera-major video clamped to [-1, 1] and the
        float32 LiDAR, both on the CPU. Other ranks get None.
        """
        latents = final_targets[0]
        latent_frames = latents.shape[2] // num_views
        tasks = [
            VaeTask(
                f"view/{index}",
                _CAMERA_DECODE_COST * num_frames * height * width,
                (1, 3, num_frames, height, width),
                self.vae.dtype,
                run=lambda _, view=latents.narrow(2, index * latent_frames, latent_frames): self._decode_latents(view),
            )
            for index in range(num_views)
        ]
        if return_lidar:
            config = self.lidar_decoder.config
            h, w = config["network_config"]["resolution"]
            projection = config["range_projection"]
            lidar_latents = final_targets[1]
            lidar_shape = (1, 3, lidar_latents.shape[2], projection["native_height"], projection["semantic_width"])
            tasks.append(
                VaeTask(
                    "lidar",
                    _LIDAR_DECODE_COST * lidar_latents.shape[2] * h * w,
                    lidar_shape,
                    torch.float32,
                    run=lambda _: self.lidar_decoder(lidar_latents),
                )
            )

        scheduler = self._vae_scheduler
        if not scheduler.is_output_rank:
            scheduler.run_to_output(tasks, None)
            return None
        # Pinned host memory lets transported results copy asynchronously.
        pin = scheduler.distributed
        video = torch.empty((1, 3, num_views * num_frames, height, width), dtype=self.vae.dtype, pin_memory=pin)
        outputs = {f"view/{index}": video.narrow(2, index * num_frames, num_frames) for index in range(num_views)}
        lidar = None
        if return_lidar:
            lidar = outputs["lidar"] = torch.empty(lidar_shape, dtype=torch.float32, pin_memory=pin)
        scheduler.run_to_output(tasks, outputs)
        return video.clamp_(-1, 1), lidar

    def _encode_lidar_condition(
        self, lidar_request: Mapping[str, Any], num_sweeps: int, target_shape: Sequence[int]
    ) -> torch.Tensor:
        """Encode the measured sweeps that condition the start of the generated LiDAR.

        The reference encodes the prefix inside an otherwise empty (zero) target
        clip. The streaming tokenizer is frame-causal within a chunk, but how much
        history a chunk keeps depends on that chunk's length, so encoding the bare
        prefix would change the latents of a trailing partial chunk. Zero-padding
        the prefix to the target's chunk boundary reproduces the full clip's chunk
        lengths for every chunk holding prefix sweeps; the causal padding cannot
        affect the prefix latents.
        """
        count = _condition_sweeps(lidar_request, num_sweeps)
        frames = load_lidar_frames(lidar_request["condition_path"], num_sweeps=count)
        chunk = self.lidar_encoder.config["streaming_chunk_frames"]
        padded = min(math.ceil(count / chunk) * chunk, num_sweeps)
        if padded > count:
            frames = torch.cat([frames, frames.new_zeros(frames.shape[0], padded - count, *frames.shape[2:])], dim=1)
        latents = self.lidar_encoder(frames)[:, :, :count].to(device=self.device, dtype=self.sampling_dtype)
        expected = (*target_shape[:2], count, *target_shape[3:])
        if tuple(latents.shape) != expected:
            raise ValueError(f"Cosmos3 LiDAR condition latents must have shape {expected}, got {tuple(latents.shape)}.")
        return latents


def _condition_sweeps(lidar_request: Mapping[str, Any], num_sweeps: int) -> int:
    count = lidar_request.get("num_conditional_sweeps", COSMOS3_MULTIVIEW_DEFAULT_LIDAR_CONDITION_SWEEPS)
    if not 0 < count < num_sweeps:
        raise ValueError(
            f"Cosmos3 lidar.num_conditional_sweeps={count} must leave at least one of the request's "
            f"{num_sweeps} LiDAR sweeps to generate."
        )
    return count


def stream_media_to_uint8_cthw(value: Any, *, height: int, width: int, max_frames: int | None = None) -> torch.Tensor:
    """``media_to_uint8_cthw`` that resizes each video frame while the next one decodes.

    One worker thread resizes frame by frame, with at most one decoded frame
    waiting for it. Resizing is independent per frame, so the result equals
    resizing the whole clip. Other media use ``media_to_uint8_cthw`` unchanged.
    """
    path = Path(value) if isinstance(value, str | Path) else None
    if path is None or path.suffix.lower() in IMAGE_EXTENSIONS or not path.exists():
        return media_to_uint8_cthw(value, height=height, width=width, max_frames=max_frames)
    import imageio.v3 as iio

    resized: list[Future] = []
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="cosmos3_resize") as pool:
        for frame in iio.imiter(path):
            if len(resized) > 1:
                resized[-2].result()
            # The same frame conversion and layout as decode_path_video_frames.
            frame = torch.from_numpy(_pil_to_uint8_rgb(frame)).permute(2, 0, 1).unsqueeze(1).contiguous()
            resized.append(pool.submit(resize_center_crop_uint8_cthw, frame, height, width))
            if len(resized) == max_frames:
                break
    if not resized:
        raise ValueError(f"Video path produced no frames: {path}")
    return torch.cat([future.result() for future in resized], dim=1)
