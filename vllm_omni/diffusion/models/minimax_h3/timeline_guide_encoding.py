# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Timeline-guide admission and VAE encoding for the MiniMax H3 pipeline.

Split out of ``pipeline_minimax_h3`` so guide profile validation, row-budget
checks and the multi-rank guide encoding collectives live in one reviewable
place instead of inside the 3.5k-line pipeline module.

The filename deliberately differs from
``vllm_omni.model_executor.models.minimax_h3.timeline_guides`` (the CPU-only
admission/decoding policy module): the pipeline imports both, so a shared name
would force import aliasing.

``MiniMaxH3TimelineGuideMixin`` is a pure behavior carrier. It reads pipeline
state (``od_config``, ``_fasth3``, ``lora_is_fused``, ``video_vae``,
``audio_vae``, ``device``) and pipeline helpers (``_sigma_schedule_for_request``,
``_transformer_for_task``, ``_component_on_device``) off ``self``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.models.minimax_h3.timeline_guides import (
    TimelineGuideBudget,
    TimelineGuideLimits,
    decode_guide_audio,
    decode_guide_image,
    decode_guide_video,
    guide_audio_limit,
    resolve_guide_start,
)

from .distributed_errors import (
    _broadcast_rank0_exception,
    _broadcast_tensor,
    _dit_rank_world,
    _synchronize_any_rank_exception,
)
from .minimax_h3_transformer import MiniMaxH3Attention
from .time_request import MINIMAX_H3_SHAPE_PLANNER


class MiniMaxH3TimelineGuideMixin:
    """Guide-profile validation, row budgets and guide VAE encoding."""

    def _validate_timeline_guide_profile(self, sampling: Any, task: str, quality_plan: Any) -> None:
        if quality_plan.cache_dit is not None:
            raise OmniClientError("MiniMax H3 timeline guides require cache-free execution; use quality=lossless")
        cache_backend = str(getattr(self.od_config, "cache_backend", "none") or "none").lower()
        if cache_backend not in ("none", "cache_dit"):
            raise OmniClientError("MiniMax H3 timeline guides do not support cache acceleration")
        if (
            self._fasth3 is not None
            or self.lora_is_fused
            or getattr(self, "_lora_is_fused", False)
            # LoraLoaderMixin records this only after in-place weight addition;
            # the dynamic manager's inactive registry is _registered_adapters.
            or getattr(self, "_lora_loaded", None)
        ):
            raise OmniClientError("MiniMax H3 timeline guides require base H3 weights, not FastH3 or fused LoRA")
        if sampling.lora_request is not None and not math.isclose(float(sampling.lora_scale), 0.0):
            raise OmniClientError("MiniMax H3 timeline guides do not support active LoRA/Turbo adapters")
        if self._sigma_schedule_for_request(sampling, task) is not None:
            raise OmniClientError("MiniMax H3 timeline guides do not support distilled schedules")
        # Inspect resolved modules, including the token-refiner attention role,
        # rather than the startup default (which role overrides can replace).
        dense_backends = {
            "FLASH_ATTN",
            "SDPA",
            "SAGE_ATTN",
            "SAGE_ATTN_3",
            "CUDNN_ATTN",
            "FLASHINFER_ATTN",
            "FLASH_ATTN_HUB",
            "FLASH_ATTN_3_HUB",
            "TRTLLM_ATTN",
        }
        for module in self._transformer_for_task(task).modules():
            if any(getattr(module, "_diffusion_lora_active_slices", ())):
                raise OmniClientError("MiniMax H3 timeline guides do not support active LoRA adapters")
            if isinstance(module, MiniMaxH3Attention):
                backend = module.attention.attn_backend.get_name()
                if backend not in dense_backends:
                    raise OmniClientError(
                        f"MiniMax H3 timeline guides require dense attention in every role; got {backend}"
                    )
                if backend == "TRTLLM_ATTN":
                    from vllm_omni.diffusion.attention.backends.trtllm_attn import QuantConfig, SkipSoftmaxConfig

                    attention = module.attention
                    implementation = attention.attention
                    spec = getattr(attention, "attn_spec", None)
                    backend_kwargs = spec.backend_kwargs() if spec is not None else None
                    if (
                        implementation.quant.enabled
                        or implementation.skip.configured
                        or QuantConfig.from_backend_kwargs(backend_kwargs).enabled
                        or SkipSoftmaxConfig.from_backend_kwargs(backend_kwargs).configured
                    ):
                        raise OmniClientError(
                            "MiniMax H3 timeline guides require dense unquantized TRTLLM_ATTN in every role; "
                            "disable quant and skip_softmax/target_sparsity options"
                        )

    @staticmethod
    def _check_timeline_rows(limits: TimelineGuideLimits, guide_rows: int, other_rows: int) -> int:
        packed_rows = ((guide_rows + other_rows + 63) // 64) * 64
        if guide_rows > limits.max_guide_rows:
            raise OmniClientError("MiniMax H3 timeline guides exceed the guide-row limit; trim guides")
        if packed_rows > limits.max_packed_rows:
            raise OmniClientError("MiniMax H3 timeline guides exceed the packed-request row limit; trim inputs")
        return packed_rows

    def _encode_timeline_guides(
        self,
        descriptors: list[dict[str, Any]],
        *,
        width: int,
        height: int,
        num_frames: int,
        limits: TimelineGuideLimits,
        other_rows: int,
        keyframe_rows: int = 0,
    ) -> tuple[list[dict[str, Any]], torch.Tensor | None, list[tuple[int, int, int]], torch.Tensor | None, list[int]]:
        group, rank, world_size = _dit_rank_world()
        decoded: list[dict[str, Any]] = []
        blocks: list[dict[str, Any]] = []
        prep_error = None
        if rank == 0:
            try:
                budget = TimelineGuideBudget(limits)
                guide_rows = keyframe_rows
                other_rows -= keyframe_rows
                for descriptor in descriptors:
                    frames = None
                    audio = None
                    if "image" in descriptor:
                        frames = [decode_guide_image(descriptor["image"], width, height, limits, budget=budget)]
                    elif "video" in descriptor:
                        frames = decode_guide_video(
                            descriptor["video"], width, height, num_frames, limits, budget=budget
                        )
                    count = len(frames) if frames is not None else 1
                    start = resolve_guide_start(descriptor["frame_index"], num_frames, count)
                    block: dict[str, Any] = {"frame_index": start}
                    if frames is not None:
                        vt = 1 if count == 1 else MINIMAX_H3_SHAPE_PLANNER.video_latent_t(count)
                        block.update(
                            kind="image" if "image" in descriptor else "video",
                            latent_t=vt,
                            latent_h=height // 16,
                            latent_w=width // 16,
                            ref_audio_t=0,
                        )
                        guide_rows += vt * (height // 32) * (width // 32)
                    if "audio" in descriptor:
                        audio = decode_guide_audio(descriptor["audio"], limits, budget=budget)
                        # Audio VAE preprocess may pad to its stride. Reserve the
                        # whole output remainder, then verify the actual cropped T.
                        limit_t = guide_audio_limit(num_frames, start)
                        if limit_t < 1:
                            raise OmniClientError("timeline guide leaves no audio latent positions")
                        block.update(kind="video_audio" if frames is not None else "audio", ref_audio_t=limit_t)
                        guide_rows += 2 * limit_t
                    self._check_timeline_rows(limits, guide_rows, other_rows)
                    blocks.append(block)
                    decoded.append({"frames": frames, "audio": audio})
            except Exception as exc:
                prep_error = OmniClientError(str(exc)) if isinstance(exc, ValueError) else exc
        _broadcast_rank0_exception(prep_error)
        if world_size > 1:
            payload = [blocks]
            dist.broadcast_object_list(payload, src=0, group=group)
            blocks = payload[0]

        visual_parts, audio_parts = [], []
        visual_shapes, audio_lengths = [], []
        # A patch-parallel video VAE runs collectives inside ``encode_image`` as
        # well as ``encode_video``, so participation follows the codec, never
        # the guide's modality. Encoding a still on rank 0 alone would strand
        # its peers in the next broadcast. The audio WVAE has no such group.
        distributed_video = self.video_vae.is_distributed_enabled()
        for index, block in enumerate(blocks):
            if block["kind"] != "audio":
                frames = decoded[index]["frames"] if rank == 0 else None
                is_clip = "video" in descriptors[index]
                if distributed_video:
                    payload = [frames]
                    dist.broadcast_object_list(payload, src=0, group=group)
                    frames = payload[0]
                rows = None
                encode_error = None
                shape = (block["latent_t"], block["latent_h"], block["latent_w"])
                try:
                    if rank == 0 or distributed_video:
                        with self._component_on_device(self.video_vae):
                            if is_clip:
                                video_frames = np.stack([np.asarray(frame) for frame in frames])
                                rows, actual_shape = self.video_vae.encode_video(video_frames)
                                if tuple(actual_shape) != shape:
                                    raise OmniClientError(
                                        f"timeline guide video latent shape {actual_shape} != {shape}"
                                    )
                            else:
                                rows = self.video_vae.encode_image(frames[0])
                        expected = shape[0] * (shape[1] // 2) * (shape[2] // 2)
                        if tuple(rows.shape) != (expected, 96):
                            raise OmniClientError("timeline guide visual VAE returned an unexpected row shape")
                except Exception as exc:
                    encode_error = exc
                if distributed_video:
                    _synchronize_any_rank_exception(encode_error)
                else:
                    _broadcast_rank0_exception(encode_error)
                visual_parts.append(_broadcast_tensor(rows, dtype=torch.float32, device=self.device))
                visual_shapes.append(shape)
            if block["kind"] in ("audio", "video_audio"):
                rows = None
                encode_error = None
                length = 0
                if rank == 0:
                    try:
                        waveform, sample_rate = decoded[index]["audio"]
                        with self._component_on_device(self.audio_vae):
                            rows, length = self.audio_vae.encode_waveform(torch.from_numpy(waveform), sample_rate)
                        if length < 1 or tuple(rows.shape) != (2 * length, 32):
                            raise OmniClientError("timeline guide audio VAE returned empty or invalid stereo rows")
                        cropped_t = min(length, block["ref_audio_t"])
                        # VAE rows are channel-major: crop time independently for
                        # both channels, never slice the already-flattened prefix.
                        rows = rows.reshape(2, length, 32)[:, :cropped_t].reshape(-1, 32)
                        length = cropped_t
                    except Exception as exc:
                        encode_error = exc
                _broadcast_rank0_exception(encode_error)
                rows = _broadcast_tensor(rows, dtype=torch.float32, device=self.device)
                length = rows.shape[0] // 2
                block["ref_audio_t"] = length
                audio_parts.append(rows)
                audio_lengths.append(length)
        return (
            blocks,
            torch.cat(visual_parts) if visual_parts else None,
            visual_shapes,
            torch.cat(audio_parts) if audio_parts else None,
            audio_lengths,
        )
