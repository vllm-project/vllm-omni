# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright 2026 Lightricks and The HuggingFace Team. All rights reserved.
#
# The decoder modules are adapted from Diffusers at commit
# d035dcd7cc7c88e0a154609b62887d50bba9fdc2 (Apache-2.0). The distributed
# execution below independently extends LTX-2.5's spatial-only topology with
# shape-aware grid selection.

"""LTX-2.5-specific distributed execution for the diffusion VAE decoder."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import torch
import torch.distributed as dist
from diffusers.utils import logging

from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedOperator,
    DistributedVaeMixin,
    GridSpec,
    TileTask,
)

from .decoder import LTX2VideoDiffusionDecoderModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

LTX2_VAE_SPATIAL_OVERLAP = 4


def _split_axis_by_count(
    length: int,
    tile_count: int,
    overlap: int,
    *,
    min_tile_size: int = 1,
) -> tuple[tuple[int, int], ...]:
    """Split an axis into nearly equal overlapping intervals."""
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")
    if tile_count < 1:
        raise ValueError(f"tile_count must be >= 1, got {tile_count}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if min_tile_size < 1:
        raise ValueError(f"min_tile_size must be >= 1, got {min_tile_size}")
    if tile_count > length:
        raise ValueError(
            f"Cannot split length {length} into {tile_count} tiles: each tile needs at least one latent cell."
        )
    if length < min_tile_size:
        raise ValueError(
            f"Cannot split length {length} with minimum tile size {min_tile_size}: the complete axis is too short."
        )
    if tile_count == 1:
        return ((0, length),)

    covered_length = length + overlap * (tile_count - 1)
    base_size, larger_tiles = divmod(covered_length, tile_count)
    if base_size <= overlap:
        raise ValueError(
            f"Cannot split length {length} into {tile_count} tiles with "
            f"overlap {overlap}: tile size {base_size} is not larger than the overlap."
        )
    if base_size < min_tile_size:
        raise ValueError(
            f"Cannot split length {length} into {tile_count} tiles with "
            f"minimum tile size {min_tile_size}: smallest tile would have size {base_size}."
        )

    intervals = []
    start = 0
    for tile_index in range(tile_count):
        tile_size = base_size + int(tile_index < larger_tiles)
        end = start + tile_size
        intervals.append((start, end))
        start = end - overlap
    if intervals[-1][1] != length:
        raise AssertionError(f"Invalid tile coverage for length {length}: {intervals}")
    return tuple(intervals)


def _minimum_raw_latent_tile_shape(
    stage_kernels: tuple[tuple[int, int, int], ...],
    stage5_kernel: tuple[int, int, int],
    upsample_strides: tuple[tuple[int, int, int], ...],
) -> tuple[int, int, int]:
    """Return the smallest raw tile accepted by every decoder NA stage."""
    if len(stage_kernels) != len(upsample_strides):
        raise ValueError(
            "decoder stage kernels and upsample strides must have the same length, "
            f"got {len(stage_kernels)} and {len(upsample_strides)}."
        )

    scale = [1, 1, 1]
    minimum = [1, 1, 1]
    for kernel, stride in zip(stage_kernels, upsample_strides, strict=True):
        for axis in range(3):
            minimum[axis] = max(minimum[axis], (kernel[axis] + scale[axis] - 1) // scale[axis])
            scale[axis] *= stride[axis]
    for axis in range(3):
        minimum[axis] = max(minimum[axis], (stage5_kernel[axis] + scale[axis] - 1) // scale[axis])
    return minimum[0], minimum[1], minimum[2]


def _axis_min_tile_size(
    model_minimum: int,
    tile_count: int,
    overlap: int,
) -> int:
    """Keep NA legal and prevent three adjacent blend ramps from overlapping."""
    if tile_count <= 1:
        return model_minimum
    blend_minimum = overlap + 1 if tile_count == 2 else 2 * overlap
    return max(model_minimum, blend_minimum)


def _shape_aware_spatial_grid(
    *,
    height: int,
    width: int,
    tile_count: int,
    overlap: int,
    min_tile_height: int = 1,
    min_tile_width: int = 1,
) -> tuple[int, int]:
    """Choose a legal HxW factorization for the actual latent shape.

    Fixed overlap makes aggregate processed area a proxy for both redundant
    decoder work and seam exposure. All candidates use ``tile_count`` tasks;
    ties prefer the smaller slowest task, then more square tiles. The last
    tie-break preserves the previous wider-grid orientation for square inputs.
    """
    if height < 1 or width < 1:
        raise ValueError(f"height and width must be >= 1, got {height}x{width}")
    if tile_count < 1:
        raise ValueError(f"tile_count must be >= 1, got {tile_count}")

    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if min_tile_height < 1 or min_tile_width < 1:
        raise ValueError(f"minimum tile dimensions must be >= 1, got {min_tile_height}x{min_tile_width}")

    candidates = []
    for height_count in range(1, tile_count + 1):
        if tile_count % height_count:
            continue
        width_count = tile_count // height_count
        height_overlap = overlap if height_count > 1 else 0
        width_overlap = overlap if width_count > 1 else 0
        try:
            height_tiles = _split_axis_by_count(
                height,
                height_count,
                height_overlap,
                min_tile_size=_axis_min_tile_size(min_tile_height, height_count, height_overlap),
            )
            width_tiles = _split_axis_by_count(
                width,
                width_count,
                width_overlap,
                min_tile_size=_axis_min_tile_size(min_tile_width, width_count, width_overlap),
            )
        except ValueError:
            continue

        height_sizes = tuple(end - start for start, end in height_tiles)
        width_sizes = tuple(end - start for start, end in width_tiles)
        aggregate_area = sum(height_sizes) * sum(width_sizes)
        redundant_area = aggregate_area - height * width
        max_task_area = max(height_sizes) * max(width_sizes)
        square_tile_mismatch = abs(height * width_count - width * height_count)
        prefer_wider_grid = height_count > width_count
        score = (
            redundant_area,
            max_task_area,
            square_tile_mismatch,
            prefer_wider_grid,
            height_count,
        )
        candidates.append((score, height_count, width_count))

    if not candidates:
        raise ValueError(
            f"Cannot split latent shape {height}x{width} into {tile_count} legal spatial tiles "
            f"with overlap {overlap} and minimum tile size {min_tile_height}x{min_tile_width}."
        )

    _, height_count, width_count = min(candidates)
    return height_count, width_count


@dataclass(frozen=True)
class LTX2VideoDiffusionTilePlan:
    """Latent-space geometry shared by all ranks for one decode."""

    height_tiles: tuple[tuple[int, int], ...]
    width_tiles: tuple[tuple[int, int], ...]
    latent_frames: int
    latent_height: int
    latent_width: int
    scale_t: int
    scale_h: int
    scale_w: int

    @property
    def output_shape(self) -> tuple[int, int, int]:
        return (
            (self.latent_frames - 1) * self.scale_t + 1,
            self.latent_height * self.scale_h,
            self.latent_width * self.scale_w,
        )


@dataclass
class LTX2VideoDiffusionTileTask(TileTask):
    """One raw latent tile and the synchronized decoder RNG."""

    noise_generator: torch.Generator | list[torch.Generator] | None = None


class DistributedLTX2VideoDiffusionDecoderModel(
    LTX2VideoDiffusionDecoderModel,
    DistributedVaeMixin,
):
    """LTX-2.5 DiffVAE with full-decoder spatial tile parallelism.

    The latent is divided into one overlapping spatial tile per participating
    rank. Every tile then runs all five decoder stages independently, and rank
    0 reconstructs the video with complementary trapezoidal blend masks.
    """

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any):
        model = super().from_pretrained(*args, **kwargs)
        model.init_distributed()
        return model

    def set_parallel_size(self, parallel_size: int, mode: str = "tile") -> None:
        if mode != "tile":
            raise ValueError(f"LTX-2.5 DiffVAE only supports vae_parallel_mode='tile', got {mode!r}.")
        super().set_parallel_size(parallel_size, mode=mode)

    def _build_tiled_decode_plan(
        self,
        z: torch.Tensor,
    ) -> LTX2VideoDiffusionTilePlan:
        parallel_size = min(
            int(self.distributed_executor.parallel_size),
            int(self.distributed_executor.world_size),
        )
        minimum_tile_shape = _minimum_raw_latent_tile_shape(
            tuple(tuple(kernel) for kernel in self.config.decoder_stage_kernels),
            tuple(self.config.decoder_stage5_kernel),
            tuple(tuple(stride) for stride in self.config.decoder_upsample_strides),
        )
        height_count, width_count = _shape_aware_spatial_grid(
            height=z.shape[3],
            width=z.shape[4],
            tile_count=parallel_size,
            overlap=LTX2_VAE_SPATIAL_OVERLAP,
            min_tile_height=minimum_tile_shape[1],
            min_tile_width=minimum_tile_shape[2],
        )
        height_tiles = _split_axis_by_count(
            z.shape[3],
            height_count,
            LTX2_VAE_SPATIAL_OVERLAP if height_count > 1 else 0,
            min_tile_size=_axis_min_tile_size(
                minimum_tile_shape[1],
                height_count,
                LTX2_VAE_SPATIAL_OVERLAP if height_count > 1 else 0,
            ),
        )
        width_tiles = _split_axis_by_count(
            z.shape[4],
            width_count,
            LTX2_VAE_SPATIAL_OVERLAP if width_count > 1 else 0,
            min_tile_size=_axis_min_tile_size(
                minimum_tile_shape[2],
                width_count,
                LTX2_VAE_SPATIAL_OVERLAP if width_count > 1 else 0,
            ),
        )
        plan = LTX2VideoDiffusionTilePlan(
            height_tiles=height_tiles,
            width_tiles=width_tiles,
            latent_frames=z.shape[2],
            latent_height=z.shape[3],
            latent_width=z.shape[4],
            scale_t=self.temporal_compression_ratio,
            scale_h=self.spatial_compression_ratio,
            scale_w=self.spatial_compression_ratio,
        )
        logger.debug(
            "LTX-2.5 distributed DiffVAE shape-aware latent tile plan: grid=%s overlap=%d",
            (height_count, width_count),
            LTX2_VAE_SPATIAL_OVERLAP,
        )
        return plan

    def _default_generator(self, device: torch.device) -> torch.Generator:
        if device.type == "cpu":
            return torch.default_generator
        device_module = getattr(torch, device.type, None)
        default_generators = getattr(device_module, "default_generators", None)
        if default_generators is None:
            raise ValueError(
                f"Distributed LTX-2.5 diffusion decode on {device.type!r} requires an explicit torch.Generator."
            )
        device_index = device.index
        if device_index is None:
            current_device = getattr(device_module, "current_device", None)
            device_index = current_device() if current_device is not None else 0
        return default_generators[device_index]

    def _sync_generators(
        self,
        generator: torch.Generator | list[torch.Generator] | None,
        device: torch.device,
    ) -> torch.Generator | list[torch.Generator]:
        """Make every rank begin its one tile decode from rank 0's RNG state."""
        generators = self._default_generator(device) if generator is None else generator
        generator_list = generators if isinstance(generators, list) else [generators]
        for item in generator_list:
            state_on_device = item.get_state().to(device=device)
            dist.broadcast(
                state_on_device,
                src=0,
                group=self.distributed_executor.group,
            )
            item.set_state(state_on_device.cpu())
        return generators

    def _distributed_tile_split(
        self,
        z: torch.Tensor,
        generator: torch.Generator | list[torch.Generator] | None,
    ) -> tuple[list[LTX2VideoDiffusionTileTask], GridSpec]:
        plan = self._build_tiled_decode_plan(z)
        generators = self._sync_generators(generator, z.device)

        tasks = []
        coords = product(
            range(len(plan.height_tiles)),
            range(len(plan.width_tiles)),
        )
        for tile_id, coord in enumerate(coords):
            h0, h1 = plan.height_tiles[coord[0]]
            w0, w1 = plan.width_tiles[coord[1]]
            latent_tile = z[:, :, :, h0:h1, w0:w1]
            tasks.append(
                LTX2VideoDiffusionTileTask(
                    tile_id=tile_id,
                    grid_coord=coord,
                    tensor=latent_tile,
                    workload=latent_tile.numel(),
                    noise_generator=generators,
                )
            )

        return tasks, GridSpec(
            split_dims=(3, 4),
            grid_shape=(len(plan.height_tiles), len(plan.width_tiles)),
            tile_spec={"plan": plan},
            output_dtype=z.dtype,
        )

    def _distributed_tile_exec(
        self,
        task: LTX2VideoDiffusionTileTask,
        num_inference_steps: int,
    ) -> torch.Tensor:
        decoded = self.decoder(
            task.tensor,
            generator=task.noise_generator,
            num_inference_steps=num_inference_steps,
        )
        target_frames = (task.tensor.shape[2] - 1) * self.temporal_compression_ratio + 1
        target_height = task.tensor.shape[3] * self.spatial_compression_ratio
        target_width = task.tensor.shape[4] * self.spatial_compression_ratio
        return decoded[:, :, :target_frames, :target_height, :target_width]

    @staticmethod
    def _spatial_blend_mask(
        intervals: tuple[tuple[int, int], ...],
        index: int,
        scale: int,
        device: torch.device,
    ) -> torch.Tensor:
        start, end = intervals[index]
        length = (end - start) * scale
        left_ramp = (intervals[index - 1][1] - start) * scale if index > 0 else 0
        right_ramp = (end - intervals[index + 1][0]) * scale if index + 1 < len(intervals) else 0
        mask = torch.ones(length, dtype=torch.float32, device=device)
        if left_ramp:
            mask[:left_ramp] = torch.linspace(
                0.0,
                1.0,
                left_ramp + 2,
                dtype=torch.float32,
                device=device,
            )[1:-1]
        if right_ramp:
            mask[-right_ramp:] = torch.linspace(
                1.0,
                0.0,
                right_ramp + 2,
                dtype=torch.float32,
                device=device,
            )[1:-1]
        return mask

    def _merge_tiled_decode(
        self,
        tiles: dict[tuple[int, int], torch.Tensor],
        plan: LTX2VideoDiffusionTilePlan,
    ) -> torch.Tensor:
        if not tiles:
            raise ValueError("Cannot merge an empty LTX-2.5 DiffVAE tile set.")
        first_tile = next(iter(tiles.values()))
        frames, height, width = plan.output_shape
        output = torch.zeros(
            first_tile.shape[0],
            first_tile.shape[1],
            frames,
            height,
            width,
            device=first_tile.device,
            dtype=first_tile.dtype,
        )
        height_masks = tuple(
            self._spatial_blend_mask(
                plan.height_tiles,
                index,
                plan.scale_h,
                first_tile.device,
            )
            for index in range(len(plan.height_tiles))
        )
        width_masks = tuple(
            self._spatial_blend_mask(
                plan.width_tiles,
                index,
                plan.scale_w,
                first_tile.device,
            )
            for index in range(len(plan.width_tiles))
        )

        for h_idx, w_idx in product(
            range(len(plan.height_tiles)),
            range(len(plan.width_tiles)),
        ):
            h0, h1 = plan.height_tiles[h_idx]
            w0, w1 = plan.width_tiles[w_idx]
            tile = tiles[(h_idx, w_idx)]
            expected_shape = (
                first_tile.shape[0],
                first_tile.shape[1],
                frames,
                (h1 - h0) * plan.scale_h,
                (w1 - w0) * plan.scale_w,
            )
            if tuple(tile.shape) != expected_shape:
                raise ValueError(
                    f"Decoded tile {(h_idx, w_idx)} has shape {tuple(tile.shape)}, expected {expected_shape}."
                )
            mask = height_masks[h_idx][None, None, None, :, None]
            mask = mask * width_masks[w_idx][None, None, None, None, :]
            output[
                :,
                :,
                :,
                h0 * plan.scale_h : h1 * plan.scale_h,
                w0 * plan.scale_w : w1 * plan.scale_w,
            ] += tile * mask
        return output

    def _distributed_tile_merge(
        self,
        tiles: dict[tuple[int, int], torch.Tensor],
        grid_spec: GridSpec,
    ) -> torch.Tensor:
        plan = grid_spec.tile_spec["plan"]
        if not isinstance(plan, LTX2VideoDiffusionTilePlan):
            raise TypeError(f"Expected an LTX2VideoDiffusionTilePlan, got {type(plan)!r}.")
        return self._merge_tiled_decode(tiles, plan)

    def tiled_decode(
        self,
        z: torch.Tensor,
        generator: torch.Generator | list[torch.Generator] | None = None,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        if not self.is_distributed_enabled():
            return super().tiled_decode(
                z,
                generator=generator,
                num_inference_steps=num_inference_steps,
            )

        logger.debug("LTX-2.5 diffusion decoder running full decoder on distributed latent tiles")
        num_inference_steps = num_inference_steps or self.decoder.default_num_inference_steps
        result = self.distributed_executor.execute(
            z,
            DistributedOperator(
                split=lambda tensor: self._distributed_tile_split(tensor, generator),
                exec=lambda task: self._distributed_tile_exec(
                    task,
                    num_inference_steps,
                ),
                merge=self._distributed_tile_merge,
            ),
            broadcast_result=False,
        )
        if result.numel() == 0:
            # The base decode method crops five dimensions before the LTX
            # runtime discards non-output-rank results.
            return torch.empty(
                (0, self.decoder.out_channels, 0, 0, 0),
                device=z.device,
                dtype=z.dtype,
            )
        return result
