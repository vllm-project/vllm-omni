# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from typing import Any

import torch
from diffusers.models.autoencoders.vae import DecoderOutput
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedOperator,
    DistributedVaeMixin,
    GridSpec,
    TileTask,
)
from vllm_omni.diffusion.models.qwen_image_21.autoencoder_kl_qwenimage21 import (
    AutoencoderKLQwenImage21,
    _cudnn_deterministic,
    _unpatchify,
)

logger = init_logger(__name__)


class DistributedAutoencoderKLQwenImage21(AutoencoderKLQwenImage21, DistributedVaeMixin):
    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any):
        model = super().from_pretrained(*args, **kwargs)
        model.init_distributed()
        return model

    def tile_split(self, z: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        # mostly copy from AutoencoderKLQwenImage21.tiled_decode
        _, _, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        if self.config.patch_size is not None:
            sample_height = sample_height // self.config.patch_size
            sample_width = sample_width // self.config.patch_size

        # Split z into overlapping tiles and decode them separately. The starts
        # are spread evenly so every tile keeps the full decode window (see
        # AutoencoderKLQwenImage21._tile_starts, vllm-omni#8850).
        starts_height = self._tile_starts(height, tile_latent_min_height, tile_latent_stride_height)
        starts_width = self._tile_starts(width, tile_latent_min_width, tile_latent_stride_width)
        # Decoded cells contributed by one latent cell, before unpatchify.
        cells_per_latent = self.spatial_compression_ratio // (self.config.patch_size or 1)
        overlap_height = [
            (starts_height[n] + tile_latent_min_height - starts_height[n + 1]) * cells_per_latent
            for n in range(len(starts_height) - 1)
        ]
        overlap_width = [
            (starts_width[n] + tile_latent_min_width - starts_width[n + 1]) * cells_per_latent
            for n in range(len(starts_width) - 1)
        ]

        tiletask_list: list[TileTask] = []
        for i_index, i in enumerate(starts_height):
            for j_index, j in enumerate(starts_width):
                time_list = []
                for k in range(num_frames):
                    tile = z[:, :, k : k + 1, i : i + tile_latent_min_height, j : j + tile_latent_min_width]
                    time_list.append(tile)
                tiletask_list.append(
                    TileTask(
                        len(tiletask_list),
                        (i_index, j_index),
                        time_list,
                        workload=time_list[0].shape[3] * time_list[0].shape[4],
                    )
                )
        tile_spec = {
            "sample_height": sample_height,
            "sample_width": sample_width,
            "starts_height": starts_height,
            "starts_width": starts_width,
            "cells_per_latent": cells_per_latent,
            "overlap_height": overlap_height,
            "overlap_width": overlap_width,
        }
        grid_spec = GridSpec(
            split_dims=(3, 4),
            grid_shape=(len(starts_height), len(starts_width)),
            tile_spec=tile_spec,
            output_dtype=self.dtype,
        )
        return tiletask_list, grid_spec

    def tile_exec(self, task: TileTask) -> torch.Tensor:
        """Decode a single latent tile into RGB space."""
        self.clear_cache()
        time = []
        with _cudnn_deterministic():
            for k in range(len(task.tensor)):
                tile = self.post_quant_conv(task.tensor[k])
                decoded = self.decoder(tile, first_chunk=(k == 0))
                time.append(decoded)
        result = torch.cat(time, dim=2)
        return result

    def tile_merge(self, coord_tensor_map: dict[tuple[int, ...], torch.Tensor], grid_spec: GridSpec) -> torch.Tensor:
        """Merge decoded tiles into a full image."""
        grid_h, grid_w = grid_spec.grid_shape
        self.clear_cache()

        starts_height = grid_spec.tile_spec["starts_height"]
        starts_width = grid_spec.tile_spec["starts_width"]
        cells_per_latent = grid_spec.tile_spec["cells_per_latent"]
        overlap_height = grid_spec.tile_spec["overlap_height"]
        overlap_width = grid_spec.tile_spec["overlap_width"]

        result_rows = []
        for i in range(grid_h):
            result_row = []
            for j in range(grid_w):
                tile = coord_tensor_map[(i, j)]
                if i > 0:
                    tile = self.blend_v(coord_tensor_map[(i - 1, j)], tile, overlap_height[i - 1])
                if j > 0:
                    tile = self.blend_h(coord_tensor_map[(i, j - 1)], tile, overlap_width[j - 1])
                # A tile contributes up to the start of the next one; the trailing
                # tile contributes the rest and is trimmed by the final crop.
                keep_h = (
                    (starts_height[i + 1] - starts_height[i]) * cells_per_latent if i + 1 < grid_h else tile.shape[-2]
                )
                keep_w = (
                    (starts_width[j + 1] - starts_width[j]) * cells_per_latent if j + 1 < grid_w else tile.shape[-1]
                )
                result_row.append(tile[:, :, :, :keep_h, :keep_w])
            result_rows.append(torch.cat(result_row, dim=-1))
        dec = torch.cat(result_rows, dim=3)[
            :, :, :, : grid_spec.tile_spec["sample_height"], : grid_spec.tile_spec["sample_width"]
        ]

        if self.config.patch_size is not None:
            dec = _unpatchify(dec, patch_size=self.config.patch_size)

        dec = torch.clamp(dec, min=-1.0, max=1.0)
        return dec

    def tiled_decode(self, z: torch.Tensor, return_dict: bool = True):
        if not self.is_distributed_enabled():
            return super().tiled_decode(z, return_dict=return_dict)

        logger.debug("Decode running with distributed executor")
        result = self.distributed_executor.execute(
            z,
            DistributedOperator(split=self.tile_split, exec=self.tile_exec, merge=self.tile_merge),
            broadcast_result=True,
        )
        if not return_dict:
            return (result,)

        return DecoderOutput(sample=result)
