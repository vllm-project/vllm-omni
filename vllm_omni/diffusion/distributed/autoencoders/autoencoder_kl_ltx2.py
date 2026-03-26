# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from typing import Any

import torch
from diffusers import AutoencoderKLLTX2Video
from diffusers.models.autoencoders.vae import DecoderOutput
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedOperator,
    DistributedVaeMixin,
    GridSpec,
    TileTask,
)

logger = init_logger(__name__)


class DistributedAutoencoderKLLTX2Video(AutoencoderKLLTX2Video, DistributedVaeMixin):
    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any):
        model = super().from_pretrained(*args, **kwargs)
        model.init_distributed()
        return model

    def tile_split(self, z: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        _, _, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio
        tile_sample_stride_height = self.tile_sample_stride_height
        tile_sample_stride_width = self.tile_sample_stride_width
        # `super().decode(...)` already returns fully decoded pixel tiles, so
        # no extra patch-size downscaling/unpatchifying should be applied here.
        blend_height = self.tile_sample_min_height - tile_sample_stride_height
        blend_width = self.tile_sample_min_width - tile_sample_stride_width

        tiletask_list = []
        for i in range(0, height, tile_latent_stride_height):
            for j in range(0, width, tile_latent_stride_width):
                tile = z[:, :, :num_frames, i : i + tile_latent_min_height, j : j + tile_latent_min_width]
                tiletask_list.append(
                    TileTask(
                        len(tiletask_list),
                        (i // tile_latent_stride_height, j // tile_latent_stride_width),
                        tile,
                        workload=tile.shape[2] * tile.shape[3] * tile.shape[4],
                    )
                )

        tile_spec = {
            "sample_height": sample_height,
            "sample_width": sample_width,
            "blend_height": blend_height,
            "blend_width": blend_width,
            "tile_sample_stride_height": tile_sample_stride_height,
            "tile_sample_stride_width": tile_sample_stride_width,
        }
        grid_spec = GridSpec(
            split_dims=(3, 4),
            grid_shape=(tiletask_list[-1].grid_coord[0] + 1, tiletask_list[-1].grid_coord[1] + 1),
            tile_spec=tile_spec,
            output_dtype=self.dtype,
        )
        return tiletask_list, grid_spec

    def tile_exec(
        self,
        task: TileTask,
        timestep: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Decode a single latent tile into video space."""
        tile = task.tensor
        if hasattr(self, "clear_cache"):
            self.clear_cache()
        return super().decode(tile, timestep, return_dict=False, *args, **kwargs)[0]

    def tile_merge(self, coord_tensor_map: dict[tuple[int, ...], torch.Tensor], grid_spec: GridSpec) -> torch.Tensor:
        """Merge decoded tiles into a full video."""
        grid_h, grid_w = grid_spec.grid_shape
        result_rows = []

        if hasattr(self, "clear_cache"):
            self.clear_cache()

        for i in range(grid_h):
            result_row = []
            for j in range(grid_w):
                tile = coord_tensor_map[(i, j)]
                if i > 0:
                    tile = self.blend_v(coord_tensor_map[(i - 1, j)], tile, grid_spec.tile_spec["blend_height"])
                if j > 0:
                    tile = self.blend_h(coord_tensor_map[(i, j - 1)], tile, grid_spec.tile_spec["blend_width"])
                result_row.append(
                    tile[
                        :,
                        :,
                        :,
                        : grid_spec.tile_spec["tile_sample_stride_height"],
                        : grid_spec.tile_spec["tile_sample_stride_width"],
                    ]
                )
            result_rows.append(torch.cat(result_row, dim=-1))

        dec = torch.cat(result_rows, dim=3)[
            :, :, :, : grid_spec.tile_spec["sample_height"], : grid_spec.tile_spec["sample_width"]
        ]
        dec = torch.clamp(dec, min=-1.0, max=1.0)
        return dec

    def patch_split(self, z: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        _, _, _, latent_h, latent_w = z.shape

        overlap_sample_h = max(0, self.tile_sample_min_height - self.tile_sample_stride_height)
        overlap_sample_w = max(0, self.tile_sample_min_width - self.tile_sample_stride_width)
        overlap_latent_h = overlap_sample_h // self.spatial_compression_ratio
        overlap_latent_w = overlap_sample_w // self.spatial_compression_ratio
        halo_base_h = max(0, overlap_latent_h // 2)
        halo_base_w = max(0, overlap_latent_w // 2)

        max_parallel_size = self.distributed_decoder.parallel_size
        root = int(math.sqrt(max_parallel_size))
        for rows in range(root, 0, -1):
            if max_parallel_size % rows == 0:
                grid_rows, grid_cols = rows, max_parallel_size // rows
                break

        tiletask_list = []
        halo_size = {}
        for i in range(grid_rows):
            for j in range(grid_cols):
                h0 = (i * latent_h) // grid_rows
                h1 = ((i + 1) * latent_h) // grid_rows
                w0 = (j * latent_w) // grid_cols
                w1 = ((j + 1) * latent_w) // grid_cols

                core_h = max(0, h1 - h0)
                core_w = max(0, w1 - w0)
                halo_h = max(halo_base_h, core_h // 2)
                halo_w = max(halo_base_w, core_w // 2)

                ph0 = max(0, h0 - halo_h)
                ph1 = min(latent_h, h1 + halo_h)
                pw0 = max(0, w0 - halo_w)
                pw1 = min(latent_w, w1 + halo_w)

                tile = z[:, :, :, ph0:ph1, pw0:pw1]
                tiletask_list.append(
                    TileTask(
                        len(tiletask_list),
                        (i, j),
                        tile,
                        workload=tile.shape[2] * tile.shape[3] * tile.shape[4],
                    )
                )
                halo_size[(i, j)] = {
                    "up": h0 - ph0,
                    "down": ph1 - h1,
                    "left": w0 - pw0,
                    "right": pw1 - w1,
                }

        grid_spec = GridSpec(
            split_dims=(3, 4),
            grid_shape=(tiletask_list[-1].grid_coord[0] + 1, tiletask_list[-1].grid_coord[1] + 1),
            tile_spec={
                "halo_size": halo_size,
                "scale": self.spatial_compression_ratio,
            },
            output_dtype=self.dtype,
        )
        return tiletask_list, grid_spec

    def patch_exec(
        self,
        task: TileTask,
        timestep: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.tile_exec(task, timestep=timestep, *args, **kwargs)

    def patch_merge(self, coord_tensor_map: dict[tuple[int, ...], torch.Tensor], grid_spec: GridSpec) -> torch.Tensor:
        grid_h, grid_w = grid_spec.grid_shape
        result_rows = []
        scale = grid_spec.tile_spec["scale"]

        if hasattr(self, "clear_cache"):
            self.clear_cache()

        for i in range(grid_h):
            result_row = []
            for j in range(grid_w):
                halo = grid_spec.tile_spec["halo_size"][(i, j)]
                tile = coord_tensor_map[(i, j)]

                halo_up = halo["up"] * scale
                halo_down = halo["down"] * scale
                halo_left = halo["left"] * scale
                halo_right = halo["right"] * scale

                core_tile = tile[
                    :,
                    :,
                    :,
                    halo_up : (None if halo_down == 0 else -halo_down),
                    halo_left : (None if halo_right == 0 else -halo_right),
                ]
                result_row.append(core_tile)
            result_rows.append(torch.cat(result_row, dim=4))

        dec = torch.cat(result_rows, dim=3)
        dec = torch.clamp(dec, min=-1.0, max=1.0)
        return dec

    def _strategy_select(self, z: torch.Tensor):
        tile_latent_min_height = getattr(self, "tile_sample_min_height", None)
        tile_latent_min_width = getattr(self, "tile_sample_min_width", None)
        if tile_latent_min_height is None or tile_latent_min_width is None:
            return None, None, None

        tile_latent_min_height = tile_latent_min_height // self.spatial_compression_ratio
        tile_latent_min_width = tile_latent_min_width // self.spatial_compression_ratio
        if z.shape[-2] > tile_latent_min_height or z.shape[-1] > tile_latent_min_width:
            return self.tile_split, self.tile_exec, self.tile_merge

        return self.patch_split, self.patch_exec, self.patch_merge

    def decode(
        self,
        z: torch.Tensor,
        timestep: torch.Tensor | None = None,
        return_dict: bool = True,
        *args: Any,
        **kwargs: Any,
    ):
        if not self.is_distributed_enabled():
            return super().decode(z, timestep, return_dict=return_dict, *args, **kwargs)

        split, exec, merge = self._strategy_select(z)
        if split is None:
            return super().decode(z, timestep, return_dict=return_dict, *args, **kwargs)

        strategy = "tile" if split == self.tile_split else "patch"
        logger.info(f"Decode run with distributed executor, split strategy is {strategy}")
        result = self.distributed_decoder.execute(
            z,
            DistributedOperator(
                split=split,
                exec=lambda task: exec(task, timestep=timestep, *args, **kwargs),
                merge=merge,
            ),
            broadcast_result=True,
        )
        if not return_dict:
            return (result,)

        return DecoderOutput(sample=result)
