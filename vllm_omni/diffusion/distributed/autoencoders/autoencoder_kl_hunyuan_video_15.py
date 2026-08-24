# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from diffusers import AutoencoderKLHunyuanVideo15
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedOperator,
    DistributedVaeMixin,
    GridSpec,
    TileTask,
)

logger = init_logger(__name__)


class DistributedAutoencoderKLHunyuanVideo15(AutoencoderKLHunyuanVideo15, DistributedVaeMixin):
    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any):
        model = super().from_pretrained(*args, **kwargs)
        model.init_distributed()
        return model

    def tile_split(self, z: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        _, _, _, height, width = z.shape
        overlap_height = int(self.tile_latent_min_height * (1 - self.tile_overlap_factor))
        overlap_width = int(self.tile_latent_min_width * (1 - self.tile_overlap_factor))
        blend_height = int(self.tile_sample_min_height * self.tile_overlap_factor)
        blend_width = int(self.tile_sample_min_width * self.tile_overlap_factor)
        row_limit_height = self.tile_sample_min_height - blend_height
        row_limit_width = self.tile_sample_min_width - blend_width

        tiletask_list = []
        for i in range(0, height, overlap_height):
            for j in range(0, width, overlap_width):
                tile = z[:, :, :, i : i + self.tile_latent_min_height, j : j + self.tile_latent_min_width]
                tiletask_list.append(
                    TileTask(
                        len(tiletask_list),
                        (i // overlap_height, j // overlap_width),
                        tile,
                        workload=tile.shape[3] * tile.shape[4],
                    )
                )

        grid_spec = GridSpec(
            split_dims=(3, 4),
            grid_shape=(tiletask_list[-1].grid_coord[0] + 1, tiletask_list[-1].grid_coord[1] + 1),
            tile_spec={
                "blend_height": blend_height,
                "blend_width": blend_width,
                "row_limit_height": row_limit_height,
                "row_limit_width": row_limit_width,
            },
            output_dtype=self.dtype,
        )
        return tiletask_list, grid_spec

    def tile_exec(self, task: TileTask) -> torch.Tensor:
        return self.decoder(task.tensor)

    def tile_merge(self, coord_tensor_map: dict[tuple[int, ...], torch.Tensor], grid_spec: GridSpec) -> torch.Tensor:
        grid_h, grid_w = grid_spec.grid_shape
        result_rows = []
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
                        : grid_spec.tile_spec["row_limit_height"],
                        : grid_spec.tile_spec["row_limit_width"],
                    ]
                )
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def encode_tile_split(self, x: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        _, _, _, height, width = x.shape
        overlap_height = int(self.tile_sample_min_height * (1 - self.tile_overlap_factor))
        overlap_width = int(self.tile_sample_min_width * (1 - self.tile_overlap_factor))
        blend_height = int(self.tile_latent_min_height * self.tile_overlap_factor)
        blend_width = int(self.tile_latent_min_width * self.tile_overlap_factor)
        row_limit_height = self.tile_latent_min_height - blend_height
        row_limit_width = self.tile_latent_min_width - blend_width

        tiletask_list = []
        for i in range(0, height, overlap_height):
            for j in range(0, width, overlap_width):
                tile = x[:, :, :, i : i + self.tile_sample_min_height, j : j + self.tile_sample_min_width]
                tiletask_list.append(
                    TileTask(
                        len(tiletask_list),
                        (i // overlap_height, j // overlap_width),
                        tile,
                        workload=tile.shape[3] * tile.shape[4],
                    )
                )

        grid_spec = GridSpec(
            split_dims=(3, 4),
            grid_shape=(tiletask_list[-1].grid_coord[0] + 1, tiletask_list[-1].grid_coord[1] + 1),
            tile_spec={
                "blend_height": blend_height,
                "blend_width": blend_width,
                "row_limit_height": row_limit_height,
                "row_limit_width": row_limit_width,
            },
            output_dtype=self.dtype,
        )
        return tiletask_list, grid_spec

    def encode_tile_exec(self, task: TileTask) -> torch.Tensor:
        return self.encoder(task.tensor)

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_distributed_enabled():
            return super().tiled_encode(x)

        logger.debug("Encode running with distributed executor")
        return self.distributed_executor.execute(
            x,
            DistributedOperator(
                split=self.encode_tile_split,
                exec=self.encode_tile_exec,
                merge=self.tile_merge,
            ),
            broadcast_result=True,
        )

    def tiled_decode(self, z: torch.Tensor) -> torch.Tensor:
        if not self.is_distributed_enabled():
            return super().tiled_decode(z)

        logger.debug("Decode running with distributed executor")
        return self.distributed_executor.execute(
            z,
            DistributedOperator(split=self.tile_split, exec=self.tile_exec, merge=self.tile_merge),
            broadcast_result=True,
        )
