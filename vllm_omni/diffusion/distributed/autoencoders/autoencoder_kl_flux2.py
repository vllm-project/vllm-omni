# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl import DistributedAutoencoderKL_base
from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedOperator,
    GridSpec,
    TileTask,
)

logger = init_logger(__name__)


class DistributedAutoencoderKLFlux2(DistributedAutoencoderKL_base, AutoencoderKLFlux2):
    def encode_tile_split(self, x: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        _, _, height, width = x.shape
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        tiletask_list = []
        for i in range(0, height, overlap_size):
            for j in range(0, width, overlap_size):
                tile = x[:, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                tiletask_list.append(
                    TileTask(
                        len(tiletask_list),
                        (i // overlap_size, j // overlap_size),
                        tile,
                        workload=tile.shape[2] * tile.shape[3],
                    )
                )

        grid_spec = GridSpec(
            split_dims=(2, 3),
            grid_shape=(tiletask_list[-1].grid_coord[0] + 1, tiletask_list[-1].grid_coord[1] + 1),
            tile_spec={
                "blend_extent": blend_extent,
                "row_limit": row_limit,
            },
            output_dtype=self.dtype,
        )
        return tiletask_list, grid_spec

    def encode_tile_exec(self, task: TileTask) -> torch.Tensor:
        tile = self.encoder(task.tensor)
        if self.quant_conv is not None:
            tile = self.quant_conv(tile)
        return tile

    def encode_tile_merge(
        self, coord_tensor_map: dict[tuple[int, ...], torch.Tensor], grid_spec: GridSpec
    ) -> torch.Tensor:
        grid_h, grid_w = grid_spec.grid_shape
        result_rows = []
        for i in range(grid_h):
            result_row = []
            for j in range(grid_w):
                tile = coord_tensor_map[(i, j)]
                if i > 0:
                    tile = self.blend_v(coord_tensor_map[(i - 1, j)], tile, grid_spec.tile_spec["blend_extent"])
                if j > 0:
                    tile = self.blend_h(coord_tensor_map[(i, j - 1)], tile, grid_spec.tile_spec["blend_extent"])
                result_row.append(tile[:, :, : grid_spec.tile_spec["row_limit"], : grid_spec.tile_spec["row_limit"]])
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        if self.use_tiling and (width > self.tile_sample_min_size or height > self.tile_sample_min_size):
            if self.is_distributed_enabled():
                logger.debug("Encode running with distributed executor")
                return self.distributed_executor.execute(
                    x,
                    DistributedOperator(
                        split=self.encode_tile_split,
                        exec=self.encode_tile_exec,
                        merge=self.encode_tile_merge,
                    ),
                    broadcast_result=True,
                )
            return self._tiled_encode(x)

        enc = self.encoder(x)
        if self.quant_conv is not None:
            enc = self.quant_conv(enc)
        return enc

    def decode(self, z: torch.Tensor, return_dict: bool = True, *args: Any, **kwargs: Any):
        if not self.is_distributed_enabled():
            return super().decode(z, return_dict=return_dict, *args, **kwargs)

        split, exec_fn, merge = self._strategy_select(z)

        if split is not None:
            strategy = "tile" if split == self.tile_split else "patch"
            logger.info(f"Decode run with distributed executor, split strategy is {strategy}")
            result = self.distributed_executor.execute(
                z,
                DistributedOperator(split=split, exec=exec_fn, merge=merge),
                broadcast_result=True,
            )
            if not return_dict:
                return (result,)

            from diffusers.models.autoencoders.vae import DecoderOutput

            return DecoderOutput(sample=result)

        return super().decode(z, return_dict=return_dict, *args, **kwargs)
