# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from diffusers.models.autoencoders.autoencoder_oobleck import AutoencoderOobleck
from diffusers.models.autoencoders.vae import DecoderOutput
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedOperator,
    DistributedVaeMixin,
    GridSpec,
    TileTask,
)

logger = init_logger(__name__)


class DistributedAutoencoderKLStableAudio(AutoencoderOobleck, DistributedVaeMixin):
    use_tiling: bool = False

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any):
        model = super().from_pretrained(*args, **kwargs)
        model.init_distributed()
        return model

    def tile_split(self, z: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        tile_latent_min_size = getattr(self, "tile_latent_min_size", None)
        tile_overlap_factor = getattr(self, "tile_overlap_factor", 0.5)

        if tile_latent_min_size is None:
            tile_latent_min_size = z.shape[2] // 4

        overlap_size = int(tile_latent_min_size * (1 - tile_overlap_factor))
        if overlap_size == 0:
            overlap_size = tile_latent_min_size // 2

        _, _, time_steps = z.shape

        tiletask_list = []
        for i in range(0, time_steps, overlap_size):
            tile = z[:, :, i : i + tile_latent_min_size]
            if tile.shape[2] == 0:
                continue
            tiletask_list.append(
                TileTask(
                    len(tiletask_list),
                    (i // overlap_size,),
                    tile,
                    workload=tile.shape[2],
                )
            )

        blend_extent = overlap_size // 2

        tile_spec = {
            "blend_extent": blend_extent,
            "tile_latent_min_size": tile_latent_min_size,
            "overlap_size": overlap_size,
        }

        grid_spec = GridSpec(
            split_dims=(2,),
            grid_shape=(len(tiletask_list),),
            tile_spec=tile_spec,
            output_dtype=self.dtype,
        )

        return tiletask_list, grid_spec

    def tile_exec(self, task: TileTask) -> torch.Tensor:
        return self.decoder(task.tensor)

    def tile_merge(self, coord_tensor_map: dict[tuple[int, ...], torch.Tensor], grid_spec: GridSpec) -> torch.Tensor:
        blend_extent = grid_spec.tile_spec["blend_extent"]

        sorted_coords = sorted(coord_tensor_map.keys(), key=lambda x: x[0])

        if len(sorted_coords) == 1:
            return coord_tensor_map[sorted_coords[0]].clone()

        result_tiles = []
        for i, coord in enumerate(sorted_coords):
            tile = coord_tensor_map[coord]

            if i > 0:
                blend_size = min(blend_extent, result_tiles[-1].shape[2], tile.shape[2])

                if blend_size > 0:
                    blend_weights = torch.linspace(0, 1, blend_size, device=tile.device, dtype=tile.dtype).view(
                        1, 1, -1
                    )

                    blended = (
                        result_tiles[-1][:, :, -blend_size:] * (1 - blend_weights)
                        + tile[:, :, :blend_size] * blend_weights
                    )

                    result_tiles[-1] = torch.cat([result_tiles[-1][:, :, :-blend_size], blended], dim=2)
                    tile = tile[:, :, blend_size:]

            if tile.shape[2] > 0:
                result_tiles.append(tile)

        if result_tiles:
            return torch.cat(result_tiles, dim=2)

        return coord_tensor_map[sorted_coords[0]].clone()

    def decode(self, z: torch.Tensor, return_dict: bool = True, *args: Any, **kwargs: Any):
        if not self.is_distributed_enabled():
            return super().decode(z, return_dict=return_dict, *args, **kwargs)

        if self.distributed_executor.rank == 0:
            logger.info(
                f"Decode run with distributed executor, parallel_size={self.distributed_executor.parallel_size}"
            )
        result = self.distributed_executor.execute(
            z,
            DistributedOperator(split=self.tile_split, exec=self.tile_exec, merge=self.tile_merge),
            broadcast_result=True,
        )
        if not return_dict:
            return (result,)

        return DecoderOutput(sample=result)
