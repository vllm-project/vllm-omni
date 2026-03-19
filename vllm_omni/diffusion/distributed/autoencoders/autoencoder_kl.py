# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from typing import Any

import torch
from diffusers.models.autoencoders import AutoencoderKL as Diffusers_AutoencoderKL
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedOperator,
    DistributedVaeMixin,
    GridSpec,
    TileTask,
)

# from vllm_omni.diffusion.models.nextstep_1_1.modeling_flux_vae import AutoencoderKL as Next_Step_AutoencoderKL

logger = init_logger(__name__)


# We use base class because some model re-implement AutoencoderKL, but split is share.
class DistributedAutoencoderKL_base(DistributedVaeMixin):
    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any):
        model = super().from_pretrained(*args, **kwargs)
        model.init_distributed()
        return model

    def tile_split(self, z: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        # mostly copy from AutoencoderKL
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        # Split z into overlapping 64x64 tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        tiletask_list = []
        for i in range(0, z.shape[2], overlap_size):
            for j in range(0, z.shape[3], overlap_size):
                tile = z[:, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
                tiletask_list.append(
                    TileTask(
                        len(tiletask_list),
                        (i // overlap_size, j // overlap_size),
                        tile,
                        workload=tile.shape[2] * tile.shape[3],
                    )
                )

        tile_spec = {
            "blend_extent": blend_extent,
            "row_limit": row_limit,
        }
        grid_spec = GridSpec(
            split_dims=(2, 3),
            grid_shape=(tiletask_list[-1].grid_coord[0] + 1, tiletask_list[-1].grid_coord[1] + 1),
            tile_spec=tile_spec,
        )
        return tiletask_list, grid_spec

    def tile_exec(self, task: TileTask) -> torch.Tensor:
        """Decode a single latent tile into RGB space."""
        tile = task.tensor
        if self.config.use_post_quant_conv:
            tile = self.post_quant_conv(tile)
        decoded = self.decoder(tile)
        return decoded

    def tile_merge(self, coord_tensor_map: dict[tuple[int, ...], torch.Tensor], grid_spec: GridSpec) -> torch.Tensor:
        """Merge decoded tiles into a full image."""

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
            result_rows.append(torch.cat(result_row, dim=3))

        dec = torch.cat(result_rows, dim=2)
        return dec

    def patch_split(self, z: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        overlap_latent = int(self.tile_latent_min_size * float(self.tile_overlap_factor))
        halo_base = max(0, overlap_latent // 2)

        _, _, latent_h, latent_w = z.shape
        scale = int(2 ** (len(self.config.block_out_channels) - 1))
        max_parallel_size = self.distributed_decoder.parallel_size

        root = int(math.sqrt(max_parallel_size))
        for rows in range(root, 0, -1):
            if max_parallel_size % rows == 0:
                grid_rows, grid_cols = rows, max_parallel_size // rows
                break
        tiletask_list = []
        halo_size = dict()
        for i in range(grid_rows):
            for j in range(grid_cols):
                h0 = (i * latent_h) // grid_rows
                h1 = ((i + 1) * latent_h) // grid_rows
                w0 = (j * latent_w) // grid_cols
                w1 = ((j + 1) * latent_w) // grid_cols

                core_h = max(0, h1 - h0)
                core_w = max(0, w1 - w0)
                halo = max(halo_base, min(core_h, core_w) // 2)
                ph0 = max(0, h0 - halo)
                ph1 = min(latent_h, h1 + halo)
                pw0 = max(0, w0 - halo)
                pw1 = min(latent_w, w1 + halo)
                tile = z[:, :, ph0:ph1, pw0:pw1]
                tiletask_list.append(TileTask(len(tiletask_list), (i, j), tile, tile.shape[2] * tile.shape[3]))
                halo_size[(i, j)] = {
                    "up": h0 - ph0,
                    "down": ph1 - h1,
                    "left": w0 - pw0,
                    "right": pw1 - w1,
                }

        tile_spec = {
            "halo_size": halo_size,
            "scale": scale,
        }
        grid_spec = GridSpec(
            split_dims=(2, 3),
            grid_shape=(tiletask_list[-1].grid_coord[0] + 1, tiletask_list[-1].grid_coord[1] + 1),
            tile_spec=tile_spec,
            output_dtype=self.dtype,
        )
        return tiletask_list, grid_spec

    def patch_exec(self, task: TileTask) -> torch.Tensor:
        return self.tile_exec(task)

    def patch_merge(self, coord_tensor_map: dict[tuple[int, ...], torch.Tensor], grid_spec: GridSpec) -> torch.Tensor:
        grid_h, grid_w = grid_spec.grid_shape
        result_rows = []
        for i in range(grid_h):
            result_row = []
            for j in range(grid_w):
                halo = grid_spec.tile_spec["halo_size"][(i, j)]
                scale = grid_spec.tile_spec["scale"]

                tile = coord_tensor_map[(i, j)]
                halo_up = halo["up"] * scale
                halo_down = halo["down"] * scale
                halo_left = halo["left"] * scale
                halo_right = halo["right"] * scale

                core_tile = tile[
                    :,
                    :,
                    halo_up : (None if halo_down == 0 else -halo_down),
                    halo_left : (None if halo_right == 0 else -halo_right),
                ]
                result_row.append(core_tile)
            result_rows.append(torch.cat(result_row, dim=3))
        dec = torch.cat(result_rows, dim=2)
        return dec

    def _strategy_select(self, z: torch.Tensor):
        tile_latent_min_size = getattr(self, "tile_latent_min_size", None)
        tile_overlap_factor = getattr(self, "tile_overlap_factor", None)
        if tile_latent_min_size is None or tile_overlap_factor is None:
            return None, None, None
        if z.shape[-1] > tile_latent_min_size or z.shape[-2] > tile_latent_min_size:
            return self.tile_split, self.tile_exec, self.tile_merge

        return self.patch_split, self.patch_exec, self.patch_merge

    # Normally, we should override tiled_decode. However, since we also need to
    # support the patch split strategy, we override decode instead.
    def decode(self, z: torch.Tensor, return_dict: bool = True, *args: Any, **kwargs: Any):
        if not self.is_distributed_enabled():
            return super().decode(z, return_dict=return_dict, *args, **kwargs)

        split, exec, merge = self._strategy_select(z)

        if split is not None:
            strategy = "tile" if split == self.tile_split else "patch"
            logger.info(f"Decode run with distributed executor, split strategy is {strategy}")
            result = self.distributed_decoder.execute(
                z, DistributedOperator(split=split, exec=exec, merge=merge), broadcast_result=False
            )
            if not return_dict:
                return (result,)

            from diffusers.models.autoencoders.vae import DecoderOutput

            return DecoderOutput(sample=result)
        else:
            return super().decode(z, return_dict=return_dict, *args, **kwargs)


class DistributedAutoencoderKL(DistributedAutoencoderKL_base, Diffusers_AutoencoderKL):
    pass


# Next_Step_AutoencoderKL not support tiling now, so we currently disable it.

# class DistributedAutoencoderKL_NextStep(DistributedAutoencoderKL_base, Next_Step_AutoencoderKL):
#    pass
