# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from diffusers import AutoencoderKLLTX2Video
from diffusers.models.autoencoders.vae import DecoderOutput
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedOperator,
    DistributedVaeExecutor,
    DistributedVaeMixin,
    GridSpec,
    TileTask,
)

logger = init_logger(__name__)


class LTX2VaeExecutor(DistributedVaeExecutor):
    def _pack_local_tiles_without_meta(self, local_results, global_padding_shape, device, dtype):
        tile_tensor = torch.zeros(global_padding_shape, device=device, dtype=dtype)
        for idx, (_, t_tensor) in enumerate(local_results):
            slices = tuple(slice(0, s) for s in t_tensor.shape)
            tile_tensor[idx][slices] = t_tensor
        return tile_tensor

    def _unpack_tiles_with_known_metadata(self, tile_gather, assigned, tile_output_shapes):
        coord_tensor_map = {}
        for r, rank_tasks in enumerate(assigned):
            tiles_src = tile_gather[r]
            for idx, task in enumerate(rank_tasks):
                output_shape = tile_output_shapes[task.tile_id]
                slices = tuple(slice(0, dim) for dim in output_shape)
                coord_tensor_map[task.grid_coord] = tiles_src[idx][slices]

        return coord_tensor_map

    def execute(self, z: torch.Tensor, operator: DistributedOperator, broadcast_result: bool = True):
        pp_size = min(self.parallel_size, self.world_size)

        tiletask_list, grid_spec = operator.split(z)
        max_tile_output_shape = grid_spec.tile_spec.get("max_tile_output_shape")
        tile_output_shapes = grid_spec.tile_spec.get("tile_output_shapes")
        if max_tile_output_shape is None or tile_output_shapes is None:
            return super().execute(z, operator, broadcast_result=broadcast_result)

        assigned = self._balance_tasks(tiletask_list, pp_size)
        local_tasks = assigned[self.rank] if self.rank < pp_size else []
        local_results = [(t.tile_id, operator.exec(t)) for t in local_tasks]

        output_dtype = grid_spec.output_dtype if grid_spec.output_dtype is not None else z.dtype
        max_local_tile_count = max(len(tasks) for tasks in assigned)
        global_padding_shape = [max_local_tile_count, *max_tile_output_shape]
        local_tile_tensor = self._pack_local_tiles_without_meta(
            local_results,
            global_padding_shape,
            z.device,
            output_dtype,
        )
        tile_gather = self.gather_tensors(local_tile_tensor)

        if self.rank != 0:
            result = torch.empty(0, device=z.device)
        else:
            coord_tensor_map = self._unpack_tiles_with_known_metadata(tile_gather, assigned, tile_output_shapes)
            result = operator.merge(coord_tensor_map, grid_spec)

        if broadcast_result:
            result = self._sync_final_result(result, z.ndim, z.device, output_dtype)
        return result


class DistributedAutoencoderKLLTX2Video(AutoencoderKLLTX2Video, DistributedVaeMixin):
    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any):
        model = super().from_pretrained(*args, **kwargs)
        model.init_distributed()
        return model

    def init_distributed(self):
        self.distributed_executor = LTX2VaeExecutor()

    def tile_split(self, z: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        _, _, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio
        sample_num_frames = (num_frames - 1) * self.temporal_compression_ratio + 1
        output_channels = int(getattr(getattr(self, "config", None), "out_channels", 3))

        tiletask_list = []
        tile_output_shapes = {}
        for i in range(0, height, tile_latent_stride_height):
            for j in range(0, width, tile_latent_stride_width):
                tile = z[:, :, :num_frames, i : i + tile_latent_min_height, j : j + tile_latent_min_width]
                tile_id = len(tiletask_list)
                tile_output_shapes[tile_id] = (
                    z.shape[0],
                    output_channels,
                    sample_num_frames,
                    tile.shape[3] * self.spatial_compression_ratio,
                    tile.shape[4] * self.spatial_compression_ratio,
                )
                tiletask_list.append(
                    TileTask(
                        tile_id,
                        (i // tile_latent_stride_height, j // tile_latent_stride_width),
                        tile,
                        workload=tile.shape[2] * tile.shape[3] * tile.shape[4],
                    )
                )

        tile_spec = {
            "sample_height": sample_height,
            "sample_width": sample_width,
            "blend_height": self.tile_sample_min_height - self.tile_sample_stride_height,
            "blend_width": self.tile_sample_min_width - self.tile_sample_stride_width,
            "tile_sample_stride_height": self.tile_sample_stride_height,
            "tile_sample_stride_width": self.tile_sample_stride_width,
            "max_tile_output_shape": (
                z.shape[0],
                output_channels,
                sample_num_frames,
                self.tile_sample_min_height,
                self.tile_sample_min_width,
            ),
            "tile_output_shapes": tile_output_shapes,
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
        temb: torch.Tensor | None = None,
        causal: bool | None = None,
    ) -> torch.Tensor:
        if hasattr(self, "clear_cache"):
            self.clear_cache()
        return self.decoder(task.tensor, temb, causal=causal)

    def tile_merge(self, coord_tensor_map: dict[tuple[int, ...], torch.Tensor], grid_spec: GridSpec) -> torch.Tensor:
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
        return dec

    def tiled_decode(
        self,
        z: torch.Tensor,
        temb: torch.Tensor | None = None,
        causal: bool | None = None,
        return_dict: bool = True,
    ):
        if not self.is_distributed_enabled():
            if causal is None:
                return super().tiled_decode(z, temb, return_dict=return_dict)
            return super().tiled_decode(z, temb, causal=causal, return_dict=return_dict)

        logger.debug("LTX2 video VAE decode running with distributed tiled executor")
        result = self.distributed_executor.execute(
            z,
            DistributedOperator(
                split=self.tile_split,
                exec=lambda task: self.tile_exec(task, temb=temb, causal=causal),
                merge=self.tile_merge,
            ),
            broadcast_result=False,
        )
        if not return_dict:
            return (result,)

        return DecoderOutput(sample=result)
