# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.distributed as dist
from diffusers.models.autoencoders import AutoencoderKLQwenImage
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import QwenImageCausalConv3d
from diffusers.models.autoencoders.vae import DecoderOutput
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedOperator,
    DistributedVaeMixin,
    GridSpec,
    TileTask,
)
from vllm_omni.diffusion.distributed.autoencoders.qwen_spatial_shard import install_qwen_spatial_shard_decode
from vllm_omni.diffusion.distributed.autoencoders.wan_spatial_shard import WanDistCausalConv3d

logger = init_logger(__name__)


class DistributedAutoencoderKLQwenImage(AutoencoderKLQwenImage, DistributedVaeMixin):
    def clear_cache(self):
        def _count_cached_conv3d(model) -> int:
            return sum(isinstance(module, (QwenImageCausalConv3d, WanDistCausalConv3d)) for module in model.modules())

        self._conv_num = _count_cached_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        self._enc_conv_num = _count_cached_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

    def _spatial_decode_requested(self) -> bool:
        executor = getattr(self, "distributed_executor", None)
        mode = getattr(executor, "parallel_mode", "tile")
        installed = getattr(self, "_qwen_spatial_shard_config", None)
        if installed is not None:
            group, direction, size = installed
            if mode != f"spatial_shard_{direction}" or executor.group is not group or executor.parallel_size != size:
                raise ValueError(
                    "Qwen spatial-shard mode, group or parallel size changed; create a fresh VAE instance."
                )
        return mode in ("spatial_shard_height", "spatial_shard_width")

    def _spatial_decode(self, z: torch.Tensor, return_dict: bool):
        executor = self.distributed_executor
        size = executor.parallel_size
        if size < 1:
            raise ValueError("Qwen spatial-shard parallel size must be positive.")
        if not dist.is_initialized():
            if size != 1:
                raise RuntimeError("Qwen spatial-shard requires initialized distributed execution for multiple ranks.")
        else:
            if size > 1 and executor.group is None:
                raise ValueError("Qwen spatial-shard requires an explicit executor process group for multiple ranks.")
            world_size = dist.get_world_size(executor.group)
            if size != world_size:
                raise ValueError(
                    f"Qwen spatial-shard vae_patch_parallel_size={size} must match executor group size={world_size}."
                )
        if size > 1:
            install_qwen_spatial_shard_decode(
                self, executor.group, executor.parallel_mode.removeprefix("spatial_shard_")
            )
        if z.shape[2] == 0:
            raise ValueError("Qwen spatial-shard decode requires at least one latent frame.")
        self.clear_cache()
        try:
            x = self.post_quant_conv(z)
            chunks = []
            for i in range(z.shape[2]):
                self._conv_idx = [0]
                chunks.append(self.decoder(x[:, :, i : i + 1], feat_cache=self._feat_map, feat_idx=self._conv_idx))
            out = torch.cat(chunks, dim=2).clamp(-1, 1)
        finally:
            self.clear_cache()
        return DecoderOutput(sample=out) if return_dict else (out,)

    def _decode(self, z: torch.Tensor, return_dict: bool = True):
        if self._spatial_decode_requested():
            return self._spatial_decode(z, return_dict)
        return super()._decode(z, return_dict=return_dict)

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any):
        model = super().from_pretrained(*args, **kwargs)
        model.init_distributed()
        return model

    def tile_split(self, z: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        # mostly copy from AutoencoderKL
        _, _, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width = self.tile_sample_min_width - self.tile_sample_stride_width

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        tiletask_list = []
        for i in range(0, height, tile_latent_stride_height):
            for j in range(0, width, tile_latent_stride_width):
                time_list = []
                for k in range(num_frames):
                    self._conv_idx = [0]
                    tile = z[:, :, k : k + 1, i : i + tile_latent_min_height, j : j + tile_latent_min_width]
                    time_list.append(tile)
                tiletask_list.append(
                    TileTask(
                        len(tiletask_list),
                        (i // tile_latent_stride_height, j // tile_latent_stride_width),
                        time_list,
                        workload=time_list[0].shape[3] * time_list[0].shape[4],
                    )
                )
        tile_spec = {
            "sample_height": sample_height,
            "sample_width": sample_width,
            "blend_height": blend_height,
            "blend_width": blend_width,
        }
        grid_spec = GridSpec(
            split_dims=(3, 4),
            grid_shape=(tiletask_list[-1].grid_coord[0] + 1, tiletask_list[-1].grid_coord[1] + 1),
            tile_spec=tile_spec,
            output_dtype=self.dtype,
        )
        return tiletask_list, grid_spec

    def tile_exec(self, task: TileTask) -> torch.Tensor:
        """Decode a single latent tile into RGB space."""
        self.clear_cache()
        time = []
        for k in range(len(task.tensor)):
            self._conv_idx = [0]
            tile = self.post_quant_conv(task.tensor[k])
            decoded = self.decoder(tile, feat_cache=self._feat_map, feat_idx=self._conv_idx)
            time.append(decoded)
        result = torch.cat(time, dim=2)
        return result

    def tile_merge(self, coord_tensor_map: dict[tuple[int, ...], torch.Tensor], grid_spec: GridSpec) -> torch.Tensor:
        """Merge decoded tiles into a full image."""
        grid_h, grid_w = grid_spec.grid_shape
        self.clear_cache()

        result_rows = []
        for i in range(grid_h):
            result_row = []
            for j in range(grid_w):
                tile = coord_tensor_map[(i, j)]
                if i > 0:
                    tile = self.blend_v(coord_tensor_map[(i - 1, j)], tile, grid_spec.tile_spec["blend_height"])
                if j > 0:
                    tile = self.blend_h(coord_tensor_map[(i, j - 1)], tile, grid_spec.tile_spec["blend_width"])
                result_row.append(tile[:, :, :, : self.tile_sample_stride_height, : self.tile_sample_stride_width])
            result_rows.append(torch.cat(result_row, dim=-1))
        dec = torch.cat(result_rows, dim=3)[
            :, :, :, : grid_spec.tile_spec["sample_height"], : grid_spec.tile_spec["sample_width"]
        ]
        return dec

    def tiled_decode(self, z: torch.Tensor, return_dict: bool = True):
        if self._spatial_decode_requested():
            return self._spatial_decode(z, return_dict)
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
