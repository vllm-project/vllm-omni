# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl import (
    DistributedAutoencoderKL_base,
)
from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedOperator,
    TileTask,
)
from vllm_omni.diffusion.models.bagel.autoencoder import AutoEncoder, AutoEncoderParams


class DistributedAutoEncoderBagel(DistributedAutoencoderKL_base, AutoEncoder):
    def __init__(self, params: AutoEncoderParams):
        AutoEncoder.__init__(self, params)

        self.config = SimpleNamespace(
            block_out_channels=tuple(params.ch_mult),
            out_channels=params.out_ch,
            use_post_quant_conv=False,
        )
        self.use_slicing = False
        self.use_tiling = False
        self.tile_sample_min_size = int(params.resolution)
        self.tile_latent_min_size = int(params.resolution // params.downsample)
        self.tile_overlap_factor = 0.25

        self.init_distributed()

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def tile_exec(self, task: TileTask) -> torch.Tensor:
        z = task.tensor / self.scale_factor + self.shift_factor
        return self.decoder(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if not self.is_distributed_enabled():
            return AutoEncoder.decode(self, z)

        split, exec, merge = self._strategy_select(z)
        if split is None:
            return AutoEncoder.decode(self, z)

        return self.distributed_decoder.execute(
            z,
            DistributedOperator(split=split, exec=exec, merge=merge),
            broadcast_result=True,
        )

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, y, :] * (y / blend_extent)
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, x] * (x / blend_extent)
        return b
