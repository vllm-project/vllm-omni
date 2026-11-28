# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Fallback for backends without HunyuanImage3 spatial VAE support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_hunyuan import DistributedAutoencoderKLHunyuan

logger = init_logger(__name__)


def spatial_vae_forward(
    vae: DistributedAutoencoderKLHunyuan, x: torch.Tensor, stage: Literal["encode", "decode"]
) -> torch.Tensor | None:
    if stage == "decode" and vae.distributed_executor.parallel_mode == "spatial_shard_height":
        logger.warning_once("HunyuanImage3 spatial VAE decode is supported only on Ascend NPU; using tile execution")
    return None
