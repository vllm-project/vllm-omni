# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Loader for Marey's in-tree WLAM VAE port."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from vllm_omni.diffusion.models.marey.vae.vae_inference import (
    TwoStageVAEInferenceConfig,
)

logger = logging.getLogger(__name__)


def load_vae(
    vae_config: dict[str, Any],
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[nn.Module | None, str | None]:
    """Load the Marey spatiotemporal VAE from the in-tree WLAM VAE code."""
    vae_path = vae_config.get("cp_path", "")
    if not os.path.exists(vae_path):
        return None, f"VAE checkpoint not found at {vae_path}."

    torch_device = device if isinstance(device, torch.device) else torch.device(device)
    decode_chunking_strategy = (
        "overlap-and-drop" if vae_config.get("extra_context_and_drop_strategy", False) else "basic"
    )

    cfg_kwargs: dict[str, Any] = {
        "checkpoint": vae_path,
        "frame_chunk_len": vae_config["frame_chunk_len"],
        "decode_chunking_strategy": decode_chunking_strategy,
        "scaling_factor": vae_config.get("scaling_factor", 1.0),
        "bias_factor": vae_config.get("bias_factor", 0.0),
        "max_batch_size": 4,
    }
    if "valid_skip_n_blocks" in vae_config:
        cfg_kwargs["valid_skip_n_blocks"] = vae_config["valid_skip_n_blocks"]
    if "torch_compile_kwargs" in vae_config:
        cfg_kwargs["torch_compile_kwargs"] = vae_config["torch_compile_kwargs"]

    vae = TwoStageVAEInferenceConfig(**cfg_kwargs).make(
        device=torch_device,
        dtype=dtype,
    )
    vae = vae.eval()
    logger.info(
        "Loaded in-tree WLAM Marey VAE (latent_dim=%s, downsample=%s) "
        "from %s. frame_chunk_len=%s max_batch_size=%s decode_chunking_strategy=%s",
        vae.latent_dim,
        vae.model.get_downsample_factors(0),
        vae_path,
        vae.cfg.frame_chunk_len,
        vae.cfg.max_batch_size,
        decode_chunking_strategy,
    )
    return vae


def build_marey_vae_sp_hooks() -> tuple[
    Callable[[torch.Tensor], torch.Tensor] | None,
    Callable[[torch.Tensor], torch.Tensor] | None,
]:
    """Build ``(sp_shard, sp_gather)`` callables over vllm-omni's SP group.

    Returns ``(None, None)`` when SP is disabled. The returned pair shares a
    closure so the pad_size produced by ``sp_shard`` is consumed by the
    matching ``sp_gather``.
    """
    from vllm_omni.diffusion.distributed.parallel_state import (
        get_sequence_parallel_world_size,
    )

    if get_sequence_parallel_world_size() <= 1:
        return None, None

    from vllm_omni.diffusion.distributed.sp_sharding import (
        sp_gather as _sp_gather_primitive,
    )
    from vllm_omni.diffusion.distributed.sp_sharding import (
        sp_shard_with_padding,
    )

    state = {"pad_size": 0}

    def sp_shard(x: torch.Tensor, dim: int) -> torch.Tensor:
        sharded, pad = sp_shard_with_padding(x, dim=dim)
        state["pad_size"] = pad
        return sharded

    def sp_gather(x: torch.Tensor, dim: int) -> torch.Tensor:
        y = _sp_gather_primitive(x, dim=dim)
        pad = state["pad_size"]
        logging.info(f"sp_gather: pad: {pad}")
        if pad > 0:
            y = y.narrow(0, 0, y.size(0) - pad)
        return y

    return sp_shard, sp_gather
