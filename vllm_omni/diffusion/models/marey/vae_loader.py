# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Loader for Marey's in-tree WLAM VAE port."""

from __future__ import annotations

import logging
import os
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
        "max_batch_size": vae_config.get("max_batch_size", 8),
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
