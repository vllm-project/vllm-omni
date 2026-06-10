# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project and the xDiT authors.
#
# This module is adapted from xDiT (https://github.com/xdit-project/xdit)

"""
PipeFusion runtime state management.

Tracks patch metadata (sizes, token ranges, split dimension) and
provides the per-patch index counter used by the pipeline, scheduler,
and transformer mixins.
"""

from __future__ import annotations

from typing import Literal

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.parallel_state import get_pipeline_parallel_world_size, get_pp_group

logger = init_logger(__name__)

_PF_RUNTIME: PipeFusionRuntime | None = None


class PipeFusionRuntime:
    patch_mode: bool
    pipeline_patch_idx: int
    pp_patches_height: list[int] | None
    pp_patches_start_end_idx: list[tuple[int, int]] | None
    pp_patches_token_num: list[int] | None
    pp_patches_token_start_end_idx: list[tuple[int, int]] | None

    def __init__(self):
        self.patch_size: tuple[int, int, int] = (1, 2, 2)
        self.patch_mode = False
        self.pipeline_patch_idx = 0
        self.warmup_steps = 1
        self.split_dim: Literal["height", "temporal"] = "height"
        self.cache_key: Literal["inputs", "inputs_uncond"] = "inputs"
        self.patch_idx_tensor = torch.tensor(0, dtype=torch.int32)
        self.warmup_cache_timestep: torch.Tensor | None = None
        self.update_warmup_cache = True

    def set_input_parameters(self, latents: torch.Tensor, dtype):
        self._calc_patches_metadata(latents)
        self._reset_recv_buffer(dtype)

    def set_cache_key(self, key: Literal["inputs", "inputs_uncond"]) -> None:
        if key not in ("inputs", "inputs_uncond"):
            raise ValueError(f"Invalid PipeFusion cache key: {key!r}. Must be 'inputs' or 'inputs_uncond'.")
        self.cache_key = key

    def set_run_config(self, warmup_steps: int | None = None, split_dim: str | None = None) -> None:
        if warmup_steps is not None:
            if not isinstance(warmup_steps, int) or warmup_steps < 1:
                raise ValueError(f"PipeFusion warmup_steps must be a positive integer, got {warmup_steps!r}.")
            self.warmup_steps = warmup_steps
        if split_dim is not None:
            if split_dim not in ("height", "temporal"):
                raise ValueError(f"Invalid PipeFusion split_dim: {split_dim!r}. Must be 'height' or 'temporal'.")
            self.split_dim = split_dim

    def set_patched_mode(self, patch_mode: bool):
        self.patch_mode = patch_mode
        self.pipeline_patch_idx = 0
        if hasattr(self, "patch_idx_tensor"):
            self.patch_idx_tensor.fill_(0)

    def next_patch(self):
        if self.patch_mode:
            self.pipeline_patch_idx += 1
            if self.pipeline_patch_idx == self.num_pipeline_patch:
                self.pipeline_patch_idx = 0
        else:
            self.pipeline_patch_idx = 0
        if hasattr(self, "patch_idx_tensor"):
            self.patch_idx_tensor.fill_(self.pipeline_patch_idx)

    def _calc_patch_metadata(self, seq_length):
        lengths = [seq_length // self.num_pipeline_patch] * (self.num_pipeline_patch - 1)
        # Give more tokens to the last patch, if it's the case.
        lengths.append(seq_length // self.num_pipeline_patch + seq_length % self.num_pipeline_patch)
        start = 0
        start_end_idx = []
        for num in lengths:
            start_end_idx.append((start, start + num))
            start += num
        return lengths, start_end_idx

    def _calc_patches_metadata(self, latents):
        self.num_pipeline_patch = get_pipeline_parallel_world_size()

        p_t, p_h, p_w = self.patch_size
        ppf = latents.size(-3) // p_t  # post-patch frames
        pph = latents.size(-2) // p_h  # post-patch height (full)
        ppw = latents.size(-1) // p_w  # post-patch width

        # Store post-patch spatial dims for KV cache reshape in attention
        self.ppf = ppf
        self.pph = pph
        self.ppw = ppw

        # Create 1-element tensors for dynamic shape tracking
        self.ppf_tensor = torch.tensor(ppf, dtype=torch.int64, device=latents.device)
        self.pph_tensor = torch.tensor(pph, dtype=torch.int64, device=latents.device)
        self.ppw_tensor = torch.tensor(ppw, dtype=torch.int64, device=latents.device)

        if self.split_dim == "height":
            # Split along spatial height
            self.latent_split_dim = -2  # dim in 5D [B,C,T,H,W]

            # Post-patch heights split among patches
            self.pp_patches_post_height, self.pp_patches_post_start_end_idx = self._calc_patch_metadata(pph)
            self.pp_patches_post_frames = None  # not split

            # Latent-space heights (multiply by p_h to ensure divisibility)
            self.pp_patches_height = [h * p_h for h in self.pp_patches_post_height]
            start = 0
            self.pp_patches_start_end_idx = []
            for h in self.pp_patches_height:
                self.pp_patches_start_end_idx.append((start, start + h))
                start += h

            # Token count: each patch covers all frames and widths but partial height
            self.pp_patches_token_num = [h * ppw * ppf for h in self.pp_patches_post_height]

        elif self.split_dim == "temporal":
            # Split along temporal (frames) dimension
            self.latent_split_dim = -3  # dim in 5D [B,C,T,H,W]

            # Post-patch frames split among patches
            self.pp_patches_post_frames, self.pp_patches_post_start_end_idx = self._calc_patch_metadata(ppf)
            self.pp_patches_post_height = None  # not split

            # Latent-space frames (multiply by p_t to ensure divisibility)
            self.pp_patches_height = [f * p_t for f in self.pp_patches_post_frames]
            start = 0
            self.pp_patches_start_end_idx = []
            for f in self.pp_patches_height:
                self.pp_patches_start_end_idx.append((start, start + f))
                start += f

            # Token count: each patch covers all heights and widths but partial frames
            self.pp_patches_token_num = [f * pph * ppw for f in self.pp_patches_post_frames]

        else:
            raise ValueError(f"Unknown split_dim: {self.split_dim}. Use 'height' or 'temporal'.")

        # Calculate start/end indices for each patch's tokens
        start = 0
        self.pp_patches_token_start_end_idx = []
        for num in self.pp_patches_token_num:
            self.pp_patches_token_start_end_idx.append((start, start + num))
            start += num

    def _reset_recv_buffer(self, dtype):
        get_pp_group().reset_buffer()
        get_pp_group().set_config(dtype)
        self.patch_idx_tensor = torch.tensor(self.pipeline_patch_idx, dtype=torch.int32, device=get_pp_group().device)


def initialize_pupefusion_runtime():
    global _PF_RUNTIME
    if _PF_RUNTIME is not None:
        RuntimeError("PipeFusion runtime is already initialized...")
    _PF_RUNTIME = PipeFusionRuntime()


def is_pipefusion_initialized() -> bool:
    return _PF_RUNTIME is not None


def get_pipefusion_runtime():
    if _PF_RUNTIME is None:
        raise RuntimeError("PipeFusion runtime has not been initialized.")
    return _PF_RUNTIME


def set_pipefusion_cache_key_if_initialized(key: Literal["inputs", "inputs_uncond"]) -> None:
    if _PF_RUNTIME is not None:
        _PF_RUNTIME.set_cache_key(key)
