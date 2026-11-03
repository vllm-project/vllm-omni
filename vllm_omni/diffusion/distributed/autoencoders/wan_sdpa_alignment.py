# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Align Wan VAE attention inputs for memory-efficient SDPA.

Diffusers' ``WanAttentionBlock`` builds Q, K, and V by chunking one QKV
projection along the last dimension. For a 1x1 spatial activation that chunk
is a size-1 sequence whose stride stays 1: PyTorch treats the tensor as
contiguous and ``contiguous()`` does not rewrite unit-dimension strides. The
CUDA memory-efficient kernel then rejects it with
``query is not correctly aligned (strideM)``.

Distributed Wan tiling (256px tiles, stride 192, spatial compression 16)
produces that 1x1 latent at the corner of some resolutions, including ones
sampled by the Cosmos3 stability mix. Repacking only those tensors gives the
kernel a canonical sequence stride and leaves every already-aligned tile on
its original storage.
"""

from __future__ import annotations

import warnings
from importlib import metadata
from types import MethodType

import torch
import torch.nn.functional as F
from torch import nn

# 128-bit vector width used by the memory-efficient SDPA strideM check.
_SDPA_ALIGNMENT_BYTES = 16
_ROOT_ATTR = "_vllm_omni_wan_attention_sdpa_aligned"


def sdpa_sequence_stride_alignment(tensor: torch.Tensor) -> int:
    """Element count the sequence stride must be divisible by for ``tensor``'s dtype."""
    element_size = tensor.element_size()
    if element_size <= 0 or element_size > _SDPA_ALIGNMENT_BYTES:
        return 1
    return _SDPA_ALIGNMENT_BYTES // element_size


def sequence_stride_is_aligned(tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` satisfies the memory-efficient SDPA sequence-stride check.

    The kernel reads ``stride(-2)`` (sequence) and requires the last dimension
    to be contiguous. Size-1 sequence dimensions are included: ``is_contiguous``
    ignores them, which is how a 1x1 query keeps sequence stride 1.
    """
    if tensor.ndim < 2 or tensor.stride(-1) != 1:
        return False
    alignment = sdpa_sequence_stride_alignment(tensor)
    return tensor.stride(-2) % alignment == 0


def _repack_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    """Copy ``tensor`` into a buffer with canonical strides, including size-1 dims."""
    repacked = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
    repacked.copy_(tensor)
    return repacked


def align_sdpa_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Q, K, V views whose sequence strides the memory-efficient kernel accepts.

    Already-aligned inputs are returned unchanged, so normal tiles keep their
    storage and values. Misaligned inputs are copied into a canonical buffer;
    the values are unchanged.
    """
    if sequence_stride_is_aligned(query) and sequence_stride_is_aligned(key) and sequence_stride_is_aligned(value):
        return query, key, value
    return _repack_contiguous(query), _repack_contiguous(key), _repack_contiguous(value)


def split_wan_attention_qkv(qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match diffusers 0.40 ``WanAttentionBlock`` QKV views from ``[B*T, 3C, H, W]``."""
    batch_time, channels_times_3, _, _ = qkv.shape
    fused = qkv.reshape(batch_time, 1, channels_times_3, -1).permute(0, 1, 3, 2).contiguous()
    query, key, value = fused.chunk(3, dim=-1)
    return query, key, value


def aligned_wan_attention_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """``WanAttentionBlock.forward`` with SDPA inputs realigned when required.

    The arithmetic matches diffusers 0.40.0. The only extra work is
    :func:`align_sdpa_qkv`, which is a no-op for every tile whose sequence
    stride is already legal.
    """
    identity = x
    batch_size, channels, time, height, width = x.size()

    hidden = x.permute(0, 2, 1, 3, 4).reshape(batch_size * time, channels, height, width)
    hidden = self.norm(hidden)
    query, key, value = align_sdpa_qkv(*split_wan_attention_qkv(self.to_qkv(hidden)))
    hidden = F.scaled_dot_product_attention(query, key, value)

    hidden = hidden.squeeze(1).permute(0, 2, 1).reshape(batch_size * time, channels, height, width)
    hidden = self.proj(hidden)
    hidden = hidden.view(batch_size, time, channels, height, width)
    hidden = hidden.permute(0, 2, 1, 3, 4)
    return hidden + identity


def install_wan_attention_sdpa_alignment(root: nn.Module) -> None:
    """Point each ``WanAttentionBlock`` under ``root`` at the aligned forward.

    Idempotent. Installed at decode/encode time so the Wan fast path, which
    rejects a custom forward at install time, still sees the diffusers method
    while the model is loading. Existing spatial-shard wrappers are preserved:
    they gather the full spatial extent before attention, rather than tiles.
    Warn on unverified diffusers versions because the forward is copied from
    0.40.0; keep the alignment fix active rather than restoring the tile crash.
    """
    if getattr(root, _ROOT_ATTR, False):
        return
    diffusers_version = metadata.version("diffusers")
    if diffusers_version != "0.40.0":
        warnings.warn(
            "Wan attention SDPA alignment was validated against diffusers 0.40.0, "
            f"but {diffusers_version} is installed. The copied WanAttentionBlock.forward "
            "must be checked against this version for compatibility.",
            RuntimeWarning,
            stacklevel=2,
        )
    for module in root.modules():
        if module.__class__.__name__ != "WanAttentionBlock":
            continue
        # LingBot installs this wrapper before the first decode. Replacing it
        # would drop the all-gather/reshard and silently attend to local shards.
        if getattr(module, "_vllm_omni_spatial_shard_attention", False):
            continue
        module.forward = MethodType(aligned_wan_attention_forward, module)
    setattr(root, _ROOT_ATTR, True)
