# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# The halo-exchange spatial-parallel decode here is adapted from SGLang's
# spatial-parallel VAE decode
#   https://github.com/sgl-project/sglang
#   (python/sglang/multimodal_gen/runtime/layers/parallel_conv.py)
# which is in turn adapted from FastVideo (https://github.com/hao-ai-lab/FastVideo).
# This version generalizes the height-only sharding to shard along height or
# width and adds the Wan causal-conv ``feat_cache`` handling.
"""Spatially-sharded Wan VAE decode.

The existing distributed Wan VAE path shards *tiles*.  This module adds an
opt-in decode backend that shards decoder feature maps along height or width and
exchanges boundary rows/columns before spatial convolutions.  It is
intentionally decode-only and keeps checkpoint loading unchanged by patching the
already-loaded decoder.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from itertools import chain
from types import MethodType
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify
from diffusers.models.autoencoders.vae import DecoderOutput
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.autoencoders.wan_decoder_utils import _persistent_input_buffer
from vllm_omni.diffusion.models.interface import DecodedChunkConsumer

logger = init_logger(__name__)


@dataclass(frozen=True)
class SpatialShardContext:
    input_extent: int
    local_input_extent: int
    split_dim: str
    rank: int
    world_size: int


_SPATIAL_SHARD_CONTEXT: ContextVar[SpatialShardContext | None] = ContextVar(
    "wan_vae_spatial_shard_context",
    default=None,
)


def _spatial_dim(split_dim: str) -> int:
    if split_dim == "height":
        return -2
    if split_dim == "width":
        return -1
    raise ValueError(f"Unsupported Wan VAE split_dim={split_dim!r}; expected 'height' or 'width'.")


def _narrow_along_dim(x: torch.Tensor, dim: int, start: int, length: int) -> torch.Tensor:
    if dim < 0:
        dim += x.dim()
    return x.narrow(dim, start, length)


def _global_rank(group: dist.ProcessGroup, group_rank: int) -> int:
    try:
        return dist.get_global_rank(group, group_rank)
    except Exception:
        return group_rank


def _rank_world(group: dist.ProcessGroup) -> tuple[int, int]:
    return dist.get_rank(group), dist.get_world_size(group)


def _pad_along_dim(x: torch.Tensor, pad: int, dim: int, value: float = 0.0) -> torch.Tensor:
    if pad <= 0:
        return x
    shape = list(x.shape)
    shape[dim] = pad
    padding = torch.full(shape, value, dtype=x.dtype, device=x.device)
    return torch.cat([x, padding], dim=dim)


def _maybe_contiguous_for_shard_gather(x: torch.Tensor) -> torch.Tensor:
    if (
        x.dim() == 5
        and hasattr(torch, "channels_last_3d")
        and x.is_contiguous(memory_format=torch.channels_last_3d)
        and not x.is_contiguous()
    ):
        return x.contiguous()
    return x


def _halo_memory_format(reference: torch.Tensor) -> torch.memory_format:
    if reference.dim() > 1 and reference.stride(1) == 1:
        if reference.dim() == 5 and hasattr(torch, "channels_last_3d"):
            return torch.channels_last_3d
        if reference.dim() == 4:
            return torch.channels_last
    return torch.contiguous_format


def _current_full_extent(local_extent: int) -> int | None:
    ctx = _SPATIAL_SHARD_CONTEXT.get()
    if ctx is None:
        return None
    if ctx.local_input_extent <= 0:
        return None
    scale = local_extent / ctx.local_input_extent
    rounded_scale = round(scale)
    if not math.isclose(scale, rounded_scale, rel_tol=0.0, abs_tol=1e-6):
        return None
    return ctx.input_extent * rounded_scale


def _local_valid_extent(local_extent: int) -> int:
    ctx = _SPATIAL_SHARD_CONTEXT.get()
    full_extent = _current_full_extent(local_extent)
    if ctx is None or full_extent is None:
        return local_extent
    start = ctx.rank * local_extent
    return max(0, min(local_extent, full_extent - start))


def _zero_invalid_extent(x: torch.Tensor, *, split_dim: str, inplace: bool = False) -> torch.Tensor:
    dim = _spatial_dim(split_dim)
    dim_size = x.shape[dim]
    valid_extent = _local_valid_extent(dim_size)
    if valid_extent >= dim_size:
        return x
    if not inplace:
        x = x.clone()
    invalid = _narrow_along_dim(x, dim, valid_extent, dim_size - valid_extent)
    invalid.zero_()
    return x


def split_for_parallel_decode(
    x: torch.Tensor,
    *,
    upsample_count: int,
    split_dim: str = "height",
    group: dist.ProcessGroup | None = None,
    rank: int | None = None,
    world_size: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Shard latent/feature spatial extent and return expected full output extent."""
    if group is not None:
        rank, world_size = _rank_world(group)
    rank = 0 if rank is None else int(rank)
    world_size = 1 if world_size is None else int(world_size)
    if world_size < 1:
        raise ValueError(f"Wan VAE world_size must be >= 1, got {world_size}.")
    if not 0 <= rank < world_size:
        raise ValueError(f"Wan VAE rank must satisfy 0 <= rank < world_size, got rank={rank}, world_size={world_size}.")

    dim = _spatial_dim(split_dim)
    expected_extent = x.shape[dim] * (2**upsample_count)
    if world_size <= 1:
        return x, expected_extent

    pad = (world_size - (x.shape[dim] % world_size)) % world_size
    if pad:
        x = _pad_along_dim(x, pad, dim=dim)
    chunk_size = x.shape[dim] // world_size
    return _narrow_along_dim(x, dim, rank * chunk_size, chunk_size).contiguous(), expected_extent


def all_gather_along_dim(
    x: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    dim: int,
    dst: int | None = None,
) -> torch.Tensor:
    rank, world_size = _rank_world(group)
    if world_size <= 1:
        return x
    x = _maybe_contiguous_for_shard_gather(x)
    gathered = [torch.empty_like(x) for _ in range(world_size)]
    # NCCL has no rank-local gather, so every rank joins the collective; only ``dst``
    # keeps the assembled tensor while the rest drop their copies.
    dist.all_gather(gathered, x.contiguous(), group=group)
    if dst is not None and rank != dst:
        return x.new_zeros(0)
    return torch.cat(gathered, dim=dim)


def reshard_from_trimmed_extent(
    x: torch.Tensor,
    *,
    local_extent: int,
    split_dim: str,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    rank, world_size = _rank_world(group)
    if world_size <= 1:
        return x

    dim = _spatial_dim(split_dim)
    ctx = _SPATIAL_SHARD_CONTEXT.get()
    if ctx is not None and ctx.rank != rank:
        raise RuntimeError(
            "Wan VAE spatial-shard rank mismatch while resharding: "
            f"group_rank={rank}, context_rank={ctx.rank}, split_dim={split_dim!r}."
        )

    valid_extent = _local_valid_extent(local_extent)
    actual_extent = x.shape[dim]
    start = min(rank * local_extent, actual_extent)
    valid_extent = min(valid_extent, actual_extent - start)
    local = _narrow_along_dim(x, dim, start, valid_extent).contiguous()
    if valid_extent < local_extent:
        local = _pad_along_dim(local, local_extent - valid_extent, dim=dim)
    return local


def gather_and_trim_extent(
    x: torch.Tensor,
    *,
    expected_extent: int | None,
    split_dim: str,
    group: dist.ProcessGroup,
    dst: int | None = None,
) -> torch.Tensor:
    dim = _spatial_dim(split_dim)
    rank, _ = _rank_world(group)
    out = all_gather_along_dim(x, group=group, dim=dim, dst=dst)
    if dst is not None and rank != dst:
        return out
    if expected_extent is not None and out.shape[dim] != expected_extent:
        out = _narrow_along_dim(out, dim, 0, expected_extent).contiguous()
    return out


def _ensure_recv_buf(
    recv_buf: torch.Tensor | None,
    reference: torch.Tensor,
    *,
    memory_format: torch.memory_format = torch.contiguous_format,
) -> torch.Tensor:
    # CPU/Gloo and the reference path use ordinary contiguous buffers. The
    # optimized CUDA path can retain the activation's channels-last ordering.
    if (
        recv_buf is None
        or recv_buf.shape != reference.shape
        or recv_buf.dtype != reference.dtype
        or recv_buf.device != reference.device
        or not recv_buf.is_contiguous(memory_format=memory_format)
    ):
        return torch.empty(
            reference.shape,
            dtype=reference.dtype,
            device=reference.device,
            memory_format=memory_format,
        )
    return recv_buf


def _halo_exchange_p2p(
    *,
    rank: int,
    world_size: int,
    group: dist.ProcessGroup,
    top_row_ref: torch.Tensor,
    bottom_row_ref: torch.Tensor,
    recv_top_buf: torch.Tensor,
    recv_bottom_buf: torch.Tensor,
) -> None:
    memory_format = _halo_memory_format(recv_top_buf)
    packed = False
    if (
        memory_format != torch.contiguous_format
        and top_row_ref.is_cuda
        and top_row_ref.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
    ):
        from .wan_vae_fastpath.triton_spatial_halo import launch_halo_pair

        top_row = torch.empty_like(recv_top_buf)
        bottom_row = torch.empty_like(recv_bottom_buf)
        packed = launch_halo_pair(top_row_ref, bottom_row_ref, top_row, bottom_row)
    p2p_ops = []
    if rank > 0:
        prev_rank = _global_rank(group, rank - 1)
        if not packed:
            top_row = top_row_ref.contiguous(memory_format=memory_format)
        p2p_ops.append(dist.P2POp(dist.irecv, recv_top_buf, prev_rank, group))
        p2p_ops.append(dist.P2POp(dist.isend, top_row, prev_rank, group))
    else:
        recv_top_buf.zero_()

    if rank < world_size - 1:
        next_rank = _global_rank(group, rank + 1)
        if not packed:
            bottom_row = bottom_row_ref.contiguous(memory_format=memory_format)
        p2p_ops.append(dist.P2POp(dist.isend, bottom_row, next_rank, group))
        p2p_ops.append(dist.P2POp(dist.irecv, recv_bottom_buf, next_rank, group))
    else:
        recv_bottom_buf.zero_()

    if p2p_ops:
        reqs = dist.batch_isend_irecv(p2p_ops)
        for req in reqs:
            req.wait()


def _copy_received_halos(
    top: torch.Tensor, bottom: torch.Tensor, out_top: torch.Tensor, out_bottom: torch.Tensor
) -> None:
    if (
        top.is_cuda
        and top.stride(1) == 1
        and top.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
    ):
        from .wan_vae_fastpath.triton_spatial_halo import launch_halo_pair

        if launch_halo_pair(top, bottom, out_top, out_bottom):
            return
    out_top.copy_(top)
    out_bottom.copy_(bottom)


def _exchange_input_halos(module: nn.Module, buf: torch.Tensor, halo: int, rank: int, world_size: int) -> None:
    """Exchange the padded interior's boundaries into its reserved halo slots."""
    dim = module.split_tensor_dim
    top = _narrow_along_dim(buf, dim, halo, halo)
    bottom = _narrow_along_dim(buf, dim, buf.shape[dim] - 2 * halo, halo)
    memory_format = _halo_memory_format(buf) if buf.is_cuda and module._backend == "nccl" else torch.contiguous_format
    module._halo_recv_top_buf = _ensure_recv_buf(module._halo_recv_top_buf, top, memory_format=memory_format)
    module._halo_recv_bottom_buf = _ensure_recv_buf(module._halo_recv_bottom_buf, bottom, memory_format=memory_format)
    _halo_exchange_p2p(
        rank=rank,
        world_size=world_size,
        group=module.group,
        top_row_ref=top,
        bottom_row_ref=bottom,
        recv_top_buf=module._halo_recv_top_buf,
        recv_bottom_buf=module._halo_recv_bottom_buf,
    )
    _copy_received_halos(
        module._halo_recv_top_buf,
        module._halo_recv_bottom_buf,
        _narrow_along_dim(buf, dim, 0, halo),
        _narrow_along_dim(buf, dim, buf.shape[dim] - halo, halo),
    )


def _spatial_conv_forward(
    module: nn.Module, x: torch.Tensor, local_extent: int, *, return_bias: bool
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
    """Convolve an assembled shard, trim boundaries, and apply its bias contract."""
    trim = module._trim_cache.get(local_extent)
    if trim is None:
        rank, world_size = _rank_world(module.group)
        trim = _compute_conv_trim_params(
            local_extent=local_extent,
            rank=rank,
            world_size=world_size,
            halo_size=module.halo_size,
            pad_before=module.split_pad_left,
            pad_after=module.split_pad_right,
            kernel_extent=module.kernel_extent,
            stride_extent=module.stride_extent,
        )
        module._trim_cache[local_extent] = trim
    shift, start, upper_bound = trim
    if shift:
        dim = module.split_tensor_dim
        x = _narrow_along_dim(x, dim, shift, x.shape[dim] - shift)
    defer_bias = (
        return_bias
        and module.input_memory_format != torch.contiguous_format
        and module.bias is not None
        and x.is_cuda
        and not torch.is_grad_enabled()
    )
    out = module._conv_forward(x, module.weight, None if defer_bias else module.bias)
    out = _trim_local_conv_output(out, module.halo_size, start, upper_bound, split_dim=module.split_dim)
    bias = module.bias.to(dtype=out.dtype) if defer_bias else None
    extent = out.shape[module.split_tensor_dim]
    if bias is not None and _local_valid_extent(extent) < extent:
        # Invalid positions must stay zero, including after the next epilogue.
        out = out + bias.view(1, -1, *((1,) * (out.ndim - 2)))
        bias = None
    # Fresh convolution outputs do not alias a caller or temporal cache.
    out = _zero_invalid_extent(out, split_dim=module.split_dim, inplace=not torch.is_grad_enabled())
    return (out, bias) if return_bias else out


def halo_exchange(
    x: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    halo_size: int,
    split_dim: str = "height",
    recv_top_buf: torch.Tensor | None = None,
    recv_bottom_buf: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if halo_size <= 0:
        return x, recv_top_buf, recv_bottom_buf

    rank, world_size = _rank_world(group)
    if world_size <= 1:
        return x, recv_top_buf, recv_bottom_buf

    dim = _spatial_dim(split_dim)
    top_row_ref = _narrow_along_dim(x, dim, 0, halo_size)
    bottom_row_ref = _narrow_along_dim(x, dim, x.shape[dim] - halo_size, halo_size)
    recv_top_buf = _ensure_recv_buf(recv_top_buf, top_row_ref)
    recv_bottom_buf = _ensure_recv_buf(recv_bottom_buf, bottom_row_ref)

    _halo_exchange_p2p(
        rank=rank,
        world_size=world_size,
        group=group,
        top_row_ref=top_row_ref,
        bottom_row_ref=bottom_row_ref,
        recv_top_buf=recv_top_buf,
        recv_bottom_buf=recv_bottom_buf,
    )

    return torch.cat([recv_top_buf, x, recv_bottom_buf], dim=dim), recv_top_buf, recv_bottom_buf


class WanDistZeroPad2d(nn.Module):
    """Apply ZeroPad2d only at global split-dimension boundaries."""

    def __init__(
        self,
        padding: tuple[int, int, int, int],
        group: dist.ProcessGroup,
        *,
        split_dim: str = "height",
        split_padding: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.padding = padding
        self.split_dim = split_dim
        default_split_padding = (padding[2], padding[3]) if split_dim == "height" else (padding[0], padding[1])
        self.split_padding = split_padding or default_split_padding
        self.group = group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rank, world_size = _rank_world(self.group)
        left, right, top, bottom = self.padding
        if world_size > 1:
            if self.split_dim == "height":
                top = top if rank == 0 else 0
                bottom = bottom if rank == world_size - 1 else 0
            else:
                left = left if rank == 0 else 0
                right = right if rank == world_size - 1 else 0
        return F.pad(x, (left, right, top, bottom))


class WanDistConv2d(nn.Conv2d):
    def __init__(
        self,
        source: nn.Conv2d,
        group: dist.ProcessGroup,
        split_dim: str = "height",
        split_padding: tuple[int, int] | None = None,
    ):
        super().__init__(
            source.in_channels,
            source.out_channels,
            source.kernel_size,
            stride=source.stride,
            padding=0,
            dilation=source.dilation,
            groups=source.groups,
            bias=source.bias is not None,
            padding_mode=source.padding_mode,
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        self.load_state_dict(source.state_dict())
        self.group = group
        self.split_dim = split_dim
        self.split_tensor_dim = _spatial_dim(split_dim)
        kernel_extent = self.kernel_size[-2] if split_dim == "height" else self.kernel_size[-1]
        self.halo_size = (kernel_extent - 1) // 2
        pad_h = source.padding[-2] if isinstance(source.padding, tuple) else source.padding
        pad_w = source.padding[-1] if isinstance(source.padding, tuple) else source.padding
        if split_dim == "height":
            if split_padding is None:
                split_padding = (pad_h, pad_h)
            self._non_split_padding = (pad_w, pad_w, 0, 0)
            self.kernel_extent = self.kernel_size[-2]
            self.stride_extent = self.stride[-2]
        else:
            if split_padding is None:
                split_padding = (pad_w, pad_w)
            self._non_split_padding = (0, 0, pad_h, pad_h)
            self.kernel_extent = self.kernel_size[-1]
            self.stride_extent = self.stride[-1]
        self.split_pad_left, self.split_pad_right = split_padding
        self._halo_recv_top_buf: torch.Tensor | None = None
        self._halo_recv_bottom_buf: torch.Tensor | None = None
        self._trim_cache: dict[int, tuple[int, int, int]] = {}
        self._backend = dist.get_backend(group) if dist.is_initialized() else None
        self.input_memory_format = torch.contiguous_format
        self._input_buf: torch.Tensor | None = None

    def forward(
        self, x: torch.Tensor, *, return_bias: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        if not torch.is_grad_enabled():
            x_padded, local_extent = self._assemble_input(x)
        else:
            x = F.pad(x, self._non_split_padding)
            x_padded, self._halo_recv_top_buf, self._halo_recv_bottom_buf = halo_exchange(
                x,
                group=self.group,
                halo_size=self.halo_size,
                split_dim=self.split_dim,
                recv_top_buf=self._halo_recv_top_buf,
                recv_bottom_buf=self._halo_recv_bottom_buf,
            )
            local_extent = x.shape[self.split_tensor_dim]
        return _spatial_conv_forward(self, x_padded, local_extent, return_bias=return_bias)

    def _assemble_input(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Copy the activation and received halos into one reusable padded input."""
        left, right, top, bottom = self._non_split_padding
        rank, world_size = _rank_world(self.group)
        halo = self.halo_size if world_size > 1 else 0
        dim = self.split_tensor_dim
        halo_h = halo if self.split_dim == "height" else 0
        halo_w = halo if self.split_dim == "width" else 0
        batch, channels, height, width = x.shape
        total_height = halo_h + top + height + bottom + halo_h
        total_width = halo_w + left + width + right + halo_w
        buf = _persistent_input_buffer(self, (batch, channels, total_height, total_width), x)
        buf[:, :, halo_h + top : halo_h + top + height, halo_w + left : halo_w + left + width].copy_(x)
        local_extent = buf.shape[dim] - 2 * halo
        if halo:
            _exchange_input_halos(self, buf, halo, rank, world_size)
        return buf, local_extent


class WanDistCausalConv3d(nn.Conv3d):
    def __init__(
        self,
        source: nn.Conv3d,
        group: dist.ProcessGroup,
        split_dim: str = "height",
    ):
        super().__init__(
            source.in_channels,
            source.out_channels,
            source.kernel_size,
            stride=source.stride,
            padding=0,
            dilation=source.dilation,
            groups=source.groups,
            bias=source.bias is not None,
            padding_mode=source.padding_mode,
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        self.load_state_dict(source.state_dict())
        self.group = group
        self.split_dim = split_dim
        self.split_tensor_dim = _spatial_dim(split_dim)
        source_padding = getattr(source, "_padding", None)
        if source_padding is None:
            p_t, p_h, p_w = source.padding
            source_padding = (p_w, p_w, p_h, p_h, 2 * p_t, 0)
        self._source_padding = tuple(source_padding)
        if split_dim == "height":
            self.split_pad_left = int(self._source_padding[2])
            self.split_pad_right = int(self._source_padding[3])
            self.kernel_extent = self.kernel_size[-2]
            self.stride_extent = self.stride[-2]
            padding = (
                self._source_padding[0],
                self._source_padding[1],
                0,
                0,
                self._source_padding[4],
                self._source_padding[5],
            )
        else:
            self.split_pad_left = int(self._source_padding[0])
            self.split_pad_right = int(self._source_padding[1])
            self.kernel_extent = self.kernel_size[-1]
            self.stride_extent = self.stride[-1]
            padding = (
                0,
                0,
                self._source_padding[2],
                self._source_padding[3],
                self._source_padding[4],
                self._source_padding[5],
            )
        self.halo_size = (self.kernel_extent - 1) // 2
        self._padding = padding if self.halo_size > 0 else self._source_padding
        self._halo_recv_top_buf: torch.Tensor | None = None
        self._halo_recv_bottom_buf: torch.Tensor | None = None
        self._trim_cache: dict[int, tuple[int, int, int]] = {}
        self._backend = dist.get_backend(group) if dist.is_initialized() else None
        self.input_memory_format = torch.contiguous_format

    def forward(
        self, x: torch.Tensor, cache_x: torch.Tensor | None = None, *, return_bias: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        x_padded, local_extent = self._assemble_input(x, cache_x)
        return _spatial_conv_forward(self, x_padded, local_extent, return_bias=return_bias)

    def _assemble_input(self, x: torch.Tensor, cache_x: torch.Tensor | None) -> tuple[torch.Tensor, int]:
        """Assemble history, padding, and halo slots in the convolution layout.

        Reuse the shared causal-input kernel when supported, otherwise copy
        into the same slots with PyTorch. Conv3d uses transient storage
        to avoid retaining one large activation per layer.
        """
        pad_w0, pad_w1, pad_h0, pad_h1, pad_t0, pad_t1 = self._padding
        cache_frames = 0
        if cache_x is not None and pad_t0 > 0:
            cache_x = cache_x.to(x.device)
            cache_frames = int(cache_x.shape[2])
            pad_t0 -= cache_frames
        assert pad_t0 >= 0, "Wan temporal cache exceeds the causal padding"
        rank, world_size = _rank_world(self.group)
        halo = self.halo_size if world_size > 1 else 0
        halo_h = halo if self.split_dim == "height" else 0
        halo_w = halo if self.split_dim == "width" else 0

        batch, channels, frames, height, width = x.shape
        top, left = pad_h0 + halo_h, pad_w0 + halo_w
        total_frames = pad_t0 + cache_frames + frames + pad_t1
        total_height = top + height + pad_h1 + halo_h
        total_width = left + width + pad_w1 + halo_w
        shape = (batch, channels, total_frames, total_height, total_width)
        buf = torch.empty(shape, dtype=x.dtype, device=x.device, memory_format=self.input_memory_format)
        fused = False
        if x.is_cuda and not torch.is_grad_enabled() and not torch.compiler.is_compiling():
            from .wan_vae_fastpath.triton_data_movement import cat_pad_5d

            # The kernel writes every element, including zero padding.
            padding = (left, pad_w1 + halo_w, top, pad_h1 + halo_h, pad_t0 + cache_frames, pad_t1)
            fused = cat_pad_5d(x, cache_x if cache_frames else None, padding, out=buf) is not None
        if not fused:
            # The CUDA assembly kernel is optional; these copies implement the
            # same layout for CPU, autograd, and unsupported kernel inputs.
            if pad_t0:
                buf[:, :, :pad_t0].zero_()
            if pad_t1:
                buf[:, :, total_frames - pad_t1 :].zero_()
            if pad_h0:
                buf[:, :, :, halo_h:top].zero_()
            if pad_h1:
                buf[:, :, :, total_height - halo_h - pad_h1 : total_height - halo_h].zero_()
            if pad_w0:
                buf[:, :, :, :, halo_w:left].zero_()
            if pad_w1:
                buf[:, :, :, :, total_width - halo_w - pad_w1 : total_width - halo_w].zero_()
            interior = buf[:, :, pad_t0 : pad_t0 + cache_frames + frames, top : top + height, left : left + width]
            if cache_frames:
                interior[:, :, :cache_frames].copy_(cache_x)
                interior[:, :, cache_frames:].copy_(x)
            else:
                interior.copy_(x)
        dim = self.split_tensor_dim
        local_extent = buf.shape[dim] - 2 * halo
        if halo:
            _exchange_input_halos(self, buf, halo, rank, world_size)
        return buf, local_extent


def _compute_conv_trim_params(
    *,
    local_extent: int,
    rank: int,
    world_size: int,
    halo_size: int,
    pad_before: int,
    pad_after: int,
    kernel_extent: int,
    stride_extent: int,
) -> tuple[int, int, int]:
    global_start = rank * local_extent
    shift = 0
    if halo_size > 0 and stride_extent > 1:
        shift = (global_start - halo_size + pad_before) % stride_extent
        if shift:
            global_start += shift

    global_extent = local_extent * world_size
    min_i = math.ceil(((-pad_before) - (global_start - halo_size)) / stride_extent)
    max_i = math.floor(
        ((global_extent - 1 + pad_after) - (kernel_extent - 1) - (global_start - halo_size)) / stride_extent
    )
    return shift, max(min_i, 0), max_i + 1


def _trim_local_conv_output(
    out: torch.Tensor,
    halo_size: int,
    start: int,
    upper_bound: int,
    *,
    split_dim: str,
) -> torch.Tensor:
    if halo_size <= 0:
        return out
    dim = _spatial_dim(split_dim)
    end = min(upper_bound, out.shape[dim])
    if start != 0 or end != out.shape[dim]:
        out = _narrow_along_dim(out, dim, start, end - start)
    return out


def _attention_local_queries(
    module: nn.Module, local: torch.Tensor, gathered: torch.Tensor, *, split_dim: str, rank: int
) -> torch.Tensor:
    """Use global keys/values, but compute attention outputs only for this shard.

    QKV projection keeps its full-frame shape. The shorter query sequence can
    change SDPA's reduction schedule, so this belongs to the tolerance-based
    channels-last level rather than the bit-exact level.
    """
    dim = _spatial_dim(split_dim)
    local_extent = local.shape[dim]
    valid = min(local_extent, gathered.shape[dim] - rank * local_extent)
    if valid <= 0:
        return torch.zeros_like(local)
    batch, channels, frames, height, width = gathered.shape
    x = gathered.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    qkv = module.to_qkv(module.norm(x))
    qkv = qkv.reshape(batch * frames, 1, channels * 3, -1).permute(0, 1, 3, 2).contiguous()
    query, key, value = qkv.chunk(3, dim=-1)
    query = query.reshape(batch * frames, 1, height, width, channels)
    query = query.narrow(2 if split_dim == "height" else 3, rank * local_extent, valid)
    query = query.reshape(batch * frames, 1, -1, channels).contiguous()
    out = F.scaled_dot_product_attention(query, key, value)
    local_height, local_width = (valid, width) if split_dim == "height" else (height, valid)
    out = out.squeeze(1).permute(0, 2, 1).reshape(batch * frames, channels, local_height, local_width)
    out = module.proj(out).view(batch, frames, channels, local_height, local_width).permute(0, 2, 1, 3, 4)
    out = out + local.narrow(dim, 0, valid)
    if valid < local_extent:
        padding = (0, local_extent - valid, 0, 0) if split_dim == "width" else (0, 0, 0, local_extent - valid)
        out = F.pad(out, padding)
    return out


def _patch_attention_block(module: nn.Module, group: dist.ProcessGroup, split_dim: str) -> None:
    if getattr(module, "_vllm_omni_spatial_shard_attention", False):
        return
    orig_forward = module.forward
    local_queries_supported = (
        isinstance(orig_forward, MethodType)
        and orig_forward.__func__ is type(module).forward
        and type(getattr(module, "to_qkv", None)) is nn.Conv2d
        and type(getattr(module, "proj", None)) is nn.Conv2d
        and all(
            conv.kernel_size == (1, 1) and conv.stride == (1, 1) and conv.padding == (0, 0)
            for conv in (module.to_qkv, module.proj)
        )
    )

    def _forward(self: nn.Module, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        from .wan_vae_fastpath.forwards import CFG_ATTR

        cfg = getattr(self, CFG_ATTR, None)
        _, world_size = _rank_world(group)
        if world_size <= 1:
            return orig_forward(x, *args, **kwargs)
        dim = _spatial_dim(split_dim)
        local_extent = x.shape[dim]
        gathered = all_gather_along_dim(x, group=group, dim=dim).contiguous()
        gathered_extent = gathered.shape[dim]
        full_extent = _current_full_extent(local_extent)
        if full_extent is not None:
            trim_extent = min(full_extent, gathered_extent)
            gathered = _narrow_along_dim(gathered, dim, 0, trim_extent).contiguous()
        else:
            trim_extent = gathered_extent
        if (
            local_queries_supported
            and cfg is not None
            and cfg.channels_last
            and x.is_cuda
            and not torch.is_grad_enabled()
            and not torch.compiler.is_compiling()
            and not args
            and not kwargs
        ):
            return _attention_local_queries(self, x, gathered, split_dim=split_dim, rank=_rank_world(group)[0])
        out = orig_forward(gathered, *args, **kwargs)
        if out.shape[dim] != trim_extent:
            raise RuntimeError(
                "Wan VAE attention changed the spatial extent during global gather: "
                f"rank={_rank_world(group)[0]}, split_dim={split_dim!r}, "
                f"local_extent={local_extent}, gathered_extent={gathered_extent}, "
                f"trimmed_extent={trim_extent}, output_extent={out.shape[dim]}, "
                f"context={_SPATIAL_SHARD_CONTEXT.get()!r}."
            )
        return reshard_from_trimmed_extent(out, local_extent=local_extent, split_dim=split_dim, group=group)

    module.forward = MethodType(_forward, module)
    module._vllm_omni_spatial_shard_attention = True  # type: ignore[attr-defined]


def _replace_child(
    parent: nn.Module,
    name: str,
    child: nn.Module,
    group: dist.ProcessGroup,
    split_dim: str,
) -> None:
    if child.__class__.__name__ == "WanCausalConv3d":
        setattr(
            parent,
            name,
            WanDistCausalConv3d(
                child,
                group,
                split_dim=split_dim,
            ),
        )
        return
    if isinstance(child, nn.ZeroPad2d):
        padding = cast(tuple[int, int, int, int], tuple(int(p) for p in child.padding))
        module_padding = padding
        if parent.__class__.__name__ == "Sequential":
            # Let the following WanDistConv2d account for global after-edge
            # padding; this module handles the non-split dimension and before edge.
            if split_dim == "height":
                module_padding = (padding[0], padding[1], padding[2], 0)
            else:
                module_padding = (padding[0], 0, padding[2], padding[3])
        setattr(
            parent,
            name,
            WanDistZeroPad2d(
                module_padding,
                group,
                split_dim=split_dim,
                split_padding=(padding[2], padding[3]) if split_dim == "height" else (padding[0], padding[1]),
            ),
        )
        return
    if isinstance(child, nn.Conv2d):
        split_padding = None
        if name == "1" and parent.__class__.__name__ == "Sequential":
            # WanResample downsample uses ZeroPad2d((0, 1, 0, 1)) before a
            # stride-2 conv with padding=0.  Only the last rank should see the
            # bottom/right global padding, which is approximated by split_padding.
            prev = getattr(parent, "0", None)
            if isinstance(prev, WanDistZeroPad2d):
                split_padding = prev.split_padding
            elif isinstance(prev, nn.ZeroPad2d):
                pad = prev.padding
                split_padding = (int(pad[2]), int(pad[3])) if split_dim == "height" else (int(pad[0]), int(pad[1]))
        setattr(
            parent,
            name,
            WanDistConv2d(
                child,
                group,
                split_dim=split_dim,
                split_padding=split_padding,
            ),
        )


def _patch_decoder_modules(
    module: nn.Module,
    group: dist.ProcessGroup,
    split_dim: str,
    inside_attention: bool = False,
) -> None:
    if module.__class__.__name__ == "WanAttentionBlock":
        _patch_attention_block(module, group, split_dim)
        inside_attention = True

    for name, child in list(module.named_children()):
        if child.__class__.__name__ == "WanCausalConv3d":
            _replace_child(module, name, child, group, split_dim)
            continue
        if isinstance(child, nn.Conv2d) and not inside_attention:
            _replace_child(module, name, child, group, split_dim)
            continue
        _patch_decoder_modules(child, group, split_dim, inside_attention=inside_attention)


def _decoder_upsample_count(decoder: nn.Module) -> int:
    count = 0
    for block in getattr(decoder, "up_blocks", []):
        if getattr(block, "upsampler", None) is not None or getattr(block, "upsamplers", None) is not None:
            count += 1
    return count


def install_wan_spatial_shard_decode(
    vae: Any,
    group: dist.ProcessGroup,
    split_dim: str = "height",
    *,
    dst: int | None = 0,
) -> None:
    """Patch ``vae.decoder`` once for spatially-sharded decode.

    This mutates the already-loaded decoder in place by swapping its spatial
    convolutions/padding for halo-exchanging variants and wrapping
    ``decoder.forward``. The patch is permanent for the lifetime of the VAE
    instance and is applied only once (subsequent calls are no-ops). A given
    instance is bound to a single ``group``, ``split_dim`` and ``dst``; switching any
    requires a fresh VAE instance and raises here otherwise.

    With ``dst=0`` (the default) only group-relative rank 0 assembles the
    final decoded frame, mirroring the distributed tiled-decode
    ``broadcast_result=False`` contract; the other ranks take part in the
    collectives but return an empty placeholder. ``dst=None`` lets every rank
    keep the assembled frame, for a caller whose every rank goes on to
    consume the decoded pixels; the gather is an all-gather either way, so
    this costs no extra communication.
    """
    _spatial_dim(split_dim)
    _, world_size = _rank_world(group)
    if dst is not None and (not isinstance(dst, int) or not 0 <= dst < world_size):
        raise ValueError(f"Wan spatial-shard dst must be None or an integer in [0, {world_size}), got {dst!r}.")
    if getattr(vae, "_vllm_omni_wan_spatial_shard_installed", False):
        installed_split_dim = getattr(vae, "_vllm_omni_wan_spatial_shard_split_dim", "height")
        if installed_split_dim != split_dim:
            raise ValueError(
                "Wan spatial-shard VAE decoder was already patched for "
                f"{installed_split_dim!r} split; create a fresh VAE instance to use {split_dim!r} split."
            )
        installed_dst = getattr(vae, "_vllm_omni_wan_spatial_shard_dst", 0)
        if installed_dst != dst:
            raise ValueError(
                "Wan spatial-shard VAE decoder was already patched to assemble on "
                f"{installed_dst!r}; create a fresh VAE instance to assemble on {dst!r}."
            )
        if getattr(vae, "_vllm_omni_wan_spatial_shard_group", None) is not group:
            raise ValueError(
                "Wan spatial-shard VAE decoder was already patched for a different process group; "
                "create a fresh VAE instance to change groups."
            )
        return
    decoder = getattr(vae, "decoder", None)
    if decoder is None:
        raise ValueError("Wan spatial-shard VAE decode requires a decoder module.")

    # Restore any direct/tile fast path before replacing convolution modules:
    # its undo callbacks refer to those original modules. Reinstall only after
    # the spatial wrapper exists, so it cannot be bypassed by a decoder forward.
    from .wan_vae_fastpath.install import (
        REPORT_ATTR,
        SPATIAL_LEVEL_ATTR,
        install_wan_vae_fastpath,
        uninstall_wan_vae_fastpath,
    )

    report = getattr(vae, REPORT_ATTR, None)
    fast_level = report.level if report is not None and report.installed else getattr(vae, SPATIAL_LEVEL_ATTR, None)
    if report is not None and report.installed:
        uninstall_wan_vae_fastpath(vae)

    _patch_decoder_modules(decoder, group, split_dim)
    upsample_count = _decoder_upsample_count(decoder)
    orig_forward = decoder.forward

    def _forward(
        self: nn.Module,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor] | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool = False,
    ) -> torch.Tensor:
        if feat_idx is None:
            feat_idx = [0]
        tensor_dim = _spatial_dim(split_dim)
        input_extent = x.shape[tensor_dim]
        x, expected_extent = split_for_parallel_decode(
            x,
            upsample_count=upsample_count,
            split_dim=split_dim,
            group=group,
        )
        rank, world_size = _rank_world(group)
        token = _SPATIAL_SHARD_CONTEXT.set(
            SpatialShardContext(
                input_extent=input_extent,
                local_input_extent=x.shape[tensor_dim],
                split_dim=split_dim,
                rank=rank,
                world_size=world_size,
            )
        )
        try:
            out = orig_forward(x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)
        finally:
            _SPATIAL_SHARD_CONTEXT.reset(token)
        return gather_and_trim_extent(out, expected_extent=expected_extent, split_dim=split_dim, group=group, dst=dst)

    decoder.forward = MethodType(_forward, decoder)
    vae._vllm_omni_wan_spatial_shard_installed = True
    vae._vllm_omni_wan_spatial_shard_split_dim = split_dim
    vae._vllm_omni_wan_spatial_shard_dst = dst
    vae._vllm_omni_wan_spatial_shard_group = group
    if fast_level is not None:
        install_wan_vae_fastpath(vae, level=fast_level)
    logger.info("Installed Wan VAE %s-sharded decode.", split_dim)


def spatial_shard_decode(
    vae: Any,
    z: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    return_dict: bool = True,
    split_dim: str = "height",
    on_chunk: DecodedChunkConsumer | None = None,
    temporal_chunk_size: int | None = None,
) -> DecoderOutput | tuple[torch.Tensor] | None:
    """Decode spatial shards, optionally batching subsequent latent frames.

    The first latent frame remains separate to initialize the temporal
    upsamplers. Larger chunks amortize halo exchanges and kernel launches but
    increase memory and may change convolution rounding, even with lossless
    local kernels. None selects min(8, group size) for an installed channels-last
    path and 1 otherwise. All ranks must use the same setting.
    """
    if temporal_chunk_size is not None and (
        isinstance(temporal_chunk_size, bool) or not isinstance(temporal_chunk_size, int) or temporal_chunk_size < 1
    ):
        raise ValueError("temporal_chunk_size must be a positive integer")
    install_wan_spatial_shard_decode(vae, group, split_dim=split_dim)

    if z.shape[2] == 0:
        raise ValueError("Wan spatial-shard VAE decode expects at least one latent frame.")

    # Non-rank-0 ranks must still run the decoder every chunk to stay in lockstep with
    # the halo/all-gather collectives; they just skip keeping/assembling the output.
    rank, world_size = _rank_world(group)
    if temporal_chunk_size is None:
        from .wan_vae_fastpath import REPORT_ATTR

        report = getattr(vae, REPORT_ATTR, None)
        channels_last = report is not None and report.installed and report.level == "channels_last"
        temporal_chunk_size = min(8, world_size) if channels_last else 1
    produce_output = world_size <= 1 or rank == 0

    vae.clear_cache()
    callback_error: BaseException | None = None
    try:
        context = vae._execution_context() if hasattr(vae, "_execution_context") else nullcontext()
        with context:
            x = vae.post_quant_conv(z)
            preallocate = on_chunk is None and not torch.is_grad_enabled()
            decoded_chunks = []
            out = None
            offset = 0
            frames_per_chunk = 2 ** sum(bool(flag) for flag in vae.decoder.temperal_upsample) if preallocate else 0
            for i in chain((0,), range(1, z.shape[2], temporal_chunk_size)):
                vae._conv_idx = [0]
                chunk = vae.decoder(
                    x[:, :, i : i + (1 if i == 0 else temporal_chunk_size), :, :],
                    feat_cache=vae._feat_map,
                    feat_idx=vae._conv_idx,
                    first_chunk=(i == 0),
                )
                if produce_output:
                    if on_chunk is not None and callback_error is None:
                        try:
                            if vae.config.patch_size is not None:
                                chunk = unpatchify(chunk, patch_size=vae.config.patch_size)
                            on_chunk(torch.clamp(chunk, min=-1.0, max=1.0))
                        except BaseException as exc:
                            # Keep all ranks in the temporal collective loop;
                            # surface the callback failure only after decode.
                            callback_error = exc
                    elif on_chunk is None:
                        if preallocate:
                            if vae.config.patch_size is not None:
                                chunk = unpatchify(chunk, patch_size=vae.config.patch_size)
                            if out is None:
                                total_frames = chunk.shape[2] + (z.shape[2] - 1) * frames_per_chunk
                                out = chunk.new_empty((*chunk.shape[:2], total_frames, *chunk.shape[3:]))
                            end = offset + chunk.shape[2]
                            torch.clamp(chunk, min=-1.0, max=1.0, out=out[:, :, offset:end])
                            offset = end
                        else:
                            decoded_chunks.append(chunk)
                del chunk

            if produce_output and on_chunk is None and preallocate:
                if out is None or offset != out.shape[2]:
                    raise RuntimeError("Wan spatial decoder produced an unexpected number of frames")
            if produce_output and on_chunk is None and not preallocate:
                out = torch.cat(decoded_chunks, dim=2)
                if vae.config.patch_size is not None:
                    out = unpatchify(out, patch_size=vae.config.patch_size)
                out = torch.clamp(out, min=-1.0, max=1.0)
    finally:
        vae.clear_cache()

    if callback_error is not None:
        raise callback_error
    if on_chunk is not None:
        return None
    if not produce_output:
        out = z.new_zeros(0)
    if not return_dict:
        return (out,)
    return DecoderOutput(sample=out)
