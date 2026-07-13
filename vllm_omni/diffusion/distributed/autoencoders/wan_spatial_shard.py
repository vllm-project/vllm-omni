# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
from types import MethodType
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify
from diffusers.models.autoencoders.vae import DecoderOutput
from vllm.logger import init_logger

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


def _zero_invalid_extent(x: torch.Tensor, *, split_dim: str) -> torch.Tensor:
    dim = _spatial_dim(split_dim)
    dim_size = x.shape[dim]
    valid_extent = _local_valid_extent(dim_size)
    if valid_extent >= dim_size:
        return x
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
    valid_extent = _local_valid_extent(local_extent)
    start = rank * local_extent
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


def _ensure_recv_buf(recv_buf: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    memory_format = _halo_memory_format(reference)
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
    p2p_ops = []
    if rank > 0:
        prev_rank = _global_rank(group, rank - 1)
        top_row = top_row_ref.contiguous(memory_format=_halo_memory_format(top_row_ref))
        p2p_ops.append(dist.P2POp(dist.irecv, recv_top_buf, prev_rank, group))
        p2p_ops.append(dist.P2POp(dist.isend, top_row, prev_rank, group))
    else:
        recv_top_buf.zero_()

    if rank < world_size - 1:
        next_rank = _global_rank(group, rank + 1)
        bottom_row = bottom_row_ref.contiguous(memory_format=_halo_memory_format(bottom_row_ref))
        p2p_ops.append(dist.P2POp(dist.isend, bottom_row, next_rank, group))
        p2p_ops.append(dist.P2POp(dist.irecv, recv_bottom_buf, next_rank, group))
    else:
        recv_bottom_buf.zero_()

    if p2p_ops:
        reqs = dist.batch_isend_irecv(p2p_ops)
        for req in reqs:
            req.wait()


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self._non_split_padding)
        x_padded, self._halo_recv_top_buf, self._halo_recv_bottom_buf = halo_exchange(
            x,
            group=self.group,
            halo_size=self.halo_size,
            split_dim=self.split_dim,
            recv_top_buf=self._halo_recv_top_buf,
            recv_bottom_buf=self._halo_recv_bottom_buf,
        )
        shift, start, upper_bound = self._get_trim_params(x.shape[self.split_tensor_dim])
        if shift:
            x_padded = _narrow_along_dim(
                x_padded,
                self.split_tensor_dim,
                shift,
                x_padded.shape[self.split_tensor_dim] - shift,
            )
        out = super().forward(x_padded)
        out = _trim_local_conv_output(out, self.halo_size, start, upper_bound, split_dim=self.split_dim)
        return _zero_invalid_extent(out, split_dim=self.split_dim)

    def _get_trim_params(self, local_extent: int) -> tuple[int, int, int]:
        trim_params = self._trim_cache.get(local_extent)
        if trim_params is None:
            rank, world_size = _rank_world(self.group)
            trim_params = _compute_conv_trim_params(
                local_extent=local_extent,
                rank=rank,
                world_size=world_size,
                halo_size=self.halo_size,
                pad_before=self.split_pad_left,
                pad_after=self.split_pad_right,
                kernel_extent=self.kernel_extent,
                stride_extent=self.stride_extent,
            )
            self._trim_cache[local_extent] = trim_params
        return trim_params


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

    def forward(self, x: torch.Tensor, cache_x: torch.Tensor | None = None) -> torch.Tensor:
        padding = list(self._padding)
        if cache_x is not None and padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]

        x = F.pad(x, padding)
        x_padded, self._halo_recv_top_buf, self._halo_recv_bottom_buf = halo_exchange(
            x,
            group=self.group,
            halo_size=self.halo_size,
            split_dim=self.split_dim,
            recv_top_buf=self._halo_recv_top_buf,
            recv_bottom_buf=self._halo_recv_bottom_buf,
        )
        shift, start, upper_bound = self._get_trim_params(x.shape[self.split_tensor_dim])
        if shift:
            x_padded = _narrow_along_dim(
                x_padded,
                self.split_tensor_dim,
                shift,
                x_padded.shape[self.split_tensor_dim] - shift,
            )
        out = super().forward(x_padded)
        out = _trim_local_conv_output(out, self.halo_size, start, upper_bound, split_dim=self.split_dim)
        return _zero_invalid_extent(out, split_dim=self.split_dim)

    def _get_trim_params(self, local_extent: int) -> tuple[int, int, int]:
        trim_params = self._trim_cache.get(local_extent)
        if trim_params is None:
            rank, world_size = _rank_world(self.group)
            trim_params = _compute_conv_trim_params(
                local_extent=local_extent,
                rank=rank,
                world_size=world_size,
                halo_size=self.halo_size,
                pad_before=self.split_pad_left,
                pad_after=self.split_pad_right,
                kernel_extent=self.kernel_extent,
                stride_extent=self.stride_extent,
            )
            self._trim_cache[local_extent] = trim_params
        return trim_params


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


def _patch_attention_block(module: nn.Module, group: dist.ProcessGroup, split_dim: str) -> None:
    if getattr(module, "_vllm_omni_spatial_shard_attention", False):
        return
    orig_forward = module.forward

    def _forward(self: nn.Module, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        _, world_size = _rank_world(group)
        if world_size <= 1:
            return orig_forward(x, *args, **kwargs)
        dim = _spatial_dim(split_dim)
        local_extent = x.shape[dim]
        gathered = all_gather_along_dim(x, group=group, dim=dim).contiguous()
        full_extent = _current_full_extent(local_extent)
        if full_extent is not None:
            gathered = _narrow_along_dim(gathered, dim, 0, full_extent).contiguous()
        out = orig_forward(gathered, *args, **kwargs)
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
        padding = tuple(int(p) for p in child.padding)
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


def install_wan_spatial_shard_decode(vae: Any, group: dist.ProcessGroup, split_dim: str = "height") -> None:
    """Patch ``vae.decoder`` once for spatially-sharded decode.

    This mutates the already-loaded decoder in place by swapping its spatial
    convolutions/padding for halo-exchanging variants and wrapping
    ``decoder.forward``. The patch is permanent for the lifetime of the VAE
    instance and is applied only once (subsequent calls are no-ops). A given
    instance is bound to a single ``split_dim``; switching between
    ``"height"`` and ``"width"`` requires a fresh VAE instance and raises here
    otherwise.

    Only group-relative rank 0 assembles the final decoded frame, mirroring the
    distributed tiled-decode ``broadcast_result=False`` contract; the other ranks
    take part in the collectives but return an empty placeholder.
    """
    _spatial_dim(split_dim)
    if getattr(vae, "_vllm_omni_wan_spatial_shard_installed", False):
        installed_split_dim = getattr(vae, "_vllm_omni_wan_spatial_shard_split_dim", "height")
        if installed_split_dim != split_dim:
            raise ValueError(
                "Wan spatial-shard VAE decoder was already patched for "
                f"{installed_split_dim!r} split; create a fresh VAE instance to use {split_dim!r} split."
            )
        return
    decoder = getattr(vae, "decoder", None)
    if decoder is None:
        raise ValueError("Wan spatial-shard VAE decode requires a decoder module.")

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
        return gather_and_trim_extent(out, expected_extent=expected_extent, split_dim=split_dim, group=group, dst=0)

    decoder.forward = MethodType(_forward, decoder)
    vae._vllm_omni_wan_spatial_shard_installed = True
    vae._vllm_omni_wan_spatial_shard_split_dim = split_dim
    logger.info("Installed Wan VAE %s-sharded decode.", split_dim)


def spatial_shard_decode(
    vae: Any,
    z: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    return_dict: bool = True,
    split_dim: str = "height",
) -> DecoderOutput | tuple[torch.Tensor]:
    install_wan_spatial_shard_decode(vae, group, split_dim=split_dim)

    if z.shape[2] == 0:
        raise ValueError("Wan spatial-shard VAE decode expects at least one latent frame.")

    # Non-rank-0 ranks must still run the decoder every chunk to stay in lockstep with
    # the halo/all-gather collectives; they just skip keeping/assembling the output.
    rank, world_size = _rank_world(group)
    produce_output = world_size <= 1 or rank == 0

    vae.clear_cache()
    try:
        context = vae._execution_context() if hasattr(vae, "_execution_context") else nullcontext()
        with context:
            x = vae.post_quant_conv(z)
            decoded_chunks = []
            for i in range(z.shape[2]):
                vae._conv_idx = [0]
                chunk = vae.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=vae._feat_map,
                    feat_idx=vae._conv_idx,
                    first_chunk=(i == 0),
                )
                if produce_output:
                    decoded_chunks.append(chunk)

            if produce_output:
                out = torch.cat(decoded_chunks, dim=2)
                if vae.config.patch_size is not None:
                    out = unpatchify(out, patch_size=vae.config.patch_size)
                out = torch.clamp(out, min=-1.0, max=1.0)
            else:
                out = z.new_zeros(0)
    finally:
        vae.clear_cache()

    if not return_dict:
        return (out,)
    return DecoderOutput(sample=out)
