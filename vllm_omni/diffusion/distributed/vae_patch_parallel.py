# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Distributed VAE patch/tile parallelism utilities."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
from vllm.logger import init_logger

logger = init_logger(__name__)


def _get_vae_spatial_scale_factor(vae: Any) -> int:
    try:
        block_out_channels = getattr(getattr(vae, "config", None), "block_out_channels", None)
        if block_out_channels:
            return 2 ** (len(block_out_channels) - 1)
    except Exception:
        pass
    return 8


def _factor_pp_grid(pp_size: int) -> tuple[int, int]:
    """Pick a (rows, cols) grid whose product equals `pp_size`."""
    if pp_size <= 1:
        return (1, 1)
    root = int(math.sqrt(pp_size))
    for rows in range(root, 0, -1):
        if pp_size % rows == 0:
            return (rows, pp_size // rows)
    return (1, pp_size)


def _get_world_rank_pp_size(
    group: dist.ProcessGroup,
    vae_patch_parallel_size: int,
) -> tuple[int, int, int]:
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    pp_size = min(int(vae_patch_parallel_size), int(world_size))
    return world_size, rank, pp_size


def _get_vae_out_channels(vae: Any) -> int:
    return int(getattr(getattr(vae, "config", None), "out_channels", 3))


def _get_vae_tile_params(vae: Any) -> tuple[int, float] | None:
    tile_latent_min_size = getattr(vae, "tile_latent_min_size", None)
    tile_overlap_factor = getattr(vae, "tile_overlap_factor", None)
    if tile_latent_min_size is None or tile_overlap_factor is None:
        return None
    return int(tile_latent_min_size), float(tile_overlap_factor)


def _get_vae_tiling_params(vae: Any) -> tuple[int, float, int] | None:
    tile_sample_min_size = getattr(vae, "tile_sample_min_size", None)
    tile_params = _get_vae_tile_params(vae)
    if tile_params is None or tile_sample_min_size is None:
        return None
    tile_latent_min_size, tile_overlap_factor = tile_params
    return tile_latent_min_size, tile_overlap_factor, int(tile_sample_min_size)


def _distributed_tiled_decode(
    *,
    vae: Any,
    orig_decode: Callable[..., Any],
    z: torch.Tensor,
    group: dist.ProcessGroup,
    vae_patch_parallel_size: int,
) -> torch.Tensor:
    """Distributed version of diffusers AutoencoderKL.tiled_decode (decode only).

    Each rank decodes a subset of tiles; rank0 gathers all tiles and performs the
    original blend + stitch logic. Non-rank0 ranks return an empty tensor; callers
    can broadcast the stitched result if needed.
    """
    world_size, rank, pp_size = _get_world_rank_pp_size(group, vae_patch_parallel_size)
    if pp_size <= 1:
        return orig_decode(z, return_dict=False)[0]

    tiling_params = _get_vae_tiling_params(vae)
    if tiling_params is None:
        return orig_decode(z, return_dict=False)[0]
    tile_latent_min_size, tile_overlap_factor, tile_sample_min_size = tiling_params

    overlap_size = int(tile_latent_min_size * (1 - tile_overlap_factor))
    if overlap_size <= 0:
        return orig_decode(z, return_dict=False)[0]

    h_starts = list(range(0, z.shape[2], overlap_size))
    w_starts = list(range(0, z.shape[3], overlap_size))
    num_rows = len(h_starts)
    num_cols = len(w_starts)
    num_tiles = num_rows * num_cols

    if num_tiles < 2:
        return orig_decode(z, return_dict=False)[0]

    blend_extent = int(tile_sample_min_size * tile_overlap_factor)
    row_limit = int(tile_sample_min_size - blend_extent)

    # Decide which ranks actively decode tiles.
    active = rank < pp_size

    local_tiles: list[torch.Tensor] = []
    local_meta: list[tuple[int, int, int]] = []

    tile_id = 0
    for i in h_starts:
        for j in w_starts:
            # Offset assignment by 1 so rank0 avoids decoding the largest (tile_id=0) tile.
            tile_rank = (tile_id + 1) % pp_size
            if active and (tile_rank == rank):
                tile = z[:, :, i : i + tile_latent_min_size, j : j + tile_latent_min_size]
                if getattr(getattr(vae, "config", None), "use_post_quant_conv", False):
                    tile = vae.post_quant_conv(tile)
                decoded = vae.decoder(tile)
                local_tiles.append(decoded)
                local_meta.append((tile_id, int(decoded.shape[-2]), int(decoded.shape[-1])))
            tile_id += 1

    # Gather per-rank tile counts.
    count_tensor = torch.tensor([len(local_tiles)], device=z.device, dtype=torch.int64)
    if rank == 0:
        count_gather = [torch.empty_like(count_tensor) for _ in range(world_size)]
    else:
        count_gather = None
    dist.gather(count_tensor, gather_list=count_gather, dst=0, group=group)
    max_count = 0
    if rank == 0:
        counts = [int(t.item()) for t in count_gather]  # type: ignore[arg-type]
        max_count = max(counts) if counts else 0
    max_count_tensor = torch.tensor([max_count], device=z.device, dtype=torch.int64)
    dist.broadcast(max_count_tensor, src=0, group=group)
    max_count = int(max_count_tensor.item())

    out_channels = _get_vae_out_channels(vae)

    # Prepare padded metadata + tiles for gather.
    meta_tensor = torch.full((max_count, 3), -1, device=z.device, dtype=torch.int64)
    tile_tensor = torch.zeros(
        (max_count, z.shape[0], out_channels, tile_sample_min_size, tile_sample_min_size),
        device=z.device,
        dtype=z.dtype,
    )
    for idx, (tile_id, h, w) in enumerate(local_meta):
        meta_tensor[idx, 0] = tile_id
        meta_tensor[idx, 1] = h
        meta_tensor[idx, 2] = w
        tile_tensor[idx, :, :, :h, :w] = local_tiles[idx]

    if rank == 0:
        meta_gather = [torch.empty_like(meta_tensor) for _ in range(world_size)]
        tile_gather = [torch.empty_like(tile_tensor) for _ in range(world_size)]
    else:
        meta_gather = None
        tile_gather = None

    dist.gather(meta_tensor, gather_list=meta_gather, dst=0, group=group)
    dist.gather(tile_tensor, gather_list=tile_gather, dst=0, group=group)

    if rank != 0:
        return torch.empty(0, device=z.device, dtype=z.dtype)

    # Reconstruct the full tile grid on rank0.
    tile_map: dict[int, torch.Tensor] = {}
    for src_rank in range(world_size):
        meta_src = meta_gather[src_rank]  # type: ignore[index]
        tiles_src = tile_gather[src_rank]  # type: ignore[index]
        for idx in range(max_count):
            tid = int(meta_src[idx, 0].item())
            if tid < 0:
                continue
            h = int(meta_src[idx, 1].item())
            w = int(meta_src[idx, 2].item())
            tile_map[tid] = tiles_src[idx, :, :, :h, :w]

    rows: list[list[torch.Tensor]] = []
    for r in range(num_rows):
        row: list[torch.Tensor] = []
        for c in range(num_cols):
            tid = r * num_cols + c
            row.append(tile_map[tid])
        rows.append(row)

    result_rows: list[torch.Tensor] = []
    for i, row in enumerate(rows):
        result_row: list[torch.Tensor] = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = vae.blend_v(rows[i - 1][j], tile, blend_extent)
            if j > 0:
                tile = vae.blend_h(row[j - 1], tile, blend_extent)
            result_row.append(tile[:, :, :row_limit, :row_limit])
        result_rows.append(torch.cat(result_row, dim=3))

    return torch.cat(result_rows, dim=2)


def _distributed_patch_decode(
    *,
    vae: Any,
    orig_decode: Callable[..., Any],
    z: torch.Tensor,
    group: dist.ProcessGroup,
    vae_patch_parallel_size: int,
    vae_scale_factor: int,
) -> torch.Tensor:
    """Decode one spatial block per rank, then stitch RGB blocks on rank0.

    Intended for sizes where diffusers tiling would not kick in, so we can still
    reduce the per-rank VAE decode activation peak. Each rank decodes a core
    block with a small latent-space halo, then crops to the core and gathers the
    RGB blocks to rank0 for final stitching.
    """
    world_size, rank, pp_size = _get_world_rank_pp_size(group, vae_patch_parallel_size)
    if pp_size <= 1:
        return orig_decode(z, return_dict=False)[0]

    tile_params = _get_vae_tile_params(vae)
    if tile_params is None:
        return orig_decode(z, return_dict=False)[0]
    tile_latent_min_size, tile_overlap_factor = tile_params

    overlap_latent = int(tile_latent_min_size * float(tile_overlap_factor))
    halo_base = max(0, overlap_latent // 2)

    # Only ranks < pp_size participate in decoding. Others send empty tensors.
    active = rank < pp_size

    bsz, _, latent_h, latent_w = z.shape
    scale = int(vae_scale_factor)
    out_h = latent_h * scale
    out_w = latent_w * scale

    out_channels = _get_vae_out_channels(vae)

    local_core = torch.empty(0, device=z.device, dtype=z.dtype)
    local_h = 0
    local_w = 0

    grid_rows, grid_cols = _factor_pp_grid(pp_size)

    if active:
        patch_idx = rank
        patch_row = patch_idx // grid_cols
        patch_col = patch_idx % grid_cols

        h0 = (patch_row * latent_h) // grid_rows
        h1 = ((patch_row + 1) * latent_h) // grid_rows
        w0 = (patch_col * latent_w) // grid_cols
        w1 = ((patch_col + 1) * latent_w) // grid_cols

        core_h = max(0, h1 - h0)
        core_w = max(0, w1 - w0)
        if core_h == 0 or core_w == 0:
            local_core = torch.empty(0, device=z.device, dtype=z.dtype)
        else:
            halo = max(halo_base, min(core_h, core_w) // 2)
            ph0 = max(0, h0 - halo)
            ph1 = min(latent_h, h1 + halo)
            pw0 = max(0, w0 - halo)
            pw1 = min(latent_w, w1 + halo)

            tile = z[:, :, ph0:ph1, pw0:pw1]
            if getattr(getattr(vae, "config", None), "use_post_quant_conv", False):
                tile = vae.post_quant_conv(tile)
            decoded = vae.decoder(tile)

            ch0 = (h0 - ph0) * scale
            cw0 = (w0 - pw0) * scale
            ch1 = ch0 + core_h * scale
            cw1 = cw0 + core_w * scale
            local_core = decoded[:, :, ch0:ch1, cw0:cw1]

        local_h = int(local_core.shape[-2]) if local_core.numel() else 0
        local_w = int(local_core.shape[-1]) if local_core.numel() else 0

    # Gather block shapes.
    shape_tensor = torch.tensor([local_h, local_w], device=z.device, dtype=torch.int64)
    if rank == 0:
        shape_gather = [torch.empty_like(shape_tensor) for _ in range(world_size)]
    else:
        shape_gather = None
    dist.gather(shape_tensor, gather_list=shape_gather, dst=0, group=group)

    max_h = 0
    max_w = 0
    if rank == 0:
        shapes = [tuple(int(x.item()) for x in t) for t in shape_gather]  # type: ignore[arg-type]
        max_h = max((h for h, _ in shapes), default=0)
        max_w = max((w for _, w in shapes), default=0)

    max_hw_tensor = torch.tensor([max_h, max_w], device=z.device, dtype=torch.int64)
    dist.broadcast(max_hw_tensor, src=0, group=group)
    max_h = int(max_hw_tensor[0].item())
    max_w = int(max_hw_tensor[1].item())

    # Pad local block for gather.
    if max_h == 0 or max_w == 0:
        padded = torch.empty(0, device=z.device, dtype=z.dtype)
    else:
        padded = torch.zeros((bsz, out_channels, max_h, max_w), device=z.device, dtype=z.dtype)
        if local_h and local_w:
            padded[:, :, :local_h, :local_w] = local_core

    if rank == 0:
        block_gather = [torch.empty_like(padded) for _ in range(world_size)]
    else:
        block_gather = None
    dist.gather(padded, gather_list=block_gather, dst=0, group=group)

    if rank != 0:
        return torch.empty(0, device=z.device, dtype=z.dtype)

    # Stitch on rank0.
    out = torch.empty((bsz, out_channels, out_h, out_w), device=z.device, dtype=z.dtype)

    for patch_idx in range(pp_size):
        src_rank = patch_idx
        patch_row = patch_idx // grid_cols
        patch_col = patch_idx % grid_cols

        h0 = (patch_row * latent_h) // grid_rows
        h1 = ((patch_row + 1) * latent_h) // grid_rows
        w0 = (patch_col * latent_w) // grid_cols
        w1 = ((patch_col + 1) * latent_w) // grid_cols

        ph = (h1 - h0) * scale
        pw = (w1 - w0) * scale
        if ph <= 0 or pw <= 0:
            continue

        tile = block_gather[src_rank]  # type: ignore[index]
        out[:, :, h0 * scale : h1 * scale, w0 * scale : w1 * scale] = tile[:, :, :ph, :pw]

    return out


class VaePatchParallelism:
    """Patch/tile-parallel VAE decode wrapper.

    This is meant to wrap `vae.decode` as an instance-level override
    so pipelines don't need model-specific code paths.
    """

    def __init__(
        self,
        vae: Any,
        *,
        vae_patch_parallel_size: int,
        group_getter: Callable[[], dist.ProcessGroup],
    ) -> None:
        self._vae = vae
        self._vae_patch_parallel_size = int(vae_patch_parallel_size)
        self._group_getter = group_getter

        self._vae_scale_factor = _get_vae_spatial_scale_factor(vae)
        self._orig_decode = vae.decode

    def decode(self, z: torch.Tensor, return_dict: bool = True, *args: Any, **kwargs: Any):
        # Keep the original path for unsupported VAE types / shapes.
        if z.ndim != 4:
            return self._orig_decode(z, return_dict=return_dict, *args, **kwargs)

        if self._vae_patch_parallel_size <= 1 or not dist.is_initialized():
            return self._orig_decode(z, return_dict=return_dict, *args, **kwargs)

        if not getattr(self._vae, "use_tiling", False):
            return self._orig_decode(z, return_dict=return_dict, *args, **kwargs)

        try:
            group = self._group_getter()
        except Exception:
            return self._orig_decode(z, return_dict=return_dict, *args, **kwargs)

        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        pp_size = min(int(self._vae_patch_parallel_size), int(world_size))
        if pp_size <= 1:
            return self._orig_decode(z, return_dict=return_dict, *args, **kwargs)

        # Match diffusers' condition for when VAE tiling would be used.
        tile_latent_min_size = getattr(self._vae, "tile_latent_min_size", None)
        if tile_latent_min_size is None:
            decoded = _distributed_tiled_decode(
                vae=self._vae,
                orig_decode=self._orig_decode,
                z=z,
                group=group,
                vae_patch_parallel_size=pp_size,
            )
        else:
            should_tile = (z.shape[-1] > tile_latent_min_size) or (z.shape[-2] > tile_latent_min_size)
            if should_tile:
                decoded = _distributed_tiled_decode(
                    vae=self._vae,
                    orig_decode=self._orig_decode,
                    z=z,
                    group=group,
                    vae_patch_parallel_size=pp_size,
                )
            else:
                decoded = _distributed_patch_decode(
                    vae=self._vae,
                    orig_decode=self._orig_decode,
                    z=z,
                    group=group,
                    vae_patch_parallel_size=pp_size,
                    vae_scale_factor=self._vae_scale_factor,
                )

        if rank == 0 and decoded.numel() == 0:
            logger.warning("VAE patch parallel decode produced empty output on rank0; falling back to vae.decode.")
            decoded = self._orig_decode(z, return_dict=False, *args, **kwargs)[0]
        if rank == 0 and decoded.dtype != z.dtype:
            decoded = decoded.to(dtype=z.dtype)
        if rank == 0 and not decoded.is_contiguous():
            decoded = decoded.contiguous()

        shape_tensor = torch.empty((4,), device=z.device, dtype=torch.int64)
        if rank == 0:
            shape_tensor.copy_(torch.tensor(tuple(decoded.shape), device=z.device, dtype=torch.int64))
        dist.broadcast(shape_tensor, src=0, group=group)

        if rank != 0:
            decoded = torch.empty(tuple(int(x) for x in shape_tensor.tolist()), device=z.device, dtype=z.dtype)
        dist.broadcast(decoded, src=0, group=group)

        if not return_dict:
            return (decoded,)

        from diffusers.models.autoencoders.vae import DecoderOutput

        return DecoderOutput(sample=decoded)


def maybe_wrap_vae_decode_with_patch_parallelism(
    pipeline: Any,
    *,
    vae_patch_parallel_size: int,
    group_getter: Callable[[], dist.ProcessGroup],
) -> None:
    """Wrap a diffusers-style pipeline's `vae.decode` with patch/tile parallel decode."""
    if vae_patch_parallel_size <= 1:
        return

    vae = getattr(pipeline, "vae", None)
    if vae is None or not hasattr(vae, "decode"):
        return
    try:
        from diffusers.models.autoencoders import AutoencoderKL
    except Exception:
        return
    if not isinstance(vae, AutoencoderKL):
        return

    if getattr(vae, "_vllm_vae_patch_parallel_installed", False):
        return

    wrapper = VaePatchParallelism(
        vae,
        vae_patch_parallel_size=vae_patch_parallel_size,
        group_getter=group_getter,
    )

    vae._vllm_vae_patch_parallel_installed = True  # type: ignore[attr-defined]
    vae._vllm_vae_patch_parallel_original_decode = vae.decode  # type: ignore[attr-defined]
    vae.decode = wrapper.decode  # type: ignore[assignment]
