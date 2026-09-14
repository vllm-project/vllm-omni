# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from __future__ import annotations

import functools
import math
import os
from contextlib import contextmanager
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.fastvideo_vsa import (
    FastVideoVSABackend,
    FastVideoVSAImpl,
    _construct_variable_block_sizes,
    _get_gate_compress,
    _get_non_pad_index,
    _get_tile_partition_indices,
)
from vllm_omni.diffusion.models.minimax_h3.ops.attention.layout import (
    H3_VSA_FUSED_TILE_PACK_ENV,
    H3_VSA_FUSED_UNTILE_ENV,
    build_h3_aligned_untile_source_rows,
    build_h3_tiled_source_rows,
    h3_vsa_fused_tile_pack_enabled,
    h3_vsa_fused_untile_enabled,
    h3_vsa_tile_pack,
    h3_vsa_tile_pack_cuda_supported,
    h3_vsa_tile_untile,
    h3_vsa_tile_untile_cuda_supported,
    h3_vsa_tile_untile_out,
)
from vllm_omni.diffusion.models.minimax_h3.ops.attention.o_bundle import (
    H3_VSA_O_BUNDLE_ACTIVE_KEY,
    H3_VSA_O_BUNDLE_ENV,
    H3_VSA_O_BUNDLE_STATE_KEY,
    h3_vsa_o_bundle_enabled,
)
from vllm_omni.diffusion.models.minimax_h3.ops.attention.owner_route import (
    H3VSAOwnerRoutePlan,
    build_h3_vsa_owner_route_plan,
    h3_vsa_owner_route_plan_is_trusted,
)
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)
H3_VSA_DIRECT_Q2K_ENV = "VLLM_OMNI_FASTVIDEO_VSA_DIRECT_Q2K"
H3_VSA_SKIP_SOFTMAX_THRESHOLD_ENV = "VLLM_OMNI_FASTVIDEO_VSA_SKIP_SOFTMAX_THRESHOLD_SCALE_FACTOR"
H3_VSA_NVTX_ENV = "VLLM_OMNI_MINIMAX_H3_VSA_NVTX"
H3_VSA_NVTX_DOMAIN = "vllm_omni.minimax_h3.vsa"
H3_VSA_EFFECTIVE_TILE_SHAPE = (4, 4, 4)
H3_VSA_EFFECTIVE_TILE_SIZE = math.prod(H3_VSA_EFFECTIVE_TILE_SHAPE)


def h3_vsa_direct_q2k_enabled() -> bool:
    return os.environ.get(H3_VSA_DIRECT_Q2K_ENV, "0") == "1"


@contextmanager
def _h3_vsa_nvtx_stage(name: str):
    """Emit nested VSA phase ranges only for an explicit diagnostic run."""
    if os.environ.get(H3_VSA_NVTX_ENV, "0") != "1":
        yield
        return
    import nvtx

    nvtx.push_range(name, domain=H3_VSA_NVTX_DOMAIN)
    try:
        yield
    finally:
        nvtx.pop_range(domain=H3_VSA_NVTX_DOMAIN)


def h3_vsa_skip_softmax_threshold_scale_factor() -> float:
    """Return the opt-in SM120 VSA skip-softmax threshold.

    Zero is the exact/default kernel specialization.  A positive value enables
    an approximate early-skip policy in FlashInfer's SM120 CuTeDSL kernel, so
    malformed values must fail closed instead of silently selecting a numeric
    path that was not requested.
    """
    raw = os.environ.get(H3_VSA_SKIP_SOFTMAX_THRESHOLD_ENV, "0").strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{H3_VSA_SKIP_SOFTMAX_THRESHOLD_ENV} must be a finite non-negative float, got {raw!r}"
        ) from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{H3_VSA_SKIP_SOFTMAX_THRESHOLD_ENV} must be a finite non-negative float, got {raw!r}")
    return value


if not hasattr(torch.ops.vllm_omni, "fastvideo_h3_vsa_bhsd"):

    @torch.library.custom_op("vllm_omni::fastvideo_h3_vsa_bhsd", mutates_args=())
    def _fastvideo_h3_vsa_bhsd_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_map: torch.Tensor,
        variable_block_sizes: torch.Tensor,
        logical_blocks: int,
    ) -> torch.Tensor:
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        if os.environ.get("FASTVIDEO_VSA_SM100A", "0") == "1":
            try:
                from fastvideo_kernel import block_sparse_attn_sm100a
                from fastvideo_kernel.triton_kernels.index import map_to_index

                if block_sparse_attn_sm100a.is_supported(q, variable_block_sizes):
                    (q2k_idx, q2k_num) = map_to_index(block_map)
                    (out, _) = block_sparse_attn_sm100a.block_sparse_attn_sm100a(
                        q,
                        k,
                        v,
                        q2k_idx.to(torch.int32).contiguous(),
                        q2k_num.to(torch.int32).contiguous(),
                        variable_block_sizes.to(torch.int32).contiguous(),
                        need_lse=False,
                    )
                    return out.transpose(1, 2).contiguous()
            except (ImportError, RuntimeError) as exc:
                logger.warning_once(
                    "FASTVIDEO_VSA_SM100A=1 requested but the native Blackwell forward is "
                    "unavailable (%s); using the Triton block-sparse route instead.",
                    exc,
                )
        from fastvideo_kernel.block_sparse_attn import block_sparse_attn

        logical_len = logical_blocks * H3_VSA_EFFECTIVE_TILE_SIZE
        (out, _) = block_sparse_attn(
            q[:, :, :logical_len].contiguous(),
            k[:, :, :logical_len].contiguous(),
            v[:, :, :logical_len].contiguous(),
            block_map[..., :logical_blocks, :logical_blocks].contiguous(),
            variable_block_sizes[:logical_blocks].to(torch.int32).contiguous(),
        )
        out = out.transpose(1, 2).contiguous()
        if out.shape[1] != query.shape[1]:
            out = torch.nn.functional.pad(out, (0, 0, 0, 0, 0, query.shape[1] - out.shape[1]))
        return out

    @_fastvideo_h3_vsa_bhsd_op.register_fake
    def _(query, key, value, block_map, variable_block_sizes, logical_blocks):
        del key, value, block_map, variable_block_sizes, logical_blocks
        return torch.empty_like(query)


_fastvideo_h3_vsa_bhsd_op = torch.ops.vllm_omni.fastvideo_h3_vsa_bhsd


def _flashinfer_h3_vsa_q2k_bshd_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Dispatch H3's blk64 VSA contract to the native FlashInfer kernel."""
    skip_softmax_threshold = h3_vsa_skip_softmax_threshold_scale_factor()
    if q2k_idx.dtype != torch.int32 or not q2k_idx.is_contiguous():
        raise ValueError("q2k_idx must be a contiguous int32 tensor")
    if q2k_num.dtype != torch.int32 or not q2k_num.is_contiguous():
        raise ValueError("q2k_num must be a contiguous int32 tensor")
    blocks = query.shape[1] // H3_VSA_EFFECTIVE_TILE_SIZE
    expected_idx_shape = (query.shape[0], query.shape[2], blocks, blocks)
    expected_num_shape = expected_idx_shape[:-1]
    if tuple(q2k_idx.shape) != expected_idx_shape:
        raise ValueError(f"q2k_idx shape {tuple(q2k_idx.shape)} must be {expected_idx_shape}")
    if tuple(q2k_num.shape) != expected_num_shape:
        raise ValueError(f"q2k_num shape {tuple(q2k_num.shape)} must be {expected_num_shape}")
    if q2k_idx.device != query.device or q2k_num.device != query.device:
        raise ValueError("q2k_idx and q2k_num must be on the same device as query")
    block_sizes = variable_block_sizes.to(torch.int32).contiguous()
    sage_mode = os.environ.get("VLLM_OMNI_H3_VSA_SAGE", "bf16")
    if sage_mode not in ("bf16", "sage"):
        raise ValueError("H3 VSA compute must be bf16 or sage")
    if sage_mode != "bf16":
        if skip_softmax_threshold:
            raise ValueError("Sage requires the zero skip-softmax threshold")
        from vllm_omni.diffusion.attention.ops.sage_block_sparse_attention import sage_block_sparse_attention
        from vllm_omni.diffusion.models.minimax_h3.attention.schedule import quantized_q

        with _h3_vsa_nvtx_stage("vsa.sage"):
            return sage_block_sparse_attention(
                query, key, value, q2k_idx, q2k_num, block_sizes, softmax_scale, prepared_q=quantized_q(query)
            )
    try:
        (major, minor) = current_omni_platform.get_device_capability(
            query.device.index if query.device.index is not None else torch.accelerator.current_device_index()
        )
    except (AssertionError, RuntimeError, ValueError) as exc:
        raise RuntimeError("FlashInfer H3 blk64 VSA requires a CUDA tensor on a supported GPU") from exc
    if (major, minor) in ((10, 0), (10, 3)):
        if skip_softmax_threshold:
            raise RuntimeError(
                f"{H3_VSA_SKIP_SOFTMAX_THRESHOLD_ENV} is supported only by FlashInfer's "
                f"SM120/SM121 blk64 CuTeDSL kernel"
            )
        if query.dtype != torch.bfloat16:
            raise ValueError(f"FlashInfer SM100/SM103 blk64 VSA requires bfloat16, got {query.dtype}")
        try:
            from flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk64 import bsa_attn_sm100_blk64_fwd
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "FlashInfer H3 VSA on SM100/SM103 requires flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk64"
            ) from exc
        kernel = bsa_attn_sm100_blk64_fwd
        kernel_name = "vsa_sm100_blk64_cuda"
    elif (major, minor) in ((12, 0), (12, 1)):
        try:
            from flashinfer.cute_dsl.sparse.bsa_attn_sm120 import bsa_attn_sm120_blk64_fwd
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "FlashInfer H3 VSA on SM120/SM121 requires the #4944 blk64 provider "
                "flashinfer.cute_dsl.sparse.bsa_attn_sm120"
            ) from exc
        kernel = bsa_attn_sm120_blk64_fwd
        kernel_name = "vsa_sm120_blk64_cute_dsl"
    else:
        raise RuntimeError(
            f"FlashInfer H3 blk64 VSA supports SM100/SM103 and SM120/SM121; current device is SM{major}{minor}"
        )
    logger.info_once("FASTVIDEO_VSA H3 compute kernel: FlashInfer %s", kernel_name)
    if skip_softmax_threshold:
        logger.warning_once(
            "FASTVIDEO_VSA H3 approximate skip-softmax enabled: %s=%g; this output is not part "
            "of the exact VSA baseline",
            H3_VSA_SKIP_SOFTMAX_THRESHOLD_ENV,
            skip_softmax_threshold,
        )
    kernel_kwargs: dict[str, Any] = {}
    if (major, minor) in ((12, 0), (12, 1)) and skip_softmax_threshold > 0:
        kernel_kwargs["skip_softmax_threshold_scale_factor"] = skip_softmax_threshold
    (output, _) = kernel(
        query,
        key,
        value,
        q2k_idx,
        int(q2k_idx.shape[-1]),
        block_sizes=block_sizes,
        q2k_block_nums=q2k_num,
        softmax_scale=softmax_scale,
        return_lse=False,
        **kernel_kwargs,
    )
    return output


def _flashinfer_h3_vsa_bshd_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Legacy FlashInfer wrapper accepting FastVideo's dense bool map."""
    from fastvideo_kernel.triton_kernels.index import map_to_index

    (q2k_idx, q2k_num) = map_to_index(block_map)
    return _flashinfer_h3_vsa_q2k_bshd_impl(
        query,
        key,
        value,
        q2k_idx.to(torch.int32).contiguous(),
        q2k_num.to(torch.int32).contiguous(),
        variable_block_sizes,
        softmax_scale,
    )


if not hasattr(torch.ops.vllm_omni, "flashinfer_h3_vsa_bshd"):

    @torch.library.custom_op("vllm_omni::flashinfer_h3_vsa_bshd", mutates_args=())
    def _flashinfer_h3_vsa_bshd_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_map: torch.Tensor,
        variable_block_sizes: torch.Tensor,
        softmax_scale: float,
    ) -> torch.Tensor:
        return _flashinfer_h3_vsa_bshd_impl(query, key, value, block_map, variable_block_sizes, softmax_scale)

    @_flashinfer_h3_vsa_bshd_op.register_fake
    def _(query, key, value, block_map, variable_block_sizes, softmax_scale):
        del key, value, block_map, variable_block_sizes, softmax_scale
        return torch.empty_like(query)


_flashinfer_h3_vsa_bshd_op = torch.ops.vllm_omni.flashinfer_h3_vsa_bshd
if not hasattr(torch.ops.vllm_omni, "flashinfer_h3_vsa_q2k_bshd"):

    @torch.library.custom_op("vllm_omni::flashinfer_h3_vsa_q2k_bshd", mutates_args=())
    def _flashinfer_h3_vsa_q2k_bshd_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q2k_idx: torch.Tensor,
        q2k_num: torch.Tensor,
        variable_block_sizes: torch.Tensor,
        softmax_scale: float,
    ) -> torch.Tensor:
        return _flashinfer_h3_vsa_q2k_bshd_impl(
            query, key, value, q2k_idx, q2k_num, variable_block_sizes, softmax_scale
        )

    @_flashinfer_h3_vsa_q2k_bshd_op.register_fake
    def _(query, key, value, q2k_idx, q2k_num, variable_block_sizes, softmax_scale):
        del key, value, q2k_idx, q2k_num, variable_block_sizes, softmax_scale
        return torch.empty_like(query)


_flashinfer_h3_vsa_q2k_bshd_op = torch.ops.vllm_omni.flashinfer_h3_vsa_q2k_bshd


@functools.lru_cache(maxsize=32)
def _get_h3_tile_metadata(
    prefix_segments: tuple[int, ...], video_shape: tuple[int, int, int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Official FastVideo H3 geometry: pure prefix chunks + 3-D video tiles."""
    block_size = H3_VSA_EFFECTIVE_TILE_SHAPE
    block_elements = H3_VSA_EFFECTIVE_TILE_SIZE
    prefix_len = sum(prefix_segments)
    prefix_sizes: list[int] = []
    for segment in prefix_segments:
        (full, remainder) = divmod(segment, block_elements)
        prefix_sizes.extend([block_elements] * full)
        if remainder:
            prefix_sizes.append(remainder)
    video_indices = _get_tile_partition_indices(video_shape, block_size, device) + prefix_len
    video_sizes = _construct_variable_block_sizes(video_shape, block_size, device)
    partition = torch.cat([torch.arange(prefix_len, device=device, dtype=torch.long), video_indices])
    sizes = torch.cat([torch.tensor(prefix_sizes, device=device, dtype=torch.int32), video_sizes.to(torch.int32)])
    non_pad = _get_non_pad_index(sizes, block_elements)
    untile = non_pad[torch.argsort(partition)]
    total = prefix_len + math.prod(video_shape)
    if int(sizes.sum()) != total or untile.numel() != total:
        raise ValueError(
            f"invalid H3 VSA geometry: prefix={prefix_segments}, video={video_shape}, "
            f"sizes_sum={int(sizes.sum())}, total={total}"
        )
    return (partition, sizes, non_pad, untile, len(prefix_sizes), int(video_sizes.numel()))


@functools.lru_cache(maxsize=32)
def _get_h3_tiled_source_rows(
    prefix_segments: tuple[int, ...], video_shape: tuple[int, int, int], padded_rows: int, device: torch.device
) -> torch.Tensor:
    """Cache the validated destination-to-source map for fused H3 tiling."""
    (partition, _sizes, non_pad, _untile, _prefix_blocks, _video_blocks) = _get_h3_tile_metadata(
        prefix_segments, video_shape, device
    )
    return build_h3_tiled_source_rows(partition, non_pad, padded_rows)


@functools.lru_cache(maxsize=32)
def _get_h3_aligned_untile_source_rows(
    prefix_segments: tuple[int, ...],
    video_shape: tuple[int, int, int],
    aligned_rows: int,
    tiled_rows: int,
    device: torch.device,
) -> torch.Tensor:
    """Cache the validated tile-64 source map for aligned H3 output."""
    (_partition, sizes, _non_pad, untile, _prefix_blocks, _video_blocks) = _get_h3_tile_metadata(
        prefix_segments, video_shape, device
    )
    expected_tiled_rows = int(sizes.numel()) * H3_VSA_EFFECTIVE_TILE_SIZE
    if tiled_rows != expected_tiled_rows:
        raise ValueError(f"H3 VSA output has {tiled_rows} logical tiled rows, expected {expected_tiled_rows}")
    return build_h3_aligned_untile_source_rows(untile, aligned_rows=aligned_rows, tiled_rows=tiled_rows)


@functools.lru_cache(maxsize=32)
@torch.compiler.disable
def get_h3_vsa_owner_route_plan(
    prefix_segments: tuple[int, ...], video_shape: tuple[int, int, int], aligned_rows: int, sp_world_size: int
) -> H3VSAOwnerRoutePlan:
    """Build and cache the exact aligned H3 row-owner route on CPU.

    Ulysses needs ``Kmax`` before the attention backend runs so it can acquire
    an expanded reverse-O landing.  Deriving the plan from the same H3 untile
    metadata used below keeps the producer and consumer on one row-order
    contract without synchronizing CUDA metadata back to the host.
    """
    if aligned_rows <= 0 or sp_world_size <= 1:
        raise ValueError(
            f"H3 VSA owner routing requires positive aligned_rows and SP world_size > 1, got "
            f"aligned_rows={aligned_rows}, sp_world_size={sp_world_size}"
        )
    if aligned_rows % sp_world_size:
        raise ValueError(f"H3 VSA aligned rows must divide the SP world: rows={aligned_rows}, world={sp_world_size}")
    cpu = torch.device("cpu")
    (_partition, sizes, _non_pad, untile, _prefix_blocks, _video_blocks) = _get_h3_tile_metadata(
        prefix_segments, video_shape, cpu
    )
    logical_blocks = int(sizes.numel())
    aligned_source_rows = build_h3_aligned_untile_source_rows(
        untile, aligned_rows=aligned_rows, tiled_rows=logical_blocks * H3_VSA_EFFECTIVE_TILE_SIZE
    )
    return build_h3_vsa_owner_route_plan(
        aligned_source_rows,
        sp_world_size=sp_world_size,
        global_rows=aligned_rows,
        local_rows=aligned_rows // sp_world_size,
        logical_block_count=logical_blocks,
    )


def _get_h3_layout(attn_metadata: AttentionMetadata | None) -> tuple[tuple[int, ...], tuple[int, int, int], int] | None:
    if attn_metadata is None or attn_metadata.video_layout is None:
        return None
    prefix = attn_metadata.extra.get("vsa_h3_prefix_segments")
    if not isinstance(prefix, (tuple, list)):
        return None
    target = next((span for span in reversed(attn_metadata.video_layout.video_spans) if span.role == "target"), None)
    if target is None:
        return None
    return (tuple(int(x) for x in prefix if int(x) > 0), target.latent_grid, target.start)


def _pool_h3_tiles(x: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    (batch, seq_len, heads, dim) = x.shape
    blocks = seq_len // H3_VSA_EFFECTIVE_TILE_SIZE
    pooled = x.view(batch, blocks, H3_VSA_EFFECTIVE_TILE_SIZE, heads, dim).sum(dim=2, dtype=torch.float32)
    pooled = pooled / sizes.view(1, -1, 1, 1).clamp_min(1)
    return pooled.permute(0, 2, 1, 3)


def _build_h3_block_map(scores: torch.Tensor, num_prefix_blocks: int, num_video_blocks: int, topk: int) -> torch.Tensor:
    """Prefix K/V are exempt and prefix queries stay dense, as in FastVideo."""
    keep_video = min(topk, num_video_blocks)
    if keep_video == num_video_blocks:
        return torch.ones_like(scores, dtype=torch.bool)
    block_map = torch.zeros_like(scores, dtype=torch.bool)
    indices = scores[..., num_prefix_blocks:].topk(keep_video, dim=-1).indices + num_prefix_blocks
    block_map.scatter_(-1, indices, True)
    block_map[..., :num_prefix_blocks] = True
    block_map[:, :, :num_prefix_blocks, :] = True
    return block_map


def _build_h3_ordered_q2k_indices(
    scores: torch.Tensor, num_prefix_blocks: int, num_video_blocks: int, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build FlashInfer's ordered sparse metadata without a dense bool map.

    ``map_to_index`` scans each bool-map row from left to right. Sorting the
    selected video block IDs and prepending the exempt prefix therefore emits
    exactly the same index order and ``-1`` tail, while avoiding both the bool
    map allocation and its full-width scan.
    """
    if scores.ndim != 4:
        raise ValueError(f"scores must be [B, H, Nq, Nkv], got {tuple(scores.shape)}")
    if num_prefix_blocks < 0 or num_video_blocks < 0:
        raise ValueError("num_prefix_blocks and num_video_blocks must be non-negative")
    total_blocks = num_prefix_blocks + num_video_blocks
    if scores.shape[-2:] != (total_blocks, total_blocks):
        raise ValueError(
            f"scores must match the H3 prefix+video block geometry: got "
            f"{tuple(scores.shape[-2:])}, expected {(total_blocks, total_blocks)}"
        )
    if topk < 0:
        raise ValueError(f"topk must be non-negative, got {topk}")
    keep_video = min(topk, num_video_blocks)
    selected: torch.Tensor | None = None
    if keep_video < num_video_blocks and keep_video:
        selected = scores[..., num_prefix_blocks:, num_prefix_blocks:].topk(keep_video, dim=-1).indices.to(torch.int32)
        selected = selected.sort(dim=-1).values
    output_shape = tuple(scores.shape)
    q2k_idx = torch.full(output_shape, -1, dtype=torch.int32, device=scores.device)
    q2k_num = torch.full(output_shape[:-1], num_prefix_blocks + keep_video, dtype=torch.int32, device=scores.device)
    all_indices = torch.arange(total_blocks, dtype=torch.int32, device=scores.device)
    if keep_video == num_video_blocks:
        q2k_idx.copy_(all_indices)
        q2k_num.fill_(total_blocks)
        return (q2k_idx, q2k_num)
    if num_prefix_blocks:
        q2k_idx[..., num_prefix_blocks:, :num_prefix_blocks] = all_indices[:num_prefix_blocks]
    if selected is not None:
        q2k_idx[..., num_prefix_blocks:, num_prefix_blocks : num_prefix_blocks + keep_video] = (
            selected + num_prefix_blocks
        )
    if num_prefix_blocks:
        q2k_idx[..., :num_prefix_blocks, :] = all_indices
        q2k_num[..., :num_prefix_blocks] = total_blocks
    return (q2k_idx, q2k_num)


class MiniMaxH3VSAImpl(FastVideoVSAImpl):
    def __init__(
        self,
        num_heads,
        head_size,
        softmax_scale,
        causal=False,
        num_kv_heads=None,
        prefix="",
        qkv_layout=None,
        backend_kwargs=None,
        **extra_impl_args,
    ):
        super().__init__(
            num_heads,
            head_size,
            softmax_scale,
            causal,
            num_kv_heads,
            prefix,
            qkv_layout,
            backend_kwargs,
            **extra_impl_args,
        )
        self.h3_kernel_backend = os.getenv("VLLM_OMNI_H3_VSA_KERNEL", "fastvideo")
        if self.h3_kernel_backend not in {"fastvideo", "flashinfer"}:
            raise ValueError("VLLM_OMNI_H3_VSA_KERNEL must be fastvideo or flashinfer")

    h3_effective_tile_shape = H3_VSA_EFFECTIVE_TILE_SHAPE
    h3_effective_tile_size = H3_VSA_EFFECTIVE_TILE_SIZE

    def _forward_h3(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
        aligned_rows: int,
    ) -> torch.Tensor:
        skip_softmax_threshold = h3_vsa_skip_softmax_threshold_scale_factor()
        if skip_softmax_threshold and self.h3_kernel_backend != "flashinfer":
            raise RuntimeError(
                f"{H3_VSA_SKIP_SOFTMAX_THRESHOLD_ENV}>0 requires FASTVIDEO_VSA h3_kernel_backend='flashinfer'"
            )
        layout = _get_h3_layout(attn_metadata)
        if layout is None:
            raise ValueError("incomplete VSA-H3 layout metadata")
        (prefix_segments, video_shape, target_start) = layout
        if sum(prefix_segments) != target_start:
            raise ValueError(f"VSA-H3 prefix segments sum to {sum(prefix_segments)}, target starts at {target_start}")
        expected = target_start + math.prod(video_shape)
        if query.shape[1] != expected:
            raise ValueError(f"VSA-H3 layout has {expected} rows but attention received {query.shape[1]}")
        if query.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"VSA-H3 requires fp16/bf16 tensors, got {query.dtype}")
        o_bundle = attn_metadata.extra.get(H3_VSA_O_BUNDLE_ACTIVE_KEY, False)
        if not isinstance(o_bundle, bool):
            raise TypeError(f"{H3_VSA_O_BUNDLE_ACTIVE_KEY} must be a bool")
        o_bundle_state = attn_metadata.extra.get(H3_VSA_O_BUNDLE_STATE_KEY)
        owner_route: H3VSAOwnerRoutePlan | None = None
        if o_bundle:
            if not h3_vsa_o_bundle_enabled():
                raise RuntimeError(f"{H3_VSA_O_BUNDLE_ACTIVE_KEY}=True requires {H3_VSA_O_BUNDLE_ENV}=1")
            if query.dtype != torch.bfloat16:
                raise TypeError("H3 VSA reverse-O bundling currently requires BF16")
            if not isinstance(o_bundle_state, dict):
                raise RuntimeError("H3 VSA reverse-O bundle mutable state is absent")
            if set(o_bundle_state) != {"plan"}:
                raise RuntimeError(f"stale H3 VSA reverse-O bundle mutable state: keys={sorted(o_bundle_state)}")
            candidate_plan = o_bundle_state["plan"]
            if not isinstance(candidate_plan, H3VSAOwnerRoutePlan) or not h3_vsa_owner_route_plan_is_trusted(
                candidate_plan
            ):
                raise RuntimeError("H3 VSA reverse-O bundle route plan is absent or untrusted")
            owner_route = candidate_plan
            if owner_route.sp_world_size * owner_route.local_rows != aligned_rows:
                raise ValueError(
                    f"H3 VSA reverse-O bundle route does not match aligned attention rows: "
                    f"route={owner_route.sp_world_size}*{owner_route.local_rows}, attention={aligned_rows}"
                )
        gate = _get_gate_compress(attn_metadata)
        if o_bundle and gate is not None:
            raise RuntimeError("deferred/bundled H3 VSA must retain gate_compress on the original SP shard")
        if gate is not None:
            if gate.shape[0] != query.shape[0] or gate.shape[2:] != query.shape[2:] or gate.shape[1] < query.shape[1]:
                raise ValueError(f"gate_compress shape {gate.shape} cannot cover query shape {query.shape}")
            gate = gate[:, : query.shape[1]]
        (partition, sizes, non_pad, untile, prefix_blocks, video_blocks) = _get_h3_tile_metadata(
            prefix_segments, video_shape, query.device
        )
        logical_blocks = int(sizes.numel())
        if owner_route is not None and owner_route.logical_block_count != logical_blocks:
            raise ValueError(
                f"H3 VSA reverse-O bundle route block count does not match the backend layout: "
                f"route={owner_route.logical_block_count}, backend={logical_blocks}"
            )
        pair_pad = logical_blocks % 2 if self.h3_kernel_backend == "fastvideo" else 0
        kernel_blocks = logical_blocks + pair_pad
        target_shape = (query.shape[0], kernel_blocks * H3_VSA_EFFECTIVE_TILE_SIZE, query.shape[2], query.shape[3])
        use_fused_tile_pack = h3_vsa_fused_tile_pack_enabled()
        use_fused_untile = h3_vsa_fused_untile_enabled()
        qkv_source_rows: torch.Tensor | None = None
        gate_source_rows: torch.Tensor | None = None
        if use_fused_tile_pack:
            qkv_source_rows = _get_h3_tiled_source_rows(prefix_segments, video_shape, target_shape[1], query.device)
            if gate is not None and pair_pad:
                gate_source_rows = _get_h3_tiled_source_rows(
                    prefix_segments, video_shape, logical_blocks * H3_VSA_EFFECTIVE_TILE_SIZE, query.device
                )
            else:
                gate_source_rows = qkv_source_rows
            pack_gate = gate is not None
            packed_tensor_count = 3 + int(pack_gate)
            fused_inputs = (query, key, value, gate) if pack_gate else (query, key, value)
            fused_maps = (
                (qkv_source_rows, qkv_source_rows, qkv_source_rows, gate_source_rows)
                if pack_gate
                else (qkv_source_rows,) * 3
            )
            use_fused_tile_pack = all(
                source_rows is not None and h3_vsa_tile_pack_cuda_supported(tensor, source_rows)
                for (tensor, source_rows) in zip(fused_inputs, fused_maps, strict=True)
            )
            if use_fused_tile_pack:
                logger.info_once(
                    "FASTVIDEO_VSA H3 tile pack: fused Triton compact->tile64 enabled "
                    "(compact_rows=%d, padded_rows=%d, tensors=%d)",
                    query.shape[1],
                    target_shape[1],
                    packed_tensor_count,
                    scope="process",
                )
            else:
                logger.warning_once(
                    "%s=1 requested but the H3 input shape/platform is unsupported; using the reference tile pack",
                    H3_VSA_FUSED_TILE_PACK_ENV,
                    scope="process",
                )
        from vllm_omni.diffusion.models.minimax_h3.attention.overlap import prepared

        overlap_prepared = prepared()
        if overlap_prepared is not None:
            assert use_fused_tile_pack and gate is None
            (q_tiled, k_tiled, q_pool, k_pool, scores, q2k_idx, q2k_num) = overlap_prepared
            with _h3_vsa_nvtx_stage("vsa.layout.tile_pack_v"):
                v_tiled = h3_vsa_tile_pack(value, qkv_source_rows)
        else:
            with _h3_vsa_nvtx_stage("vsa.layout.tile_pack"):
                if use_fused_tile_pack:
                    assert qkv_source_rows is not None
                    q_tiled = h3_vsa_tile_pack(query, qkv_source_rows)
                    k_tiled = h3_vsa_tile_pack(key, qkv_source_rows)
                    v_tiled = h3_vsa_tile_pack(value, qkv_source_rows)
                else:
                    assert True
                    q_tiled = torch.zeros(target_shape, device=query.device, dtype=query.dtype)
                    k_tiled = torch.zeros_like(q_tiled)
                    v_tiled = torch.zeros_like(q_tiled)
                    q_tiled[:, non_pad] = query[:, partition]
                    k_tiled[:, non_pad] = key[:, partition]
                    v_tiled[:, non_pad] = value[:, partition]
            with _h3_vsa_nvtx_stage("vsa.coarse.pool_qk"):
                q_pool = _pool_h3_tiles(q_tiled[:, : logical_blocks * H3_VSA_EFFECTIVE_TILE_SIZE], sizes)
                k_pool = _pool_h3_tiles(k_tiled[:, : logical_blocks * H3_VSA_EFFECTIVE_TILE_SIZE], sizes)
            with _h3_vsa_nvtx_stage("vsa.coarse.qk_scores"):
                scores = torch.matmul(q_pool, k_pool.transpose(-2, -1)) * self.softmax_scale
        kernel_sizes = sizes
        from vllm_omni.diffusion.models.minimax_h3.attention.schedule import coarse_ready, coarse_scope

        coarse_ticket = coarse_ready(v_tiled, scores) if gate is not None or o_bundle else None

        def compute_coarse():
            with coarse_scope(coarse_ticket):
                with _h3_vsa_nvtx_stage("vsa.coarse.pool_v"):
                    v_pool = _pool_h3_tiles(v_tiled[:, : logical_blocks * H3_VSA_EFFECTIVE_TILE_SIZE], sizes)
                with _h3_vsa_nvtx_stage("vsa.coarse.softmax"):
                    coarse_weights = torch.softmax(scores, dim=-1)
                with _h3_vsa_nvtx_stage("vsa.coarse.pv"):
                    coarse = torch.matmul(coarse_weights, v_pool)
                return coarse.permute(0, 2, 1, 3).to(q_tiled.dtype)

        compressed = None
        logger.info_once(
            "FASTVIDEO_VSA H3 routing: seq_len=%d, prefix_segments=%s, video_shape=%s, "
            "prefix_blocks=%d, video_blocks=%d, topk=%d, kernel_blocks=%d, compute_backend=%s",
            query.shape[1],
            prefix_segments,
            video_shape,
            prefix_blocks,
            video_blocks,
            min(self.topk, video_blocks),
            kernel_blocks,
            self.h3_kernel_backend,
        )
        if self.h3_kernel_backend == "flashinfer":
            if h3_vsa_direct_q2k_enabled():
                logger.info_once(
                    "FASTVIDEO_VSA H3 metadata: direct ordered q2k enabled by %s=1",
                    H3_VSA_DIRECT_Q2K_ENV,
                    scope="process",
                )
                with _h3_vsa_nvtx_stage("vsa.route.topk_q2k"):
                    if overlap_prepared is None:
                        (q2k_idx, q2k_num) = _build_h3_ordered_q2k_indices(
                            scores, prefix_blocks, video_blocks, self.topk
                        )
                with _h3_vsa_nvtx_stage("vsa.fine.block_sparse_attention"):
                    fine_op = _flashinfer_h3_vsa_q2k_bshd_op
                    output = fine_op(
                        q_tiled.contiguous(),
                        k_tiled.contiguous(),
                        v_tiled.contiguous(),
                        q2k_idx,
                        q2k_num,
                        kernel_sizes.contiguous(),
                        self.softmax_scale,
                    )
            else:
                logger.info_once(
                    "FASTVIDEO_VSA H3 metadata: legacy bool-map conversion active; set %s=1 to "
                    "qualify direct ordered q2k",
                    H3_VSA_DIRECT_Q2K_ENV,
                    scope="process",
                )
                with _h3_vsa_nvtx_stage("vsa.route.topk_q2k"):
                    block_map = _build_h3_block_map(scores, prefix_blocks, video_blocks, self.topk)
                with _h3_vsa_nvtx_stage("vsa.fine.block_sparse_attention"):
                    output = _flashinfer_h3_vsa_bshd_op(
                        q_tiled.contiguous(),
                        k_tiled.contiguous(),
                        v_tiled.contiguous(),
                        block_map.contiguous(),
                        kernel_sizes.contiguous(),
                        self.softmax_scale,
                    )
        else:
            with _h3_vsa_nvtx_stage("vsa.route.topk_q2k"):
                block_map = _build_h3_block_map(scores, prefix_blocks, video_blocks, self.topk)
                if pair_pad:
                    block_map = torch.nn.functional.pad(block_map, (0, 1, 0, 1), value=False)
                    kernel_sizes = torch.nn.functional.pad(sizes, (0, 1), value=0)
            with _h3_vsa_nvtx_stage("vsa.fine.block_sparse_attention"):
                output = _fastvideo_h3_vsa_bhsd_op(
                    q_tiled.contiguous(),
                    k_tiled.contiguous(),
                    v_tiled.contiguous(),
                    block_map.contiguous(),
                    kernel_sizes.contiguous(),
                    logical_blocks,
                )
        output = output[:, : logical_blocks * H3_VSA_EFFECTIVE_TILE_SIZE]
        if gate is not None or o_bundle:
            compressed = compute_coarse()
        aligned_untile_source_rows: torch.Tensor | None = None
        o_input_landing = attn_metadata.extra.get("ulysses_o_input_landing")
        if o_input_landing is not None and (not isinstance(o_input_landing, torch.Tensor)):
            raise TypeError(f"ulysses_o_input_landing must be a torch.Tensor, got {type(o_input_landing)!r}")
        if o_bundle:
            assert compressed is not None
            assert owner_route is not None
            assert isinstance(o_bundle_state, dict)
            with _h3_vsa_nvtx_stage("vsa.output.untile"):
                aligned_untile_source_rows = _get_h3_aligned_untile_source_rows(
                    prefix_segments, video_shape, aligned_rows, output.shape[1], output.device
                )
                if use_fused_untile:
                    if not h3_vsa_tile_untile_cuda_supported(output, aligned_untile_source_rows):
                        raise RuntimeError(
                            "H3 VSA reverse-O bundling requested with fused untile, but the "
                            "tile64 output geometry is unsupported"
                        )
                    fine_aligned = h3_vsa_tile_untile(output, aligned_untile_source_rows)
                    logger.info_once(
                        "FASTVIDEO_VSA H3 output untile: fused Triton tile64->aligned enabled "
                        "(valid_rows=%d, aligned_rows=%d, tiled_rows=%d, destination=o_bundle_staging)",
                        untile.numel(),
                        aligned_rows,
                        output.shape[1],
                        scope="process",
                    )
                else:
                    fine_aligned = torch.zeros(
                        (output.shape[0], aligned_rows, output.shape[2], output.shape[3]),
                        dtype=output.dtype,
                        device=output.device,
                    )
                    fine_aligned[:, : untile.numel()].copy_(output[:, untile])
            from vllm_omni.diffusion.models.minimax_h3.attention.overlap import publish_reverse

            if publish_reverse(fine_aligned, compressed, owner_route):
                return fine_aligned
            raise RuntimeError("H3 reverse-O bundling requires the active model overlap scope")
        if use_fused_untile:
            aligned_untile_source_rows = _get_h3_aligned_untile_source_rows(
                prefix_segments, video_shape, aligned_rows, output.shape[1], output.device
            )
        if gate is not None:
            assert compressed is not None
            if use_fused_tile_pack:
                if gate_source_rows is None:
                    gate_source_rows = _get_h3_tiled_source_rows(
                        prefix_segments, video_shape, logical_blocks * H3_VSA_EFFECTIVE_TILE_SIZE, query.device
                    )
                if h3_vsa_tile_pack_cuda_supported(gate, gate_source_rows):
                    gate_tiled = h3_vsa_tile_pack(gate, gate_source_rows)
                else:
                    gate_tiled = torch.zeros_like(q_tiled[:, : logical_blocks * H3_VSA_EFFECTIVE_TILE_SIZE])
                    gate_tiled[:, non_pad] = gate[:, partition]
            else:
                gate_tiled = torch.zeros_like(q_tiled[:, : logical_blocks * H3_VSA_EFFECTIVE_TILE_SIZE])
                gate_tiled[:, non_pad] = gate[:, partition]
            output = (
                output.view(
                    output.shape[0], logical_blocks, H3_VSA_EFFECTIVE_TILE_SIZE, output.shape[2], output.shape[3]
                )
                + compressed.unsqueeze(2)
                * gate_tiled.view(
                    gate_tiled.shape[0],
                    logical_blocks,
                    H3_VSA_EFFECTIVE_TILE_SIZE,
                    gate_tiled.shape[2],
                    gate_tiled.shape[3],
                )
            ).view_as(output)
        if use_fused_untile:
            assert aligned_untile_source_rows is not None
            if h3_vsa_tile_untile_cuda_supported(output, aligned_untile_source_rows):
                logger.info_once(
                    "FASTVIDEO_VSA H3 output untile: fused Triton tile64->aligned enabled "
                    "(valid_rows=%d, aligned_rows=%d, tiled_rows=%d, destination=%s)",
                    untile.numel(),
                    aligned_rows,
                    output.shape[1],
                    "registered_reverse_o" if o_input_landing is not None else "functional",
                    scope="process",
                )
                if o_input_landing is not None:
                    return h3_vsa_tile_untile_out(output, aligned_untile_source_rows, out=o_input_landing)
                return h3_vsa_tile_untile(output, aligned_untile_source_rows)
            logger.warning_once(
                "FASTVIDEO_VSA H3 output untile fallback: %s=1 requested but the shape/platform "
                "is unsupported; using the reference tile64->compact->aligned path "
                "(valid_rows=%d, aligned_rows=%d, tiled_rows=%d)",
                H3_VSA_FUSED_UNTILE_ENV,
                untile.numel(),
                aligned_rows,
                output.shape[1],
                scope="process",
            )
        return output[:, untile].contiguous()

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        (original_query, original_key, original_value) = (query, key, value)
        original_seq_len = query.shape[1]
        valid_seq_len = original_seq_len
        if attn_metadata is not None and attn_metadata.packed_padding is not None:
            valid_seq_len = attn_metadata.packed_padding.q_length
            if attn_metadata.packed_padding.kv_length != valid_seq_len:
                return self._fallback(
                    original_query, original_key, original_value, attn_metadata, "packed Q/KV lengths must match"
                )
            query = query[:, :valid_seq_len]
            key = key[:, :valid_seq_len]
            value = value[:, :valid_seq_len]
        h3_layout = _get_h3_layout(attn_metadata)
        if attn_metadata is not None and h3_layout is not None:
            try:
                output = self._forward_h3(query, key, value, attn_metadata, original_seq_len)
                o_bundle = attn_metadata.extra.get(H3_VSA_O_BUNDLE_ACTIVE_KEY, False)
                if o_bundle is True:
                    state = attn_metadata.extra.get(H3_VSA_O_BUNDLE_STATE_KEY)
                    plan = state.get("plan") if isinstance(state, dict) else None
                    if not isinstance(plan, H3VSAOwnerRoutePlan):
                        raise RuntimeError("H3 VSA reverse-O bundle plan disappeared after producer execution")
                    from vllm_omni.diffusion.models.minimax_h3.attention.overlap import ACTIVE

                    overlap_state = ACTIVE.get()
                    if overlap_state is not None:
                        (fine, _coarse, owner_plan) = overlap_state["reverse"]
                        expected_fine_rows = original_seq_len
                        if output is not fine or owner_plan is not plan or output.shape[1] != expected_fine_rows:
                            raise RuntimeError("H3 chunked reverse-O producer lost its fine/route identity")
                        return output
                    expected_bundle_rows = plan.sp_world_size * (plan.local_rows + plan.kmax)
                    if output.shape[1] != expected_bundle_rows:
                        raise ValueError(
                            f"H3 VSA reverse-O producer returned the wrong bundle rows: "
                            f"got={output.shape[1]}, expected={expected_bundle_rows}"
                        )
                    return output
                if output.shape[1] == original_seq_len:
                    return output
                if output.shape[1] != valid_seq_len:
                    raise ValueError(
                        f"VSA-H3 output must contain either the valid or aligned sequence, got "
                        f"{output.shape[1]} rows for valid={valid_seq_len}, aligned={original_seq_len}"
                    )
                restored = torch.zeros(original_query.shape, dtype=output.dtype, device=output.device)
                restored[:, :valid_seq_len] = output
                return restored
            except Exception as exc:
                o_bundle = attn_metadata.extra.get(H3_VSA_O_BUNDLE_ACTIVE_KEY, False)
                if (
                    o_bundle is True
                    or self.h3_kernel_backend == "flashinfer"
                    or isinstance(exc, torch.AcceleratorError)
                ):
                    raise
                if not self.fallback_on_error:
                    raise
                return self._fallback(
                    original_query, original_key, original_value, attn_metadata, f"VSA-H3 kernel failed: {exc}"
                )
        return super().forward_cuda(original_query, original_key, original_value, attn_metadata)


class MiniMaxH3VSABackend(FastVideoVSABackend):
    """Prefix-dense tile64 VSA; Wan retains the shared tile256 backend."""

    @staticmethod
    def get_impl_cls():
        return MiniMaxH3VSAImpl
