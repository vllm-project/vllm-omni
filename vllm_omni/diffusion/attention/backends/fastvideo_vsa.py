# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import functools
import math
from collections.abc import Mapping
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available

logger = init_logger(__name__)


# Keep the external FastVideo pybind/CUDA kernel opaque to torch.compile.
# This mirrors the SageAttention3 backend pattern: tracing the raw extension
# through Dynamo can reach Inductor scheduling with unstable internal op names
# (e.g. KeyError: "op12").  The custom op gives Dynamo a single Tensor->Tensor
# boundary and lets Inductor schedule the surrounding Wan block normally.
if not hasattr(torch.ops.vllm_omni, "fastvideo_vsa_bshd"):

    @torch.library.custom_op("vllm_omni::fastvideo_vsa_bshd", mutates_args=())
    def _fastvideo_vsa_bshd_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        variable_block_sizes: torch.Tensor,
        q_variable_block_sizes: torch.Tensor,
        compress_attn_weight: torch.Tensor,
        topk: int,
        block_t: int,
        block_h: int,
        block_w: int,
    ) -> torch.Tensor:
        from fastvideo_kernel import video_sparse_attn_bshd

        return video_sparse_attn_bshd(
            query,
            key,
            value,
            variable_block_sizes=variable_block_sizes,
            q_variable_block_sizes=q_variable_block_sizes,
            topk=topk,
            block_size=(block_t, block_h, block_w),
            compress_attn_weight=compress_attn_weight if compress_attn_weight.numel() else None,
        )

    @_fastvideo_vsa_bshd_op.register_fake
    def _(
        query,
        key,
        value,
        variable_block_sizes,
        q_variable_block_sizes,
        compress_attn_weight,
        topk,
        block_t,
        block_h,
        block_w,
    ):
        del (
            key,
            value,
            variable_block_sizes,
            q_variable_block_sizes,
            compress_attn_weight,
            topk,
            block_t,
            block_h,
            block_w,
        )
        return torch.empty_like(query)


_fastvideo_vsa_bshd_op = torch.ops.vllm_omni.fastvideo_vsa_bshd


@functools.lru_cache(maxsize=32)
def _get_tile_partition_indices(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    t_size, h_size, w_size = dit_seq_shape
    tile_t, tile_h, tile_w = tile_size
    indices = torch.arange(t_size * h_size * w_size, device=device, dtype=torch.long).reshape(t_size, h_size, w_size)
    tiles = []
    for tile_t_idx in range(math.ceil(t_size / tile_t)):
        for tile_h_idx in range(math.ceil(h_size / tile_h)):
            for tile_w_idx in range(math.ceil(w_size / tile_w)):
                tiles.append(
                    indices[
                        tile_t_idx * tile_t : min((tile_t_idx + 1) * tile_t, t_size),
                        tile_h_idx * tile_h : min((tile_h_idx + 1) * tile_h, h_size),
                        tile_w_idx * tile_w : min((tile_w_idx + 1) * tile_w, w_size),
                    ].flatten()
                )
    return torch.cat(tiles, dim=0)


@functools.lru_cache(maxsize=32)
def _construct_variable_block_sizes(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    num_tiles = tuple(math.ceil(seq_dim / tile_dim) for seq_dim, tile_dim in zip(dit_seq_shape, tile_size))

    def _sizes(dim_len: int, tile: int, n_tiles: int) -> torch.Tensor:
        sizes = torch.full((n_tiles,), tile, dtype=torch.int32, device=device)
        remainder = dim_len - (n_tiles - 1) * tile
        sizes[-1] = remainder if remainder > 0 else tile
        return sizes

    t_sizes = _sizes(dit_seq_shape[0], tile_size[0], num_tiles[0])
    h_sizes = _sizes(dit_seq_shape[1], tile_size[1], num_tiles[1])
    w_sizes = _sizes(dit_seq_shape[2], tile_size[2], num_tiles[2])
    return (t_sizes[:, None, None] * h_sizes[None, :, None] * w_sizes[None, None, :]).reshape(-1)


@functools.lru_cache(maxsize=32)
def _get_non_pad_index(variable_block_sizes: torch.Tensor, max_block_size: int) -> torch.Tensor:
    num_blocks = variable_block_sizes.shape[0]
    device = variable_block_sizes.device
    starts = torch.arange(num_blocks, device=device) * max_block_size
    padded_index = starts[:, None] + torch.arange(max_block_size, device=device)[None, :]
    valid = torch.arange(max_block_size, device=device)[None, :] < variable_block_sizes[:, None]
    return padded_index[valid]


@torch.compiler.disable
def _get_tile_metadata(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    block_elements: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tile_partition_indices = _get_tile_partition_indices(dit_seq_shape, tile_size, device)
    variable_block_sizes = _construct_variable_block_sizes(dit_seq_shape, tile_size, device)
    non_pad_index = _get_non_pad_index(variable_block_sizes, block_elements)
    untile_combined_index = non_pad_index[torch.argsort(tile_partition_indices)]
    return tile_partition_indices, variable_block_sizes, non_pad_index, untile_combined_index


def _get_vsa_dit_seq_shape(attn_metadata: AttentionMetadata | None) -> tuple[int, int, int] | None:
    if attn_metadata is None:
        return None
    value = attn_metadata.extra.get("vsa_dit_seq_shape")
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    return (int(value[0]), int(value[1]), int(value[2]))


def _get_gate_compress(attn_metadata: AttentionMetadata | None) -> torch.Tensor | None:
    if attn_metadata is None:
        return None
    value = attn_metadata.extra.get("gate_compress")
    return value if isinstance(value, torch.Tensor) else None


def _preserve_vsa_all_blocks(attn_metadata: AttentionMetadata | None) -> bool:
    if attn_metadata is None:
        return False
    return attn_metadata.extra.get("preserve_vsa_all_blocks") is True


class FastVideoVSABackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        # FastVideo VSA is intended for video DiT head sizes such as 64/128.
        # Keep this permissive and let the runtime fallback handle unsupported
        # cases from the installed fastvideo-kernel build.
        return []

    @staticmethod
    def get_name() -> str:
        return "FASTVIDEO_VSA"

    @staticmethod
    def get_impl_cls() -> type[FastVideoVSAImpl]:
        return FastVideoVSAImpl


class FastVideoVSAImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        qkv_layout: str | None = None,
        backend_kwargs: Mapping[str, Any] | None = None,
        **extra_impl_args,
    ) -> None:
        backend_kwargs = backend_kwargs or {}
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.qkv_layout = qkv_layout

        self.topk = int(backend_kwargs.get("topk", 64))
        self.block_size = self._parse_block_size(backend_kwargs.get("block_size", (4, 8, 8)))
        self.block_elements = self.block_size[0] * self.block_size[1] * self.block_size[2]
        self.min_seq_len = int(backend_kwargs.get("min_seq_len", self.block_elements * 2))
        self.fallback_on_error = bool(backend_kwargs.get("fallback_on_error", True))
        self.disable_when_sp_active = bool(backend_kwargs.get("disable_when_sp_active", True))

        self.sdpa_fallback = SDPAImpl(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
            qkv_layout=qkv_layout,
        )

        if self.block_elements != 256:
            logger.warning(
                "FASTVIDEO_VSA currently uses fastvideo_kernel.video_sparse_attn_bshd, "
                "which supports only 256-token blocks. Configured block_size=%s "
                "(product=%d) will fall back to SDPA.",
                self.block_size,
                self.block_elements,
            )

    @staticmethod
    def _parse_block_size(value: Any) -> tuple[int, int, int]:
        if isinstance(value, int):
            return (value, value, value)
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return (int(value[0]), int(value[1]), int(value[2]))
        raise ValueError(f"FASTVIDEO_VSA block_size must be an int or length-3 tuple/list, got {value!r}")

    def _fallback(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
        reason: str,
    ) -> torch.Tensor:
        logger.warning_once("FASTVIDEO_VSA falling back to SDPA: %s", reason)
        return self.sdpa_fallback.forward(query, key, value, attn_metadata)

    def _fallback_reason(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> str | None:
        if self.causal:
            return "causal attention is not supported"
        if self.block_elements != 256:
            return f"block_elements must be 256, got {self.block_elements}"
        if self.topk <= 0:
            return f"topk must be positive, got {self.topk}"
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            return f"expected [B, S, H, D] tensors, got {query.shape}, {key.shape}, {value.shape}"
        if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
            return "batch dimensions must match"
        if query.shape[2:] != key.shape[2:] or query.shape[2:] != value.shape[2:]:
            return "head/head_dim dimensions must match"
        if query.shape[1] != key.shape[1] or query.shape[1] != value.shape[1]:
            return "initial VSA backend supports self-attention with Sq == Skv only"
        if query.shape[1] < self.min_seq_len:
            return f"sequence length {query.shape[1]} is below min_seq_len {self.min_seq_len}"
        dit_seq_shape = _get_vsa_dit_seq_shape(attn_metadata)
        if dit_seq_shape is None:
            return "vsa_dit_seq_shape metadata is required"
        if math.prod(dit_seq_shape) != query.shape[1]:
            return f"vsa_dit_seq_shape product {math.prod(dit_seq_shape)} != sequence length {query.shape[1]}"
        num_blocks = math.prod(
            math.ceil(seq_dim / tile_dim) for seq_dim, tile_dim in zip(dit_seq_shape, self.block_size)
        )
        if self.topk > num_blocks:
            return f"topk {self.topk} > num_blocks {num_blocks}"
        if query.dtype not in (torch.float16, torch.bfloat16):
            return f"dtype {query.dtype} is not supported"
        if key.dtype != query.dtype or value.dtype != query.dtype:
            return "q/k/v dtypes must match"
        if query.device.type != "cuda" or key.device.type != "cuda" or value.device.type != "cuda":
            return "q/k/v must be CUDA tensors"
        expected_scale = self.head_size**-0.5
        if abs(float(self.softmax_scale) - float(expected_scale)) > 1e-6:
            return f"softmax_scale {self.softmax_scale} differs from FastVideo VSA scale {expected_scale}"
        if attn_metadata is not None and attn_metadata.attn_mask is not None:
            return "attention masks are not supported"
        if attn_metadata is not None and attn_metadata.full_attn_spans is not None:
            return "piecewise/full attention spans are not supported"
        if self.num_heads != self.num_kv_heads:
            return "GQA/MQA is not supported"
        if self.disable_when_sp_active and is_forward_context_available():
            ctx = get_forward_context()
            if getattr(ctx, "sp_active", False):
                return "sequence parallel context is active"
        return None

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        reason = self._fallback_reason(query, key, value, attn_metadata)
        if reason is not None:
            return self._fallback(query, key, value, attn_metadata, reason)

        seq_len = query.shape[1]
        dit_seq_shape = _get_vsa_dit_seq_shape(attn_metadata)
        assert dit_seq_shape is not None
        num_blocks = math.prod(
            math.ceil(seq_dim / tile_dim) for seq_dim, tile_dim in zip(dit_seq_shape, self.block_size)
        )
        preserve_all_blocks = _preserve_vsa_all_blocks(attn_metadata)
        use_native_sdpa = self.topk == num_blocks and not preserve_all_blocks
        route = "SDPA" if use_native_sdpa else "VSA_ALL_BLOCKS" if self.topk == num_blocks else "VSA"
        checkpoint_mode = "fastvideo_dmd" if preserve_all_blocks else "native"
        logger.info_once(
            "FASTVIDEO_VSA routing: seq_len=%d, dit_seq_shape=%s, block_size=%s, num_blocks=%d, "
            "topk=%d, keep_ratio=%.1f%%, checkpoint_mode=%s, route=%s",
            seq_len,
            dit_seq_shape,
            self.block_size,
            num_blocks,
            self.topk,
            100.0 * self.topk / num_blocks,
            checkpoint_mode,
            route,
        )
        if use_native_sdpa:
            return self._fallback(
                query,
                key,
                value,
                attn_metadata,
                f"topk {self.topk} selects all blocks for a native checkpoint",
            )

        try:
            tile_partition_indices, variable_block_sizes, non_pad_index, untile_combined_index = _get_tile_metadata(
                dit_seq_shape,
                self.block_size,
                self.block_elements,
                query.device,
            )

            padded_len = variable_block_sizes.numel() * self.block_elements
            target_shape = (query.shape[0], padded_len, query.shape[2], query.shape[3])
            query_tiled = torch.zeros(target_shape, device=query.device, dtype=query.dtype)
            key_tiled = torch.zeros_like(query_tiled)
            value_tiled = torch.zeros_like(query_tiled)
            query_tiled[:, non_pad_index] = query[:, tile_partition_indices]
            key_tiled[:, non_pad_index] = key[:, tile_partition_indices]
            value_tiled[:, non_pad_index] = value[:, tile_partition_indices]
            # Gate behavior is checkpoint-driven, not user-configured.
            # Wan VSA layers always provide a gate projection. Its zero
            # initialization makes checkpoints without gate weights sparse-only;
            # checkpoints containing to_gate_compress weights use the learned gate.
            gate_compress = _get_gate_compress(attn_metadata)
            if gate_compress is None:
                gate_compress = torch.zeros_like(query)
            elif gate_compress.shape != query.shape:
                raise ValueError(f"gate_compress shape {gate_compress.shape} must match query shape {query.shape}")
            gate_tiled = torch.zeros_like(query_tiled)
            gate_tiled[:, non_pad_index] = gate_compress[:, tile_partition_indices]
            compress_attn_weight = gate_tiled

            output = _fastvideo_vsa_bshd_op(
                query_tiled.contiguous(),
                key_tiled.contiguous(),
                value_tiled.contiguous(),
                variable_block_sizes,
                variable_block_sizes,
                compress_attn_weight,
                self.topk,
                self.block_size[0],
                self.block_size[1],
                self.block_size[2],
            )
            return output[:, untile_combined_index].contiguous()
        except Exception as exc:
            if not self.fallback_on_error:
                raise
            return self._fallback(query, key, value, attn_metadata, f"VSA kernel failed: {exc}")

    def forward_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        return self.sdpa_fallback.forward_npu(query, key, value, attn_metadata)

    def forward_xpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        return self.sdpa_fallback.forward_xpu(query, key, value, attn_metadata)

    def forward_musa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        return self.sdpa_fallback.forward_musa(query, key, value, attn_metadata)
