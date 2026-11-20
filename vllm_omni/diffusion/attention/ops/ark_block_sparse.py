# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""ARK XPU execution for explicit block-sparse attention."""

from __future__ import annotations

import functools
import importlib
import importlib.util
import inspect

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_ARK_SPARSE_MODULE = "auto_round_kernel.sparse_attention"


def _load_ark_block_sparse():
    try:
        module = importlib.import_module(_ARK_SPARSE_MODULE)
    except ImportError as exc:
        raise ImportError(
            "FASTVIDEO_VSA on XPU requires auto-round-lib with auto_round_kernel.sparse_attention.block_sparse_sdpa."
        ) from exc
    kernel = getattr(module, "block_sparse_sdpa", None)
    if not callable(kernel):
        raise ImportError(
            "The installed auto-round-lib does not expose auto_round_kernel.sparse_attention.block_sparse_sdpa."
        )
    return kernel


@functools.lru_cache(maxsize=1)
def _ark_block_sparse_parameters() -> frozenset[str]:
    return frozenset(inspect.signature(_load_ark_block_sparse()).parameters)


@functools.lru_cache(maxsize=1)
def supports_ark_q256_multirow() -> bool:
    """Return whether ARK can execute independent 64Q rows in one Q256 tile."""
    module = importlib.import_module(_ARK_SPARSE_MODULE)
    resolver = getattr(module, "_resolve_sparse_prefill_q_tile_override", None)
    if resolver is None or "allow_multi_row_tile" not in inspect.signature(resolver).parameters:
        return False
    try:
        return (
            resolver(
                head_dim=128,
                quant_block_size=64,
                q_tile_override=256,
                sparse_q_block_tokens=64,
                sparse_k_block_tokens=64,
                tensor_layout="NHD",
                allow_multi_row_tile=True,
            )
            == 256
        )
    except (TypeError, ValueError):
        return False


def supports_ark_sparse() -> bool:
    """Return whether auto-round-lib is installed without importing its extension."""
    return importlib.util.find_spec("auto_round_kernel") is not None


def require_ark_sparse(
    device: torch.device,
    dtype: torch.dtype,
    head_size: int,
    block_size: int,
    key_block_size: int | None = None,
) -> None:
    key_block_size = block_size if key_block_size is None else key_block_size
    if device.type != "xpu":
        raise ValueError("ARK block-sparse attention requires XPU tensors")
    if dtype != torch.bfloat16:
        raise ValueError("ARK block-sparse attention currently requires BF16")
    if head_size not in (64, 128):
        raise ValueError(f"ARK block-sparse attention requires head_dim 64 or 128, got {head_size}")
    if block_size not in (64, 128, 256):
        raise ValueError(f"ARK block-sparse attention requires block_size 64, 128 or 256, got {block_size}")
    if key_block_size not in (64, 128):
        raise ValueError(f"ARK block-sparse attention requires key_block_size 64 or 128, got {key_block_size}")
    if block_size == 256 and key_block_size != 64:
        raise ValueError("ARK 256Q routing requires key_block_size=64")
    _load_ark_block_sparse()


def block_map_to_lut(block_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a boolean block map as ARK's run-length LUT and valid counts."""
    if block_map.ndim != 4 or block_map.dtype != torch.bool or min(block_map.shape) < 1:
        raise ValueError("block_map must be a nonempty boolean [B,H,Qblocks,Kblocks] tensor")
    key_blocks = block_map.shape[-1]
    sentinel = key_blocks + 1
    block_ids = torch.arange(1, key_blocks + 1, device=block_map.device, dtype=torch.int32)
    positions = torch.where(
        block_map,
        block_ids.view(1, 1, 1, -1),
        torch.full_like(block_map, sentinel, dtype=torch.int32),
    )
    positions = positions.sort(dim=-1).values
    lut = positions - 1
    lut[..., 1:] = lut[..., 1:] - lut[..., :-1]
    valid_block_num = block_map.sum(dim=-1, dtype=torch.int32).contiguous()
    invalid = torch.arange(key_blocks, device=block_map.device, dtype=torch.int32) >= valid_block_num.unsqueeze(-1)
    lut = torch.where(invalid, torch.zeros_like(lut), lut)
    return lut.to(torch.int32).contiguous(), valid_block_num.to(torch.int32)


if not hasattr(torch.ops.vllm_omni, "ark_block_sparse_attn_bshd"):

    @torch.library.custom_op("vllm_omni::ark_block_sparse_attn_bshd", mutates_args=())
    def ark_block_sparse_attn_bshd(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        lut: torch.Tensor,
        valid_block_num: torch.Tensor,
        block_sizes: torch.Tensor,
        softmax_scale: float,
        q_tile_override: int,
        block_size: int,
        key_block_size: int,
    ) -> torch.Tensor:
        kernel = _load_ark_block_sparse()
        kwargs = {
            "scale": softmax_scale,
            "q_tile_override": q_tile_override,
            "sparse_q_block_tokens": block_size,
            "sparse_k_block_tokens": key_block_size,
            "tensor_layout": "NHD",
        }
        parameters = _ark_block_sparse_parameters()
        if block_sizes.numel():
            if "k_block_valid" in parameters:
                kwargs["k_block_valid"] = block_sizes
            else:
                logger.warning_once(
                    "The installed ARK block_sparse_sdpa has no k_block_valid support; "
                    "partial edge key blocks are executed with zero-padded keys."
                )
        return kernel(query, key, value, lut, valid_block_num, **kwargs)

    @ark_block_sparse_attn_bshd.register_fake
    def _(
        query,
        key,
        value,
        lut,
        valid_block_num,
        block_sizes,
        softmax_scale,
        q_tile_override,
        block_size,
        key_block_size,
    ):
        del key, value, lut, valid_block_num, block_sizes, softmax_scale, q_tile_override, block_size, key_block_size
        return torch.empty_like(query)


ark_block_sparse_attn_bshd = torch.ops.vllm_omni.ark_block_sparse_attn_bshd


def ark_block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_map: torch.Tensor,
    block_sizes: torch.Tensor | None,
    softmax_scale: float,
    block_size: int = 64,
    key_block_size: int | None = None,
) -> torch.Tensor:
    """Execute an explicit H3-style block map through ARK on XPU."""
    key_block_size = block_size if key_block_size is None else key_block_size
    if query.ndim != 4 or query.shape != key.shape or query.shape != value.shape:
        raise ValueError("ARK block-sparse attention requires matching BSHD tensors")
    if query.dtype != torch.bfloat16:
        raise ValueError("ARK block-sparse attention currently requires BF16")
    if query.shape[1] % block_size:
        raise ValueError("ARK block-sparse inputs must be padded to complete tiles")
    query_blocks = query.shape[1] // block_size
    if block_map.ndim != 4 or block_map.shape[:3] != (query.shape[0], query.shape[2], query_blocks):
        raise ValueError("ARK block_map must be boolean [B,H,Qblocks,Kblocks]")
    if block_map.dtype != torch.bool:
        raise ValueError("ARK block_map must be boolean")
    if block_size % key_block_size:
        raise ValueError(f"key_block_size {key_block_size} must divide block_size {block_size}")
    if key.shape[1] % key_block_size:
        raise ValueError("ARK block-sparse keys must be padded to complete key blocks")
    logical_key_blocks = key.shape[1] // key_block_size
    if block_map.shape[-1] != logical_key_blocks:
        raise ValueError(
            "ARK block_map key dimension must match the key block count; "
            f"got {block_map.shape[-1]} columns for {logical_key_blocks} key blocks"
        )
    key_blocks = block_map.shape[-1]
    if block_size == 128:
        from .block_sparse import split_tile128_map

        if block_sizes is None:
            block_sizes = torch.full((key_blocks,), block_size, dtype=torch.int32, device=query.device)
        block_map, block_sizes = split_tile128_map(block_map, block_sizes.to(torch.int32))
        block_size = 64
        key_block_size = 64
    if block_sizes is not None and block_sizes.numel() != block_map.shape[-1]:
        raise ValueError(
            "ARK block_sizes must provide one valid-token count per key block; "
            f"got {block_sizes.numel()} for {block_map.shape[-1]} key blocks"
        )
    lut, valid_block_num = block_map_to_lut(block_map)
    if block_sizes is None:
        block_sizes = torch.empty(0, dtype=torch.int32, device=query.device)
    else:
        block_sizes = block_sizes.to(torch.int32).contiguous()
    q_tile_override = 256 if block_size == 256 or supports_ark_q256_multirow() else 64
    return ark_block_sparse_attn_bshd(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        lut,
        valid_block_num,
        block_sizes,
        float(softmax_scale),
        q_tile_override,
        block_size,
        key_block_size,
    )
