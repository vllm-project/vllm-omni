from __future__ import annotations

import os
import threading
from typing import Any

import torch

FISH_KVCACHE_LONG_SPLIT_TOKENS = 1024
FISH_KVCACHE_SMALL_PATH_MAX_SEQ_LEN = 1024

_FISH_KVCACHE_ATTN_ENV = "VLLM_OMNI_FISH_KVCACHE_ATTN"
_ENABLED_VALUES = frozenset({"1", "true", "yes", "on", "required"})
_DISABLED_VALUES = frozenset({"0", "false", "no", "off", "disabled", "disable"})
_REQUIRED_VALUES = frozenset({"required"})
_WORKSPACE_CACHE: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_WORKSPACE_CACHE_LOCK = threading.Lock()


def _triton_backend() -> Any:
    from vllm_omni.attention import fish_kvcache_triton

    return fish_kvcache_triton


def is_available() -> bool:
    return _triton_backend().is_available()


def load_error() -> Exception | None:
    return _triton_backend().load_error()


def is_fish_kvcache_attn_enabled() -> bool:
    value = os.environ.get(_FISH_KVCACHE_ATTN_ENV)
    if value is None:
        return True
    value = value.lower()
    if value in _DISABLED_VALUES:
        return False
    return value in _ENABLED_VALUES or value == ""


def is_fish_kvcache_attn_required() -> bool:
    return os.environ.get(_FISH_KVCACHE_ATTN_ENV, "").lower() in _REQUIRED_VALUES


def _is_sliding_window_disabled(sliding_window: Any) -> bool:
    if sliding_window is None:
        return True
    if isinstance(sliding_window, (list, tuple)) and len(sliding_window) == 2:
        return int(sliding_window[0]) == -1 and int(sliding_window[1]) == -1
    return False


def can_use_fish_kvcache_attn(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor | None,
    seq_lens: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    dcp_world_size: int,
    use_cascade: bool,
    alibi_slopes: Any,
    sliding_window: Any,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
) -> bool:
    if not is_fish_kvcache_attn_enabled():
        return False
    if not is_available():
        return False
    if max_query_len != 1 or use_cascade or dcp_world_size != 1:
        return False
    if block_table is None or alibi_slopes is not None or not _is_sliding_window_disabled(sliding_window):
        return False
    if output_scale is not None or output_block_scale is not None:
        return False
    if query.dim() != 3 or key_cache.dim() != 4 or value_cache.dim() != 4:
        return False
    if block_table.dim() != 2 or seq_lens.dim() != 1:
        return False
    if block_table.shape[0] != query.shape[0] or seq_lens.shape[0] != query.shape[0]:
        return False
    if query.shape[-1] != 128 or key_cache.shape[-1] != 128:
        return False
    if key_cache.shape[1] != 16:
        return False
    if query.dtype not in (torch.float16, torch.bfloat16):
        return False
    if key_cache.dtype != query.dtype or value_cache.dtype != query.dtype:
        return False
    if block_table.dtype != torch.int32 or seq_lens.dtype != torch.int32:
        return False
    if max_seq_len <= 0:
        return False
    if max_seq_len > block_table.shape[1] * key_cache.shape[1]:
        return False
    if not (
        query.is_contiguous()
        and key_cache.is_contiguous()
        and value_cache.is_contiguous()
        and block_table.is_contiguous()
        and seq_lens.is_contiguous()
    ):
        return False
    return True


def _is_cuda_graph_capturing() -> bool:
    if not torch.cuda.is_available():
        return False
    return bool(torch.cuda.is_current_stream_capturing())


def _raise_workspace_capture_miss() -> None:
    raise RuntimeError("Fish kvcache attention workspace was not prewarmed before CUDA graph capture")


def _get_decode_workspace(
    query: torch.Tensor,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, num_q_heads, head_dim = query.shape
    if int(max_seq_len) <= FISH_KVCACHE_SMALL_PATH_MAX_SEQ_LEN:
        key = (query.device.type, query.device.index, "empty")
        partial_shapes = ((0,), (0,), (0,))
    else:
        num_splits = (int(max_seq_len) + FISH_KVCACHE_LONG_SPLIT_TOKENS - 1) // FISH_KVCACHE_LONG_SPLIT_TOKENS
        key = (
            query.device.type,
            query.device.index,
            int(batch_size),
            int(num_q_heads),
            int(head_dim),
            int(num_splits),
        )
        partial_shapes = (
            (num_splits, batch_size, num_q_heads),
            (num_splits, batch_size, num_q_heads),
            (num_splits, batch_size, num_q_heads, head_dim),
        )
    with _WORKSPACE_CACHE_LOCK:
        workspace = _WORKSPACE_CACHE.get(key)
        if workspace is None:
            if _is_cuda_graph_capturing():
                _raise_workspace_capture_miss()
            workspace = (
                torch.empty(partial_shapes[0], device=query.device, dtype=torch.float32),
                torch.empty(partial_shapes[1], device=query.device, dtype=torch.float32),
                torch.empty(partial_shapes[2], device=query.device, dtype=torch.float32),
            )
            _WORKSPACE_CACHE[key] = workspace
        return workspace


def prewarm_fish_kvcache_attn_workspace(
    query: torch.Tensor,
    max_seq_len: int,
) -> None:
    _get_decode_workspace(query, int(max_seq_len))


def fish_decode_kvcache_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    *,
    scale: float,
    max_seq_len: int,
) -> torch.Tensor:
    partial_m, partial_l, partial_acc = _get_decode_workspace(query, int(max_seq_len))
    return _triton_backend().fish_decode_kvcache_attn_triton(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        out,
        scale=float(scale),
        max_seq_len=int(max_seq_len),
        small_path_max_seq_len=FISH_KVCACHE_SMALL_PATH_MAX_SEQ_LEN,
        long_split_tokens=FISH_KVCACHE_LONG_SPLIT_TOKENS,
        partial_m=partial_m,
        partial_l=partial_l,
        partial_acc=partial_acc,
    )
