from __future__ import annotations

import types
from collections import Counter
from collections.abc import Callable
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.attention.fish_kvcache_attn import (
    FISH_KVCACHE_SMALL_PATH_MAX_SEQ_LEN,
    can_use_fish_kvcache_attn,
    fish_decode_kvcache_attn,
    is_available,
    is_fish_kvcache_attn_enabled,
    is_fish_kvcache_attn_required,
    load_error,
    prewarm_fish_kvcache_attn_workspace,
)

logger = init_logger(__name__)

_FISH_SLOW_AR_ARCH = "FishSpeechSlowARForConditionalGeneration"
_FIRST_SMALL_HIT_LOGGED = False
_FIRST_LONG_HIT_LOGGED = False
_FIRST_MISS_LOGGED = False
_SMALL_HIT_COUNT = 0
_LONG_HIT_COUNT = 0
_FALLBACK_COUNTS: Counter[str] = Counter()


def _fish_kvcache_enabled() -> bool:
    return is_fish_kvcache_attn_enabled()


def is_fish_kvcache_attn_active_for_model(model_config: Any) -> bool:
    return getattr(model_config, "model_arch", None) == _FISH_SLOW_AR_ARCH and is_fish_kvcache_attn_enabled()


def reset_fish_kvcache_attn_stats() -> None:
    global _LONG_HIT_COUNT, _SMALL_HIT_COUNT
    _SMALL_HIT_COUNT = 0
    _LONG_HIT_COUNT = 0
    _FALLBACK_COUNTS.clear()


def get_fish_kvcache_attn_stats() -> dict[str, Any]:
    return {
        "small_hit_count": _SMALL_HIT_COUNT,
        "long_hit_count": _LONG_HIT_COUNT,
        "fallback_count_by_reason": dict(_FALLBACK_COUNTS),
    }


def _record_fallback(reason: str) -> None:
    _FALLBACK_COUNTS[reason] += 1


def _small_decode_bucket(max_seq_len: int) -> int:
    if max_seq_len <= 64:
        return 64
    if max_seq_len <= 128:
        return 128
    if max_seq_len <= 256:
        return 256
    if max_seq_len <= 512:
        return 512
    return FISH_KVCACHE_SMALL_PATH_MAX_SEQ_LEN


def prewarm_fish_kvcache_attn_capture_workspaces(
    *,
    model_config: Any,
    device: torch.device,
    dtype: torch.dtype,
    capture_sizes: list[int] | tuple[int, ...],
) -> int:
    """Preallocate Fish attention workspaces used during CUDA graph capture."""
    if not _fish_kvcache_enabled() or not is_available():
        return 0
    if not is_fish_kvcache_attn_active_for_model(model_config):
        return 0

    text_config = model_config.hf_config.text_config
    num_heads = int(text_config.num_attention_heads)
    head_dim = int(text_config.head_dim)
    max_seq_len = int(model_config.max_model_len)
    sizes = sorted({int(size) for size in capture_sizes if int(size) > 0})
    if not sizes:
        return 0

    # The small path uses an empty workspace keyed only by device. Prewarm it
    # once, then prewarm every CUDA graph capture batch size for the long path.
    small_query = torch.empty((1, num_heads, head_dim), device=device, dtype=dtype)
    prewarm_fish_kvcache_attn_workspace(small_query, 1)
    prewarmed = 1

    for batch_size in sizes:
        query = torch.empty((batch_size, num_heads, head_dim), device=device, dtype=dtype)
        prewarm_fish_kvcache_attn_workspace(query, max_seq_len)
        prewarmed += 1

    logger.info(
        "Prewarmed Fish kvcache attention workspaces for capture sizes %s (num_heads=%d head_dim=%d max_seq_len=%d)",
        sizes,
        num_heads,
        head_dim,
        max_seq_len,
    )
    return prewarmed


def _attach_seq_lens_cpu_upper_bound(attn_metadata: Any, seq_lens_cpu_upper_bound: torch.Tensor) -> None:
    if attn_metadata is None:
        return
    if isinstance(attn_metadata, dict):
        for value in attn_metadata.values():
            _attach_seq_lens_cpu_upper_bound(value, seq_lens_cpu_upper_bound)
        return
    if isinstance(attn_metadata, (list, tuple)):
        for value in attn_metadata:
            _attach_seq_lens_cpu_upper_bound(value, seq_lens_cpu_upper_bound)
        return
    setattr(attn_metadata, "seq_lens_cpu_upper_bound", seq_lens_cpu_upper_bound)


def _graph_capture_seq_len_upper_bound(
    *,
    model_config: Any,
    max_query_len: int,
    for_cudagraph_capture: bool,
) -> int | None:
    if not for_cudagraph_capture or int(max_query_len) != 1:
        return None
    max_model_len = int(getattr(model_config, "max_model_len", 0))
    return max(max_model_len, FISH_KVCACHE_SMALL_PATH_MAX_SEQ_LEN)


def maybe_attach_fish_kvcache_seq_lens_upper_bound(
    *,
    model_config: Any,
    attn_metadata: Any,
    input_batch: Any,
    optimistic_seq_lens_cpu: torch.Tensor,
    num_reqs: int,
    num_reqs_padded: int,
    max_query_len: int,
    pad_attn: bool,
    for_cudagraph_capture: bool = False,
    num_scheduled_tokens_np: Any = None,
) -> None:
    """Attach CPU seq-len upper bounds for the Fish-only Triton decode fastpath."""
    if not is_fish_kvcache_attn_active_for_model(model_config) or int(max_query_len) != 1:
        return

    upper_bound_rows = int(num_reqs_padded if pad_attn else num_reqs)
    capture_upper_bound = _graph_capture_seq_len_upper_bound(
        model_config=model_config,
        max_query_len=max_query_len,
        for_cudagraph_capture=for_cudagraph_capture,
    )
    if capture_upper_bound is not None:
        optimistic_seq_lens_cpu[:upper_bound_rows].fill_(capture_upper_bound)
    else:
        if num_scheduled_tokens_np is None:
            return
        seq_lens_cpu_upper_bound = torch.as_tensor(
            input_batch.num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens_np,
            dtype=torch.int32,
            device="cpu",
        )
        optimistic_seq_lens_cpu[:num_reqs] = seq_lens_cpu_upper_bound
        if upper_bound_rows > num_reqs:
            optimistic_seq_lens_cpu[num_reqs:upper_bound_rows].fill_(1)
    _attach_seq_lens_cpu_upper_bound(
        attn_metadata,
        optimistic_seq_lens_cpu[:upper_bound_rows],
    )


def _dispatch_max_seq_len(attn_metadata: Any, active_batch: int, seq_lens: Any) -> int | None:
    max_seq_len = int(attn_metadata.max_seq_len)
    if int(attn_metadata.max_query_len) != 1:
        return max_seq_len

    upper_bound = getattr(attn_metadata, "seq_lens_cpu_upper_bound", None)
    if upper_bound is None:
        if is_fish_kvcache_attn_required():
            raise RuntimeError("Fish kvcache attention requires seq_lens_cpu_upper_bound for decode dispatch")
        return None
    if not isinstance(upper_bound, torch.Tensor) or upper_bound.device.type != "cpu":
        if is_fish_kvcache_attn_required():
            raise RuntimeError("Fish kvcache attention seq_lens_cpu_upper_bound must be a CPU tensor")
        return None
    if int(upper_bound.shape[0]) < active_batch:
        if is_fish_kvcache_attn_required():
            raise RuntimeError(
                "Fish kvcache attention seq_lens_cpu_upper_bound does not cover the active decode batch: "
                f"upper_bound_rows={int(upper_bound.shape[0])}, active_batch={active_batch}"
            )
        return None
    upper_bound = upper_bound[:active_batch]
    real_upper_bound = max(1, int(upper_bound.max().item()))
    if is_fish_kvcache_attn_required():
        real_max_seq_len = int(seq_lens.max().item())
        if real_upper_bound < real_max_seq_len:
            raise RuntimeError(
                "Fish kvcache attention seq_lens_cpu_upper_bound "
                f"underestimates real seq_lens: upper_bound={real_upper_bound}, "
                f"real={real_max_seq_len}"
            )
    if real_upper_bound <= FISH_KVCACHE_SMALL_PATH_MAX_SEQ_LEN:
        return _small_decode_bucket(real_upper_bound)
    return real_upper_bound


def _forward_with_fish_kvcache(
    impl: Any,
    original_forward: Callable[..., Any],
    layer: Any,
    query: Any,
    key: Any,
    value: Any,
    kv_cache: Any,
    attn_metadata: Any,
    output: Any,
    output_scale: Any = None,
    output_block_scale: Any = None,
) -> Any:
    global _FIRST_LONG_HIT_LOGGED, _FIRST_MISS_LOGGED, _FIRST_SMALL_HIT_LOGGED
    global _LONG_HIT_COUNT, _SMALL_HIT_COUNT

    if attn_metadata is not None and not attn_metadata.use_cascade:
        num_actual_tokens = attn_metadata.num_actual_tokens
        key_cache, value_cache = kv_cache.unbind(0)
        q = query[:num_actual_tokens]
        out = output[:num_actual_tokens]
        active_batch = int(q.shape[0])
        block_table = attn_metadata.block_table[:active_batch] if attn_metadata.block_table is not None else None
        seq_lens = attn_metadata.seq_lens[:active_batch]
        dispatch_max_seq_len = _dispatch_max_seq_len(attn_metadata, active_batch, seq_lens)
        can_use = dispatch_max_seq_len is not None and can_use_fish_kvcache_attn(
            query=q,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            max_query_len=attn_metadata.max_query_len,
            max_seq_len=dispatch_max_seq_len,
            dcp_world_size=impl.dcp_world_size,
            use_cascade=attn_metadata.use_cascade,
            alibi_slopes=impl.alibi_slopes,
            sliding_window=impl.sliding_window,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )
        if can_use:
            is_long_path = dispatch_max_seq_len > FISH_KVCACHE_SMALL_PATH_MAX_SEQ_LEN
            if is_long_path:
                _LONG_HIT_COUNT += 1
            else:
                _SMALL_HIT_COUNT += 1
            should_log_hit = (is_long_path and not _FIRST_LONG_HIT_LOGGED) or (
                not is_long_path and not _FIRST_SMALL_HIT_LOGGED
            )
            if should_log_hit:
                if is_long_path:
                    _FIRST_LONG_HIT_LOGGED = True
                    path_name = "long"
                else:
                    _FIRST_SMALL_HIT_LOGGED = True
                    path_name = "small"
                logger.info(
                    "Fish decode-only kvcache attention fast path hit: "
                    "query_shape=%s key_cache_shape=%s block_table_shape=%s "
                    "seq_lens_shape=%s max_seq_len=%s path=%s",
                    tuple(q.shape),
                    tuple(key_cache.shape),
                    tuple(attn_metadata.block_table.shape),
                    tuple(attn_metadata.seq_lens.shape),
                    dispatch_max_seq_len,
                    path_name,
                )
            fish_decode_kvcache_attn(
                q,
                key_cache,
                value_cache,
                block_table,
                seq_lens,
                out,
                scale=float(impl.scale),
                max_seq_len=dispatch_max_seq_len,
            )
            return output

        is_decode_request = int(attn_metadata.max_query_len) == 1
        if is_decode_request and dispatch_max_seq_len is None:
            _record_fallback("missing_cpu_upper_bound")
        elif is_decode_request:
            _record_fallback("guard_miss")
        else:
            _record_fallback("non_decode")
        if is_decode_request and is_fish_kvcache_attn_required():
            raise RuntimeError(
                "Fish kvcache attention is required but guard rejected the request: "
                f"query_shape={tuple(q.shape)} query_dtype={q.dtype} "
                f"max_query_len={attn_metadata.max_query_len} max_seq_len={dispatch_max_seq_len} "
                f"dcp_world_size={impl.dcp_world_size} use_cascade={attn_metadata.use_cascade}"
            )

        if not _FIRST_MISS_LOGGED:
            _FIRST_MISS_LOGGED = True
            logger.info(
                "Fish decode-only kvcache attention fast path miss: "
                "query_shape=%s query_dtype=%s key_cache_shape=%s key_dtype=%s "
                "block_table_shape=%s block_dtype=%s seq_lens_shape=%s seq_dtype=%s "
                "max_query_len=%s max_seq_len=%s dcp_world_size=%s use_cascade=%s",
                tuple(q.shape),
                q.dtype,
                tuple(key_cache.shape),
                key_cache.dtype,
                tuple(block_table.shape) if block_table is not None else None,
                getattr(block_table, "dtype", None),
                tuple(seq_lens.shape),
                seq_lens.dtype,
                attn_metadata.max_query_len,
                dispatch_max_seq_len,
                impl.dcp_world_size,
                attn_metadata.use_cascade,
            )

    return original_forward(
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=output_scale,
        output_block_scale=output_block_scale,
    )


def install_fish_kvcache_attn_backend(model: Any) -> int:
    """Install the Fish kvcache fast path on this Fish SlowAR model only."""
    if not _fish_kvcache_enabled():
        return 0
    if not is_available():
        _record_fallback("implementation_unavailable")
        if is_fish_kvcache_attn_required():
            raise RuntimeError(
                f"VLLM_OMNI_FISH_KVCACHE_ATTN=required but Fish kvcache attention is unavailable: {load_error()!r}"
            )
        logger.warning(
            "Fish kvcache attention is enabled but unavailable: %r",
            load_error(),
        )
        return 0

    installed = 0
    for layer in model.layers:
        impl = layer.self_attn.attn.impl
        if getattr(impl, "_fish_kvcache_attn_installed", False):
            continue

        original_forward = impl.forward

        def fish_forward(
            this_impl: Any,
            layer: Any,
            query: Any,
            key: Any,
            value: Any,
            kv_cache: Any,
            attn_metadata: Any,
            output: Any,
            output_scale: Any = None,
            output_block_scale: Any = None,
            *,
            _original_forward: Callable[..., Any] = original_forward,
        ) -> Any:
            return _forward_with_fish_kvcache(
                this_impl,
                _original_forward,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )

        impl.forward = types.MethodType(fish_forward, impl)
        impl._fish_kvcache_attn_installed = True
        installed += 1

    if installed:
        logger.info("Installed Fish decode-only kvcache attention backend on %d attention layers", installed)
    elif is_fish_kvcache_attn_required():
        raise RuntimeError("Fish kvcache attention is required but installed 0 attention layers")
    return installed
