# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Physical KV cache layout selection for the native Diffusion paged cache.

vLLM 0.29 removed ``AttentionSpec.indexes_kv_by_block_stride`` and replaced it
with a single physical layout per model, resolved once and recorded on
``CacheConfig.kv_cache_layout``.  ``KVCacheLayout.is_block_outermost`` is the
direct successor of the old per-spec flag: upstream derives
``interleaved_block_stride`` from it when it lays out ``KVCacheTensor`` regions
(``vllm/v1/core/kv_cache_utils.py``).

Diffusion never reaches vLLM's engine-core resolution.  Its backends subclass
Omni's own ``AttentionBackend`` ABC, which upstream's
``get_supported_kv_cache_layouts`` never sees, so the layout is pinned here
before ``get_kv_cache_configs`` runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.v1.kv_cache_interface import KVCacheSpec, KVCacheTensor, compute_layout_strides
from vllm.v1.kv_cache_layout import KVCacheLayout

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Layer-outermost, matching the head of upstream's ``_DEFAULT_LAYOUT_PREFERENCE``
# and reproducing the ``indexes_kv_by_block_stride=False`` that every dense
# diffusion backend declared before 0.29.
DEFAULT_DIFFUSION_KV_CACHE_LAYOUT = KVCacheLayout.LBNHC
# Block-outermost counterpart, for a paged backend whose kernel really does read
# K/V pages by the runtime block stride.
BLOCK_STRIDE_DIFFUSION_KV_CACHE_LAYOUT = KVCacheLayout.BLNHC


def resolve_diffusion_kv_cache_layout(
    vllm_config: VllmConfig,
    *,
    indexes_kv_by_block_stride: bool = False,
) -> KVCacheLayout:
    """Pin the physical KV layout for the Diffusion cache and return it.

    A layout already present on the config wins, mirroring upstream's
    ``resolve_kv_cache_layout``; it is validated against the requirement rather
    than silently overwritten, because an inconsistent layout reads K/V
    from the wrong offsets instead of raising.
    """

    required_block_outermost = bool(indexes_kv_by_block_stride)
    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is None:
        # A config double with nowhere to record the choice; report what the
        # requirement implies without pretending it was pinned.
        return BLOCK_STRIDE_DIFFUSION_KV_CACHE_LAYOUT if required_block_outermost else DEFAULT_DIFFUSION_KV_CACHE_LAYOUT

    current = getattr(cache_config, "kv_cache_layout", None)
    if current is not None:
        layout = KVCacheLayout[current]
        if layout.is_block_outermost != required_block_outermost:
            raise ValueError(
                "Diffusion KV cache layout "
                f"{layout.name} (is_block_outermost="
                f"{layout.is_block_outermost}) contradicts the attention "
                "backend, which requires is_block_outermost="
                f"{required_block_outermost}."
            )
        return layout

    layout = BLOCK_STRIDE_DIFFUSION_KV_CACHE_LAYOUT if required_block_outermost else DEFAULT_DIFFUSION_KV_CACHE_LAYOUT
    cache_config.kv_cache_layout = layout.name
    return layout


def adopt_kv_cache_layout(vllm_config: VllmConfig, kv_cache_config) -> KVCacheLayout:
    """Adopt the control plane's resolved layout inside a worker.

    The engine core resolves the layout once and every worker adopts it before
    allocating (``CacheConfig.kv_cache_layout``).  Diffusion workers receive only
    a ``KVCacheConfig``, so the name rides along on ``kv_cache_config`` and is
    copied onto the worker's own config here.  Without this, anything calling
    ``get_resolved_kv_cache_layout`` in a worker -- ``init_kv_cache`` does --
    raises.
    """

    name = getattr(kv_cache_config, "kv_cache_layout", None)
    if name is None:
        # Older configs, or a standalone harness: fall back to the same default
        # the control plane pins so both sides still agree.
        return resolve_diffusion_kv_cache_layout(vllm_config)

    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is None:
        return KVCacheLayout[name]

    current = getattr(cache_config, "kv_cache_layout", None)
    if current is None:
        cache_config.kv_cache_layout = name
    elif current != name:
        raise ValueError(
            f"Worker KV cache layout {current!r} disagrees with the layout resolved by the control plane ({name!r})."
        )
    return KVCacheLayout[name]


def build_kv_cache_tensor(
    spec: KVCacheSpec,
    num_blocks: int,
    layers: Sequence[str],
    *,
    layout: KVCacheLayout | None = None,
    offset: int = 0,
) -> KVCacheTensor:
    """A ``KVCacheTensor`` covering ``layers`` for ``num_blocks`` blocks.

    Before 0.29 a tensor named the layers sharing it
    (``KVCacheTensor(size=..., shared_by=[...])``); 0.29 replaced that with an
    explicit layout, so layer ``l``'s block ``b`` starts at
    ``offset + l * layer_stride + b * block_stride``.  Note this lays the layers
    out at distinct offsets rather than aliasing them, which is what upstream
    now does for a single cache group.

    The strides come from upstream's own ``compute_layout_strides``, called the
    way ``vllm.v1.core.kv_cache_utils`` calls it, because hand-computed strides
    read K/V from the wrong offsets silently instead of raising.
    """

    layout = layout or DEFAULT_DIFFUSION_KV_CACHE_LAYOUT
    layer_names = list(layers)
    # Mirrors kv_cache_utils: the block stride is pinned up front only for
    # block-outermost layouts, where every block repeats the same packing.
    bytes_per_block = spec.page_size_bytes * len(layer_names)
    interleaved_block_stride = bytes_per_block if layout.is_block_outermost else None
    layer_stride, block_stride, _, _, _ = compute_layout_strides(
        spec,
        num_blocks,
        len(layer_names),
        layout,
        fixed_strides=(None, interleaved_block_stride, None, None, None),
    )
    return KVCacheTensor(
        size=bytes_per_block * num_blocks,
        layers=layer_names,
        layer_stride=layer_stride,
        block_stride=block_stride,
        offset=offset,
    )


def assert_backend_layout_supported(vllm_config: VllmConfig, attn_backend: type | None) -> None:
    """Fail loudly when a backend's block-stride need contradicts the layout.

    Skipped while the layout is still unresolved -- worker processes collect
    specs before the control plane hands them the resolved name -- so this
    enforces the contract wherever it is knowable and never crashes a rank that
    has simply not been told yet.
    """

    if attn_backend is None:
        return
    # Spec discovery runs against lightweight config doubles in tests and against
    # worker configs that have not been handed the resolved name yet; in both
    # cases there is simply nothing to check.
    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is None or getattr(cache_config, "kv_cache_layout", None) is None:
        return
    indexes_by_block_stride = getattr(attn_backend, "indexes_kv_by_block_stride", None)
    if indexes_by_block_stride is None:
        return
    resolve_diffusion_kv_cache_layout(
        vllm_config,
        indexes_kv_by_block_stride=indexes_by_block_stride(),
    )
