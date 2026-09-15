# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Physical KV cache layout selection and propagation for the Diffusion cache."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.config import CacheConfig
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_omni.diffusion.diffusion_kv.layout import (
    DEFAULT_DIFFUSION_KV_CACHE_LAYOUT,
    adopt_kv_cache_layout,
    assert_backend_layout_supported,
    build_kv_cache_tensor,
    resolve_diffusion_kv_cache_layout,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _config():
    return SimpleNamespace(cache_config=CacheConfig())


class _DenseBackend:
    @classmethod
    def indexes_kv_by_block_stride(cls) -> bool:
        return False


class _BlockStrideBackend:
    @classmethod
    def indexes_kv_by_block_stride(cls) -> bool:
        return True


def test_default_layout_is_not_block_outermost() -> None:
    """The default reproduces the pre-0.29 indexes_kv_by_block_stride=False."""
    vllm_config = _config()
    layout = resolve_diffusion_kv_cache_layout(vllm_config)

    assert layout is DEFAULT_DIFFUSION_KV_CACHE_LAYOUT
    assert layout.is_block_outermost is False
    assert vllm_config.cache_config.get_resolved_kv_cache_layout() is layout


def test_block_stride_backend_selects_block_outermost_layout() -> None:
    layout = resolve_diffusion_kv_cache_layout(_config(), indexes_kv_by_block_stride=True)

    assert layout.is_block_outermost is True


def test_resolution_is_idempotent() -> None:
    vllm_config = _config()

    assert resolve_diffusion_kv_cache_layout(vllm_config) is resolve_diffusion_kv_cache_layout(vllm_config)


def test_contradicting_requirement_raises_instead_of_misaddressing() -> None:
    vllm_config = _config()
    resolve_diffusion_kv_cache_layout(vllm_config)

    with pytest.raises(ValueError, match="contradicts the attention"):
        resolve_diffusion_kv_cache_layout(vllm_config, indexes_kv_by_block_stride=True)


def test_preset_layout_is_honoured() -> None:
    vllm_config = _config()
    vllm_config.cache_config.kv_cache_layout = KVCacheLayout.BLNHC.name

    layout = resolve_diffusion_kv_cache_layout(vllm_config, indexes_kv_by_block_stride=True)

    assert layout is KVCacheLayout.BLNHC


def test_backend_assertion_is_quiet_until_the_layout_is_resolved() -> None:
    """Workers collect specs before the control plane hands them the layout."""
    vllm_config = _config()

    assert_backend_layout_supported(vllm_config, _BlockStrideBackend)  # unresolved: nothing to check

    resolve_diffusion_kv_cache_layout(vllm_config)
    assert_backend_layout_supported(vllm_config, _DenseBackend)
    assert_backend_layout_supported(vllm_config, None)
    with pytest.raises(ValueError, match="contradicts the attention"):
        assert_backend_layout_supported(vllm_config, _BlockStrideBackend)


def test_worker_adopts_the_layout_carried_on_the_kv_cache_config() -> None:
    """Diffusion workers receive only a KVCacheConfig, so the name rides on it."""
    worker = _config()
    shipped = SimpleNamespace(kv_cache_layout=KVCacheLayout.BLNHC.name)

    assert adopt_kv_cache_layout(worker, shipped) is KVCacheLayout.BLNHC
    assert worker.cache_config.get_resolved_kv_cache_layout() is KVCacheLayout.BLNHC


def test_worker_without_a_carried_layout_falls_back_to_the_same_default() -> None:
    worker = _config()

    assert adopt_kv_cache_layout(worker, SimpleNamespace(kv_cache_layout=None)) is DEFAULT_DIFFUSION_KV_CACHE_LAYOUT


def test_worker_disagreeing_with_the_control_plane_raises() -> None:
    worker = _config()
    resolve_diffusion_kv_cache_layout(worker)

    with pytest.raises(ValueError, match="disagrees with the layout"):
        adopt_kv_cache_layout(worker, SimpleNamespace(kv_cache_layout=KVCacheLayout.BLNHC.name))


def test_block_stride_requirement_propagates_control_plane_to_worker() -> None:
    """A block-stride backend must reach the worker as a block-outermost layout.

    Covers the whole seam: the control plane resolves with the requirement, the
    name rides on each rank-local KVCacheConfig, the worker adopts it, and the
    backend contract check then passes instead of raising.
    """
    control_plane = _config()
    layout = resolve_diffusion_kv_cache_layout(control_plane, indexes_kv_by_block_stride=True)
    assert layout.is_block_outermost is True

    shipped = SimpleNamespace(kv_cache_layout=layout.name)

    worker = _config()
    assert adopt_kv_cache_layout(worker, shipped) is layout
    # The worker's own backend requirement now agrees with what it was handed.
    assert_backend_layout_supported(worker, _BlockStrideBackend)
    with pytest.raises(ValueError, match="contradicts the attention"):
        assert_backend_layout_supported(worker, _DenseBackend)


def test_tensor_strides_come_from_the_upstream_derivation() -> None:
    spec = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=8, dtype=torch.bfloat16)

    tensor = build_kv_cache_tensor(spec, 4, ["layer0", "layer1"])

    assert tensor.layers == ["layer0", "layer1"]
    assert tensor.size == spec.page_size_bytes * 2 * 4
    # Layer-outermost: each layer owns a contiguous run of its own blocks.
    assert tensor.layer_stride == spec.page_size_bytes * 4
    assert tensor.block_stride == spec.page_size_bytes
