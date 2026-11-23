# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
import types
from types import SimpleNamespace

import pytest
import torch
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability

from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.selector import _cached_get_backend_cls
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import AttentionConfig
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.envs import PACKAGES_CHECKER

# Importing CudaOmniPlatform pulls in vllm.platforms.cuda, which imports the
# CUDA-only ``vllm._C_stable_libtorch`` extension at module top level. That
# extension is absent on non-CUDA builds (e.g. Intel/XPU), so skip the whole
# module there before the import can crash collection. ``is_cuda()`` resolves
# the platform without importing cuda.py.
if not current_platform.is_cuda():
    pytest.skip("CUDA-only diffusion backend selection tests", allow_module_level=True)

from vllm_omni.platforms.cuda.platform import CudaOmniPlatform  # noqa: E402

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _blackwell_sm120(monkeypatch: pytest.MonkeyPatch, *, cudnn_version: int = 90500) -> None:
    monkeypatch.setattr(
        CudaOmniPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(12, 0)),
    )
    monkeypatch.setattr(PACKAGES_CHECKER, "get_packages_info", lambda: {"has_flash_attn": False})
    monkeypatch.setattr(CudaOmniPlatform, "has_flash_attn_4", classmethod(lambda cls: False))
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: cudnn_version)


def _install_dummy_flashinfer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Top-level ``flashinfer`` only — no ``prefill`` wrapper (partial install)."""
    monkeypatch.setitem(sys.modules, "flashinfer", types.ModuleType("flashinfer"))
    # A previously imported real submodule must not satisfy the wrapper probe.
    monkeypatch.delitem(sys.modules, "flashinfer.prefill", raising=False)


def _install_dummy_flashinfer_prefill(monkeypatch: pytest.MonkeyPatch) -> None:
    """FlashInfer stub that exposes the prefill wrapper FLASHINFER_ATTN needs."""
    prefill = types.ModuleType("flashinfer.prefill")
    setattr(prefill, "BatchPrefillWithRaggedKVCacheWrapper", type("BatchPrefillWithRaggedKVCacheWrapper", (), {}))
    pkg = types.ModuleType("flashinfer")
    setattr(pkg, "prefill", prefill)
    monkeypatch.setitem(sys.modules, "flashinfer", pkg)
    monkeypatch.setitem(sys.modules, "flashinfer.prefill", prefill)


def test_auto_selects_cudnn_for_supported_blackwell_head_size(monkeypatch: pytest.MonkeyPatch):
    _blackwell_sm120(monkeypatch)

    path = CudaOmniPlatform.get_diffusion_attn_backend_cls(None, head_size=128)

    assert path == DiffusionAttentionBackendEnum.CUDNN_ATTN.get_path()


def test_auto_routes_incompatible_head_size_to_sdpa(monkeypatch: pytest.MonkeyPatch):
    _blackwell_sm120(monkeypatch)
    _install_dummy_flashinfer(monkeypatch)

    # 320 is outside cuDNN FMHA (<=256, multiple of 8) and FlashInfer {64,128,256}.
    path = CudaOmniPlatform.get_diffusion_attn_backend_cls(None, head_size=320)

    assert path == DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()


def test_auto_routes_non_multiple_head_size_to_sdpa_without_flashinfer(monkeypatch: pytest.MonkeyPatch):
    _blackwell_sm120(monkeypatch)
    monkeypatch.setitem(sys.modules, "flashinfer", None)

    path = CudaOmniPlatform.get_diffusion_attn_backend_cls(None, head_size=12)

    assert path == DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()


def test_auto_selects_flashinfer_when_cudnn_too_old(monkeypatch: pytest.MonkeyPatch):
    _blackwell_sm120(monkeypatch, cudnn_version=90499)
    _install_dummy_flashinfer_prefill(monkeypatch)

    path = CudaOmniPlatform.get_diffusion_attn_backend_cls(None, head_size=128)

    assert path == DiffusionAttentionBackendEnum.FLASHINFER_ATTN.get_path()


def test_auto_skips_flashinfer_when_only_toplevel_package(monkeypatch: pytest.MonkeyPatch):
    """A stub that only imports ``flashinfer`` must not be auto-selected."""
    _blackwell_sm120(monkeypatch, cudnn_version=90499)
    _install_dummy_flashinfer(monkeypatch)

    path = CudaOmniPlatform.get_diffusion_attn_backend_cls(None, head_size=128)

    assert path == DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()


def test_explicit_flashinfer_raises_without_prefill_wrapper(monkeypatch: pytest.MonkeyPatch):
    _blackwell_sm120(monkeypatch)
    _install_dummy_flashinfer(monkeypatch)

    with pytest.raises(ValueError, match="BatchPrefillWithRaggedKVCacheWrapper"):
        CudaOmniPlatform.get_diffusion_attn_backend_cls("FLASHINFER_ATTN", head_size=128)


def test_auto_skips_flashinfer_for_unsupported_head_size_when_cudnn_too_old(monkeypatch: pytest.MonkeyPatch):
    _blackwell_sm120(monkeypatch, cudnn_version=90499)
    _install_dummy_flashinfer(monkeypatch)

    path = CudaOmniPlatform.get_diffusion_attn_backend_cls(None, head_size=72)

    assert path == DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()


def test_explicit_cudnn_raises_for_incompatible_head_size(monkeypatch: pytest.MonkeyPatch):
    _blackwell_sm120(monkeypatch)

    with pytest.raises(ValueError, match="head_size=12 is unsupported"):
        CudaOmniPlatform.get_diffusion_attn_backend_cls("CUDNN_ATTN", head_size=12)


def test_explicit_cudnn_accepts_supported_head_size(monkeypatch: pytest.MonkeyPatch):
    _blackwell_sm120(monkeypatch)

    path = CudaOmniPlatform.get_diffusion_attn_backend_cls("CUDNN_ATTN", head_size=72)

    assert path == DiffusionAttentionBackendEnum.CUDNN_ATTN.get_path()


def test_explicit_cudnn_skips_head_size_check_for_capability_probe(monkeypatch: pytest.MonkeyPatch):
    _blackwell_sm120(monkeypatch)

    path = CudaOmniPlatform.get_diffusion_attn_backend_cls("CUDNN_ATTN", head_size=-1)

    assert path == DiffusionAttentionBackendEnum.CUDNN_ATTN.get_path()


@pytest.mark.parametrize("head_size", [32, 320])
def test_explicit_flashinfer_raises_for_unsupported_head_size(monkeypatch: pytest.MonkeyPatch, head_size: int):
    _blackwell_sm120(monkeypatch)
    _install_dummy_flashinfer_prefill(monkeypatch)

    with pytest.raises(ValueError, match=f"head_size={head_size} is unsupported"):
        CudaOmniPlatform.get_diffusion_attn_backend_cls("FLASHINFER_ATTN", head_size=head_size)


def test_explicit_flashinfer_accepts_supported_head_size(monkeypatch: pytest.MonkeyPatch):
    _blackwell_sm120(monkeypatch)
    _install_dummy_flashinfer_prefill(monkeypatch)

    path = CudaOmniPlatform.get_diffusion_attn_backend_cls("FLASHINFER_ATTN", head_size=128)

    assert path == DiffusionAttentionBackendEnum.FLASHINFER_ATTN.get_path()


def test_explicit_flashinfer_skips_head_size_check_for_capability_probe(monkeypatch: pytest.MonkeyPatch):
    _blackwell_sm120(monkeypatch)
    _install_dummy_flashinfer_prefill(monkeypatch)

    path = CudaOmniPlatform.get_diffusion_attn_backend_cls("FLASHINFER_ATTN", head_size=-1)

    assert path == DiffusionAttentionBackendEnum.FLASHINFER_ATTN.get_path()


def test_auto_selects_cudnn_for_unknown_head_size_probe(monkeypatch: pytest.MonkeyPatch):
    _blackwell_sm120(monkeypatch)
    _install_dummy_flashinfer_prefill(monkeypatch)

    path = CudaOmniPlatform.get_diffusion_attn_backend_cls(None, head_size=-1)

    assert path == DiffusionAttentionBackendEnum.CUDNN_ATTN.get_path()


def test_explicit_flash_attn_rejects_blackwell_without_fa4(monkeypatch: pytest.MonkeyPatch):
    _blackwell_sm120(monkeypatch)
    monkeypatch.setattr(PACKAGES_CHECKER, "get_packages_info", lambda: {"has_flash_attn": True})

    with pytest.raises(ValueError, match="requires CuTe FlashAttention-4"):
        CudaOmniPlatform.get_diffusion_attn_backend_cls("FLASH_ATTN", head_size=64)


def test_explicit_flash_attn_accepts_blackwell_when_fa4_available(monkeypatch: pytest.MonkeyPatch):
    _blackwell_sm120(monkeypatch)
    monkeypatch.setattr(CudaOmniPlatform, "has_flash_attn_4", classmethod(lambda cls: True))

    path = CudaOmniPlatform.get_diffusion_attn_backend_cls("FLASH_ATTN", head_size=64)

    assert path == DiffusionAttentionBackendEnum.FLASH_ATTN.get_path()


def test_blackwell_dense_flash_capability_tracks_fa4(monkeypatch: pytest.MonkeyPatch):
    _blackwell_sm120(monkeypatch)

    assert CudaOmniPlatform.supports_diffusion_dense_flash_attention() is False

    monkeypatch.setattr(CudaOmniPlatform, "has_flash_attn_4", classmethod(lambda cls: True))

    assert CudaOmniPlatform.supports_diffusion_dense_flash_attention() is True


def test_hopper_dense_flash_capability_tracks_fa_package(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        CudaOmniPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(9, 0)),
    )
    monkeypatch.setattr(CudaOmniPlatform, "has_flash_attn_package", classmethod(lambda cls: False))

    assert CudaOmniPlatform.supports_diffusion_dense_flash_attention() is False

    monkeypatch.setattr(CudaOmniPlatform, "has_flash_attn_package", classmethod(lambda cls: True))

    assert CudaOmniPlatform.supports_diffusion_dense_flash_attention() is True


def test_auto_selects_flash_attn_on_blackwell_when_cudnn_old_and_fa4_available(
    monkeypatch: pytest.MonkeyPatch,
):
    _blackwell_sm120(monkeypatch, cudnn_version=90499)
    monkeypatch.setitem(sys.modules, "flashinfer", None)
    monkeypatch.setattr(CudaOmniPlatform, "has_flash_attn_4", classmethod(lambda cls: True))

    path = CudaOmniPlatform.get_diffusion_attn_backend_cls(None, head_size=128)

    assert path == DiffusionAttentionBackendEnum.FLASH_ATTN.get_path()


def test_marked_paged_layer_constructs_on_blackwell_cudnn_without_fa4(monkeypatch: pytest.MonkeyPatch):
    """Scheduler-paged remap must not require dense FA4 when CUDNN is the default."""
    _blackwell_sm120(monkeypatch)
    _cached_get_backend_cls.cache_clear()

    od_config = SimpleNamespace(
        diffusion_attention_config=AttentionConfig(),
        diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER,
        parallel_config=SimpleNamespace(ring_degree=1, allgather_degree=1),
        diffusion_kv_cache_dtype=None,
        diffusion_kv_cache_skip_step_indices=None,
        diffusion_kv_cache_skip_layer_indices=None,
        model_class_name=None,
    )

    with set_current_diffusion_config(od_config):
        attention = Attention(
            num_heads=4,
            head_size=64,
            causal=False,
            softmax_scale=1.0,
            paged_kv_cache_role="primary",
        )

    assert attention.attn_backend.get_name() == "FLASH_ATTN"
    assert attention.backend_pref == "FLASH_ATTN"
    assert attention.backend_explicit is False
    assert attention.attn_spec is None
