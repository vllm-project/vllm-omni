# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for seeded rotations and the quantized NPU attention backend.

These tests load ``kv_quant_npu`` from its source file via ``importlib`` so
the test module itself does not ``import vllm_omni`` (which would pull
``patch`` → ``aenum``, vLLM, etc.).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _repo_root() -> Path:
    """Resolve checkout root (parent of ``vllm_omni/``), not ``tests/``."""
    here = Path(__file__).resolve()
    marker = Path("vllm_omni") / "platforms" / "npu" / "quant" / "kv_quant_npu.py"
    for parent in here.parents:
        if (parent / marker).is_file():
            return parent
    msg = f"could not locate repo root (no {marker}) starting from {here}"
    raise FileNotFoundError(msg)


def _load_kv_quant_npu() -> ModuleType:
    path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "quant" / "kv_quant_npu.py"
    if not path.is_file():
        msg = f"kv_quant_npu source not found: {path}"
        raise FileNotFoundError(msg)
    name = "vllm_omni_test_kv_quant_npu_standalone"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        msg = f"cannot load import spec for {path}"
        raise RuntimeError(msg)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kv_quant_npu = _load_kv_quant_npu()


def _npu_smoke_available() -> bool:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return bool(hasattr(torch, "npu") and torch.npu.is_available())


npu_smoke = pytest.mark.skipif(not _npu_smoke_available(), reason="NPU device or torch_npu not available.")


def test_rotation_is_fixed_orthogonal_cached_and_preserves_rng():
    kv_quant_npu.get_quant_attention_rotation.cache_clear()
    before = torch.random.get_rng_state()
    rot = kv_quant_npu.get_quant_attention_rotation(torch.device("cpu"), torch.float32, 64)
    assert rot is kv_quant_npu.get_quant_attention_rotation(torch.device("cpu"), torch.float32, 64)
    torch.testing.assert_close(rot @ rot.T, torch.eye(64))
    assert torch.equal(before, torch.random.get_rng_state())


@npu_smoke
@pytest.mark.npu
@pytest.mark.parametrize("method", ["fp8", "mxfp8", "mxfp4"])
@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
@pytest.mark.parametrize("seq_len", [256, 75600])
def test_dense_runtime_backend_real_npu(method, layout, seq_len):
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
    from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl

    # On NPU, missing Runtime symbols must fail this qualification test.
    query = torch.randn(1, seq_len, 2, 128, dtype=torch.bfloat16, device="npu") * 0.1
    # Distinct constant values per head detect head/layout mixing and tail writes.
    value = torch.tensor([0.5, -0.5], dtype=query.dtype, device=query.device)
    value = value.view(1, 1, 2, 1).expand_as(query).contiguous()
    if layout == "BNSD":
        query, value = query.transpose(1, 2), value.transpose(1, 2)
    impl = FlashAttentionImpl(
        num_heads=2,
        head_size=128,
        softmax_scale=128**-0.5,
        qkv_layout=layout,
    )
    metadata = AttentionMetadata(extra={"kv_cache_dtype": method})
    out = impl.forward_npu(query, query, value, metadata)
    assert out.shape == query.shape and out.dtype == query.dtype
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float().cpu(), value.float().cpu(), rtol=0.02, atol=0.02)
