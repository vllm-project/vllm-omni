# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for shared AdaLayerNorm layers used by FLUX and other models."""

import os

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _init_distributed():
    """Initialize the minimal distributed environment required by
    ReplicatedLinear (tensor-parallel group must exist)."""
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29501")
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="env://",
    )
    initialize_model_parallel()
    yield
    cleanup_dist_env_and_memory()


@pytest.fixture(autouse=True)
def _force_default_gemm(monkeypatch):
    """Force CPU-compatible GEMM dispatch for tests using CPU tensors.

    vLLM's dispatch_unquantized_gemm() selects the backend by platform
    (e.g. rocm_unquantized_gemm on AMD machines), not by tensor device.
    CPU test tensors crash with NotImplementedError on ROCm.  Monkeypatch
    the dispatcher to always return the default (torch.nn.functional.linear)
    implementation which works on any device."""
    from vllm.model_executor.layers.utils import default_unquantized_gemm

    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        lambda: default_unquantized_gemm,
    )


def test_adalayernorm_import_from_shared_module():
    """Verify imports work from the shared adalayernorm module."""
    from vllm_omni.diffusion.layers.adalayernorm import (  # noqa: F401
        AdaLayerNormContinuous,
        AdaLayerNormZero,
        AdaLayerNormZeroSingle,
    )


def test_adalayernorm_zero_forward_shape():
    """AdaLayerNormZero produces correct output shapes (x, gate, shift, scale, gate)."""
    from vllm_omni.diffusion.layers.adalayernorm import AdaLayerNormZero

    dim = 64
    batch = 2
    seq_len = 4
    norm = AdaLayerNormZero(dim)

    x = torch.randn(batch, seq_len, dim)
    emb = torch.randn(batch, dim)

    out_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = norm(x, emb)

    assert out_x.shape == (batch, seq_len, dim)
    assert gate_msa.shape == (batch, dim)
    assert shift_mlp.shape == (batch, dim)
    assert scale_mlp.shape == (batch, dim)
    assert gate_mlp.shape == (batch, dim)


def test_adalayernorm_zero_single_forward_shape():
    """AdaLayerNormZeroSingle produces (x, gate) with correct shapes."""
    from vllm_omni.diffusion.layers.adalayernorm import AdaLayerNormZeroSingle

    dim = 64
    batch = 2
    seq_len = 4
    norm = AdaLayerNormZeroSingle(dim)

    x = torch.randn(batch, seq_len, dim)
    emb = torch.randn(batch, dim)

    out_x, gate = norm(x, emb)

    assert out_x.shape == (batch, seq_len, dim)
    assert gate.shape == (batch, dim)


def test_adalayernorm_continuous_forward_shape():
    """AdaLayerNormContinuous produces correct output shape."""
    from vllm_omni.diffusion.layers.adalayernorm import AdaLayerNormContinuous

    dim = 64
    cond_dim = 64
    batch = 2
    seq_len = 4
    norm = AdaLayerNormContinuous(dim, cond_dim)

    x = torch.randn(batch, seq_len, dim)
    conditioning = torch.randn(batch, cond_dim)

    out = norm(x, conditioning)

    assert out.shape == (batch, seq_len, dim)


def test_adalayernorm_zero_accepts_quant_config():
    """Constructor accepts quant_config=None and prefix='test' without error."""
    from vllm_omni.diffusion.layers.adalayernorm import (
        AdaLayerNormContinuous,
        AdaLayerNormZero,
        AdaLayerNormZeroSingle,
    )

    # Should not raise with quant_config=None and prefix
    AdaLayerNormZero(64, quant_config=None, prefix="test.norm1")
    AdaLayerNormZeroSingle(64, quant_config=None, prefix="test.norm")
    AdaLayerNormContinuous(64, 64, quant_config=None, prefix="test.norm_out")


def test_adalayernorm_uses_replicated_linear():
    """Verify .linear is a ReplicatedLinear instance (not nn.Linear)."""
    from vllm.model_executor.layers.linear import ReplicatedLinear

    from vllm_omni.diffusion.layers.adalayernorm import (
        AdaLayerNormContinuous,
        AdaLayerNormZero,
        AdaLayerNormZeroSingle,
    )

    norm_zero = AdaLayerNormZero(64)
    assert isinstance(norm_zero.linear, ReplicatedLinear)

    norm_zero_single = AdaLayerNormZeroSingle(64)
    assert isinstance(norm_zero_single.linear, ReplicatedLinear)

    norm_continuous = AdaLayerNormContinuous(64, 64)
    assert isinstance(norm_continuous.linear, ReplicatedLinear)


# ── Numerical equivalence tests against diffusers originals ──


def _copy_weights(src_linear, dst_replicated_linear):
    """Copy weights from nn.Linear to ReplicatedLinear for comparison."""
    dst_replicated_linear.weight.data.copy_(src_linear.weight.data)
    if src_linear.bias is not None and dst_replicated_linear.bias is not None:
        dst_replicated_linear.bias.data.copy_(src_linear.bias.data)


def test_adalayernorm_zero_matches_diffusers():
    """Verify AdaLayerNormZero produces identical output to diffusers original."""
    from diffusers.models.normalization import (
        AdaLayerNormZero as DiffusersAdaLayerNormZero,
    )

    from vllm_omni.diffusion.layers.adalayernorm import AdaLayerNormZero

    dim = 64
    torch.manual_seed(42)
    ours = AdaLayerNormZero(dim)
    ref = DiffusersAdaLayerNormZero(dim)

    # Copy weights: nn.Linear -> ReplicatedLinear
    _copy_weights(ref.linear, ours.linear)

    x = torch.randn(2, 4, dim)
    emb = torch.randn(2, dim)

    out_ours = ours(x, emb)
    out_ref = ref(x, emb=emb)

    for o, r in zip(out_ours, out_ref):
        torch.testing.assert_close(o, r, atol=1e-5, rtol=1e-5)


def test_adalayernorm_zero_single_matches_diffusers():
    """Verify AdaLayerNormZeroSingle produces identical output to diffusers original."""
    from diffusers.models.normalization import (
        AdaLayerNormZeroSingle as DiffusersAdaLayerNormZeroSingle,
    )

    from vllm_omni.diffusion.layers.adalayernorm import AdaLayerNormZeroSingle

    dim = 64
    torch.manual_seed(42)
    ours = AdaLayerNormZeroSingle(dim)
    ref = DiffusersAdaLayerNormZeroSingle(dim)

    _copy_weights(ref.linear, ours.linear)

    x = torch.randn(2, 4, dim)
    emb = torch.randn(2, dim)

    out_ours = ours(x, emb)
    out_ref = ref(x, emb=emb)

    for o, r in zip(out_ours, out_ref):
        torch.testing.assert_close(o, r, atol=1e-5, rtol=1e-5)


def test_adalayernorm_continuous_matches_diffusers():
    """Verify AdaLayerNormContinuous produces identical output to diffusers original."""
    from diffusers.models.normalization import (
        AdaLayerNormContinuous as DiffusersAdaLayerNormContinuous,
    )

    from vllm_omni.diffusion.layers.adalayernorm import AdaLayerNormContinuous

    dim = 64
    cond_dim = 64
    torch.manual_seed(42)
    # Match constructor args: diffusers defaults elementwise_affine=True, eps=1e-5
    ours = AdaLayerNormContinuous(dim, cond_dim, elementwise_affine=False, eps=1e-6)
    ref = DiffusersAdaLayerNormContinuous(dim, cond_dim, elementwise_affine=False, eps=1e-6)

    _copy_weights(ref.linear, ours.linear)

    x = torch.randn(2, 4, dim)
    cond = torch.randn(2, cond_dim)

    out_ours = ours(x, cond)
    out_ref = ref(x, cond)

    torch.testing.assert_close(out_ours, out_ref, atol=1e-5, rtol=1e-5)
