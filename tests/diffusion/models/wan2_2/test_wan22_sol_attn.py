# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention

from vllm_omni.diffusion.attention.backends import sol_attn as sol_backend
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.diffusion.attention.backends.sol_attn import (
    SolAttnBackend,
    SolAttnImpl,
    get_morton_indices,
    wan_sol_layout,
)
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, OmniDiffusionConfig, SolAttnSpec
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import _morton_block_inputs

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_sol_is_explicit_and_scoped_to_self():
    assert AttentionConfig().resolve_with_source("self")[0] is None
    config = AttentionConfig(per_role={"self": {"backend": "SOL_ATTN", "sol_attn": {"tau": 1.25}}})
    assert config.resolve_with_source("cross")[0] is None
    spec, _ = config.resolve_with_source("self")
    assert spec.backend_kwargs() == {"tau": 1.25, "dense_layers": 1, "dense_steps": 4}
    assert DiffusionAttentionBackendEnum.SOL_ATTN.get_class() is SolAttnBackend


@pytest.mark.parametrize("options", [{"tau": float("nan")}, {"dense_steps": -1}, {"dense_layers": 1.5}])
def test_sol_options(options):
    with pytest.raises(ValueError):
        SolAttnSpec(**options)
    with pytest.raises(ValueError, match="require the SOL_ATTN"):
        AttentionSpec(backend="FLASH_ATTN", sol_attn={})


def test_sol_rejects_fp16_at_startup():
    with set_current_diffusion_config(OmniDiffusionConfig(dtype=torch.float16)):
        with pytest.raises(ValueError, match="requires BF16"):
            SolAttnImpl(2, 128, 0.2, prefix="blocks.0.attn1", qkv_layout="BSND")


@pytest.mark.parametrize(
    ("step", "layer", "sparse"), [(None, 1, False), (3, 29, False), (4, 0, False), (4, 1, True), (49, 29, True)]
)
def test_sol_step_and_layer_dispatch(monkeypatch, step, layer, sparse):
    monkeypatch.setattr(sol_backend, "_sol_available", lambda: True)
    spec = AttentionSpec(backend="SOL_ATTN")
    impl = SolAttnImpl(
        2, 128, 0.2, prefix=f"blocks.{layer}.attn1", qkv_layout="BSND", backend_kwargs=spec.backend_kwargs()
    )
    provider = ModuleType("mindiesd")
    kernel = Mock(side_effect=lambda q, k, v, **kw: q + 1)
    monkeypatch.setattr(provider, "sol_attn", kernel, raising=False)
    monkeypatch.setitem(sys.modules, "mindiesd", provider)
    dense = Mock(side_effect=lambda q, *args: q)
    monkeypatch.setattr(impl.dense, "forward_npu", dense)
    q = torch.zeros(1, 2, 4, 128, dtype=torch.bfloat16).transpose(1, 2)
    last_attention = Mock(spec=Attention)
    last_impl = SolAttnImpl(
        2, 128, 0.2, prefix="blocks.29.attn1", qkv_layout="BSND", backend_kwargs=spec.backend_kwargs()
    )
    last_attention.effective_attention.return_value = (last_impl, SolAttnBackend, spec)
    with set_forward_context(denoise_step_idx=step):
        meta = AttentionMetadata(extra=wan_sol_layout(last_attention))
    output = impl.forward_npu(q, q, q, meta)
    assert dense.call_count == int(not sparse)
    assert kernel.call_count == int(sparse)
    torch.testing.assert_close(output, q + int(sparse))
    if sparse:
        assert all(arg.is_contiguous() for arg in kernel.call_args.args)
        assert kernel.call_args.kwargs == {"scale": 0.2, "tau": 1.0}
    with pytest.raises(ValueError, match="declared token order"):
        impl.forward_npu(q, q, q, AttentionMetadata())


def test_wan_sol_layout_follows_active_backend_and_all_dense_layers():
    last_attention = Mock(spec=Attention)
    spec = AttentionSpec(backend="SOL_ATTN", sol_attn={"dense_layers": 30, "dense_steps": 0})
    impl = SolAttnImpl(2, 128, 0.2, prefix="blocks.29.attn1", qkv_layout="BSND", backend_kwargs=spec.backend_kwargs())
    last_attention.effective_attention.side_effect = [(impl, SolAttnBackend, spec), (impl.dense, None, None)]
    with set_forward_context(denoise_step_idx=10):
        assert wan_sol_layout(last_attention) == {"sol_morton_order": False}
        assert wan_sol_layout(last_attention) == {}


def test_unavailable_sol_uses_flash_for_all_layers_without_morton(monkeypatch):
    monkeypatch.setitem(sys.modules, "mindiesd", ModuleType("mindiesd"))
    warning = Mock()
    monkeypatch.setattr(sol_backend.logger, "warning_once", warning)
    sol_backend._sol_available.cache_clear()
    try:
        impls = [SolAttnImpl(2, 128, 0.2, prefix=f"blocks.{i}.attn1", qkv_layout="BSND") for i in (0, 1, 29)]
        last = Mock(spec=Attention)
        last.effective_attention.return_value = (impls[-1], SolAttnBackend, None)
        q = torch.ones(1, 3, 2, 128)
        for step in (0, 4, 49):
            with set_forward_context(denoise_step_idx=step):
                extra = wan_sol_layout(last)
            assert extra == {"sol_morton_order": False}
            for impl in impls:
                dense = Mock(return_value=q)
                monkeypatch.setattr(impl.dense, "forward_npu", dense)
                meta = AttentionMetadata(extra=extra)
                assert impl.forward_npu(q, q, q, meta) is q
                dense.assert_called_once_with(q, q, q, meta)
        warning.assert_called_once()
        message, reason = warning.call_args.args
        assert "FLASH_ATTN" in message and "callable sol_attn" in reason
    finally:
        sol_backend._sol_available.cache_clear()


@pytest.mark.parametrize(("soc", "arch"), [(220, "ascend910b"), (260, "ascend950")])
def test_sol_device_kernels_and_opp_coverage(monkeypatch, tmp_path, soc, arch):
    monkeypatch.setitem(sys.modules, "mindiesd", SimpleNamespace(sol_attn=Mock()))
    npu = Mock()
    npu.get_soc_version.return_value = soc
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu=npu))
    ops = Mock()
    monkeypatch.setattr(torch.ops, "mindiesd", ops)
    registered = Mock(return_value=True)
    monkeypatch.setattr(torch._C, "_dispatch_has_kernel_for_dispatch_key", registered)
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", str(tmp_path))
    if arch == "ascend950":
        binaries = ["sol_attn_a5_" + name for name in ("summary", "route", "forward")]
    else:
        binaries = [
            "sol_attn_" + name + "_kernel"
            for name in (
                "prepare_qkv",
                "prepare_route_pack",
                "fwd_m128",
                "prepare_diag_thresh",
                "prepare_short",
                "prepare_route_stats",
                "short_fwd_m128",
            )
        ]
    root = tmp_path / "op_impl/ai_core/tbe/kernel"
    config = root / "config" / arch
    config.mkdir(parents=True)
    for binary in binaries:
        directory = root / arch / binary
        directory.mkdir(parents=True)
        metadata = directory / "variant.json"
        metadata.write_text(json.dumps({"binFileName": "kernel", "binFileSuffix": ".o"}))
        entry = {"binInfo": {"jsonFilePath": metadata.relative_to(root).as_posix()}}
        (config / f"{binary}.json").write_text(json.dumps({"binList": [entry]}))
        (directory / "kernel.o").touch()
    assert sol_backend._sol_unavailable_reason() is None
    directory = root / arch / binaries[-1]
    (directory / "kernel.o").rename(directory / "obsolete_build.o")
    assert binaries[-1] in sol_backend._sol_unavailable_reason()
    registered.return_value = False
    assert "missing NPU operator" in sol_backend._sol_unavailable_reason()
    registered.reset_mock()
    del ops.sol_attn_prepare_qkv
    assert "mindiesd::sol_attn_prepare_qkv" in sol_backend._sol_unavailable_reason()
    registered.assert_not_called()
    npu.get_soc_version.return_value = 250
    assert "SoC 250" in sol_backend._sol_unavailable_reason()
    npu.get_soc_version.side_effect = RuntimeError("driver error")
    with pytest.raises(RuntimeError, match="driver error"):
        sol_backend._sol_unavailable_reason()


def test_missing_mindiesd_uses_native_flash(monkeypatch):
    monkeypatch.setitem(sys.modules, "mindiesd", None)
    assert "not installed" in sol_backend._sol_unavailable_reason()
    impl = SolAttnImpl(2, 128, 0.2, prefix="blocks.1.attn1", qkv_layout="BSND")
    q = torch.zeros(1, 3, 2, 128)
    native = Mock(return_value=q)
    monkeypatch.setattr(impl.dense, "_npu_fusion_attention", native)
    assert impl.forward_npu(q, q, q, AttentionMetadata(extra={"sol_morton_order": False})) is q
    native.assert_called_once_with(q, q, q, atten_mask=None, sparse_mode=0, softmax_scale=0.2)


@pytest.mark.parametrize("exc", [ValueError("invalid shape"), torch.OutOfMemoryError("out of memory")])
def test_sol_execution_errors_propagate(monkeypatch, exc):
    impl = SolAttnImpl(2, 128, 0.2, prefix="blocks.1.attn1", qkv_layout="BSND")
    monkeypatch.setitem(sys.modules, "mindiesd", SimpleNamespace(sol_attn=Mock(side_effect=exc)))
    q = torch.zeros(1, 3, 2, 128)
    with pytest.raises(type(exc), match=str(exc)):
        impl.forward_npu(q, q, q, AttentionMetadata(extra={"sol_morton_order": True}))


def test_morton_alignment_and_dense_round_trip():
    grid = (2, 2, 3)
    indices = get_morton_indices(grid, torch.device("cpu"))
    assert get_morton_indices(grid, torch.device("cpu")) is indices
    order, inverse = indices
    assert order.tolist() == [0, 1, 3, 4, 6, 7, 9, 10, 2, 5, 8, 11]
    assert torch.equal(order[inverse], torch.arange(12))
    generator = torch.Generator().manual_seed(17)
    tokens = torch.randn(2, 12, 8, generator=generator)
    cos, sin = torch.randn(2, 1, 12, 1, 8, generator=generator)
    per_token = torch.randn(2, 12, 6, 8, generator=generator)
    hidden, (mc, ms), mt = _morton_block_inputs(tokens, (cos, sin), per_token, order)
    torch.testing.assert_close(mc, cos[:, order])
    torch.testing.assert_close(ms, sin[:, order])
    torch.testing.assert_close(mt, per_token[:, order])
    reference = scaled_dot_product_attention(
        tokens + cos.squeeze(2), tokens + sin.squeeze(2), tokens + per_token[:, :, 0]
    )
    actual = scaled_dot_product_attention(hidden + mc.squeeze(2), hidden + ms.squeeze(2), hidden + mt[:, :, 0])
    torch.testing.assert_close(actual[:, inverse], reference)

    per_sample = per_token[:, 0]
    assert _morton_block_inputs(tokens, (cos, sin), per_sample, order)[2] is per_sample
