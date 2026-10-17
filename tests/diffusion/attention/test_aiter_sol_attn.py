# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
import types

import pytest
import torch

from vllm_omni.diffusion.attention.backends import aiter_sage, aiter_sol
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, VideoTokenLayout, VideoTokenSpan
from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.diffusion.attention.capabilities import CompilationMode, ExecutionContext, SupportStatus
from vllm_omni.diffusion.data import AttentionSpec, SolAttnSpec
from vllm_omni.diffusion.forward_context import set_forward_context

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

B, S, H, D = 1, 64, 4, 128


@pytest.fixture
def kernels(monkeypatch):
    calls: list[tuple] = []

    def fake_sage(query, key, value, softmax_scale):
        assert all(t.is_contiguous() for t in (query, key, value))
        calls.append(("sage", query.shape[1]))
        return query + 1

    def fake_sol(query, key, value, *, tau, sink_tokens, sink_start, softmax_scale):
        assert all(t.is_contiguous() for t in (query, key, value))
        calls.append(("sol", query.shape[1], tau, sink_tokens, sink_start, softmax_scale))
        return query + 2

    monkeypatch.setattr(aiter_sage, "_sage_kernel", lambda: fake_sage)
    monkeypatch.setattr(aiter_sol, "_sol_kernel", lambda: fake_sol)
    return calls


def _impl(prefix="blocks.5.attn", **sol_attn):
    kwargs = AttentionSpec(backend="AITER_SOL", sol_attn=sol_attn).backend_kwargs()
    return aiter_sol.AiterSolAttentionImpl(
        num_heads=H, head_size=D, softmax_scale=0.125, prefix=prefix, backend_kwargs=kwargs
    )


def _qkv(seq=S):
    return torch.randn(B, seq, 3, H, D, dtype=torch.bfloat16).unbind(2)


def test_backend_is_registered():
    backend = DiffusionAttentionBackendEnum.AITER_SOL.get_class()
    assert backend is aiter_sol.AiterSolAttentionBackend
    assert backend.get_name() == "AITER_SOL"
    assert backend.get_impl_cls() is aiter_sol.AiterSolAttentionImpl
    assert backend.supported_platforms == ("rocm",)
    assert backend.supports_prefix_kv_slicing


def test_spec_defaults_and_kwargs():
    spec = AttentionSpec(backend="AITER_SOL")
    assert spec.sol_attn == SolAttnSpec()
    assert spec.backend_kwargs() == {
        "sol_attn": {
            "tau": 1.0,
            "thresh_type": "diag",
            "sink_tokens": 0,
            "sink_start": 0,
            "prefix_sink": True,
            "dense_steps": 10,
            "dense_layers": [0, 1],
        }
    }


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({"tau": -1.0}, "tau"),
        ({"thresh_type": "exact"}, "thresh_type"),
        ({"sink_tokens": -1}, "sink_tokens"),
        ({"sink_start": -1}, "sink_start"),
        ({"dense_steps": -1}, "dense_steps"),
    ],
)
def test_spec_rejects_invalid_values(config, match):
    with pytest.raises(ValueError, match=match):
        SolAttnSpec(**config)


def test_sol_attn_config_requires_sol_backend():
    with pytest.raises(ValueError, match="sol_attn is only supported"):
        AttentionSpec(backend="AITER_SAGE", sol_attn={"tau": 1.0})


@pytest.mark.parametrize("has_op", [False, True])
def test_validate_available_requires_sol_op(monkeypatch, has_op):
    monkeypatch.setattr(aiter_sage, "_gcn_arch", lambda: "gfx1201")
    package = types.ModuleType("aiter")
    setattr(package, "gfx1201_sage_attention", object())
    if has_op:
        setattr(package, "gfx1201_sol_attention", object())
    monkeypatch.setitem(sys.modules, "aiter", package)
    if has_op:
        aiter_sol.AiterSolAttentionBackend.validate_available()
    else:
        with pytest.raises(ValueError, match="AITER_SOL requires an aiter build with gfx1201_sol_attention"):
            aiter_sol.AiterSolAttentionBackend.validate_available()


def test_sparse_step_calls_sol_with_config(kernels):
    q, k, v = _qkv()
    impl = _impl(tau=1.5, sink_tokens=16, sink_start=None, dense_steps=2, dense_layers="0")
    with set_forward_context(denoise_step_idx=2):
        out = impl.forward_hip(q, k, v)
    assert kernels == [("sol", S, 1.5, 16, None, 0.125)]
    torch.testing.assert_close(out, q + 2, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("prefix", "step"),
    [
        ("blocks.5.attn", 1),  # warmup step
        ("blocks.0.attn", 7),  # dense layer
        ("blocks.5.attn", None),  # no published step
    ],
)
def test_dense_guard_runs_sage(kernels, prefix, step):
    q, k, v = _qkv()
    impl = _impl(prefix, dense_steps=2, dense_layers="0")
    with set_forward_context(denoise_step_idx=step):
        out = impl.forward_hip(q, k, v)
    assert kernels == [("sage", S)]
    torch.testing.assert_close(out, q + 1, rtol=0, atol=0)


def test_no_dense_steps_is_sparse_without_step(kernels):
    q, k, v = _qkv()
    _impl(dense_steps=0, dense_layers=None).forward_hip(q, k, v)
    assert [call[0] for call in kernels] == ["sol"]


@pytest.mark.parametrize("valid", [1, 37, S - 1])
def test_valid_kv_prefix_is_sliced_and_padding_rows_are_zero(kernels, valid):
    q, k, v = _qkv()
    impl = _impl(dense_steps=0, sink_tokens=48)
    out = impl.forward_hip(q, k, v, AttentionMetadata(extra={"valid_kv_length": valid}))
    sink = min(48, valid)
    assert kernels == [("sol", valid, 1.0, sink, min(0, valid - sink), 0.125)]
    torch.testing.assert_close(out[:, :valid], q[:, :valid] + 2, rtol=0, atol=0)
    assert torch.count_nonzero(out[:, valid:]) == 0


def test_sink_start_is_clamped_to_sequence(kernels):
    q, k, v = _qkv()
    _impl(dense_steps=0, sink_tokens=16, sink_start=60).forward_hip(q, k, v)
    assert kernels == [("sol", S, 1.0, 16, S - 16, 0.125)]


def _sdpa(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=0.125
    ).transpose(1, 2)


@pytest.mark.parametrize(
    ("layout", "prefix"),
    [
        (VideoTokenLayout(prefix_len=20, latent_grid=(1, 4, 8)), 20),
        (
            VideoTokenLayout(
                used_len=S,
                video_spans=(
                    VideoTokenSpan(start=4, latent_grid=(1, 2, 4), role="reference"),
                    VideoTokenSpan(start=24, latent_grid=(1, 4, 10), role="target"),
                ),
            ),
            24,
        ),
    ],
)
def test_layout_prefix_is_exact_key_and_query_sink(kernels, layout, prefix):
    q, k, v = _qkv()
    impl = _impl(dense_steps=0, sink_tokens=16, sink_start=None)
    out = impl.forward_hip(q, k, v, AttentionMetadata(video_layout=layout))
    assert kernels == [("sol", S, 1.0, prefix, 0, 0.125)]
    torch.testing.assert_close(out[:, :prefix], _sdpa(q[:, :prefix], k, v), rtol=0, atol=0)
    torch.testing.assert_close(out[:, prefix:], q[:, prefix:] + 2, rtol=0, atol=0)


def test_layout_prefix_sink_can_be_disabled(kernels):
    q, k, v = _qkv()
    impl = _impl(dense_steps=0, sink_tokens=16, sink_start=None, prefix_sink=False)
    out = impl.forward_hip(q, k, v, AttentionMetadata(video_layout=VideoTokenLayout(prefix_len=20)))
    assert kernels == [("sol", S, 1.0, 16, None, 0.125)]
    torch.testing.assert_close(out, q + 2, rtol=0, atol=0)


def test_unsupported_input_reports_dense_path():
    q, k, v = _qkv()
    result = _impl(dense_steps=0).resolve_execution_path(ExecutionContext(platform="rocm"), q.float(), k, v, None)
    assert result.support.status is SupportStatus.UNSUPPORTED
    assert result.path == "sage_dense"


def test_unsupported_input_is_rejected_on_sparse_path(kernels):
    q, k, v = _qkv()
    with pytest.raises(ValueError, match="AITER_SOL: Q, K, and V must be BF16"):
        _impl(dense_steps=0).forward_hip(q.float(), k.float(), v.float())
    assert kernels == []


def test_execution_path_reports_backend_and_path():
    q, k, v = _qkv()
    impl = _impl(dense_steps=2, dense_layers="0")
    context = ExecutionContext(platform="rocm")
    with set_forward_context(denoise_step_idx=1):
        dense = impl.resolve_execution_path(context, q, k, v, None)
    with set_forward_context(denoise_step_idx=2):
        sparse = impl.resolve_execution_path(context, q, k, v, None)
    assert (dense.backend, dense.path) == ("AITER_SOL", "sage_dense")
    assert (sparse.backend, sparse.path) == ("AITER_SOL", "sol_sparse")
    assert sparse.support.status is SupportStatus.SUPPORTED
    assert sparse.compilation_mode is CompilationMode.CUSTOM_OP
    unsupported = impl.resolve_execution_path(ExecutionContext(platform="cuda"), q, k, v, None)
    assert unsupported.support.status is SupportStatus.UNSUPPORTED
    assert unsupported.path == "sage_dense"
    assert "AITER_SOL runs on ROCm only" in unsupported.support.reason


@pytest.mark.parametrize(
    ("prefix", "layer"), [("blocks.12.attn", 12), ("transformer.blocks.3.attn", 3), ("attn", None)]
)
def test_layer_index_from_prefix(prefix, layer):
    assert _impl(prefix).layer_idx == layer
