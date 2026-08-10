# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the Sol-Attn sparse attention backend."""

import importlib.util
import math
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.backends import sol_attn as sol_mod
from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.diffusion.attention.backends.sol_attn import (
    SolAttnBackend,
    SolAttnConfig,
    SolAttnImpl,
    _parse_layer_ranges,
)
from vllm_omni.diffusion.data import AttentionSpec, SolAttnSpec

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _impl(prefix: str = "", **backend_kwargs) -> SolAttnImpl:
    return SolAttnImpl(
        num_heads=8,
        head_size=128,
        softmax_scale=128**-0.5,
        causal=False,
        num_kv_heads=8,
        prefix=prefix,
        backend_kwargs=backend_kwargs or None,
    )


def _set_denoise_step(monkeypatch, step: int | None) -> None:
    ctx = SimpleNamespace(denoise_step_idx=step)
    monkeypatch.setattr(sol_mod, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(sol_mod, "get_forward_context", lambda: ctx)


def test_registry_enum_resolves_backend():
    backend = DiffusionAttentionBackendEnum.SOL_ATTN
    assert backend.get_path().endswith("sol_attn.SolAttnBackend")
    assert backend.get_class() is SolAttnBackend


def test_parse_layer_ranges():
    assert _parse_layer_ranges(None) == frozenset()
    assert _parse_layer_ranges(3) == frozenset({3})
    assert _parse_layer_ranges("0,1,3-5") == frozenset({0, 1, 3, 4, 5})
    assert _parse_layer_ranges(" 2 , 4 ") == frozenset({2, 4})


def test_backend_contract():
    assert SolAttnBackend.get_supported_head_sizes() == [128]
    assert SolAttnBackend.get_name() == "SOL_ATTN"
    assert SolAttnBackend.get_impl_cls() is SolAttnImpl
    assert SolAttnBackend.accept_output_buffer is True
    assert SolAttnBackend.supports_prefix_kv_slicing is True


def test_impl_rejects_unsupported_head_size():
    with pytest.raises(ValueError, match="head_size=128"):
        SolAttnImpl(8, 64, math.sqrt(64) ** -0.5, causal=False, prefix="blocks.0.attn")


def test_parse_layer_idx_from_prefix():
    assert _impl("blocks.5.attn").layer_idx == 5
    assert _impl("token_refiner.blocks.0.attn").layer_idx == 0
    assert _impl("some_other_attn").layer_idx is None


def test_config_defaults():
    cfg = SolAttnConfig.from_backend_kwargs(None)
    assert cfg.tau == 1.0
    assert cfg.thresh_type == "diag"
    assert cfg.kv_splits == "auto"
    assert cfg.sink_tokens == 0
    assert cfg.dense_steps == 10
    assert cfg.dense_layers == frozenset({0, 1})


def test_config_from_backend_kwargs():
    cfg = SolAttnConfig.from_backend_kwargs(
        {
            "sol_attn": {
                "tau": 2.0,
                "thresh_type": "exact",
                "sink_tokens": 951,
                "sink_start": 0,
                "dense_steps": 20,
                "dense_layers": "0,1,3-5",
                "kv_splits": 4,
            }
        }
    )
    assert (cfg.tau, cfg.thresh_type, cfg.sink_tokens, cfg.dense_steps, cfg.kv_splits) == (
        2.0,
        "exact",
        951,
        20,
        4,
    )
    assert cfg.dense_layers == frozenset({0, 1, 3, 4, 5})


def test_dense_guard_uses_early_steps(monkeypatch):
    _set_denoise_step(monkeypatch, 3)
    impl = _impl("blocks.5.attn", sol_attn={"dense_steps": 10, "dense_layers": "0,1"})
    assert impl._should_use_dense() is True


def test_sparse_layer_after_dense_guard(monkeypatch):
    _set_denoise_step(monkeypatch, 20)
    impl = _impl("blocks.5.attn", sol_attn={"dense_steps": 10, "dense_layers": "0,1"})
    assert impl._should_use_dense() is False


def test_dense_layer_by_index(monkeypatch):
    _set_denoise_step(monkeypatch, 20)
    impl = _impl("blocks.0.attn", sol_attn={"dense_steps": 10, "dense_layers": "0,1"})
    assert impl._should_use_dense() is True


def test_dense_when_no_forward_context():
    impl = _impl("blocks.5.attn")
    assert impl._should_use_dense() is False


@pytest.mark.parametrize(
    ("offsets", "total", "expected"),
    [
        ([0, 26, 26], 26, [0, 26]),
        ([0, 26, 32], 32, [0, 26, 32]),
    ],
)
def test_dense_fallback_drops_only_empty_packed_documents(
    monkeypatch,
    offsets,
    total,
    expected,
):
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
    from vllm_omni.diffusion.attention.backends.utils import fa as fa_utils

    captured = {}

    def fake_flash_attn_varlen_func(**kwargs):
        captured["cu_seqlens_q"] = kwargs["cu_seqlens_q"].clone()
        captured["cu_seqlens_k"] = kwargs["cu_seqlens_k"].clone()
        return torch.zeros_like(kwargs["q"])

    monkeypatch.setattr(fa_utils, "flash_attn_varlen_func", fake_flash_attn_varlen_func)
    query = torch.randn(1, total, 8, 128)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int32)
    metadata = AttentionMetadata(
        extra={
            "cu_seqlens_q": cu_seqlens,
            "cu_seqlens_k": cu_seqlens,
            "max_seqlen_q": 26,
            "max_seqlen_k": 26,
            "valid_kv_length": 26,
        }
    )

    output = _impl("token_refiner.blocks.0.attn")._forward_dense_varlen(
        query,
        query,
        query,
        metadata,
    )

    assert output.shape == query.shape
    assert captured["cu_seqlens_q"].tolist() == expected
    assert captured["cu_seqlens_k"].tolist() == expected


def test_sol_attn_spec_validation():
    spec = AttentionSpec(backend="SOL_ATTN", sol_attn={"tau": 1.0, "sink_tokens": 951})
    assert spec.sol_attn.tau == 1.0
    assert spec.sol_attn.sink_tokens == 951
    with pytest.raises(ValueError, match="sol_attn.thresh_type"):
        SolAttnSpec(thresh_type="bogus")
    with pytest.raises(ValueError, match="only supported by the SOL_ATTN"):
        AttentionSpec(backend="FLASH_ATTN", sol_attn={"tau": 1.0})


def test_sol_attn_spec_serialized_in_backend_kwargs():
    spec = AttentionSpec(
        backend="SOL_ATTN",
        sol_attn={"tau": 2.0, "thresh_type": "exact", "dense_layers": "0,1", "kv_splits": "auto"},
    )
    serialized = spec.backend_kwargs()["sol_attn"]
    assert serialized["tau"] == 2.0
    assert serialized["thresh_type"] == "exact"
    assert serialized["dense_layers"] == "0,1"
    assert serialized["kv_splits"] == "auto"


def test_sink_range_clamped_to_short_sequence():
    # The text-refiner attention runs on short text rows; a configured sink
    # larger than the sequence must be clamped instead of failing.
    assert SolAttnImpl._clamp_sink_range(used=200, sink_start=0, sink_tokens=951) == (0, 200)
    assert SolAttnImpl._clamp_sink_range(used=38000, sink_start=0, sink_tokens=951) == (0, 951)
    assert SolAttnImpl._clamp_sink_range(used=500, sink_start=400, sink_tokens=951) == (400, 100)
    assert SolAttnImpl._clamp_sink_range(used=500, sink_start=None, sink_tokens=951) == (0, 500)


def test_used_length_respects_packed_padding():
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

    impl = _impl("blocks.0.attn")
    metadata = AttentionMetadata(
        extra={
            "max_seqlen_q": 38000,
            "max_seqlen_k": 38000,
            "valid_kv_length": 38000,
        }
    )
    assert impl._used_length(metadata, seq_len=38064) == 38000
    assert impl._used_length(None, seq_len=38064) == 38064


@pytest.mark.skipif(
    importlib.util.find_spec("sol_attn") is None,
    reason="sol_attn package is not available",
)
def test_cuda_resolver_requires_sol_attn_package():
    from vllm_omni.platforms.cuda.platform import CudaOmniPlatform

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    path = CudaOmniPlatform.get_diffusion_attn_backend_cls(
        selected_backend="SOL_ATTN",
        head_size=128,
    )
    assert path.endswith("sol_attn.SolAttnBackend")
