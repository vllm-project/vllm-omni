# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
import types
from typing import Any

import pytest
import torch

from vllm_omni.diffusion.attention.ops.ark_block_sparse import (
    _ark_block_sparse_parameters,
    block_map_to_lut,
    supports_ark_q256_multirow,
)
from vllm_omni.diffusion.attention.ops.block_sparse import (
    block_sparse_attn_bshd,
    resolve_block_sparse_provider,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _lut_to_block_map(lut: torch.Tensor, valid: torch.Tensor, key_blocks: int) -> torch.Tensor:
    positions = torch.cumsum(lut.to(torch.int64), dim=-1)
    block_map = torch.zeros_like(lut, dtype=torch.bool)
    block_map.scatter_(-1, positions, True)
    return block_map & (valid.unsqueeze(-1) > 0)


def test_block_map_to_lut_round_trips_including_empty_rows():
    torch.manual_seed(11)
    block_map = torch.rand(2, 3, 5, 7) < 0.45
    block_map[1, 2, 1] = False
    lut, valid = block_map_to_lut(block_map)

    assert lut.dtype == torch.int32 and valid.dtype == torch.int32
    assert valid.tolist() == block_map.sum(dim=-1).tolist()
    assert torch.equal(_lut_to_block_map(lut, valid, block_map.shape[-1]), block_map)

    # The invalid LUT tail must be zeroed so ARK never chases padding slots.
    invalid = torch.arange(block_map.shape[-1])[None, None, None, :] >= valid.unsqueeze(-1)
    assert torch.count_nonzero(lut[invalid]) == 0


def _install_fake_ark(monkeypatch, calls, *, with_k_block_valid=False, q256_multirow=True):
    package = types.ModuleType("auto_round_kernel")
    package.__path__ = []
    module = types.ModuleType("auto_round_kernel.sparse_attention")

    if q256_multirow:

        def _resolve_sparse_prefill_q_tile_override(
            *,
            head_dim,
            quant_block_size,
            q_tile_override,
            sparse_q_block_tokens,
            sparse_k_block_tokens,
            tensor_layout,
            allow_multi_row_tile=False,
        ):
            del head_dim, quant_block_size, sparse_k_block_tokens, tensor_layout
            if allow_multi_row_tile and q_tile_override == 256 and sparse_q_block_tokens == 64:
                return 256
            raise ValueError("not a q256 multi-row configuration")

        setattr(module, "_resolve_sparse_prefill_q_tile_override", _resolve_sparse_prefill_q_tile_override)

    if with_k_block_valid:

        def block_sparse_sdpa(query, key, value, lut, valid_block_num, *, k_block_valid=None, **kwargs):
            calls.append((lut, valid_block_num, k_block_valid, kwargs))
            return query.clone()

    else:

        def block_sparse_sdpa(query, key, value, lut, valid_block_num, **kwargs):
            calls.append((lut, valid_block_num, None, kwargs))
            return query.clone()

    setattr(module, "block_sparse_sdpa", block_sparse_sdpa)
    monkeypatch.setitem(sys.modules, "auto_round_kernel", package)
    monkeypatch.setitem(sys.modules, "auto_round_kernel.sparse_attention", module)
    _ark_block_sparse_parameters.cache_clear()
    supports_ark_q256_multirow.cache_clear()


def test_ark_provider_dispatches_lut_and_tile64_multirrow(monkeypatch):
    calls: list[Any] = []
    _install_fake_ark(monkeypatch, calls)
    query = torch.randn(1, 128, 2, 128, dtype=torch.bfloat16)
    block_map = torch.tensor([[[[True, False], [False, True]]]]).expand(1, 2, -1, -1)

    output = block_sparse_attn_bshd(
        query,
        query,
        query,
        block_map,
        torch.tensor([64, 64], dtype=torch.int32),
        128**-0.5,
        provider="xpu",
        block_size=64,
    )

    assert output is not None
    lut, valid, k_block_valid, kwargs = calls[0]
    assert lut.tolist() == [[[[0, 0], [1, 0]], [[0, 0], [1, 0]]]]
    assert valid.tolist() == [[[1, 1], [1, 1]]]
    assert k_block_valid is None
    assert kwargs["q_tile_override"] == 256
    assert kwargs["sparse_q_block_tokens"] == 64
    assert kwargs["sparse_k_block_tokens"] == 64
    assert kwargs["tensor_layout"] == "NHD"


def test_ark_provider_passes_partial_block_sizes_when_supported(monkeypatch):
    calls: list[Any] = []
    _install_fake_ark(monkeypatch, calls, with_k_block_valid=True)
    query = torch.randn(1, 256, 2, 128, dtype=torch.bfloat16)
    block_map = torch.ones(1, 2, 2, 2, dtype=torch.bool)
    sizes = torch.tensor([128, 96], dtype=torch.int32)

    block_sparse_attn_bshd(
        query,
        query,
        query,
        block_map,
        sizes,
        128**-0.5,
        provider="xpu",
        block_size=128,
    )

    _, _, k_block_valid, kwargs = calls[0]
    assert torch.equal(k_block_valid, torch.tensor([64, 64, 64, 32], dtype=torch.int32))
    assert kwargs["q_tile_override"] == 256
    assert kwargs["sparse_q_block_tokens"] == 64


def test_ark_tile64_prefers_q256_multirow_when_supported(monkeypatch):
    calls: list[Any] = []
    _install_fake_ark(monkeypatch, calls, q256_multirow=True)
    query = torch.randn(1, 256, 2, 128, dtype=torch.bfloat16)
    block_map = torch.ones(1, 2, 4, 4, dtype=torch.bool)

    block_sparse_attn_bshd(
        query,
        query,
        query,
        block_map,
        torch.full((4,), 64, dtype=torch.int32),
        128**-0.5,
        provider="xpu",
        block_size=64,
    )

    _, _, _, kwargs = calls[0]
    assert kwargs["q_tile_override"] == 256
    assert kwargs["sparse_q_block_tokens"] == 64


def test_ark_old_build_uses_q64_fallback(monkeypatch):
    calls: list[Any] = []
    _install_fake_ark(monkeypatch, calls, q256_multirow=False)
    query = torch.randn(1, 256, 2, 128, dtype=torch.bfloat16)
    block_map = torch.ones(1, 2, 4, 4, dtype=torch.bool)

    block_sparse_attn_bshd(
        query,
        query,
        query,
        block_map,
        torch.full((4,), 64, dtype=torch.int32),
        128**-0.5,
        provider="xpu",
        block_size=64,
    )

    _, _, _, kwargs = calls[0]
    assert kwargs["q_tile_override"] == 64


@pytest.mark.parametrize("block_size", [64, 128])
def test_auto_selects_xpu_on_xpu(monkeypatch, block_size):
    from vllm_omni.diffusion.attention.ops import ark_block_sparse

    monkeypatch.setattr(ark_block_sparse, "require_ark_sparse", lambda *args: None)
    resolve_block_sparse_provider.cache_clear()
    try:
        assert (
            resolve_block_sparse_provider(
                "auto",
                "bf16",
                torch.device("xpu"),
                torch.bfloat16,
                128,
                block_size,
            )
            == "xpu"
        )
    finally:
        resolve_block_sparse_provider.cache_clear()


def test_xpu_tile128_rejects_non_xpu_provider():
    resolve_block_sparse_provider.cache_clear()
    with pytest.raises(ValueError, match="unavailable on XPU"):
        resolve_block_sparse_provider(
            "fastvideo",
            "bf16",
            torch.device("xpu"),
            torch.bfloat16,
            128,
            128,
        )
