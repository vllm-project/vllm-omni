# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Provider contracts need neither an H3 model nor a CUDA runtime."""

import sys

import pytest
import torch

from vllm_omni.diffusion.attention.ops import flashinfer_block_sparse as provider
from vllm_omni.diffusion.attention.ops.block_sparse import block_map_to_indices, block_sparse_attn_bshd
from vllm_omni.diffusion.attention.ops.video_tiles import get_tile_metadata

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_sparse_map_conversion_keeps_scattered_and_empty_selections():
    block_map = torch.tensor([[[[False, True, False, True], [False, False, False, False], [True, True, True, True]]]])
    indices, counts = block_map_to_indices(block_map)
    assert indices.tolist() == [[[[1, 3, -1, -1], [-1, -1, -1, -1], [0, 1, 2, 3]]]]
    assert counts.tolist() == [[[2, 0, 4]]]
    assert indices.dtype == counts.dtype == torch.int32


@pytest.mark.parametrize("tile", [(4, 4, 4), (4, 8, 8)])
def test_shared_video_tiling_round_trips_ragged_grid(tile):
    grid = (5, 9, 10)
    rows = 5 * 9 * 10
    elements = tile[0] * tile[1] * tile[2]
    partition, sizes, non_pad, untile = get_tile_metadata(grid, tile, elements, torch.device("cpu"))
    source = torch.arange(rows)
    tiled = torch.full((sizes.numel() * elements,), -1)
    tiled[non_pad] = source[partition]
    torch.testing.assert_close(tiled[untile], source)
    assert sizes.sum().item() == rows


@pytest.mark.parametrize(
    "precision,capability,supported",
    [
        ("bf16", (12, 0), True),
        ("bf16", (12, 1), True),
        ("sage", (12, 0), True),
        ("sage", (12, 1), False),
        ("sage", (10, 0), False),
        ("bf16", (10, 0), False),
        ("bf16", (9, 0), False),
        ("bf16", (8, 0), False),
        ("sage", None, False),
    ],
)
def test_precision_specific_hardware_contract(precision, capability, supported):
    if supported:
        provider.validate_flashinfer_sparse_capability(precision, capability)
    else:
        with pytest.raises(ValueError, match="requires SM"):
            provider.validate_flashinfer_sparse_capability(precision, capability)


def _inputs():
    q = torch.randn(2, 129, 3, 128, dtype=torch.bfloat16)
    k = torch.randn(2, 191, 3, 128, dtype=torch.bfloat16)
    return q, k, k.clone(), torch.ones(2, 3, 3, 3, dtype=torch.bool), None


@pytest.mark.parametrize("precision", ["bf16", "sage"])
@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_both_precisions_reject_invalid_scale_before_import(precision, scale):
    with pytest.raises(ValueError, match="finite and positive"):
        block_sparse_attn_bshd(*_inputs(), scale, provider="flashinfer", precision=precision)


@pytest.mark.parametrize("precision", ["bf16", "sage"])
def test_rejects_wrong_key_block_count_even_without_sizes(precision):
    q, k, v, block_map, sizes = _inputs()
    with pytest.raises(ValueError, match="key blocks"):
        block_sparse_attn_bshd(q, k, v, block_map[..., :2], sizes, 0.125, provider="flashinfer", precision=precision)


def test_bf16_abi_is_bshd_tuple_without_sage_dependency(monkeypatch):
    q, k, v, block_map, sizes = _inputs()
    monkeypatch.setitem(sys.modules, "vllm_omni.diffusion.attention.ops.sage_quantization", None)
    monkeypatch.setitem(sys.modules, "vllm_omni.diffusion.attention.ops.sage_block_sparse_attention", None)
    calls = []

    def kernel(q_arg, k_arg, v_arg, indices, capacity, **kwargs):
        assert q_arg.shape == (2, 129, 3, 128)
        assert k_arg.shape == v_arg.shape == (2, 191, 3, 128)
        assert indices.shape == (2, 3, 3, 3) and capacity == 3
        assert kwargs["q2k_block_nums"].eq(3).all()
        return q_arg.clone(), None

    def require(precision, device):
        calls.append((precision, device))
        return kernel

    monkeypatch.setattr(provider, "require_flashinfer_sparse", require)
    actual = block_sparse_attn_bshd(q, k, v, block_map, sizes, 0.125, provider="flashinfer")
    torch.testing.assert_close(actual, q)
    assert actual.is_contiguous()
    assert calls == [("bf16", q.device)]


def test_missing_provider_api_is_clear(monkeypatch):
    from vllm_omni.platforms import current_omni_platform

    monkeypatch.setattr(current_omni_platform, "get_device_capability", lambda *_: (12, 0))
    monkeypatch.setitem(sys.modules, "flashinfer", None)
    with pytest.raises(ImportError, match="requires a build with bsa_attn_sm120_blk64_sage_fwd"):
        provider.require_flashinfer_sparse("sage")


def test_provider_uses_operand_device_for_hardware_check(monkeypatch):
    from vllm_omni.platforms import current_omni_platform

    seen = []

    def capability(device_id):
        seen.append(device_id)
        return (9, 0)

    monkeypatch.setattr(current_omni_platform, "get_device_capability", capability)
    with pytest.raises(ValueError, match="requires SM120"):
        provider.require_flashinfer_sparse("sage", torch.device("cuda:3"))
    assert seen == [3]
