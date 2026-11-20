# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Strict Ulysses with the all-to-all overlapped against head-chunked attention."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from vllm.utils.network_utils import get_file_store_init_method

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention import layer as layer_mod
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


# --------------------------------------------------------------------------- config


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"ulysses_overlap_chunks": -1}, ">= 0"),
        ({"ulysses_overlap_int8": True}, "requires ulysses_overlap_chunks > 0"),
        ({"ulysses_overlap_chunks": 2, "ring_degree": 2}, "strict pure Ulysses"),
        ({"ulysses_overlap_chunks": 2, "ulysses_mode": "advanced_uaa"}, "strict pure Ulysses"),
        ({"ulysses_overlap_chunks": 2, "ulysses_a2a_permute": True}, "strict pure Ulysses"),
    ],
)
def test_overlap_config_rejects_unsupported_combinations(kwargs, match):
    with pytest.raises(ValueError, match=match):
        DiffusionParallelConfig(ulysses_degree=2, **kwargs)


@pytest.mark.cpu
def test_overlap_config_accepts_strict_pure_ulysses():
    config = DiffusionParallelConfig(ulysses_degree=2, ulysses_overlap_chunks=2, ulysses_overlap_int8=True)
    assert config.ulysses_overlap_chunks == 2 and config.ulysses_overlap_int8 is True


# --------------------------------------------------------------------------- gating


class _Shape:
    """Stands in for a CUDA tensor: the gate only reads shape, rank and device."""

    def __init__(self, shape, is_cuda=True):
        self.shape = torch.Size(shape)
        self.is_cuda = is_cuda

    def dim(self):
        return len(self.shape)


def _strategy(chunks=2, int8=False, ulysses=4, ring=1, permute=False, name="ulysses"):
    return SimpleNamespace(
        name=name,
        overlap_chunks=chunks,
        overlap_int8=int8,
        _ulysses_a2a_permute=permute,
        _scatter_idx=2,
        _gather_idx=1,
        _sp_group=SimpleNamespace(ring_world_size=ring, ulysses_world_size=ulysses),
    )


def _gate(strategy, shape=(1, 16, 56, 128), metadata=None, *, paged=False, use_ring=False, cross=False, cuda=True):
    attention = SimpleNamespace(use_ring=use_ring, _explicit_cross_contract=cross)
    q = _Shape(shape, is_cuda=cuda)
    return Attention._ulysses_overlap_chunks(attention, strategy, q, q, q, metadata, paged)


@pytest.mark.cpu
def test_gate_enables_eligible_strict_ulysses_call():
    assert _gate(_strategy()) == 2
    # MiniMax-H3 publishes a contiguous valid K/V prefix instead of a mask; that stays eligible.
    assert _gate(_strategy(), metadata=AttentionMetadata(extra={"valid_kv_length": 9})) == 2
    # int8 with a non-power-of-two head dim keeps the overlap; the runner sends bf16 instead.
    assert _gate(_strategy(int8=True), shape=(1, 16, 56, 96)) == 2


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("case", "kwargs"),
    [
        ("disabled", {"strategy": _strategy(chunks=0)}),
        ("not ulysses", {"strategy": _strategy(name="ring")}),
        ("hybrid ring", {"strategy": _strategy(ring=2)}),
        ("a2a permute", {"strategy": _strategy(permute=True)}),
        ("heads not divisible", {"strategy": _strategy(chunks=3)}),
        ("batch > 1", {"strategy": _strategy(), "shape": (2, 16, 56, 128)}),
        ("paged kv", {"strategy": _strategy(), "paged": True}),
        ("ring layer", {"strategy": _strategy(), "use_ring": True}),
        ("explicit cross", {"strategy": _strategy(), "cross": True}),
        ("cpu tensor", {"strategy": _strategy(), "cuda": False}),
        (
            "mask",
            {"strategy": _strategy(), "metadata": AttentionMetadata(attn_mask=torch.ones(1, 4, dtype=torch.bool))},
        ),
        ("joint", {"strategy": _strategy(), "metadata": AttentionMetadata(joint_query=torch.zeros(1))}),
        ("gate", {"strategy": _strategy(), "metadata": AttentionMetadata(extra={"gate_compress": torch.zeros(1)})}),
    ],
)
def test_gate_falls_back_to_standard_path(case, kwargs):
    strategy = kwargs.pop("strategy")
    assert _gate(strategy, **kwargs) == 0, case


@pytest.mark.cpu
def test_gate_requires_strict_mode(monkeypatch):
    monkeypatch.setattr(layer_mod, "get_ulysses_mode", lambda default="strict": "advanced_uaa")
    assert _gate(_strategy()) == 0


# --------------------------------------------------------------------------- int8 pack / unpack (1 GPU)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("chunks", [1, 2])
def test_int8_pack_unpack_round_trip_reads_strided_source(chunks):
    from vllm_omni.diffusion.attention.parallel.ulysses_overlap import ROW_BYTES_EXTRA, pack_int8, unpack_int8

    P, S, H, D = 4, 37, 16, 128
    hc = H // (P * chunks)
    torch.manual_seed(0)
    fused = torch.randn(S, 3, H, D, device="cuda", dtype=torch.bfloat16)  # (S, H, D) views are strided
    srcs = fused.unbind(1)
    for c in range(chunks):
        buf = torch.empty(P, S, 3, hc, D + ROW_BYTES_EXTRA, device="cuda", dtype=torch.uint8)
        for t, x in enumerate(srcs):
            pack_int8(x, buf, chunk=c, t=t, n_t=3, world_size=P, hc=hc)
        for t, x in enumerate(srcs):
            out = torch.empty(P * S, hc, D, device="cuda", dtype=torch.bfloat16)
            unpack_int8(buf, out, t=t, n_t=3)
            # Without an exchange, slot p holds the rows destined for rank p: heads c*P*hc + p*hc + j.
            expected = x.view(S, chunks, P, hc, D)[:, c].permute(1, 0, 2, 3).reshape(P * S, hc, D)
            step = expected.float().abs().amax(dim=-1, keepdim=True) / 127
            err = (out.float() - expected.float()).abs()
            assert torch.all(err <= step * 0.51 + expected.float().abs() * 2**-7 + 1e-6)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_int8_pack_unpack_offsets_past_int32():
    from vllm_omni.diffusion.attention.parallel.ulysses_overlap import ROW_BYTES_EXTRA, pack_int8, unpack_int8

    # Source element offsets (s * 3*H*D) and packed byte offsets (row * row_bytes) both pass 2**31 at the tail.
    P, S, H, D = 4, 352_000, 16, 128
    hc = H // P
    assert (S - 1) * 3 * H * D >= 2**31 and P * S * 3 * hc * (D + ROW_BYTES_EXTRA) >= 2**31
    fused = torch.zeros(S, 3, H, D, device="cuda", dtype=torch.bfloat16)
    tail = torch.randn(8, 3, H, D, device="cuda", dtype=torch.bfloat16)
    fused[-8:] = tail
    buf = torch.empty(P, S, 3, hc, D + ROW_BYTES_EXTRA, device="cuda", dtype=torch.uint8)
    for t, x in enumerate(fused.unbind(1)):
        pack_int8(x, buf, chunk=0, t=t, n_t=3, world_size=P, hc=hc)
    del fused
    out = torch.empty(P * S, hc, D, device="cuda", dtype=torch.bfloat16)
    unpack_int8(buf, out, t=2, n_t=3)
    got = out.view(P, S, hc, D)[:, -8:].permute(1, 0, 2, 3).reshape(8, H, D).float()
    expected = tail[:, 2].float()
    step = expected.abs().amax(dim=-1, keepdim=True) / 127
    assert torch.all((got - expected).abs() <= step * 0.51 + expected.abs() * 2**-7 + 1e-6)


# --------------------------------------------------------------------------- overlapped exchange (2 GPUs)


def _sdpa_nhd(q, k, v):
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)


def _run_overlap_case(rank: int, world_size: int, init_method: str) -> None:
    import torch.distributed as dist

    from vllm_omni.diffusion.attention.parallel.ulysses_overlap import run_ulysses_overlap

    current_omni_platform.set_device(torch.device("cuda", rank))
    dist.init_process_group("nccl", init_method=init_method, world_size=world_size, rank=rank)
    try:
        S, H = 96, 8
        # D=96: int8 falls back to the bf16 exchange. transposed: the head dim is not contiguous.
        for D, transposed in ((128, False), (96, False), (128, True)):
            g = torch.Generator(device="cuda").manual_seed(0)
            q_g, k_g, v_g = (
                torch.randn(1, world_size * S, H, D, device="cuda", dtype=torch.bfloat16, generator=g) for _ in range(3)
            )
            ref = _sdpa_nhd(q_g, k_g, v_g)[:, rank * S : (rank + 1) * S].float()
            q, k, v = (x[:, rank * S : (rank + 1) * S].contiguous() for x in (q_g, k_g, v_g))
            if transposed:
                q, k, v = (x.transpose(-1, -2).contiguous().transpose(-1, -2) for x in (q, k, v))
                assert q.stride(-1) != 1
            for chunks in (1, 2):
                exact = run_ulysses_overlap(dist.group.WORLD, q, k, v, _sdpa_nhd, chunks=chunks, int8=False)
                assert exact.shape == q.shape
                torch.testing.assert_close(exact.float(), ref, atol=2e-2, rtol=2e-2)
                lossy = run_ulysses_overlap(dist.group.WORLD, q, k, v, _sdpa_nhd, chunks=chunks, int8=True)
                cos = F.cosine_similarity(lossy.float().flatten(), ref.flatten(), dim=0).item()
                assert cos > 0.9995, (D, transposed, chunks, cos)
    finally:
        dist.destroy_process_group()


@hardware_test(res={"cuda": "L4"}, num_cards=2)
def test_overlapped_exchange_matches_full_attention():
    world_size = 2
    torch.multiprocessing.spawn(_run_overlap_case, args=(world_size, get_file_store_init_method()), nprocs=world_size)
