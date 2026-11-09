# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The copy-engine ring transport (``VLLM_OMNI_RING_CE_TRANSPORT=1``) must be bit-identical to ``RingComm``."""

import os

import pytest
import torch
import torch.distributed as dist

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import get_open_port
from vllm_omni.diffusion.attention.backends import ring_flash_attn
from vllm_omni.diffusion.attention.backends.ring import ce_transport
from vllm_omni.diffusion.attention.backends.ring.ring_selector import AttnType
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

# (local sequence, heads, head dim, dtype): the workspace grows, is reused smaller, changes dtype.
_SHAPES = [
    (19, 4, 64, torch.bfloat16),
    (64, 4, 64, torch.bfloat16),
    (33, 4, 64, torch.bfloat16),
    (64, 8, 128, torch.float16),
    (7, 4, 64, torch.bfloat16),
]


def _ring(q, k, v, causal: bool, ce: bool):
    os.environ["VLLM_OMNI_RING_CE_TRANSPORT"] = "1" if ce else "0"
    return ring_flash_attn.ring_flash_attn_forward(
        dist.group.WORLD,
        q,
        k,
        v,
        softmax_scale=q.shape[-1] ** -0.5,
        causal=causal,
        attn_type=AttnType.VLLM_FA,
    )


def _local_qkv(world_size, rank, seqlen, heads, head_dim, dtype, seed):
    torch.manual_seed(seed)
    return tuple(
        torch.randn(1, seqlen * world_size, heads, head_dim, device="cuda", dtype=dtype)
        .chunk(world_size, dim=1)[rank]
        .contiguous()
        for _ in range(3)
    )


def _run_ce_matches_ring_comm(local_rank: int, world_size: int, master_port: int) -> None:
    device = torch.device("cuda", local_rank)
    current_omni_platform.set_device(device)
    os.environ.update(
        RANK=str(local_rank),
        LOCAL_RANK=str(local_rank),
        WORLD_SIZE=str(world_size),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(master_port),
    )
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    finished = []
    finish = ce_transport.CeRingComm.finish

    def counting_finish(self):
        finished.append(1)
        finish(self)

    ce_transport.CeRingComm.finish = counting_finish
    try:
        with torch.inference_mode():
            for i, (seqlen, heads, head_dim, dtype) in enumerate(_SHAPES):
                q, k, v = _local_qkv(world_size, local_rank, seqlen, heads, head_dim, dtype, seed=100 + i)
                refs = {causal: _ring(q, k, v, causal, ce=False) for causal in (False, True)}
                # Back-to-back copy-engine calls, as consecutive layers issue them.
                for causal in (False, True, False):
                    before = len(finished)
                    out, lse = _ring(q, k, v, causal, ce=True)
                    assert len(finished) == before + 1, "the copy-engine transport was not used"
                    ref_out, ref_lse = refs[causal]
                    assert torch.equal(out, ref_out), (seqlen, heads, head_dim, dtype, causal)
                    assert torch.equal(lse, ref_lse), (seqlen, heads, head_dim, dtype, causal)
    finally:
        ce_transport.CeRingComm.finish = finish
        os.environ.pop("VLLM_OMNI_RING_CE_TRANSPORT", None)
        dist.destroy_process_group()


def _run_ce_falls_back(local_rank: int, world_size: int, master_port: int) -> None:
    device = torch.device("cuda", local_rank)
    current_omni_platform.set_device(device)
    os.environ.update(
        RANK=str(local_rank),
        LOCAL_RANK=str(local_rank),
        WORLD_SIZE=str(world_size),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(master_port),
    )
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    get_buffer = ce_transport._get_symm_buffer

    def failing_get_buffer(*args, **kwargs):
        raise RuntimeError("no P2P mapping")

    try:
        with torch.inference_mode():
            q, k, v = _local_qkv(world_size, local_rank, 19, 4, 64, torch.bfloat16, seed=7)
            ref_out, ref_lse = _ring(q, k, v, False, ce=False)
            # A failed rendezvous or peer mapping falls back to RingComm.
            ce_transport._get_symm_buffer = failing_get_buffer
            out, lse = _ring(q, k, v, False, ce=True)
            assert torch.equal(out, ref_out) and torch.equal(lse, ref_lse)
            assert not ce_transport._CREDITED
    finally:
        ce_transport._get_symm_buffer = get_buffer
        os.environ.pop("VLLM_OMNI_RING_CE_TRANSPORT", None)
        dist.destroy_process_group()


def _spawn(fn, world_size: int, needs_peer_access: bool = False) -> None:
    if current_omni_platform.get_device_count() < world_size:
        pytest.skip(f"test requires {world_size} CUDA devices")
    if needs_peer_access:
        # The transport maps every peer's buffer, so it needs CUDA peer access between all GPU
        # pairs. PCIe hosts without P2P (such as the L4 lane) fall back to RingComm, which
        # test_ce_transport_falls_back_when_peer_mapping_fails covers.
        missing = [
            (i, j)
            for i in range(world_size)
            for j in range(world_size)
            if i != j and not torch.cuda.can_device_access_peer(i, j)
        ]
        if missing:
            pytest.skip(f"the copy-engine transport needs CUDA peer access; GPU pairs without it: {missing[:4]}")
    torch.multiprocessing.spawn(fn, args=(world_size, get_open_port()), nprocs=world_size, join=True)


@hardware_test(res={"cuda": ["L4", "H100"]}, num_cards=2)
@pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="the copy-engine transport requires CUDA")
def test_ce_transport_matches_ring_comm_2_ranks():
    _spawn(_run_ce_matches_ring_comm, 2, needs_peer_access=True)


@hardware_test(res={"cuda": ["H100"]}, num_cards=8)
@pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="the copy-engine transport requires CUDA")
def test_ce_transport_matches_ring_comm_8_ranks():
    _spawn(_run_ce_matches_ring_comm, 8, needs_peer_access=True)


@hardware_test(res={"cuda": ["L4", "H100"]}, num_cards=2)
@pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="the copy-engine transport requires CUDA")
def test_ce_transport_falls_back_when_peer_mapping_fails():
    _spawn(_run_ce_falls_back, 2)


@pytest.mark.cpu
def test_ce_transport_falls_back_to_ring_comm_off_cuda(monkeypatch):
    """CPU tensors cannot use the copy-engine transport, so the ring keeps RingComm."""

    class SingleRankComm:
        rank = 0
        world_size = 1
        constructed = 0

        def __init__(self, _group):
            SingleRankComm.constructed += 1

    def fake_attention(q, _k, _v, **_kwargs):
        return torch.zeros_like(q), torch.zeros(q.shape[0], q.shape[2], q.shape[1])

    monkeypatch.setenv("VLLM_OMNI_RING_CE_TRANSPORT", "1")
    monkeypatch.setattr(ring_flash_attn, "RingComm", SingleRankComm)
    monkeypatch.setattr(ring_flash_attn, "select_flash_attn_impl", lambda *_args, **_kwargs: fake_attention)
    qkv = torch.randn(1, 4, 2, 8)

    output, _ = ring_flash_attn.ring_flash_attn_forward(
        object(), qkv, qkv, qkv, softmax_scale=0.25, causal=False, attn_type=AttnType.FA
    )

    assert output.shape == qkv.shape
    assert SingleRankComm.constructed == 1
