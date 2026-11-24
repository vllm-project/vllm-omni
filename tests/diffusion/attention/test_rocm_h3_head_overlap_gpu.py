# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


def _exchange(tensor, group):
    received = torch.empty_like(tensor)
    dist.all_to_all_single(received, tensor, group=group)
    return received


@torch.inference_mode()
def _compare_on_rank(rank, rendezvous):
    from aiter import flash_attn_varlen_func

    from vllm_omni.diffusion.attention.backends.utils.rocm_h3_head_fmha import prepare_head_fmha
    from vllm_omni.diffusion.attention.parallel.rocm_h3_head_overlap import run_head_overlap

    device = torch.device("cuda", rank)
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(minutes=3))
    try:
        group = dist.new_group(list(range(4)), backend="nccl")
        generator = torch.Generator(device=device).manual_seed(1234 + rank)
        shape = (9440, 56, 128)
        q, k = [torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator) for _ in range(2)]
        v = torch.empty_strided(shape, (21504, 128, 1), device=device, dtype=q.dtype).normal_(generator=generator)
        cut = 37750
        bounds = torch.tensor([0, cut, 37760], dtype=torch.int32, device=device)
        compute = prepare_head_fmha(device)
        assert compute is not None
        result = run_head_overlap(q, k, v, bounds, group, compute)

        # The reference exchanges Q, K and V individually and attends to all 14
        # local heads together. Each rank owns different data and a different output slice.
        full_qkv = []
        for src in (q, k, v):
            send = src.reshape(9440, 4, 14, 128).transpose(0, 1).contiguous()
            full_qkv.append(_exchange(send, group).reshape(37760, 14, 128))
        expected = flash_attn_varlen_func(*full_qkv, bounds, bounds, cut, cut, softmax_scale=128**-0.5, causal=False)
        expected = _exchange(expected.reshape(4, 9440, 14, 128).contiguous(), group)
        expected = expected.transpose(0, 1).reshape(shape)
        assert result.shape == expected.shape and result.dtype == torch.bfloat16 and result.is_contiguous()
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@hardware_test(res={"rocm": "MI350"}, num_cards=4)
def test_head_overlap_matches_unsplit_ulysses(tmp_path):
    if torch.version.hip is None or not torch.accelerator.is_available() or torch.accelerator.device_count() < 4:
        pytest.skip("requires four gfx950 devices with RCCL")
    from vllm_omni.diffusion.attention.backends.utils.rocm_h3_head_fmha import prepare_head_fmha

    if any(prepare_head_fmha(torch.device("cuda", rank)) is None for rank in range(4)):
        pytest.skip("requires the supported gfx950 grouped AITER code object on all four devices")
    mp.spawn(_compare_on_rank, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=4, join=True)
