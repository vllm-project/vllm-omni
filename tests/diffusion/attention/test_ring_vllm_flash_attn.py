# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Real-CUDA coverage for vLLM FlashAttention inside the ring path."""

import os

import pytest
import torch
import torch.distributed as dist

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import get_open_port
from vllm_omni.diffusion.attention.backends.ring.ring_selector import AttnType
from vllm_omni.diffusion.attention.backends.ring_flash_attn import ring_flash_attn_forward
from vllm_omni.diffusion.attention.backends.utils import fa
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _run_vllm_ring_parity(local_rank: int, world_size: int, master_port: int) -> None:
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
    try:
        dtype = torch.bfloat16
        batch_size, local_seqlen, num_heads, head_dim = 1, 19, 4, 64
        global_seqlen = local_seqlen * world_size
        softmax_scale = head_dim**-0.5

        torch.manual_seed(31)
        query = torch.randn(batch_size, global_seqlen, num_heads, head_dim, device=device, dtype=dtype)
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        local_query = query.chunk(world_size, dim=1)[local_rank].contiguous()
        local_key = key.chunk(world_size, dim=1)[local_rank].contiguous()
        local_value = value.chunk(world_size, dim=1)[local_rank].contiguous()

        capability = torch.cuda.get_device_capability(device)
        expected_version = 3 if capability[0] == 9 else 2
        assert capability[0] in (8, 9), (
            f"test expects an Ampere/Ada or Hopper GPU, got SM{capability[0]}{capability[1]}"
        )
        assert fa.resolve_vllm_flash_attn_version() == expected_version

        output, lse = ring_flash_attn_forward(
            dist.group.WORLD,
            local_query,
            local_key,
            local_value,
            softmax_scale=softmax_scale,
            causal=False,
            attn_type=AttnType.VLLM_FA,
        )

        query_bhsd = local_query.transpose(1, 2)
        key_bhsd = key.transpose(1, 2)
        value_bhsd = value.transpose(1, 2)
        output_ref = torch.nn.functional.scaled_dot_product_attention(
            query_bhsd,
            key_bhsd,
            value_bhsd,
            scale=softmax_scale,
        ).transpose(1, 2)
        logits = torch.matmul(query_bhsd.float(), key_bhsd.float().transpose(-2, -1)) * softmax_scale
        lse_ref = torch.logsumexp(logits, dim=-1)

        torch.testing.assert_close(output, output_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(lse, lse_ref, rtol=2e-3, atol=2e-3)
    finally:
        dist.destroy_process_group()


@hardware_test(res={"cuda": ["L4", "H100"]}, num_cards=2)
@pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="vLLM FlashAttention requires CUDA")
def test_vllm_flash_attention_ring_cuda_matches_full_attention():
    """Circulate K/V through two ranks and check output plus accumulated LSE."""
    world_size = 2
    if current_omni_platform.get_device_count() < world_size:
        pytest.skip(f"test requires {world_size} CUDA devices")
    torch.multiprocessing.spawn(
        _run_vllm_ring_parity,
        args=(world_size, get_open_port()),
        nprocs=world_size,
        join=True,
    )
