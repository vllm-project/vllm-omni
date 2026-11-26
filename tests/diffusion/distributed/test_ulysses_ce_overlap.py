# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Copy-engine head-chunk overlap against plain copy-engine Ulysses, bit for bit.

Run through the shared GPU scheduler with fresh worker processes.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_sp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.local_model, *hardware_marks(res={"cuda": "H100"}, num_cards=2)]


def _attention(q, k, v):
    # One head per call, so results do not depend on how many heads a call groups.
    heads = [
        F.scaled_dot_product_attention(*(x[:, :, h : h + 1].transpose(1, 2) for x in (q, k, v))).transpose(1, 2)
        for h in range(q.shape[2])
    ]
    return torch.cat(heads, dim=2)


def _worker(rank, init_method, world_size=2):
    from vllm_omni.diffusion.attention.parallel import ulysses_ce_overlap
    from vllm_omni.diffusion.distributed import a2a_permute

    device = torch.device("cuda", rank)
    current_omni_platform.set_device(device)
    init_distributed_environment(world_size=world_size, rank=rank, local_rank=rank, distributed_init_method=init_method)
    initialize_model_parallel(data_parallel_size=1, sequence_parallel_size=world_size, ulysses_degree=world_size)
    try:
        group = get_sp_group()
        strategy = UlyssesParallelAttention(group, 2, 1, False, ulysses_a2a_permute=True)
        torch.manual_seed(1234 + rank)
        checks = 0
        with torch.inference_mode():
            # MiniMax-H3's head layout (56 heads of 128) scaled down in sequence length. Mixed shapes share one
            # grow-only runner: smaller shapes reuse it, a longer sequence or more chunks grow it.
            shapes = ((384, 56, 1), (384, 56, 2), (384, 56, 7), (256, 56, 7), (384, 16, 4), (512, 56, 2))
            for seq, heads, chunks in shapes:
                for step in range(3):
                    q, k, v = (torch.randn(1, seq, heads, 128, device=device, dtype=torch.bfloat16) for _ in range(3))
                    gq, gk, gv, _, ctx = strategy.pre_attention(q, k, v, None)
                    expected = strategy.post_attention(_attention(gq, gk, gv), ctx).clone()
                    actual = ulysses_ce_overlap.run_ulysses_ce_overlap(
                        group.ulysses_group, q, k, v, _attention, chunks=chunks
                    )
                    torch.testing.assert_close(
                        actual, expected, rtol=0, atol=0, msg=lambda m: f"{seq=} {heads=} {chunks=}: {m}"
                    )
                    checks += 1
            assert len(ulysses_ce_overlap._RUNNERS) == 1
            print(f"CE overlap rank={rank}: {checks} bit-exact checks passed", flush=True)
    finally:
        a2a_permute.clear_a2a_permute_workspaces()
        assert not ulysses_ce_overlap._RUNNERS
        destroy_distributed_env()


def test_ce_overlap_matches_plain_ulysses(tmp_path):
    if torch.accelerator.device_count() < 2:
        pytest.skip("Requires two scheduler-allocated NVLink GPUs")
    torch.multiprocessing.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2)
