# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Production Ulysses transports against an independent distributed reference.

Run through the shared GPU scheduler; each backend gets fresh worker processes.
"""

import pytest
import torch
import torch.distributed as dist

from tests.helpers.runtime import get_distributed_init_method
from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_sp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.local_model, pytest.mark.cuda, pytest.mark.gpu, pytest.mark.cards_4]


def _reference(x, mode, pg):
    peers = [torch.empty_like(x) for _ in range(dist.get_world_size(pg))]
    dist.all_gather(peers, x.contiguous(), group=pg)
    rank, size = dist.get_rank(pg), len(peers)
    axis = 2 if mode == 0 else 1
    width = x.shape[axis] // size
    slices = [peer.narrow(axis, rank * width, width) for peer in peers]
    return torch.cat(slices, dim=1 if mode == 0 else 2)


def _worker(rank, init_method, backend):
    from vllm_omni.diffusion.distributed import a2a_permute, ce_ulysses

    device = torch.device("cuda", rank)
    current_omni_platform.set_device(device)
    init_distributed_environment(world_size=4, rank=rank, local_rank=rank, distributed_init_method=init_method)
    initialize_model_parallel(data_parallel_size=1, sequence_parallel_size=4, ulysses_degree=4)
    try:
        group = get_sp_group()
        strategy = UlyssesParallelAttention(group, 2, 1, False, ulysses_a2a_backend=backend)
        checks = 0
        torch.manual_seed(1234 + rank)
        with torch.inference_mode():
            for dtype in (torch.float16, torch.bfloat16):
                for batch, sequence, heads in ((1, 17, 8), (2, 65, 16), (1, 33, 8), (1, 17, 8)):
                    # Strided inputs exercise the functional API's packing contract.
                    x = torch.randn(batch, sequence, heads, 128, device=device, dtype=dtype)[..., ::2]
                    q, k, v, _, ctx = strategy.pre_attention(x, x + 1, x + 2, None)
                    torch.testing.assert_close(q, _reference(x, 0, group.ulysses_group), rtol=0, atol=0)
                    torch.testing.assert_close(k, _reference(x + 1, 0, group.ulysses_group), rtol=0, atol=0)
                    torch.testing.assert_close(v, _reference(x + 2, 0, group.ulysses_group), rtol=0, atol=0)
                    retained = q.clone()
                    arbitrary_output = torch.randn_like(q)
                    result = strategy.post_attention(arbitrary_output, ctx)
                    torch.testing.assert_close(
                        result, _reference(arbitrary_output, 1, group.ulysses_group), rtol=0, atol=0
                    )
                    torch.testing.assert_close(q, retained, rtol=0, atol=0)
                    checks += 5
            # Full-graph Dynamo capture must keep the group-bound op opaque.
            compiled_exchange = torch.compile(
                lambda tensor, mode: strategy._fused_exchange(tensor, mode, slot=0),
                backend="eager",
                fullgraph=True,
            )
            output = compiled_exchange(x, 0)
            torch.testing.assert_close(output, _reference(x, 0, group.ulysses_group), rtol=0, atol=0)
            print(f"native {backend} rank={rank}: {checks + 1} exact reference/lifetime checks passed", flush=True)
    finally:
        ce_ulysses.clear_ulysses_groups()
        a2a_permute.clear_a2a_permute_workspaces()
        destroy_distributed_env()


@pytest.mark.parametrize("backend", ["lsa", "ce"])
def test_native_ulysses_transport(backend):
    if torch.accelerator.device_count() < 4:
        pytest.skip("Requires four scheduler-allocated NVLink GPUs")
    if backend == "ce":
        pytest.importorskip("fast_ulysses")
    torch.multiprocessing.spawn(_worker, args=(get_distributed_init_method(), backend), nprocs=4)
