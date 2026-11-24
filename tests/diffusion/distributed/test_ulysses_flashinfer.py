# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""FlashInfer Ulysses transport against an independent distributed reference.

Run through the shared GPU scheduler with fresh worker processes.
"""

import pytest
import torch
import torch.distributed as dist

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_sp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.local_model, *hardware_marks(res={"cuda": "H100"}, num_cards=2)]


def _reference(x, mode, pg):
    peers = [torch.empty_like(x) for _ in range(dist.get_world_size(pg))]
    dist.all_gather(peers, x.contiguous(), group=pg)
    rank, size = dist.get_rank(pg), len(peers)
    axis = 2 if mode == 0 else 1
    width = x.shape[axis] // size
    slices = [peer.narrow(axis, rank * width, width) for peer in peers]
    return torch.cat(slices, dim=1 if mode == 0 else 2)


def _worker(rank, init_method, world_size=2):
    from vllm_omni.diffusion.distributed import a2a_permute, flashinfer_ulysses

    device = torch.device("cuda", rank)
    current_omni_platform.set_device(device)
    init_distributed_environment(world_size=world_size, rank=rank, local_rank=rank, distributed_init_method=init_method)
    initialize_model_parallel(data_parallel_size=1, sequence_parallel_size=world_size, ulysses_degree=world_size)
    try:
        group = get_sp_group()
        pg = group.ulysses_group
        strategy = UlyssesParallelAttention(
            group, 2, 1, False, ulysses_a2a_permute=True, ulysses_a2a_backend="flashinfer"
        )
        checks = 0
        torch.manual_seed(1234 + rank)
        with torch.inference_mode():
            for dtype in (torch.float16, torch.bfloat16):
                for batch, sequence, heads in ((1, 17, 8), (2, 65, 16), (1, 33, 8), (1, 17, 8)):
                    x = torch.randn(batch, sequence, heads, 128, device=device, dtype=dtype)[..., ::2]
                    kv = x[:, :, :4]
                    q, k, v, _, ctx = strategy.pre_attention(x, kv + 1, kv + 2, AttentionMetadata())
                    torch.testing.assert_close(q, _reference(x, 0, pg), rtol=0, atol=0)
                    torch.testing.assert_close(k, _reference(kv + 1, 0, pg), rtol=0, atol=0)
                    torch.testing.assert_close(v, _reference(kv + 2, 0, pg), rtol=0, atol=0)
                    arbitrary_output = torch.randn_like(q)
                    result = strategy.post_attention(arbitrary_output, ctx)
                    torch.testing.assert_close(result, _reference(arbitrary_output, 1, pg), rtol=0, atol=0)
                    checks += 4
            # gate_compress keeps Q/K/V on the copy engines; the output still returns over FlashInfer.
            gate = x + 3
            q, k, _, metadata, ctx = strategy.pre_attention(
                x, kv + 1, kv + 2, AttentionMetadata(extra={"gate_compress": gate})
            )
            torch.testing.assert_close(metadata.extra["gate_compress"], _reference(gate, 0, pg), rtol=0, atol=0)
            torch.testing.assert_close(q, _reference(x, 0, pg), rtol=0, atol=0)
            torch.testing.assert_close(k, _reference(kv + 1, 0, pg), rtol=0, atol=0)
            torch.testing.assert_close(strategy.post_attention(q, ctx), x, rtol=0, atol=0)
            checks += 4
            # Full-graph Dynamo capture must keep the group-bound ops opaque.
            compiled = torch.compile(
                lambda a, b, c: flashinfer_ulysses.ulysses_fi_qkv(a, b, c, pg.group_name, world_size),
                backend="eager",
                fullgraph=True,
            )
            for actual, source in zip(compiled(x, kv + 1, kv + 2), (x, kv + 1, kv + 2)):
                torch.testing.assert_close(actual, _reference(source, 0, pg), rtol=0, atol=0)
                checks += 1
            print(f"flashinfer rank={rank}: {checks} exact reference checks passed", flush=True)
    finally:
        for _, _, comm in flashinfer_ulysses._COMMUNICATORS.values():
            comm.close()
        flashinfer_ulysses._COMMUNICATORS.clear()
        flashinfer_ulysses._OUTPUTS.clear()
        a2a_permute.clear_a2a_permute_workspaces()
        destroy_distributed_env()


def test_flashinfer_ulysses_transport(tmp_path):
    pytest.importorskip("flashinfer.comm", exc_type=ImportError)
    if torch.accelerator.device_count() < 2:
        pytest.skip("Requires two scheduler-allocated NVLink GPUs")
    torch.multiprocessing.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2)
