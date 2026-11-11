# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CE Ulysses transport against an independent distributed reference.

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

pytestmark = [pytest.mark.local_model, *hardware_marks(res={"cuda": "H100"}, num_cards=4)]


def _reference(x, mode, pg):
    peers = [torch.empty_like(x) for _ in range(dist.get_world_size(pg))]
    dist.all_gather(peers, x.contiguous(), group=pg)
    rank, size = dist.get_rank(pg), len(peers)
    axis = 2 if mode == 0 else 1
    width = x.shape[axis] // size
    slices = [peer.narrow(axis, rank * width, width) for peer in peers]
    return torch.cat(slices, dim=1 if mode == 0 else 2)


def _worker(rank, init_method, world_size=4):
    from vllm_omni.diffusion.distributed import a2a_permute

    device = torch.device("cuda", rank)
    current_omni_platform.set_device(device)
    init_distributed_environment(world_size=world_size, rank=rank, local_rank=rank, distributed_init_method=init_method)
    initialize_model_parallel(data_parallel_size=1, sequence_parallel_size=world_size, ulysses_degree=world_size)
    try:
        group = get_sp_group()
        strategy = UlyssesParallelAttention(group, 2, 1, False, ulysses_a2a_permute=True)

        def exchange_qkv(q, k, v):
            return a2a_permute.ulysses_qkv_fwd_group(q, k, v, group.ulysses_group.group_name, world_size)

        checks = 0
        torch.manual_seed(1234 + rank)
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream), torch.inference_mode():
            for dtype in (torch.float16, torch.bfloat16):
                for batch, sequence, heads in ((1, 17, 8), (2, 65, 16), (1, 33, 8), (1, 17, 8)):
                    # Strided inputs exercise the functional API's packing contract.
                    x = torch.randn(batch, sequence, heads, 128, device=device, dtype=dtype)[..., ::2]
                    # CE groups Q/K/V while allowing fewer K/V heads (GQA).
                    kv = x[:, :, :4]
                    gate = x + 3
                    metadata = AttentionMetadata(extra={"gate_compress": gate})
                    q, k, v, metadata, ctx = strategy.pre_attention(x, kv + 1, kv + 2, metadata)
                    torch.testing.assert_close(
                        metadata.extra["gate_compress"], _reference(gate, 0, group.ulysses_group), rtol=0, atol=0
                    )
                    torch.testing.assert_close(q, _reference(x, 0, group.ulysses_group), rtol=0, atol=0)
                    torch.testing.assert_close(k, _reference(kv + 1, 0, group.ulysses_group), rtol=0, atol=0)
                    torch.testing.assert_close(v, _reference(kv + 2, 0, group.ulysses_group), rtol=0, atol=0)
                    retained = q.clone()
                    arbitrary_output = torch.randn_like(q)
                    result = strategy.post_attention(arbitrary_output, ctx)
                    torch.testing.assert_close(
                        result, _reference(arbitrary_output, 1, group.ulysses_group), rtol=0, atol=0
                    )
                    torch.testing.assert_close(q, retained, rtol=0, atol=0)
                    checks += 6
            # Full-graph Dynamo capture must keep the group-bound op opaque.
            compiled_exchange = torch.compile(exchange_qkv, backend="eager", fullgraph=True)
            for actual, source in zip(compiled_exchange(x, kv + 1, kv + 2), (x, kv + 1, kv + 2)):
                torch.testing.assert_close(actual, _reference(source, 0, group.ulysses_group), rtol=0, atol=0)
                checks += 1
            # Capture without nesting the transport-local graph, then replay
            # with new input values.
            exchange_qkv(x, kv + 1, kv + 2)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured_q, _, _ = exchange_qkv(x, kv + 1, kv + 2)
                captured_round_trip = a2a_permute.ulysses_o_rev(captured_q, group.ulysses_group.group_name, world_size)
            x.add_(1)
            graph.replay()
            torch.testing.assert_close(captured_q, _reference(x, 0, group.ulysses_group), rtol=0, atol=0)
            torch.testing.assert_close(captured_round_trip, x, rtol=0, atol=0)
            checks += 2
            print(f"native CE rank={rank}: {checks} exact reference/lifetime checks passed", flush=True)
    finally:
        a2a_permute.clear_a2a_permute_workspaces()
        destroy_distributed_env()


def test_native_ulysses_transport(tmp_path):
    if torch.accelerator.device_count() < 4:
        pytest.skip("Requires four scheduler-allocated NVLink GPUs")
    torch.multiprocessing.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=4)
