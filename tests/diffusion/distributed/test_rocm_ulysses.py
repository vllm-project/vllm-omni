# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Packed RCCL Ulysses versus independently gathered rank/head slices."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention

pytestmark = [pytest.mark.local_model, *hardware_marks(res={"rocm": "MI325"}, num_cards=4)]


def _reference(tensor, rank, scatter, gather):
    peers = [torch.empty_like(tensor.contiguous()) for _ in range(4)]
    dist.all_gather(peers, tensor.contiguous())
    return torch.cat([x.chunk(4, dim=scatter)[rank] for x in peers], dim=gather)


def _worker(rank, rendezvous):
    device = torch.device("cuda", rank)
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=4)
    try:
        group = SimpleNamespace(
            ulysses_group=dist.group.WORLD,
            ulysses_world_size=4,
            ulysses_rank=rank,
            ring_world_size=1,
            allgather_world_size=1,
        )
        strategy = UlyssesParallelAttention(group, 2, 1, False, ulysses_a2a_permute=True)
        torch.manual_seed(10 + rank)
        with torch.inference_mode():
            for rows in (17, 33):
                q, k, v = torch.randn((1, rows, 3, 56, 128), device=device, dtype=torch.bfloat16).unbind(2)
                # Only one rank needs scalar-safe packing. Its peers keep vectorizing.
                if rank == 1:
                    storage = q.new_empty(q.numel() + 1)
                    v = storage[1:].view_as(q).copy_(v)
                # Only two ranks accept interleaved Q/K/V; all use the same collective.
                strided = rank % 2 == 0
                metadata = AttentionMetadata(extra={"accepts_strided_ulysses_qkv": strided})
                Q, K, V, _, ctx = strategy.pre_attention(q, k, v, metadata)
                assert ctx.rocm_packed
                for received, tensor in zip((Q, K, V), (q, k, v)):
                    torch.testing.assert_close(received, _reference(tensor, rank, 2, 1), rtol=0, atol=0)
                    assert received.is_contiguous() == (not strided)
                    if strided:
                        assert received.stride()[1:] == (3 * 14 * 128, 128, 1)
                # The reverse exchange must also handle a non-contiguous consumer result.
                out = torch.randn(1, 4 * rows, 14, 256, device=device, dtype=q.dtype)[..., ::2]
                actual = strategy.post_attention(out, ctx)
                assert actual.is_contiguous()
                torch.testing.assert_close(actual, _reference(out, rank, 1, 2), rtol=0, atol=0)
                torch.testing.assert_close(strategy.post_attention(Q, ctx), q, rtol=0, atol=0)

            # Replay the receive views and vectorized restore with new contents.
            def round_trip():
                received, _, _, _, context = strategy.pre_attention(q, k, v, metadata)
                return received, strategy.post_attention(received, context)

            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                round_trip()
            torch.cuda.current_stream().wait_stream(stream)
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured, restored = round_trip()
            q.add_(1)
            graph.replay()
            torch.testing.assert_close(captured, _reference(q, rank, 2, 1), rtol=0, atol=0)
            torch.testing.assert_close(restored, q, rtol=0, atol=0)
            # Release references to the captured communicator before destroying it.
            graph.reset()
            # A different head count should continue to use generic Ulysses on every rank.
            q = q[:, :, :8]
            Q, _, _, _, ctx = strategy.pre_attention(q, q, q, None)
            assert not ctx.rocm_packed
            torch.testing.assert_close(strategy.post_attention(Q, ctx), q, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_packed_rocm_ulysses(tmp_path):
    if torch.version.hip is None or torch.accelerator.device_count() < 4:
        pytest.skip("Requires four ROCm GPUs")
    torch.multiprocessing.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=4)
