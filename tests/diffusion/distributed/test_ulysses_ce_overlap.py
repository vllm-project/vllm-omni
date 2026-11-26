# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Copy-engine head-chunk overlap against plain copy-engine Ulysses, bit for bit.

Run through the shared GPU scheduler with fresh worker processes.
"""

from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
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

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "H100"}, num_cards=2)]


def _attention(q, k, v):
    # One head per call, so results do not depend on how many heads a call groups.
    heads = [
        F.scaled_dot_product_attention(*(x[:, :, h : h + 1].transpose(1, 2) for x in (q, k, v))).transpose(1, 2)
        for h in range(q.shape[2])
    ]
    return torch.cat(heads, dim=2)


def _check_uneven(group, device, rank):
    from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionBackend
    from vllm_omni.diffusion.attention.parallel.head_overlap import head_policy
    from vllm_omni.diffusion.attention.parallel.ulysses_ce_overlap import CEOverlapUlysses

    plan = CEOverlapUlysses(group.ulysses_group, device=device)
    s, h, d = 127, 16, 64
    torch.manual_seed(31)
    prefix = tuple(torch.randn(1, 19, h, d, device=device, dtype=torch.bfloat16) for _ in range(2))
    torch.manual_seed(101 + rank)
    packed = torch.randn(1, s, 3, h, d, device=device, dtype=torch.bfloat16)
    inputs = packed.unbind(2)
    # A recoverable OOM on just one rank is voted on before the symmetric
    # rendezvous. Both ranks can still use native collectives afterwards.
    policy = head_policy.HeadPolicy(group.cpu_group, 120)
    strategy = SimpleNamespace(_sp_group=group, _ulysses_pg=group.ulysses_group, head_layer_name="self")
    layer = SimpleNamespace(
        causal=False,
        softmax_scale=d**-0.5,
        borrow_ulysses_output=False,
        effective_attention=lambda: (None, FlashAttentionBackend, None),
        _with_kv_cache_dtype=lambda md: md,
    )
    layer._run_local_attention = MethodType(lambda self, q, k, v, md: _attention(q, k, v), layer)

    def prepare(*args):
        if rank:
            raise torch.OutOfMemoryError("injected before rendezvous")
        return head_policy.HeadPolicy._prepare(*args)

    with (
        patch.object(head_policy, "candidate", return_value=(((8,), (3, 5)), 0)),
        patch.object(policy, "_prepare", prepare),
    ):
        policy.begin_component(None, (), {})
        assert policy.try_run(strategy, layer, *inputs, None) is None
        policy.finish_component(None, (), None)
    assert not policy.pending and not next(iter(policy.layouts.values())).selected
    # Other callers may retain views across any number of later calls.
    trial = next(iter(policy.layouts.values()))
    plan._configure(s, h, d, (3, 5), packed.dtype)
    trial.plan, trial.selected = plan, True
    policy.begin_component(None, (), {})
    retained = policy.try_run(strategy, layer, *inputs, None)[:, :3]
    copy = retained.clone()
    for scale in (0.5, -1.0, 2.0):
        scaled = (packed * scale).unbind(2)
        policy.try_run(strategy, layer, *scaled, None)
    torch.testing.assert_close(retained, copy, rtol=0, atol=0)
    peers = [torch.empty_like(packed) for _ in range(2)]
    dist.all_gather(peers, packed, group=group.ulysses_group)
    q, k, v = torch.cat(peers, dim=1).unbind(2)
    for front in (False, True):
        kv = [torch.cat((p, x) if front else (x, p), dim=1) for p, x in zip(prefix, (k, v))]
        expected = _attention(q, *kv)[:, rank * s : (rank + 1) * s]
        for tail in (0, s // 2, s):
            if tail == s:
                # Use production staging prep on the same strided QKV and both prefix placements.
                qkv_spec = tuple(head_policy.tensor_spec(x) for x in inputs)
                joint_spec = tuple(head_policy.tensor_spec(x) for x in (q[:, :0], *prefix))
                key = (2, qkv_spec, joint_spec, "front" if front else "rear")
                plan, _ = policy._prepare(key, head_policy.Trial((3, 5), tail), device, group.ulysses_group)
                plan.connect()
                assert len(plan.staging) == 2
            else:
                plan._configure(s, h, d, (3, 5), q.dtype, prefix=19, front=front, tail=tail)
            actual = plan.run(*inputs, _attention, joint_kv=prefix)
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=5e-3)

    # Change input values between graph replays. Stale peer signals otherwise
    # pass a constant-input comparison even if a rank reads the previous call.
    static = torch.empty_like(packed)
    static.copy_(packed)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = plan.run(*static.unbind(2), _attention, joint_kv=prefix).clone()
    for scale in (0.5, 2.0, -1.0):
        static.copy_(packed * scale)
        graph.replay()
        kv = [torch.cat((p, x * scale), dim=1) for p, x in zip(prefix, (k, v))]
        expected = _attention(q * scale, *kv)[:, rank * s : (rank + 1) * s]
        torch.testing.assert_close(captured, expected, rtol=1e-2, atol=5e-3)


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
            _check_uneven(group, device, rank)
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
