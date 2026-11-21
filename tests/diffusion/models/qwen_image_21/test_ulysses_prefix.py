# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project


import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from vllm_omni.diffusion.models.qwen_image_21.ulysses_prefix import PrefixLayout

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class Attention:
    @staticmethod
    def _run_local_attention(q, k, v, mask):
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask
        ).transpose(1, 2)

    @staticmethod
    def _with_kv_cache_dtype(mask):
        return mask


def make_mask(p, t):
    # Three prefix spans: text causal, one bidirectional reference block, more text.
    ids = torch.cat((torch.tensor([-1, -1, 1, 1, 1]), torch.full((p - 5,), -1), torch.full((t,), 2)))
    if p >= 13:
        ids[8:13] = 3  # Independent second reference; first reference must not see it.
    causal = torch.ones((p + t, p + t), dtype=torch.bool).tril()
    return (causal | ((ids[:, None] == ids[None, :]) & (ids[:, None] >= 0)))[None, None]


def run_case(rank, p, t, dtype):
    h, d = 4, 8
    torch.manual_seed(1947 + p)
    native = Attention()
    cache = [{}, {}]
    gold_prefix = []
    ops = [torch.randn(3, d, d, dtype=torch.float32) / d**0.5 for _ in cache]
    for step in range(3):
        decode = step > 0
        layout = PrefixLayout(dist.group.WORLD, rank, p, t // 2, decode)
        full = torch.randn(1, t if decode else p + t, h, d).to(dtype)
        target = full[:, -t:]
        part = target.chunk(2, dim=1)[rank]
        if not decode:
            part = torch.cat((layout.shard_prefix(full[:, :p], 1), part), dim=1)
        mask = None if decode else make_mask(p, t)
        for i, op in enumerate(ops):
            q, k, v = ((full.float() @ w).to(dtype) for w in op)
            local_qkv = [(part.float() @ w).to(dtype) for w in op]
            if decode:
                ck, cv = gold_prefix[i]
                k, v = torch.cat((ck, k), dim=1), torch.cat((cv, v), dim=1)
            else:
                gold_prefix.append((k[:, :p], v[:, :p]))
            full = full + native._run_local_attention(q, k, v, mask)
            part = part + layout.attention(native, *local_qkv, mask, cache[i], "cond")
            torch.testing.assert_close(
                cache[i]["cond"]["key"],
                gold_prefix[i][0][:, :, rank * 2 : rank * 2 + 2],
                atol=2e-2 if dtype == torch.bfloat16 else 2e-6,
                rtol=0,
            )
        result = part[:, layout.local_prefix :].contiguous()
        gathered = [torch.empty_like(result) for _ in range(2)]
        dist.all_gather(gathered, result)
        torch.testing.assert_close(
            torch.cat(gathered, dim=1), full[:, -t:], atol=3.2e-2 if dtype == torch.bfloat16 else 2e-6, rtol=0
        )


def worker(rank, rendezvous, backend):
    torch.set_num_threads(1)
    dist.init_process_group(backend, init_method=rendezvous, rank=rank, world_size=2)
    try:
        for dtype in (torch.float32, torch.bfloat16):
            for prefix, target in ((6, 8), (7, 8), (17, 4)):
                run_case(rank, prefix, target, dtype)
    finally:
        dist.destroy_process_group()


def test_ulysses_prefix_prefill_and_cached_decode(tmp_path):
    mp.spawn(worker, args=((tmp_path / "rendezvous").as_uri(), "gloo"), nprocs=2, join=True)


def test_prefill_marks_fresh_cache_and_rejects_partial_decode(monkeypatch):
    from types import SimpleNamespace

    from vllm_omni.diffusion.distributed import parallel_state
    from vllm_omni.diffusion.models.qwen_image_21 import ulysses_prefix as prefix

    parallel = SimpleNamespace(sequence_parallel_size=2, ulysses_degree=2, ring_degree=1, allgather_degree=1)
    model = SimpleNamespace(shard_edit_prefix=True, parallel_config=parallel)
    hidden = SimpleNamespace(shape=(2, 12, 32), device=SimpleNamespace(type="npu"))
    encoder = SimpleNamespace(shape=(2, 4, 32))
    group = SimpleNamespace(
        world_size=2, ulysses_world_size=2, ring_world_size=1, allgather_world_size=1, cpu_group=None
    )
    monkeypatch.setattr(parallel_state, "get_sp_group", lambda: group)
    monkeypatch.setattr(prefix, "_eligible", lambda *args: True)

    def vote(outputs, ballot, group):
        for output in outputs:
            output.copy_(ballot)

    monkeypatch.setattr(dist, "all_gather", vote)
    cache = [{} for _ in range(32)]
    assert prefix.select_group(model, hidden, encoder, 8, cache, "cond", "auto", False) is group
    assert prefix.cache_route_key(cache)
    with pytest.raises(RuntimeError, match="Incomplete or misaligned"):
        prefix.select_group(model, hidden, encoder, 8, cache, "cond", "auto", True)
