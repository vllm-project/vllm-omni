# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Full distributed head-bucket attention versus an unbucketed reference.

CPU uses real Gloo collectives and SDPA with synchronous stream shims. NPU
uses real HCCL, streams and fused attention, including H3's Q/K norm + RoPE.
The reference gathers input tokens and uses untouched weights, never the
head-packing plan or output-assembly helpers under test.
"""

import copy
import sys
from contextlib import nullcontext
from datetime import timedelta
from importlib.machinery import ModuleSpec
from itertools import product
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _sdpa(q, k, v, ends):
    outputs, start = [], 0
    for end in ends:
        inputs = [t[start:end].transpose(0, 1).unsqueeze(0) for t in (q, k, v)]
        outputs.append(F.scaled_dot_product_attention(*inputs).squeeze(0).transpose(0, 1))
        start = end
    return torch.cat(outputs)


def _cpu_stream_shims(monkeypatch):
    # Patch only in spawned workers; the parent and other tests remain untouched.
    stream = SimpleNamespace(device="cpu", npu_stream=1, wait_event=lambda event: None)
    api = getattr(torch, "npu", SimpleNamespace())
    monkeypatch.setattr(torch, "npu", api, raising=False)
    for name, value in dict(
        Stream=lambda: stream,
        Event=lambda: SimpleNamespace(record=lambda stream: None),
        current_stream=lambda: stream,
        stream=lambda stream: nullcontext(),
    ).items():
        monkeypatch.setattr(api, name, value, raising=False)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None)
    npu = sys.modules.get("torch_npu", ModuleType("torch_npu"))
    if npu.__spec__ is None:
        npu.__spec__ = ModuleSpec("torch_npu", loader=None)
    monkeypatch.setattr(
        npu,
        "npu_fusion_attention",
        lambda q, k, v, heads, **kw: (_sdpa(q, k, v, kw["actual_seq_qlen"]),),
        raising=False,
    )
    monkeypatch.setitem(sys.modules, "torch_npu", npu)


class _Attention(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.num_heads = self.num_kv_heads = 16
        self.head_dim = 128
        self.qkv_proj = nn.Linear(128, 3 * 16 * 128, bias=bias)
        self.qkv_proj.total_num_heads = 16
        self.out_proj = nn.Linear(16 * 128, 128, bias=bias)
        method = type("UnquantizedLinearMethod", (), {})
        self.qkv_proj.quant_method = self.out_proj.quant_method = method()
        self.q_norm = nn.RMSNorm(128, eps=1e-6)
        self.k_norm = nn.RMSNorm(128, eps=1e-6)
        self.q_norm.variance_epsilon = 1e-6
        # Nonuniform scales also exercise correct norm/RoPE broadcasting.
        with torch.no_grad():
            self.q_norm.weight.uniform_(0.8, 1.2)
            self.k_norm.weight.uniform_(0.8, 1.2)
        self.to_gate_compress = None


def _gather(tensor, world):
    gathered = [torch.empty_like(tensor) for _ in range(world)]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered)


def _reference(module, x, rope, ends, rank, world):
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_qk_norm_rope

    tokens = x.shape[0]
    full_x = _gather(x, world)
    q, k, v = module.qkv_proj(full_x).reshape(world * tokens, 3, 16, 128).unbind(1)
    if rope is None:
        q, k = module.q_norm(q), module.k_norm(k)
    else:
        q, k = fused_qk_norm_rope(q, k, module.q_norm.weight, module.k_norm.weight, _gather(rope, world), 1e-6)
    if x.device.type == "cpu":
        attended = _sdpa(q, k, v, ends)
    else:
        import torch_npu

        attended = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            16,
            input_layout="TND",
            actual_seq_qlen=ends,
            actual_seq_kvlen=ends,
            scale=128**-0.5,
            keep_prob=1.0,
            sparse_mode=0,
        )[0]
    return module.out_proj(attended[rank * tokens : (rank + 1) * tokens].flatten(1))


def _worker(rank, world, init_method, device_type):
    # Spawned workers do not execute the root conftest bootstrap.
    from tests.model_executor.helpers import bootstrap_vllm_layer_custom_op_modules

    bootstrap_vllm_layer_custom_op_modules()
    # Import model dependencies before replacing the NPU API for CPU execution.
    from vllm_omni.diffusion import offloader  # noqa: F401
    from vllm_omni.diffusion.layers import fused_qk_norm_rope  # noqa: F401

    torch.set_num_threads(2)
    with pytest.MonkeyPatch.context() as monkeypatch:
        if device_type == "cpu":
            _cpu_stream_shims(monkeypatch)
            device, dtype = torch.device("cpu"), torch.float32
        else:
            import torch_npu  # noqa: F401

            torch.npu.set_device(rank)
            device, dtype = torch.device("npu", rank), torch.bfloat16
        from vllm_omni.diffusion.offloader.submodule.models.minimax_h3.h3_bucket_adapter import H3BucketAdapter

        dist.init_process_group(
            "gloo" if device_type == "cpu" else "hccl",
            init_method=init_method,
            rank=rank,
            world_size=world,
            timeout=timedelta(seconds=120),
        )
        try:
            with torch.inference_mode():
                for buckets, bias, padded, use_rope in product(
                    (1, 2, 3, 4), (False, True), (False, True), (False, True)
                ):
                    shared: dict = {}
                    layers = []
                    for layer in range(2):
                        torch.manual_seed(3100 + layer)
                        module = _Attention(bias).to(dtype=dtype)
                        reference = copy.deepcopy(module).to(device)
                        adapter = H3BucketAdapter(module, dist.group.WORLD, buckets, workspaces=shared)
                        layers.append((module.to(device), adapter, reference))
                    pending = []
                    # Reuse, then evict both cache entries; use distinct inputs on every call/rank.
                    for step, tokens in enumerate((16, 24, 16, 32, 24)):
                        torch.manual_seed(42 + rank + step * world)
                        x = torch.randn(tokens, 128, dtype=dtype).to(device)
                        theta = torch.randn(tokens, 48).to(device)
                        rope = torch.cat((theta.cos(), theta.sin()), dim=-1) if use_rope else None
                        total = tokens * world
                        ends = [total - 3, total] if padded else [total]
                        kwargs = dict(
                            rope_table=rope,
                            cu_seqlens=torch.tensor([0, *ends], dtype=torch.int32, device=device),
                            max_seqlen=ends[0],
                            packed_total=total,
                        )
                        for module, _, reference in layers:
                            pending.append((module(x, **kwargs), reference, x, rope, ends))
                    # No reference collectives or device synchronization between bucketed calls.
                    for actual, reference, x, rope, ends in pending:
                        expected = _reference(reference, x, rope, ends, rank, world)
                        rtol, atol = (2e-5, 2e-6) if device_type == "cpu" else (2e-2, 2e-3)
                        torch.testing.assert_close(
                            actual,
                            expected,
                            rtol=rtol,
                            atol=atol,
                            msg=lambda message: f"{rank=}, {buckets=}, {bias=}, {padded=}, {use_rope=}\n{message}",
                        )
                    if device_type == "npu":
                        torch.npu.synchronize()
                    for module, adapter, _ in layers:
                        module.cpu()
                        adapter.close()
        finally:
            dist.destroy_process_group()


@pytest.mark.cpu
@pytest.mark.parametrize("world", [2, 4])
def test_head_bucket_attention_gloo(tmp_path, world):
    mp.spawn(_worker, args=(world, (tmp_path / "rendezvous").as_uri(), "cpu"), nprocs=world, join=True)


@hardware_test(res={"npu": "A2"}, num_cards=4)
def test_head_bucket_attention_hccl(tmp_path, monkeypatch):
    pytest.importorskip("torch_npu")
    if torch.npu.device_count() < 4:
        pytest.skip("Requires four NPUs")
    monkeypatch.setenv("VLLM_TARGET_DEVICE", "npu")
    mp.spawn(_worker, args=(4, (tmp_path / "rendezvous").as_uri(), "npu"), nprocs=4, join=True)
