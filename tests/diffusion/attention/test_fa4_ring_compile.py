# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Real FA4/native-P2P regression tests; no vendor or collective mocks."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from packaging.version import Version

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.ring.ring_globals import HAS_FA4
from vllm_omni.diffusion.attention.backends.ring.ring_kernels import _fa4_ring_attention_op, flash_attn4_func_forward
from vllm_omni.diffusion.attention.backends.ring.ring_selector import AttnType
from vllm_omni.diffusion.attention.backends.ring_flash_attn import ring_flash_attn_forward
from vllm_omni.diffusion.attention.parallel.ring import RingParallelAttention
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _require_fa4():
    if not current_omni_platform.is_cuda() or not HAS_FA4:
        pytest.skip("Requires CuTe FA4 on Blackwell")
    capability = current_omni_platform.get_device_capability()
    if capability is None or capability.major < 10:
        pytest.skip("Requires Blackwell")


@hardware_test(res={"cuda": "B200"}, num_cards=1)
def test_fa4_ring_op_schema_and_dynamic_replay():
    _require_fa4()
    torch.compiler.reset()
    with torch.inference_mode():
        compiled = torch.compile(flash_attn4_func_forward, fullgraph=True, dynamic=True)
        for seq in (128, 256, 128):
            q = torch.randn(1, 4, seq, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
            k = torch.randn(1, seq + 23, 2, 128, device="cuda", dtype=q.dtype)
            v = torch.randn_like(k)
            before = tuple(t.clone() for t in (q, k, v))
            expected = flash_attn4_func_forward(q, k, v)
            actual = compiled(q, k, v)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            for a, b in zip((q, k, v), before):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.library.opcheck(_fa4_ring_attention_op, (q, k, v, None, False, -1, -1, 0.0))
    torch.compiler.reset()


def _ring_worker(rank, rendezvous):
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(minutes=10))
    try:
        groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        with torch._dynamo.config.patch(enable_p2p_compilation=True), torch.inference_mode():
            for group in (groups[rank // 2], dist.group.WORLD):
                runner = RingParallelAttention(SimpleNamespace(ring_group=group), "FLASH_ATTN", True)
                compiled = torch.compile(runner.run_attention, fullgraph=True, dynamic=True)
                # Isolate a 128-to-120 trim without a prefix, then cover GQA/prefix
                # padding and replay the original unpadded shape.
                cases = ((128, 4, 0, 0), (128, 4, 0, 8), (256, 2, 23, 7), (128, 4, 0, 0))
                for seq, kv_heads, prefix, trim in cases:
                    torch.manual_seed(42 + rank)
                    q = torch.randn(1, seq, 4, 128, device="cuda", dtype=torch.bfloat16)
                    k = torch.randn(1, seq, kv_heads, 128, device="cuda", dtype=q.dtype)
                    v = torch.randn_like(k)
                    # Replicated prefix must be identical on every rank.
                    torch.manual_seed(123)
                    jk = torch.randn(1, prefix, kv_heads, 128, device="cuda", dtype=q.dtype) if prefix else None
                    jv = torch.randn_like(jk) if prefix else None
                    valid = seq * dist.get_world_size(group) - trim
                    metadata = AttentionMetadata(
                        joint_key=jk,
                        joint_value=jv,
                        joint_strategy="front",
                        extra={"valid_kv_length": valid} if trim else {},
                    )
                    inputs = (q, k, v, jk, jv) if prefix else (q, k, v)
                    before = tuple(t.clone() for t in inputs)
                    eager = runner.run_attention(q, k, v, metadata)
                    actual = compiled(q, k, v, metadata)
                    gathered_k = [torch.empty_like(k) for _ in range(dist.get_world_size(group))]
                    gathered_v = [torch.empty_like(v) for _ in gathered_k]
                    dist.all_gather(gathered_k, k, group=group)
                    dist.all_gather(gathered_v, v, group=group)
                    full_k, full_v = torch.cat(gathered_k, 1)[:, :valid], torch.cat(gathered_v, 1)[:, :valid]
                    if prefix:
                        full_k, full_v = torch.cat((jk, full_k), 1), torch.cat((jv, full_v), 1)
                    full_k = full_k.repeat_interleave(4 // kv_heads, dim=2)
                    full_v = full_v.repeat_interleave(4 // kv_heads, dim=2)
                    reference = torch.nn.functional.scaled_dot_product_attention(
                        q.float().transpose(1, 2), full_k.float().transpose(1, 2), full_v.float().transpose(1, 2)
                    ).transpose(1, 2)
                    torch.testing.assert_close(actual.float(), reference, atol=1e-2, rtol=1e-2)
                    torch.testing.assert_close(actual, eager, atol=1e-2, rtol=1e-2)
                    for a, b in zip(inputs, before):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)

                # Nested inference capture returning LSE reproduces the alias
                # failure that the output-only strategy alone does not expose.
                @torch.inference_mode()
                def forward_with_lse(q, k, v, jk, jv):
                    return ring_flash_attn_forward(
                        group,
                        q,
                        k,
                        v,
                        128**-0.5,
                        causal=False,
                        attn_type=AttnType.FA4,
                        joint_tensor_key=jk,
                        joint_tensor_value=jv,
                    )

                q = torch.randn(1, 128, 32, 128, device="cuda", dtype=torch.bfloat16)
                k = torch.randn(1, 128, 8, 128, device="cuda", dtype=q.dtype)
                v = torch.randn_like(k)
                jk = torch.randn(1, 23, 8, 128, device="cuda", dtype=q.dtype)
                jv = torch.randn_like(jk)
                compiled_with_lse = torch.compile(forward_with_lse, fullgraph=True, dynamic=True)
                expected = forward_with_lse(q, k, v, jk, jv)
                actual = compiled_with_lse(q, k, v, jk, jv)
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, atol=1e-2, rtol=1e-2)
                torch.compiler.reset()
    finally:
        dist.destroy_process_group()


@hardware_test(res={"cuda": "B200"}, num_cards=4)
def test_fa4_ring_native_p2p_fullgraph(tmp_path):
    _require_fa4()
    if Version(torch.__version__.split("+")[0]) < Version("2.14.0"):
        pytest.skip("Native subgroup P2P compilation requires PyTorch 2.14")
    if torch.accelerator.device_count() < 4:
        pytest.skip("Requires four GPUs")
    mp.spawn(_ring_worker, args=(f"file://{tmp_path / 'ring-init'}",), nprocs=4, join=True)
