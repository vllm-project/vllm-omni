# SPDX-License-Identifier: Apache-2.0
"""CUDA-only regression test for the AR-Diffusion paged KV write op.

Split out of ``test_kv_cache.py``: that module is swept into CI's
``core_model and cpu`` job via a module-level ``pytest.mark.cpu``, and
``hardware_test(..., num_cards=1)`` adds no ``skipif`` guard for a missing
GPU, so this test was being collected (and would fail to import a CUDA
device) on CPU-only runners. Keeping it in its own, non-``cpu`` module means
it is only ever collected by a job that explicitly selects ``cuda`` for this
path -- none does yet, so for now this is intentionally not wired into any
CI job.
"""

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.experimental.ar_diffusion.kv_cache import allocate_kv_pool_with_views

BLOCK = 16

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_real_paged_write_op_does_not_clone_the_pool_on_cuda():
    """Same assertion as the CPU test, against the real op and CUDA codegen.

    The CPU test compiles a stand-in with the same ``mutates_args`` contract,
    because the real op needs a device and an attention kernel. That leaves two
    things unverified: whether the real op's full signature behaves the same,
    and whether Triton codegen does, which is the path the reported regression
    was observed on. This closes both.

    Note the pools passed here are the flat views, since that is what the op
    takes -- it indexes them by slot and unflattens internally.
    """
    import re

    from torch._inductor import config as inductor_config
    from torch._inductor.utils import run_and_get_code

    device = torch.device("cuda")
    num_blocks, heads, dim, num_tokens = 8, 4, 64, 8
    pool_numel = num_blocks * BLOCK * heads * dim

    class _Holder(torch.nn.Module):
        def __init__(self, key_flat, value_flat):
            super().__init__()
            self.register_buffer("key_flat", key_flat)
            self.register_buffer("value_flat", value_flat)

        def forward(self, query, k_curr, v_curr, video_slots, action_slots, block_table, query_start_loc, seq_lens):
            return torch.ops.vllm_omni.ar_diffusion_paged_write_attn(
                query,
                k_curr,
                v_curr,
                None,
                None,
                self.key_flat,
                self.value_flat,
                BLOCK,
                video_slots,
                action_slots,
                block_table,
                query_start_loc,
                seq_lens,
                num_tokens,
                num_tokens,
                dim**-0.5,
            )

    def compiled_pool_allocations(key_flat, value_flat) -> list[int]:
        args = (
            torch.randn(num_tokens, heads, dim, dtype=torch.bfloat16, device=device),
            torch.randn(num_tokens, heads, dim, dtype=torch.bfloat16, device=device),
            torch.randn(num_tokens, heads, dim, dtype=torch.bfloat16, device=device),
            torch.arange(num_tokens, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.arange(num_blocks, device=device, dtype=torch.int32).unsqueeze(0),
            torch.tensor([0, num_tokens], device=device, dtype=torch.int32),
            torch.tensor([num_tokens], device=device, dtype=torch.int32),
        )
        torch._dynamo.reset()
        # See the CPU test: inductor's cache outlives torch._dynamo.reset().
        with inductor_config.patch(force_disable_caches=True):
            _, codes = run_and_get_code(
                torch.compile(_Holder(key_flat, value_flat), backend="inductor", fullgraph=True), *args
            )
        sizes = []
        for shape in re.findall(r"empty_strided_cuda\(\(([\d, ]*?)\)", "\n".join(codes)):
            dims = [int(part) for part in shape.replace(" ", "").strip(",").split(",") if part]
            numel = 1
            for dim_size in dims:
                numel *= dim_size
            if dims and numel >= pool_numel:
                sizes.append(numel)
        return sizes

    # Positive control: one allocation, K and V as its two halves.
    shared = torch.empty(2, num_blocks * BLOCK, heads, dim, dtype=torch.bfloat16, device=device)
    control = compiled_pool_allocations(shared[0], shared[1])
    assert control, (
        "the shared layout produced no pool-sized allocation, so this test can no longer tell the two layouts apart"
    )

    _, k_pools, v_pools = allocate_kv_pool_with_views(
        num_blocks=num_blocks,
        block_size=BLOCK,
        num_layers=1,
        num_kv_heads=heads,
        head_dim=dim,
        dtype=torch.bfloat16,
        device=device,
    )
    separate = compiled_pool_allocations(k_pools[0], v_pools[0])
    assert not separate, (
        "the compiled graph allocates a pool-sized buffer, so the pool is being cloned "
        f"rather than written in place: {separate} (control saw {control})"
    )
