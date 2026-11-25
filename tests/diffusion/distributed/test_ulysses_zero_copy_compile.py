# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Zero-copy CE Q/K/V views under torch.compile.

The views alias the persistent symmetric workspace. Compiled callers must see the same values as the copy-out path,
and no compiled result may live in the workspace, where the next exchange would overwrite it.

Run through the shared GPU scheduler with fresh worker processes.
"""

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_sp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.local_model, *hardware_marks(res={"cuda": "H100"}, num_cards=2)]


def _worker(rank, init_method, world_size=2):
    from vllm_omni.diffusion.distributed import a2a_permute

    device = torch.device("cuda", rank)
    current_omni_platform.set_device(device)
    init_distributed_environment(world_size=world_size, rank=rank, local_rank=rank, distributed_init_method=init_method)
    initialize_model_parallel(data_parallel_size=1, sequence_parallel_size=world_size, ulysses_degree=world_size)
    try:
        group_name = get_sp_group().ulysses_group.group_name

        def step(q, k, v, zero_copy):
            q, k, v = a2a_permute.ulysses_qkv_fwd_group(q, k, v, group_name, world_size, zero_copy)
            # Power-of-two scales and FP32 keep eager and Inductor arithmetic bit-identical.
            q = q * 0.5  # pointwise on a view of the workspace
            out = q + k * 2.0 + v * 4.0  # attention stand-in, (B, S, H/P, D)
            # The reverse exchange overwrites the workspace.
            o = a2a_permute.ulysses_o_rev(out, group_name, world_size)
            # Same size as the packed Q/K/V exchange output, which is dead here: a buffer reuse candidate.
            stacked = torch.cat([o, o * 2.0, o * 4.0])
            return stacked, o

        compiled = torch.compile(step)
        torch.manual_seed(1234 + rank)
        # Above the CUDA-graph size limit, so the eager CE path returns workspace views.
        shape = (1, 4096, 8, 128)
        inputs = [tuple(torch.randn(shape, device=device, dtype=torch.float32) for _ in range(3)) for _ in range(3)]
        expected = [tuple(t.clone() for t in step(*x, zero_copy=False)) for x in inputs]

        q, k, v = a2a_permute.ulysses_qkv_fwd_group(*inputs[0], group_name, world_size, True)
        assert q.untyped_storage().data_ptr() == k.untyped_storage().data_ptr(), "Q/K/V are not workspace views"
        workspace = q.untyped_storage()
        lo, hi = workspace.data_ptr(), workspace.data_ptr() + workspace.nbytes()

        results = [compiled(*x, zero_copy=True) for x in inputs]
        torch.accelerator.synchronize(device)
        checks = 0
        # Every result must still hold its own values after the later calls' exchanges.
        for result, reference in zip(results, expected):
            for got, want in zip(result, reference):
                assert not lo <= got.data_ptr() < hi, "a compiled result lives in the exchange workspace"
                torch.testing.assert_close(got, want, rtol=0, atol=0)
                checks += 1
        print(f"zero-copy compile rank={rank}: {checks} exact checks passed", flush=True)
    finally:
        a2a_permute.clear_a2a_permute_workspaces()
        destroy_distributed_env()


def test_zero_copy_views_under_torch_compile(tmp_path):
    if torch.accelerator.device_count() < 2:
        pytest.skip("Requires two scheduler-allocated NVLink GPUs")
    torch.multiprocessing.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2)
