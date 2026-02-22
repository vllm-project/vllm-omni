# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import socket
import tempfile

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import get_forward_context, set_forward_context
from vllm_omni.platforms import current_omni_platform


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _set_dist_env(*, rank: int, world_size: int, master_port: int) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)


def _run_attention_case(
    local_rank: int,
    world_size: int,
    master_port: int,
    input_file: str,
    output_file: str,
    num_heads: int,
    head_size: int,
    ulysses_degree: int,
    ulysses_mode: str,
    ring_degree: int = 1,
    split_sizes: list[int] | None = None,
    sdp_kernel_mode: str = "math",
) -> None:
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)

    _set_dist_env(rank=local_rank, world_size=world_size, master_port=master_port)
    os.environ["DIFFUSION_ATTENTION_BACKEND"] = "TORCH_SDPA"

    init_distributed_environment(world_size=world_size, rank=local_rank)
    initialize_model_parallel(
        data_parallel_size=1,
        cfg_parallel_size=1,
        sequence_parallel_size=world_size,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )

    parallel_config = DiffusionParallelConfig(
        pipeline_parallel_size=1,
        data_parallel_size=1,
        tensor_parallel_size=1,
        sequence_parallel_size=world_size,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        cfg_parallel_size=1,
        ulysses_mode=ulysses_mode,
    )
    od_config = OmniDiffusionConfig(model="test", dtype=torch.float32, parallel_config=parallel_config)

    with set_forward_context(omni_diffusion_config=od_config):
        attn = Attention(
            num_heads=num_heads,
            head_size=head_size,
            causal=False,
            softmax_scale=1.0 / (head_size**0.5),
        ).to(device=device, dtype=torch.float32)

        with np.load(input_file, allow_pickle=False) as payload:
            q_full = torch.from_numpy(payload["q"]).to(device=device)
            k_full = torch.from_numpy(payload["k"]).to(device=device)
            v_full = torch.from_numpy(payload["v"]).to(device=device)

        if world_size == 1:
            q, k, v = q_full, k_full, v_full
        else:
            if split_sizes is None:
                # NOTE: torch.chunk may return fewer than `world_size` chunks for some
                # uneven lengths (e.g. seq_len=9, world_size=4 -> 3 chunks of len 3).
                # We need an exact `world_size`-way split to simulate uneven SP shards.
                q = torch.tensor_split(q_full, world_size, dim=1)[local_rank].contiguous()
                k = torch.tensor_split(k_full, world_size, dim=1)[local_rank].contiguous()
                v = torch.tensor_split(v_full, world_size, dim=1)[local_rank].contiguous()
            else:
                if len(split_sizes) != world_size:
                    raise ValueError(f"split_sizes length ({len(split_sizes)}) must equal world_size ({world_size}).")
                if sum(int(x) for x in split_sizes) != q_full.shape[1]:
                    raise ValueError(
                        "split_sizes must sum to full seq_len "
                        f"(got sum={sum(int(x) for x in split_sizes)}, seq_len={q_full.shape[1]})."
                    )
                q = torch.split(q_full, split_sizes, dim=1)[local_rank].contiguous()
                k = torch.split(k_full, split_sizes, dim=1)[local_rank].contiguous()
                v = torch.split(v_full, split_sizes, dim=1)[local_rank].contiguous()
            # The Attention layer only enables SP communication when ForwardContext.sp_active is True.
            # In production this is managed by SequenceParallelSplitHook/GatherHook, but here we
            # shard manually for testing.
            get_forward_context()._sp_shard_depth = 1

        if sdp_kernel_mode == "math":
            sdp_ctx = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        elif sdp_kernel_mode == "mem_efficient":
            sdp_ctx = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=False)
        else:
            raise ValueError(f"Invalid sdp_kernel_mode: {sdp_kernel_mode!r}")

        with sdp_ctx:
            out_local = attn(q, k, v).contiguous()

        if world_size == 1:
            out_full = out_local
        else:
            local_len = torch.tensor([out_local.shape[1]], device=device, dtype=torch.int64)
            gathered_lens = [torch.empty_like(local_len) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_lens, local_len)
            lens = [int(t.item()) for t in gathered_lens]
            max_len = max(lens)

            if out_local.shape[1] < max_len:
                pad = max_len - out_local.shape[1]
                out_local = torch.nn.functional.pad(out_local, (0, 0, 0, 0, 0, pad)).contiguous()

            gathered = [torch.empty_like(out_local) for _ in range(world_size)]
            torch.distributed.all_gather(gathered, out_local)
            if local_rank == 0:
                out_full = torch.cat([t[:, : lens[i]].contiguous() for i, t in enumerate(gathered)], dim=1)
            else:
                out_full = None

        if local_rank == 0:
            np.save(output_file, out_full.detach().cpu().numpy())

    destroy_distributed_env()


@pytest.mark.parametrize(
    "sp_world_size,seq_len,num_heads",
    [
        (2, 6, 3),  # head_cnt not divisible by P=2
        (2, 5, 4),  # seq_len not divisible by P=2
        (4, 9, 30),  # Z-Image-like: head_cnt not divisible by P=4
        (4, 10, 8),  # seq_len not divisible by P=4
    ],
)
def test_ulysses_uaa_matches_baseline(sp_world_size: int, seq_len: int, num_heads: int) -> None:
    if current_omni_platform.get_device_count() < sp_world_size:
        pytest.skip(f"Test requires {sp_world_size} GPUs")

    batch_size = 2
    head_size = 8

    base_port = _find_free_port()
    sp_port = _find_free_port()
    while sp_port == base_port:
        sp_port = _find_free_port()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as f_in:
        input_file = f_in.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as f_base:
        baseline_file = f_base.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as f_sp:
        sp_file = f_sp.name

    try:
        torch.manual_seed(0)
        q = torch.randn(batch_size, seq_len, num_heads, head_size, dtype=torch.float32)
        k = torch.randn(batch_size, seq_len, num_heads, head_size, dtype=torch.float32)
        v = torch.randn(batch_size, seq_len, num_heads, head_size, dtype=torch.float32)
        np.savez(input_file, q=q.numpy(), k=k.numpy(), v=v.numpy())

        # Baseline (no SP)
        torch.multiprocessing.spawn(
            _run_attention_case,
            args=(1, base_port, input_file, baseline_file, num_heads, head_size, 1, "strict"),
            nprocs=1,
        )

        # SP (Ulysses-P with UAA)
        torch.multiprocessing.spawn(
            _run_attention_case,
            args=(
                sp_world_size,
                sp_port,
                input_file,
                sp_file,
                num_heads,
                head_size,
                sp_world_size,
                "advanced_uaa",
            ),
            nprocs=sp_world_size,
        )

        baseline = np.load(baseline_file, allow_pickle=False)
        sp = np.load(sp_file, allow_pickle=False)

        baseline_t = torch.from_numpy(baseline)
        sp_t = torch.from_numpy(sp)
        assert baseline_t.shape == sp_t.shape
        torch.testing.assert_close(sp_t, baseline_t, atol=1e-5, rtol=1e-5)
    finally:
        for path in (input_file, baseline_file, sp_file):
            try:
                os.remove(path)
            except OSError:
                pass


def test_ulysses_uaa_hybrid_ring_matches_baseline() -> None:
    sp_world_size = 4
    ulysses_degree = 2
    ring_degree = 2

    if current_omni_platform.get_device_count() < sp_world_size:
        pytest.skip(f"Test requires {sp_world_size} GPUs")

    batch_size = 2
    head_size = 8
    seq_len = 10
    num_heads = 3  # head_cnt not divisible by ulysses_degree=2 -> triggers head padding

    # Ensure ring ranks see equal post-Ulysses seq_len:
    # rank0/1 -> 3+2=5, rank2/3 -> 3+2=5
    split_sizes = [3, 2, 3, 2]

    base_port = _find_free_port()
    sp_port = _find_free_port()
    while sp_port == base_port:
        sp_port = _find_free_port()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as f_in:
        input_file = f_in.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as f_base:
        baseline_file = f_base.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as f_sp:
        sp_file = f_sp.name

    try:
        torch.manual_seed(0)
        q = torch.randn(batch_size, seq_len, num_heads, head_size, dtype=torch.float32)
        k = torch.randn(batch_size, seq_len, num_heads, head_size, dtype=torch.float32)
        v = torch.randn(batch_size, seq_len, num_heads, head_size, dtype=torch.float32)
        np.savez(input_file, q=q.numpy(), k=k.numpy(), v=v.numpy())

        # Baseline (no SP)
        torch.multiprocessing.spawn(
            _run_attention_case,
            args=(1, base_port, input_file, baseline_file, num_heads, head_size, 1, "strict", 1, None, "mem_efficient"),
            nprocs=1,
        )

        # Hybrid SP: Ulysses (P=2) + Ring (P=2) with advanced_uaa
        torch.multiprocessing.spawn(
            _run_attention_case,
            args=(
                sp_world_size,
                sp_port,
                input_file,
                sp_file,
                num_heads,
                head_size,
                ulysses_degree,
                "advanced_uaa",
                ring_degree,
                split_sizes,
                "mem_efficient",
            ),
            nprocs=sp_world_size,
        )

        baseline = np.load(baseline_file, allow_pickle=False)
        sp = np.load(sp_file, allow_pickle=False)

        baseline_t = torch.from_numpy(baseline)
        sp_t = torch.from_numpy(sp)
        assert baseline_t.shape == sp_t.shape
        # Hybrid (Ulysses+Ring) typically has slightly larger numerical differences
        # than pure Ulysses due to different communication/reduction order and
        # the SDPA kernel path used by Ring attention. Use a looser tolerance to
        # keep the test stable across GPUs/kernels while still catching regressions.
        torch.testing.assert_close(sp_t, baseline_t, atol=5e-4, rtol=5e-4)
    finally:
        for path in (input_file, baseline_file, sp_file):
            try:
                os.remove(path)
            except OSError:
                pass
