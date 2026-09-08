# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for SeqAllToAll4D, SeqAllToAll5D, and RingComm communication primitives.

CPU tests use gloo on CPU tensors (no GPU). Nightly parity tests run the same
checks on real multi-GPU NCCL collectives.
"""

from __future__ import annotations

import os
from typing import Literal

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.distributed.comm import RingComm, SeqAllToAll4D, SeqAllToAll5D
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_sp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.platforms import current_omni_platform

_L4_TWO_GPU = hardware_marks(res={"cuda": "L4"}, num_cards=2)
_L4_FOUR_GPU = hardware_marks(res={"cuda": "L4"}, num_cards=4)

DeviceKind = Literal["cpu", "cuda"]


def _update_environment_variables(envs_dict: dict[str, str]) -> None:
    for key, value in envs_dict.items():
        os.environ[key] = value


def _worker_device(local_rank: int, device_kind: DeviceKind) -> torch.device:
    if device_kind == "cpu":
        return torch.device("cpu")
    return torch.device(f"{current_omni_platform.device_type}:{local_rank}")


def _init_worker(
    local_rank: int,
    world_size: int,
    master_port: int,
    device_kind: DeviceKind,
) -> None:
    _update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(master_port),
        }
    )
    backend = "gloo" if device_kind == "cpu" else None
    init_distributed_environment(backend=backend)


def _close_tolerance(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.bfloat16:
        return 1e-3, 1e-3
    if dtype == torch.float16:
        return 1e-3, 1e-3
    return 1e-5, 1e-5


def _run_4d_identity(
    local_rank: int,
    world_size: int,
    dtype: torch.dtype,
    batch_size: int,
    seq_len_per_rank: int,
    num_heads: int,
    head_size: int,
    use_sync: bool,
    device_kind: DeviceKind,
    master_port: int,
) -> None:
    device = _worker_device(local_rank, device_kind)
    if device_kind == "cuda":
        current_omni_platform.set_device(device)

    _init_worker(local_rank, world_size, master_port, device_kind)
    initialize_model_parallel(ulysses_degree=world_size)
    sp_group = get_sp_group().ulysses_group

    torch.manual_seed(42 + local_rank)
    input_tensor = torch.randn(
        batch_size,
        seq_len_per_rank,
        num_heads,
        head_size,
        dtype=dtype,
        device=device,
    )
    original_input = input_tensor.clone()

    intermediate = SeqAllToAll4D.apply(sp_group, input_tensor, 2, 1, use_sync)
    expected_intermediate_shape = (
        batch_size,
        seq_len_per_rank * world_size,
        num_heads // world_size,
        head_size,
    )
    assert intermediate.shape == expected_intermediate_shape, (
        f"Intermediate shape mismatch: expected {expected_intermediate_shape}, got {intermediate.shape}"
    )

    output = SeqAllToAll4D.apply(sp_group, intermediate, 1, 2, use_sync)
    assert output.shape == original_input.shape, (
        f"Output shape mismatch: expected {original_input.shape}, got {output.shape}"
    )

    rtol, atol = _close_tolerance(dtype)
    torch.testing.assert_close(
        output,
        original_input,
        rtol=rtol,
        atol=atol,
        msg="Output does not match original input after two all-to-all operations",
    )
    destroy_distributed_env()


def _run_5d_identity(
    local_rank: int,
    world_size: int,
    dtype: torch.dtype,
    batch_size: int,
    seq_len_per_rank: int,
    num_heads: int,
    head_size: int,
    use_sync: bool,
    device_kind: DeviceKind,
    master_port: int,
) -> None:
    device = _worker_device(local_rank, device_kind)
    if device_kind == "cuda":
        current_omni_platform.set_device(device)

    _init_worker(local_rank, world_size, master_port, device_kind)
    initialize_model_parallel(ulysses_degree=world_size)
    sp_group = get_sp_group().ulysses_group

    torch.manual_seed(42 + local_rank)
    input_tensor = torch.randn(
        batch_size,
        seq_len_per_rank,
        3,
        num_heads,
        head_size,
        dtype=dtype,
        device=device,
    )
    original_input = input_tensor.clone()

    intermediate = SeqAllToAll5D.apply(sp_group, input_tensor, 3, 1, use_sync)
    expected_intermediate_shape = (
        batch_size,
        seq_len_per_rank * world_size,
        3,
        num_heads // world_size,
        head_size,
    )
    assert intermediate.shape == expected_intermediate_shape, (
        f"Intermediate shape mismatch: expected {expected_intermediate_shape}, got {intermediate.shape}"
    )

    output = SeqAllToAll5D.apply(sp_group, intermediate, 1, 3, use_sync)
    assert output.shape == original_input.shape, (
        f"Output shape mismatch: expected {original_input.shape}, got {output.shape}"
    )

    rtol, atol = _close_tolerance(dtype)
    torch.testing.assert_close(
        output,
        original_input,
        rtol=rtol,
        atol=atol,
        msg="Output does not match original input after two all-to-all operations",
    )
    destroy_distributed_env()


def _run_ring_p2p(
    local_rank: int,
    world_size: int,
    dtype: torch.dtype,
    batch_size: int,
    num_heads: int,
    head_size: int,
    device_kind: DeviceKind,
    master_port: int,
) -> None:
    device = _worker_device(local_rank, device_kind)
    if device_kind == "cuda":
        current_omni_platform.set_device(device)

    try:
        _init_worker(local_rank, world_size, master_port, device_kind)
        initialize_model_parallel(ring_degree=world_size)
        sp_group = get_sp_group()
        comm = RingComm(sp_group.ring_group)

        input_tensor = torch.full(
            (batch_size, num_heads, head_size),
            fill_value=float(local_rank + 1),
            dtype=dtype,
            device=device,
        )
        recv_tensor = comm.send_recv(input_tensor)
        comm.commit()
        comm.wait()

        prev_rank = (local_rank - 1 + world_size) % world_size
        expected_value = float(prev_rank + 1)
        expected_tensor = torch.full_like(recv_tensor, fill_value=expected_value)
        rtol, atol = _close_tolerance(dtype)
        torch.testing.assert_close(
            recv_tensor,
            expected_tensor,
            rtol=rtol,
            atol=atol,
            msg=f"[Rank {local_rank}] Ring P2P data mismatch",
        )
    finally:
        destroy_distributed_env()


def _spawn_4d_identity(
    *,
    world_size: int,
    dtype: torch.dtype,
    batch_size: int,
    seq_len_per_rank: int,
    num_heads: int,
    head_size: int,
    use_sync: bool,
    device_kind: DeviceKind,
    master_port: int,
) -> None:
    torch.multiprocessing.spawn(
        _run_4d_identity,
        args=(
            world_size,
            dtype,
            batch_size,
            seq_len_per_rank,
            num_heads,
            head_size,
            use_sync,
            device_kind,
            master_port,
        ),
        nprocs=world_size,
    )


def _spawn_5d_identity(
    *,
    world_size: int,
    dtype: torch.dtype,
    batch_size: int,
    seq_len_per_rank: int,
    num_heads: int,
    head_size: int,
    use_sync: bool,
    device_kind: DeviceKind,
    master_port: int,
) -> None:
    torch.multiprocessing.spawn(
        _run_5d_identity,
        args=(
            world_size,
            dtype,
            batch_size,
            seq_len_per_rank,
            num_heads,
            head_size,
            use_sync,
            device_kind,
            master_port,
        ),
        nprocs=world_size,
    )


def _spawn_ring_p2p(
    *,
    world_size: int,
    dtype: torch.dtype,
    batch_size: int,
    num_heads: int,
    head_size: int,
    device_kind: DeviceKind,
    master_port: int,
) -> None:
    torch.multiprocessing.spawn(
        _run_ring_p2p,
        args=(world_size, dtype, batch_size, num_heads, head_size, device_kind, master_port),
        nprocs=world_size,
    )


def _require_heads_divisible(num_heads: int, world_size: int) -> None:
    if num_heads % world_size != 0:
        pytest.skip(f"num_heads ({num_heads}) not divisible by world_size ({world_size})")


def _require_gpus(world_size: int) -> None:
    available_gpus = current_omni_platform.get_device_count()
    if available_gpus < world_size:
        pytest.skip(f"Test requires {world_size} GPUs but only {available_gpus} available")


# ---------------------------------------------------------------------------
# CPU: gloo collectives on CPU tensors (no GPU)
# ---------------------------------------------------------------------------


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("use_sync", [False, True])
def test_4d_identity(world_size: int, use_sync: bool):
    _require_heads_divisible(8, world_size)
    _spawn_4d_identity(
        world_size=world_size,
        dtype=torch.float32,
        batch_size=2,
        seq_len_per_rank=8,
        num_heads=8,
        head_size=32,
        use_sync=use_sync,
        device_kind="cpu",
        master_port=29610,
    )


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("use_sync", [False, True])
def test_5d_identity(world_size: int, use_sync: bool):
    _require_heads_divisible(8, world_size)
    _spawn_5d_identity(
        world_size=world_size,
        dtype=torch.float32,
        batch_size=2,
        seq_len_per_rank=8,
        num_heads=8,
        head_size=32,
        use_sync=use_sync,
        device_kind="cpu",
        master_port=29611,
    )


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
@pytest.mark.parametrize("world_size", [2, 4])
def test_ring_p2p(world_size: int):
    _spawn_ring_p2p(
        world_size=world_size,
        dtype=torch.float32,
        batch_size=2,
        num_heads=8,
        head_size=128,
        device_kind="cpu",
        master_port=29612,
    )


# ---------------------------------------------------------------------------
# Nightly: same checks on real multi-GPU NCCL collectives
# ---------------------------------------------------------------------------


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parallel
@pytest.mark.parametrize(
    "world_size",
    [
        pytest.param(2, marks=_L4_TWO_GPU),
        pytest.param(4, marks=_L4_FOUR_GPU),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_sync", [False, True])
def test_4d_identity_parity(world_size: int, dtype: torch.dtype, use_sync: bool):
    _require_gpus(world_size)
    _require_heads_divisible(8, world_size)
    _spawn_4d_identity(
        world_size=world_size,
        dtype=dtype,
        batch_size=2,
        seq_len_per_rank=8,
        num_heads=8,
        head_size=32,
        use_sync=use_sync,
        device_kind="cuda",
        master_port=29500,
    )


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parallel
@pytest.mark.parametrize(
    "world_size",
    [
        pytest.param(2, marks=_L4_TWO_GPU),
        pytest.param(4, marks=_L4_FOUR_GPU),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_sync", [False, True])
def test_5d_identity_parity(world_size: int, dtype: torch.dtype, use_sync: bool):
    _require_gpus(world_size)
    _require_heads_divisible(8, world_size)
    _spawn_5d_identity(
        world_size=world_size,
        dtype=dtype,
        batch_size=2,
        seq_len_per_rank=8,
        num_heads=8,
        head_size=32,
        use_sync=use_sync,
        device_kind="cuda",
        master_port=29502,
    )


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parallel
@pytest.mark.parametrize(
    "world_size",
    [
        pytest.param(2, marks=_L4_TWO_GPU),
        pytest.param(4, marks=_L4_FOUR_GPU),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ring_p2p_parity(world_size: int, dtype: torch.dtype):
    _require_gpus(world_size)
    _spawn_ring_p2p(
        world_size=world_size,
        dtype=dtype,
        batch_size=2,
        num_heads=8,
        head_size=128,
        device_kind="cuda",
        master_port=29501,
    )
