# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for SeqAllToAll4D, SeqAllToAll5D, and RingComm communication primitives.

CPU tests use gloo on CPU tensors (no GPU). Nightly parity tests run the same
checks on real multi-GPU NCCL collectives.
"""

from __future__ import annotations

from typing import Literal

import pytest
import torch
import torch.distributed as dist

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import get_distributed_init_method
from vllm_omni.diffusion.distributed.comm import RingComm, SeqAllToAll4D, SeqAllToAll5D
from vllm_omni.platforms import current_omni_platform

_L4_TWO_GPU = hardware_marks(res={"cuda": "L4"}, num_cards=2)
_L4_FOUR_GPU = hardware_marks(res={"cuda": "L4"}, num_cards=4)

DeviceKind = Literal["cpu", "cuda"]


def _worker_device(local_rank: int, device_kind: DeviceKind) -> torch.device:
    if device_kind == "cpu":
        return torch.device("cpu")
    return torch.device(f"{current_omni_platform.device_type}:{local_rank}")


def _close_tolerance(dtype: torch.dtype) -> tuple[float, float]:
    if dtype in (torch.bfloat16, torch.float16):
        return 1e-3, 1e-3
    return 1e-5, 1e-5


def _assert_4d_identity(
    group: dist.ProcessGroup,
    *,
    dtype: torch.dtype,
    device: torch.device,
    world_size: int,
    use_sync: bool,
) -> None:
    batch_size = 2
    seq_len_per_rank = 8
    num_heads = 8
    head_size = 32

    torch.manual_seed(42 + dist.get_rank(group))
    input_tensor = torch.randn(
        batch_size,
        seq_len_per_rank,
        num_heads,
        head_size,
        dtype=dtype,
        device=device,
    )
    original_input = input_tensor.clone()

    intermediate = SeqAllToAll4D.apply(group, input_tensor, 2, 1, use_sync)
    expected_intermediate_shape = (
        batch_size,
        seq_len_per_rank * world_size,
        num_heads // world_size,
        head_size,
    )
    assert intermediate.shape == expected_intermediate_shape, (
        f"Intermediate shape mismatch: expected {expected_intermediate_shape}, got {intermediate.shape}"
    )

    output = SeqAllToAll4D.apply(group, intermediate, 1, 2, use_sync)
    assert output.shape == original_input.shape, (
        f"Output shape mismatch: expected {original_input.shape}, got {output.shape}"
    )

    rtol, atol = _close_tolerance(dtype)
    torch.testing.assert_close(
        output,
        original_input,
        rtol=rtol,
        atol=atol,
        msg="Output does not match original input after two 4D all-to-all operations",
    )


def _assert_5d_identity(
    group: dist.ProcessGroup,
    *,
    dtype: torch.dtype,
    device: torch.device,
    world_size: int,
    use_sync: bool,
) -> None:
    batch_size = 2
    seq_len_per_rank = 8
    num_heads = 8
    head_size = 32

    torch.manual_seed(42 + dist.get_rank(group))
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

    intermediate = SeqAllToAll5D.apply(group, input_tensor, 3, 1, use_sync)
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

    output = SeqAllToAll5D.apply(group, intermediate, 1, 3, use_sync)
    assert output.shape == original_input.shape, (
        f"Output shape mismatch: expected {original_input.shape}, got {output.shape}"
    )

    rtol, atol = _close_tolerance(dtype)
    torch.testing.assert_close(
        output,
        original_input,
        rtol=rtol,
        atol=atol,
        msg="Output does not match original input after two 5D all-to-all operations",
    )


def _assert_ring_p2p(
    group: dist.ProcessGroup,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    local_rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    comm = RingComm(group)
    input_tensor = torch.full(
        (2, 8, 128),
        fill_value=float(local_rank + 1),
        dtype=dtype,
        device=device,
    )
    recv_tensor = comm.send_recv(input_tensor)
    comm.commit()
    comm.wait()

    prev_rank = (local_rank - 1 + world_size) % world_size
    expected_tensor = torch.full_like(recv_tensor, fill_value=float(prev_rank + 1))
    rtol, atol = _close_tolerance(dtype)
    torch.testing.assert_close(
        recv_tensor,
        expected_tensor,
        rtol=rtol,
        atol=atol,
        msg=f"[Rank {local_rank}] Ring P2P data mismatch",
    )


def _run_comm_checks(
    local_rank: int,
    world_size: int,
    dtype: torch.dtype,
    checks: tuple[str, ...],
    use_sync_values: tuple[bool, ...],
    device_kind: DeviceKind,
    init_method: str,
    run_ring: bool,
) -> None:
    device = _worker_device(local_rank, device_kind)
    if device_kind == "cuda":
        current_omni_platform.set_device(device)

    backend = "gloo" if device_kind == "cpu" else "nccl"
    try:
        dist.init_process_group(
            backend,
            init_method=init_method,
            rank=local_rank,
            world_size=world_size,
        )
        group = dist.group.WORLD
        for use_sync in use_sync_values:
            if "4d" in checks:
                _assert_4d_identity(
                    group,
                    dtype=dtype,
                    device=device,
                    world_size=world_size,
                    use_sync=use_sync,
                )
            if "5d" in checks:
                _assert_5d_identity(
                    group,
                    dtype=dtype,
                    device=device,
                    world_size=world_size,
                    use_sync=use_sync,
                )
        if run_ring:
            _assert_ring_p2p(group, dtype=dtype, device=device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _spawn_comm_checks(
    *,
    world_size: int,
    dtype: torch.dtype,
    checks: tuple[str, ...],
    use_sync_values: tuple[bool, ...],
    device_kind: DeviceKind,
    run_ring: bool = False,
) -> None:
    torch.multiprocessing.spawn(
        _run_comm_checks,
        args=(
            world_size,
            dtype,
            checks,
            use_sync_values,
            device_kind,
            get_distributed_init_method(),
            run_ring,
        ),
        nprocs=world_size,
    )


def _require_gpus(world_size: int) -> None:
    available_gpus = current_omni_platform.get_device_count()
    if available_gpus < world_size:
        pytest.skip(f"Test requires {world_size} GPUs but only {available_gpus} available")


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
def test_cpu_comm_primitives():
    _spawn_comm_checks(
        world_size=2,
        dtype=torch.float32,
        checks=("4d", "5d"),
        use_sync_values=(False,),
        device_kind="cpu",
        run_ring=True,
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
def test_4d_identity_parity(world_size: int, dtype: torch.dtype):
    _require_gpus(world_size)
    _spawn_comm_checks(
        world_size=world_size,
        dtype=dtype,
        checks=("4d",),
        use_sync_values=(False, True),
        device_kind="cuda",
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
def test_5d_identity_parity(world_size: int, dtype: torch.dtype):
    _require_gpus(world_size)
    _spawn_comm_checks(
        world_size=world_size,
        dtype=dtype,
        checks=("5d",),
        use_sync_values=(False, True),
        device_kind="cuda",
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
    _spawn_comm_checks(
        world_size=world_size,
        dtype=dtype,
        checks=(),
        use_sync_values=(),
        device_kind="cuda",
        run_ring=True,
    )
