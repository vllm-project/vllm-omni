# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SeqAllToAll4D and SeqAllToAll5D communication primitives."""

import os

import pytest
import torch

from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D, SeqAllToAll5D
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_sp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.utils.platform_utils import detect_device_type

device_type = detect_device_type()
if device_type == "cuda":
    torch_device = torch.cuda
elif device_type == "npu":
    torch_device = torch.npu
else:
    raise ValueError(f"Unsupported device type: {device_type} for this test script! Expected GPU or NPU.")


def update_environment_variables(envs_dict: dict[str, str]):
    """Update multiple environment variables with logging."""
    for k, v in envs_dict.items():
        os.environ[k] = v


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seq_len_per_rank", [8])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("head_size", [32])
@pytest.mark.parametrize("use_sync", [False, True])
def test_4d_identity(
    world_size: int,
    dtype: torch.dtype,
    batch_size: int,
    seq_len_per_rank: int,
    num_heads: int,
    head_size: int,
    use_sync: bool,
):
    """Test that two consecutive all-to-all operations return the original input."""
    # Ensure num_heads is divisible by world_size
    if num_heads % world_size != 0:
        pytest.skip(f"num_heads ({num_heads}) not divisible by world_size ({world_size})")

    # Run test with multiprocessing spawn
    torch.multiprocessing.spawn(
        _test_4d_identity_worker,
        args=(
            world_size,
            dtype,
            batch_size,
            seq_len_per_rank,
            num_heads,
            head_size,
            use_sync,
        ),
        nprocs=world_size,
    )


def _test_4d_identity_worker(
    local_rank: int,
    world_size: int,
    dtype: torch.dtype,
    batch_size: int,
    seq_len_per_rank: int,
    num_heads: int,
    head_size: int,
    use_sync: bool,
):
    """Worker function for test_4d_identity."""
    # Set device
    device = torch.device(f"{device_type}:{local_rank}")
    torch_device.set_device(device)

    # Set environment variables for distributed training
    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29500",
        }
    )

    # Initialize distributed environment
    init_distributed_environment()
    initialize_model_parallel(ulysses_degree=world_size)  # test ulysses sp by default
    sp_group = get_sp_group().ulysses_group  # get ulysses sp group not ring sp group

    # Create input tensor: (bs, seqlen/P, hc, hs)
    torch.manual_seed(42 + local_rank)
    input_tensor = torch.randn(
        batch_size,
        seq_len_per_rank,
        num_heads,
        head_size,
        dtype=dtype,
        device=device,
    )

    # Save original input for comparison
    original_input = input_tensor.clone()

    # First all-to-all: (bs, seqlen/P, hc, hs) -> (bs, seqlen, hc/P, hs)
    intermediate = SeqAllToAll4D.apply(
        sp_group,
        input_tensor,
        2,  # scatter head dimension
        1,  # gather sequence dimension
        use_sync,
    )

    # Verify intermediate shape
    expected_shape = (
        batch_size,
        seq_len_per_rank * world_size,
        num_heads // world_size,
        head_size,
    )
    assert intermediate.shape == expected_shape, (
        f"Intermediate shape mismatch: expected {expected_shape}, got {intermediate.shape}"
    )

    # Second all-to-all: (bs, seqlen, hc/P, hs) -> (bs, seqlen/P, hc, hs)
    output = SeqAllToAll4D.apply(
        sp_group,
        intermediate,
        1,  # scatter sequence dimension
        2,  # gather head dimension
        use_sync,
    )

    # Verify output shape matches input
    assert output.shape == original_input.shape, (
        f"Output shape mismatch: expected {original_input.shape}, got {output.shape}"
    )

    # Verify output matches original input
    torch.testing.assert_close(
        output,
        original_input,
        rtol=1e-5,
        atol=1e-5,
        msg="Output does not match original input after two all-to-all operations",
    )

    # Cleanup distributed environment
    destroy_distributed_env()


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seq_len_per_rank", [8])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("head_size", [32])
@pytest.mark.parametrize("use_sync", [False, True])
def test_5d_identity(
    world_size: int,
    dtype: torch.dtype,
    batch_size: int,
    seq_len_per_rank: int,
    num_heads: int,
    head_size: int,
    use_sync: bool,
):
    """Test that two consecutive all-to-all operations return the original input."""
    # Ensure num_heads is divisible by world_size
    if num_heads % world_size != 0:
        pytest.skip(f"num_heads ({num_heads}) not divisible by world_size ({world_size})")

    # Run test with multiprocessing spawn
    torch.multiprocessing.spawn(
        _test_5d_identity_worker,
        args=(
            world_size,
            dtype,
            batch_size,
            seq_len_per_rank,
            num_heads,
            head_size,
            use_sync,
        ),
        nprocs=world_size,
    )


def _test_5d_identity_worker(
    local_rank: int,
    world_size: int,
    dtype: torch.dtype,
    batch_size: int,
    seq_len_per_rank: int,
    num_heads: int,
    head_size: int,
    use_sync: bool,
):
    """Worker function for test_5d_identity."""
    # Set device
    device = torch.device(f"{device_type}:{local_rank}")
    torch_device.set_device(device)

    # Set environment variables for distributed training
    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29500",
        }
    )

    # Initialize distributed environment
    init_distributed_environment()
    initialize_model_parallel(ulysses_degree=world_size)  # test ulysses sp by default
    sp_group = get_sp_group().ulysses_group  # get ulysses sp group not ring sp group

    # Create input tensor: (bs, seqlen/P, 3, hc, hs)
    # The '3' dimension is for Q, K, V
    torch.manual_seed(42 + local_rank)
    input_tensor = torch.randn(
        batch_size,
        seq_len_per_rank,
        3,  # Q, K, V
        num_heads,
        head_size,
        dtype=dtype,
        device=device,
    )

    # Save original input for comparison
    original_input = input_tensor.clone()

    # First all-to-all: (bs, seqlen/P, 3, hc, hs) -> (bs, seqlen, 3, hc/P, hs)
    intermediate = SeqAllToAll5D.apply(
        sp_group,
        input_tensor,
        3,  # scatter head dimension
        1,  # gather sequence dimension
        use_sync,
    )

    # Verify intermediate shape
    expected_shape = (
        batch_size,
        seq_len_per_rank * world_size,
        3,
        num_heads // world_size,
        head_size,
    )
    assert intermediate.shape == expected_shape, (
        f"Intermediate shape mismatch: expected {expected_shape}, got {intermediate.shape}"
    )

    # Second all-to-all: (bs, seqlen, 3, hc/P, hs) -> (bs, seqlen/P, 3, hc, hs)
    output = SeqAllToAll5D.apply(
        sp_group,
        intermediate,
        1,  # scatter sequence dimension
        3,  # gather head dimension
        use_sync,
    )

    # Verify output shape matches input
    assert output.shape == original_input.shape, (
        f"Output shape mismatch: expected {original_input.shape}, got {output.shape}"
    )

    # Verify output matches original input
    torch.testing.assert_close(
        output,
        original_input,
        rtol=1e-5,
        atol=1e-5,
        msg="Output does not match original input after two all-to-all operations",
    )

    # Cleanup distributed environment
    destroy_distributed_env()
