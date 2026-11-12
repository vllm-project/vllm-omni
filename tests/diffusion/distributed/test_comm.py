# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for SeqAllToAll4D, SeqAllToAll5D, and RingComm communication primitives.

CPU tests use gloo on CPU tensors (no GPU). Nightly parity tests run the same
checks on real multi-GPU NCCL/RCCL collectives.
"""

from __future__ import annotations

import os
from typing import Literal

import pytest
import torch

from tests.helpers.distributed import configure_cpu_collective_worker, start_collective_workers
from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import get_open_port
from vllm_omni.diffusion.distributed.comm import RingComm, SeqAllToAll4D, SeqAllToAll5D
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_sp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.platforms import current_omni_platform

_L4_TWO_GPU = hardware_marks(res={"cuda": ["L4", "B200"], "rocm": "MI325"}, num_cards=2)
_L4_FOUR_GPU = hardware_marks(res={"cuda": ["L4", "B200"], "rocm": "MI325"}, num_cards=4)

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
    if device_kind == "cpu":
        configure_cpu_collective_worker()
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
        if device_kind == "cpu":
            # P2P completion only synchronizes neighbors. Keep fast CPU ranks
            # alive until every rank has finished constructing and using groups.
            torch.distributed.barrier()
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
    start_collective_workers(
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
        device_kind=device_kind,
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
    start_collective_workers(
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
        device_kind=device_kind,
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
    start_collective_workers(
        _run_ring_p2p,
        args=(world_size, dtype, batch_size, num_heads, head_size, device_kind, master_port),
        nprocs=world_size,
        device_kind=device_kind,
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
        master_port=get_open_port(),
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
        master_port=get_open_port(),
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
        master_port=get_open_port(),
    )


# ---------------------------------------------------------------------------
# Nightly: same checks on real multi-GPU NCCL/RCCL collectives
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


def _run_fused_qkv_matches_three_4d(
    local_rank: int,
    world_size: int,
    device_kind: DeviceKind,
    master_port: int,
) -> None:
    """LingBot's fused (B, S/N, 3, H, D) exchange must equal three per-tensor 4D exchanges."""
    from vllm_omni.diffusion.distributed.comm import all_to_all_5D

    device = _worker_device(local_rank, device_kind)
    if device_kind == "cuda":
        current_omni_platform.set_device(device)
    _init_worker(local_rank, world_size, master_port, device_kind)
    initialize_model_parallel(ulysses_degree=world_size)
    sp_group = get_sp_group().ulysses_group
    try:
        batch, seq_per_rank, heads, head_size = 1, 6, 4 * world_size, 8
        # Distinguishable values: rank, q/k/v index, position, head, feature.
        base = torch.arange(seq_per_rank * heads * head_size, dtype=torch.float32, device=device).reshape(
            batch, seq_per_rank, heads, head_size
        )
        tensors = [base + 1000.0 * local_rank + 100000.0 * which for which in range(3)]
        separate = [SeqAllToAll4D.apply(sp_group, t, 2, 1, False) for t in tensors]
        fused = all_to_all_5D(torch.stack(tensors, dim=2), scatter_idx=3, gather_idx=1, group=sp_group)
        assert fused.shape == (batch, seq_per_rank * world_size, 3, heads // world_size, head_size)
        for which, expected in enumerate(separate):
            assert torch.equal(fused.unbind(2)[which], expected), f"q/k/v index {which} differs on rank {local_rank}"
    finally:
        destroy_distributed_env()


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
@pytest.mark.parametrize("world_size", [2, 4])
def test_fused_qkv_all_to_all_matches_three_4d_exchanges_cpu(world_size: int) -> None:
    start_collective_workers(
        _run_fused_qkv_matches_three_4d,
        args=(world_size, "cpu", 29613),
        nprocs=world_size,
        device_kind="cpu",
    )


def _packed_test_inputs(
    shape: tuple[int, int, int, int], rank: int, device: torch.device, dtype: torch.dtype, layout: str
) -> tuple[torch.Tensor, ...]:
    batch, tokens, heads, width = shape
    values = torch.arange(batch * tokens * heads * width * 3, dtype=torch.int32).reshape(*shape, 3)
    if dtype == torch.int32:
        values = values + (rank + 1) * 1000000
    else:
        # Bounded, exactly representable values retain rank/coordinate tags in
        # low precision without rounding every coordinate to the same value.
        values = (values % 47 + rank * 64).to(dtype)
    if layout == "row_strided":
        storage = torch.empty((batch, 2 * tokens, heads, width, 3), dtype=dtype, device=device)
        storage[:, ::2].copy_(values)
        return tuple(storage[:, ::2, :, :, index] for index in range(3))
    if layout == "projection":
        storage = values.permute(0, 1, 2, 4, 3).contiguous().to(device)
        return tuple(storage[..., index, :] for index in range(3))
    storage = values.to(device)
    if layout == "interleaved":
        return tuple(storage[..., index] for index in range(3))
    return tuple(storage[..., index].contiguous() for index in range(3))


def _check_packed_strategy(group, device: torch.device) -> None:
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
    from vllm_omni.diffusion.attention.parallel import ulysses
    from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention

    qkv = _packed_test_inputs((2, 11, 8, 4), group.ulysses_rank, device, torch.bfloat16, "projection")
    for joint in (None, "front", "rear"):
        outputs, strategies, metadata = [], [], []
        for enabled in (False, True):
            strategy = UlyssesParallelAttention(group, 2, 1, False)
            # Exercise the portable layout on CPU/Gloo and GPU collectives.
            # Public platform selection is checked separately without a group.
            strategy._rocm_packed_qkv = enabled
            strategies.append(strategy)
            if joint is None:
                mask = torch.ones((2, 11 * group.ulysses_world_size), device=device, dtype=torch.bool)
                mask[:, -3:] = False
                md = AttentionMetadata(attn_mask=mask, extra={"sentinel": "unchanged"})
            else:
                joint_q = torch.arange(2 * 3 * 8 * 4, device=device, dtype=torch.float32).reshape(2, 3, 8, 4)
                joint_q = joint_q.to(torch.bfloat16)
                md = AttentionMetadata(
                    joint_query=joint_q,
                    joint_key=joint_q + 1,
                    joint_value=joint_q + 2,
                    joint_strategy=joint,
                    extra={"sentinel": "unchanged"},
                )
            metadata.append(md)
            outputs.append(strategy.pre_attention(*qkv, md))
        assert outputs[0][4].use_packed_qkv is False
        assert outputs[1][4].use_packed_qkv is True
        for slot in range(3):
            assert torch.equal(outputs[0][slot], outputs[1][slot])
        assert metadata[0].extra == metadata[1].extra
        if joint is None:
            assert torch.equal(metadata[0].attn_mask, metadata[1].attn_mask)
        restored = [strategy.post_attention(out[0], out[4]) for strategy, out in zip(strategies, outputs)]
        assert torch.equal(restored[0], restored[1])

    # Opting in must preserve the existing gate, GQA, and variable-size UAA
    # routes, including their actual process-group exchanges.
    for case in ("gate", "gqa", "uaa"):
        inputs = (qkv[0], qkv[1][:, :, :4], qkv[2][:, :, :4]) if case == "gqa" else qkv
        outputs, gates, restored = [], [], []
        with pytest.MonkeyPatch.context() as patch:
            mode = "advanced_uaa" if case == "uaa" else "strict"
            patch.setattr(ulysses, "get_ulysses_mode", lambda **kwargs: mode)
            for enabled in (False, True):
                strategy = UlyssesParallelAttention(group, 2, 1, False)
                strategy._rocm_packed_qkv = enabled
                md = AttentionMetadata(extra={"gate_compress": inputs[0].clone()}) if case == "gate" else None
                out = strategy.pre_attention(*inputs, md)
                assert out[4].use_packed_qkv is False
                assert out[4].use_uaa is (case == "uaa")
                outputs.append(out)
                restored.append(strategy.post_attention(out[0], out[4]))
                if md is not None:
                    gates.append(md.extra["gate_compress"])
        for slot in range(3):
            assert torch.equal(outputs[0][slot], outputs[1][slot])
        assert torch.equal(restored[0], restored[1])
        if gates:
            assert torch.equal(gates[0], gates[1])


def _run_packed_qkv(local_rank: int, world_size: int, device_kind: DeviceKind, master_port: int) -> None:
    from vllm_omni.diffusion.distributed.comm import all_to_all_4D
    from vllm_omni.diffusion.distributed.packed_qkv import packed_output_all_to_all, packed_qkv_all_to_all

    device = _worker_device(local_rank, device_kind)
    if device_kind == "cuda":
        current_omni_platform.set_device(device)
    _init_worker(local_rank, world_size, master_port, device_kind)
    initialize_model_parallel(ulysses_degree=world_size)
    group = get_sp_group().ulysses_group
    cases = [
        ((1, 7, 8, 5), torch.int32, "contiguous"),
        ((2, 11, 8, 7), torch.int32, "row_strided"),
        ((1, 13, 56, 16), torch.bfloat16, "projection"),
        ((2, 17, 56, 16), torch.bfloat16, "interleaved"),
        ((2, 19, 56, 16), torch.float16, "row_strided"),
    ]
    if device_kind == "cuda":
        cases.append(((1, 1032, 56, 128), torch.bfloat16, "projection"))
    try:
        with torch.inference_mode():
            for shape, dtype, layout in cases:
                qkv = _packed_test_inputs(shape, local_rank, device, dtype, layout)
                expected = tuple(all_to_all_4D(t, group=group) for t in qkv)
                original_collective = torch.distributed.all_to_all_single
                calls = []

                def counted(*args, **kwargs):
                    calls.append(1)
                    return original_collective(*args, **kwargs)

                with pytest.MonkeyPatch.context() as patch:
                    patch.setattr(torch.distributed, "all_to_all_single", counted)
                    actual = packed_qkv_all_to_all(group, *qkv)
                    assert len(calls) == 1
                    reverse = packed_output_all_to_all(group, actual[0])
                    assert len(calls) == 2
                for tensor, reference, producer in zip(actual, expected, qkv):
                    assert tensor.is_contiguous()
                    assert tensor.dtype == producer.dtype
                    assert torch.equal(tensor, reference)
                    # Independent oracle: rank-ordered sequence concatenation
                    # of each producer's destination head slice.
                    gathered = [torch.empty_like(producer) for _ in range(world_size)]
                    torch.distributed.all_gather(gathered, producer.contiguous(), group=group)
                    heads = shape[2] // world_size
                    oracle = torch.cat(
                        [t[:, :, local_rank * heads : (local_rank + 1) * heads] for t in gathered], dim=1
                    )
                    assert torch.equal(tensor, oracle)
                assert reverse.is_contiguous()
                assert torch.equal(reverse, qkv[0])
                assert torch.equal(reverse, all_to_all_4D(expected[0], 1, 2, group=group))
                padded = torch.empty(
                    (shape[0], actual[0].shape[1] + 3, actual[0].shape[2], shape[3]), dtype=dtype, device=device
                )
                sliced = padded[:, 1:-2]
                sliced.copy_(actual[0])
                assert torch.equal(packed_output_all_to_all(group, sliced), reverse)
                saved = tuple(t.clone() for t in actual)
                packed_qkv_all_to_all(group, *(t + 1 for t in qkv))
                for tensor, reference in zip(actual, saved):
                    assert torch.equal(tensor, reference)
            _check_packed_strategy(get_sp_group(), device)
    finally:
        destroy_distributed_env()


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parallel
@pytest.mark.cpu
@pytest.mark.parametrize("world_size", [2, 4])
def test_packed_qkv_layout_and_strategy_cpu(world_size: int) -> None:
    start_collective_workers(
        _run_packed_qkv,
        args=(world_size, "cpu", get_open_port()),
        nprocs=world_size,
        device_kind="cpu",
    )


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parallel
@pytest.mark.parametrize("world_size", [pytest.param(2, marks=_L4_TWO_GPU), pytest.param(4, marks=_L4_FOUR_GPU)])
def test_packed_qkv_layout_and_strategy_gpu(world_size: int) -> None:
    _require_gpus(world_size)
    start_collective_workers(
        _run_packed_qkv,
        args=(world_size, "cuda", get_open_port()),
        nprocs=world_size,
        device_kind="cuda",
    )
