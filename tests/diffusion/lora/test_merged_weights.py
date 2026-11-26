# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import socket

import pytest
import torch

from vllm_omni.lora.merged_weights import MergedLinearWeights, check_merge_error

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _layer(dtype=torch.bfloat16, slices=(3, 2), active=(True, True)):
    generator = torch.Generator().manual_seed(17)
    base = torch.nn.Linear(4, sum(slices), bias=False, dtype=dtype)
    with torch.no_grad():
        base.weight.copy_(torch.randn(base.weight.shape, generator=generator))
    layer = torch.nn.Module()
    layer.base_layer = base
    layer.output_slices = slices
    layer._diffusion_lora_active_slices = active
    layer.lora_a_stacked = tuple(torch.randn(1, 1, 2, 4, generator=generator) for _ in slices)
    layer.lora_b_stacked = tuple(torch.randn(1, 1, rows, 2, generator=generator) for rows in slices)
    return layer


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("active", [(True, True), (True, False), (False, True)])
def test_packed_merge_rounds_once_and_preserves_storage(dtype, active):
    layer = _layer(dtype, active=active)
    weight = layer.base_layer.weight
    base = weight.detach().clone()
    pointer = weight.data_ptr()
    bank = MergedLinearWeights()
    prepared = bank.prepare({"qkv": layer})
    torch.testing.assert_close(weight, base, rtol=0, atol=0)
    bank.apply(prepared)
    expected = base.float().clone()
    offset = 0
    for enabled, rows, a, b in zip(active, layer.output_slices, layer.lora_a_stacked, layer.lora_b_stacked):
        if enabled:
            expected[offset : offset + rows] += b[0, 0] @ a[0, 0]
        offset += rows
    torch.testing.assert_close(weight, expected.to(dtype), rtol=0, atol=0)
    assert weight.data_ptr() == pointer
    bank.close()
    torch.testing.assert_close(weight, base, rtol=0, atol=0)
    assert weight.data_ptr() == pointer


def test_repeated_updates_do_not_accumulate_bfloat16_drift():
    layer = _layer()
    base = layer.base_layer.weight.detach().clone()
    bank = MergedLinearWeights()
    for step in range(50):
        layer.lora_b_stacked[0].fill_(step / 127)
        bank.apply(bank.prepare({"proj": layer}))
        expected = base.float()
        expected[:3] += layer.lora_b_stacked[0][0, 0] @ layer.lora_a_stacked[0][0, 0]
        expected[3:] += layer.lora_b_stacked[1][0, 0] @ layer.lora_a_stacked[1][0, 0]
        torch.testing.assert_close(layer.base_layer.weight, expected.bfloat16(), rtol=0, atol=0)


def test_omitted_targets_restore_pristine_base():
    first, second = _layer(), _layer()
    base = first.base_layer.weight.detach().clone()
    bank = MergedLinearWeights()
    bank.apply(bank.prepare({"first": first, "second": second}))
    first._diffusion_lora_active_slices = (False, False)
    bank.apply(bank.prepare({"first": first, "second": second}))
    torch.testing.assert_close(first.base_layer.weight, base, rtol=0, atol=0)
    assert not torch.equal(second.base_layer.weight, base)


@pytest.mark.parametrize("failure", ["shape", "nonfinite"])
def test_failed_prepare_leaves_previous_policy_live(failure):
    first, second = _layer(), _layer()
    bank = MergedLinearWeights()
    modules = {"first": first, "second": second}
    bank.apply(bank.prepare(modules))
    previous = [layer.base_layer.weight.detach().clone() for layer in modules.values()]
    if failure == "shape":
        second.lora_b_stacked = (torch.zeros(1, 1, 2, 2), second.lora_b_stacked[1])
    else:
        second.lora_a_stacked[0].fill_(float("nan"))
    with pytest.raises(ValueError):
        bank.prepare(modules)
    for layer, value in zip(modules.values(), previous):
        torch.testing.assert_close(layer.base_layer.weight, value, rtol=0, atol=0)


def test_prepare_batches_finite_checks_per_device(monkeypatch):
    modules = {"first": _layer(), "second": _layer()}
    item = torch.Tensor.item
    item_calls = 0

    def count_item(tensor, *args, **kwargs):
        nonlocal item_calls
        item_calls += 1
        return item(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "item", count_item)
    MergedLinearWeights().prepare(modules)

    assert item_calls == 1


def test_copy_failure_restores_every_previously_written_weight(monkeypatch):
    modules = {"first": _layer(), "second": _layer()}
    bank = MergedLinearWeights()
    bank.apply(bank.prepare(modules))
    previous = [layer.base_layer.weight.detach().clone() for layer in modules.values()]
    for layer in modules.values():
        layer.lora_a_stacked[0].mul_(2)
    prepared = bank.prepare(modules)
    copy = torch.Tensor.copy_
    failed = False

    def fail_second_copy(tensor, value, *args, **kwargs):
        nonlocal failed
        if tensor is modules["second"].base_layer.weight and not failed:
            failed = True
            raise RuntimeError("injected copy failure")
        return copy(tensor, value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", fail_second_copy)
    with pytest.raises(RuntimeError, match="injected copy"):
        bank.apply(prepared)
    for layer, value in zip(modules.values(), previous):
        torch.testing.assert_close(layer.base_layer.weight, value, rtol=0, atol=0)


def test_rollback_snapshot_is_a_host_copy():
    layer = _layer(torch.float32)
    weight = layer.base_layer.weight
    snapshot = MergedLinearWeights._rollback_snapshot(weight)
    assert snapshot.device.type == "cpu"
    assert snapshot.data_ptr() != weight.data_ptr()
    torch.testing.assert_close(snapshot, weight, rtol=0, atol=0)


def test_dense_base_refresh_rebuilds_bank():
    layer = _layer()
    bank = MergedLinearWeights()
    bank.apply(bank.prepare({"proj": layer}))
    bank.close()
    with torch.no_grad():
        layer.base_layer.weight.fill_(3)
    bank.apply(bank.prepare({"proj": layer}))
    bank.close()
    torch.testing.assert_close(layer.base_layer.weight, torch.full_like(layer.base_layer.weight, 3))


def test_forward_is_dense_with_no_adapter_operations():
    layer = _layer(torch.float32)
    bank = MergedLinearWeights()
    bank.apply(bank.prepare({"proj": layer}))
    x = torch.randn(2, 4)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        output = layer.base_layer(x)
    torch.testing.assert_close(output, torch.nn.functional.linear(x, layer.base_layer.weight))
    assert sum(event.count for event in trace.key_averages() if event.key == "aten::mm") == 1


def _failed_prepare_rank(rank: int, port: int) -> None:
    torch.distributed.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    try:
        layer = _layer()
        bank = MergedLinearWeights()
        bank.apply(bank.prepare({"proj": layer}))
        previous = layer.base_layer.weight.detach().clone()
        if rank == 1:
            layer.lora_b_stacked = (torch.ones(1, 1, 2, 2), *layer.lora_b_stacked[1:])
        error = None
        try:
            bank.prepare({"proj": layer})
        except ValueError as exc:
            error = exc
        with pytest.raises((ValueError, RuntimeError)):
            check_merge_error(error, torch.distributed.group.WORLD)
        torch.testing.assert_close(layer.base_layer.weight, previous, rtol=0, atol=0)
    finally:
        torch.distributed.destroy_process_group()


def test_prepare_failure_aborts_both_ranks_without_changing_policy():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    torch.multiprocessing.spawn(_failed_prepare_rank, args=(port,), nprocs=2)


def test_pristine_bank_is_host_resident_and_built_once():
    layer = _layer(torch.bfloat16)
    bank = MergedLinearWeights()
    prepared1 = bank.prepare({"proj": layer})
    assert bank._base["proj"].device.type == "cpu"
    base_ptr = bank._base["proj"].data_ptr()
    bank.apply(prepared1)

    layer.lora_b_stacked[0].mul_(2.0)
    prepared2 = bank.prepare({"proj": layer})
    assert bank._base["proj"].device.type == "cpu"
    assert bank._base["proj"].data_ptr() == base_ptr
    bank.apply(prepared2)

    bank.close()
    assert not bank._base
    assert not bank._fp32_scratch
    assert not bank._cast_scratch


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_parking_branches_select_device_or_host_from_free_hbm(dtype):
    first, second = _layer(dtype), _layer(dtype)
    modules = {"first": first, "second": second}
    active_bytes = first.base_layer.weight.nbytes + second.base_layer.weight.nbytes
    margin = 1024
    threshold = 2 * active_bytes + margin

    # Exact threshold -> device parking (and float32 prepared must not alias scratch).
    dev_bank = MergedLinearWeights(
        parking_hbm_margin_bytes=margin,
        free_device_memory_fn=lambda _dev: threshold,
    )
    dev_prepared = dev_bank.prepare(modules)
    assert dev_bank.last_parking_modes[torch.device("cpu")] == "device"
    assert dev_prepared["first"].data_ptr() != dev_prepared["second"].data_ptr()
    assert dev_prepared["first"].data_ptr() != dev_bank._fp32_scratch[torch.device("cpu")].data_ptr()

    # One byte below threshold -> host parking with cast scratch.
    host_bank = MergedLinearWeights(
        parking_hbm_margin_bytes=margin,
        free_device_memory_fn=lambda _dev: threshold - 1,
    )
    host_prepared = host_bank.prepare(modules)
    assert host_bank.last_parking_modes[torch.device("cpu")] == "host"
    assert host_prepared["first"].device.type == "cpu"
    assert (torch.device("cpu"), dtype) in host_bank._cast_scratch

    # Both parking branches produce bitwise-identical results.
    for name in modules:
        torch.testing.assert_close(dev_prepared[name], host_prepared[name], rtol=0, atol=0)

    host_bank.apply(host_prepared)
    for name, layer in modules.items():
        torch.testing.assert_close(layer.base_layer.weight, dev_prepared[name], rtol=0, atol=0)


def test_device_scratch_is_reused_across_syncs():
    layer = _layer(torch.bfloat16)
    bank = MergedLinearWeights(
        parking_hbm_margin_bytes=1024,
        free_device_memory_fn=lambda _dev: 0,
    )
    bank.apply(bank.prepare({"proj": layer}))
    fp32_ptr = bank._fp32_scratch[torch.device("cpu")].data_ptr()
    cast_ptr = bank._cast_scratch[(torch.device("cpu"), torch.bfloat16)].data_ptr()

    for step in range(3):
        layer.lora_b_stacked[0].fill_(float(step + 1))
        bank.apply(bank.prepare({"proj": layer}))
        assert bank._fp32_scratch[torch.device("cpu")].data_ptr() == fp32_ptr
        assert bank._cast_scratch[(torch.device("cpu"), torch.bfloat16)].data_ptr() == cast_ptr


def test_host_parked_copy_failure_rolls_back_all_weights(monkeypatch):
    modules = {"first": _layer(torch.bfloat16), "second": _layer(torch.bfloat16)}
    bank = MergedLinearWeights(
        parking_hbm_margin_bytes=1024,
        free_device_memory_fn=lambda _dev: 0,
    )
    bank.apply(bank.prepare(modules))
    previous = {name: layer.base_layer.weight.detach().clone() for name, layer in modules.items()}

    for layer in modules.values():
        layer.lora_a_stacked[0].mul_(3.0)
    prepared = bank.prepare(modules)
    assert bank.last_parking_modes[torch.device("cpu")] == "host"

    orig_copy = torch.Tensor.copy_
    failed = False

    def fail_second(tensor, value, *args, **kwargs):
        nonlocal failed
        if tensor is modules["second"].base_layer.weight and not failed:
            failed = True
            raise RuntimeError("injected host-parked copy failure")
        return orig_copy(tensor, value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", fail_second)
    with pytest.raises(RuntimeError, match="injected host-parked copy failure"):
        bank.apply(prepared)

    for name, layer in modules.items():
        torch.testing.assert_close(layer.base_layer.weight, previous[name], rtol=0, atol=0)


def test_pinned_arena_falls_back_to_chunked_pinned_and_pageable(caplog):
    modules = {"first": _layer(torch.float32), "second": _layer(torch.float32), "third": _layer(torch.float32)}
    expected_bank = MergedLinearWeights()
    expected = expected_bank.prepare(modules)

    # Tier 1 -> Tier 2: fail single arena (> 512 bytes), succeed per-chunk pinned (<= 512 bytes).
    chunk_calls: list[int] = []

    def small_arena_alloc(nbytes: int) -> torch.Tensor:
        chunk_calls.append(nbytes)
        if nbytes > 512:
            raise RuntimeError("forced arena OOM")
        return torch.empty(nbytes, dtype=torch.uint8, device="cpu")

    with caplog.at_level("WARNING"):
        chunked_bank = MergedLinearWeights(
            pinned_chunk_max_bytes=256,
            free_device_memory_fn=lambda _dev: 0,
            use_pinned_host_fn=lambda _dev: True,
            alloc_pinned_bytes_fn=small_arena_alloc,
        )
        chunked_prepared = chunked_bank.prepare(modules)
    assert chunked_bank.last_host_alloc_mode == "chunked_pinned"
    assert any(size > 512 for size in chunk_calls)
    assert any(size <= 512 for size in chunk_calls)
    assert "falling back to chunked pinned" in caplog.text
    for name in modules:
        torch.testing.assert_close(chunked_prepared[name], expected[name], rtol=0, atol=0)

    # Tier 2 -> Tier 3: fail all pinned allocations, complete on pageable memory with warning.
    caplog.clear()

    def always_fail_pinned(nbytes: int) -> torch.Tensor:
        raise RuntimeError(f"pinned alloc disabled ({nbytes})")

    with caplog.at_level("WARNING"):
        pageable_bank = MergedLinearWeights(
            pinned_chunk_max_bytes=256,
            free_device_memory_fn=lambda _dev: 0,
            use_pinned_host_fn=lambda _dev: True,
            alloc_pinned_bytes_fn=always_fail_pinned,
        )
        pageable_prepared = pageable_bank.prepare(modules)
    assert pageable_bank.last_host_alloc_mode == "pageable"
    assert "falling back to pageable host memory" in caplog.text
    for name in modules:
        torch.testing.assert_close(pageable_prepared[name], expected[name], rtol=0, atol=0)


def test_failed_first_prepare_cleans_up_newly_captured_base():
    first, second = _layer(), _layer()
    second.lora_a_stacked[0].fill_(float("nan"))
    bank = MergedLinearWeights()
    with pytest.raises(ValueError, match="non-finite"):
        bank.prepare({"first": first, "second": second})
    assert not bank._base
    assert not bank._weights


def test_non_blocking_used_for_pinned_copies(monkeypatch):
    layer = _layer(torch.bfloat16)
    bank = MergedLinearWeights(
        free_device_memory_fn=lambda _dev: 0,
        use_pinned_host_fn=lambda _dev: True,
        alloc_pinned_bytes_fn=lambda nbytes: torch.empty(nbytes, dtype=torch.uint8, device="cpu"),
    )
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: True)

    orig_copy = torch.Tensor.copy_
    non_blocking_flags: list[bool] = []

    def record_copy(self, src, non_blocking=False):
        non_blocking_flags.append(bool(non_blocking))
        return orig_copy(self, src, non_blocking=False)

    monkeypatch.setattr(torch.Tensor, "copy_", record_copy)
    prepared = bank.prepare({"proj": layer})
    bank.apply(prepared)
    assert any(non_blocking_flags), "expected non_blocking=True copies for pinned host buffers"
