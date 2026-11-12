# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project


import pytest
import torch
from vllm.utils.network_utils import get_file_store_init_method

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


def _failed_prepare_rank(rank: int, init_method: str) -> None:
    torch.distributed.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2)
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
    torch.multiprocessing.spawn(_failed_prepare_rank, args=(get_file_store_init_method(),), nprocs=2)
