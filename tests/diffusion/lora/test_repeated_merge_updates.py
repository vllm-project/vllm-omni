# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for repeated diffusion LoRA merge-on-load updates."""

from __future__ import annotations

import pytest
import torch
from vllm.lora.lora_weights import LoRALayerWeights

from tests.diffusion.lora.helpers import FakeLinearBase
from tests.diffusion.lora.test_merge_on_load import (
    IN_DIM,
    _make_lora,
    _make_manager,
    _MergeableLoRALayer,
    _register,
    _StubLoRAModel,
)
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_named_lora(name: str, rank: int, out_dim: int, seed: int) -> LoRALayerWeights:
    gen = torch.Generator().manual_seed(seed)
    return LoRALayerWeights(
        module_name=name,
        rank=rank,
        lora_alpha=rank,
        lora_a=torch.randn(rank, IN_DIM, generator=gen),
        lora_b=torch.randn(out_dim, rank, generator=gen),
    )


def _make_two_layer_manager() -> tuple[DiffusionLoRAManager, _MergeableLoRALayer, _MergeableLoRALayer]:
    pipeline = torch.nn.Module()
    pipeline.transformer = torch.nn.Module()
    pipeline.transformer.foo = FakeLinearBase()
    pipeline.transformer.bar = FakeLinearBase()
    manager = DiffusionLoRAManager(
        pipeline=pipeline,
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_cached_adapters=3,
        merge_on_load=True,
    )
    foo = _MergeableLoRALayer((6,))
    bar = _MergeableLoRALayer((5,))
    manager._lora_modules = {
        "transformer.foo": foo,
        "transformer.bar": bar,
    }
    return manager, foo, bar


def _register_many(
    manager: DiffusionLoRAManager,
    adapter_id: int,
    loras: dict[str, LoRALayerWeights],
) -> None:
    manager._registered_adapters[adapter_id] = _StubLoRAModel(adapter_id, loras)


def _canonical(base: torch.Tensor, lora: LoRALayerWeights) -> torch.Tensor:
    expected = base.clone()
    delta_fp32 = lora.lora_b.float() @ lora.lora_a.float()
    expected.add_(delta_fp32.to(expected.dtype))
    return expected


def test_direct_switch_writes_shared_target_once():
    layer = _MergeableLoRALayer((6,))
    manager = _make_manager(merge_on_load=True, layer=layer)
    first = _make_lora(rank=2, out_dim=6, seed=10)
    second = _make_lora(rank=2, out_dim=6, seed=11)
    _register(manager, 1, first)
    _register(manager, 2, second)

    base = layer.base_layer.weight.detach().clone()
    address = layer.base_layer.weight.data_ptr()
    manager._activate_adapter(1, scale=1.0)
    version_before_switch = layer.base_layer.weight._version

    manager._activate_adapter(2, scale=1.0)

    assert torch.equal(layer.base_layer.weight, _canonical(base, second))
    assert layer.base_layer.weight.data_ptr() == address
    # A direct pristine + delta overwrite mutates the destination once. The old
    # restore-then-add path increments the tensor version twice.
    assert layer.base_layer.weight._version == version_before_switch + 1
    assert manager._merged_layer_names == {"transformer.foo"}


def test_target_set_change_restores_old_only_layer():
    manager, foo, bar = _make_two_layer_manager()
    foo_first = _make_named_lora("foo", rank=2, out_dim=6, seed=20)
    bar_first = _make_named_lora("bar", rank=2, out_dim=5, seed=21)
    foo_second = _make_named_lora("foo", rank=2, out_dim=6, seed=22)
    _register_many(
        manager,
        1,
        {
            "transformer.foo": foo_first,
            "transformer.bar": bar_first,
        },
    )
    _register_many(manager, 2, {"transformer.foo": foo_second})
    foo_base = foo.base_layer.weight.detach().clone()
    bar_base = bar.base_layer.weight.detach().clone()

    manager._activate_adapter(1, scale=1.0)
    manager._activate_adapter(2, scale=1.0)

    assert torch.equal(foo.base_layer.weight, _canonical(foo_base, foo_second))
    assert torch.equal(bar.base_layer.weight, bar_base)
    assert manager._merged_layer_names == {"transformer.foo"}


def test_disjoint_target_switch_restores_old_and_snapshots_new():
    manager, foo, bar = _make_two_layer_manager()
    foo_first = _make_named_lora("foo", rank=2, out_dim=6, seed=23)
    bar_second = _make_named_lora("bar", rank=2, out_dim=5, seed=24)
    _register_many(manager, 1, {"transformer.foo": foo_first})
    _register_many(manager, 2, {"transformer.bar": bar_second})
    foo_base = foo.base_layer.weight.detach().clone()
    bar_base = bar.base_layer.weight.detach().clone()

    manager._activate_adapter(1, scale=1.0)
    manager._activate_adapter(2, scale=1.0)

    assert torch.equal(foo.base_layer.weight, foo_base)
    assert torch.equal(bar.base_layer.weight, _canonical(bar_base, bar_second))
    assert manager._merged_layer_names == {"transformer.bar"}


@pytest.mark.parametrize("scale", [0.0, -0.5, 2.0])
def test_zero_negative_and_changed_scales_follow_reference(scale: float):
    layer = _MergeableLoRALayer((6,))
    manager = _make_manager(merge_on_load=True, layer=layer)
    lora = _make_lora(rank=2, out_dim=6, seed=25)
    _register(manager, 1, lora)
    base = layer.base_layer.weight.detach().clone()

    manager._activate_adapter(1, scale=scale)

    scaled = LoRALayerWeights(
        module_name=lora.module_name,
        rank=lora.rank,
        lora_alpha=lora.lora_alpha,
        lora_a=lora.lora_a,
        lora_b=lora.lora_b * scale,
    )
    assert torch.equal(layer.base_layer.weight, _canonical(base, scaled))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_repeated_switches_always_use_canonical_base(dtype: torch.dtype):
    layer = _MergeableLoRALayer((6,))
    layer.base_layer.weight = torch.nn.Parameter(layer.base_layer.weight.detach().to(dtype))
    manager = _make_manager(merge_on_load=True, layer=layer)
    first = _make_lora(rank=2, out_dim=6, seed=30)
    second = _make_lora(rank=2, out_dim=6, seed=31)
    _register(manager, 1, first)
    _register(manager, 2, second)
    base = layer.base_layer.weight.detach().clone()

    for update in range(1000):
        adapter_id, lora = (1, first) if update % 2 == 0 else (2, second)
        manager._activate_adapter(adapter_id, scale=1.0)
        assert torch.equal(layer.base_layer.weight, _canonical(base, lora))
        assert torch.equal(manager._pristine_weights["transformer.foo"], base)

    manager._deactivate_all_adapters()
    assert torch.equal(layer.base_layer.weight, base)


def test_mid_update_failure_rolls_back_to_no_lora(monkeypatch: pytest.MonkeyPatch):
    manager, foo, bar = _make_two_layer_manager()
    first = {
        "transformer.foo": _make_named_lora("foo", 2, 6, 40),
        "transformer.bar": _make_named_lora("bar", 2, 5, 41),
    }
    second = {
        "transformer.foo": _make_named_lora("foo", 2, 6, 42),
        "transformer.bar": _make_named_lora("bar", 2, 5, 43),
    }
    _register_many(manager, 1, first)
    _register_many(manager, 2, second)
    foo_base = foo.base_layer.weight.detach().clone()
    bar_base = bar.base_layer.weight.detach().clone()
    manager._activate_adapter(1, scale=1.0)

    real_add = torch.add
    calls = 0

    def fail_second_add(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected direct-overwrite failure")
        return real_add(*args, **kwargs)

    monkeypatch.setattr(torch, "add", fail_second_add)
    with pytest.raises(RuntimeError, match="injected direct-overwrite failure"):
        manager._activate_adapter(2, scale=1.0)

    assert torch.equal(foo.base_layer.weight, foo_base)
    assert torch.equal(bar.base_layer.weight, bar_base)
    assert manager._active_adapter_id is None
    assert manager._merged_layer_names == set()
    assert not manager._merged
    assert not any(foo._diffusion_lora_active_slices)
    assert not any(bar._diffusion_lora_active_slices)


def test_same_adapter_id_and_scale_remains_immutable_noop():
    layer = _MergeableLoRALayer((6,))
    manager = _make_manager(merge_on_load=True, layer=layer)
    first = _make_lora(rank=2, out_dim=6, seed=50)
    replacement = _make_lora(rank=2, out_dim=6, seed=51)
    _register(manager, 7, first)
    manager._activate_adapter(7, scale=1.0)
    applied = layer.base_layer.weight.detach().clone()

    manager._registered_adapters[7].loras["transformer.foo"] = replacement
    manager._activate_adapter(7, scale=1.0)

    assert torch.equal(layer.base_layer.weight, applied)


def test_unmerged_external_base_update_refreshes_pristine_snapshot():
    layer = _MergeableLoRALayer((6,))
    manager = _make_manager(merge_on_load=True, layer=layer)
    first = _make_lora(rank=2, out_dim=6, seed=60)
    second = _make_lora(rank=2, out_dim=6, seed=61)
    _register(manager, 1, first)
    _register(manager, 2, second)
    manager._activate_adapter(1, scale=1.0)
    manager._deactivate_all_adapters()

    with torch.no_grad():
        layer.base_layer.weight.add_(3.0)
    new_base = layer.base_layer.weight.detach().clone()
    manager._activate_adapter(2, scale=1.0)

    assert torch.equal(layer.base_layer.weight, _canonical(new_base, second))
    assert torch.equal(manager._pristine_weights["transformer.foo"], new_base)


def test_external_write_while_merged_is_rejected():
    layer = _MergeableLoRALayer((6,))
    manager = _make_manager(merge_on_load=True, layer=layer)
    _register(manager, 1, _make_lora(rank=2, out_dim=6, seed=62))
    _register(manager, 2, _make_lora(rank=2, out_dim=6, seed=63))
    base = layer.base_layer.weight.detach().clone()
    manager._activate_adapter(1, scale=1.0)

    with torch.no_grad():
        layer.base_layer.weight.add_(1.0)

    with pytest.raises(RuntimeError, match="modified outside the manager"):
        manager._activate_adapter(2, scale=1.0)

    assert torch.equal(layer.base_layer.weight, base)
    assert manager._active_adapter_id is None
    assert manager._merged_layer_names == set()


def test_removing_active_adapter_restores_canonical_base():
    layer = _MergeableLoRALayer((6,))
    manager = _make_manager(merge_on_load=True, layer=layer)
    _register(manager, 1, _make_lora(rank=2, out_dim=6, seed=64))
    base = layer.base_layer.weight.detach().clone()
    manager._activate_adapter(1, scale=1.0)

    assert manager.remove_adapter(1)

    assert torch.equal(layer.base_layer.weight, base)
    assert manager._active_adapter_id is None
    assert not manager._merged


def test_switch_with_zero_binding_restores_no_lora_state():
    layer = _MergeableLoRALayer((6,))
    manager = _make_manager(merge_on_load=True, layer=layer)
    _register(manager, 1, _make_lora(rank=2, out_dim=6, seed=70))
    manager._registered_adapters[2] = _StubLoRAModel(
        2,
        {"transformer.not_present": _make_named_lora("not_present", 2, 6, 71)},
    )
    base = layer.base_layer.weight.detach().clone()
    manager._activate_adapter(1, scale=1.0)

    with pytest.raises(ValueError, match="applies to no layer"):
        manager._activate_adapter(2, scale=1.0)

    assert torch.equal(layer.base_layer.weight, base)
    assert manager._active_adapter_id is None
    assert manager._merged_layer_names == set()


def test_shared_weight_storage_is_rejected_before_merge():
    manager, foo, bar = _make_two_layer_manager()
    bar = _MergeableLoRALayer((6,))
    manager._lora_modules["transformer.bar"] = bar
    bar.base_layer.weight = foo.base_layer.weight
    first = {
        "transformer.foo": _make_named_lora("foo", 2, 6, 80),
        "transformer.bar": _make_named_lora("bar", 2, 6, 81),
    }
    _register_many(manager, 1, first)
    base = foo.base_layer.weight.detach().clone()

    with pytest.raises(RuntimeError, match="shared storage"):
        manager._activate_adapter(1, scale=1.0)

    assert torch.equal(foo.base_layer.weight, base)
    assert manager._active_adapter_id is None
    assert manager._merged_layer_names == set()


def test_noncontiguous_weight_preserves_layout_and_identity():
    layer = _MergeableLoRALayer((6,))
    noncontiguous = torch.randn(IN_DIM, 6).T
    assert not noncontiguous.is_contiguous()
    layer.base_layer.weight = torch.nn.Parameter(noncontiguous)
    manager = _make_manager(merge_on_load=True, layer=layer)
    lora = _make_lora(rank=2, out_dim=6, seed=90)
    _register(manager, 1, lora)
    base = layer.base_layer.weight.detach().clone()
    parameter = layer.base_layer.weight
    address = parameter.data_ptr()
    stride = parameter.stride()

    manager._activate_adapter(1, scale=1.0)

    assert torch.equal(parameter, _canonical(base, lora))
    assert layer.base_layer.weight is parameter
    assert parameter.data_ptr() == address
    assert parameter.stride() == stride
