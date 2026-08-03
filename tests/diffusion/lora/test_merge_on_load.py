# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DiffusionLoRAManager merge-on-load (single-adapter fast path)."""

from __future__ import annotations

import pytest
import torch
from vllm.lora.lora_weights import LoRALayerWeights
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from tests.diffusion.lora.helpers import FakeLinearBase
from vllm_omni.diffusion.lora.layers.base_linear import DiffusionBaseLinearLayerWithLoRA
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

IN_DIM = 8
MAX_RANK = 4


class _MergeableLoRALayer(torch.nn.Module):
    """Real-tensor stand-in matching DiffusionBaseLinearLayerWithLoRA's buffer layout."""

    def __init__(self, out_slices: tuple[int, ...], quant_method=None):
        super().__init__()
        self.n_slices = len(out_slices)
        self.output_slices = tuple(out_slices)
        out_dim = sum(out_slices)
        base = torch.nn.Module()
        base.weight = torch.nn.Parameter(torch.randn(out_dim, IN_DIM))
        base.quant_method = quant_method
        self.base_layer = base
        self.lora_a_stacked = tuple(torch.zeros(1, 1, MAX_RANK, IN_DIM) for _ in out_slices)
        self.lora_b_stacked = tuple(torch.zeros(1, 1, s, MAX_RANK) for s in out_slices)
        self._diffusion_lora_active_slices = (False,) * self.n_slices
        self.reset_calls = 0

    def set_lora(self, index: int, lora_a, lora_b):
        assert index == 0
        a_list = lora_a if isinstance(lora_a, list) else [lora_a]
        b_list = lora_b if isinstance(lora_b, list) else [lora_b]
        active = []
        for i, (a, b) in enumerate(zip(a_list, b_list)):
            if a is None or b is None:
                active.append(False)
                continue
            self.lora_a_stacked[i].zero_()
            self.lora_b_stacked[i].zero_()
            self.lora_a_stacked[i][0, 0, : a.shape[0], :] = a
            self.lora_b_stacked[i][0, 0, :, : b.shape[1]] = b
            active.append(True)
        active += [False] * (self.n_slices - len(active))
        self._diffusion_lora_active_slices = tuple(active)

    def reset_lora(self, index: int):
        assert index == 0
        self.reset_calls += 1
        for a in self.lora_a_stacked:
            a.zero_()
        for b in self.lora_b_stacked:
            b.zero_()
        self._diffusion_lora_active_slices = (False,) * self.n_slices

    def create_lora_weights(self, max_loras, lora_config, model_config=None):
        max_rank = lora_config.max_lora_rank
        self.lora_a_stacked = tuple(torch.zeros(1, 1, max_rank, IN_DIM) for _ in self.output_slices)
        self.lora_b_stacked = tuple(torch.zeros(1, 1, s, max_rank) for s in self.output_slices)
        self._diffusion_lora_active_slices = (False,) * self.n_slices


class _StubLoRAModel:
    def __init__(self, loras: dict[str, LoRALayerWeights]):
        self._loras = loras

    def get_lora(self, name: str):
        return self._loras.get(name)


def _make_manager(merge_on_load: bool, layer: torch.nn.Module) -> DiffusionLoRAManager:
    pipeline = torch.nn.Module()
    pipeline.transformer = torch.nn.Module()
    pipeline.transformer.foo = FakeLinearBase()
    manager = DiffusionLoRAManager(
        pipeline=pipeline,
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_cached_adapters=2,
        merge_on_load=merge_on_load,
    )
    manager._lora_modules = {"transformer.foo": layer}
    return manager


def _make_lora(rank: int, out_dim: int, seed: int) -> LoRALayerWeights:
    gen = torch.Generator().manual_seed(seed)
    return LoRALayerWeights(
        module_name="foo",
        rank=rank,
        lora_alpha=rank,
        lora_a=torch.randn(rank, IN_DIM, generator=gen),
        lora_b=torch.randn(out_dim, rank, generator=gen),
    )


def _register(manager: DiffusionLoRAManager, adapter_id: int, lora: LoRALayerWeights):
    manager._registered_adapters[adapter_id] = _StubLoRAModel({"transformer.foo": lora})


def test_merge_applies_delta_and_resets_buffers():
    layer = _MergeableLoRALayer((6,))
    manager = _make_manager(merge_on_load=True, layer=layer)
    lora = _make_lora(rank=2, out_dim=6, seed=0)
    _register(manager, 7, lora)

    w0 = layer.base_layer.weight.detach().clone()
    manager._activate_adapter(7, scale=1.0)

    expected = w0 + lora.lora_b @ lora.lora_a
    assert torch.allclose(layer.base_layer.weight, expected, atol=1e-6)
    assert manager._merged
    assert torch.equal(manager._pristine_weights["transformer.foo"], w0)
    # Buffers consumed: apply() must take the all-inactive fast path.
    assert not any(layer._diffusion_lora_active_slices)
    assert all(a.abs().sum() == 0 for a in layer.lora_a_stacked)


def test_deactivate_restores_bitwise():
    layer = _MergeableLoRALayer((6,))
    manager = _make_manager(merge_on_load=True, layer=layer)
    _register(manager, 7, _make_lora(rank=2, out_dim=6, seed=0))

    w0 = layer.base_layer.weight.detach().clone()
    manager._activate_adapter(7, scale=1.0)
    assert not torch.equal(layer.base_layer.weight, w0)

    manager._deactivate_all_adapters()
    assert torch.equal(layer.base_layer.weight, w0)
    assert not manager._merged


def test_switch_adapters_does_not_accumulate():
    layer = _MergeableLoRALayer((6,))
    manager = _make_manager(merge_on_load=True, layer=layer)
    lora1 = _make_lora(rank=2, out_dim=6, seed=1)
    lora2 = _make_lora(rank=2, out_dim=6, seed=2)
    _register(manager, 1, lora1)
    _register(manager, 2, lora2)

    w0 = layer.base_layer.weight.detach().clone()
    manager._activate_adapter(1, scale=1.0)
    manager._activate_adapter(2, scale=1.0)

    expected = w0 + lora2.lora_b @ lora2.lora_a
    assert torch.allclose(layer.base_layer.weight, expected, atol=1e-6)
    # Pristine copy still reflects the original weights, not adapter 1's merge.
    assert torch.equal(manager._pristine_weights["transformer.foo"], w0)


def test_scale_change_remerges():
    layer = _MergeableLoRALayer((6,))
    manager = _make_manager(merge_on_load=True, layer=layer)
    lora = _make_lora(rank=2, out_dim=6, seed=3)
    _register(manager, 7, lora)

    w0 = layer.base_layer.weight.detach().clone()
    manager._activate_adapter(7, scale=1.0)
    manager._activate_adapter(7, scale=2.0)

    expected = w0 + 2.0 * (lora.lora_b @ lora.lora_a)
    assert torch.allclose(layer.base_layer.weight, expected, atol=1e-6)


def test_rank_growth_while_merged():
    """A larger-rank adapter arriving while another is merged goes through
    _ensure_max_lora_rank, which recreates buffers and re-activates the merged
    adapter mid-flight; the final weight must be pristine + the new delta."""
    layer = _MergeableLoRALayer((6,))
    manager = _make_manager(merge_on_load=True, layer=layer)
    small = _make_lora(rank=2, out_dim=6, seed=5)
    big = _make_lora(rank=16, out_dim=6, seed=6)
    _register(manager, 1, small)
    _register(manager, 2, big)

    w0 = layer.base_layer.weight.detach().clone()
    manager._ensure_max_lora_rank(small.rank)
    manager._activate_adapter(1, scale=1.0)
    assert manager._merged
    assert manager._max_lora_rank < big.rank  # next adapter must force a rank bump

    # Mirrors _load_adapter: the bump happens while adapter 1 is still merged.
    manager._ensure_max_lora_rank(big.rank)
    manager._activate_adapter(2, scale=1.0)

    expected = w0 + big.lora_b @ big.lora_a
    assert torch.allclose(layer.base_layer.weight, expected, atol=1e-5)
    assert torch.equal(manager._pristine_weights["transformer.foo"], w0)

    manager._deactivate_all_adapters()
    assert torch.equal(layer.base_layer.weight, w0)


def test_packed_layer_merges_only_active_slices():
    layer = _MergeableLoRALayer((4, 3))
    manager = _make_manager(merge_on_load=True, layer=layer)

    w0 = layer.base_layer.weight.detach().clone()
    A0 = torch.randn(2, IN_DIM)
    B0 = torch.randn(4, 2)
    layer.set_lora(0, [A0, None], [B0, None])
    manager._merge_active_adapter()

    assert torch.allclose(layer.base_layer.weight[:4], w0[:4] + B0 @ A0, atol=1e-6)
    assert torch.equal(layer.base_layer.weight[4:], w0[4:])


def test_quantized_layer_at_merge_time_raises():
    # Merging only runs in shadow mode (creation-time fallback installs
    # wrappers instead), so an ineligible layer at merge time must fail loudly
    # rather than silently dropping the adapter.
    layer = _MergeableLoRALayer((6,), quant_method=object())
    manager = _make_manager(merge_on_load=True, layer=layer)
    layer.set_lora(0, torch.randn(2, IN_DIM), torch.randn(6, 2))
    with pytest.raises(RuntimeError, match="cannot be weight-merged"):
        manager._merge_active_adapter()


def test_merge_disabled_by_default():
    layer = _MergeableLoRALayer((6,))
    manager = _make_manager(merge_on_load=False, layer=layer)
    _register(manager, 7, _make_lora(rank=2, out_dim=6, seed=4))

    w0 = layer.base_layer.weight.detach().clone()
    manager._activate_adapter(7, scale=1.0)

    assert torch.equal(layer.base_layer.weight, w0)
    assert not manager._merged
    assert any(layer._diffusion_lora_active_slices)


def _run_replace(monkeypatch, merge_on_load: bool, quant_method=None):
    """Drive _replace_layers_with_lora with fakes; returns (manager, replace_calls)."""
    import vllm_omni.diffusion.lora.manager as manager_mod

    class _Wrapper(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base_layer = base

    monkeypatch.setattr(manager_mod, "BaseLayerWithLoRA", _Wrapper)
    monkeypatch.setattr(
        manager_mod,
        "from_layer_diffusion",
        lambda *, layer, **_k: _Wrapper(layer) if isinstance(layer, FakeLinearBase) else layer,
    )
    replace_calls: list[str] = []
    monkeypatch.setattr(manager_mod, "replace_submodule", lambda _root, name, _sub: replace_calls.append(name))

    pipeline = torch.nn.Module()
    pipeline.transformer = torch.nn.Module()
    target = FakeLinearBase()
    target.weight = torch.nn.Parameter(torch.randn(4, 4))
    target.quant_method = quant_method
    pipeline.transformer.foo = target

    manager = DiffusionLoRAManager(
        pipeline=pipeline,
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_cached_adapters=1,
        merge_on_load=merge_on_load,
    )
    peft_helper = type("_PH", (), {"r": 2})()
    manager._replace_layers_with_lora(peft_helper)
    return manager, replace_calls


def test_merge_mode_shadow_wraps_without_installing(monkeypatch):
    manager, replace_calls = _run_replace(monkeypatch, merge_on_load=True)
    # Module tree untouched (torch.compile graphs unaffected), wrapper tracked.
    assert replace_calls == []
    assert "transformer.foo" in manager._lora_modules
    assert manager._merge_enabled
    assert not manager._wrappers_installed


def test_standard_mode_installs_wrappers(monkeypatch):
    manager, replace_calls = _run_replace(monkeypatch, merge_on_load=False)
    assert replace_calls == ["foo"]
    assert manager._wrappers_installed


def test_merge_mode_quantized_target_falls_back_to_install(monkeypatch):
    manager, replace_calls = _run_replace(monkeypatch, merge_on_load=True, quant_method=object())
    assert replace_calls == ["foo"]
    # The user's request is preserved; only the effective mode is cleared.
    assert manager.merge_on_load
    assert not manager._merge_enabled
    assert manager._wrappers_installed


class _WeightReadingQuantMethod(UnquantizedLinearMethod):
    """Reads base_layer.weight at call time so merged weights take effect.

    Subclasses UnquantizedLinearMethod so the merge eligibility check passes.
    """

    def apply(self, base_layer, x: torch.Tensor, bias: torch.Tensor | None):
        y = x @ base_layer.weight.t()
        if bias is not None:
            y = y + bias
        return y


def test_merged_output_matches_lora_apply():
    """End-to-end numerical consistency: real apply() before vs after merge."""
    out_dim, rank = 6, 2
    layer = DiffusionBaseLinearLayerWithLoRA.__new__(DiffusionBaseLinearLayerWithLoRA)
    layer.tp_size = 1
    # Plain object, not nn.Module: the __new__ skeleton has no _modules dict,
    # so assigning an nn.Module (or Parameter) to it would raise.
    base = type("Base", (), {})()
    base.weight = torch.randn(out_dim, IN_DIM)
    base.quant_method = _WeightReadingQuantMethod()
    layer.base_layer = base
    layer.n_slices = 1
    layer.output_slices = (out_dim,)
    layer.lora_a_stacked = (torch.zeros(1, 1, MAX_RANK, IN_DIM),)
    layer.lora_b_stacked = (torch.zeros(1, 1, out_dim, MAX_RANK),)

    A = torch.randn(rank, IN_DIM)
    B = torch.randn(out_dim, rank)
    layer.lora_a_stacked[0][0, 0, :rank, :] = A
    layer.lora_b_stacked[0][0, 0, :, :rank] = B
    layer._diffusion_lora_active_slices = (True,)

    x = torch.randn(5, IN_DIM)
    y_lora = layer.apply(x)

    manager = _make_manager(merge_on_load=True, layer=layer)
    manager._merge_active_adapter()
    assert manager._merged
    y_merged = layer.apply(x)

    assert torch.allclose(y_lora, y_merged, rtol=1e-5, atol=1e-5)
