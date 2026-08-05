# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch
from vllm.lora.lora_weights import LoRALayerWeights
from vllm.lora.utils import get_supported_lora_modules

from tests.diffusion.lora.helpers import (
    DummyBaseLayerWithLoRA,
    FakeLinearBase,
    fake_replace_submodule,
)
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.lora.request import LoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummyLoRALayer:
    def __init__(self, n_slices: int, output_slices: tuple[int, ...]):
        self.n_slices = n_slices
        self.output_slices = output_slices
        # Keyed by slot index
        self.set_calls: list[
            tuple[int, list[torch.Tensor | None] | torch.Tensor, list[torch.Tensor | None] | torch.Tensor]
        ] = []
        self.reset_calls: list[int] = []
        self._n_active_adapters: int = 0

    def set_lora(self, index: int, lora_a, lora_b):
        self.set_calls.append((index, lora_a, lora_b))

    def reset_lora(self, index: int):
        self.reset_calls.append(index)


# Aliases for backward compatibility within this file
_FakeLinearBase = FakeLinearBase
_DummyBaseLayerWithLoRA = DummyBaseLayerWithLoRA


class _DummyPipeline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.foo = _FakeLinearBase()


class _DummyLM(torch.nn.Module):
    """LoRA enabled wrapper for _DummyPipeline."""

    def __init__(self, rank: int, lora_a_val: float = 1.0, lora_b_val: float = 1.0):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.foo = _DummyBaseLayerWithLoRA(_FakeLinearBase())
        self.rank = rank
        self.loras = self.get_lora_modules(lora_a_val, lora_b_val)

    def get_lora_modules(self, lora_a_val: float = 1.0, lora_b_val: float = 1.0):
        return {"transformer.foo": self._get_initial_lora(self.rank, lora_a_val, lora_b_val)}

    def get_lora(self, k: str) -> LoRALayerWeights:
        """Get the unscaled LoRA weights for transformer.foo"""
        return self.loras[k]

    def _get_initial_lora(self, rank: int, lora_a_val: float = 1.0, lora_b_val: float = 1.0) -> LoRALayerWeights:
        """Initializes a dummy LoRA for the current rank."""
        A = torch.ones((rank, 4)) * lora_a_val
        B = torch.ones((4, rank)) * lora_b_val
        return LoRALayerWeights(
            module_name="foo",
            rank=rank,
            lora_alpha=rank,
            lora_a=A,
            lora_b=B,
        )


def test_lora_manager_supported_modules_are_stable_with_wrapped_layers(monkeypatch):
    # Simulate a pipeline that already contains LoRA wrappers where the original
    # LinearBase is nested under ".base_layer".
    import vllm_omni.diffusion.lora.manager as manager_mod

    monkeypatch.setattr(manager_mod, "BaseLayerWithLoRA", _DummyBaseLayerWithLoRA)

    pipeline = _DummyLM(rank=2)

    # vLLM helper would see only the nested LinearBase and yield "base_layer".
    assert get_supported_lora_modules(pipeline) == ["base_layer"]

    manager = DiffusionLoRAManager(
        pipeline=pipeline,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_cached_adapters=1,
    )

    assert "foo" in manager._supported_lora_modules
    assert "base_layer" not in manager._supported_lora_modules


def test_lora_manager_replace_layers_does_not_rewrap_base_layer(monkeypatch):
    import vllm_omni.diffusion.lora.manager as manager_mod

    monkeypatch.setattr(manager_mod, "BaseLayerWithLoRA", _DummyBaseLayerWithLoRA)

    def _fake_from_layer_diffusion(*, layer: torch.nn.Module, **_kwargs):
        if isinstance(layer, _FakeLinearBase):
            return _DummyBaseLayerWithLoRA(layer)
        return layer

    replace_calls: list[str] = []

    def _fake_replace_submodule(root: torch.nn.Module, module_name: str, submodule: torch.nn.Module):
        replace_calls.append(module_name)
        setattr(root, module_name, submodule)

    monkeypatch.setattr(manager_mod, "from_layer_diffusion", _fake_from_layer_diffusion)
    monkeypatch.setattr(manager_mod, "replace_submodule", _fake_replace_submodule)

    pipeline = torch.nn.Module()
    pipeline.transformer = torch.nn.Module()
    pipeline.transformer.foo = _FakeLinearBase()

    manager = DiffusionLoRAManager(
        pipeline=pipeline,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_cached_adapters=1,
    )

    peft_helper = type("_PH", (), {"r": 1})()

    manager._replace_layers_with_lora(peft_helper)
    manager._replace_layers_with_lora(peft_helper)

    # Only the top-level layer should have been replaced; nested ".base_layer"
    # must be skipped to avoid nesting LoRA wrappers.
    assert replace_calls == ["foo"]


def test_lora_manager_replaces_packed_layer_when_targeting_sublayers(monkeypatch):
    import vllm_omni.diffusion.lora.manager as manager_mod

    monkeypatch.setattr(manager_mod, "BaseLayerWithLoRA", _DummyBaseLayerWithLoRA)

    def _fake_from_layer_diffusion(*, layer: torch.nn.Module, **_kwargs):
        return _DummyBaseLayerWithLoRA(layer)

    replace_calls: list[str] = []

    def _fake_replace_submodule(root: torch.nn.Module, module_name: str, submodule: torch.nn.Module):
        replace_calls.append(module_name)
        setattr(root, module_name, submodule)

    monkeypatch.setattr(manager_mod, "from_layer_diffusion", _fake_from_layer_diffusion)
    monkeypatch.setattr(manager_mod, "replace_submodule", _fake_replace_submodule)

    pipeline = torch.nn.Module()
    pipeline.stacked_params_mapping = [
        (".to_qkv.", ".to_q.", "q"),
        (".to_qkv.", ".to_k.", "k"),
        (".to_qkv.", ".to_v.", "v"),
    ]
    pipeline.transformer = torch.nn.Module()
    pipeline.transformer.to_qkv = _FakeLinearBase()

    manager = DiffusionLoRAManager(
        pipeline=pipeline,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_cached_adapters=1,
    )

    # Treat the dummy layer as a packed 3-slice projection so the manager uses
    # `stacked_params_mapping` to decide replacement based on target_modules.
    monkeypatch.setattr(manager, "_get_packed_modules_list", lambda _module: ["q", "k", "v"])

    peft_helper = type("_PH", (), {"r": 1, "target_modules": ["to_q"]})()
    manager._replace_layers_with_lora(peft_helper)

    assert replace_calls == ["to_qkv"]


def test_lora_manager_activates_fused_lora_on_packed_layer():
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_cached_adapters=1,
    )

    packed_layer = _DummyLoRALayer(n_slices=3, output_slices=(2, 1, 1))
    manager._lora_modules = {"transformer.blocks.0.attn.to_qkv": packed_layer}

    rank = 2
    A = torch.ones((rank, 4))
    B = torch.arange(0, sum(packed_layer.output_slices) * rank, dtype=torch.bfloat16).view(-1, rank)
    lora = LoRALayerWeights(
        module_name="transformer.blocks.0.attn.to_qkv",
        rank=rank,
        lora_alpha=rank,
        lora_a=A,
        lora_b=B,
    )
    manager._registered_adapters = {
        7: type(
            "LM",
            (),
            {
                "id": 7,
                "loras": {"transformer.blocks.0.attn.to_qkv": lora},
                "get_lora": lambda self, k: self.loras.get(k),
            },
        )()
    }

    manager._activate_adapters([7], [0.5])

    # Filter set_calls for slot 0
    slot0_sets = [(a, b) for idx, a, b in packed_layer.set_calls if idx == 0]
    assert len(slot0_sets) == 1
    lora_a_list, lora_b_list = slot0_sets[0]
    assert isinstance(lora_a_list, list)
    assert isinstance(lora_b_list, list)
    assert len(lora_a_list) == 3
    assert len(lora_b_list) == 3
    assert all(torch.allclose(a, A) for a in lora_a_list)
    # B should be split into 3 slices and scaled.
    b0, b1, b2 = lora_b_list
    assert b0.shape[0] == 2 and b1.shape[0] == 1 and b2.shape[0] == 1
    assert torch.allclose(torch.cat([b0, b1, b2], dim=0), B * 0.5)


def test_lora_manager_activates_packed_lora_from_sublayers():
    pipeline = torch.nn.Module()
    pipeline.stacked_params_mapping = [
        (".to_qkv", ".to_q", "q"),
        (".to_qkv", ".to_k", "k"),
        (".to_qkv", ".to_v", "v"),
    ]
    manager = DiffusionLoRAManager(
        pipeline=pipeline,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_cached_adapters=1,
    )

    packed_layer = _DummyLoRALayer(n_slices=3, output_slices=(2, 1, 1))
    manager._lora_modules = {"transformer.blocks.0.attn.to_qkv": packed_layer}

    rank = 2
    loras: dict[str, LoRALayerWeights] = {}
    for name, out_dim in zip(["to_q", "to_k", "to_v"], [2, 1, 1]):
        loras[f"transformer.blocks.0.attn.{name}"] = LoRALayerWeights(
            module_name=f"transformer.blocks.0.attn.{name}",
            rank=rank,
            lora_alpha=rank,
            lora_a=torch.ones((rank, 4)) * (1 if name == "to_q" else 2),
            lora_b=torch.ones((out_dim, rank)) * (3 if name == "to_q" else 4),
        )

    manager._registered_adapters = {
        1: type("LM", (), {"id": 1, "loras": loras, "get_lora": lambda self, k: self.loras.get(k)})()
    }

    manager._activate_adapters([1], [2.0])

    slot0_sets = [(a, b) for idx, a, b in packed_layer.set_calls if idx == 0]
    assert len(slot0_sets) == 1
    lora_a_list, lora_b_list = slot0_sets[0]
    assert isinstance(lora_a_list, list)
    assert isinstance(lora_b_list, list)
    assert len(lora_a_list) == 3
    assert len(lora_b_list) == 3
    # Scale should apply to B only.
    assert torch.allclose(lora_b_list[0], torch.ones((2, rank)) * 3 * 2.0)
    assert torch.allclose(lora_b_list[1], torch.ones((1, rank)) * 4 * 2.0)
    assert torch.allclose(lora_b_list[2], torch.ones((1, rank)) * 4 * 2.0)


def _dummy_lora_request(adapter_id: int) -> LoRARequest:
    return LoRARequest(
        lora_name=f"adapter_{adapter_id}",
        lora_int_id=adapter_id,
        lora_path=f"/tmp/adapter_{adapter_id}",
    )


def test_lora_manager_evicts_lru_adapter_when_cache_full(monkeypatch):
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_cached_adapters=2,
    )

    def _fake_load(_req: LoRARequest):
        lora_model = type("LM", (), {"id": _req.lora_int_id})()
        peft_helper = type("PH", (), {})()
        return lora_model, peft_helper

    monkeypatch.setattr(manager, "_load_adapter", _fake_load)
    monkeypatch.setattr(manager, "_replace_layers_with_lora", lambda _peft: None)
    monkeypatch.setattr(manager, "_activate_adapters", lambda _ids, _scales: None)

    req1 = _dummy_lora_request(1)
    req2 = _dummy_lora_request(2)
    req3 = _dummy_lora_request(3)

    manager.set_active_adapters([req1], [1.0])
    manager.set_active_adapters([req2], [1.0])

    # Touch adapter 1 so adapter 2 becomes LRU.
    manager.set_active_adapters([req1], [1.0])

    manager.set_active_adapters([req3], [1.0])

    assert set(manager.list_adapters()) == {1, 3}


def test_lora_manager_does_not_evict_pinned_adapter(monkeypatch):
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_cached_adapters=2,
    )

    def _fake_load(_req: LoRARequest):
        lora_model = type("LM", (), {"id": _req.lora_int_id})()
        peft_helper = type("PH", (), {})()
        return lora_model, peft_helper

    monkeypatch.setattr(manager, "_load_adapter", _fake_load)
    monkeypatch.setattr(manager, "_replace_layers_with_lora", lambda _peft: None)
    monkeypatch.setattr(manager, "_activate_adapters", lambda _ids, _scales: None)

    manager.set_active_adapters([_dummy_lora_request(1)], [1.0])
    assert manager.pin_adapter(1)

    manager.set_active_adapters([_dummy_lora_request(2)], [1.0])
    manager.set_active_adapters([_dummy_lora_request(3)], [1.0])

    assert set(manager.list_adapters()) == {1, 3}


def test_lora_manager_warns_when_all_adapters_pinned(monkeypatch):
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_cached_adapters=2,
    )

    def _fake_load(_req: LoRARequest):
        lora_model = type("LM", (), {"id": _req.lora_int_id})()
        peft_helper = type("PH", (), {})()
        return lora_model, peft_helper

    monkeypatch.setattr(manager, "_load_adapter", _fake_load)
    monkeypatch.setattr(manager, "_replace_layers_with_lora", lambda _peft: None)
    monkeypatch.setattr(manager, "_activate_adapters", lambda _ids, _scales: None)

    manager.set_active_adapters([_dummy_lora_request(1)], [1.0])
    manager.set_active_adapters([_dummy_lora_request(2)], [1.0])

    assert manager.pin_adapter(1)
    assert manager.pin_adapter(2)

    manager.max_cached_adapters = 1
    manager._evict_for_new_adapter()

    assert set(manager.list_adapters()) == {1, 2}


def test_lora_manager_applies_multiple_scales_correctly(monkeypatch):
    """Ensure that the LoRA manager applies scales correctly when the
    active adapter receives a different scale, i.e., the rank is unchanged.
    """
    import vllm_omni.diffusion.lora.manager as manager_mod

    monkeypatch.setattr(manager_mod, "BaseLayerWithLoRA", _DummyLoRALayer)

    rank = 2
    adapter_id = 7
    req1 = _dummy_lora_request(adapter_id)
    scale_1 = 0.25
    scale_2 = 0.5

    lora_model = _DummyLM(rank=rank)
    manager = DiffusionLoRAManager(
        pipeline=_DummyPipeline(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    def _fake_load(_req: LoRARequest):
        peft_helper = type("PH", (), {"r": rank})()
        return lora_model, peft_helper

    monkeypatch.setattr(manager, "_load_adapter", _fake_load)
    manager._registered_adapters = {
        adapter_id: lora_model,
    }
    manager._lora_modules = {"transformer.foo": lora_model.transformer.foo}

    # After the first scale, all B values should go from 1 -> scale_1
    manager.set_active_adapters([req1], [scale_1])
    slot0_sets = [(a, b) for idx, a, b in lora_model.transformer.foo.set_calls if idx == 0]
    assert len(slot0_sets) == 1
    lora_a, lora_b = slot0_sets[0]
    assert torch.all(lora_a == 1)
    assert torch.all(lora_b == scale_1)

    # After the second scale, all B values should go from 1 -> scale_2
    manager.set_active_adapters([req1], [scale_2])
    slot0_sets = [(a, b) for idx, a, b in lora_model.transformer.foo.set_calls if idx == 0]
    assert len(slot0_sets) == 2

    lora_a, lora_b = slot0_sets[1]
    assert torch.all(lora_a == 1)
    assert torch.all(lora_b == scale_2)


def test_lora_manager_scales_correctly_with_rank_changes(monkeypatch):
    """Ensure that the LoRA manager correctly handles scaling when the rank
    is changed and the buffers are reset + we reactivate.
    """
    import vllm_omni.diffusion.lora.manager as manager_mod

    monkeypatch.setattr(manager_mod, "BaseLayerWithLoRA", _DummyBaseLayerWithLoRA)

    rank = 2
    adapter_id = 7
    req1 = _dummy_lora_request(adapter_id)
    initial_scale = 0.5

    lora_model = _DummyLM(rank=rank)
    manager = DiffusionLoRAManager(
        pipeline=_DummyPipeline(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    def _fake_load(_req: LoRARequest):
        peft_helper = type("PH", (), {"r": rank})()
        return lora_model, peft_helper

    monkeypatch.setattr(manager, "_load_adapter", _fake_load)
    manager._registered_adapters = {
        adapter_id: lora_model,
    }
    manager._lora_modules = {"transformer.foo": lora_model.transformer.foo}

    # Activate adapter with initial scale
    manager.set_active_adapters([req1], [initial_scale])
    assert lora_model.transformer.foo.create_calls == 0
    slot0_sets = [(a, b) for idx, a, b in lora_model.transformer.foo.set_calls if idx == 0]
    assert len(slot0_sets) == 1
    lora_a, lora_b = slot0_sets[0]
    assert torch.all(lora_a == 1)
    assert torch.all(lora_b == initial_scale)

    # Increase the rank; this resets the buffers, so the adapter is activated again
    manager._ensure_max_lora_rank(8)

    # Ensure we actually took the rank expansion path, which recreates
    # and sets the weight buffers, but that the scale didn't change
    assert lora_model.transformer.foo.create_calls == 1
    slot0_sets = [(a, b) for idx, a, b in lora_model.transformer.foo.set_calls if idx == 0]
    assert len(slot0_sets) == 2
    lora_a, lora_b = slot0_sets[1]
    assert torch.all(lora_a == 1)
    assert torch.all(lora_b == initial_scale)


def test_scale_rounding():
    """Ensure that scales are rounded for comparison."""
    assert DiffusionLoRAManager._get_rounded_scale(0.0031) == 0.003


def test_lora_manager_uses_valid_max_rank(monkeypatch):
    """Ensure that the LoRA manager uses a valid max rank for vLLM."""
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    # Ensure that the rank is correctly adjusted to the smallest valid max rank
    supported_max_rank = 64
    unsupported_max_rank = 63
    assert supported_max_rank in DiffusionLoRAManager._VALID_MAX_RANKS
    assert unsupported_max_rank not in DiffusionLoRAManager._VALID_MAX_RANKS

    def _fake_load(_req: LoRARequest):
        lora_model = type("LM", (), {"id": _req.lora_int_id})()
        peft_helper = type("PH", (), {"r": unsupported_max_rank})()
        return lora_model, peft_helper

    monkeypatch.setattr(manager, "_load_adapter", _fake_load)
    req1 = _dummy_lora_request(1)
    manager.add_adapter(req1)
    assert manager._max_lora_rank == supported_max_rank


@pytest.mark.parametrize("rank", [-1, 0, DiffusionLoRAManager._VALID_MAX_RANKS[-1] + 1])
def test_lora_manager_max_rank_validation(monkeypatch, rank):
    """Check that invalid max ranks are handled correctly."""
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    lora_rank = rank

    def _fake_load(_req: LoRARequest):
        lora_model = type("LM", (), {"id": _req.lora_int_id})()
        peft_helper = type("PH", (), {"r": lora_rank})()
        return lora_model, peft_helper

    monkeypatch.setattr(manager, "_load_adapter", _fake_load)
    req1 = _dummy_lora_request(1)
    with pytest.raises(ValueError):
        manager.add_adapter(req1)


# ============================================================
# Multi-LoRA composition tests
# ============================================================


def test_multi_adapter_activation():
    """Verify multiple adapters are set into separate slots."""
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_loras=3,
        max_cached_adapters=3,
    )

    layer = _DummyLoRALayer(n_slices=1, output_slices=(4,))
    manager._lora_modules = {"transformer.foo": layer}

    rank = 2
    adapters = {}
    for aid in [1, 2, 3]:
        lora = LoRALayerWeights(
            module_name="foo",
            rank=rank,
            lora_alpha=rank,
            lora_a=torch.ones((rank, 4)) * aid,
            lora_b=torch.ones((4, rank)) * aid,
        )
        adapters[aid] = type(
            "LM", (), {"id": aid, "loras": {"transformer.foo": lora}, "get_lora": lambda self, k: self.loras.get(k)}
        )()

    manager._registered_adapters = adapters

    manager._activate_adapters([1, 2, 3], [0.5, 0.75, 1.0])

    # Should have 3 set_lora calls (one per adapter) for the single layer
    set_by_slot = {idx: (a, b) for idx, a, b in layer.set_calls}
    assert 0 in set_by_slot
    assert 1 in set_by_slot
    assert 2 in set_by_slot

    # Verify weights are correct per slot
    a0, b0 = set_by_slot[0]
    assert torch.allclose(a0, torch.ones((rank, 4)) * 1)
    assert torch.allclose(b0, torch.ones((4, rank)) * 1 * 0.5)

    a1, b1 = set_by_slot[1]
    assert torch.allclose(a1, torch.ones((rank, 4)) * 2)
    assert torch.allclose(b1, torch.ones((4, rank)) * 2 * 0.75)

    a2, b2 = set_by_slot[2]
    assert torch.allclose(a2, torch.ones((rank, 4)) * 3)
    assert torch.allclose(b2, torch.ones((4, rank)) * 3 * 1.0)

    assert layer._n_active_adapters == 3


def test_multi_adapter_unused_slots_are_reset():
    """When going from 3 adapters to 1, unused slots should be reset."""
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_loras=3,
        max_cached_adapters=3,
    )

    layer = _DummyLoRALayer(n_slices=1, output_slices=(4,))
    manager._lora_modules = {"transformer.foo": layer}

    rank = 2
    adapters = {}
    for aid in [1, 2, 3]:
        lora = LoRALayerWeights(
            module_name="foo",
            rank=rank,
            lora_alpha=rank,
            lora_a=torch.ones((rank, 4)),
            lora_b=torch.ones((4, rank)),
        )
        adapters[aid] = type(
            "LM", (), {"id": aid, "loras": {"transformer.foo": lora}, "get_lora": lambda self, k: self.loras.get(k)}
        )()

    manager._registered_adapters = adapters

    # Activate 3
    manager._activate_adapters([1, 2, 3], [1.0, 1.0, 1.0])
    assert layer._n_active_adapters == 3

    # Now activate only 1 — slots 1 and 2 should be reset
    layer.set_calls.clear()
    layer.reset_calls.clear()

    manager._activate_adapters([1], [1.0])

    assert layer._n_active_adapters == 1
    # Slots 1 and 2 should have been reset
    assert 1 in layer.reset_calls
    assert 2 in layer.reset_calls


def test_multi_adapter_exceeds_max_loras():
    """Requesting more adapters than max_loras should raise ValueError."""
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_loras=2,
        max_cached_adapters=3,
    )

    with pytest.raises(ValueError, match="max_loras"):
        manager.set_active_adapters(
            [_dummy_lora_request(1), _dummy_lora_request(2), _dummy_lora_request(3)],
            [1.0, 1.0, 1.0],
        )


def test_multi_adapter_mismatched_lengths():
    """lora_requests and lora_scales must have the same length."""
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_loras=2,
    )

    with pytest.raises(ValueError, match="same length"):
        manager.set_active_adapters(
            [_dummy_lora_request(1), _dummy_lora_request(2)],
            [1.0],
        )


def test_multi_adapter_empty_list_deactivates():
    """Empty request list should deactivate all adapters."""
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_loras=2,
        max_cached_adapters=2,
    )

    layer = _DummyLoRALayer(n_slices=1, output_slices=(4,))
    manager._lora_modules = {"transformer.foo": layer}

    # Activate something first
    manager._active_adapter_ids = [1]
    manager._active_adapter_scales = [1.0]

    manager.set_active_adapters([], [])

    assert manager._active_adapter_ids == []
    assert manager._active_adapter_scales == []


def test_multi_adapter_skips_zero_scale(monkeypatch):
    """Adapters with scale 0 should be filtered out."""
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_loras=2,
        max_cached_adapters=2,
    )

    layer = _DummyLoRALayer(n_slices=1, output_slices=(4,))
    manager._lora_modules = {"transformer.foo": layer}

    rank = 2
    lora = LoRALayerWeights(
        module_name="foo",
        rank=rank,
        lora_alpha=rank,
        lora_a=torch.ones((rank, 4)),
        lora_b=torch.ones((4, rank)),
    )
    for aid in [1, 2]:
        manager._registered_adapters[aid] = type(
            "LM", (), {"id": aid, "loras": {"transformer.foo": lora}, "get_lora": lambda self, k: self.loras.get(k)}
        )()

    manager.set_active_adapters(
        [_dummy_lora_request(1), _dummy_lora_request(2)],
        [1.0, 0.0],  # adapter 2 has scale 0
    )

    # Only adapter 1 should be active
    assert manager._active_adapter_ids == [1]
    assert layer._n_active_adapters == 1


def test_multi_adapter_are_active_at_scales():
    """Test the _are_active_at_scales comparison."""
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_loras=3,
    )
    manager._active_adapter_ids = [1, 2]
    manager._active_adapter_scales = [0.5, 0.75]

    assert manager._are_active_at_scales([1, 2], [0.5, 0.75])
    assert not manager._are_active_at_scales([1, 2], [0.5, 0.8])
    assert not manager._are_active_at_scales([1], [0.5])
    assert not manager._are_active_at_scales([2, 1], [0.75, 0.5])


def test_lora_manager_discovers_bagel_component(monkeypatch):
    """Verify that _replace_layers_with_lora finds layers under 'bagel'."""
    import vllm_omni.diffusion.lora.manager as manager_mod

    monkeypatch.setattr(manager_mod, "BaseLayerWithLoRA", _DummyBaseLayerWithLoRA)

    def _fake_from_layer_diffusion(*, layer: torch.nn.Module, **_kwargs):
        if isinstance(layer, _FakeLinearBase):
            return _DummyBaseLayerWithLoRA(layer)
        return layer

    replace_calls: list[str] = []

    monkeypatch.setattr(manager_mod, "from_layer_diffusion", _fake_from_layer_diffusion)
    monkeypatch.setattr(
        manager_mod,
        "replace_submodule",
        lambda root, name, sub: fake_replace_submodule(root, name, sub, replace_calls),
    )

    # Pipeline with a 'bagel' component (no 'transformer')
    pipeline = torch.nn.Module()
    pipeline.bagel = torch.nn.Module()
    pipeline.bagel.language_model = torch.nn.Module()
    pipeline.bagel.language_model.qkv_proj = _FakeLinearBase()

    manager = DiffusionLoRAManager(
        pipeline=pipeline,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_cached_adapters=1,
    )

    peft_helper = type("_PH", (), {"r": 1})()
    manager._replace_layers_with_lora(peft_helper)

    assert "language_model.qkv_proj" in replace_calls
    assert "bagel.language_model.qkv_proj" in manager._lora_modules
    # Verify the module was actually replaced in the tree (not just recorded)
    assert isinstance(pipeline.bagel.language_model.qkv_proj, _DummyBaseLayerWithLoRA)


def test_lora_manager_discovers_unet_component(monkeypatch):
    """Verify that _replace_layers_with_lora finds layers under 'unet'."""
    import vllm_omni.diffusion.lora.manager as manager_mod

    monkeypatch.setattr(manager_mod, "BaseLayerWithLoRA", _DummyBaseLayerWithLoRA)

    def _fake_from_layer_diffusion(*, layer: torch.nn.Module, **_kwargs):
        if isinstance(layer, _FakeLinearBase):
            return _DummyBaseLayerWithLoRA(layer)
        return layer

    replace_calls: list[str] = []

    monkeypatch.setattr(manager_mod, "from_layer_diffusion", _fake_from_layer_diffusion)
    monkeypatch.setattr(
        manager_mod,
        "replace_submodule",
        lambda root, name, sub: fake_replace_submodule(root, name, sub, replace_calls),
    )

    # Pipeline with a 'unet' component (no 'transformer')
    pipeline = torch.nn.Module()
    pipeline.unet = torch.nn.Module()
    pipeline.unet.down_block = torch.nn.Module()
    pipeline.unet.down_block.proj = _FakeLinearBase()

    manager = DiffusionLoRAManager(
        pipeline=pipeline,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_cached_adapters=1,
    )

    peft_helper = type("_PH", (), {"r": 1})()
    manager._replace_layers_with_lora(peft_helper)

    assert "down_block.proj" in replace_calls
    assert "unet.down_block.proj" in manager._lora_modules
    # Verify the module was actually replaced in the tree (not just recorded)
    assert isinstance(pipeline.unet.down_block.proj, _DummyBaseLayerWithLoRA)
