# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch
from vllm.lora.lora_weights import LoRALayerWeights
from vllm.lora.utils import get_supported_lora_modules
from vllm.model_executor.layers.linear import LinearBase

from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.lora.request import LoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummyLoRALayer:
    def __init__(self, n_slices: int, output_slices: tuple[int, ...]):
        self.n_slices = n_slices
        self.output_slices = output_slices
        self.set_calls: list[
            tuple[list[torch.Tensor | None] | torch.Tensor, list[torch.Tensor | None] | torch.Tensor]
        ] = []
        self.reset_calls: int = 0

    def set_lora(self, index: int, lora_a, lora_b):
        assert index == 0
        self.set_calls.append((lora_a, lora_b))

    def reset_lora(self, index: int):
        assert index == 0
        self.reset_calls += 1


class _FakeLinearBase(LinearBase):
    def __init__(self):
        torch.nn.Module.__init__(self)


def test_lora_manager_supported_modules_are_stable_with_wrapped_layers(monkeypatch):
    # Simulate a pipeline that already contains LoRA wrappers where the original
    # LinearBase is nested under ".base_layer".
    import vllm_omni.diffusion.lora.manager as manager_mod

    class _DummyBaseLayerWithLoRA(torch.nn.Module):
        def __init__(self, base_layer: torch.nn.Module):
            super().__init__()
            self.base_layer = base_layer

    monkeypatch.setattr(manager_mod, "BaseLayerWithLoRA", _DummyBaseLayerWithLoRA)

    pipeline = torch.nn.Module()
    pipeline.transformer = torch.nn.Module()
    pipeline.transformer.foo = _DummyBaseLayerWithLoRA(_FakeLinearBase())

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

    class _DummyBaseLayerWithLoRA(torch.nn.Module):
        def __init__(self, base_layer: torch.nn.Module):
            super().__init__()
            self.base_layer = base_layer

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

    class _DummyBaseLayerWithLoRA(torch.nn.Module):
        def __init__(self, base_layer: torch.nn.Module):
            super().__init__()
            self.base_layer = base_layer

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
    pipeline.packed_modules_mapping = {"to_qkv": ["to_q", "to_k", "to_v"]}
    pipeline.transformer = torch.nn.Module()
    pipeline.transformer.to_qkv = _FakeLinearBase()

    manager = DiffusionLoRAManager(
        pipeline=pipeline,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_cached_adapters=1,
    )

    # Treat the dummy layer as a packed 3-slice projection so the manager uses
    # `packed_modules_mapping` to decide replacement based on target_modules.
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
    manager._adapter_scales = {7: 0.5}

    manager._activate_adapter(7)

    assert packed_layer.reset_calls == 0
    assert len(packed_layer.set_calls) == 1
    lora_a_list, lora_b_list = packed_layer.set_calls[0]
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
    pipeline.packed_modules_mapping = {"to_qkv": ["to_q", "to_k", "to_v"]}
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
    manager._adapter_scales = {1: 2.0}

    manager._activate_adapter(1)

    assert packed_layer.reset_calls == 0
    assert len(packed_layer.set_calls) == 1
    lora_a_list, lora_b_list = packed_layer.set_calls[0]
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
    monkeypatch.setattr(manager, "_activate_adapter", lambda _adapter_id: None)

    req1 = _dummy_lora_request(1)
    req2 = _dummy_lora_request(2)
    req3 = _dummy_lora_request(3)

    manager.set_active_adapter(req1, lora_scale=1.0)
    manager.set_active_adapter(req2, lora_scale=1.0)

    # Touch adapter 1 so adapter 2 becomes LRU.
    manager.set_active_adapter(req1, lora_scale=1.0)

    manager.set_active_adapter(req3, lora_scale=1.0)

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
    monkeypatch.setattr(manager, "_activate_adapter", lambda _adapter_id: None)

    manager.set_active_adapter(_dummy_lora_request(1), lora_scale=1.0)
    assert manager.pin_adapter(1)

    manager.set_active_adapter(_dummy_lora_request(2), lora_scale=1.0)
    manager.set_active_adapter(_dummy_lora_request(3), lora_scale=1.0)

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
    monkeypatch.setattr(manager, "_activate_adapter", lambda _adapter_id: None)

    manager.set_active_adapter(_dummy_lora_request(1), lora_scale=1.0)
    manager.set_active_adapter(_dummy_lora_request(2), lora_scale=1.0)

    assert manager.pin_adapter(1)
    assert manager.pin_adapter(2)

    manager.max_cached_adapters = 1
    manager._evict_if_needed()

    assert set(manager.list_adapters()) == {1, 2}
