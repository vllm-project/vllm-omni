# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

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
from vllm_omni.diffusion.lora.loader import LoadedDiffusionLoRA
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.lora.plan import AdditiveBiasUpdate, DiffusionLoRAApplyPlan
from vllm_omni.diffusion.lora.types import WeightedLoRA
from vllm_omni.lora.request import LoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummyLoRALayer:
    def __init__(self, n_slices: int, output_slices: tuple[int, ...]):
        self.n_slices = n_slices
        self.output_slices = output_slices
        self.set_calls: list[
            tuple[list[torch.Tensor | None] | torch.Tensor, list[torch.Tensor | None] | torch.Tensor]
        ] = []
        self.bias_calls: list[torch.Tensor | list[torch.Tensor | None] | None] = []
        self.reset_calls: int = 0

    def set_lora(self, index: int, lora_a, lora_b):
        assert index == 0
        self.set_calls.append((lora_a, lora_b))

    def set_additive_bias(self, bias):
        self.bias_calls.append(bias)

    def reset_lora(self, index: int):
        assert index == 0
        self.reset_calls += 1


class _FusableLoRALayer:
    def __init__(self, in_features: int, out_features: int, max_rank: int):
        self.n_slices = 1
        self.output_slices = (out_features,)
        self.base_layer = torch.nn.Linear(in_features, out_features, bias=True)
        torch.nn.init.zeros_(self.base_layer.weight)
        torch.nn.init.zeros_(self.base_layer.bias)
        self.lora_a_stacked = [torch.zeros(1, 1, max_rank, in_features)]
        self.lora_b_stacked = [torch.zeros(1, 1, out_features, max_rank)]
        self._diffusion_lora_active_slices = (False,)
        self._diffusion_additive_bias = (None,)

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
    ):
        assert index == 0
        rank = lora_a.shape[0]
        self.lora_a_stacked[0].zero_()
        self.lora_b_stacked[0].zero_()
        self.lora_a_stacked[0][0, 0, :rank].copy_(lora_a)
        self.lora_b_stacked[0][0, 0, :, :rank].copy_(lora_b)
        self._diffusion_lora_active_slices = (True,)

    def set_additive_bias(self, bias: torch.Tensor | None):
        self._diffusion_additive_bias = (bias,)

    def reset_lora(self, index: int):
        assert index == 0
        self.lora_a_stacked[0].zero_()
        self.lora_b_stacked[0].zero_()
        self._diffusion_lora_active_slices = (False,)
        self._diffusion_additive_bias = (None,)


# Aliases for backward compatibility within this file
_FakeLinearBase = FakeLinearBase
_DummyBaseLayerWithLoRA = DummyBaseLayerWithLoRA


def _loaded_lora(model, *updates) -> LoadedDiffusionLoRA:
    peft_helper = type("PH", (), {})()
    return LoadedDiffusionLoRA(model, peft_helper, tuple(updates))


class _DummyPipeline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.foo = _FakeLinearBase()


class _CustomApplyPlanPipeline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.container = torch.nn.Module()
        self.container.custom_dit = torch.nn.Module()
        self.container.custom_dit.proj = _FakeLinearBase()

    def get_lora_apply_plan(self) -> DiffusionLoRAApplyPlan:
        return DiffusionLoRAApplyPlan(
            component_names=("container.custom_dit",),
            target_modules=("proj",),
            packed_modules_mapping={"proj": ("query", "key")},
        )


class _DummyLM(torch.nn.Module):
    """LoRA enabled wrapper for _DummyPipeline."""

    def __init__(self, rank: int):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.foo = _DummyBaseLayerWithLoRA(_FakeLinearBase())
        self.rank = rank
        self.loras = self.get_lora_modules()

    def get_lora_modules(self):
        return {"transformer.foo": self._get_initial_lora(self.rank)}

    def get_lora(self, k: str) -> LoRALayerWeights:
        """Get the unscaled LoRA weights for transformer.foo"""
        return self.loras[k]

    def _get_initial_lora(self, rank: int) -> LoRALayerWeights:
        """Initializes a dummy LoRA for the current rank."""
        A = torch.ones((rank, 4))
        B = torch.ones((4, rank))
        return LoRALayerWeights(
            module_name="foo",
            rank=rank,
            lora_alpha=rank,
            lora_a=A,
            lora_b=B,
        )


def test_model_owned_apply_plan_describes_custom_application() -> None:
    manager = DiffusionLoRAManager(
        pipeline=_CustomApplyPlanPipeline(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert manager._component_names() == ("container.custom_dit",)
    assert manager._packed_modules_mapping == {"proj": ["query", "key"]}
    assert manager._component_relative_name("container.custom_dit.block.proj") == "block.proj"


@pytest.mark.parametrize(
    ("lora_names", "module_names", "match"),
    [
        (("transformer.typo.proj",), ("transformer.block.proj",), "unbound"),
        (("proj",), ("transformer.a.proj", "transformer.b.proj"), "ambiguous"),
    ],
)
def test_lora_manager_rejects_non_unique_weight_bindings(lora_names, module_names, match) -> None:
    manager = DiffusionLoRAManager(torch.nn.Module(), torch.device("cpu"), torch.bfloat16)
    manager._lora_modules = {name: _DummyLoRALayer(1, (2,)) for name in module_names}
    loras = dict.fromkeys(lora_names, object())
    model = type("LM", (), {"loras": loras, "get_lora": lambda self, name: self.loras.get(name)})()

    with pytest.raises(ValueError, match=match):
        manager._validate_lora_bindings(_loaded_lora(model))


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
        max_registered_adapters=1,
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
        max_registered_adapters=1,
    )

    peft_helper = type("_PH", (), {"r": 1})()

    manager._frozen = False
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
        max_registered_adapters=1,
    )

    # Treat the dummy layer as a packed 3-slice projection so the manager uses
    # `stacked_params_mapping` to decide replacement based on target_modules.
    monkeypatch.setattr(manager, "_get_packed_modules_list", lambda _module: ["q", "k", "v"])

    peft_helper = type("_PH", (), {"r": 1, "target_modules": ["to_q"]})()
    manager._frozen = False
    manager._replace_layers_with_lora(peft_helper)

    assert replace_calls == ["to_qkv"]


def test_lora_manager_activates_fused_lora_on_packed_layer():
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_registered_adapters=1,
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
        7: _loaded_lora(
            type(
                "LM",
                (),
                {
                    "id": 7,
                    "loras": {"transformer.blocks.0.attn.to_qkv": lora},
                    "get_lora": lambda self, k: self.loras.get(k),
                },
            )()
        )
    }
    manager._max_lora_rank = manager._get_smallest_valid_max_rank(rank)

    _activate_single_composition(manager, 7, 0.5)

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


def test_lora_manager_splits_fused_checkpoint_with_global_tp_sizes():
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_registered_adapters=1,
    )

    # Runtime buffers are rank-local, while the checkpoint B tensor is global.
    packed_layer = _DummyLoRALayer(n_slices=2, output_slices=(2, 2))
    packed_layer.output_sizes = (8, 8)
    manager._lora_modules = {"transformer.blocks.0.mlp.fc1": packed_layer}

    rank = 2
    a = torch.ones((rank, 4))
    b = torch.arange(16 * rank, dtype=torch.bfloat16).view(16, rank)
    lora = LoRALayerWeights(
        module_name="transformer.blocks.0.mlp.fc1",
        rank=rank,
        lora_alpha=rank,
        lora_a=a,
        lora_b=b,
    )
    manager._registered_adapters = {
        7: _loaded_lora(
            type(
                "LM",
                (),
                {
                    "id": 7,
                    "loras": {"transformer.blocks.0.mlp.fc1": lora},
                    "get_lora": lambda self, key: self.loras.get(key),
                },
            )()
        )
    }
    manager._max_lora_rank = manager._get_smallest_valid_max_rank(rank)

    _activate_single_composition(manager, 7, 1.0)

    lora_a, lora_b = packed_layer.set_calls[-1]
    assert isinstance(lora_a, list)
    assert isinstance(lora_b, list)
    assert [tensor.shape for tensor in lora_b] == [(8, rank), (8, rank)]
    torch.testing.assert_close(torch.cat(lora_b), b)


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
        max_registered_adapters=1,
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
        1: _loaded_lora(type("LM", (), {"id": 1, "loras": loras, "get_lora": lambda self, k: self.loras.get(k)})())
    }
    manager._max_lora_rank = manager._get_smallest_valid_max_rank(rank)

    _activate_single_composition(manager, 1, scale=2.0)

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


def test_lora_manager_composes_multiple_adapters_with_exact_math():
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_registered_adapters=2,
    )
    layer = _DummyLoRALayer(n_slices=1, output_slices=(2,))
    manager._lora_modules = {"transformer.proj": layer}

    a_1 = torch.tensor([[1.0, 2.0]])
    b_1 = torch.tensor([[3.0], [4.0]])
    a_2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    b_2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    def _model(adapter_id: int, a: torch.Tensor, b: torch.Tensor):
        weights = LoRALayerWeights(
            module_name="transformer.proj",
            rank=a.shape[0],
            lora_alpha=a.shape[0],
            lora_a=a,
            lora_b=b,
        )
        return type(
            "LM",
            (),
            {
                "id": adapter_id,
                "loras": {"transformer.proj": weights},
                "get_lora": lambda self, key: self.loras.get(key),
            },
        )()

    manager._registered_adapters = {
        1: _loaded_lora(_model(1, a_1, b_1)),
        2: _loaded_lora(_model(2, a_2, b_2)),
    }
    manager._max_lora_rank = manager._get_smallest_valid_max_rank(3)
    requests = (_dummy_lora_request(1), _dummy_lora_request(2))
    scales = (0.25, 0.75)

    manager.set_active_adapter(requests, scales)

    composed_a, composed_b = layer.set_calls[-1]
    x = torch.tensor([[2.0, -1.0]])
    actual = (x @ composed_a.T) @ composed_b.T
    expected = 0.25 * ((x @ a_1.T) @ b_1.T) + 0.75 * ((x @ a_2.T) @ b_2.T)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_prefused_and_dynamic_routes_share_the_same_delta_math():
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_registered_adapters=2,
    )
    layer = _FusableLoRALayer(in_features=2, out_features=2, max_rank=8)
    manager._lora_modules = {"transformer.proj": layer}
    manager._max_lora_rank = 8

    a_1 = torch.tensor([[1.0, 2.0]])
    b_1 = torch.tensor([[3.0], [4.0]])
    bias_1 = torch.tensor([2.0, -2.0])
    a_2 = torch.tensor([[2.0, -1.0]])
    b_2 = torch.tensor([[5.0], [6.0]])
    bias_2 = torch.tensor([4.0, 8.0])

    def _model(adapter_id: int, a: torch.Tensor, b: torch.Tensor):
        weights = LoRALayerWeights(
            module_name="transformer.proj",
            rank=a.shape[0],
            lora_alpha=a.shape[0],
            lora_a=a,
            lora_b=b,
        )
        return type(
            "LM",
            (),
            {
                "id": adapter_id,
                "loras": {"transformer.proj": weights},
                "get_lora": lambda self, key: self.loras.get(key),
            },
        )()

    manager._registered_adapters = {
        1: _loaded_lora(_model(1, a_1, b_1), AdditiveBiasUpdate("transformer.proj", bias_1)),
        2: _loaded_lora(_model(2, a_2, b_2), AdditiveBiasUpdate("transformer.proj", bias_2)),
    }
    manager._adapter_requests = {
        1: _dummy_lora_request(1),
        2: _dummy_lora_request(2),
    }

    _activate_single_composition(manager, 1, scale=0.25)
    manager._fuse_active_composition()
    torch.testing.assert_close(layer.base_layer.weight, 0.25 * (b_1 @ a_1), rtol=0, atol=0)
    torch.testing.assert_close(layer.base_layer.bias, 0.25 * bias_1, rtol=0, atol=0)

    _activate_single_composition(manager, 2, scale=0.75)
    dynamic_a = layer.lora_a_stacked[0][0, 0]
    dynamic_b = layer.lora_b_stacked[0][0, 0]
    effective_weight = layer.base_layer.weight + dynamic_b @ dynamic_a
    expected = 0.25 * (b_1 @ a_1) + 0.75 * (b_2 @ a_2)
    torch.testing.assert_close(effective_weight, expected, rtol=0, atol=0)
    dynamic_bias = layer._diffusion_additive_bias[0]
    assert dynamic_bias is not None
    torch.testing.assert_close(layer.base_layer.bias + dynamic_bias, 0.25 * bias_1 + 0.75 * bias_2, rtol=0, atol=0)


def test_additive_bias_update_does_not_require_low_rank_weights():
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    layer = _DummyLoRALayer(n_slices=1, output_slices=(2,))
    manager._lora_modules = {"transformer.proj": layer}
    lora_model = type("LM", (), {"id": 1, "get_lora": lambda _self, _key: None})()
    update = AdditiveBiasUpdate("transformer.proj", torch.tensor([2.0, -4.0]))
    manager._registered_adapters = {1: _loaded_lora(lora_model, update)}

    _activate_single_composition(manager, 1, scale=0.25)

    assert layer.reset_calls == 1
    torch.testing.assert_close(layer.bias_calls[-1], torch.tensor([0.5, -1.0]))


def test_auxiliary_update_must_bind_to_an_installed_lora_module():
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    manager._lora_modules = {"transformer.proj": _DummyLoRALayer(n_slices=1, output_slices=(2,))}
    loaded = _loaded_lora(
        type("LM", (), {})(),
        AdditiveBiasUpdate("transformer.missing", torch.ones(2)),
    )

    with pytest.raises(ValueError, match="does not match any installed LoRA module"):
        manager._validate_auxiliary_update_bindings(loaded)


def test_auxiliary_update_shape_is_validated_when_bound():
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    manager._lora_modules = {"transformer.proj": _DummyLoRALayer(n_slices=1, output_slices=(2,))}
    loaded = _loaded_lora(
        type("LM", (), {})(),
        AdditiveBiasUpdate("transformer.proj", torch.ones(1)),
    )

    with pytest.raises(ValueError, match=r"got \(1,\), expected \(2,\)"):
        manager._validate_auxiliary_update_bindings(loaded)


def test_auxiliary_update_binds_to_a_packed_logical_sublayer():
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    manager._packed_modules_mapping = {"to_qkv": ["to_q", "to_k", "to_v"]}
    manager._lora_modules = {
        "transformer.block.attn.to_qkv": _DummyLoRALayer(
            n_slices=3,
            output_slices=(4, 2, 2),
        )
    }
    loaded = _loaded_lora(
        type("LM", (), {})(),
        AdditiveBiasUpdate("transformer.block.attn.to_q", torch.ones(4)),
    )

    manager._validate_auxiliary_update_bindings(loaded)


def test_prefused_bf16_weight_uses_single_rounding():
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    layer = _FusableLoRALayer(in_features=1, out_features=1, max_rank=4)
    layer.base_layer.to(dtype=torch.bfloat16)
    with torch.no_grad():
        layer.base_layer.weight.fill_(1.0)
    layer.lora_a_stacked = [torch.zeros(1, 1, 4, 1, dtype=torch.bfloat16)]
    layer.lora_b_stacked = [torch.zeros(1, 1, 1, 4, dtype=torch.bfloat16)]

    lora_a = torch.tensor(
        [-0.033203125, 0.10205078125, 0.0517578125, -0.08544921875],
        dtype=torch.bfloat16,
    ).view(4, 1)
    lora_b = torch.tensor(
        [0.0286865234375, -0.2001953125, -0.087890625, -0.006195068359375],
        dtype=torch.bfloat16,
    ).view(1, 4)
    layer.set_lora(index=0, lora_a=lora_a, lora_b=lora_b)
    manager._lora_modules = {"transformer.proj": layer}
    manager._active_composition = (WeightedLoRA(request=_dummy_lora_request(1), scale=1.0),)

    original_weight = layer.base_layer.weight.detach().clone()
    delta = lora_b.float() @ lora_a.float()
    expected = (original_weight.float() + delta).to(torch.bfloat16)
    double_rounded = original_weight + delta.to(torch.bfloat16)
    assert not torch.equal(double_rounded, expected)

    manager._fuse_active_composition()

    torch.testing.assert_close(layer.base_layer.weight, expected, rtol=0, atol=0)


def test_prefused_only_restores_dense_layers_and_releases_runtime_slots(monkeypatch):
    pipeline = _DummyPipeline()
    base_layer = pipeline.transformer.foo
    wrapper = _DummyBaseLayerWithLoRA(base_layer)
    prefused = (WeightedLoRA(request=_dummy_lora_request(1)),)

    def _load_prefused(manager, _composition):
        pipeline.transformer.foo = wrapper
        manager._lora_modules = {"transformer.foo": wrapper}
        manager._max_lora_rank = 8

    monkeypatch.setattr(DiffusionLoRAManager, "_load_startup_composition", _load_prefused)
    monkeypatch.setattr(
        DiffusionLoRAManager,
        "_activate_composition",
        lambda manager, composition: setattr(manager, "_active_composition", composition),
    )
    monkeypatch.setattr(
        DiffusionLoRAManager,
        "_fuse_active_composition",
        lambda manager: setattr(manager, "_active_composition", ()),
    )
    monkeypatch.setattr(DiffusionLoRAManager, "_discard_startup_adapter", lambda *_args: None)

    manager = DiffusionLoRAManager(
        pipeline=pipeline,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        prefused_loras=prefused,
    )

    assert pipeline.transformer.foo is base_layer
    assert manager._lora_modules == {}
    assert manager._max_lora_rank == 0


def _dummy_lora_request(adapter_id: int, path: str | None = None) -> LoRARequest:
    return LoRARequest(
        lora_name=f"adapter_{adapter_id}",
        lora_int_id=adapter_id,
        lora_path=path or f"/tmp/adapter_{adapter_id}",
    )


def _activate_single_composition(manager: DiffusionLoRAManager, adapter_id: int, scale: float) -> None:
    manager._activate_composition((WeightedLoRA(request=_dummy_lora_request(adapter_id), scale=scale),))


def test_lora_manager_request_activation_never_loads_adapter():
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    assert manager._loader is None

    with pytest.raises(ValueError, match="not registered.*--dynamic-lora"):
        manager.set_active_adapter(_dummy_lora_request(1), lora_scale=1.0)


def test_lora_manager_rejects_registry_larger_than_startup_capacity() -> None:
    with pytest.raises(ValueError, match="registry exceeds max_cpu_loras"):
        DiffusionLoRAManager(
            pipeline=torch.nn.Module(),
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            max_registered_adapters=1,
            dynamic_loras=(_dummy_lora_request(1), _dummy_lora_request(2)),
        )


def test_lora_manager_rejects_nonpositive_registry_capacity() -> None:
    with pytest.raises(ValueError, match="must be at least 1"):
        DiffusionLoRAManager(
            pipeline=torch.nn.Module(),
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            max_registered_adapters=0,
        )


def test_lora_manager_validates_adapter_path_before_active_fast_path() -> None:
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    existing = _dummy_lora_request(7, "/tmp/a")
    manager._adapter_requests[7] = existing
    manager._active_composition = (WeightedLoRA(request=existing, scale=1.0),)

    with pytest.raises(ValueError, match="already registered from '/tmp/a'.*not '/tmp/b'"):
        manager.set_active_adapter(_dummy_lora_request(7, "/tmp/b"), lora_scale=1.0)


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

    manager._registered_adapters = {
        adapter_id: _loaded_lora(lora_model),
    }
    manager._lora_modules = {"transformer.foo": lora_model.transformer.foo}
    manager._max_lora_rank = manager._get_smallest_valid_max_rank(rank)

    # After the first scale, all B values should go from 1 -> scale_1
    manager.set_active_adapter(req1, lora_scale=scale_1)
    assert len(lora_model.transformer.foo.set_calls) == 1
    lora_a, lora_b = lora_model.transformer.foo.set_calls[0]
    assert torch.all(lora_a == 1)
    assert torch.all(lora_b == scale_1)

    # After the second scale, all B values should go from 1 -> scale_2
    manager.set_active_adapter(req1, lora_scale=scale_2)
    assert len(lora_model.transformer.foo.set_calls) == 2

    lora_a, lora_b = lora_model.transformer.foo.set_calls[1]
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

    manager._registered_adapters = {
        adapter_id: _loaded_lora(lora_model),
    }
    manager._lora_modules = {"transformer.foo": lora_model.transformer.foo}
    manager._max_lora_rank = manager._get_smallest_valid_max_rank(rank)

    # Activate adapter with initial scale
    manager.set_active_adapter(req1, lora_scale=initial_scale)
    assert lora_model.transformer.foo.create_calls == 0
    assert len(lora_model.transformer.foo.set_calls) == 1
    lora_a, lora_b = lora_model.transformer.foo.set_calls[0]
    assert torch.all(lora_a == 1)
    assert torch.all(lora_b == initial_scale)

    # Increase the rank; this resets the buffers, so the adapter is activated again
    expanded_rank = next(valid_rank for valid_rank in manager._VALID_MAX_RANKS if valid_rank > manager._max_lora_rank)
    manager._frozen = False
    manager._ensure_max_lora_rank(expanded_rank)

    # Ensure we actually took the rank expansion path, which recreates
    # and sets the weight buffets, but that the scale didn't change
    assert lora_model.transformer.foo.create_calls == 1
    assert len(lora_model.transformer.foo.set_calls) == 2
    lora_a, lora_b = lora_model.transformer.foo.set_calls[1]
    assert torch.all(lora_a == 1)
    assert torch.all(lora_b == initial_scale)


def test_lora_manager_uses_valid_max_rank():
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

    manager._frozen = False
    manager._replace_layers_with_lora(type("PH", (), {"r": unsupported_max_rank})())
    assert manager._max_lora_rank == supported_max_rank


@pytest.mark.parametrize("rank", [-1, 0, DiffusionLoRAManager._VALID_MAX_RANKS[-1] + 1])
def test_lora_manager_max_rank_validation(rank):
    """Check that invalid max ranks are handled correctly."""
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    manager._frozen = False
    with pytest.raises(ValueError):
        manager._replace_layers_with_lora(type("PH", (), {"r": rank})())


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
        max_registered_adapters=1,
    )

    peft_helper = type("_PH", (), {"r": 1})()
    manager._frozen = False
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
        max_registered_adapters=1,
    )

    peft_helper = type("_PH", (), {"r": 1})()
    manager._frozen = False
    manager._replace_layers_with_lora(peft_helper)

    assert "down_block.proj" in replace_calls
    assert "unet.down_block.proj" in manager._lora_modules
    # Verify the module was actually replaced in the tree (not just recorded)
    assert isinstance(pipeline.unet.down_block.proj, _DummyBaseLayerWithLoRA)
