# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.lora.lora_model import LoRAModel
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
    base_layer: torch.nn.Module

    def __init__(self, n_slices: int, output_slices: tuple[int, ...], tp_size: int = 1):
        self.n_slices = n_slices
        self.output_slices = output_slices
        self.tp_size = tp_size
        self.set_calls: list[
            tuple[list[torch.Tensor | None] | torch.Tensor, list[torch.Tensor | None] | torch.Tensor]
        ] = []
        self.reset_calls: int = 0
        self.suspend_calls: int = 0
        self.resume_calls: int = 0
        self.active_slices: tuple[bool, ...] = ()
        self.suspended_slices: tuple[bool, ...] | None = None

    def set_lora(self, index: int, lora_a, lora_b):
        assert index == 0
        self.set_calls.append((lora_a, lora_b))
        if isinstance(lora_b, list):
            self.active_slices = tuple(b is not None for b in lora_b)
        else:
            self.active_slices = (True,)
        self.suspended_slices = None

    def reset_lora(self, index: int):
        assert index == 0
        self.reset_calls += 1
        self.active_slices = ()
        self.suspended_slices = None

    def suspend_lora(self) -> None:
        if self.suspended_slices is not None:
            return
        self.suspended_slices = self.active_slices
        self.active_slices = (False,) * len(self.active_slices)
        self.suspend_calls += 1

    def resume_lora(self) -> None:
        if self.suspended_slices is not None:
            self.active_slices = self.suspended_slices
            self.suspended_slices = None
        self.resume_calls += 1


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


def test_lora_manager_keeps_dlo_lora_buffers_on_compute_device(monkeypatch):
    import vllm_omni.diffusion.lora.manager as manager_mod

    class _DLOLoRALayer(DummyBaseLayerWithLoRA):
        lora_buffer_device = None

        def _set_diffusion_lora_buffer_device(self, device):
            self.lora_buffer_device = device

    monkeypatch.setattr(manager_mod, "BaseLayerWithLoRA", _DLOLoRALayer)
    monkeypatch.setattr(
        manager_mod,
        "from_layer_diffusion",
        lambda *, layer, **_kwargs: _DLOLoRALayer(layer),
    )

    pipeline = torch.nn.Module()
    pipeline.od_config = SimpleNamespace(enable_distributed_layerwise_offload=True)
    pipeline.transformer = torch.nn.Module()
    pipeline.transformer.foo = _FakeLinearBase()
    manager = DiffusionLoRAManager(
        pipeline=pipeline,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    manager._replace_layers_with_lora(type("_PH", (), {"r": 1})())

    assert pipeline.transformer.foo.lora_buffer_device == torch.device("cpu")


@pytest.mark.parametrize(
    "target_modules",
    [
        ["experts.0.gate_proj"],
        r".*\.experts\.\d+\.gate_proj$",
    ],
)
def test_lora_manager_matches_scoped_moe_targets(monkeypatch, target_modules):
    import vllm.model_executor.layers.fused_moe as upstream_moe

    import vllm_omni.diffusion.lora.manager as manager_mod

    class _TargetMoERunner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.moe_config = SimpleNamespace(
                is_act_and_mul=True,
                num_experts=2,
            )
            self.global_num_experts = 2

    monkeypatch.setattr(upstream_moe, "MoERunner", _TargetMoERunner)
    monkeypatch.setattr(manager_mod, "BaseLayerWithLoRA", _DummyBaseLayerWithLoRA)
    monkeypatch.setattr(
        manager_mod,
        "from_layer_diffusion",
        lambda *, layer, **_kwargs: _DummyBaseLayerWithLoRA(layer),
    )
    monkeypatch.setattr(
        manager_mod,
        "replace_submodule",
        lambda root, name, sub: fake_replace_submodule(root, name, sub),
    )

    pipeline = torch.nn.Module()
    pipeline.transformer = torch.nn.Module()
    pipeline.transformer.experts = _TargetMoERunner()
    manager = DiffusionLoRAManager(
        pipeline=pipeline,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    peft_helper = type(
        "_PH",
        (),
        {"r": 1, "target_modules": target_modules},
    )()
    manager._replace_layers_with_lora(peft_helper)

    assert "transformer.experts" in manager._lora_modules
    assert isinstance(pipeline.transformer.experts, _DummyBaseLayerWithLoRA)


def test_moe_punica_initializes_public_mapping_metadata(monkeypatch):
    import vllm.lora.punica_wrapper as punica_module

    class _FakePunica:
        def __init__(self):
            self.update_calls = []

        def update_metadata(self, mapping, lora_index_to_id, max_loras, vocab_size):
            self.update_calls.append((mapping, lora_index_to_id, max_loras, vocab_size))

    punica = _FakePunica()
    monkeypatch.setattr(punica_module, "get_punica_wrapper", lambda *args, **kwargs: punica)

    manager = object.__new__(DiffusionLoRAManager)
    manager._moe_punica_wrapper = None
    manager._max_lora_rank = 8
    manager.max_cached_adapters = 1
    manager.dtype = torch.bfloat16
    manager.device = torch.device("cpu")
    manager._max_num_batched_tokens = 8

    assert manager._get_moe_punica_wrapper() is punica
    assert len(punica.update_calls) == 1
    mapping, lora_index_to_id, max_loras, vocab_size = punica.update_calls[0]
    assert mapping.index_mapping == (1,) * 8
    assert mapping.prompt_mapping == (1,)
    assert lora_index_to_id == [1]
    assert max_loras == 1
    assert vocab_size == 0


def test_moe_lora_token_capacity_is_validated_before_forward(monkeypatch):
    import vllm_omni.diffusion.lora.utils as utils_mod

    class _Runner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_forward_pre_hook(lambda *_args: None)

        def forward(self, hidden_states):
            return hidden_states

    class _Wrapper(torch.nn.Module):
        def __init__(self, base_layer):
            super().__init__()
            self.base_layer = base_layer
            self.adapter_enabled = torch.ones(1, dtype=torch.int)

        def create_lora_weights(self, *_args, **_kwargs):
            pass

        def set_lora(self, *_args, **_kwargs):
            pass

        def reset_lora(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(utils_mod, "MoERunner", _Runner)
    monkeypatch.setattr(utils_mod, "_select_moe_lora_wrapper_cls", lambda: _Wrapper)

    wrapper = utils_mod.from_layer_diffusion(
        _Runner(),
        max_loras=1,
        lora_config=SimpleNamespace(),  # type: ignore[arg-type]
        packed_modules_list=[],
        max_num_batched_tokens=8,
    )

    assert wrapper(torch.zeros((8, 4))).shape == (8, 4)
    with pytest.raises(ValueError, match="token count 9 exceeds the configured capacity 8"):
        wrapper(torch.zeros((9, 4)))


def test_move_moe_lora_buffers_keeps_all_stacks_and_views_on_device():
    from vllm_omni.diffusion.lora.utils import _move_moe_lora_buffers

    wrapper = SimpleNamespace(
        max_loras=1,
        local_num_experts=2,
        enable_moe_shared_loras=False,
        _w13_slices=2,
        w13_lora_a_stacked=(torch.ones((1, 2, 3, 4)), torch.ones((1, 2, 3, 4))),
        w13_lora_b_stacked=(torch.ones((1, 2, 5, 3)), torch.ones((1, 2, 5, 3))),
        w2_lora_a_stacked=(torch.ones((1, 2, 3, 5)),),
        w2_lora_b_stacked=(torch.ones((1, 2, 4, 3)),),
        adapter_enabled=torch.ones(2, dtype=torch.int),
    )

    _move_moe_lora_buffers(wrapper, torch.device("meta"))  # type: ignore[arg-type]

    for stacks in (
        wrapper.w13_lora_a_stacked,
        wrapper.w13_lora_b_stacked,
        wrapper.w2_lora_a_stacked,
        wrapper.w2_lora_b_stacked,
    ):
        assert all(tensor.device.type == "meta" for tensor in stacks)
    assert wrapper.adapter_enabled.device.type == "meta"
    assert len(wrapper.lora_a_stacked) == len(wrapper.lora_b_stacked) == 6
    assert [tensor.shape for tensor in wrapper.lora_a_stacked] == [
        (3, 4),
        (3, 5),
        (3, 4),
        (3, 4),
        (3, 5),
        (3, 4),
    ]


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

    manager._activate_adapter(7, 0.5)

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


@pytest.mark.parametrize("name_prefix", ["transformer.blocks.0.attn.", "blocks.0.attn.", ""])
@pytest.mark.parametrize("target_modules", [("to_q",), ("to_q", "to_v"), ("to_q", "to_k", "to_v")])
def test_lora_manager_activates_packed_lora_from_sublayers(name_prefix, target_modules):
    pipeline = torch.nn.Module()
    bound_names = None

    def validate_binding(*, lora_model, bound_lora_names):
        nonlocal bound_names
        assert lora_model.id == 1
        bound_names = bound_lora_names

    pipeline._validate_diffusion_lora_binding = validate_binding
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
        if name not in target_modules:
            continue
        loras[f"{name_prefix}{name}"] = LoRALayerWeights(
            module_name=f"{name_prefix}{name}",
            rank=rank,
            lora_alpha=rank,
            lora_a=torch.ones((rank, 4)) * (1 if name == "to_q" else 2),
            lora_b=torch.ones((out_dim, rank)) * (3 if name == "to_q" else 4),
        )

    manager._registered_adapters = {
        1: type("LM", (), {"id": 1, "loras": loras, "get_lora": lambda self, k: self.loras.get(k)})()
    }

    manager._activate_adapter(1, scale=2.0)

    assert packed_layer.reset_calls == 0
    assert len(packed_layer.set_calls) == 1
    lora_a_list, lora_b_list = packed_layer.set_calls[0]
    assert isinstance(lora_a_list, list)
    assert isinstance(lora_b_list, list)
    assert len(lora_a_list) == 3
    assert len(lora_b_list) == 3
    # Scale should apply to B only.
    for index, name in enumerate(("to_q", "to_k", "to_v")):
        if name in target_modules:
            weights = loras[f"{name_prefix}{name}"]
            assert torch.equal(lora_a_list[index], weights.lora_a)
            assert torch.equal(lora_b_list[index], weights.lora_b * 2.0)
        else:
            assert lora_a_list[index] is None
            assert lora_b_list[index] is None
    assert manager._active_adapter_id == 1
    assert bound_names == frozenset(loras)


def test_lora_manager_rolls_back_all_layers_when_activation_fails():
    class _FailingLoRALayer(_DummyLoRALayer):
        fail = False

        def set_lora(self, index: int, lora_a, lora_b):
            if self.fail:
                raise RuntimeError("bind failed")
            super().set_lora(index, lora_a, lora_b)

    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_cached_adapters=2,
    )
    first = _DummyLoRALayer(n_slices=1, output_slices=(2,))
    second = _FailingLoRALayer(n_slices=1, output_slices=(2,))
    manager._lora_modules = {
        "transformer.first": first,
        "transformer.second": second,
    }

    def adapter(adapter_id: int):
        loras = {
            name: LoRALayerWeights(
                module_name=name,
                rank=2,
                lora_alpha=2,
                lora_a=torch.full((2, 2), float(adapter_id)),
                lora_b=torch.full((2, 2), float(adapter_id)),
            )
            for name in manager._lora_modules
        }
        return type(
            "LM",
            (),
            {
                "id": adapter_id,
                "loras": loras,
                "get_lora": lambda self, key: self.loras.get(key),
            },
        )()

    manager._registered_adapters = {1: adapter(1), 2: adapter(2)}
    manager._activate_adapter(1, scale=1.0)
    first_calls_after_success = len(first.set_calls)

    second.fail = True
    with pytest.raises(RuntimeError, match="bind failed"):
        manager._activate_adapter(2, scale=1.0)

    assert manager._active_adapter_id is None
    assert first.reset_calls == 1
    assert second.reset_calls == 1

    # The failed activation must not leave the old adapter eligible for the
    # fast path: reactivating it binds every layer again.
    second.fail = False
    manager._activate_adapter(1, scale=1.0)
    assert manager._active_adapter_id == 1
    assert len(first.set_calls) == first_calls_after_success + 2


def test_lora_manager_rejects_adapter_that_binds_no_layer():
    """An adapter whose target modules match nothing must fail, not silently no-op."""
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_cached_adapters=1,
    )
    layer = _DummyLoRALayer(n_slices=1, output_slices=(2,))
    manager._lora_modules = {"transformer.blocks.0.attn.to_q": layer}
    manager._expected_lora_modules = {"to_q"}

    # Adapter-side name the engine does not expose, e.g. a diffusers-style
    # checkpoint against a differently named engine layout.
    mismatched = LoRALayerWeights(
        module_name="unet.down_blocks.0.attn.to_q",
        rank=2,
        lora_alpha=2,
        lora_a=torch.ones((2, 2)),
        lora_b=torch.ones((2, 2)),
    )
    manager._registered_adapters = {
        3: type(
            "LM",
            (),
            {
                "id": 3,
                "loras": {"unet.down_blocks.0.attn.to_q": mismatched},
                "get_lora": lambda self, key: self.loras.get(key),
            },
        )()
    }

    with pytest.raises(ValueError, match="binding is incomplete") as excinfo:
        manager._activate_adapter(3, scale=1.0)

    # The message must name what was received so the mismatch is diagnosable.
    assert "bound=0/1" in str(excinfo.value)
    assert "unet.down_blocks.0.attn.to_q" in str(excinfo.value)
    assert "expected target modules in ['to_q']" in str(excinfo.value)

    # Nothing was bound and the adapter must not be left marked active.
    assert manager._active_adapter_id is None
    assert layer.set_calls == []
    assert layer.reset_calls >= 1


def test_lora_manager_rejects_empty_adapter():
    """An empty adapter must not be considered successfully bound."""
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_cached_adapters=1,
    )
    layer = _DummyLoRALayer(n_slices=1, output_slices=(2,))
    manager._lora_modules = {"transformer.attn.to_q": layer}
    manager._registered_adapters = {1: LoRAModel(1, rank=2, loras={})}

    with pytest.raises(ValueError, match="bound=0/0"):
        manager._activate_adapter(1, scale=1.0)

    assert manager._active_adapter_id is None
    assert manager._suspended_adapter_id is None
    assert 1 not in manager._adapter_scales
    assert layer.set_calls == []
    assert layer.active_slices == ()
    assert layer.reset_calls >= 1


@pytest.mark.parametrize("suspend_previous", [False, True])
@pytest.mark.parametrize("unbound_name", ["transformer.attn.to_out.0", "transformer.attn.to_qkv"])
def test_lora_manager_rejects_partial_binding_and_rolls_back(unbound_name, suspend_previous):
    """A partial bind must clear uploaded weights and invalidate fast paths."""
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_cached_adapters=2,
    )
    layer = _DummyLoRALayer(n_slices=1, output_slices=(2,))
    packed_layer = _DummyLoRALayer(n_slices=3, output_slices=(2, 1, 1))
    manager._lora_modules = {
        "transformer.attn.to_out": layer,
        "transformer.attn.to_qkv": packed_layer,
    }
    manager._expected_lora_modules = {"to_out", "to_qkv"}

    def weights(name, value):
        return LoRALayerWeights(
            module_name=name,
            rank=2,
            lora_alpha=2,
            lora_a=torch.full((2, 2), value),
            lora_b=torch.full((2, 2), value),
        )

    valid_name = "transformer.attn.to_out"
    previous = LoRAModel(1, rank=2, loras={valid_name: weights(valid_name, 1.0)})
    # The extra weights either use an unmatched checkpoint name or have a B
    # shape that cannot be split across the fused layer's four output rows.
    partial = LoRAModel(
        2,
        rank=2,
        loras={valid_name: weights(valid_name, 2.0), unbound_name: weights(unbound_name, 2.0)},
    )
    manager._registered_adapters = {1: previous, 2: partial}
    manager._activate_adapter(1, scale=0.5)
    if suspend_previous:
        manager._deactivate_all_adapters()

    with pytest.raises(ValueError, match="binding is incomplete") as excinfo:
        manager._activate_adapter(2, scale=1.0)

    assert "LoRA adapter 2" in str(excinfo.value)
    assert unbound_name in str(excinfo.value)
    assert valid_name not in str(excinfo.value).replace(unbound_name, "")
    if unbound_name.endswith("to_qkv"):
        assert "lora_b.shape[0]=2" in str(excinfo.value)
        assert "sum(output_slices)=4" in str(excinfo.value)
        assert "output_slices=(2, 1, 1)" in str(excinfo.value)
    else:
        assert "expected target modules in ['to_out', 'to_qkv']" in str(excinfo.value)
    assert len(layer.set_calls) == 2
    assert layer.reset_calls == 1
    assert layer.active_slices == ()
    assert packed_layer.active_slices == ()
    assert manager._active_adapter_id is None
    assert manager._suspended_adapter_id is None
    assert 2 not in manager._adapter_scales

    manager._activate_adapter(1, scale=0.5)
    assert len(layer.set_calls) == 3
    assert layer.resume_calls == 0
    assert manager._active_adapter_id == 1
    assert torch.equal(layer.set_calls[-1][1], previous.loras[valid_name].lora_b * 0.5)


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
    monkeypatch.setattr(manager, "_activate_adapter", lambda _adapter_id, scale: None)

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
    monkeypatch.setattr(manager, "_activate_adapter", lambda _adapter_id, scale: None)

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
    monkeypatch.setattr(manager, "_activate_adapter", lambda _adapter_id, scale: None)

    manager.set_active_adapter(_dummy_lora_request(1), lora_scale=1.0)
    manager.set_active_adapter(_dummy_lora_request(2), lora_scale=1.0)

    assert manager.pin_adapter(1)
    assert manager.pin_adapter(2)

    manager.max_cached_adapters = 1
    manager._evict_for_new_adapter()

    assert set(manager.list_adapters()) == {1, 2}


def _make_hunyuan_image3_pipeline(num_heads=4, num_kv_heads=2, head_dim=2, hidden_size=8):
    """Use production hooks and the production base-weight splitter without
    allocating the full checkpoint or initializing distributed workers.
    """
    from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import HunyuanImage3Model
    from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import HunyuanImage3Pipeline

    model = HunyuanImage3Model.__new__(HunyuanImage3Model)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        attention_head_dim=head_dim,
        hidden_size=hidden_size,
    )
    pipeline = HunyuanImage3Pipeline.__new__(HunyuanImage3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.model = model
    pipeline.transformer = model
    return pipeline


def _make_hunyuan_image3_qkv_base():
    """An un-initialized QKVParallelLinear instance for isinstance() checks.

    Supply the type and checkpoint head metadata for binding-only tests.
    The PEFT forward test below constructs the real distributed projection.
    """
    from vllm.model_executor.layers.linear import QKVParallelLinear

    layer = object.__new__(QKVParallelLinear)
    layer.total_num_heads = 4
    layer.total_num_kv_heads = 2
    layer.head_size = layer.v_head_size = 2
    return layer


def test_lora_manager_activates_hunyuan_image3_fused_qkv_lora():
    """A fused HI3 ``qkv_proj`` LoRA-B (GQA-interleaved in the adapter) is
    de-interleaved to the block [Q; K; V] layout the QKV output slices expect,
    with the namespace alias resolving the PEFT adapter tensors."""
    manager = DiffusionLoRAManager(
        pipeline=_make_hunyuan_image3_pipeline(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_cached_adapters=1,
    )

    wrapper = "transformer.layers.0.self_attn.qkv_proj"
    packed_layer = _DummyLoRALayer(
        n_slices=3,
        output_slices=(8, 4, 4),
        tp_size=1,
    )
    # Use QKV type identity and checkpoint head metadata for the manager's
    # de-interleave branch; the forward test below constructs real layers.
    packed_layer.base_layer = _make_hunyuan_image3_qkv_base()
    manager._lora_modules = {wrapper: packed_layer}

    rank = 3
    # GQA-interleaved fused QKV rows: [Q-group0, K0, V0, Q-group1, K1, V1].
    interleaved = torch.arange(16, dtype=torch.bfloat16).unsqueeze(1).repeat(1, rank)
    lora = LoRALayerWeights(
        module_name="model.layers.0.self_attn.qkv_proj",
        rank=rank,
        lora_alpha=rank,
        lora_a=torch.ones((rank, 8)),
        lora_b=interleaved,
    )
    manager._registered_adapters = {
        11: type(
            "LM",
            (),
            {
                "id": 11,
                "loras": {"model.layers.0.self_attn.qkv_proj": lora},
                "get_lora": lambda self, k: self.loras.get(k),
            },
        )()
    }

    manager._activate_adapter(11, 0.5)

    assert packed_layer.reset_calls == 0
    assert len(packed_layer.set_calls) == 1
    lora_a_list, lora_b_list = packed_layer.set_calls[0]
    assert len(lora_a_list) == 3 and len(lora_b_list) == 3
    # De-interleaved full [Q; K; V] slices, scaled by the lora scale.
    expected_q = torch.cat([interleaved[0:4], interleaved[8:12]], dim=0) * 0.5
    expected_k = torch.cat([interleaved[4:6], interleaved[12:14]], dim=0) * 0.5
    expected_v = torch.cat([interleaved[6:8], interleaved[14:16]], dim=0) * 0.5
    assert torch.equal(lora_b_list[0], expected_q)
    assert torch.equal(lora_b_list[1], expected_k)
    assert torch.equal(lora_b_list[2], expected_v)


def test_lora_manager_rejects_hunyuan_image3_qkv_lora_with_unexpected_rows():
    """If the fused HI3 QKV layout cannot be established (wrong row count), the
    manager must fail closed instead of applying the LoRA to wrong slices."""
    manager = DiffusionLoRAManager(
        pipeline=_make_hunyuan_image3_pipeline(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_cached_adapters=1,
    )

    wrapper = "transformer.layers.0.self_attn.qkv_proj"
    packed_layer = _DummyLoRALayer(n_slices=3, output_slices=(8, 4, 4))
    packed_layer.base_layer = _make_hunyuan_image3_qkv_base()
    manager._lora_modules = {wrapper: packed_layer}

    rank = 3
    # Row count (12) does not match the HI3 QKV layout (16): fail closed.
    lora = LoRALayerWeights(
        module_name="model.layers.0.self_attn.qkv_proj",
        rank=rank,
        lora_alpha=rank,
        lora_a=torch.ones((rank, 8)),
        lora_b=torch.arange(12 * rank, dtype=torch.bfloat16).view(-1, rank),
    )
    manager._registered_adapters = {
        12: type(
            "LM",
            (),
            {
                "id": 12,
                "loras": {"model.layers.0.self_attn.qkv_proj": lora},
                "get_lora": lambda self, k: self.loras.get(k),
            },
        )()
    }

    with pytest.raises(ValueError, match="cannot establish HunyuanImage-3 fused-QKV layout"):
        manager._activate_adapter(12, 0.5)

    assert packed_layer.set_calls == []
    assert packed_layer.reset_calls >= 1
    assert manager._active_adapter_id is None


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

    def _fake_load(_req: LoRARequest):
        peft_helper = type("PH", (), {"r": rank})()
        return lora_model, peft_helper

    monkeypatch.setattr(manager, "_load_adapter", _fake_load)
    manager._registered_adapters = {
        adapter_id: lora_model,
    }
    manager._lora_modules = {"transformer.foo": lora_model.transformer.foo}

    # Activate adapter with initial scale
    manager.set_active_adapter(req1, lora_scale=initial_scale)
    assert lora_model.transformer.foo.create_calls == 0
    assert len(lora_model.transformer.foo.set_calls) == 1
    lora_a, lora_b = lora_model.transformer.foo.set_calls[0]
    assert torch.all(lora_a == 1)
    assert torch.all(lora_b == initial_scale)

    # Increase the rank; this resets the buffers, so the adapter is activated again.
    manager._ensure_max_lora_rank(8)

    # Ensure we actually took the rank expansion path, which recreates
    # and sets the weight buffers, but that the scale didn't change.
    assert lora_model.transformer.foo.create_calls == 1
    assert len(lora_model.transformer.foo.set_calls) == 2
    lora_a, lora_b = lora_model.transformer.foo.set_calls[1]
    assert torch.all(lora_a == 1)
    assert torch.all(lora_b == initial_scale)


def test_rank_growth_invalidates_moe_punica_before_republish(monkeypatch):
    adapter_id = 7
    lora_model = _DummyLM(rank=8)
    manager = DiffusionLoRAManager(
        pipeline=_DummyPipeline(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    manager._max_lora_rank = 8
    manager._lora_modules = {"transformer.foo": lora_model.transformer.foo}
    manager._registered_adapters = {adapter_id: lora_model}
    manager._active_adapter_id = adapter_id
    manager._adapter_scales = {adapter_id: 0.5}
    manager._moe_punica_wrapper = object()

    published_cache_values = []

    def _publish_with_current_rank():
        published_cache_values.append(manager._moe_punica_wrapper)
        manager._moe_punica_wrapper = SimpleNamespace(max_lora_rank=manager._max_lora_rank)

    monkeypatch.setattr(manager, "_bind_adapter_weights", lambda *_args: None)
    monkeypatch.setattr(manager, "_publish_moe_lora_context", _publish_with_current_rank)

    manager._ensure_max_lora_rank(128)

    assert published_cache_values == [None]
    assert manager._moe_punica_wrapper.max_lora_rank == 128


def test_scale_keys_are_rounded():
    """Ensure that added adapter scales are rounded to avoid lookup
    issues due to precision differences, e.g., computed scales.
    """
    manager = DiffusionLoRAManager(
        pipeline=_DummyPipeline(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    adapter_id = 1
    # Currently we round keys to 3 decimal places
    manager._update_adapter_scale(adapter_id, 0.0031)
    assert manager._adapter_scales[adapter_id] == 0.003


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

    # Pipeline with a 'bagel' component (no 'transformer') declared via
    # _lora_components, matching how BagelPipeline opts in.
    pipeline = torch.nn.Module()
    pipeline._lora_components = ["bagel"]
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


def _suspend_harness(monkeypatch, adapter_id: int = 7, rank: int = 2):
    """Manager wired to a single dummy layer, ready to activate `adapter_id`."""
    import vllm_omni.diffusion.lora.manager as manager_mod

    monkeypatch.setattr(manager_mod, "BaseLayerWithLoRA", _DummyBaseLayerWithLoRA)

    lora_model = _DummyLM(rank=rank)
    manager = DiffusionLoRAManager(
        pipeline=_DummyPipeline(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    monkeypatch.setattr(manager, "_load_adapter", lambda _req: (lora_model, type("PH", (), {"r": rank})()))
    manager._registered_adapters = {adapter_id: lora_model}
    manager._lora_modules = {"transformer.foo": lora_model.transformer.foo}
    return manager, lora_model.transformer.foo


def test_reactivating_same_adapter_resumes_without_rebinding(monkeypatch):
    """activate -> deactivate -> activate at the same scale must re-arm the
    saved mask instead of re-uploading every layer."""
    adapter_id = 7
    req = _dummy_lora_request(adapter_id)
    manager, layer = _suspend_harness(monkeypatch, adapter_id)

    manager.set_active_adapter(req, lora_scale=0.5)
    assert len(layer.set_calls) == 1
    assert layer.active_slices == (True,)

    manager.set_active_adapter(None)
    assert manager._suspended_adapter_id == adapter_id
    assert layer.suspend_calls == 1
    assert layer.active_slices == (False,)
    # Suspending must not tear the upload down.
    assert layer.reset_calls == 0

    manager.set_active_adapter(req, lora_scale=0.5)
    assert layer.resume_calls == 1
    assert len(layer.set_calls) == 1, "resume must not rebind"
    assert layer.active_slices == (True,)
    assert manager._active_adapter_id == adapter_id
    assert manager._suspended_adapter_id is None


def test_suspended_scope_restores_exact_scale_after_another_adapter(monkeypatch):
    manager, layer = _suspend_harness(monkeypatch)
    manager._registered_adapters[8] = manager._registered_adapters[7]
    manager.set_active_adapter(_dummy_lora_request(7), lora_scale=0.123456)
    original_b = layer.set_calls[-1][1].clone()
    with manager.suspended():
        manager.set_active_adapter(_dummy_lora_request(8), lora_scale=0.7)
        with manager.suspended():
            assert manager._active_adapter_id is None
        assert manager._active_adapter_id == 8
    assert manager._active_adapter_id == 7
    torch.testing.assert_close(layer.set_calls[-1][1], original_b, rtol=0, atol=0)


@pytest.mark.parametrize("second_scale", [0.5, 0.25])
def test_scale_or_adapter_change_forces_rebind(monkeypatch, second_scale):
    """A different scale, or a different adapter, must take the full bind
    path rather than re-arming a stale mask."""
    adapter_id = 7
    manager, layer = _suspend_harness(monkeypatch, adapter_id)

    manager.set_active_adapter(_dummy_lora_request(adapter_id), lora_scale=0.5)
    manager.set_active_adapter(None)
    assert manager._suspended_adapter_id == adapter_id

    # Same id at a new scale, or a different id entirely.
    other_id = adapter_id if second_scale != 0.5 else adapter_id + 1
    if other_id != adapter_id:
        manager._registered_adapters[other_id] = manager._registered_adapters[adapter_id]

    manager.set_active_adapter(_dummy_lora_request(other_id), lora_scale=second_scale)
    assert len(layer.set_calls) == 2, "must rebind"
    assert manager._suspended_adapter_id is None
    assert manager._active_adapter_id == other_id


def test_partial_packed_mask_survives_suspend_resume(monkeypatch):
    """A packed layer where only some slices carry LoRA must come back with
    the same mask, not an all-True one."""
    adapter_id = 7
    manager, layer = _suspend_harness(monkeypatch, adapter_id)

    # Emulate a packed bind that left slice 1 empty.
    layer.set_lora(0, [torch.ones(2, 2), None], [torch.ones(2, 2), None])
    manager._active_adapter_id = adapter_id
    manager._update_adapter_scale(adapter_id, 0.5)
    assert layer.active_slices == (True, False)

    manager.set_active_adapter(None)
    assert layer.active_slices == (False, False)

    manager.set_active_adapter(_dummy_lora_request(adapter_id), lora_scale=0.5)
    assert layer.active_slices == (True, False), "mask must survive unchanged"


def test_removing_suspended_adapter_drops_the_upload(monkeypatch):
    """A removed adapter can never be resumed, so its weights must not stay
    in the stacked buffers."""
    adapter_id = 7
    manager, layer = _suspend_harness(monkeypatch, adapter_id)

    manager.set_active_adapter(_dummy_lora_request(adapter_id), lora_scale=0.5)
    manager.set_active_adapter(None)
    assert manager._suspended_adapter_id == adapter_id

    manager.remove_adapter(adapter_id)
    assert manager._suspended_adapter_id is None
    assert layer.reset_calls == 1, "the upload must be torn down"
    assert layer.suspended_slices is None


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_hunyuan_image3_peft_qkv_forward(tmp_path, monkeypatch, tp_size):
    """Disk PEFT -> real manager/hooks -> vLLM sharding -> numerical forward.

    TP4 also covers replicated KV heads. Only distributed rank discovery is
    stubbed: projection construction, adapter loading and all LoRA operations
    execute production code. The oracle indexes checkpoint rows explicitly.
    """
    import json

    from safetensors.torch import save_file
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear

    def cpu_unquantized_gemm(_layer, inputs, weight, bias=None):
        return torch.nn.functional.linear(inputs, weight, bias)

    # This numerical contract intentionally runs on CPU. Accelerator builds
    # select the unquantized GEMM from the process-wide platform, which would
    # otherwise dispatch CPU tensors to the ROCm-only custom operator.
    monkeypatch.setattr(
        linear,
        "dispatch_unquantized_gemm",
        lambda *_args, **_kwargs: cpu_unquantized_gemm,
    )

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    rank = 2
    a = torch.arange(16, dtype=torch.float32).reshape(rank, 8) / 16
    b = torch.arange(32, dtype=torch.float32).reshape(16, rank) / 32
    prefix = "base_model.model.model.layers.0.self_attn.qkv_proj"
    save_file(
        {f"{prefix}.lora_A.weight": a, f"{prefix}.lora_B.weight": b},
        str(adapter_dir / "adapter_model.safetensors"),
    )
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"r": rank, "lora_alpha": rank, "target_modules": ["qkv_proj"]})
    )
    request = LoRARequest(lora_name="hi3", lora_int_id=6411, lora_path=str(adapter_dir))
    x = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 24
    q_rows = [0, 1, 2, 3, 8, 9, 10, 11]
    k_rows = [4, 5, 12, 13]
    v_rows = [6, 7, 14, 15]
    for tp_rank in range(tp_size):
        monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: tp_size)
        monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: tp_rank)
        monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: tp_rank)
        monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: tp_size)
        with set_current_vllm_config(VllmConfig()):
            pipeline = _make_hunyuan_image3_pipeline()
            base = linear.QKVParallelLinear(8, 2, 4, 2, bias=False, params_dtype=torch.float32)
            base.weight.data.zero_()
            block = torch.nn.Module()
            block.self_attn = torch.nn.Module()
            block.self_attn.qkv_proj = base
            pipeline.model.layers = torch.nn.ModuleList([block])
            manager = DiffusionLoRAManager(pipeline, torch.device("cpu"), torch.float32)
            manager.set_active_adapter(request, lora_scale=0.5)
            wrapped = pipeline.model.layers[0].self_attn.qkv_proj
            assert wrapped is not base
            q_size = 8 // tp_size
            kv_size = 4 // min(tp_size, 2)
            kv_rank = tp_rank // max(tp_size // 2, 1)
            rows = (
                q_rows[tp_rank * q_size : (tp_rank + 1) * q_size]
                + k_rows[kv_rank * kv_size : (kv_rank + 1) * kv_size]
                + v_rows[kv_rank * kv_size : (kv_rank + 1) * kv_size]
            )
            expected = (x @ a.T) @ b[rows].T * 0.5
            with torch.inference_mode():
                torch.testing.assert_close(wrapped.apply(x), expected)
                manager.set_active_adapter(None)
                torch.testing.assert_close(wrapped.apply(x), torch.zeros_like(expected))
                manager.set_active_adapter(request, lora_scale=0.5)
                torch.testing.assert_close(wrapped.apply(x), expected)


@pytest.mark.parametrize("projection", ["qkv_proj", "o_proj"])
def test_hunyuan_image3_lora_namespace_is_model_scoped(projection):
    from vllm.lora.lora_model import LoRAModel

    name = f"model.layers.0.self_attn.{projection}"
    weights = LoRALayerWeights(name, 2, 2, torch.ones(2, 8), torch.ones(8, 2))
    adapter = LoRAModel(6411, 2, {name: weights})
    wrapper = f"transformer.layers.0.self_attn.{projection}"
    hi3 = DiffusionLoRAManager(_make_hunyuan_image3_pipeline(), torch.device("cpu"), torch.float32)
    other = DiffusionLoRAManager(_DummyPipeline(), torch.device("cpu"), torch.float32)
    assert hi3._get_lora_weights(adapter, wrapper) is weights
    assert other._get_lora_weights(adapter, wrapper) is None


@pytest.mark.parametrize("head_dim_key", ["head_dim", "attention_head_dim", None])
def test_hunyuan_image3_real_qkv_hook_head_dimension(head_dim_key):
    pipeline = _make_hunyuan_image3_pipeline()
    del pipeline.model.config.attention_head_dim
    if head_dim_key is not None:
        setattr(pipeline.model.config, head_dim_key, 2)
    b = torch.arange(32, dtype=torch.float32).reshape(16, 2)
    expected_rows = [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]
    torch.testing.assert_close(pipeline._deinterleave_fused_qkv_lora_b(b), b[expected_rows])
    assert pipeline._deinterleave_fused_qkv_lora_b(b[:-1]) is None


def _make_lora_model(adapter_id: int, modules: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> LoRAModel:
    """Build a LoRAModel with optimized (scaling-folded) per-module weights."""
    loras = {}
    for name, (lora_a, lora_b) in modules.items():
        rank = lora_a.shape[0]
        assert lora_b.shape[1] == rank
        loras[name] = LoRALayerWeights(
            module_name=name,
            rank=rank,
            lora_alpha=rank,
            lora_a=lora_a,
            lora_b=lora_b,
        ).optimize()
    rank = max((lora.rank for lora in loras.values()), default=1)
    return LoRAModel(lora_model_id=adapter_id, rank=rank, loras=loras)


def test_compose_lora_models_concatenates_rank_dim():
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_cached_adapters=3,
    )
    a1 = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 8
    b1 = torch.arange(12, dtype=torch.float32).reshape(6, 2) / 12
    a2 = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 12
    b2 = torch.arange(18, dtype=torch.float32).reshape(6, 3) / 18
    model1 = _make_lora_model(1, {"transformer.to_q": (a1, b1)})
    model2 = _make_lora_model(2, {"transformer.to_q": (a2, b2)})

    composed, peft_helper = manager._compose_lora_models(99, [(model1, 0.5), (model2, 2.0)])

    assert composed.id == 99
    assert composed.rank == 5
    assert peft_helper.r == 5
    assert peft_helper.target_modules == ["to_q"]
    composed_lora = composed.loras["transformer.to_q"]
    assert composed_lora.rank == 5
    assert composed_lora.scaling == 1.0
    torch.testing.assert_close(composed_lora.lora_a, torch.cat([a1, a2], dim=0))
    torch.testing.assert_close(composed_lora.lora_b, torch.cat([b1 * 0.5, b2 * 2.0], dim=1))

    # The composed delta must equal the manual per-adapter sum.
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 8
    expected = 0.5 * (x @ a1.t()) @ b1.t() + 2.0 * (x @ a2.t()) @ b2.t()
    actual = (x @ composed_lora.lora_a.t()) @ composed_lora.lora_b.t()
    torch.testing.assert_close(actual, expected)


def test_compose_lora_models_partial_coverage():
    """A module covered by only one adapter takes just its contribution."""
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_cached_adapters=3,
    )
    a1 = torch.ones(2, 4)
    b1 = torch.ones(6, 2)
    a2 = torch.ones(3, 4) * 2
    b2 = torch.ones(6, 3) * 3
    a3 = torch.ones(4, 5) * 4
    b3 = torch.ones(7, 4) * 5
    model1 = _make_lora_model(1, {"transformer.to_q": (a1, b1), "transformer.to_k": (a3, b3)})
    model2 = _make_lora_model(2, {"transformer.to_q": (a2, b2)})

    composed, peft_helper = manager._compose_lora_models(100, [(model1, 0.5), (model2, 1.0)])

    assert set(composed.loras) == {"transformer.to_q", "transformer.to_k"}
    torch.testing.assert_close(composed.loras["transformer.to_q"].lora_b, torch.cat([b1 * 0.5, b2], dim=1))
    # to_k is only covered by model1: keep its rank and scaled B unchanged.
    assert composed.loras["transformer.to_k"].rank == 4
    torch.testing.assert_close(composed.loras["transformer.to_k"].lora_a, a3)
    torch.testing.assert_close(composed.loras["transformer.to_k"].lora_b, b3 * 0.5)
    assert peft_helper.r == 5
    assert composed.rank == 5


def test_compose_lora_models_rejects_rank_sum_over_limit():
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_cached_adapters=3,
    )
    max_rank = DiffusionLoRAManager._VALID_MAX_RANKS[-1]
    big_a = torch.ones(max_rank, 2)
    big_b = torch.ones(2, max_rank)
    model1 = _make_lora_model(1, {"transformer.to_q": (big_a, big_b)})
    model2 = _make_lora_model(2, {"transformer.to_q": (torch.ones(1, 2), torch.ones(2, 1))})

    with pytest.raises(ValueError, match="exceeding the maximum supported rank"):
        manager._compose_lora_models(101, [(model1, 1.0), (model2, 1.0)])


def test_compose_lora_models_rejects_mixed_packed_and_fused_layouts():
    from vllm.lora.lora_weights import PackedLoRALayerWeights

    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_cached_adapters=3,
    )
    packed = PackedLoRALayerWeights(
        module_name="transformer.to_qkv",
        rank=2,
        lora_alphas=[2, 2],
        lora_a=[torch.ones(2, 4), torch.ones(2, 4)],
        lora_b=[torch.ones(6, 2), torch.ones(6, 2)],
        scaling=[1.0, 1.0],
    )
    model1 = LoRAModel(lora_model_id=1, rank=2, loras={"transformer.to_qkv": packed})
    model2 = _make_lora_model(2, {"transformer.to_qkv": (torch.ones(2, 4), torch.ones(6, 2))})

    with pytest.raises(ValueError, match="packed and fused"):
        manager._compose_lora_models(102, [(model1, 1.0), (model2, 1.0)])


def test_compose_lora_models_rejects_empty_coverage():
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_cached_adapters=3,
    )
    empty1 = LoRAModel(lora_model_id=1, rank=2, loras={})
    empty2 = LoRAModel(lora_model_id=2, rank=2, loras={})

    with pytest.raises(ValueError, match="none of the adapters cover any module"):
        manager._compose_lora_models(103, [(empty1, 1.0), (empty2, 1.0)])


def _composed_harness(monkeypatch, max_cached_adapters: int = 4):
    """Manager with stubbed load/replace/activate plus two source adapters."""
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_cached_adapters=max_cached_adapters,
    )
    models = {
        1: _make_lora_model(1, {"transformer.foo": (torch.ones(2, 4), torch.ones(4, 2))}),
        2: _make_lora_model(2, {"transformer.foo": (torch.ones(3, 4) * 2, torch.ones(4, 3) * 3)}),
    }
    load_calls: list[int] = []

    def _fake_load(req: LoRARequest):
        load_calls.append(req.lora_int_id)
        return models[req.lora_int_id], type("PH", (), {"r": models[req.lora_int_id].rank})()

    activations: list[tuple[int, float]] = []
    monkeypatch.setattr(manager, "_load_adapter", _fake_load)
    monkeypatch.setattr(manager, "_replace_layers_with_lora", lambda _peft: None)
    monkeypatch.setattr(manager, "_activate_adapter", lambda aid, scale: activations.append((aid, scale)))
    return manager, models, load_calls, activations


def test_set_active_adapters_composes_and_caches(monkeypatch):
    manager, models, load_calls, activations = _composed_harness(monkeypatch)
    req1, req2 = _dummy_lora_request(1), _dummy_lora_request(2)

    manager.set_active_adapters([req1, req2], [0.5, 2.0])

    assert sorted(load_calls) == [1, 2]
    assert len(manager._composed_adapters) == 1
    composed_id = next(iter(manager._composed_adapters.values()))
    assert composed_id in manager._registered_adapters
    assert activations == [(composed_id, 1.0)]

    composed = manager._registered_adapters[composed_id]
    foo = composed.loras["transformer.foo"]
    torch.testing.assert_close(
        foo.lora_a,
        torch.cat([models[1].loras["transformer.foo"].lora_a, models[2].loras["transformer.foo"].lora_a], dim=0),
    )
    torch.testing.assert_close(
        foo.lora_b,
        torch.cat(
            [models[1].loras["transformer.foo"].lora_b * 0.5, models[2].loras["transformer.foo"].lora_b * 2.0],
            dim=1,
        ),
    )

    # Same combo again: composed cache hit, no reload, no re-registration.
    manager.set_active_adapters([req2, req1], [2.0, 0.5])
    assert sorted(load_calls) == [1, 2]
    assert len(manager._composed_adapters) == 1
    assert activations == [(composed_id, 1.0), (composed_id, 1.0)]

    # A different scale combination composes a distinct adapter.
    manager.set_active_adapters([req1, req2], [0.5, 1.0])
    assert len(manager._composed_adapters) == 2
    assert activations[-1][0] != composed_id


def test_set_active_adapters_single_and_empty_fall_back(monkeypatch):
    manager, models, load_calls, activations = _composed_harness(monkeypatch)
    deactivate_calls: list[int] = []
    monkeypatch.setattr(manager, "_deactivate_all_adapters", lambda: deactivate_calls.append(1))
    manager._active_adapter_id = 1

    # None and single-request inputs delegate to the single-adapter path.
    manager.set_active_adapters(None)
    assert deactivate_calls == [1]
    manager._active_adapter_id = 1

    manager.set_active_adapters(_dummy_lora_request(1), 0.5)
    assert activations == [(1, 0.5)]
    assert manager._composed_adapters == {}

    # A single-entry list takes the same path.
    manager.set_active_adapters([_dummy_lora_request(2)], [0.25])
    assert activations == [(1, 0.5), (2, 0.25)]
    assert manager._composed_adapters == {}

    # All-zero scales deactivate instead of composing.
    manager.set_active_adapters([_dummy_lora_request(1), _dummy_lora_request(2)], [0.0, 0.0])
    assert deactivate_calls == [1, 1]
    assert manager._composed_adapters == {}
    assert sorted(load_calls) == [1, 2]


def test_set_active_adapters_composed_eviction_coexists_with_sources(monkeypatch):
    """Composed entries share the adapter LRU: with capacity 3 the two sources
    plus one composed adapter fit; a second combo evicts the LRU (the first
    composed adapter) while both sources stay registered."""
    manager, _, load_calls, activations = _composed_harness(monkeypatch, max_cached_adapters=3)
    req1, req2 = _dummy_lora_request(1), _dummy_lora_request(2)

    manager.set_active_adapters([req1, req2], [1.0, 1.0])
    first_composed_id = activations[-1][0]
    assert set(manager.list_adapters()) == {1, 2, first_composed_id}
    assert len(manager._composed_adapters) == 1

    # A new combo needs a 4th slot: registering it evicts the LRU entry, which
    # is the first composed adapter; its composed-cache key is purged too.
    manager.set_active_adapters([req1, req2], [0.5, 1.0])
    second_composed_id = activations[-1][0]
    assert second_composed_id != first_composed_id
    assert set(manager.list_adapters()) == {1, 2, second_composed_id}
    assert sorted(load_calls) == [1, 2], "sources must stay cached across combos"
    assert all(cid != first_composed_id for cid in manager._composed_adapters.values())

    # Removing a composed adapter also drops its composed-cache key.
    manager.remove_adapter(second_composed_id)
    assert second_composed_id not in manager._registered_adapters
    assert manager._composed_adapters == {}


@pytest.mark.parametrize("scales", [(0.65, 1.0), (0.0, 1.0), (-0.5, 0.7)])
def test_compose_h3_packed_and_separate_projections(scales):
    from vllm.lora.lora_weights import PackedLoRALayerWeights

    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(), device=torch.device("cpu"), dtype=torch.float32, max_cached_adapters=3
    )
    manager._packed_modules_mapping = {"qkv_proj": ["to_q", "to_k", "to_v"]}
    generator = torch.Generator().manual_seed(42)
    packed_weights = {}
    separate_weights = {}
    expected = {}
    for module, slices in (("attn.qkv_proj", 3), ("mlp.fc1", 2)):
        name = f"transformer.blocks.0.{module}"
        a = [torch.randn(2, 4, generator=generator) for _ in range(slices)]
        b = [torch.randn(6, 2, generator=generator) for _ in range(slices)]
        packed_weights[name] = PackedLoRALayerWeights(
            module_name=name, rank=2, lora_alphas=[2] * slices, lora_a=a, lora_b=b, scaling=[1.0] * slices
        )
        a2 = [torch.randn(3, 4, generator=generator) for _ in range(slices)]
        b2 = [torch.randn(6, 3, generator=generator) for _ in range(slices)]
        if module == "attn.qkv_proj":
            for index, suffix in enumerate(("to_q", "to_k", "to_v")):
                sub_name = f"transformer.blocks.0.attn.{suffix}"
                separate_weights[sub_name] = LoRALayerWeights(
                    sub_name, 3, 3, lora_a=a2[index], lora_b=b2[index], scaling=1.0
                )
                expected[sub_name] = scales[0] * b[index] @ a[index] + scales[1] * b2[index] @ a2[index]
        else:
            separate_weights[name] = PackedLoRALayerWeights(name, 3, [3] * slices, a2, b2, scaling=[1.0] * slices)
            expected[name] = [scales[0] * b[i] @ a[i] + scales[1] * b2[i] @ a2[i] for i in range(slices)]
    first = LoRAModel(lora_model_id=1, rank=2, loras=packed_weights)
    second = LoRAModel(lora_model_id=2, rank=3, loras=separate_weights)
    composed, _ = manager._compose_lora_models(101, [(first, scales[0]), (second, scales[1])])
    for name, delta in expected.items():
        weights = composed.loras[name]
        if isinstance(delta, list):
            for slice_a, slice_b, target in zip(weights.lora_a, weights.lora_b, delta):
                torch.testing.assert_close(slice_b @ slice_a, target)
        else:
            assert isinstance(weights.lora_a, torch.Tensor)
            assert isinstance(weights.lora_b, torch.Tensor)
            torch.testing.assert_close(weights.lora_b @ weights.lora_a, delta)
    assert set(first.loras) == set(packed_weights)
    assert first.loras["transformer.blocks.0.attn.qkv_proj"] is packed_weights["transformer.blocks.0.attn.qkv_proj"]
