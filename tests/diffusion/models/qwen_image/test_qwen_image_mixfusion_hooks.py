# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the MixFusion block-forward-hook guard (CPU).

The packed MixFusion path runs blocks on the chunk-stacked layout (several
chunks per request stacked on the batch dimension), where block-hook
semantics are unvalidated: under layerwise offload the block-weight prefetch
step size no longer matches the request count, and HSDP's ``fully_shard``
registers native forward pre-hooks to all-gather weights against a batch
dimension that no longer matches the request count. The packed path must
refuse to run while blocks carry hooks -- the adapter's ``prepare_inputs``
raises before any packing work -- and the driver must fall back to
the serial path, whose per-request loops route every block through the hooked
module call.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context
from vllm_omni.diffusion.hooks import HookRegistry, ModelHook, module_has_active_hooks
from vllm_omni.diffusion.hooks.mixfusion import plan_foreign_hooks
from vllm_omni.diffusion.mixfusion.driver import MixFusionDriver
from vllm_omni.diffusion.models.qwen_image.mixfusion_adapter import QwenImageMixFusionAdapter
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    QwenImageTransformer2DModel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _PlainBlock(nn.Module):
    def forward(self, x):
        return x


def _hooked_block() -> nn.Module:
    """A block with a registered hook, as installed by layerwise offload."""
    block = _PlainBlock()
    HookRegistry.get_or_create(block).register_hook("layerwise_offload", ModelHook())
    return block


def _hsdp_like_block() -> nn.Module:
    """A block with a PyTorch-native forward pre-hook, as registered by
    HSDP's ``fully_shard`` (the unshard/all-gather pre-hook)."""
    block = _PlainBlock()
    block.register_forward_pre_hook(lambda module, args: args)
    return block


def _transformer(blocks: list[nn.Module]) -> QwenImageTransformer2DModel:
    """A transformer with only the mixfusion guards' dependencies.

    Built via ``__new__`` so the vLLM parallel layers are never constructed;
    only non-Module attributes are set, so ``object.__setattr__`` bypasses the
    nn.Module machinery.
    """
    transformer = QwenImageTransformer2DModel.__new__(QwenImageTransformer2DModel)
    object.__setattr__(transformer, "zero_cond_t", False)
    object.__setattr__(transformer, "parallel_config", SimpleNamespace(sequence_parallel_size=1))
    # ``plan_foreign_hooks`` expands ``*`` only over ``nn.ModuleList``.
    object.__setattr__(transformer, "transformer_blocks", nn.ModuleList(blocks))
    return transformer


def _pipeline(transformer):
    """The driver's pipeline surface, without constructing the real pipeline."""
    return SimpleNamespace(
        transformer=transformer,
        # ``interrupt`` is a read-only property on the real pipeline.
        interrupt=False,
        attention_kwargs=None,
        parallel_config=SimpleNamespace(sequence_parallel_size=1),
    )


class _StubTransformer:
    """Transformer stub recording packed-path calls for the driver tests."""

    def __init__(self, *, plan=None, blocks=None):
        self._mixfusion_plan = plan
        self.transformer_blocks = nn.ModuleList(blocks or [])
        self.packed_calls = 0
        # Stamped by ``apply_mixfusion``; the driver refuses the packed
        # call when the registry never installed the hooks.
        self._mixfusion_hooks_applied = True

    def __call__(self, **kwargs):
        self.packed_calls += 1
        return "packed"


class _StubAdapter:
    """Minimal model-side adapter; the transformer stub consumes any payload."""

    def build_packed_call(self, input_batch):
        return {"hidden_states": input_batch.latents}


def _driver(transformer):
    driver = MixFusionDriver(_pipeline(transformer), transformer, _StubAdapter())
    serial_inputs = []
    driver.serial_ragged = lambda batch, states, dense_step: serial_inputs.append(batch) or "serial"
    return driver, serial_inputs


def _packed_input_batch():
    return SimpleNamespace(
        latents=[torch.zeros(1, 8, 16), torch.zeros(1, 8, 16)],
        do_true_cfg=False,
        image_latents=None,
        # The driver aggregates per-request thresholds; lowering the minimum
        # lets this tiny fixture stay packable.
        request_extras=[{"mixfusion_min_chunk_tokens": 1}],
        timesteps=torch.tensor([500.0]),
    )


def _dense_step(batch, **kwargs):
    return "dense"


def test_module_has_active_hooks():
    plain = _PlainBlock()
    registry_only = _PlainBlock()
    HookRegistry.get_or_create(registry_only)  # empty registry must not count

    assert not module_has_active_hooks(plain)
    assert not module_has_active_hooks(registry_only)
    assert module_has_active_hooks(_hooked_block())


def test_plan_foreign_hooks_reports_block_hooks():
    plan = QwenImageTransformer2DModel._mixfusion_plan
    assert plan_foreign_hooks(_transformer([_PlainBlock()]), plan) == []
    assert plan_foreign_hooks(_transformer([_PlainBlock(), _hooked_block()]), plan) == [
        "transformer_blocks.*.img_mod:layerwise_offload"
    ]


def test_plan_foreign_hooks_reports_native_block_hooks():
    # HSDP's fully_shard registers PyTorch-native hooks (invisible to the
    # Omni HookRegistry); the guard must fall back for those too.
    plan = QwenImageTransformer2DModel._mixfusion_plan
    assert plan_foreign_hooks(_transformer([_PlainBlock()]), plan) == []
    assert plan_foreign_hooks(_transformer([_PlainBlock(), _hsdp_like_block()]), plan) == [
        "transformer_blocks.*.img_mod:torch-native"
    ]


def _prepare(transformer, **overrides):
    kwargs = dict(
        hidden_states=[torch.zeros(1, 8, 16)],
        encoder_hidden_states=torch.zeros(1, 4, 8),
        img_shapes=[(16, 2, 2)],
        txt_seq_lens=[4],
    )
    kwargs.update(overrides)
    QwenImageMixFusionAdapter().prepare_inputs(transformer, **kwargs)


def test_packed_prepare_rejects_hooked_blocks():
    with pytest.raises(ValueError, match="unvalidated semantics"):
        _prepare(_transformer([_hooked_block()]))


def test_packed_prepare_rejects_native_hooked_blocks():
    with pytest.raises(ValueError, match="unvalidated semantics"):
        _prepare(_transformer([_hsdp_like_block()]))


def test_packed_prepare_passes_the_hook_guard_without_hooks():
    # Without hooks the guard must not fire; execution continues to the
    # regular input validation (empty hidden states).
    with pytest.raises(ValueError, match="at least one hidden-state tensor"):
        _prepare(_transformer([_PlainBlock()]), hidden_states=[])


def test_driver_falls_back_to_serial_when_blocks_are_hooked():
    transformer = _StubTransformer(
        plan={"transformer_blocks.*": "img_mod"},
        blocks=[_hooked_block()],
    )
    assert plan_foreign_hooks(transformer, transformer._mixfusion_plan) == ["transformer_blocks.*:layerwise_offload"]
    driver, serial_inputs = _driver(transformer)
    input_batch = _packed_input_batch()

    assert driver.run(input_batch, states=[], dense_step=_dense_step) == "serial"
    assert serial_inputs == [input_batch]
    assert transformer.packed_calls == 0


def test_driver_falls_back_to_serial_when_blocks_carry_native_hooks():
    transformer = _StubTransformer(
        plan={"transformer_blocks.*": "img_mod"},
        blocks=[_hsdp_like_block()],
    )
    driver, serial_inputs = _driver(transformer)
    input_batch = _packed_input_batch()

    assert driver.run(input_batch, states=[], dense_step=_dense_step) == "serial"
    assert serial_inputs == [input_batch]
    assert transformer.packed_calls == 0


def test_driver_packs_when_blocks_are_not_hooked():
    transformer = _StubTransformer(plan={"transformer_blocks.*": "img_mod"}, blocks=[_PlainBlock()])
    driver, serial_inputs = _driver(transformer)
    input_batch = _packed_input_batch()

    with override_forward_context(ForwardContext()):
        assert driver.run(input_batch, states=[], dense_step=_dense_step) == "packed"
    assert transformer.packed_calls == 1
    assert serial_inputs == []
