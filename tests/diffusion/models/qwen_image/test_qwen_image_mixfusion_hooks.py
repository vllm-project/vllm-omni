# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the MixFusion block-forward-hook guard (CPU).

``_forward_mixfusion`` invokes ``block.forward_mixfusion`` directly, which
bypasses the HookRegistry wrapper around ``block.forward``; under layerwise
offload only the first block is materialized, so the packed path must refuse
to run while blocks carry hooks, and the pipeline must fall back to the serial
path, whose per-request loops route every block through the hooked module call.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.hooks import HookRegistry, ModelHook, module_has_active_hooks
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import QwenImagePipeline
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


def _transformer(blocks: list[nn.Module]) -> QwenImageTransformer2DModel:
    """A transformer with only the mixfusion guards' dependencies.

    Built via ``__new__`` so the vLLM parallel layers are never constructed;
    only non-Module attributes are set, so ``object.__setattr__`` bypasses the
    nn.Module machinery.
    """
    transformer = QwenImageTransformer2DModel.__new__(QwenImageTransformer2DModel)
    object.__setattr__(transformer, "zero_cond_t", False)
    object.__setattr__(transformer, "parallel_config", SimpleNamespace(sequence_parallel_size=1))
    object.__setattr__(transformer, "transformer_blocks", blocks)
    return transformer


def _pipeline(transformer) -> QwenImagePipeline:
    pipeline = QwenImagePipeline.__new__(QwenImagePipeline)
    object.__setattr__(pipeline, "parallel_config", SimpleNamespace(sequence_parallel_size=1))
    object.__setattr__(pipeline, "transformer", transformer)
    return pipeline


class _StubTransformer:
    """Transformer stub recording packed-path calls for the pipeline tests."""

    def __init__(self, *, hooked: bool):
        self.hooked = hooked
        self.packed_calls = 0

    def has_active_block_hooks(self) -> bool:
        return self.hooked

    def __call__(self, **kwargs):
        self.packed_calls += 1
        return "packed"


def test_module_has_active_hooks():
    plain = _PlainBlock()
    registry_only = _PlainBlock()
    HookRegistry.get_or_create(registry_only)  # empty registry must not count

    assert not module_has_active_hooks(plain)
    assert not module_has_active_hooks(registry_only)
    assert module_has_active_hooks(_hooked_block())


def test_has_active_block_hooks_reflects_block_hooks():
    assert not _transformer([_PlainBlock()]).has_active_block_hooks()
    assert _transformer([_PlainBlock(), _hooked_block()]).has_active_block_hooks()


def test_forward_mixfusion_rejects_hooked_blocks():
    transformer = _transformer([_hooked_block()])

    with pytest.raises(ValueError, match="bypasses block forward hooks"):
        transformer._forward_mixfusion(
            hidden_states=[torch.zeros(1, 8, 16)],
            encoder_hidden_states=torch.zeros(1, 4, 8),
            encoder_hidden_states_mask=None,
            timestep=torch.zeros(1),
            img_shapes=[(16, 2, 2)],
            txt_seq_lens=[4],
            guidance=None,
        )


def test_forward_mixfusion_passes_the_hook_guard_without_hooks():
    # Without hooks the guard must not fire; execution continues to the
    # regular input validation (empty hidden states).
    transformer = _transformer([_PlainBlock()])

    with pytest.raises(ValueError, match="at least one hidden-state tensor"):
        transformer._forward_mixfusion(
            hidden_states=[],
            encoder_hidden_states=torch.zeros(1, 4, 8),
            encoder_hidden_states_mask=None,
            timestep=torch.zeros(1),
            img_shapes=[(16, 2, 2)],
            txt_seq_lens=[4],
            guidance=None,
        )


def test_denoise_step_mixfusion_falls_back_to_serial_when_blocks_are_hooked():
    pipeline = _pipeline(_StubTransformer(hooked=True))
    serial_inputs = []
    object.__setattr__(
        pipeline,
        "_denoise_step_serial_ragged",
        lambda batch: serial_inputs.append(batch) or "serial",
    )
    input_batch = SimpleNamespace(
        latents=[torch.zeros(1, 8, 16)],
        do_true_cfg=False,
        image_latents=None,
    )

    assert pipeline._denoise_step_mixfusion(input_batch) == "serial"
    assert serial_inputs == [input_batch]


def test_denoise_step_mixfusion_packs_when_blocks_are_not_hooked():
    transformer = _StubTransformer(hooked=False)
    pipeline = _pipeline(transformer)
    serial_inputs = []
    object.__setattr__(
        pipeline,
        "_denoise_step_serial_ragged",
        lambda batch: serial_inputs.append(batch) or "serial",
    )
    object.__setattr__(
        pipeline,
        "_qwen_mixfusion_candidate",
        lambda latents, *, min_chunk_tokens, max_chunks: (True, "", 8, 2),
    )
    # ``attention_kwargs`` is a read-only property over ``_attention_kwargs``.
    object.__setattr__(pipeline, "_attention_kwargs", None)
    input_batch = SimpleNamespace(
        latents=[torch.zeros(1, 8, 16), torch.zeros(1, 8, 16)],
        do_true_cfg=False,
        image_latents=None,
        request_extras=None,
        timesteps=torch.tensor([500.0]),
        guidance=None,
        prompt_embeds=torch.zeros(2, 4, 8),
        prompt_embeds_mask=None,
        img_shapes=[(16, 2, 2), (16, 2, 2)],
        txt_seq_lens=[4, 4],
    )

    assert pipeline._denoise_step_mixfusion(input_batch) == "packed"
    assert transformer.packed_calls == 1
    assert serial_inputs == []
