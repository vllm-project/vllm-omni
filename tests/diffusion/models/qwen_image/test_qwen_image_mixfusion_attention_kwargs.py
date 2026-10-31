# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests that ``attention_kwargs`` survive the MixFusion dispatch (CPU).

The packed layout drives blocks through the layout-aware ``block.forward``,
which has no ``attention_kwargs`` channel, so a non-empty ``attention_kwargs``
used to be silently dropped when ``hidden_states`` was a list. The pipeline
must fall back to the serial path (which threads them into every block), and a
direct packed call with ``attention_kwargs`` must fail loudly at the prepare
step instead of ignoring them.

The tests also cover the driver arming: the packed call arms a
``MixFusionStep`` on the forward context, forwards the layout arguments the
root hook needs (``img_shapes`` / ``txt_seq_lens``), and restores the previous
step afterwards.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.forward_context import (
    ForwardContext,
    get_forward_context,
    override_forward_context,
)
from vllm_omni.diffusion.mixfusion.driver import MixFusionDriver
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    QwenImageMixFusionAdapter,
    QwenImageTransformer2DModel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

# A non-empty sentinel: any truthy dict stands in for diffusers-style
# per-attention arguments (e.g. IP-Adapter or LoRA scale).
SENTINEL_KWARGS = {"scale": 0.5}


class _PlainBlock(nn.Module):
    def forward(self, x):
        return x


def _transformer() -> QwenImageTransformer2DModel:
    """A transformer with only the mixfusion guards' dependencies.

    Built via ``__new__`` so the vLLM parallel layers are never constructed;
    only non-Module attributes are set, so ``object.__setattr__`` bypasses the
    nn.Module machinery.
    """
    transformer = QwenImageTransformer2DModel.__new__(QwenImageTransformer2DModel)
    object.__setattr__(transformer, "zero_cond_t", False)
    object.__setattr__(transformer, "parallel_config", SimpleNamespace(sequence_parallel_size=1))
    object.__setattr__(transformer, "transformer_blocks", [_PlainBlock()])
    return transformer


class _StubTransformer:
    """Transformer stub recording packed-path calls for the driver tests."""

    def __init__(self):
        self.packed_calls = 0
        self.calls = []
        self.armed_at_call = None
        self._mixfusion_hooks_applied = True

    def __call__(self, **kwargs):
        self.packed_calls += 1
        self.calls.append(kwargs)
        # The driver must arm the step for exactly the transformer call.
        self.armed_at_call = get_forward_context().mixfusion_step is not None
        return "packed"


def _pipeline(transformer):
    """The driver's pipeline surface, without constructing the real pipeline."""
    return SimpleNamespace(
        transformer=transformer,
        # ``interrupt`` is a read-only property on the real pipeline.
        interrupt=False,
        attention_kwargs=None,
        parallel_config=SimpleNamespace(sequence_parallel_size=1),
    )


def _packed_input_batch():
    return SimpleNamespace(
        latents=[torch.zeros(1, 8, 16), torch.zeros(1, 8, 16)],
        do_true_cfg=False,
        image_latents=None,
        # The driver aggregates per-request thresholds; lowering the minimum
        # lets this tiny fixture stay packable.
        request_extras=[{"mixfusion_min_chunk_tokens": 1}],
        timesteps=torch.tensor([500.0]),
        guidance=None,
        prompt_embeds=torch.zeros(2, 4, 8),
        prompt_embeds_mask=None,
        img_shapes=[(16, 2, 2), (16, 2, 2)],
        txt_seq_lens=[4, 4],
    )


def _serial_fallback_driver(transformer):
    driver = MixFusionDriver(_pipeline(transformer), transformer, QwenImageMixFusionAdapter())
    serial_inputs = []
    driver.serial_ragged = lambda batch, states, dense_step: serial_inputs.append(batch) or "serial"
    return driver, serial_inputs


def _dense_step(batch, **kwargs):
    return "dense"


def test_prepare_rejects_attention_kwargs_on_packed_input():
    with pytest.raises(ValueError, match="does not consume attention_kwargs"):
        _transformer().prepare_mixfusion_inputs(
            hidden_states=[torch.zeros(1, 8, 16)],
            encoder_hidden_states=torch.zeros(1, 4, 8),
            img_shapes=[(16, 2, 2)],
            txt_seq_lens=[4],
            attention_kwargs=dict(SENTINEL_KWARGS),
        )


def test_prepare_accepts_empty_attention_kwargs_on_packed_input():
    # An empty dict carries no per-attention arguments, so the packed dispatch
    # must proceed (here: into the regular input validation).
    with pytest.raises(ValueError, match="at least one hidden-state tensor"):
        _transformer().prepare_mixfusion_inputs(
            hidden_states=[],
            encoder_hidden_states=torch.zeros(1, 4, 8),
            img_shapes=[(16, 2, 2)],
            txt_seq_lens=[4],
            attention_kwargs={},
        )


def test_driver_falls_back_to_serial_with_attention_kwargs():
    transformer = _StubTransformer()
    driver, serial_inputs = _serial_fallback_driver(transformer)
    driver.pipeline.attention_kwargs = dict(SENTINEL_KWARGS)
    input_batch = SimpleNamespace(
        latents=[torch.zeros(1, 8, 16)],
        do_true_cfg=False,
        image_latents=None,
        request_extras=None,
        timesteps=torch.tensor([500.0]),
    )

    assert driver.run(input_batch, states=[], dense_step=_dense_step) == "serial"
    assert serial_inputs == [input_batch]
    assert transformer.packed_calls == 0


def test_driver_falls_back_when_hooks_missing():
    transformer = _StubTransformer()
    transformer._mixfusion_hooks_applied = False
    driver, serial_inputs = _serial_fallback_driver(transformer)
    input_batch = _packed_input_batch()

    with override_forward_context(ForwardContext()):
        assert driver.run(input_batch, states=[], dense_step=_dense_step) == "serial"
    assert serial_inputs == [input_batch]
    assert transformer.packed_calls == 0


def test_driver_falls_back_with_foreign_root_hooks():
    transformer = _StubTransformer()
    # A non-MixFusion hook on the root sees the swapped-in packed batch;
    # cache-style hooks would build keys on the wrong layout.
    transformer._hook_registry = SimpleNamespace(_hooks={"cache": object()})
    driver, serial_inputs = _serial_fallback_driver(transformer)
    input_batch = _packed_input_batch()

    assert driver.run(input_batch, states=[], dense_step=_dense_step) == "serial"
    assert serial_inputs == [input_batch]
    assert transformer.packed_calls == 0


def test_driver_arms_step_and_forwards_layout_kwargs():
    transformer = _StubTransformer()
    driver = MixFusionDriver(_pipeline(transformer), transformer, QwenImageMixFusionAdapter())
    input_batch = _packed_input_batch()

    with override_forward_context(ForwardContext()):
        assert driver.run(input_batch, states=[], dense_step=_dense_step) == "packed"
        # Restored by the try/finally so later calls see no armed step.
        assert get_forward_context().mixfusion_step is None

    assert transformer.packed_calls == 1
    assert transformer.armed_at_call is True
    call = transformer.calls[0]
    # The root hook packs the list itself; the adapter forwards the layout
    # arguments it needs instead of a pre-built bundle.
    assert call["hidden_states"] is input_batch.latents
    assert call["img_shapes"] == input_batch.img_shapes
    assert call["txt_seq_lens"] == input_batch.txt_seq_lens
    assert call["return_dict"] is False
