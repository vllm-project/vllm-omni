# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MAGI-2 shared experts activate each fc1 projection separately, bit for bit."""

from __future__ import annotations

import pytest
import torch

from tests.diffusion.models.magi2.test_native_packing import _tiny_model
from vllm_omni.diffusion.models.magi2.layers import ModalityDispatcher, swiglu7
from vllm_omni.diffusion.models.magi2.modeling_magi2 import Magi2MultiHeadMoELayer

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


@pytest.fixture(autouse=True)
def _deterministic_inductor(monkeypatch):
    # Reduction configs are otherwise benchmarked per compile, which can change the bits.
    # Dynamo resets ``deterministic`` after every traced frame; the config filter stays on.
    monkeypatch.setattr(torch._inductor.config, "deterministic", True)
    monkeypatch.setattr(torch._inductor.config.test_configs, "force_filter_reduction_configs", True)


GROUP_SIZES = [(7, 4, 3), (1, 0, 0), (0, 0, 1), (0, 0, 0), (5, 0, 2), (0, 6, 0), (13, 1, 9)]


def _concatenated_shared_experts(
    layer: Magi2MultiHeadMoELayer,
    normalized: torch.Tensor,
    dispatcher: ModalityDispatcher,
) -> torch.Tensor:
    """Shared experts with one SwiGLU7 over the concatenated fc1 outputs.

    Both halves reach fc2 dense, as in the per-projection form, so the comparison covers the
    activation split and not how a down projection reads a column slice.
    """
    shared = layer.shared_expert_fc1(normalized)
    modality = layer.modality_specific_shared_expert_fc1(normalized, dispatcher)
    activated = swiglu7(torch.cat((shared, modality), dim=-1))
    shared, modality = activated.split((shared.shape[-1] // 2, modality.shape[-1] // 2), dim=-1)
    return layer.shared_expert_fc2(shared.contiguous()) + layer.modality_specific_shared_expert_fc2(
        modality.contiguous(), dispatcher
    )


def _moe_layer(params_dtype: torch.dtype) -> Magi2MultiHeadMoELayer:
    layer = _tiny_model(params_dtype=params_dtype).block.layers[0].mlp
    assert isinstance(layer, Magi2MultiHeadMoELayer)
    return layer


def _inputs(group_sizes: tuple[int, ...], hidden_size: int, dtype: torch.dtype, seed: int = 3):
    modalities = torch.cat([torch.full((size,), index, dtype=torch.int32) for index, size in enumerate(group_sizes)])
    generator = torch.Generator().manual_seed(seed)
    # Large activations reach both SwiGLU7 clamps.
    normalized = (torch.randn(sum(group_sizes), hidden_size, generator=generator) * 40).to(dtype)
    return normalized, ModalityDispatcher(modalities, len(group_sizes))


@pytest.mark.cpu
@pytest.mark.parametrize("group_sizes", GROUP_SIZES, ids=lambda sizes: "-".join(map(str, sizes)))
# BF16 only: on CPU an FP32 sigmoid over the strided gate columns can take a scalar exp instead of
# the vectorized one, so FP32 results depend on how the activation input is laid out.
@pytest.mark.parametrize("params_dtype", [torch.bfloat16])
@pytest.mark.parametrize("compiling", [False, True], ids=["eager", "native"])
def test_shared_experts_match_the_concatenated_activation_bitwise(monkeypatch, group_sizes, params_dtype, compiling):
    layer = _moe_layer(params_dtype)
    normalized, dispatcher = _inputs(group_sizes, layer.config.hidden_size, params_dtype)
    if compiling:
        # Select the native expression that compiled regions trace.
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    with torch.no_grad():
        expected = _concatenated_shared_experts(layer, normalized, dispatcher)
        actual = layer._shared_experts(normalized, dispatcher)

    assert actual.shape == (sum(group_sizes), layer.config.hidden_size)
    assert actual.dtype == expected.dtype == params_dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.cpu
def test_shared_experts_feed_dense_down_projections(monkeypatch):
    layer = _moe_layer(torch.bfloat16)
    normalized, dispatcher = _inputs((5, 0, 2), layer.config.hidden_size, torch.bfloat16)
    fc2_inputs = []

    def record(module):
        forward = module.forward

        def wrapped(tensor, *args):
            fc2_inputs.append(tensor)
            return forward(tensor, *args)

        monkeypatch.setattr(module, "forward", wrapped)

    record(layer.shared_expert_fc2)
    record(layer.modality_specific_shared_expert_fc2)
    with torch.no_grad():
        layer._shared_experts(normalized, dispatcher)

    assert len(fc2_inputs) == 2
    assert all(tensor.is_contiguous() for tensor in fc2_inputs)


@pytest.mark.cpu
@pytest.mark.parametrize("width", [2, 8, 2560])
@pytest.mark.parametrize("tokens", [0, 1, 37])
def test_swiglu7_of_a_column_concatenation_is_the_concatenation_of_swiglu7(width, tokens):
    generator = torch.Generator().manual_seed(width + tokens)
    left = (torch.randn(tokens, width, generator=generator) * 10).to(torch.bfloat16)
    right = (torch.randn(tokens, 3 * width, generator=generator) * 10).to(torch.bfloat16)

    expected = swiglu7(torch.cat((left, right), dim=-1))
    actual = torch.cat((swiglu7(left), swiglu7(right)), dim=-1)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.musa
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_musa_device_shared_experts_match_the_concatenated_activation_bitwise(compiled):
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("requires a MUSA device")
    layer = _moe_layer(torch.bfloat16).to("musa")
    shared_experts, concatenated = layer._shared_experts, _concatenated_shared_experts
    if compiled:
        options = {"fullgraph": True, "dynamic": False, "options": {"emulate_precision_casts": True}}
        shared_experts, concatenated = torch.compile(shared_experts, **options), torch.compile(concatenated, **options)

    for group_sizes in ((29, 0, 0), (0, 0, 31), (13, 4, 9)):
        normalized, dispatcher = _inputs(group_sizes, layer.config.hidden_size, torch.bfloat16)
        normalized = normalized.to("musa")
        dispatcher = ModalityDispatcher(dispatcher.modality_mapping.to("musa"), dispatcher.num_modalities)
        with torch.no_grad():
            expected = concatenated(layer, normalized, dispatcher)
            actual = shared_experts(normalized, dispatcher)
        assert torch.equal(actual.cpu().view(torch.int16), expected.cpu().view(torch.int16))
    torch._dynamo.reset()


@pytest.mark.musa
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_musa_device_swiglu7_splits_at_production_widths(compiled):
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("requires a MUSA device")
    generator = torch.Generator().manual_seed(17)
    shared = (torch.randn(3702, 2560, generator=generator) * 10).to(torch.bfloat16).to("musa")
    modality = (torch.randn(3702, 2560, generator=generator) * 10).to(torch.bfloat16).to("musa")

    def concatenated(shared, modality):
        return swiglu7(torch.cat((shared, modality), dim=-1)).split((1280, 1280), dim=-1)

    def separate(shared, modality):
        return swiglu7(shared), swiglu7(modality)

    if compiled:
        options = {"fullgraph": True, "dynamic": False, "options": {"emulate_precision_casts": True}}
        concatenated, separate = torch.compile(concatenated, **options), torch.compile(separate, **options)
    with torch.no_grad():
        expected = concatenated(shared, modality)
        actual = separate(shared, modality)
    for actual_part, expected_part in zip(actual, expected, strict=True):
        assert torch.equal(actual_part.cpu().view(torch.int16), expected_part.cpu().view(torch.int16))
    torch._dynamo.reset()
