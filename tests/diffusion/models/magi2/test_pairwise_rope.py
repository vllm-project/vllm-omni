# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import Mock, patch

import pytest
import torch

from tests.diffusion.models.magi2.test_native_preview import _initialize_tiny_model, _tiny_config
from vllm_omni.diffusion.models.magi2 import modeling_magi2 as modeling
from vllm_omni.diffusion.models.magi2.attention import VarlenHandler, apply_rotary_emb
from vllm_omni.diffusion.models.magi2.modeling_magi2 import Magi2PreviewTransformer, Modality

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


def _reference(x, cos, sin, *, interleaved=False):
    """Pre-extraction formula, with its dtype promotion and temporary tables."""
    width = cos.shape[-1] * 2
    if width > x.shape[-1]:
        raise ValueError("RoPE dimension exceeds head dimension")
    x_rot = x[..., :width]
    if interleaved:
        cos = cos.unsqueeze(-2).repeat_interleave(2, dim=-1)
        sin = sin.unsqueeze(-2).repeat_interleave(2, dim=-1)
        first, second = x_rot[..., ::2], x_rot[..., 1::2]
        rotated_half = torch.stack((-second, first), dim=-1).flatten(-2)
    else:
        cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2)
        sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2)
        first, second = x_rot.chunk(2, dim=-1)
        rotated_half = torch.cat((-second, first), dim=-1)
    return torch.cat((x_rot * cos + rotated_half * sin, x[..., width:]), dim=-1)


def _inputs(dtype, table_dtype, width=8, *, empty=False, batched=False, head_dim=11):
    gen = torch.Generator().manual_seed(63)
    length = 0 if empty else 5
    leading = (2, length) if batched else (length,)
    x = torch.randn(*leading, 3, head_dim * 2, generator=gen).to(dtype)[..., ::2]
    cos = torch.randn(*leading, width, generator=gen).to(table_dtype)[..., ::2]
    sin = torch.randn(*leading, width, generator=gen).to(table_dtype)[..., ::2]
    return x, cos, sin


def _check(x, cos, sin, interleaved=False):
    originals = [tensor.clone() for tensor in (x, cos, sin)]
    rng = torch.get_rng_state()
    actual = apply_rotary_emb(x, cos, sin, interleaved=interleaved)
    expected = _reference(x, cos, sin, interleaved=interleaved)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape and actual.is_contiguous()
    for tensor, original in zip((x, cos, sin), originals):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(torch.get_rng_state(), rng)
    return actual


@pytest.mark.cpu
@pytest.mark.parametrize(
    "dtype,table_dtype",
    [
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.float32),
    ],
)
@pytest.mark.parametrize(
    "width,empty,batched", [(8, False, False), (10, False, True), (0, False, False), (8, True, False)]
)
@pytest.mark.parametrize("interleaved", [False, True])
def test_rope_preserves_original_values_and_promotion(dtype, table_dtype, width, empty, batched, interleaved):
    x, cos, sin = _inputs(dtype, table_dtype, width, empty=empty, batched=batched)
    _check(x, cos, sin, interleaved)


@pytest.mark.cpu
def test_bf16_input_does_not_downcast_fp32_trig_or_tail():
    x, cos, sin = _inputs(torch.bfloat16, torch.float32)
    actual = _check(x, cos, sin)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual[..., 8:], x[..., 8:].float(), rtol=0, atol=0)
    assert not torch.equal(actual, _reference(x, cos.bfloat16(), sin.bfloat16()).float())


@pytest.mark.cpu
def test_compact_tables_avoid_the_two_duplicate_concatenations():
    x, cos, sin = _inputs(torch.float32, torch.float32)
    original = torch.cat
    with patch.object(torch, "cat", wraps=original) as optimized:
        apply_rotary_emb(x, cos, sin)
    with patch.object(torch, "cat", wraps=original) as reference:
        _reference(x, cos, sin)
    assert optimized.call_count == 2  # rotate_half and final output, not trig tables
    assert reference.call_count == 4


@pytest.mark.cpu
def test_mixed_tables_and_unusual_broadcast_shapes_keep_reference_semantics():
    x, cos, sin = _inputs(torch.float16, torch.float16)
    _check(x, cos, sin.float())
    _check(x, cos[:1], sin)  # Existing broadcast-compatible but unequal table shapes.
    with pytest.raises(RuntimeError):
        _reference(x, cos, sin[:, :1])
    with pytest.raises(RuntimeError):
        apply_rotary_emb(x, cos, sin[:, :1])
    with pytest.raises(ValueError, match="exceeds head dimension"):
        apply_rotary_emb(x, torch.ones(5, 6), torch.ones(5, 6))


@pytest.mark.cpu
def test_zero_signs_nonfinite_and_integer_negation():
    x = torch.tensor([[[0.0, -0.0, 1.0, -1.0, torch.inf, -torch.inf, torch.nan, 0.0, 2.0]]])
    cos, sin = torch.tensor([[0.0, 1.0, -1.0, torch.inf]]), torch.tensor([[-0.0, 1.0, 0.0, 1.0]])
    actual = _check(x, cos, sin)
    expected = _reference(x, cos, sin)
    finite = torch.isfinite(expected)
    assert torch.equal(torch.signbit(actual[finite]), torch.signbit(expected[finite]))
    _check(torch.tensor([[[1, 2, 3, 4, 5]]], dtype=torch.uint8), torch.ones(1, 2), torch.ones(1, 2))


@pytest.mark.cpu
def test_gradient_parity():
    x, cos, sin = _inputs(torch.float32, torch.float32, batched=True)
    inputs = (x.clone().requires_grad_(), cos.clone().requires_grad_(), sin.clone().requires_grad_())
    actual, expected = apply_rotary_emb(*inputs), _reference(*inputs)
    for grad, ref in zip(
        torch.autograd.grad(actual.square().sum(), inputs), torch.autograd.grad(expected.square().sum(), inputs)
    ):
        torch.testing.assert_close(grad, ref, rtol=1e-5, atol=2e-6)


@pytest.mark.cpu
def test_compile_frontend_accepts_repeated_shapes():
    graphs = []

    def backend(graph, _inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(apply_rotary_emb, backend=backend, fullgraph=True)
    try:
        for batched in (False, True):
            x, cos, sin = _inputs(torch.bfloat16, torch.float32, batched=batched)
            torch.testing.assert_close(compiled(x, cos, sin), _reference(x, cos, sin), rtol=0, atol=0)
        assert graphs  # This exercises Dynamo, not a silently disabled compile call.
    finally:
        torch._dynamo.reset()


@pytest.mark.cpu
def test_native_transformer_parity():
    model = Magi2PreviewTransformer(_tiny_config(num_layers=2)).eval()
    _initialize_tiny_model(model, seed=7)
    gen = torch.Generator().manual_seed(11)
    packed = torch.randn(6, 4, generator=gen)
    coords = torch.ones(6, 9)
    modalities = torch.tensor(
        [Modality.VIDEO, Modality.VIDEO, Modality.AUDIO, Modality.AUDIO, Modality.TEXT, Modality.TEXT]
    )
    cu = torch.tensor([0, 6], dtype=torch.int32)
    with torch.no_grad(), patch.object(modeling, "apply_rotary_emb", _reference):
        expected = model(packed, coords, modalities, VarlenHandler(cu, cu, 6, 6))
    spy = Mock(wraps=apply_rotary_emb)
    with torch.no_grad(), patch.object(modeling, "apply_rotary_emb", spy):
        actual = model(packed, coords, modalities, VarlenHandler(cu, cu, 6, 6))
    assert spy.call_count == 4
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.musa
@pytest.mark.parametrize(
    "dtype,table_dtype",
    [
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.float32),
    ],
)
def test_musa_rope_matches_original_formula(dtype, table_dtype):
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("requires a MUSA device")
    for width, empty, batched, head_dim in (
        (8, False, False, 11),
        (0, False, False, 11),
        (8, True, False, 11),
        (10, False, True, 11),
        (128, False, False, 128),
        (96, False, True, 128),
    ):
        x, cos, sin = (
            tensor.to("musa").repeat_interleave(2, dim=-1)[..., ::2]
            for tensor in _inputs(dtype, table_dtype, width, empty=empty, batched=batched, head_dim=head_dim)
        )
        for interleaved in (False, True):
            _check(x, cos, sin, interleaved)
