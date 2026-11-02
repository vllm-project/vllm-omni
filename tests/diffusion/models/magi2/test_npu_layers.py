# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MAGI-2-specific Ascend RMSNorm and RoPE layout adapters."""

import pytest
import torch

pytest.importorskip("torch_npu")

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.magi2.common_layers import ModalityDispatcher
from vllm_omni.diffusion.models.magi2.layers.native.normalization import (
    MultiModalityRMSNorm as NativeMultiModalityRMSNorm,
)
from vllm_omni.diffusion.models.magi2.layers.native.rope import apply_rotary_emb as native_apply_rotary_emb
from vllm_omni.diffusion.models.magi2.layers.npu import normalization as npu_normalization
from vllm_omni.diffusion.models.magi2.layers.npu import rope as npu_rope
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


@pytest.fixture
def npu_runtime():
    torch_npu = pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an available NPU")
    return torch_npu


def _norm_pair(dim=64, *, num_modality=1, num_patterns=1, out_dtype=None):
    kwargs = dict(eps=1e-5, num_modality=num_modality, num_patterns=num_patterns, out_dtype=out_dtype)
    reference = NativeMultiModalityRMSNorm(dim, **kwargs)
    adapter = npu_normalization.MultiModalityRMSNorm(dim, **kwargs)
    with torch.no_grad():
        reference.weight.copy_(torch.linspace(-0.4, 0.6, reference.weight.numel()))
        adapter.load_state_dict(reference.state_dict())
    return reference, adapter


@pytest.mark.cpu
def test_npu_rmsnorm_cpu_inputs_match_native():
    reference, adapter = _norm_pair(num_modality=3, out_dtype=torch.float32)
    x = torch.randn(5, 2, 64, generator=torch.Generator().manual_seed(7)).to(torch.bfloat16)
    dispatcher = ModalityDispatcher(torch.tensor([0, 0, 2, 2, 2]), 3)

    with torch.no_grad():
        expected = reference(x, dispatcher)
        actual = adapter(x, dispatcher)

    assert actual.shape == x.shape and actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@hardware_test(res={"npu": "A5"}, num_cards=1)
@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires NPU")
@pytest.mark.parametrize(
    "group_sizes,heads,out_dtype",
    [
        pytest.param((5,), None, None, id="single-modality-hidden"),
        pytest.param((2, 1, 2), 2, torch.float32, id="multimodal-qk-fp32-output"),
        pytest.param((2, 0, 3), 2, None, id="multimodal-empty-group"),
    ],
)
def test_real_npu_rmsnorm_matches_native(mocker, npu_runtime, group_sizes, heads, out_dtype):
    reference, adapter = _norm_pair(num_modality=len(group_sizes), out_dtype=out_dtype)
    shape = (sum(group_sizes), 64) if heads is None else (sum(group_sizes), heads, 64)
    x = torch.randn(shape, generator=torch.Generator().manual_seed(11)).to(torch.bfloat16)
    modalities = torch.cat([torch.full((size,), index, dtype=torch.int64) for index, size in enumerate(group_sizes)])
    dispatcher = ModalityDispatcher(modalities, len(group_sizes))
    device_dispatcher = ModalityDispatcher(modalities.to("npu"), len(group_sizes))
    adapter = adapter.to("npu")
    provider = mocker.spy(npu_runtime, "npu_rms_norm")

    with torch.no_grad():
        expected = reference(x, dispatcher)
        actual = adapter(x.to("npu"), device_dispatcher)

    assert provider.call_count == sum(size > 0 for size in group_sizes)
    weights = reference.weight.view(len(group_sizes), 64) + 1.0
    nonempty = [index for index, size in enumerate(group_sizes) if size]
    for call, index in zip(provider.call_args_list, nonempty, strict=True):
        assert call.args[0].dtype == torch.float32
        assert call.args[0].shape[0] == group_sizes[index]
        assert call.kwargs["epsilon"] == 1e-5
        assert call.kwargs["gamma"].dtype == torch.float32
        torch.testing.assert_close(call.kwargs["gamma"].cpu(), weights[index], rtol=0, atol=0)
    assert actual.shape == x.shape and actual.dtype == (out_dtype or x.dtype)
    assert torch.isfinite(actual).all()
    tolerance = dict(rtol=2e-5, atol=3e-6) if out_dtype == torch.float32 else dict(rtol=1e-2, atol=2e-3)
    torch.testing.assert_close(actual.cpu(), expected, **tolerance)
    torch.testing.assert_close(adapter.weight.cpu(), reference.weight, rtol=0, atol=0)


@hardware_test(res={"npu": "A5"}, num_cards=1)
@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires NPU")
@pytest.mark.parametrize(
    "shape,dim,num_patterns",
    [
        pytest.param((5, 2, 32), 64, 1, id="split-features"),
        pytest.param((5, 2, 64), 64, 2, id="multiple-patterns"),
    ],
)
def test_npu_rmsnorm_unsupported_layouts_use_native(mocker, npu_runtime, shape, dim, num_patterns):
    reference, adapter = _norm_pair(dim, num_patterns=num_patterns, out_dtype=torch.float32)
    x = torch.randn(shape, generator=torch.Generator().manual_seed(13)).to(torch.bfloat16)
    provider = mocker.spy(npu_runtime, "npu_rms_norm")

    with torch.no_grad():
        expected = reference(x)
        actual = adapter.to("npu")(x.to("npu"))

    provider.assert_not_called()
    assert actual.shape == x.shape and actual.dtype == torch.float32
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=3e-6)


@hardware_test(res={"npu": "A5"}, num_cards=1)
@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires NPU")
@pytest.mark.parametrize("angle_ndim", [2, 3])
def test_real_npu_rope_fp32_partial_rotation_matches_native(mocker, npu_runtime, angle_ndim):
    pytest.importorskip("mindiesd")
    generator = torch.Generator().manual_seed(17)
    # MAGI-2 normalizes Q/K to FP32 and casts to checkpoint dtype after RoPE.
    x = torch.randn(1, 11, 3, 128, generator=generator)
    angles = torch.randn(11, 48, generator=generator)
    cos, sin = angles.cos(), angles.sin()
    if angle_ndim == 3:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    expected = native_apply_rotary_emb(x, cos, sin)
    device_x = x.to("npu").transpose(1, 2).contiguous().transpose(1, 2)
    assert not device_x.is_contiguous()
    assert npu_rope._HAS_MINDIESD
    provider = mocker.spy(npu_rope, "apply_rotary_emb_mindiesd")
    fallback = mocker.spy(npu_rope, "native_apply_rotary_emb")

    with torch.inference_mode():
        actual = npu_rope.apply_rotary_emb(device_x, cos.to("npu"), sin.to("npu"))

    provider.assert_called_once()
    fallback.assert_not_called()
    args, kwargs = provider.call_args
    assert args[0].shape == (1, 11, 3, 96) and args[0].is_contiguous()
    assert all(tensor.dtype == torch.float32 for tensor in args)
    assert kwargs == {"interleaved": False, "half_head_dim": True}
    assert actual.shape == x.shape and actual.dtype == torch.float32
    assert torch.equal(actual[..., 96:], device_x[..., 96:])
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(device_x.cpu(), x, rtol=0, atol=0)


@hardware_test(res={"npu": "A5"}, num_cards=1)
@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires NPU")
@pytest.mark.parametrize(
    "batch,shared_angles,interleaved,has_mindiesd",
    [
        pytest.param(2, False, False, True, id="batch-specific-angles"),
        pytest.param(1, True, True, True, id="interleaved"),
        pytest.param(1, True, False, False, id="without-mindiesd"),
    ],
)
def test_npu_rope_unsupported_contracts_use_native(
    mocker, monkeypatch, npu_runtime, batch, shared_angles, interleaved, has_mindiesd
):
    generator = torch.Generator().manual_seed(23)
    x = torch.randn(batch, 11, 3, 128, generator=generator)
    angle_shape = (11, 48) if shared_angles else (batch, 11, 48)
    angles = torch.randn(angle_shape, generator=generator)
    cos, sin = angles.cos(), angles.sin()
    expected = native_apply_rotary_emb(x, cos, sin, interleaved=interleaved)
    monkeypatch.setattr(npu_rope, "_HAS_MINDIESD", has_mindiesd)
    provider = mocker.spy(npu_rope, "apply_rotary_emb_mindiesd")
    fallback = mocker.spy(npu_rope, "native_apply_rotary_emb")

    with torch.inference_mode():
        actual = npu_rope.apply_rotary_emb(x.to("npu"), cos.to("npu"), sin.to("npu"), interleaved=interleaved)

    provider.assert_not_called()
    fallback.assert_called_once()
    assert actual.shape == x.shape and actual.dtype == torch.float32
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-5)


@pytest.mark.cpu
@pytest.mark.parametrize("interleaved", [False, True])
def test_npu_rope_cpu_inputs_use_native(mocker, monkeypatch, interleaved):
    generator = torch.Generator().manual_seed(29)
    x = torch.randn(1, 7, 2, 64, generator=generator)
    angles = torch.randn(7, 24, generator=generator)
    cos, sin = angles.cos(), angles.sin()
    expected = native_apply_rotary_emb(x, cos, sin, interleaved=interleaved)
    monkeypatch.setattr(npu_rope, "_HAS_MINDIESD", True)
    provider = mocker.spy(npu_rope, "apply_rotary_emb_mindiesd")
    fallback = mocker.spy(npu_rope, "native_apply_rotary_emb")

    actual = npu_rope.apply_rotary_emb(x, cos, sin, interleaved=interleaved)

    provider.assert_not_called()
    fallback.assert_called_once()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
