# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import copy

import pytest
import torch
import torch.nn.functional as F

from tests.diffusion.models.minimax_h3.h3_vae_encoder_replica import checkpoint_pad_chain, make_encoder
from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.minimax_h3.ops.vae.dispatch import resolve_h3_vae_operators
from vllm_omni.diffusion.models.minimax_h3.ops.vae.encoder import install_h3_vae_encoder_fusion
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

PADS = {"front": 2, "top": 1, "bottom": 1, "left": 1, "right": 1, "apply_silu": True}


def _pad_operator():
    if not current_omni_platform.is_available():
        pytest.skip("No accelerator is available")
    device = current_omni_platform.get_torch_device()
    pad_op = getattr(resolve_h3_vae_operators(device), "causal_reflect_pad", None)
    if pad_op is None:
        pytest.skip("No exact H3 VAE encoder operator is registered for this target")
    return device, pad_op


def _assert_same_bits(actual, expected):
    assert actual is not None and actual.shape == expected.shape and actual.is_contiguous()
    assert torch.equal(actual.view(torch.int32), expected.contiguous().view(torch.int32))


@hardware_test(res={"cuda": ["H100"]})
@pytest.mark.parametrize("shape", [(1, 128, 17, 64, 64), (1, 64, 9, 33, 17), (2, 32, 1, 16, 16), (1, 32, 5, 2, 300)])
@pytest.mark.parametrize("pads", [(2, 1, 1, 1, 1), (2, 0, 1, 0, 1), (2, 0, 0, 0, 0), (0, 1, 1, 1, 1), (3, 1, 0, 0, 1)])
def test_fused_padding_is_bit_exact(shape, pads):
    device, pad_op = _pad_operator()
    batch, channels, frames, height, width = shape
    normalized = 4 * torch.randn(batch, frames, channels, height + 2, width + 2, device=device)
    source = normalized[..., 1:-1, 1:-1].permute(0, 2, 1, 3, 4)
    pads = dict(zip(("front", "top", "bottom", "left", "right"), pads))
    with torch.inference_mode():
        for x in (source, source.contiguous(), source[:, :, :1]):
            for apply_silu in (True, False):
                expected = checkpoint_pad_chain(x, **pads, apply_silu=apply_silu)
                _assert_same_bits(pad_op(x, **pads, apply_silu=apply_silu), expected)


@hardware_test(res={"cuda": ["H100"]})
def test_fused_silu_matches_eager_for_every_fp32_value():
    device, pad_op = _pad_operator()
    chunk = 1 << 26
    with torch.inference_mode():
        for start in range(-(1 << 31), 1 << 31, chunk):
            bits = torch.arange(start, start + chunk, dtype=torch.int64, device=device).to(torch.int32)
            values = bits.view(torch.float32).view(1, 1, 1, chunk // 1024, 1024)
            actual = pad_op(values, front=0, top=0, bottom=0, left=0, right=0, apply_silu=True)
            _assert_same_bits(actual, F.silu(values))


@hardware_test(res={"cuda": ["H100"]})
def test_fused_padding_rejects_inputs_outside_its_contract():
    device, pad_op = _pad_operator()
    source = torch.randn(1, 2, 3, 4, 5, device=device)
    with torch.inference_mode():
        assert pad_op(source.half(), **PADS) is None
        assert pad_op(source.cpu(), **PADS) is None
        assert pad_op(source[:, :, :, :1], **PADS) is None
        assert pad_op(source, **{**PADS, "left": -1}) is None
    with torch.enable_grad():
        assert pad_op(source.requires_grad_(), **PADS) is None


@hardware_test(res={"cuda": ["H100"]}, num_cards=2)
def test_fused_padding_declines_a_tensor_off_the_current_device():
    _device, pad_op = _pad_operator()
    if torch.accelerator.device_count() < 2:
        pytest.skip("Needs two CUDA devices")
    other = torch.device("cuda", (torch.accelerator.current_device_index() + 1) % torch.accelerator.device_count())
    source = torch.randn(1, 2, 3, 4, 5, device=other)
    with torch.inference_mode():
        assert pad_op(source, **PADS) is None
        with torch.accelerator.device_index(other.index):
            _assert_same_bits(pad_op(source, **PADS), checkpoint_pad_chain(source, **PADS))


@hardware_test(res={"cuda": ["H100"]})
@pytest.mark.parametrize("time_isolated", [True, False])
def test_fused_encoder_matches_checkpoint_without_extra_memory(time_isolated):
    device, pad_op = _pad_operator()
    reference = make_encoder(time_isolated).to(device)
    fused = copy.deepcopy(reference)
    with torch.device(device):
        assert install_h3_vae_encoder_fusion(fused, device=device)
    assert fused.conv_in._omni_causal_reflect_pad is pad_op
    video = torch.randn(1, 3, 40, 80, 72, device=device)
    clip = torch.randn(1, 3, 17, 128, 128, device=device)

    def peak(encoder):
        torch.accelerator.synchronize(device)
        torch.accelerator.reset_peak_memory_stats(device)
        base = torch.accelerator.memory_allocated(device)
        encoder(clip)
        torch.accelerator.synchronize(device)
        return torch.accelerator.max_memory_allocated(device) - base

    with torch.inference_mode():
        for x in (video[:, :, 3:20, 8:72, 4:68], video[:, :, 7:8, :64, :64], video[:, :, :5, :32, :48], clip):
            assert torch.equal(fused(x), reference(x))
        assert peak(fused) <= peak(reference)
