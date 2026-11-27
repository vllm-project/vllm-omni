# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Check the H3 audio decoder configuration and platform boundary."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3 import vae
from vllm_omni.diffusion.models.minimax_h3.npu import audio_tconv
from vllm_omni.diffusion.models.minimax_h3.vae_audio_fir_channel_pack import _fir_forward

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    ("config", "expected"), [(None, False), ({}, False), ({"audio_vae_optimized_decode": True}, True)]
)
def test_audio_decoder_requires_explicit_boolean_opt_in(config, expected):
    assert vae.resolve_audio_vae_optimized_decode(config) is expected


def test_audio_decoder_rejects_string_false():
    with pytest.raises(ValueError, match="must be a boolean"):
        vae.resolve_audio_vae_optimized_decode({"audio_vae_optimized_decode": "false"})


@pytest.mark.parametrize(("device_type", "device_name"), [("cpu", "cpu"), ("npu", "Ascend910B2"), ("npu", "Ascend950")])
def test_audio_decoder_preserves_reference_outside_validated_platform(device_type, device_name, mocker):
    get_name = mocker.patch.object(vae.current_omni_platform, "get_device_name", return_value=device_name)
    install = mocker.patch.object(audio_tconv, "install_bigvgan_upsamplers")
    vae.MiniMaxH3AudioVAE._install_optimized_decode(SimpleNamespace(), SimpleNamespace(type=device_type, index=1))
    assert get_name.call_count == int(device_type == "npu")
    install.assert_not_called()


def test_width_one_fir_fallback_is_not_a_fast_path_execution(mocker):
    module = SimpleNamespace(
        filter=torch.zeros(1, 1, 12),
        training=False,
        _h3_fir_original=mocker.Mock(return_value=mocker.sentinel.reference),
    )
    tensor = mocker.Mock(ndim=3, dtype=torch.float32, shape=(1, 3, 12))
    tensor.device.type = "npu"
    tensor.numel.return_value = 36
    packed_forward = mocker.Mock()
    assert _fir_forward(module, tensor, packed_forward=packed_forward) is mocker.sentinel.reference
    packed_forward.assert_not_called()
    assert module._h3_fir_fallback_calls == 1
    assert not hasattr(module, "_h3_fir_c16_calls")
