# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Check the H3 audio decoder configuration and platform boundary."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

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


def test_stage_installer_maps_the_latent_threshold_per_stage(mocker):
    decoder = nn.Module()
    first = nn.ConvTranspose1d(2, 2, kernel_size=4, stride=2, padding=1)
    second = nn.ConvTranspose1d(2, 1, kernel_size=4, stride=2, padding=1)
    decoder.ups = nn.ModuleList([nn.ModuleList([first]), nn.ModuleList([second])])
    phase = mocker.patch.object(audio_tconv, "_conv_transpose1d_as_conv1d", return_value="phase")
    assert audio_tconv.install_bigvgan_upsamplers(decoder) == 2

    def probe(layer, length):
        layer.eval()
        layer._h3_stage_original_forward = mocker.Mock(return_value="native")
        tensor = SimpleNamespace(
            device=SimpleNamespace(type="npu"),
            dtype=torch.float32,
            ndim=3,
            shape=(1, 2, length),
            numel=lambda: 2 * length,
        )
        return layer(tensor)

    assert probe(first, 127) == "native"
    assert probe(first, 128) == "phase"
    assert probe(second, 255) == "native"
    assert probe(second, 256) == "phase"
    assert phase.call_count == 2


def test_decoder_pipeline_forwards_the_audio_decoder_switch(mocker, tmp_path):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3_decoder as decoder_module

    mocker.patch.object(decoder_module, "resolve_minimax_h3_decoder_model_path", return_value=str(tmp_path))
    mocker.patch.object(decoder_module, "resolve_h3_lightvae_decoder_path", return_value=None)
    mocker.patch.object(decoder_module, "get_local_device", return_value=torch.device("cpu"))
    video = mocker.patch.object(decoder_module, "MiniMaxH3VideoVAE")
    audio = mocker.patch.object(decoder_module, "MiniMaxH3AudioVAE")
    mocker.patch.object(decoder_module.MiniMaxH3DecoderPipeline, "setup_diffusion_pipeline_profiler")
    config = SimpleNamespace(
        model="unused",
        revision=None,
        task_type=None,
        trust_remote_code=False,
        additional_config={"audio_vae_optimized_decode": True},
        vae_stack_tiles=True,
        enable_diffusion_pipeline_profiler=False,
        parallel_config=SimpleNamespace(),
    )
    decoder_module.MiniMaxH3DecoderPipeline(od_config=config)
    assert audio.call_args.kwargs["optimize_decode"] is True
    assert video.called
