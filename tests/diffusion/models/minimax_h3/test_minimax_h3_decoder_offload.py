# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import nullcontext

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3_decoder as decoder
from vllm_omni.diffusion.models.minimax_h3 import vae as native_vae
from vllm_omni.diffusion.offloader.base import OffloadConfig
from vllm_omni.diffusion.offloader.component_utils import prepare_pipeline_components
from vllm_omni.diffusion.offloader.plan_resolver import resolve_offload_plan

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def native_decoders(monkeypatch, mocker):
    class NativeVAE(nn.Module):
        def __init__(self, path, *, device, load_device=None, pin_memory=True, **kwargs):
            super().__init__()
            self.initial_device = load_device or device
            self.residency = self.initial_device
            self.pin_memory = pin_memory
            self.decode_dtype = torch.float32
            self.load_to_device = mocker.Mock(side_effect=lambda: setattr(self, "residency", device))
            self.offload_to_cpu = mocker.Mock(side_effect=lambda: setattr(self, "residency", torch.device("cpu")))
            self.set_omni_component_cache = mocker.Mock()

        @property
        def decoder_component(self):
            return self

    monkeypatch.setattr(decoder, "MiniMaxH3VideoVAE", NativeVAE)
    monkeypatch.setattr(decoder, "MiniMaxH3AudioVAE", NativeVAE)
    monkeypatch.setattr(decoder, "get_local_device", lambda: torch.device("cuda:0"))
    monkeypatch.setattr(decoder, "resolve_minimax_h3_decoder_model_path", lambda *args: "/fake/FL2VA")
    monkeypatch.setattr(decoder, "BoundedAllocatorCache", mocker.Mock(), raising=False)
    monkeypatch.setattr(decoder.MiniMaxH3DecoderPipeline, "setup_diffusion_pipeline_profiler", lambda *args, **kw: None)
    return NativeVAE


def _config(dlo, pin_memory=True):
    return OmniDiffusionConfig(
        model="/fake",
        revision=None,
        task_type="fl2va",
        trust_remote_code=True,
        enable_diffusion_pipeline_profiler=False,
        diffusion_offload_config=None,
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        enable_distributed_layerwise_offload=dlo,
        dlo_use_allgather=False,
        dlo_resident_layers=0,
        dlo_host_registration_limit_gib=0.0,
        host_weight_runtime_mode="disabled",
        pin_cpu_memory=pin_memory,
    )


@pytest.mark.parametrize(
    "pipeline_class",
    [
        decoder.MiniMaxH3DecoderPipeline,
        decoder.MiniMaxH3VideoDecoderPipeline,
        decoder.MiniMaxH3AudioDecoderPipeline,
    ],
)
@pytest.mark.parametrize("dlo", [False, True])
@pytest.mark.parametrize("pin_memory", [False, True])
def test_decoder_initial_placement_and_backend_plan(
    native_decoders, monkeypatch, mocker, pipeline_class, dlo, pin_memory
):
    # Exercise construction and the generic backend's real placement policy.
    monkeypatch.setattr(decoder.current_omni_platform, "empty_cache", lambda: None)
    pipeline = pipeline_class(od_config=_config(dlo, pin_memory))
    resolved = resolve_offload_plan(pipeline, OffloadConfig.from_od_config(pipeline.od_config))
    expected = torch.device("cpu" if dlo else "cuda:0")
    assert resolved.vaes
    for component in resolved.vaes:
        assert component.module.initial_device == expected
        assert component.on_demand is dlo
        assert component.module.pin_memory is pin_memory
    if dlo:
        staged: list[nn.Module] = []
        prepare_pipeline_components(
            resolved, device=pipeline.device, staged_components=staged, enable_encoder_blocks=lambda _: False
        )
        assert staged == [component.module for component in resolved.vaes]
        for component in resolved.vaes:
            component.module.set_omni_component_cache.assert_called_once_with(pipeline._dlo_component_cache)
    for component in resolved.vaes:
        vae = component.module
        for _ in range(2):
            with pipeline._component_on_device(vae):
                assert vae.residency == pipeline.device
            assert vae.residency == expected
        if dlo:
            with pytest.raises(RuntimeError, match="decode failure"):
                with pipeline._component_on_device(vae):
                    raise RuntimeError("decode failure")
            assert vae.residency.type == "cpu"
            assert vae.load_to_device.call_count == 3
        else:
            vae.load_to_device.assert_not_called()
            vae.offload_to_cpu.assert_not_called()


def test_nonleader_audio_does_not_load_or_register_vae(native_decoders, monkeypatch):
    monkeypatch.setattr(decoder.MiniMaxH3DecoderPipeline, "_is_output_owner_rank", staticmethod(lambda: False))
    pipeline = decoder.MiniMaxH3AudioDecoderPipeline(od_config=_config(True))
    assert pipeline.audio_vae is None
    resolved = resolve_offload_plan(pipeline, OffloadConfig.from_od_config(pipeline.od_config))
    assert not resolved.vaes


def test_public_dit_selector_does_not_implicitly_stage_vae(native_decoders):
    config = OmniDiffusionConfig(
        model="/fake",
        task_type="fl2va",
        diffusion_offload_config={"mode": "layer", "components": ["dit"]},
    )
    pipeline = decoder.MiniMaxH3VideoDecoderPipeline(od_config=config)
    assert pipeline.video_vae.initial_device == pipeline.device
    assert not pipeline._uses_manual_component_offload(pipeline.video_vae)
    with pytest.raises(ValueError, match="No DiT"):
        resolve_offload_plan(pipeline, OffloadConfig.from_od_config(config))


@pytest.mark.parametrize("pin_memory", [False, True])
@pytest.mark.parametrize("modality", ["video", "audio"])
def test_native_vae_passes_host_pinning_to_stager(monkeypatch, mocker, pin_memory, modality):
    remote = nn.Module()
    remote.model = nn.Module()
    remote.model.decoder = nn.Linear(2, 2)
    monkeypatch.setattr(native_vae, "_load_component_config", lambda _: {"sample_rate": 32000})
    monkeypatch.setattr(native_vae, "_load_remote_component", lambda *args, **kwargs: remote)
    monkeypatch.setattr(native_vae, "install_h3_vae_optimizations", lambda *args, **kwargs: None)
    monkeypatch.setattr(native_vae, "install_temporal_stream_patches", lambda *args, **kwargs: None)
    stager = mocker.Mock()
    monkeypatch.setattr(native_vae, "PinnedModuleStager", stager)
    cls = native_vae.MiniMaxH3VideoVAE if modality == "video" else native_vae.MiniMaxH3AudioVAE
    cls("/fake", device=torch.device("cuda:0"), load_device=torch.device("cpu"), pin_memory=pin_memory)
    stager.assert_called_once_with(remote, torch.device("cuda:0"), pin_memory=pin_memory)


@pytest.mark.parametrize("owner", [False, True])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
def test_video_reply_materialization(native_decoders, monkeypatch, mocker, owner, dtype):
    """Non-reply ranks must still decode every latent, without preparing frames."""
    pipeline = decoder.MiniMaxH3VideoDecoderPipeline(od_config=_config(False))
    pipeline.device = torch.device("cpu")
    monkeypatch.setattr(pipeline, "_is_output_owner_rank", lambda: owner)
    monkeypatch.setattr(pipeline, "_component_on_device", lambda _: nullcontext())
    monkeypatch.setattr(decoder.current_omni_platform, "create_autocast_context", lambda **kw: nullcontext())
    monkeypatch.setattr(pipeline, "_release_stage_cache", lambda: None)
    # Temporal-major native output is noncontiguous in B,C,T,H,W; cropping
    # also leaves row padding. The reply must preserve the original output.
    frames = torch.arange(120).reshape(1, 2, 3, 4, 5).to(dtype)
    if dtype == torch.float32:
        frames.div_(119)
    decoded = frames.transpose(1, 2)
    decode = mocker.Mock(side_effect=lambda _: decoded.clone())
    pipeline.video_vae.decode_latent = decode
    prepare = mocker.spy(decoder, "_prepare_minimax_h3_video_output")
    transfer = mocker.Mock(side_effect=lambda value: value)
    monkeypatch.setattr(pipeline, "_offload_model_cpu_stage_output", transfer)
    request = mocker.Mock(
        prompts=[
            {
                "additional_information": {
                    "minimax_h3_decode": {
                        "video_latents": [torch.zeros(1), torch.ones(1)],
                        "height": 3,
                        "width": 4,
                    }
                }
            }
        ]
    )
    result = pipeline.forward(request)
    assert decode.call_count == 2
    if owner:
        expected = decoded[..., :3, :4].float().clamp(0, 1).mul(255).round().to(torch.uint8)
        if dtype == torch.uint8:
            expected = decoded[..., :3, :4]
        expected = expected.permute(0, 2, 3, 4, 1).contiguous()
        assert torch.equal(result.output, torch.cat([expected, expected]))
        assert result.output.is_contiguous()
        assert prepare.call_count == transfer.call_count == 2
    else:
        assert result.output is None
        assert result.post_process_func is None
        prepare.assert_not_called()
        transfer.assert_not_called()
