# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VAE initialization and residency contracts for the explicit INT8 path."""

from contextlib import nullcontext

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.minimax_h3 import vae as vae_module
from vllm_omni.diffusion.models.minimax_h3 import vae_int8, vae_nvfp4
from vllm_omni.diffusion.offloader.module_residency import PinnedModuleStager
from vllm_omni.quantization.int8_config import DiffusionInt8Config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _StoredInt8Linear(nn.Module):
    """Storage fixture; numerical delegation is covered by adapter tests."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(16, dtype=torch.int8).reshape(4, 4), requires_grad=False)
        self.weight_scale = nn.Parameter(torch.arange(1, 5, dtype=torch.float32), requires_grad=False)
        self.bias = nn.Parameter(torch.ones(4, dtype=torch.float32), requires_grad=False)


class _Remote(nn.Module):
    def __init__(self, events):
        super().__init__()
        self.events = events
        self.model = nn.Module()
        self.model.encoder = nn.Linear(4, 4)
        self.model.quant_conv = nn.Linear(4, 4)
        self.model.post_quant_conv = nn.Linear(4, 4)
        self.model.decoder = nn.Sequential(nn.Linear(4, 4))

    def to(self, *args, **kwargs):
        self.events.append(("place", kwargs.copy()))
        return super().to(*args, **kwargs)


@pytest.fixture
def quantized_vae(monkeypatch, mocker):
    monkeypatch.delenv("MINIMAX_H3_VAE_USE_OMNI_OPS", raising=False)
    events: list[tuple[str, object]] = []
    remote = _Remote(events)
    quant_config = DiffusionInt8Config()
    report = object()
    monkeypatch.setattr(
        vae_module,
        "_load_component_config",
        lambda _: {
            "latent_channels": 1,
            "latents_mean": [0.0],
            "latents_std": [1.0],
        },
    )
    monkeypatch.setattr(vae_module, "_load_remote_component", lambda *args, **kwargs: remote)
    monkeypatch.setattr(vae_module, "install_temporal_stream_patches", mocker.Mock())
    validation = mocker.Mock()
    monkeypatch.setattr(vae_int8, "validate_h3_video_vae_quant_config", validation)
    optimization = mocker.Mock()
    monkeypatch.setattr(vae_module, "install_h3_vae_optimizations", optimization)

    def install(decoder, config, **kwargs):
        assert config is quant_config
        assert kwargs == {
            "quant_device": torch.device("cuda"),
            "storage_device": torch.device("cpu"),
            "compute_dtype": torch.float16,
        }
        assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in remote.parameters())
        events.append(("quantize", None))
        decoder[0] = _StoredInt8Linear()
        return report

    def stager(modules, device, **kwargs):
        assert device == torch.device("cuda")
        assert kwargs == {"pin_memory": True}
        modules = list(modules)
        if modules[-1] is remote.model.decoder:
            assert modules[-1][0].weight.dtype == torch.int8
            assert modules[-1][0].weight_scale.dtype == torch.float32
        assert events[-1][0] in ("place", "snapshot")
        events.append(("snapshot", None))
        return mocker.Mock()

    monkeypatch.setattr(vae_int8, "install_h3_video_vae_int8", install)
    monkeypatch.setattr(vae_module, "PinnedModuleStager", stager)
    vae = vae_module.MiniMaxH3VideoVAE(
        "unused",
        device=torch.device("cuda"),
        load_device=torch.device("cpu"),
        quant_config=quant_config,
    )
    validation.assert_called_once_with(
        quant_config,
        device=torch.device("cuda"),
        encode_only=False,
        decoder_source=None,
    )
    optimization.assert_not_called()
    assert vae._int8_report is report
    return vae, events


def test_quantization_precedes_residency_snapshot(quantized_vae):
    vae, events = quantized_vae
    assert [name for name, _ in events] == ["place", "quantize", "place", "snapshot", "snapshot"]
    assert events[0][1] == {"device": torch.device("cpu"), "dtype": torch.float32}
    assert events[2][1] == {"device": torch.device("cpu")}
    assert next(vae.parameters()).dtype == torch.float32
    assert vae.model.encoder.weight.dtype == torch.float32
    assert vae.model.post_quant_conv.weight.dtype == torch.float32
    assert vae.model.decoder[0].weight.dtype == torch.int8


def test_keyframe_encode_does_not_recast_quantized_decoder(quantized_vae, monkeypatch, mocker):
    from PIL import Image

    vae, _ = quantized_vae
    vae.model.parallel_tiling = False
    vae.model.encode_images = lambda image, **kwargs: [torch.ones(1, 1, 2, 2)]
    # The reference first parameter must stay FP32; otherwise encode_image's
    # whole-adapter dtype conversion would recast the decoder scales/biases.
    monkeypatch.setattr(vae, "to", mocker.Mock(side_effect=AssertionError("whole VAE dtype cast")))
    monkeypatch.setattr(vae, "_encoder_tiling_context", lambda *args: nullcontext())
    monkeypatch.setattr(vae_module, "_minimax_h3_keyframe_encode_context", lambda *args: nullcontext())
    torch.testing.assert_close(vae.encode_image(Image.new("RGB", (2, 2))), torch.ones(1, 4))
    assert vae.model.decoder[0].weight.dtype == torch.int8
    assert vae.model.decoder[0].weight_scale.dtype == torch.float32


def test_invalid_quantization_rejected_before_nvfp4_resolution_or_weight_load(monkeypatch, mocker):
    resolver = mocker.Mock(side_effect=AssertionError("should validate first"))
    loader = mocker.Mock(side_effect=AssertionError("should validate first"))
    monkeypatch.setattr(vae_nvfp4, "resolve_video_decoder_checkpoint", resolver)
    monkeypatch.setattr(vae_module, "_load_remote_component", loader)
    # A CPU target cannot enable this NPU-only path. The actual validator runs.
    with pytest.raises(ValueError):
        vae_module.MiniMaxH3VideoVAE("unused", device=torch.device("cpu"), quant_config=DiffusionInt8Config())
    resolver.assert_not_called()
    loader.assert_not_called()


def test_mixed_int8_fp32_parameters_survive_residency_cycle():
    # Exercise the real storage snapshot/bind logic on CPU without pinning.
    layer = _StoredInt8Linear()
    masters = {name: p.detach().clone() for name, p in layer.named_parameters()}
    parameter_ids = {name: id(p) for name, p in layer.named_parameters()}
    stager = PinnedModuleStager(layer, torch.device("cpu"), pin_memory=False)
    for _ in range(2):
        stager.load()
        stager.offload()
        for name, parameter in layer.named_parameters():
            assert id(parameter) == parameter_ids[name]
            assert parameter.dtype == masters[name].dtype
            torch.testing.assert_close(parameter, masters[name])
