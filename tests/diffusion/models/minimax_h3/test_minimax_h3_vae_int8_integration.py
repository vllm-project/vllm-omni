# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VAE initialization and residency contracts for the explicit INT8 path."""

from contextlib import nullcontext
from types import FunctionType, MethodType, SimpleNamespace

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


def _native_transformer_block(mocker):
    """Small remote structure for exercising the real adapter installation."""

    def forward(self, hidden_states, rotary_pos_emb=None, pack_info=None):
        return hidden_states

    block = nn.Module()
    block.norm1 = nn.RMSNorm(4, eps=1e-5)
    block.norm2 = nn.RMSNorm(4, eps=1e-5)
    block.attn = nn.Module()
    block.attn.to_qkv = nn.Linear(4, 12)
    block.attn.to_out = nn.Linear(4, 4)
    block.attn.norm_q = nn.RMSNorm(4, eps=1e-5, elementwise_affine=False)
    block.attn.norm_k = nn.RMSNorm(4, eps=1e-5, elementwise_affine=False)
    block.attn.dim_head = 4
    block.attn.spatial_parallel = False
    block.attn.perform_attention = nn.Identity()
    # Model the checkpoint's module-local Python helpers without modifying
    # this test module's globals. No forward/operator is executed here.
    namespace = {name: mocker.Mock() for name in ("all_to_all_4D", "get_parallel_state", "_vit_norm_input")}
    block.attn.forward = MethodType(FunctionType(forward.__code__, namespace), block.attn)
    return block


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


@pytest.fixture
def omni_vae_factory(monkeypatch, mocker):
    """Use the real constructor with CPU storage and fake hardware boundaries."""

    from vllm_omni.diffusion.models.minimax_h3 import vae_ops

    install_omni_ops = vae_ops.patch_minimax_h3_video_vae

    def create(
        env,
        *,
        target_type="npu",
        platform_npu=True,
        encode_only=False,
        missing_decoder=False,
        replacement_decoder=False,
        quantized=True,
        break_first_parameter=False,
    ):
        if env is None:
            monkeypatch.delenv("MINIMAX_H3_VAE_USE_OMNI_OPS", raising=False)
        else:
            monkeypatch.setenv("MINIMAX_H3_VAE_USE_OMNI_OPS", env)
        # torch.device("npu") need not be registered on this CPU test host.
        # Only the target is fake; every parameter and placement is real CPU.
        target = SimpleNamespace(type=target_type)
        events: list[tuple[str, object]] = []
        remote = _Remote(events)
        remote.model.decoder.transformer_blocks = nn.ModuleList([_native_transformer_block(mocker)])
        if encode_only or missing_decoder:
            del remote.model.decoder
            del remote.model.post_quant_conv
        config = DiffusionInt8Config() if quantized else None
        monkeypatch.setattr(
            vae_module,
            "_load_component_config",
            lambda _: {"latent_channels": 1, "latents_mean": [0.0], "latents_std": [1.0]},
        )
        monkeypatch.setattr(vae_module, "_load_remote_component", lambda *args, **kwargs: remote)
        monkeypatch.setattr(vae_module, "_load_video_vae_encoder", lambda *args, **kwargs: remote)
        monkeypatch.setattr(vae_module, "current_omni_platform", SimpleNamespace(is_npu=lambda: platform_npu))
        monkeypatch.setattr(
            vae_nvfp4,
            "resolve_video_decoder_checkpoint",
            mocker.Mock(return_value="replacement.safetensors" if replacement_decoder else None),
        )
        monkeypatch.setattr(vae_nvfp4, "install_nvfp4_video_decoder", mocker.Mock())
        monkeypatch.setattr(vae_int8, "validate_h3_video_vae_quant_config", mocker.Mock())
        optimization = mocker.Mock()
        monkeypatch.setattr(vae_module, "install_h3_vae_optimizations", optimization)

        def patch(model):
            assert model is remote.model
            assert events[-1][0] == "place"
            assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters())
            install_omni_ops(model)
            events.append(("omni_patch", None))

        patch_mock = mocker.Mock(side_effect=patch)
        monkeypatch.setattr(vae_ops, "patch_minimax_h3_video_vae", patch_mock)

        def quantize(decoder, quant_config, **kwargs):
            assert quant_config is config
            assert kwargs == {
                "quant_device": target,
                "storage_device": torch.device("cpu"),
                "compute_dtype": torch.float16,
            }
            assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in remote.parameters())
            decoder[0] = _StoredInt8Linear()
            if break_first_parameter:
                remote.model.encoder.half()
            events.append(("quantize", None))
            return "int8-report"

        quantize_mock = mocker.Mock(side_effect=quantize)
        monkeypatch.setattr(vae_int8, "install_h3_video_vae_int8", quantize_mock)

        def temporal(model, **kwargs):
            assert model is remote.model
            events.append(("temporal", None))

        monkeypatch.setattr(vae_module, "install_temporal_stream_patches", temporal)

        def stager(modules, device, **kwargs):
            assert device is target
            assert kwargs == {"pin_memory": True}
            assert events[-1][0] in ("temporal", "snapshot")
            if quantized:
                assert remote.model.decoder[0].weight.dtype == torch.int8
                assert remote.model.decoder[0].weight_scale.dtype == torch.float32
            if patch_mock.called:
                assert remote.model.decoder.transformer_blocks[0].norm1.weight.dtype == torch.float32
            events.append(("snapshot", None))
            return mocker.Mock()

        monkeypatch.setattr(vae_module, "PinnedModuleStager", stager)
        vae = vae_module.MiniMaxH3VideoVAE(
            "unused",
            device=target,
            load_device=torch.device("cpu"),
            encode_only=encode_only,
            decoder_source="replacement" if replacement_decoder else None,
            quant_config=config,
        )
        return SimpleNamespace(
            vae=vae,
            remote=remote,
            events=events,
            patch=patch_mock,
            quantize=quantize_mock,
            optimization=optimization,
        )

    return create


@pytest.mark.parametrize("env", [None, "0", "false", " OFF ", "", "unknown"])
def test_omni_ops_disabled_by_default_and_false_values(omni_vae_factory, env):
    result = omni_vae_factory(env)
    result.patch.assert_not_called()
    result.quantize.assert_called_once()
    assert result.vae._int8_report == "int8-report"
    block = result.remote.model.decoder.transformer_blocks[0]
    assert isinstance(block.norm1, nn.RMSNorm)
    assert not getattr(block.attn, "_minimax_h3_omni_ops_patched", False)
    assert not hasattr(block.attn, "omni_rope")


@pytest.mark.parametrize("env", ["1", "true", "TRUE", " yes ", "On"])
def test_omni_ops_enabled_for_npu_target_with_cpu_stored_weights(omni_vae_factory, env):
    from vllm_omni.diffusion.layers.norm import RMSNorm

    result = omni_vae_factory(env)
    result.patch.assert_called_once_with(result.remote.model)
    result.quantize.assert_called_once()
    result.optimization.assert_not_called()
    assert next(result.vae.parameters()).device.type == "cpu"
    block = result.remote.model.decoder.transformer_blocks[0]
    assert isinstance(block.norm1, RMSNorm)
    assert isinstance(block.norm2, RMSNorm)
    assert block.attn._minimax_h3_omni_ops_patched
    assert hasattr(block.attn, "omni_rope")
    assert isinstance(block.attn.norm_q, nn.RMSNorm)
    assert block.attn.norm_q.weight is None


def test_omni_ops_can_be_enabled_without_int8(omni_vae_factory):
    from vllm_omni.diffusion.layers.norm import RMSNorm

    result = omni_vae_factory("1", quantized=False)
    result.patch.assert_called_once_with(result.remote.model)
    result.quantize.assert_not_called()
    block = result.remote.model.decoder.transformer_blocks[0]
    assert isinstance(block.norm1, RMSNorm)
    assert block.attn._minimax_h3_omni_ops_patched
    assert [name for name, _ in result.events] == ["place", "omni_patch", "temporal", "snapshot", "snapshot"]


@pytest.mark.parametrize(
    "options",
    [
        {"target_type": "cpu"},
        {"target_type": "cuda"},
        {"platform_npu": False},
        {"encode_only": True},
        {"missing_decoder": True},
        {"target_type": "cuda", "replacement_decoder": True},
    ],
    ids=["cpu-target", "cuda-target", "non-npu-platform", "encode-only", "no-decoder", "nvfp4-decoder"],
)
def test_omni_ops_skips_unsupported_targets_and_decoders(omni_vae_factory, options):
    result = omni_vae_factory("1", quantized=False, **options)
    result.patch.assert_not_called()
    result.quantize.assert_not_called()


def test_omni_ops_precedes_int8_and_residency_snapshots(omni_vae_factory):
    result = omni_vae_factory("1")
    assert [name for name, _ in result.events] == [
        "place",
        "omni_patch",
        "quantize",
        "place",
        "temporal",
        "snapshot",
        "snapshot",
    ]
    assert result.events[0][1] == {"device": torch.device("cpu"), "dtype": torch.float32}
    # Adding Omni ops must not turn the post-quantization device move into a
    # dtype cast of registered INT8 weights or FP32 scales, bias, and norms.
    assert result.events[3][1] == {"device": torch.device("cpu")}
    assert next(result.vae.parameters()).dtype == torch.float32
    assert result.remote.model.decoder.transformer_blocks[0].norm1.weight.dtype == torch.float32


def test_omni_ops_does_not_bypass_int8_first_parameter_guard(omni_vae_factory):
    with pytest.raises(ValueError, match="FP32 encoder/boundary first parameter"):
        omni_vae_factory("1", break_first_parameter=True)


def test_omni_norm_and_int8_parameters_survive_residency_cycle(mocker):
    from vllm_omni.diffusion.layers.norm import RMSNorm
    from vllm_omni.diffusion.models.minimax_h3 import vae_ops

    model = nn.Module()
    model.decoder = nn.Module()
    model.decoder.transformer_blocks = nn.ModuleList([_native_transformer_block(mocker)])
    block = model.decoder.transformer_blocks[0]
    with torch.no_grad():
        block.norm1.weight.copy_(torch.arange(1, 5, dtype=torch.float32))
    vae_ops.patch_minimax_h3_video_vae(model)
    assert isinstance(block.norm1, RMSNorm)
    # Match production order: Omni replacement, then mixed INT8 registration,
    # then the real CPU residency snapshot. No numerical operator is mocked.
    block.attn.to_out = _StoredInt8Linear()
    masters = {name: p.detach().clone() for name, p in model.named_parameters()}
    parameter_ids = {name: id(p) for name, p in model.named_parameters()}
    state_keys = set(model.state_dict())
    stager = PinnedModuleStager(model, torch.device("cpu"), pin_memory=False)
    for _ in range(2):
        stager.load()
        stager.offload()
        assert set(model.state_dict()) == state_keys
        assert block.attn._minimax_h3_omni_ops_patched
        for name, parameter in model.named_parameters():
            assert id(parameter) == parameter_ids[name]
            assert parameter.dtype == masters[name].dtype
            torch.testing.assert_close(parameter, masters[name])
