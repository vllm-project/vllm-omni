# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VAE initialization and residency contracts for the optional Omni operators and their INT8 composition."""

from importlib.util import find_spec
from types import FunctionType, MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.minimax_h3 import vae as vae_module
from vllm_omni.diffusion.models.minimax_h3 import vae_nvfp4
from vllm_omni.diffusion.offloader.module_residency import PinnedModuleStager
from vllm_omni.quantization.int8_config import DiffusionInt8Config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

HAS_VAE_INT8 = find_spec("vllm_omni.diffusion.models.minimax_h3.vae_int8") is not None
if HAS_VAE_INT8:
    from vllm_omni.diffusion.models.minimax_h3 import vae_int8

INT8_CASES = [False, pytest.param(True, marks=pytest.mark.skipif(not HAS_VAE_INT8, reason="Requires VAE INT8"))]


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
        quantized=False,
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
        if quantized and not HAS_VAE_INT8:
            pytest.skip("Requires VAE INT8")
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
        if HAS_VAE_INT8:
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
        if HAS_VAE_INT8:
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
            **({"quant_config": config} if HAS_VAE_INT8 else {}),
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


@pytest.mark.parametrize("quantized", INT8_CASES)
@pytest.mark.parametrize("env", [None, "0", "false", " OFF ", "", "unknown"])
def test_omni_ops_disabled_by_default_and_false_values(omni_vae_factory, env, quantized):
    result = omni_vae_factory(env, quantized=quantized)
    result.patch.assert_not_called()
    assert result.quantize.call_count == int(quantized)
    if quantized:
        assert result.vae._int8_report == "int8-report"
    block = result.remote.model.decoder.transformer_blocks[0]
    assert isinstance(block.norm1, nn.RMSNorm)
    assert not getattr(block.attn, "_minimax_h3_omni_ops_patched", False)
    assert not hasattr(block.attn, "omni_rope")


@pytest.mark.parametrize("quantized", INT8_CASES)
@pytest.mark.parametrize("env", ["1", "true", "TRUE", " yes ", "On"])
def test_omni_ops_enabled_for_npu_target_with_cpu_stored_weights(omni_vae_factory, env, quantized):
    from vllm_omni.diffusion.layers.norm import RMSNorm

    result = omni_vae_factory(env, quantized=quantized)
    result.patch.assert_called_once_with(result.remote.model)
    assert result.quantize.call_count == int(quantized)
    assert result.optimization.call_count == int(not quantized)
    assert next(result.vae.parameters()).device.type == "cpu"
    block = result.remote.model.decoder.transformer_blocks[0]
    assert isinstance(block.norm1, RMSNorm)
    assert isinstance(block.norm2, RMSNorm)
    assert block.attn._minimax_h3_omni_ops_patched
    assert not hasattr(block.attn, "omni_rope")
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


@pytest.mark.skipif(not HAS_VAE_INT8, reason="Requires VAE INT8")
def test_omni_ops_precedes_int8_and_residency_snapshots(omni_vae_factory):
    result = omni_vae_factory("1", quantized=True)
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


@pytest.mark.skipif(not HAS_VAE_INT8, reason="Requires VAE INT8")
def test_omni_ops_does_not_bypass_int8_first_parameter_guard(omni_vae_factory):
    with pytest.raises(ValueError, match="FP32 encoder/boundary first parameter"):
        omni_vae_factory("1", quantized=True, break_first_parameter=True)


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
    unit_weights = [block.attn.norm_q._unit_weight, block.attn.norm_k._unit_weight]
    assert all(weight.dtype == torch.float32 and not weight.requires_grad for weight in unit_weights)
    state_keys = set(model.state_dict())
    assert not any("_unit_weight" in name for name in state_keys)
    stager = PinnedModuleStager(model, torch.device("cpu"), pin_memory=False)
    for _ in range(2):
        stager.load()
        value = torch.randn(1, 3, 2, 4)
        for norm, weight in zip((block.attn.norm_q, block.attn.norm_k), unit_weights):
            assert norm._unit_weight is weight
            torch.testing.assert_close(norm(value), nn.functional.rms_norm(value, (4,), None, 1e-5))
        stager.offload()
        assert set(model.state_dict()) == state_keys
        assert block.attn._minimax_h3_omni_ops_patched
        for weight in unit_weights:
            torch.testing.assert_close(weight, torch.ones(4), rtol=0, atol=0)
        for name, parameter in model.named_parameters():
            assert id(parameter) == parameter_ids[name]
            assert parameter.dtype == masters[name].dtype
            torch.testing.assert_close(parameter, masters[name])
