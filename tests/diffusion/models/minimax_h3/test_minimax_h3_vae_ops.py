# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.gpu, pytest.mark.diffusion]


def _selected_operators():
    from vllm_omni.diffusion.models.minimax_h3.ops.vae.dispatch import (
        resolve_h3_vae_operators,
    )

    if not current_omni_platform.is_available():
        pytest.skip("No accelerator is available")
    device = current_omni_platform.get_torch_device()
    operators = resolve_h3_vae_operators(device)
    if operators is None:
        pytest.skip("No H3 VAE operator implementation is registered for this target")
    return device, operators


def _operator_set(*, supports=lambda _device: True):
    from vllm_omni.diffusion.models.minimax_h3.ops.vae.dispatch import H3VAEOperatorSet

    return H3VAEOperatorSet(
        supports=supports,
        qk_norm_rope=lambda *_args: None,
        scaled_residual=lambda *_args: None,
    )


def _qk_reference(x, cos, sin):
    normalized = F.rms_norm(x.float(), (64,), None, 1e-5).to(x.dtype)
    rotary = normalized[..., :48]
    first, second = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return torch.cat(
        (rotary * cos + rotated * sin, normalized[..., 48:]),
        dim=-1,
    )


def _vit_norm_input(_module, hidden_states):
    return hidden_states.float()


def _same_dtype_norm_input(_module, hidden_states):
    return hidden_states


def _failing_norm_input(_module, _hidden_states):
    raise RuntimeError("unsupported remote normalization semantics")


@pytest.mark.parametrize(("batch", "sequence"), [(1, 1), (1, 195), (2, 1797)])
def test_h3_vae_qk_norm_rope_is_bit_exact(batch, sequence):
    device, operators = _selected_operators()

    torch.manual_seed(17)
    qkv = torch.randn(
        batch,
        sequence,
        32,
        192,
        device=device,
        dtype=torch.float16,
    )
    q, k, _ = qkv.chunk(3, dim=-1)
    cos = torch.randn(batch, sequence, 1, 48, device=device, dtype=torch.float16)
    sin = torch.randn_like(cos)

    expected_q = _qk_reference(q, cos, sin)
    expected_k = _qk_reference(k, cos, sin)
    with torch.inference_mode():
        actual = operators.qk_norm_rope(q, k, (cos, sin), 1e-5)

    assert actual is not None
    actual_q, actual_k = actual
    assert torch.equal(actual_q, expected_q)
    assert torch.equal(actual_k, expected_k)


def test_h3_vae_scaled_residual_is_bit_exact():
    device, operators = _selected_operators()

    torch.manual_seed(29)
    residual = torch.randn(195, 2048, device=device, dtype=torch.float32)
    branch = torch.randn(195, 2048, device=device, dtype=torch.float16)
    scale = torch.randn(2048, device=device, dtype=torch.float32)
    expected = residual + branch * scale

    with torch.inference_mode():
        actual = operators.scaled_residual(residual, branch, scale)

    assert actual is not None
    assert torch.equal(actual, expected)


def test_h3_vae_exact_ops_reject_unsupported_inputs():
    from vllm_omni.diffusion.models.minimax_h3.ops.vae.qk_norm_rope import (
        try_qk_norm_rope_exact,
    )
    from vllm_omni.diffusion.models.minimax_h3.ops.vae.scaled_residual import try_scaled_residual_exact

    q = torch.randn(1, 2, 32, 64, dtype=torch.float32)
    cos = torch.randn(1, 2, 1, 48, dtype=torch.float32)
    residual = torch.randn(2, 2048)
    branch = torch.randn(2, 2048, dtype=torch.float16)
    scale = torch.randn(2048)
    with torch.inference_mode():
        assert try_qk_norm_rope_exact(q, q, (cos, cos), 1e-5) is None
        assert try_scaled_residual_exact(residual, branch, scale) is None


def _make_decoder():
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_qkv = nn.Linear(16, 48)
            self.to_out = nn.Linear(16, 16)
            self.norm_q = nn.RMSNorm(16, elementwise_affine=False)
            self.norm_k = nn.RMSNorm(16, elementwise_affine=False)
            self.spatial_parallel = False
            self.dim_head = 16

        def perform_attention(self, query, _key, _value, _pack_info):
            return query

        def forward(self, hidden_states, rotary_pos_emb=None, pack_info=None):
            return hidden_states

    class FeedForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Linear(16, 64)
            self.w2 = nn.Linear(32, 16)
            self.use_gated = True
            self.act_fn = nn.SiLU()
            self._compile_forward_enabled = False
            self._compile_forward_fatal = False
            self._compiled_forward = None

        def forward(self, hidden_states):
            hidden_states = self.w1(hidden_states)
            gate, hidden_states = hidden_states.chunk(2, dim=-1)
            return self.w2(self.act_fn(gate) * hidden_states)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attention()
            self.ff = FeedForward()
            self.norm1 = nn.RMSNorm(16)
            self.norm2 = nn.RMSNorm(16)
            self.scale1 = nn.Parameter(torch.zeros(16))
            self.scale2 = nn.Parameter(torch.zeros(16))
            self.use_scale = True

        def forward(self, hidden_states, rotary_pos_emb=None, pack_info=None):
            return hidden_states

    decoder = nn.Module()
    decoder.transformer_blocks = nn.ModuleList([Block(), Block()])
    decoder.proj_out = nn.Linear(16, 16)
    return decoder


def test_video_vae_installs_exact_optimizations(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import vae as vae_module

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.Module()

    class FakeRemote(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeModel()

    monkeypatch.setattr(
        vae_module,
        "_load_component_config",
        lambda _path: {
            "latent_channels": 1,
            "latents_mean": [0.0],
            "latents_std": [1.0],
        },
    )
    monkeypatch.setattr(
        vae_module,
        "_load_remote_component",
        lambda _path, _config: FakeRemote(),
    )
    install = Mock(return_value=True)
    monkeypatch.setattr(vae_module, "install_h3_vae_optimizations", install)
    monkeypatch.setattr(vae_module, "PinnedModuleStager", Mock())

    video_vae = vae_module.MiniMaxH3VideoVAE(
        "unused",
        device=torch.device("cuda"),
        load_device=torch.device("cpu"),
    )

    install.assert_called_once_with(
        video_vae.model.decoder,
        device=torch.device("cuda"),
    )


def test_video_vae_installs_requested_fp8_layers(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import vae as vae_module

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.Module()

    class FakeRemote(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeModel()

    monkeypatch.setattr(
        vae_module,
        "_load_component_config",
        lambda _path: {
            "latent_channels": 1,
            "latents_mean": [0.0],
            "latents_std": [1.0],
        },
    )
    monkeypatch.setattr(
        vae_module,
        "_load_remote_component",
        lambda _path, _config: FakeRemote(),
    )
    install = Mock(return_value=True)
    install_fp8 = Mock()
    monkeypatch.setattr(vae_module, "install_h3_vae_optimizations", install)
    monkeypatch.setattr(vae_module, "install_h3_vae_fp8_quantization", install_fp8)

    vae_module.MiniMaxH3VideoVAE(
        "unused",
        device=torch.device("cpu"),
        fp8_layers=frozenset({"attn.to_qkv", "ff.w1", "ff.w2"}),
    )

    install.assert_called_once()
    assert isinstance(install.call_args.args[0], torch.nn.Module)
    assert install.call_args.kwargs == {
        "device": torch.device("cpu"),
    }
    install_fp8.assert_called_once()
    assert install_fp8.call_args.args == install.call_args.args
    assert install_fp8.call_args.kwargs == {
        "execution_device": torch.device("cpu"),
        "storage_device": torch.device("cpu"),
        "layers": frozenset({"attn.to_qkv", "ff.w1", "ff.w2"}),
    }


def test_video_vae_fp8_config_requires_explicit_component_config():
    from vllm_omni.diffusion.models.minimax_h3.vae import (
        resolve_minimax_h3_video_vae_fp8_layers,
    )
    from vllm_omni.quantization import build_quant_config

    assert resolve_minimax_h3_video_vae_fp8_layers(build_quant_config("fp8")) is None

    disabled_config = build_quant_config(
        {
            "default": {"method": "fp8"},
            "video_vae": None,
        }
    )
    assert resolve_minimax_h3_video_vae_fp8_layers(disabled_config) is None

    selective_config = build_quant_config(
        {
            "video_vae": {
                "method": "fp8",
                "ignored_layers": ["ff.w2"],
            }
        }
    )
    assert resolve_minimax_h3_video_vae_fp8_layers(selective_config) == frozenset({"attn.to_qkv", "ff.w1"})

    aggressive_config = build_quant_config({"video_vae": {"method": "fp8"}})
    assert resolve_minimax_h3_video_vae_fp8_layers(aggressive_config) == frozenset({"attn.to_qkv", "ff.w1", "ff.w2"})


def test_video_vae_fp8_config_rejects_generic_component_fp8():
    from vllm_omni.diffusion.models.minimax_h3.vae import (
        resolve_minimax_h3_video_vae_fp8_layers,
    )
    from vllm_omni.quantization import ComponentQuantizationConfig

    unsupported = Mock()
    unsupported.get_name.return_value = "unsupported"
    config = ComponentQuantizationConfig({"video_vae": unsupported})
    with pytest.raises(ValueError, match="only supports online fp8"):
        resolve_minimax_h3_video_vae_fp8_layers(config)


@pytest.mark.parametrize(
    "video_vae_config",
    [
        {"method": "fp8", "activation_scheme": "static"},
        {"method": "fp8", "ignored_layers": ["attn.to_out"]},
    ],
)
def test_video_vae_fp8_config_rejects_unsupported_policy(video_vae_config):
    from vllm_omni.diffusion.models.minimax_h3.vae import (
        resolve_minimax_h3_video_vae_fp8_layers,
    )
    from vllm_omni.quantization import build_quant_config

    config = build_quant_config({"video_vae": video_vae_config})
    with pytest.raises(ValueError, match="MiniMax H3 video_vae"):
        resolve_minimax_h3_video_vae_fp8_layers(config)


def test_h3_vae_install_precasts_only_block_linears(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3.ops import vae as vae_ops

    operators = _operator_set()
    monkeypatch.setattr(vae_ops, "resolve_h3_vae_operators", lambda _device: operators)
    decoder = _make_decoder()

    assert vae_ops.install_h3_vae_optimizations(
        decoder,
        device=torch.device("meta"),
    )

    for block in decoder.transformer_blocks:
        assert block.attn.to_qkv.weight.dtype == torch.float16
        assert block.attn.to_out.weight.dtype == torch.float16
        assert block.ff.w1.weight.dtype == torch.float16
        assert block.ff.w2.weight.dtype == torch.float16
        assert block.forward.__func__.__name__ == "_optimized_transformer_block"
        assert block.attn.forward.__func__.__name__ == "_optimized_attention"
        assert block.ff.forward.__func__.__name__ == "_optimized_feed_forward"
        assert block.attn._omni_qk_norm_rope is operators.qk_norm_rope
        assert block._omni_scaled_residual is operators.scaled_residual
    assert decoder.proj_out.weight.dtype == torch.float32

    # Repeated installation is idempotent.
    assert vae_ops.install_h3_vae_optimizations(
        decoder,
        device=torch.device("meta"),
    )


def test_h3_vae_fp8_is_independent_from_eager_operator_dispatch(monkeypatch):
    import vllm.model_executor.parameter as parameter_module
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.layers.quantization.online.fp8 import Fp8PtpcOnlineLinearMethod

    from vllm_omni.diffusion.models.minimax_h3 import vae_fp8
    from vllm_omni.diffusion.models.minimax_h3.ops import vae as vae_ops

    resolver = Mock(side_effect=AssertionError("eager-op dispatch must not be used"))
    monkeypatch.setattr(vae_ops, "resolve_h3_vae_operators", resolver)
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_world_size", lambda: 1)
    device, _ = _selected_operators()
    decoder = _make_decoder().to(device)

    vae_fp8.install_h3_vae_fp8_quantization(
        decoder,
        execution_device=device,
        storage_device=device,
        layers=frozenset({"attn.to_qkv", "ff.w1", "ff.w2"}),
    )
    resolver.assert_not_called()

    for block in decoder.transformer_blocks:
        for linear in (
            block.attn.to_qkv,
            block.ff.w1,
            block.ff.w2,
        ):
            assert isinstance(linear, ReplicatedLinear)
            assert isinstance(linear.quant_method, Fp8PtpcOnlineLinearMethod)
            assert linear.weight.dtype == current_omni_platform.fp8_dtype()
            assert linear.weight_scale.dtype == torch.float32
        assert not tuple(block.attn.to_out.buffers())


def test_h3_vae_fp8_backend_preserves_linear_contract(monkeypatch):
    import vllm.model_executor.parameter as parameter_module

    from vllm_omni.diffusion.models.minimax_h3 import vae_fp8

    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_world_size", lambda: 1)
    device, _ = _selected_operators()
    decoder = _make_decoder().to(device)
    feed_forward = decoder.transformer_blocks[0].ff
    hidden_states = torch.randn(37, 16, device=device, dtype=torch.float32)
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16):
        expected = feed_forward(hidden_states)

    vae_fp8.install_h3_vae_fp8_quantization(
        decoder,
        execution_device=device,
        storage_device=device,
        layers=frozenset({"ff.w1", "ff.w2"}),
    )
    for linear in (feed_forward.w1, feed_forward.w2):
        assert linear.quant_method.apply.__func__ is vae_fp8._H3VAEFp8LinearMethod.apply
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16):
        actual = feed_forward(hidden_states)

    assert actual.shape == expected.shape
    assert actual.dtype == torch.float16
    relative_l2 = torch.linalg.vector_norm(actual.float() - expected.float()) / torch.linalg.vector_norm(
        expected.float()
    )
    cosine = F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)
    assert relative_l2 < 0.08
    assert cosine > 0.995


def test_h3_vae_fp8_is_not_processed_twice_by_model_finalizer(monkeypatch):
    import vllm.model_executor.parameter as parameter_module
    from vllm.model_executor.model_loader.reload import finalize_layerwise_processing

    from vllm_omni.diffusion.models.minimax_h3 import vae_fp8

    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_world_size", lambda: 1)
    device, _ = _selected_operators()
    decoder = _make_decoder().to(device)

    vae_fp8.install_h3_vae_fp8_quantization(
        decoder,
        execution_device=device,
        storage_device=device,
        layers=frozenset({"ff.w1", "ff.w2"}),
    )
    weights_before = {
        name: parameter.detach().clone()
        for name, parameter in decoder.named_parameters()
        if name.endswith(("ff.w1.weight", "ff.w2.weight"))
    }

    finalize_layerwise_processing(decoder, model_config=None)

    for name, expected in weights_before.items():
        assert torch.equal(dict(decoder.named_parameters())[name], expected)


def test_h3_vae_fp8_backend_supports_cpu_storage(monkeypatch):
    import vllm.model_executor.parameter as parameter_module

    from vllm_omni.diffusion.models.minimax_h3 import vae_fp8

    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_world_size", lambda: 1)
    device, _ = _selected_operators()
    decoder = _make_decoder().to(device)

    vae_fp8.install_h3_vae_fp8_quantization(
        decoder,
        execution_device=device,
        storage_device=torch.device("cpu"),
        layers=frozenset({"ff.w1", "ff.w2"}),
    )
    for block in decoder.transformer_blocks:
        assert block.ff.w1.weight.device.type == "cpu"
        assert block.ff.w1.weight_scale.device.type == "cpu"
        assert block.ff.w2.weight.device.type == "cpu"
        assert block.ff.w2.weight_scale.device.type == "cpu"

    decoder.to(device)
    hidden_states = torch.randn(37, 16, device=device, dtype=torch.float32)
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16):
        output = decoder.transformer_blocks[0].ff(hidden_states)
    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all()


def test_h3_vae_fp8_is_not_enabled_by_hardware_dispatch_alone(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3.ops import vae as vae_ops

    monkeypatch.setattr(
        vae_ops,
        "resolve_h3_vae_operators",
        lambda _device: _operator_set(),
    )
    decoder = _make_decoder()

    assert vae_ops.install_h3_vae_optimizations(
        decoder,
        device=torch.device("meta"),
    )

    for block in decoder.transformer_blocks:
        assert not tuple(block.ff.w1.buffers())
        assert not tuple(block.ff.w2.buffers())
        assert block.ff.w1.forward.__func__.__name__ == "forward"
        assert block.ff.w2.forward.__func__.__name__ == "forward"


def test_h3_vae_install_accepts_remote_integer_parallel_flag(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3.ops import vae as vae_ops

    monkeypatch.setattr(
        vae_ops,
        "resolve_h3_vae_operators",
        lambda _device: _operator_set(),
    )
    decoder = _make_decoder()
    for block in decoder.transformer_blocks:
        # The official checkpoint loads this JSON boolean-like field as int 0.
        block.attn.spatial_parallel = 0

    assert vae_ops.install_h3_vae_optimizations(
        decoder,
        device=torch.device("meta"),
    )
    assert getattr(decoder, "_omni_h3_vae_optimizations_installed", False)


def test_h3_vae_swiglu_uses_post_linear_fp16_output(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3.ops import vae as vae_ops

    device, _ = _selected_operators()
    operators = _operator_set()
    monkeypatch.setattr(vae_ops, "resolve_h3_vae_operators", lambda _device: operators)
    decoder = _make_decoder().to(device)
    feed_forward = decoder.transformer_blocks[0].ff
    reference_forward = type(feed_forward).forward
    assert vae_ops.install_h3_vae_optimizations(
        decoder,
        device=device,
    )

    hidden_states = torch.randn(4, 16, device=device, dtype=torch.float32)
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16):
        expected = reference_forward(feed_forward, hidden_states)
        actual = feed_forward(hidden_states)

    assert expected.dtype == torch.float16
    assert torch.equal(actual, expected)


def test_h3_vae_install_leaves_unsupported_target_untouched(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3.ops import vae as vae_ops

    monkeypatch.setattr(vae_ops, "resolve_h3_vae_operators", lambda _device: None)
    decoder = _make_decoder()

    assert not vae_ops.install_h3_vae_optimizations(
        decoder,
        device=torch.device("meta"),
    )
    assert not hasattr(decoder, "_omni_h3_vae_optimizations_installed")
    for block in decoder.transformer_blocks:
        assert block.attn.to_qkv.weight.dtype == torch.float32
        assert block.ff.w1.weight.dtype == torch.float32
        assert block.forward.__func__.__name__ == "forward"
        assert block.attn.forward.__func__.__name__ == "forward"


@pytest.mark.parametrize("norm_input", [None, _same_dtype_norm_input, _failing_norm_input])
def test_h3_vae_install_requires_fp32_attention_norm_semantics(monkeypatch, norm_input):
    from vllm_omni.diffusion.models.minimax_h3.ops import vae as vae_ops

    monkeypatch.setattr(
        vae_ops,
        "resolve_h3_vae_operators",
        lambda _device: _operator_set(),
    )
    decoder = _make_decoder()
    forward_globals = type(decoder.transformer_blocks[0].attn).forward.__globals__
    monkeypatch.setitem(forward_globals, "_vit_norm_input", norm_input)

    assert not vae_ops.install_h3_vae_optimizations(
        decoder,
        device=torch.device("meta"),
    )
    assert not hasattr(decoder, "_omni_h3_vae_optimizations_installed")
    assert decoder.transformer_blocks[0].attn.to_qkv.weight.dtype == torch.float32


@pytest.mark.parametrize(
    "break_contract",
    [
        lambda block: delattr(block.attn, "spatial_parallel"),
        lambda block: setattr(block.attn, "spatial_parallel", 2),
        lambda block: setattr(block.attn, "perform_attention", None),
        lambda block: setattr(block, "scale1", nn.Parameter(torch.zeros(7))),
        lambda block: setattr(block.ff, "_compile_forward_enabled", True),
    ],
)
def test_h3_vae_install_rejects_incompatible_remote_contract(monkeypatch, break_contract):
    from vllm_omni.diffusion.models.minimax_h3.ops import vae as vae_ops

    monkeypatch.setattr(
        vae_ops,
        "resolve_h3_vae_operators",
        lambda _device: _operator_set(),
    )
    decoder = _make_decoder()
    break_contract(decoder.transformer_blocks[0])

    assert not vae_ops.install_h3_vae_optimizations(
        decoder,
        device=torch.device("meta"),
    )
    assert decoder.transformer_blocks[0].attn.to_qkv.weight.dtype == torch.float32


def test_h3_vae_dispatch_is_extended_by_adding_an_operator_set(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3.ops.vae import dispatch

    first = _operator_set(supports=lambda _device: False)
    added = _operator_set(supports=lambda device: device.type == "meta")
    monkeypatch.setattr(dispatch, "H3_VAE_OPERATOR_TABLE", (first, added))

    assert dispatch.resolve_h3_vae_operators(torch.device("meta")) is added
    assert dispatch.resolve_h3_vae_operators(torch.device("cpu")) is None


def test_h3_vae_dispatch_selects_supported_cuda_capabilities(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3.ops.vae import dispatch

    platform = Mock()
    platform.is_cuda.return_value = True
    platform.is_available.return_value = True
    monkeypatch.setattr(dispatch, "HAS_TRITON", True)
    monkeypatch.setattr(dispatch, "current_omni_platform", platform)

    for capability in (90, 100, 103):
        platform.get_device_capability.return_value.to_int.return_value = capability
        assert dispatch.resolve_h3_vae_operators(torch.device("cuda:0")) is not None

    for capability in (89, 101, 110):
        platform.get_device_capability.return_value.to_int.return_value = capability
        assert dispatch.resolve_h3_vae_operators(torch.device("cuda:0")) is None
