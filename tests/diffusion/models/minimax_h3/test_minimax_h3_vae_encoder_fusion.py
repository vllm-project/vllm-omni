# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import copy
from types import MethodType
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from tests.diffusion.models.minimax_h3.h3_vae_encoder_replica import checkpoint_pad_chain, make_encoder
from vllm_omni.diffusion.models.minimax_h3 import vae as vae_module
from vllm_omni.diffusion.models.minimax_h3.ops.vae import dispatch
from vllm_omni.diffusion.models.minimax_h3.ops.vae import encoder as encoder_ops
from vllm_omni.diffusion.models.minimax_h3.ops.vae.causal_reflect_pad import try_causal_reflect_pad_exact

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

ENV = "VLLM_OMNI_H3_VAE_ENCODER_FUSION"


def _use_operator(monkeypatch, pad_op=checkpoint_pad_chain):
    calls = []

    def recording(source, **pads):
        calls.append(pads)
        return pad_op(source, **pads)

    monkeypatch.setattr(encoder_ops, "resolve_h3_vae_operators", lambda _device: Mock(causal_reflect_pad=recording))
    return calls


def _install(encoder):
    return encoder_ops.install_h3_vae_encoder_fusion(encoder, device=torch.device("cpu"))


def _patched(encoder):
    return [module for module in encoder.modules() if "forward" in vars(module)]


def _installed_pair(monkeypatch, pad_op=checkpoint_pad_chain, time_isolated=True):
    calls = _use_operator(monkeypatch, pad_op)
    reference = make_encoder(time_isolated)
    fused = copy.deepcopy(reference)
    for module in fused.modules():
        if hasattr(module, "spatial_parallel"):
            module.spatial_parallel = 0
    with torch.device("meta"):
        assert _install(fused)
    return calls, reference, fused


def _inputs():
    torch.manual_seed(3)
    video = torch.randn(1, 3, 40, 48, 40)
    return video[:, :, 3:20, 8:40, 4:36], video[:, :, 7:8, :32, :32], torch.randn(2, 3, 5, 32, 16)


@pytest.mark.parametrize(
    ("time_isolated", "pad_op", "calls_per_input"),
    [(True, checkpoint_pad_chain, 16), (False, checkpoint_pad_chain, 16), (True, lambda _source, **_pads: None, 31)],
)
def test_encoder_fusion_matches_checkpoint_forward(monkeypatch, time_isolated, pad_op, calls_per_input):
    calls, reference, fused = _installed_pair(monkeypatch, pad_op, time_isolated)
    forward = fused.down[0].block[0].forward
    assert _install(fused) and fused.down[0].block[0].forward is forward
    with torch.inference_mode():
        for x in _inputs():
            assert torch.equal(fused(x), reference(x))
    assert len(calls) == 3 * calls_per_input
    assert {"front": 2, "top": 0, "bottom": 1, "left": 0, "right": 1, "apply_silu": False} in calls
    assert sum(call["apply_silu"] for call in calls) == 3 * 13


def test_encoder_fusion_keeps_autograd_conditioned_and_training_calls(monkeypatch):
    calls, reference, fused = _installed_pair(monkeypatch)
    hidden, x = torch.randn(1, 32, 3, 8, 8), _inputs()[0]
    actual = fused(x)
    assert actual.requires_grad and torch.equal(actual, reference(x))
    assert len(calls) == 16 and not any(call["apply_silu"] for call in calls)
    calls.clear()
    with torch.inference_mode():
        assert torch.equal(fused.down[0].block[0](hidden, torch.zeros(1)), reference.down[0].block[0](hidden))
        assert len(calls) == 2 and not any(call["apply_silu"] for call in calls)
        fused.train()
        assert torch.equal(fused(x), reference(x))
    assert len(calls) == 2 + 16


def test_encoder_fusion_falls_back_for_unbatched_and_spatial_parallel_calls(monkeypatch):
    calls, reference, fused = _installed_pair(monkeypatch)
    frames, hidden, x = torch.randn(3, 5, 16, 16), torch.randn(32, 5, 8, 8), _inputs()[0]
    with torch.inference_mode():
        assert torch.equal(fused.conv_in(frames), reference.conv_in(frames))
        assert torch.equal(fused.down[0].downsample(hidden), reference.down[0].downsample(hidden))
        assert not calls
        for path in ("conv_in", "down.0.block.1.conv2", "down.1.downsample.conv", "conv_out"):
            fused.get_submodule(path).spatial_parallel = True
        assert torch.equal(fused(x), reference(x))
    assert len(calls) == 16 - 4


def test_unpadded_shortcut_skips_the_copy_only_for_aligned_contiguous_inputs(monkeypatch):
    _calls, reference, fused = _installed_pair(monkeypatch)
    numel = 32 * 3 * 8 * 8
    buffer = torch.randn(numel + 256)
    start = (-buffer.data_ptr() % 512) // buffer.element_size()
    aligned = buffer[start : start + numel].view(1, 32, 3, 8, 8)
    misaligned = buffer[start + 1 : start + 1 + numel].view(1, 32, 3, 8, 8)
    inputs = ((aligned, False), (misaligned, True), (aligned.transpose(3, 4), True))
    shortcut = fused.down[1].block[0].nin_shortcut
    with torch.inference_mode():
        expected = [reference.down[1].block[0].nin_shortcut(x) for x, _ in inputs]
        checkpoint_forward = Mock(wraps=type(shortcut).forward)
        monkeypatch.setattr(type(shortcut), "forward", checkpoint_forward)
        for (x, copied), y in zip(inputs, expected):
            checkpoint_forward.reset_mock()
            assert torch.equal(shortcut(x), y) and checkpoint_forward.called is copied


def test_encoder_fusion_runs_forward_hooks_like_the_checkpoint(monkeypatch):
    calls, reference, fused = _installed_pair(monkeypatch)
    names = {module: name for root in (reference, fused) for name, module in root.named_modules()}
    hooked = ("down.0.block.0.norm1", "down.0.block.0.conv1", "down.0.downsample.conv", "norm_out", "conv_out")
    seen, order = [], []
    for path in hooked:
        fused.get_submodule(path).register_forward_hook(lambda module, _args, _out: seen.append(names[module]))
    x = _inputs()[0]
    with torch.inference_mode():
        expected = reference(x)
        assert torch.equal(fused(x), expected)
        assert sorted(seen) == sorted(hooked) and len(calls) == 16
        handle = torch.nn.modules.module.register_module_forward_pre_hook(
            lambda module, _args: order.append(names[module])
        )
        try:
            reference(x)
            checkpoint_order = order.copy()
            order.clear()
            assert torch.equal(fused(x), expected)
        finally:
            handle.remove()
    assert order == checkpoint_order


def _own_forward(module):
    module.forward = MethodType(type(module).forward, module)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda encoder: setattr(encoder, "use_fused_norm", True),
        lambda encoder: setattr(encoder.down[1].block[0], "use_fused_norm", True),
        lambda encoder: setattr(encoder.down[0].block[1].conv1, "pad_mode", "replicate"),
        lambda encoder: setattr(encoder.conv_out, "causal", False),
        lambda encoder: setattr(encoder.conv_in, "groups", 3),
        lambda encoder: setattr(encoder.conv_in, "padding", (1, 2, 2)),
        lambda encoder: setattr(encoder.norm_out, "spatial_parallel", True),
        lambda encoder: setattr(encoder.down[0].downsample, "pad_mode", "constant"),
        lambda encoder: setattr(encoder.down[0].downsample, "space_stride", 4),
        lambda encoder: setattr(encoder.down[1].downsample.conv, "padding", (1, 1, 0)),
        lambda encoder: setattr(encoder.down[1].block[0].nin_shortcut, "padding_mode", "zeros"),
        lambda encoder: setattr(encoder.down[0].block[0].norm2, "affine", False),
        lambda encoder: setattr(encoder.conv_in, "_apply_padding", lambda x: F.pad(x, (1, 1, 1, 1, 1, 1))),
        lambda encoder: encoder.to("meta"),
        lambda encoder: _own_forward(encoder.down[0].block[0].norm1),
        lambda encoder: _own_forward(encoder.down[0].block[0].conv1),
        lambda encoder: _own_forward(encoder.down[1].block[0].nin_shortcut),
        lambda encoder: _own_forward(encoder.down[0].downsample),
    ],
)
def test_encoder_fusion_rejects_unknown_contracts(monkeypatch, mutate):
    _use_operator(monkeypatch)
    encoder = make_encoder()
    mutate(encoder)
    patched = _patched(encoder)
    assert not _install(encoder)
    assert _patched(encoder) == patched


@pytest.mark.parametrize("probed", ["conv_in", "norm_out"])
@pytest.mark.parametrize(
    ("error", "declines"),
    [(error("probe"), True) for error in (RuntimeError, NotImplementedError, TypeError, ValueError, AttributeError)]
    + [(error("probe"), False) for error in (KeyError, AssertionError, torch.OutOfMemoryError, torch.AcceleratorError)],
)
def test_encoder_fusion_declines_only_on_probe_contract_errors(monkeypatch, probed, error, declines):
    _use_operator(monkeypatch)
    logger = Mock()
    monkeypatch.setattr(encoder_ops, "logger", logger)
    encoder = make_encoder()

    def raise_error(*_args):
        raise error

    if probed == "conv_in":
        encoder.conv_in._apply_padding = raise_error
    else:
        monkeypatch.setattr(type(encoder.norm_out), "forward", raise_error)
    if declines:
        assert not _install(encoder)
        logger.debug.assert_called_once()
        assert error in logger.debug.call_args.args
    else:
        with pytest.raises(type(error)):
            _install(encoder)
    assert not _patched(encoder)


@pytest.mark.parametrize("operators", [None, Mock(causal_reflect_pad=None)])
def test_encoder_fusion_requires_an_encoder_operator(monkeypatch, operators):
    monkeypatch.setattr(encoder_ops, "resolve_h3_vae_operators", lambda _device: operators)
    encoder = make_encoder()
    assert not _install(encoder) and not _patched(encoder)


def test_encoder_operator_is_registered_for_sm90_only(monkeypatch):
    platform = Mock(**{"is_cuda.return_value": True, "is_available.return_value": True})
    monkeypatch.setattr(dispatch, "HAS_TRITON", True)
    monkeypatch.setattr(dispatch, "current_omni_platform", platform)
    monkeypatch.setattr(dispatch.torch.version, "hip", None)
    selected = {}
    for capability in (89, 90, 100, 103):
        platform.get_device_capability.return_value.to_int.return_value = capability
        selected[capability] = dispatch.resolve_h3_vae_operators(torch.device("cuda:0")).causal_reflect_pad
    assert selected == {89: None, 90: try_causal_reflect_pad_exact, 100: None, 103: None}


@pytest.mark.parametrize(
    ("value", "expected"),
    {None: True, "": True, "1": True, "true": True, " ON ": True, " 0 ": False, "False": False, "off": False}.items(),
)
def test_encoder_fusion_env_switch(monkeypatch, value, expected):
    monkeypatch.delenv(ENV, raising=False)
    if value is not None:
        monkeypatch.setenv(ENV, value)
    assert vae_module._encoder_fusion_enabled() is expected


def test_encoder_fusion_env_switch_rejects_unknown_values(monkeypatch):
    monkeypatch.setenv(ENV, "2")
    with pytest.raises(ValueError, match=ENV):
        vae_module._encoder_fusion_enabled()


def _fake_remote(*names):
    remote = torch.nn.Module()
    remote.model = torch.nn.ModuleDict({name: torch.nn.Module() for name in ("encoder", "quant_conv", *names)})
    return remote


@pytest.mark.parametrize(
    ("value", "mode", "installed"),
    [("1", {}, True), ("1", {"encode_only": True}, True), ("0", {}, False), ("1", {"decode_only": True}, False)],
)
def test_video_vae_installs_encoder_fusion_unless_disabled(monkeypatch, value, mode, installed):
    monkeypatch.setenv(ENV, value)
    config = {"latent_channels": 1, "latents_mean": [0.0], "latents_std": [1.0]}
    monkeypatch.setattr(vae_module, "_load_component_config", lambda _path: config)
    monkeypatch.setattr(vae_module, "_check_trust_remote_code", lambda *_args: None)
    monkeypatch.setattr(vae_module, "_load_remote_component", lambda _path, _config, **_kwargs: _fake_remote("decoder"))
    monkeypatch.setattr(vae_module, "_load_video_vae_encoder", lambda _path, _config: _fake_remote())
    monkeypatch.setattr(vae_module, "install_h3_vae_optimizations", Mock(return_value=True))
    install = Mock(return_value=True)
    monkeypatch.setattr(vae_module, "install_h3_vae_encoder_fusion", install)
    video_vae = vae_module.MiniMaxH3VideoVAE("unused", device=torch.device("cpu"), **mode)
    if installed:
        install.assert_called_once_with(video_vae.model.encoder, device=torch.device("cpu"))
    else:
        install.assert_not_called()
