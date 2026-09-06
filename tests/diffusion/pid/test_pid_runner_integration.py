# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Plan-B Runner-layer PiD orchestration.

All heavyweight modules (PidDecoder / PidNet / Gemma) are mocked; the tests
exercise gating, passthrough bookkeeping, and output replacement only.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.pid import PidDecodeConfig
from vllm_omni.diffusion.pid.runner_integration import (
    PidPassthrough,
    _resolve_pid_config,
    decode_stepwise_output,
    init_pid_decoder_on,
    maybe_pid_passthrough,
    stepwise_pid_active,
    validate_pid_override,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _cfg(**kw):
    kw.setdefault("enabled", True)
    kw.setdefault("checkpoint_path", "/tmp/pid.pth")
    kw.setdefault("gemma_model", "/tmp/gemma")
    return PidDecodeConfig(**kw)


class FluxPipeline:
    """Minimal pipeline stand-in whose class name hits LATENT_FORMS["FluxPipeline"]."""

    vae_scale_factor = 8

    def __init__(self, decoder=None, config=None):
        self._pid_decoder = decoder
        self._pid_config = config
        self._resident_modules = []


_Pipe = FluxPipeline  # alias used by the unregistered-family tests


class _Req:
    def __init__(self, output_type="pil", pid_decode=None, prompt="a cat", height=512, width=512, **sp):
        self.request_id = "r0"
        self.prompt = prompt
        self.sampling_params = SimpleNamespace(
            output_type=output_type,
            pid_decode=pid_decode,
            height=height,
            width=width,
            latents=None,
            image_latent=None,
            strength=None,
            **sp,
        )


# -- _resolve_pid_config ------------------------------------------------------


def test_resolve_pid_config_none():
    assert _resolve_pid_config(SimpleNamespace(pid_decode=None)) is None


def test_resolve_pid_config_dict():
    cfg = _resolve_pid_config(SimpleNamespace(pid_decode={"enabled": True, "scale": 2}))
    assert isinstance(cfg, PidDecodeConfig) and cfg.scale == 2


def test_resolve_pid_config_bad_type():
    with pytest.raises(TypeError, match="pid_decode"):
        _resolve_pid_config(SimpleNamespace(pid_decode=123))


# -- init_pid_decoder_on -------------------------------------------------------


def test_init_disabled_is_noop(mocker):
    patch = mocker.patch("vllm_omni.diffusion.pid.runner_integration.PidDecoder")
    init_pid_decoder_on(_Pipe(), SimpleNamespace(pid_decode=_cfg(enabled=False)))
    patch.assert_not_called()


def test_init_mounts_decoder_and_declares_resident(mocker):
    decoder_cls = mocker.patch("vllm_omni.diffusion.pid.runner_integration.PidDecoder")
    pipe = _Pipe()
    init_pid_decoder_on(pipe, SimpleNamespace(pid_decode=_cfg(), enforce_eager=True))
    decoder_cls.assert_called_once()
    assert decoder_cls.call_args.kwargs["enforce_eager"] is True
    # backbone comes from the LatentForm table, not from the pipeline
    assert decoder_cls.call_args.kwargs["backbone"] == "flux"
    decoder_cls.return_value.load_weights.assert_called_once()
    assert pipe._pid_decoder is decoder_cls.return_value
    assert "_pid_decoder" in pipe._resident_modules


def test_init_unregistered_family_warns_and_skips(mocker, caplog):
    decoder_cls = mocker.patch("vllm_omni.diffusion.pid.runner_integration.PidDecoder")
    pipe = object.__new__(torch.nn.Module)  # any class not in LATENT_FORMS
    init_pid_decoder_on(pipe, SimpleNamespace(pid_decode=_cfg()))
    decoder_cls.assert_not_called()
    assert not hasattr(pipe, "_pid_decoder")


# -- maybe_pid_passthrough gating ----------------------------------------------


def test_gate_no_decoder_no_request_returns_none():
    assert maybe_pid_passthrough(_Pipe(), [_Req()], SimpleNamespace(pid_decode=None)) is None


def test_gate_request_without_enable_raises():
    pipe = _Pipe()
    with pytest.raises(RuntimeError, match="--pid-enable"):
        maybe_pid_passthrough(pipe, [_Req(pid_decode={"enabled": True})], SimpleNamespace(pid_decode=None))


def test_gate_request_disabled_falls_back():
    pipe = _Pipe(decoder=object(), config=_cfg())
    out = maybe_pid_passthrough(pipe, [_Req(pid_decode={"enabled": False})], SimpleNamespace(pid_decode=_cfg()))
    assert out is None


def test_gate_user_latent_output_returns_none():
    pipe = _Pipe(decoder=object(), config=_cfg())
    out = maybe_pid_passthrough(pipe, [_Req(output_type="latent")], SimpleNamespace(pid_decode=_cfg()))
    assert out is None


def test_gate_img2img_falls_back():
    pipe = _Pipe(decoder=object(), config=_cfg())
    req = _Req()
    req.sampling_params.latents = torch.zeros(1)
    assert maybe_pid_passthrough(pipe, [req], SimpleNamespace(pid_decode=_cfg())) is None
    req2 = _Req()
    req2.sampling_params.strength = 0.7
    assert maybe_pid_passthrough(pipe, [req2], SimpleNamespace(pid_decode=_cfg())) is None


def test_gate_enabled_returns_passthrough():
    pipe = _Pipe(decoder=object(), config=_cfg())
    pt = maybe_pid_passthrough(pipe, [_Req()], SimpleNamespace(pid_decode=_cfg()))
    assert isinstance(pt, PidPassthrough)


# -- PidPassthrough force / restore / decode ------------------------------------


def test_force_and_restore_output_type():
    pt = PidPassthrough(_Pipe(), None, None, _cfg())
    reqs = [_Req(output_type="pil"), _Req(output_type="latent")]
    pt.force_latent_output(reqs)
    assert all(r.sampling_params.output_type == "latent" for r in reqs)
    pt.restore_output_type(reqs)
    assert [r.sampling_params.output_type for r in reqs] == ["pil", "latent"]


def test_decode_outputs_replaces_latent_with_image(mocker):
    mocker.patch(
        "vllm_omni.diffusion.pid.runner_integration.decode_with_pid",
        return_value=torch.zeros(1, 3, 512, 512),
    )
    pt = PidPassthrough.__new__(PidPassthrough)
    pt.pipeline = _Pipe()
    pt.form = SimpleNamespace(
        backbone="qwenimage",
        to_x0=lambda latent, h, w, v, pipeline=None: (torch.zeros(1, 16, 64, 64), 512, 512),
    )
    pt.decoder = mocker.Mock()
    pt.config = _cfg()
    pt._saved_output_types = []

    # Flux-family packed tokens: [B, T, 4C] = [1, 1024, 64]
    out = DiffusionOutput(output=torch.zeros(1, 1024, 64))
    results = pt.decode_outputs([out], [_Req()])
    assert results[0].output.shape == (1, 3, 512, 512)


def test_decode_outputs_keeps_error_outputs(mocker):
    pt = PidPassthrough.__new__(PidPassthrough)
    pt.pipeline = _Pipe()
    pt.form = SimpleNamespace(to_x0=lambda *a: (torch.zeros(1, 16, 64, 64), 512, 512))
    pt.decoder = mocker.Mock()
    pt.config = _cfg()
    pt._saved_output_types = []
    out = DiffusionOutput(output=torch.zeros(1), error="boom")
    results = pt.decode_outputs([out], [_Req()])
    assert results[0].error == "boom"


# -- stepwise path ----------------------------------------------------------------


def test_stepwise_pid_active_gates():
    pipe = _Pipe(decoder=object(), config=_cfg())
    state = SimpleNamespace(
        sampling=SimpleNamespace(output_type="pil", pid_decode=None, latents=None, image_latent=None, strength=None),
        prompt="a cat",
    )
    assert stepwise_pid_active(pipe, state) is True
    state.sampling.output_type = "latent"
    assert stepwise_pid_active(pipe, state) is False
    state.sampling.output_type = "pil"
    state.sampling.pid_decode = {"enabled": False}
    assert stepwise_pid_active(pipe, state) is False


def test_decode_stepwise_output_replaces_output(mocker):
    from vllm_omni.diffusion.pid import latent_forms

    pipe = _Pipe(decoder=mocker.Mock(), config=_cfg())
    pipe.vae_scale_factor = 8
    # lookup_latent_form is imported into runner_integration's namespace; patch there.
    mocker.patch(
        "vllm_omni.diffusion.pid.runner_integration.lookup_latent_form",
        return_value=latent_forms.LATENT_FORMS["FluxPipeline"],
    )
    mocker.patch(
        "vllm_omni.diffusion.pid.runner_integration.decode_with_pid",
        return_value=torch.zeros(1, 3, 512, 512),
    )
    state = SimpleNamespace(
        sampling=SimpleNamespace(height=512, width=512, pid_decode=None),
        prompt="a cat",
    )
    # Flux-family packed tokens: [B, T, 4C] = [1, 1024, 64]
    result = DiffusionOutput(output=torch.zeros(1, 1024, 64))
    out = decode_stepwise_output(pipe, state, result)
    assert out.output.shape == (1, 3, 512, 512)


def test_decode_stepwise_output_noop_without_decoder():
    pipe = _Pipe()
    state = SimpleNamespace(sampling=SimpleNamespace(height=512, width=512, pid_decode=None), prompt="x")
    result = DiffusionOutput(output=torch.zeros(1))
    assert decode_stepwise_output(pipe, state, result) is result


# -- validate_pid_override -------------------------------------------------------


@pytest.mark.parametrize(
    "ov",
    [
        None,
        {},
        {"enabled": True},
        {"enabled": False, "scale": 2, "num_steps": 4, "seed": 7, "degrade_sigma": 0.1},
    ],
)
def test_validate_pid_override_accepts_valid(ov):
    validate_pid_override(ov)


@pytest.mark.parametrize(
    "ov, match",
    [
        ("bad", "must be a dict"),
        ({"extra": 1}, "unknown key 'extra'"),
        ({"scales": 4}, "unknown key 'scales'"),
        ({"scale": -1}, "scale must be >= 1"),
        ({"scale": "4"}, "scale must be int"),
        ({"scale": True}, "scale must be int"),
        ({"num_steps": 0}, "num_steps must be >= 1"),
        ({"num_steps": 4.0}, "num_steps must be int"),
        ({"enabled": 1}, "enabled must be bool"),
        ({"degrade_sigma": "x"}, "degrade_sigma must be"),
    ],
)
def test_validate_pid_override_rejects_invalid(ov, match):
    with pytest.raises(ValueError, match=match):
        validate_pid_override(ov)


def test_validate_pid_override_called_in_batch_path():
    """Unknown keys fail the request in maybe_pid_passthrough (not silently filtered)."""
    pipe = _Pipe(decoder=object(), config=_cfg())
    reqs = [_Req(pid_decode={"extra": 1})]
    with pytest.raises(ValueError, match="unknown key 'extra'"):
        maybe_pid_passthrough(pipe, reqs, None)


def test_stepwise_enabled_true_without_decoder_raises():
    """Streaming path converges with batch path: explicit enabled=True raises."""
    pipe = _Pipe()
    state = SimpleNamespace(
        sampling=SimpleNamespace(
            output_type="pil", pid_decode={"enabled": True}, latents=None, image_latent=None, strength=None
        ),
        prompt="a cat",
    )
    with pytest.raises(RuntimeError, match="--pid-enable"):
        stepwise_pid_active(pipe, state)
