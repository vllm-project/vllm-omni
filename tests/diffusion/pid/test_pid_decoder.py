# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.diffusion.pid import PidDecodeConfig, get_pid_net_config
from vllm_omni.diffusion.pid.config import PID_SAMPLING_CONFIG
from vllm_omni.diffusion.pid.decoder import PidDecoder

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _decoder(mocker, config=None, backbone="qwenimage", enforce_eager=False):
    mocker.patch("vllm_omni.diffusion.pid.decoder.get_local_device", return_value=torch.device("cpu"))
    return PidDecoder(config or PidDecodeConfig(), backbone=backbone, enforce_eager=enforce_eager)


def test_load_weights_constructs_model_and_loads_checkpoint(mocker):
    """Enabled loading path: build PidNet+Gemma, load checkpoint, eval, resident."""
    model_cls = mocker.patch("vllm_omni.diffusion.pid.decoder.PidInferenceModel")
    load_ckpt = mocker.patch("vllm_omni.diffusion.pid.decoder.load_pid_checkpoint")

    config = PidDecodeConfig(
        enabled=True,
        checkpoint_path="/tmp/pid.pth",
        gemma_model="/tmp/gemma",
        precision="float16",
    )
    decoder = _decoder(mocker, config)
    decoder.load_weights()

    model = decoder._model
    assert model is not None
    model_cls.assert_called_once_with(
        net_kwargs=get_pid_net_config("qwenimage"),
        gemma_model_id="/tmp/gemma",
        sampling_overrides=dict(PID_SAMPLING_CONFIG),
        precision="float16",
        enforce_eager=False,
    )
    load_ckpt.assert_called_once_with(model, "/tmp/pid.pth", backbone="qwenimage")
    model.eval.assert_called_once()
    model.to.assert_called_once_with(decoder.device)


def test_load_weights_honors_enforce_eager(mocker):
    """PiD compile follows the model's --enforce-eager (eager in, eager out)."""
    model_cls = mocker.patch("vllm_omni.diffusion.pid.decoder.PidInferenceModel")
    mocker.patch("vllm_omni.diffusion.pid.decoder.load_pid_checkpoint")
    decoder = _decoder(mocker, PidDecodeConfig(enabled=True), enforce_eager=True)
    decoder.load_weights()
    assert model_cls.call_args.kwargs["enforce_eager"] is True


def test_load_weights_idempotent(mocker):
    """Repeated load_weights() does not rebuild the model."""
    mocker.patch("vllm_omni.diffusion.pid.decoder.PidInferenceModel")
    mocker.patch("vllm_omni.diffusion.pid.decoder.load_pid_checkpoint")

    decoder = _decoder(mocker)
    decoder._model = mocker.Mock()  # already loaded
    decoder.load_weights()  # early return since _model is set
    decoder.load_weights()

    import vllm_omni.diffusion.pid.decoder as decoder_mod

    assert decoder_mod.PidInferenceModel.call_count == 0


def test_decode_forwards_to_generate_samples(mocker):
    """decode() forwards resolved args to the PiD sampling entry point."""
    decoder = _decoder(mocker)
    decoder._model = mocker.Mock()
    decoder._model.generate_samples_from_batch = mocker.Mock(return_value=torch.zeros(1, 3, 1024, 1024))

    lq_latent = torch.zeros(1, 4, 64, 64)
    out = decoder.decode(lq_latent, "a cat", (1024, 1024), num_steps=2, seed=7, degrade_sigma=0.1)

    assert out is not None
    decoder._model.generate_samples_from_batch.assert_called_once_with(
        lq_latent=lq_latent,
        caption="a cat",
        output_size=(1024, 1024),
        degrade_sigma=0.1,
        num_steps=2,
        seed=7,
    )


def test_decode_uses_config_defaults_when_args_none(mocker):
    """Without explicit args, decode() falls back to config defaults."""
    config = PidDecodeConfig(num_steps=3, seed=11, degrade_sigma=0.5)
    decoder = _decoder(mocker, config)
    decoder._model = mocker.Mock()
    decoder._model.generate_samples_from_batch = mocker.Mock(return_value=torch.zeros(1, 3, 64, 64))

    decoder.decode(torch.zeros(1, 4, 8, 8), "a cat", (64, 64))

    kwargs = decoder._model.generate_samples_from_batch.call_args.kwargs
    assert kwargs["num_steps"] == 3
    assert kwargs["seed"] == 11
    assert kwargs["degrade_sigma"] == 0.5
