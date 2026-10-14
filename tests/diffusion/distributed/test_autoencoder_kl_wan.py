import pytest
import torch

from vllm_omni.diffusion.distributed.autoencoders import autoencoder_kl_wan as wan_vae_module
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import OmniAutoencoderKLWan

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummyOmniAutoencoderKLWan(OmniAutoencoderKLWan):
    def __init__(self, *, dtype: torch.dtype):
        torch.nn.Module.__init__(self)
        self.register_parameter("dummy_weight", torch.nn.Parameter(torch.ones(1, dtype=dtype)))


def test_wan_vae_execution_context_handles_fp32():
    model = _DummyOmniAutoencoderKLWan(dtype=torch.float32)
    with model._execution_context():
        output = model.dummy_weight + 1
    assert output.dtype == torch.float32


def test_wan_vae_execution_context_handles_bf16():
    model = _DummyOmniAutoencoderKLWan(dtype=torch.bfloat16)
    with model._execution_context():
        output = model.dummy_weight + 1
    assert output.dtype == torch.bfloat16


def test_wan_vae_execution_context_uses_platform_autocast(mocker):
    sentinel = object()
    platform = mocker.Mock()
    platform.create_autocast_context.return_value = sentinel
    mocker.patch.object(wan_vae_module, "current_omni_platform", platform)

    model = _DummyOmniAutoencoderKLWan(dtype=torch.bfloat16)

    assert model._execution_context() is sentinel
    platform.create_autocast_context.assert_called_once_with(
        device_type=model.dummy_weight.device.type,
        dtype=torch.bfloat16,
        enabled=True,
    )
