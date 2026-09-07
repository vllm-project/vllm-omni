import pytest
import torch

import vllm_omni.diffusion.layers.custom_op as custom_op
from vllm_omni.diffusion.layers.mot.mot_layernorm import MoTRMSNorm

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _NpuPlatform:
    is_rocm = staticmethod(lambda: False)
    is_cuda = staticmethod(lambda: False)
    is_npu = staticmethod(lambda: True)
    is_xpu = staticmethod(lambda: False)
    is_musa = staticmethod(lambda: False)


@pytest.mark.parametrize("use_mot_routing", [False, True])
def test_mot_rms_norm_npu_falls_back_to_native(monkeypatch, use_mot_routing: bool):
    monkeypatch.setattr(
        custom_op,
        "current_omni_platform",
        _NpuPlatform(),
    )

    layer = MoTRMSNorm(hidden_size=8)
    assert layer._forward_method.__func__ is MoTRMSNorm.forward_npu

    x = torch.randn(6, 8)
    text_indices = torch.tensor([0, 2, 4]) if use_mot_routing else None
    vae_indices = torch.tensor([1, 3, 5]) if use_mot_routing else None

    expected = layer.forward_native(x, text_indices, vae_indices)
    actual = layer(x, text_indices, vae_indices)

    torch.testing.assert_close(actual, expected)
