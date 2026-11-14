# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.minimax_h3.ops import vae as vae_ops

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("is_a5", [True, False])
def test_video_vae_fused_operators_are_enabled_on_a5_only(mocker, is_a5):
    mocker.patch.object(vae_ops, "is_a5", return_value=is_a5)
    install = mocker.patch.object(vae_ops, "_install_npu_operators", return_value=True)
    device = mocker.Mock(spec=torch.device, type="npu")

    assert vae_ops.install_h3_vae_optimizations(nn.Module(), device=device) is is_a5
    assert install.call_count == int(is_a5)


def test_native_video_vae_installs_operators_without_mxfp8(mocker):
    from vllm_omni.diffusion.models.minimax_h3 import vae as vae_module

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Module()

    class FakeRemote(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeModel()

    mocker.patch.object(vae_module, "_load_component_config", return_value={})
    mocker.patch.object(vae_module, "_check_trust_remote_code")
    mocker.patch.object(vae_module, "_load_remote_component", return_value=FakeRemote())
    mocker.patch.object(vae_module, "install_temporal_stream_patches")
    mocker.patch.object(vae_module.MiniMaxH3VideoVAE, "_build_residency_stagers")
    mocker.patch.object(vae_module.MiniMaxH3VideoVAE, "_tile_gather_reuse_enabled", return_value=False)
    install = mocker.patch.object(vae_module, "install_h3_vae_optimizations")
    device = mocker.Mock(spec=torch.device, type="npu")

    vae = vae_module.MiniMaxH3VideoVAE("unused", device=device, load_device=torch.device("cpu"))

    install.assert_called_once_with(vae.model.decoder, device=device)


def test_npu_installer_preserves_cpu_norm_fallback_and_is_idempotent():
    block = nn.Module()
    block.attn = nn.Module()
    block.attn.norm_q = nn.RMSNorm(8, eps=1e-6, elementwise_affine=False)
    block.attn.norm_k = nn.RMSNorm(8, eps=1e-6, elementwise_affine=False)
    block.norm1 = nn.RMSNorm(8, eps=1e-6)
    block.norm2 = nn.RMSNorm(8, eps=1e-6)
    decoder = nn.Module()
    decoder.transformer_blocks = nn.ModuleList([block])
    x = torch.randn(2, 8)
    expected = block.attn.norm_q(x)

    assert vae_ops._install_npu_operators(decoder)
    assert vae_ops._install_npu_operators(decoder)
    torch.testing.assert_close(block.attn.norm_q(x), expected, rtol=0, atol=0)
    assert "_omni_unit_gamma" not in block.attn.norm_q.state_dict()
