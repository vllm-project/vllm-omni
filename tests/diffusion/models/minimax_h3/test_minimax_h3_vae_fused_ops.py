# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.minimax_h3.ops import vae as vae_ops

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("is_a5", [True, False])
@pytest.mark.parametrize("opt_in", [None, False, True])
def test_video_vae_fused_operators_require_a5_and_opt_in(mocker, is_a5, opt_in):
    mocker.patch.object(vae_ops, "is_a5", return_value=is_a5)
    install = mocker.patch.object(vae_ops, "_install_npu_operators", return_value=True)
    device = mocker.Mock(spec=torch.device, type="npu")
    additional_config = {} if opt_in is None else {"video_vae_fused_ops": opt_in}
    config = OmniDiffusionConfig(additional_config=additional_config)
    expected = is_a5 and opt_in is True

    with set_current_diffusion_config(config):
        assert vae_ops.install_h3_vae_optimizations(nn.Module(), device=device) is expected
    assert install.call_count == int(expected)


def test_video_vae_fused_operators_reject_non_boolean_config_on_a5(mocker):
    mocker.patch.object(vae_ops, "is_a5", return_value=True)
    install = mocker.patch.object(vae_ops, "_install_npu_operators")
    device = mocker.Mock(spec=torch.device, type="npu")
    config = OmniDiffusionConfig(additional_config={"video_vae_fused_ops": "false"})

    with set_current_diffusion_config(config), pytest.raises(ValueError, match="video_vae_fused_ops must be a boolean"):
        vae_ops.install_h3_vae_optimizations(nn.Module(), device=device)

    install.assert_not_called()


@pytest.mark.parametrize("device_type", ["npu", "cuda", "xpu", "cpu"])
def test_non_a5_decoder_ignores_video_vae_fusion_config(mocker, device_type):
    mocker.patch.object(vae_ops, "is_a5", return_value=False)
    mocker.patch.object(vae_ops, "resolve_h3_vae_operators", return_value=None)
    device = mocker.Mock(spec=torch.device, type=device_type)
    config = OmniDiffusionConfig(additional_config={"video_vae_fused_ops": "false"})

    with set_current_diffusion_config(config):
        assert not vae_ops.install_h3_vae_optimizations(nn.Module(), device=device)


@pytest.mark.parametrize("blocks", [None, nn.ModuleList()])
def test_npu_installer_keeps_reference_for_unsupported_decoder(blocks):
    decoder = nn.Identity()
    decoder.transformer_blocks = blocks
    x = torch.randn(2, 8)

    assert not vae_ops._install_npu_operators(decoder)
    assert "_omni_h3_vae_npu_operators_installed" not in vars(decoder)
    assert decoder(x) is x


@pytest.mark.parametrize("lightvae_decoder_path", [None, "unused-lite"])
def test_npu_vae_operator_installation_preserves_litevae(mocker, lightvae_decoder_path):
    from vllm_omni.diffusion.models.minimax_h3 import lightvae
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

    if lightvae_decoder_path is not None:
        checkpoint = mocker.Mock(kept_layers=[0])
        mocker.patch.object(lightvae, "read_h3_lite_decoder_checkpoint", return_value=checkpoint)
        mocker.patch.object(lightvae, "install_h3_lite_decoder")

    vae = vae_module.MiniMaxH3VideoVAE(
        "unused",
        device=device,
        load_device=torch.device("cpu"),
        lightvae_decoder_path=lightvae_decoder_path,
    )

    if lightvae_decoder_path is None:
        install.assert_called_once_with(vae.model.decoder, device=device)
    else:
        lightvae.install_h3_lite_decoder.assert_called_once()
        install.assert_not_called()


@pytest.mark.parametrize(
    ("additional_config", "expected"),
    [({}, False), ({"video_vae_fused_ops": False}, False), ({"video_vae_fused_ops": True}, True)],
)
def test_standalone_decoder_uses_stage_vae_fusion_config(mocker, tmp_path, additional_config, expected):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3_decoder as decoder_module
    from vllm_omni.diffusion.registry import initialize_model

    def init_video_vae(_path, *, device, **_kwargs):
        decoder = nn.Module()
        vae_ops.install_h3_vae_optimizations(decoder, device=device)
        return decoder

    device = mocker.Mock(spec=torch.device, type="npu")
    mocker.patch.object(decoder_module, "get_local_device", return_value=device)
    mocker.patch.object(decoder_module, "MiniMaxH3VideoVAE", side_effect=init_video_vae)
    mocker.patch.object(decoder_module, "MiniMaxH3AudioVAE")
    mocker.patch.object(vae_ops, "is_a5", return_value=True)
    install = mocker.patch.object(vae_ops, "_install_npu_operators", return_value=True)
    config = OmniDiffusionConfig(
        model=str(tmp_path),
        model_class_name="MiniMaxH3DecoderPipeline",
        stage_id=2,
        additional_config=additional_config,
    )

    initialize_model(config)

    assert install.called is expected


def apply_rotary_pos_emb(x, _rotary):
    return x


@pytest.mark.parametrize(("compile_flag", "fuse_rope"), [(None, True), ("0", True), ("1", False), (" TRUE ", False)])
def test_npu_installer_keeps_native_rope_when_compile_is_requested(monkeypatch, mocker, compile_flag, fuse_rope):
    class Attention(nn.Module):
        def forward(self, x, rotary):
            return apply_rotary_pos_emb(x, rotary)

    flag = "MINIMAX_H3_VAE_DECODER_VIT_ROPE_TORCH_COMPILE"
    if compile_flag is None:
        monkeypatch.delenv(flag, raising=False)
    else:
        monkeypatch.setenv(flag, compile_flag)

    block = nn.Module()
    block.attn = Attention()
    block.norm1 = nn.RMSNorm(8, eps=1e-6)
    decoder = nn.Module()
    decoder.transformer_blocks = nn.ModuleList([block])
    x = torch.randn(2, 8)
    fused = torch.zeros_like(x)
    try_fused = mocker.patch.object(vae_ops, "try_rotary_npu", return_value=fused)

    assert vae_ops._install_npu_operators(decoder)
    assert ("forward" in vars(block.attn)) is fuse_rope
    assert block.attn(x, (x, x)) is (fused if fuse_rope else x)
    assert try_fused.call_count == int(fuse_rope)
    assert "forward" in vars(block.norm1)


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
