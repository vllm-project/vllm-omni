# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ming Audio's checkpoint-specific reference encoder contract."""

import pytest
import torch

from vllm_omni.model_executor.models.common.ming.audio_vae import AudioVAEConfig
from vllm_omni.model_executor.models.ming_flash_omni.ming_flash_omni_talker import (
    MingFlashOmniTalkerForConditionalGeneration,
)
from vllm_omni.transformers_utils.configs.ming_talker import MingFlashOmniTalkerConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("local", [True, False], ids=["local", "hub"])
@pytest.mark.parametrize("patch_size", [4, -1])
def test_reference_encoder_preserves_native_depth(tmp_path, mocker, local, patch_size):
    backbone = dict(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=6,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=1,
        _attn_implementation="sdpa",
        use_cache=False,
    )
    config = AudioVAEConfig(
        enc_kwargs=dict(backbone=backbone, input_dim=4, hop_size=4, latent_dim=2),
        dec_kwargs=dict(backbone=backbone, output_dim=4, latent_dim=2),
        patch_size=patch_size,
    )
    mocker.patch.object(AudioVAEConfig, "from_pretrained", return_value=config)
    vae_path = tmp_path / "vae"
    if local:
        vae_path.mkdir()
    vae, _ = MingFlashOmniTalkerForConditionalGeneration._init_audio_vae(
        MingFlashOmniTalkerConfig(audio_vae_path=str(vae_path)), str(tmp_path), "unused/model"
    )
    assert vae is not None
    vae.eval()
    visited = []
    for i, layer in enumerate(vae.encoder.encoder.layers):
        layer.register_forward_hook(lambda module, args, output, i=i: visited.append(i))
    with torch.inference_mode():
        latents, _ = vae.encode_latent(
            torch.zeros(1, 16), torch.tensor([16]), generator=torch.Generator().manual_seed(17)
        )
    assert torch.isfinite(latents).all()
    assert visited == list(range(4 if patch_size != -1 else 6))
    # Keep every checkpoint parameter loadable, even the inactive encoder layers.
    assert len(vae.encoder.encoder.layers) == 6
    assert "encoder.encoder.layers.5.self_attn.q_proj.weight" in vae.state_dict()
    assert vae.decoder.decoder.config.num_hidden_layers == 6
    assert config.enc_kwargs["backbone"]["num_hidden_layers"] == 6
