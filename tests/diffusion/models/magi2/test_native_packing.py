# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.magi2.configuration_magi2 import (
    Magi2MHCConfig,
    Magi2MoEConfig,
    Magi2PreviewConfig,
)
from vllm_omni.diffusion.models.magi2.layers import MultiModalityRMSNorm
from vllm_omni.diffusion.models.magi2.mh_moe import Magi2MultiHeadMoE
from vllm_omni.diffusion.models.magi2.modeling_magi2 import Magi2PreviewTransformer
from vllm_omni.diffusion.models.magi2.preview_data_proxy import (
    Magi2DataProxy,
    Magi2PreviewDataProxyConfig,
)
from vllm_omni.diffusion.models.magi2.sampler_magi2 import CFGConfig, Magi2PreviewSampler

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


def _tiny_config(params_dtype: torch.dtype = torch.float32, *, moe_layers: int = 1) -> Magi2PreviewConfig:
    # The leading layers are multimodal with MoE, the rest single-modality
    # dense, so every layer belongs to one of two kinds.
    moe_indices = tuple(range(moe_layers))
    return Magi2PreviewConfig(
        num_layers=2 * moe_layers,
        hidden_size=16,
        head_dim=8,
        num_query_groups=2,
        video_in_channels=4,
        audio_in_channels=4,
        text_in_channels=4,
        intermediate_factor=2,
        multimodal_layers=moe_indices,
        params_dtype=params_dtype,
        mhc=Magi2MHCConfig(num_streams=2),
        moe=Magi2MoEConfig(
            num_heads=2,
            num_experts=4,
            top_k=2,
            expert_intermediate_size=8,
            shared_expert_intermediate_size=8,
            modality_shared_expert_intermediate_size=8,
            layers=moe_indices,
        ),
    )


def _tiny_model(
    seed: int = 11,
    params_dtype: torch.dtype = torch.float32,
    *,
    moe_layers: int = 1,
) -> Magi2PreviewTransformer:
    model = Magi2PreviewTransformer(_tiny_config(params_dtype, moe_layers=moe_layers))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype) * 0.02)
        for module in model.modules():
            if isinstance(module, MultiModalityRMSNorm):
                module.weight.zero_()
            elif isinstance(module, Magi2MultiHeadMoE):
                module.router.expert_bias.zero_()
                module.router.expert_bias_ema.zero_()
    return model


def _tiny_sampler(model: torch.nn.Module) -> Magi2PreviewSampler:
    return Magi2PreviewSampler(model, Magi2DataProxy(Magi2PreviewDataProxyConfig(time_channel_dim=8)))


def _sampler_tensors(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return {
        "latent": torch.randn(1, 4, 2, 1, 3, generator=generator),
        "audio_latent": torch.randn(1, 5, 4, generator=generator),
        "txt_feat": torch.randn(1, 3, 4, generator=generator),
        "null_txt_feat": torch.randn(1, 2, 4, generator=generator),
    }


def _longer_text_tensors(seed: int) -> dict[str, torch.Tensor]:
    tensors = _sampler_tensors(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed + 100)
    tensors["txt_feat"] = torch.randn(1, 4, 4, generator=generator)
    return tensors


def test_prepare_model_input_keeps_lengths_on_host() -> None:
    sampler = Magi2PreviewSampler(torch.nn.Identity())
    model_input = sampler.prepare_model_input(**_sampler_tensors(0), t=torch.tensor([500.0]), cfg_config=CFGConfig())

    assert model_input.audio_feat_len == [5, 5]
    assert model_input.txt_feat_len == [3, 2]
    assert model_input.ref_audio_feat_len == [0, 0]
    assert model_input.ref_video_feat_len == [0, 0]

    positive, negative = Magi2PreviewSampler._split_cfg_model_input(model_input)
    assert positive.txt_feat_len == [3]
    assert negative.txt_feat_len == [2]


def test_pre_adapter_embeds_directly_in_checkpoint_dtype() -> None:
    model = _tiny_model(params_dtype=torch.bfloat16)
    packed = torch.randn(6, 4)
    indices = torch.tensor([0, 1]), torch.tensor([2, 3]), torch.tensor([4, 5])

    with torch.no_grad():
        hidden = model.pre_adapter(packed, *indices)

    assert hidden.dtype == torch.bfloat16
    assert hidden.shape == (6, model.config.virtual_width)
