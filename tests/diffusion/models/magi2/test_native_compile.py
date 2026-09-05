# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest
import torch

import vllm_omni.diffusion.models.magi2.layers as layers_module
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
    Magi2PackedLayout,
    Magi2PreviewDataProxyConfig,
)
from vllm_omni.diffusion.models.magi2.sampler_magi2 import CFGConfig, Magi2PreviewSampler

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


def _tiny_config(params_dtype: torch.dtype = torch.float32) -> Magi2PreviewConfig:
    # Layer 0 is multimodal with MoE, layer 1 is single-modality dense.
    return Magi2PreviewConfig(
        num_layers=2,
        hidden_size=16,
        head_dim=8,
        num_query_groups=2,
        video_in_channels=4,
        audio_in_channels=4,
        text_in_channels=4,
        intermediate_factor=2,
        multimodal_layers=(0,),
        params_dtype=params_dtype,
        mhc=Magi2MHCConfig(num_streams=2),
        moe=Magi2MoEConfig(
            num_heads=2,
            num_experts=4,
            top_k=2,
            expert_intermediate_size=8,
            shared_expert_intermediate_size=8,
            modality_shared_expert_intermediate_size=8,
            layers=(0,),
        ),
    )


def _tiny_model(seed: int = 11, params_dtype: torch.dtype = torch.float32) -> Magi2PreviewTransformer:
    model = Magi2PreviewTransformer(_tiny_config(params_dtype))
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


def test_shared_layout_matches_fresh_packing_bitwise() -> None:
    sampler = _tiny_sampler(_tiny_model())
    layout = Magi2PackedLayout()
    timesteps = (torch.tensor([900.0]), torch.tensor([450.0]))

    with torch.no_grad():
        reused = [
            sampler.forward(
                sampler.prepare_model_input(**_sampler_tensors(seed), t=t, cfg_config=CFGConfig(), layout=layout)
            )
            for seed, t in enumerate(timesteps)
        ]
        fresh = [
            sampler.forward(sampler.prepare_model_input(**_sampler_tensors(seed), t=t, cfg_config=CFGConfig()))
            for seed, t in enumerate(timesteps)
        ]

    for (video_a, audio_a), (video_b, audio_b) in zip(reused, fresh, strict=True):
        assert torch.equal(video_a, video_b)
        assert torch.equal(audio_a, audio_b)
    assert not torch.equal(reused[0][0], reused[1][0])


def test_shared_layout_builds_token_metadata_once(monkeypatch: pytest.MonkeyPatch) -> None:
    sampler = _tiny_sampler(_tiny_model())
    layout = Magi2PackedLayout()
    counts = {"nonzero": 0, "dispatcher": 0, "rope": 0}

    real_nonzero = torch.nonzero
    real_dispatcher_init = layers_module.ModalityDispatcher.__init__
    real_rope_forward = layers_module.ElementWiseFourierEmbed.forward

    def counting_nonzero(*args, **kwargs):
        counts["nonzero"] += 1
        return real_nonzero(*args, **kwargs)

    def counting_dispatcher_init(self, *args, **kwargs):
        counts["dispatcher"] += 1
        real_dispatcher_init(self, *args, **kwargs)

    def counting_rope_forward(self, *args, **kwargs):
        counts["rope"] += 1
        return real_rope_forward(self, *args, **kwargs)

    monkeypatch.setattr(torch, "nonzero", counting_nonzero)
    monkeypatch.setattr(layers_module.ModalityDispatcher, "__init__", counting_dispatcher_init)
    monkeypatch.setattr(layers_module.ElementWiseFourierEmbed, "forward", counting_rope_forward)

    with torch.no_grad():
        sampler.forward(
            sampler.prepare_model_input(
                **_sampler_tensors(0), t=torch.tensor([900.0]), cfg_config=CFGConfig(), layout=layout
            )
        )
        first_step = dict(counts)
        sampler.forward(
            sampler.prepare_model_input(
                **_sampler_tensors(1), t=torch.tensor([450.0]), cfg_config=CFGConfig(), layout=layout
            )
        )

    assert first_step == {"nonzero": 3, "dispatcher": 1, "rope": 1}
    assert counts == first_step
    assert layout.sequence is not None
    assert layout.tokens is not None


def test_cfg_branches_carry_their_own_layout() -> None:
    sampler = Magi2PreviewSampler(torch.nn.Identity())
    layouts = (Magi2PackedLayout(), Magi2PackedLayout())
    model_input = sampler.prepare_model_input(**_sampler_tensors(0), t=torch.tensor([500.0]), cfg_config=CFGConfig())

    positive, negative = Magi2PreviewSampler._split_cfg_model_input(model_input, layouts)

    assert positive.layout is layouts[0]
    assert negative.layout is layouts[1]


def test_pre_adapter_embeds_directly_in_checkpoint_dtype() -> None:
    model = _tiny_model(params_dtype=torch.bfloat16)
    packed = torch.randn(6, 4)
    indices = torch.tensor([0, 1]), torch.tensor([2, 3]), torch.tensor([4, 5])

    with torch.no_grad():
        hidden = model.pre_adapter(packed, *indices)

    assert hidden.dtype == torch.bfloat16
    assert hidden.shape == (6, model.config.virtual_width)
