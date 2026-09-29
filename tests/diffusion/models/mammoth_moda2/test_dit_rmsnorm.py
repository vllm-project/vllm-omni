# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MammothModa2 DiT norms run on the shared RMSNorm instead of transformers' Qwen2RMSNorm."""

from types import SimpleNamespace

import pytest
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.layers.norm import RMSNorm
from vllm_omni.diffusion.models.mammoth_moda2.mammothmoda2_dit_model import (
    SimpleQFormerImageRefiner,
    Transformer2DModel,
)
from vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit import MammothModa2DiTPipeline

from .test_dit_attention import _SDPA_CONFIG

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_dit_norms_are_shared_rmsnorm_with_their_eps():
    # norm_eps differs from the fixed 1e-5 of the QK norms and the reinitialized caption
    # embedder, so a site that lost its eps shows up here.
    with set_current_diffusion_config(_SDPA_CONFIG):
        transformer = Transformer2DModel(
            hidden_size=48,
            num_layers=1,
            num_refiner_layers=1,
            num_attention_heads=6,
            num_kv_heads=2,
            multiple_of=8,
            norm_eps=1e-6,
            axes_dim_rope=(4, 2, 2),
            text_feat_dim=16,
        )
    refiner = SimpleQFormerImageRefiner(
        hidden_size=32, output_hidden_size=48, num_queries=4, num_layers=1, norm_eps=1e-6
    )

    norms = [m for m in (*transformer.modules(), *refiner.modules()) if isinstance(m, (RMSNorm, Qwen2RMSNorm))]
    assert norms and all(isinstance(m, RMSNorm) for m in norms)

    block = transformer.layers[0]
    assert block.attn.norm_q.variance_epsilon == block.attn.norm_k.variance_epsilon == 1e-5
    assert block.norm1.norm.variance_epsilon == block.norm2.variance_epsilon == 1e-6
    assert transformer.context_refiner[0].norm1.variance_epsilon == 1e-6
    assert transformer.time_caption_embed.caption_embedder[0].variance_epsilon == 1e-6
    assert refiner.layers[0]["ln_ffn"].variance_epsilon == 1e-6

    MammothModa2DiTPipeline._reinit_caption_embedder(SimpleNamespace(gen_transformer=transformer), 32)
    caption_norm = transformer.time_caption_embed.caption_embedder[0]
    assert isinstance(caption_norm, RMSNorm) and caption_norm.variance_epsilon == 1e-5


def test_shared_rmsnorm_loads_qwen2_rmsnorm_weights():
    RMSNorm(120, eps=1e-5).load_state_dict(Qwen2RMSNorm(120, eps=1e-5).state_dict())
