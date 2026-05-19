# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for F3: Gr00tN1d7ActionHead.

Shape, determinism with a seeded RNG, and embodiment sensitivity.  Numerical
parity vs Isaac-GR00T's Gr00tPolicy is F10.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def small_config():
    """A small Gr00tN1d7Config with the real DiT/VL-self-attn shapes shrunk
    so tests stay CPU-friendly.  Backbone overlay disabled by passing
    model_name=""."""
    from vllm_omni.transformers_utils.configs.gr00t import Gr00tN1d7Config

    return Gr00tN1d7Config(
        model_name="",
        hidden_size=64,
        input_embedding_dim=64,
        backbone_embedding_dim=128,
        max_action_dim=16,
        max_state_dim=16,
        action_horizon=4,
        max_num_embodiments=4,
        state_history_length=1,
        max_seq_len=64,
        add_pos_embed=True,
        use_vlln=True,
        use_alternate_vl_dit=True,
        attend_text_every_n_blocks=2,
        diffusion_model_cfg={
            "num_layers": 4,
            "num_attention_heads": 4,
            "attention_head_dim": 16,
            "output_dim": 64,
            "norm_type": "ada_norm",
            "interleave_self_attention": True,
            "final_dropout": True,
            "dropout": 0.0,
            "positional_embeddings": None,
        },
        vl_self_attention_cfg={
            "num_layers": 2,
            "num_attention_heads": 4,
            "attention_head_dim": 32,
            "dropout": 0.0,
            "final_dropout": True,
            "positional_embeddings": None,
        },
        use_vl_self_attention=True,
        num_inference_timesteps=4,
        num_timestep_buckets=1000,
    )


def _build_inputs(cfg, B: int = 2, S: int = 5, seed: int = 1234):
    torch.manual_seed(seed)
    vl_embeds = torch.randn(B, S, cfg.backbone_embedding_dim)
    vl_attn_mask = torch.ones(B, S, dtype=torch.bool)
    image_mask = torch.zeros(B, S, dtype=torch.bool)
    image_mask[:, : S // 2] = True
    state = torch.randn(B, cfg.state_history_length, cfg.max_state_dim)
    embodiment_id = torch.tensor([0, 1], dtype=torch.long)[:B]
    return vl_embeds, vl_attn_mask, image_mask, state, embodiment_id


def test_action_head_submodule_names(small_config):
    from vllm_omni.diffusion.models.gr00t.modeling.action_head import (
        Gr00tN1d7ActionHead,
    )

    head = Gr00tN1d7ActionHead(small_config)
    names = {n for n, _ in head.named_modules() if n}
    for required in (
        "state_encoder",
        "action_encoder",
        "action_decoder",
        "model",
        "vlln",
        "vl_self_attention",
        "position_embedding",
    ):
        assert required in names, f"missing submodule: {required}"


def test_action_head_get_action_shape_and_determinism(small_config):
    from vllm_omni.diffusion.models.gr00t.modeling.action_head import (
        Gr00tN1d7ActionHead,
    )

    torch.manual_seed(0)
    head = Gr00tN1d7ActionHead(small_config)
    vl, vl_mask, img_mask, state, emb = _build_inputs(small_config)

    torch.manual_seed(42)
    out1 = head.get_action(
        vl_embeds=vl,
        vl_attn_mask=vl_mask,
        image_mask=img_mask,
        state=state,
        embodiment_id=emb,
    )
    torch.manual_seed(42)
    out2 = head.get_action(
        vl_embeds=vl,
        vl_attn_mask=vl_mask,
        image_mask=img_mask,
        state=state,
        embodiment_id=emb,
    )
    assert out1.shape == (2, small_config.action_horizon, small_config.max_action_dim)
    torch.testing.assert_close(out1, out2, rtol=0, atol=0)


def test_action_head_embodiment_sensitivity(small_config):
    from vllm_omni.diffusion.models.gr00t.modeling.action_head import (
        Gr00tN1d7ActionHead,
    )

    torch.manual_seed(0)
    head = Gr00tN1d7ActionHead(small_config)
    vl, vl_mask, img_mask, state, _ = _build_inputs(small_config, B=1, seed=7)

    torch.manual_seed(99)
    out_a = head.get_action(
        vl_embeds=vl,
        vl_attn_mask=vl_mask,
        image_mask=img_mask,
        state=state,
        embodiment_id=torch.tensor([0], dtype=torch.long),
    )
    torch.manual_seed(99)
    out_b = head.get_action(
        vl_embeds=vl,
        vl_attn_mask=vl_mask,
        image_mask=img_mask,
        state=state,
        embodiment_id=torch.tensor([1], dtype=torch.long),
    )
    assert not torch.allclose(out_a, out_b)


def test_action_head_rejects_wrong_state_history(small_config):
    from vllm_omni.diffusion.models.gr00t.modeling.action_head import (
        Gr00tN1d7ActionHead,
    )

    head = Gr00tN1d7ActionHead(small_config)
    vl, vl_mask, img_mask, state, emb = _build_inputs(small_config)
    state_bad = torch.randn(state.shape[0], 3, state.shape[-1])
    with pytest.raises(ValueError, match="state history mismatch"):
        head.get_action(
            vl_embeds=vl,
            vl_attn_mask=vl_mask,
            image_mask=img_mask,
            state=state_bad,
            embodiment_id=emb,
        )


def test_action_head_runs_with_plain_dit(small_config):
    """Without use_alternate_vl_dit, head wires the plain DiT and omits the
    image_mask / backbone_attention_mask kwargs."""
    from vllm_omni.diffusion.models.gr00t.modeling.action_head import (
        Gr00tN1d7ActionHead,
    )

    small_config.use_alternate_vl_dit = False
    torch.manual_seed(0)
    head = Gr00tN1d7ActionHead(small_config)
    vl, vl_mask, img_mask, state, emb = _build_inputs(small_config)

    torch.manual_seed(1)
    out = head.get_action(
        vl_embeds=vl,
        vl_attn_mask=vl_mask,
        image_mask=img_mask,
        state=state,
        embodiment_id=emb,
    )
    assert out.shape == (2, small_config.action_horizon, small_config.max_action_dim)


def test_action_head_disables_vl_self_attn_when_configured(small_config):
    from vllm_omni.diffusion.models.gr00t.modeling.action_head import (
        Gr00tN1d7ActionHead,
    )
    from torch import nn

    small_config.use_vl_self_attention = False
    head = Gr00tN1d7ActionHead(small_config)
    assert isinstance(head.vl_self_attention, nn.Identity)
