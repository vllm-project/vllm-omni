# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for F2: action-head primitive + DiT modules.

Validates shapes, parameter-name compatibility with upstream checkpoint
tensors (so F5 load_weights works without remapping), and numerical
determinism with a seeded RNG.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Embodiment-conditioned MLP primitives
# ---------------------------------------------------------------------------


def test_category_specific_linear_param_names_and_shapes():
    from vllm_omni.diffusion.models.gr00t.modeling.action_head_modules import (
        CategorySpecificLinear,
    )

    layer = CategorySpecificLinear(num_categories=4, input_dim=8, hidden_dim=16)
    state = dict(layer.named_parameters())
    assert set(state.keys()) == {"W", "b"}
    assert state["W"].shape == (4, 8, 16)
    assert state["b"].shape == (4, 16)


def test_category_specific_mlp_param_names():
    from vllm_omni.diffusion.models.gr00t.modeling.action_head_modules import (
        CategorySpecificMLP,
    )

    mlp = CategorySpecificMLP(
        num_categories=4, input_dim=8, hidden_dim=16, output_dim=12
    )
    names = set(dict(mlp.named_parameters()).keys())
    assert names == {"layer1.W", "layer1.b", "layer2.W", "layer2.b"}


def test_multi_embodiment_action_encoder_param_names():
    from vllm_omni.diffusion.models.gr00t.modeling.action_head_modules import (
        MultiEmbodimentActionEncoder,
    )

    enc = MultiEmbodimentActionEncoder(
        action_dim=132, hidden_size=64, num_embodiments=32
    )
    names = set(dict(enc.named_parameters()).keys())
    # pos_encoding has no params (parameter-free sinusoidal)
    assert names == {"W1.W", "W1.b", "W2.W", "W2.b", "W3.W", "W3.b"}


def test_embodiment_mlp_shapes_and_determinism():
    from vllm_omni.diffusion.models.gr00t.modeling.action_head_modules import (
        MultiEmbodimentActionEncoder,
    )

    B, T, action_dim, hidden, num_emb = 2, 40, 132, 64, 32
    torch.manual_seed(0)
    enc = MultiEmbodimentActionEncoder(
        action_dim=action_dim, hidden_size=hidden, num_embodiments=num_emb
    )

    torch.manual_seed(123)
    actions = torch.randn(B, T, action_dim)
    timesteps = torch.tensor([0, 7], dtype=torch.long)
    cat_ids = torch.tensor([1, 5], dtype=torch.long)

    out1 = enc(actions, timesteps, cat_ids)
    out2 = enc(actions, timesteps, cat_ids)
    assert out1.shape == (B, T, hidden)
    torch.testing.assert_close(out1, out2, rtol=0, atol=0)


def test_embodiment_mlp_distinct_embodiments_produce_distinct_outputs():
    from vllm_omni.diffusion.models.gr00t.modeling.action_head_modules import (
        MultiEmbodimentActionEncoder,
    )

    torch.manual_seed(0)
    enc = MultiEmbodimentActionEncoder(
        action_dim=16, hidden_size=32, num_embodiments=4
    )
    actions = torch.randn(1, 5, 16)
    timesteps = torch.tensor([3], dtype=torch.long)
    out_a = enc(actions, timesteps, torch.tensor([0]))
    out_b = enc(actions, timesteps, torch.tensor([1]))
    assert not torch.allclose(out_a, out_b)


# ---------------------------------------------------------------------------
# DiT stack
# ---------------------------------------------------------------------------


@pytest.fixture
def dit_kwargs():
    return dict(
        num_attention_heads=4,
        attention_head_dim=16,
        output_dim=12,
        num_layers=4,
        dropout=0.0,
        attention_bias=True,
        activation_fn="gelu-approximate",
        norm_type="ada_norm",
        max_num_positional_embeddings=64,
        final_dropout=True,
        positional_embeddings="sinusoidal",
        interleave_self_attention=True,
        cross_attention_dim=24,
    )


def test_basic_transformer_block_uses_diffusers_attention_param_names(dit_kwargs):
    from vllm_omni.diffusion.models.gr00t.modeling.dit import BasicTransformerBlock

    block = BasicTransformerBlock(
        dim=64,
        num_attention_heads=dit_kwargs["num_attention_heads"],
        attention_head_dim=dit_kwargs["attention_head_dim"],
        cross_attention_dim=dit_kwargs["cross_attention_dim"],
        norm_type="ada_norm",
        positional_embeddings="sinusoidal",
        num_positional_embeddings=64,
        final_dropout=True,
        attention_bias=True,
        activation_fn="gelu-approximate",
    )
    names = set(dict(block.named_parameters()).keys())
    # Sanity: attn1 should expose diffusers-style to_q / to_k / to_v / to_out
    assert any(n.startswith("attn1.to_q.") for n in names)
    assert any(n.startswith("attn1.to_k.") for n in names)
    assert any(n.startswith("attn1.to_v.") for n in names)
    assert any(n.startswith("attn1.to_out.0.") for n in names)


def test_dit_shapes_and_determinism(dit_kwargs):
    from vllm_omni.diffusion.models.gr00t.modeling.dit import DiT

    torch.manual_seed(0)
    dit = DiT(**dit_kwargs)
    inner_dim = dit_kwargs["num_attention_heads"] * dit_kwargs["attention_head_dim"]

    B, T_a, T_v = 2, 10, 7
    torch.manual_seed(123)
    hidden = torch.randn(B, T_a, inner_dim)
    enc = torch.randn(B, T_v, dit_kwargs["cross_attention_dim"])
    t = torch.tensor([3, 5], dtype=torch.long)

    out1 = dit(hidden, enc, timestep=t)
    out2 = dit(hidden, enc, timestep=t)
    assert out1.shape == (B, T_a, dit_kwargs["output_dim"])
    torch.testing.assert_close(out1, out2, rtol=0, atol=0)


def test_alternate_vl_dit_image_mask_required(dit_kwargs):
    from vllm_omni.diffusion.models.gr00t.modeling.dit import AlternateVLDiT

    torch.manual_seed(0)
    dit = AlternateVLDiT(attend_text_every_n_blocks=2, **dit_kwargs)
    B, T_a, T_v = 2, 10, 7
    inner_dim = dit_kwargs["num_attention_heads"] * dit_kwargs["attention_head_dim"]
    hidden = torch.randn(B, T_a, inner_dim)
    enc = torch.randn(B, T_v, dit_kwargs["cross_attention_dim"])
    t = torch.tensor([3, 5], dtype=torch.long)

    with pytest.raises(ValueError, match="image_mask"):
        dit(hidden, enc, timestep=t)

    image_mask = torch.zeros(B, T_v, dtype=torch.bool)
    image_mask[:, :3] = True
    backbone_mask = torch.ones(B, T_v, dtype=torch.bool)
    out = dit(
        hidden,
        enc,
        timestep=t,
        image_mask=image_mask,
        backbone_attention_mask=backbone_mask,
    )
    assert out.shape == (B, T_a, dit_kwargs["output_dim"])


def test_alternate_vl_dit_requires_interleave_self_attention(dit_kwargs):
    from vllm_omni.diffusion.models.gr00t.modeling.dit import AlternateVLDiT

    dit_kwargs = {**dit_kwargs, "interleave_self_attention": False}
    dit = AlternateVLDiT(**dit_kwargs)
    B, T_a, T_v = 1, 4, 4
    inner_dim = dit_kwargs["num_attention_heads"] * dit_kwargs["attention_head_dim"]
    hidden = torch.randn(B, T_a, inner_dim)
    enc = torch.randn(B, T_v, dit_kwargs["cross_attention_dim"])
    t = torch.tensor([1], dtype=torch.long)
    with pytest.raises(ValueError, match="interleave_self_attention"):
        dit(
            hidden,
            enc,
            timestep=t,
            image_mask=torch.ones(B, T_v, dtype=torch.bool),
            backbone_attention_mask=torch.ones(B, T_v, dtype=torch.bool),
        )


def test_self_attention_transformer_shapes_and_determinism():
    from vllm_omni.diffusion.models.gr00t.modeling.dit import (
        SelfAttentionTransformer,
    )

    torch.manual_seed(0)
    sat = SelfAttentionTransformer(
        num_attention_heads=4,
        attention_head_dim=16,
        num_layers=2,
        positional_embeddings="sinusoidal",
        max_num_positional_embeddings=32,
        final_dropout=True,
        activation_fn="gelu-approximate",
        attention_bias=True,
        dropout=0.0,
    )
    B, T = 2, 8
    inner_dim = 4 * 16
    torch.manual_seed(123)
    x = torch.randn(B, T, inner_dim)

    out1 = sat(x)
    out2 = sat(x)
    assert out1.shape == (B, T, inner_dim)
    torch.testing.assert_close(out1, out2, rtol=0, atol=0)


def test_dit_load_state_dict_roundtrip(dit_kwargs):
    """Save → load roundtrip succeeds, validating module parameter names
    remain compatible with `load_state_dict(strict=True)`."""
    from vllm_omni.diffusion.models.gr00t.modeling.dit import DiT

    torch.manual_seed(0)
    src = DiT(**dit_kwargs)
    torch.manual_seed(1)
    dst = DiT(**dit_kwargs)

    missing, unexpected = dst.load_state_dict(src.state_dict(), strict=True)
    assert not missing
    assert not unexpected

    # Assert dst now matches src on a random input.
    inner = dit_kwargs["num_attention_heads"] * dit_kwargs["attention_head_dim"]
    torch.manual_seed(99)
    h = torch.randn(1, 5, inner)
    e = torch.randn(1, 4, dit_kwargs["cross_attention_dim"])
    t = torch.tensor([2], dtype=torch.long)
    torch.testing.assert_close(src(h, e, timestep=t), dst(h, e, timestep=t))
