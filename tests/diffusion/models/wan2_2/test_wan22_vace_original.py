# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the original (non-diffusers) Wan2.2 VACE loading helpers.

Covers the two pure functions that the original-format path relies on:

* ``convert_original_to_diffusers_key`` — renames original checkpoint keys
  into diffusers form. It runs *unconditionally* in the shared
  ``WanTransformer3DModel.load_weights`` (inherited by T2V / I2V / VACE).
* ``convert_original_vace_config`` — translates the original ``VaceWanModel``
  config field names into the diffusers keys the transformer ``__init__`` wants.
"""

import pytest

from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_vace import convert_original_vace_config
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import convert_original_to_diffusers_key

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


# (original key, expected diffusers key)
_ORIGINAL_TO_DIFFUSERS = [
    ("blocks.0.self_attn.q.weight", "blocks.0.attn1.to_q.weight"),
    ("blocks.0.self_attn.k.bias", "blocks.0.attn1.to_k.bias"),
    ("blocks.0.self_attn.v.weight", "blocks.0.attn1.to_v.weight"),
    ("blocks.0.self_attn.o.weight", "blocks.0.attn1.to_out.0.weight"),
    ("blocks.7.cross_attn.q.weight", "blocks.7.attn2.to_q.weight"),
    ("blocks.7.cross_attn.o.bias", "blocks.7.attn2.to_out.0.bias"),
    ("blocks.0.ffn.0.weight", "blocks.0.ffn.net.0.proj.weight"),
    ("blocks.0.ffn.2.weight", "blocks.0.ffn.net.2.weight"),
    ("blocks.0.norm3.weight", "blocks.0.norm2.weight"),
    ("blocks.0.modulation", "blocks.0.scale_shift_table"),
    ("head.modulation", "scale_shift_table"),
    ("head.head.weight", "proj_out.weight"),
    ("text_embedding.0.weight", "condition_embedder.text_embedder.linear_1.weight"),
    ("text_embedding.2.bias", "condition_embedder.text_embedder.linear_2.bias"),
    ("time_embedding.0.weight", "condition_embedder.time_embedder.linear_1.weight"),
    ("time_embedding.2.weight", "condition_embedder.time_embedder.linear_2.weight"),
    ("time_projection.1.weight", "condition_embedder.time_proj.weight"),
    ("vace_blocks.0.before_proj.weight", "vace_blocks.0.proj_in.weight"),
    ("vace_blocks.0.after_proj.bias", "vace_blocks.0.proj_out.bias"),
    ("vace_blocks.0.self_attn.q.weight", "vace_blocks.0.attn1.to_q.weight"),
]

# Real diffusers-format keys that MUST pass through untouched. If any rename rule
# accidentally matched one of these, it would silently corrupt every existing
# diffusers Wan (T2V / I2V / VACE) checkpoint at load time.
_DIFFUSERS_NOOP_KEYS = [
    "patch_embedding.weight",
    "condition_embedder.text_embedder.linear_1.weight",
    "condition_embedder.text_embedder.linear_2.weight",
    "condition_embedder.time_embedder.linear_1.weight",
    "condition_embedder.time_proj.weight",
    "blocks.0.attn1.to_q.weight",
    "blocks.0.attn1.to_k.weight",
    "blocks.0.attn1.to_v.weight",
    "blocks.0.attn1.to_out.0.weight",
    "blocks.0.attn2.to_q.weight",
    "blocks.0.norm2.weight",
    "blocks.0.ffn.net.0.proj.weight",
    "blocks.0.ffn.net.2.weight",
    "blocks.0.scale_shift_table",
    "vace_blocks.0.proj_in.weight",
    "vace_blocks.0.proj_out.weight",
    "proj_out.weight",
    "scale_shift_table",
]


@pytest.mark.parametrize(("original_name", "diffusers_name"), _ORIGINAL_TO_DIFFUSERS)
def test_convert_original_to_diffusers_key_maps_original_names(original_name: str, diffusers_name: str) -> None:
    assert convert_original_to_diffusers_key(original_name) == diffusers_name


@pytest.mark.parametrize("diffusers_name", _DIFFUSERS_NOOP_KEYS)
def test_convert_original_to_diffusers_key_is_noop_on_diffusers_names(diffusers_name: str) -> None:
    # The regression guard: already-diffusers keys come back byte-for-byte.
    assert convert_original_to_diffusers_key(diffusers_name) == diffusers_name


@pytest.mark.parametrize("qk_norm_name", ["blocks.0.attn1.norm_q.weight", "blocks.0.attn1.norm_k.weight"])
def test_convert_original_to_diffusers_key_leaves_qk_norm_untouched(qk_norm_name: str) -> None:
    # ``.q.weight``/``.k.weight`` are anchored on a leading dot, so the qk-norm
    # params (``..._q.weight``) must NOT be rewritten into ``to_q``/``to_k``.
    assert convert_original_to_diffusers_key(qk_norm_name) == qk_norm_name


def test_convert_original_vace_config_translates_original_fields() -> None:
    original_config = {
        "dim": 16,
        "num_heads": 4,
        "in_dim": 8,
        "out_dim": 8,
        "ffn_dim": 32,
        "freq_dim": 256,
        "num_layers": 2,
        "eps": 1e-6,
        "vace_layers": [0, 1],
        "vace_in_dim": 12,
        "model_type": "vace",  # an omitted-on-purpose field falls to model defaults
    }

    assert convert_original_vace_config(original_config) == {
        "num_attention_heads": 4,
        "attention_head_dim": 4,  # dim // num_heads = 16 // 4
        "in_channels": 8,
        "out_channels": 8,
        "ffn_dim": 32,
        "freq_dim": 256,
        "num_layers": 2,
        "eps": 1e-6,
        "vace_layers": [0, 1],
        "vace_in_channels": 12,
    }
