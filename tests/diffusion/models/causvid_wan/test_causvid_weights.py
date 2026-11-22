# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CausVid Wan-native → Diffusers key remap."""

from __future__ import annotations

import torch

from vllm_omni.diffusion.models.causvid_wan.weights import (
    iter_causvid_diffusers_weights,
    remap_causvid_key,
)


def test_remap_core_block_and_head_keys():
    assert remap_causvid_key("model.blocks.0.self_attn.q.weight") == "blocks.0.attn1.to_q.weight"
    assert remap_causvid_key("blocks.2.cross_attn.o.bias") == "blocks.2.attn2.to_out.0.bias"
    assert remap_causvid_key("blocks.1.ffn.0.weight") == "blocks.1.ffn.net.0.proj.weight"
    assert remap_causvid_key("blocks.1.modulation") == "blocks.1.scale_shift_table"
    assert remap_causvid_key("blocks.0.norm3.weight") == "blocks.0.norm2.weight"
    assert remap_causvid_key("blocks.0.norm3.bias") == "blocks.0.norm2.bias"
    assert remap_causvid_key("head.head.weight") == "proj_out.weight"
    assert remap_causvid_key("patch_embedding.weight") == "patch_embedding.weight"
    assert remap_causvid_key("freqs") is None


def test_iter_generator_dict():
    raw = {
        "model.blocks.0.self_attn.q.weight": torch.zeros(4, 4),
        "freqs": torch.zeros(8),
        "generator.ignored": "not-a-tensor",
    }
    mapped = dict(iter_causvid_diffusers_weights(raw))
    assert "blocks.0.attn1.to_q.weight" in mapped
    assert "freqs" not in mapped
