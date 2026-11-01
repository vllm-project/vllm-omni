# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.models.krea2.krea2_transformer import Krea2Transformer2DModel

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.usefixtures("mock_tp")]

_FUSED_TO_CHECKPOINT = {
    ".to_qkv.": (".to_q.", ".to_k.", ".to_v."),
    ".gate_up_proj.": (".gate.", ".up."),
}


def _tiny_model() -> Krea2Transformer2DModel:
    # GQA in the main blocks (4 heads / 2 kv heads) and MHA in the text fusion blocks, like the real checkpoint.
    return Krea2Transformer2DModel(
        in_channels=8,
        num_layers=2,
        attention_head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=96,
        timestep_embed_dim=32,
        text_hidden_dim=32,
        num_text_layers=3,
        text_num_attention_heads=2,
        text_num_key_value_heads=2,
        text_intermediate_size=48,
        num_layerwise_text_blocks=1,
        num_refiner_text_blocks=1,
        axes_dims_rope=(4, 6, 6),
    )


def _diffusers_checkpoint(model: Krea2Transformer2DModel) -> dict[str, torch.Tensor]:
    """Random weights under the unfused diffusers names (to_q/to_k/to_v, ff.gate/ff.up)."""
    generator = torch.Generator().manual_seed(0)
    checkpoint = {}
    for name, param in model.named_parameters():
        fused = next((token for token in _FUSED_TO_CHECKPOINT if token in name), None)
        if fused is None:
            checkpoint[name] = torch.randn(param.shape, generator=generator)
            continue
        module = model.get_submodule(name.rsplit(".", 1)[0])
        if fused == ".to_qkv.":
            sizes = [module.num_heads * module.head_size, *[module.num_kv_heads * module.head_size] * 2]
        else:
            sizes = list(module.output_sizes)
        for shard, size in zip(_FUSED_TO_CHECKPOINT[fused], sizes):
            checkpoint[name.replace(fused, shard)] = torch.randn((size, param.shape[1]), generator=generator)
    return checkpoint


def test_load_weights_stacks_unfused_checkpoint_into_fused_projections():
    model = _tiny_model()
    checkpoint = _diffusers_checkpoint(model)

    loaded = model.load_weights(checkpoint.items())

    assert loaded == {name for name, _ in model.named_parameters()}
    for name, param in model.named_parameters():
        for fused, shards in _FUSED_TO_CHECKPOINT.items():
            if fused in name:
                expected = torch.cat([checkpoint[name.replace(fused, shard)] for shard in shards], dim=0)
                torch.testing.assert_close(param.data, expected, rtol=0, atol=0)


def test_fused_projections_match_unfused_reference():
    model = _tiny_model()
    checkpoint = _diffusers_checkpoint(model)
    model.load_weights(checkpoint.items())

    for prefix, attn_dim in (
        ("transformer_blocks.0", 64),
        ("text_fusion.refiner_blocks.0", 32),
    ):
        x = torch.randn(2, 5, attn_dim)

        attn = model.get_submodule(f"{prefix}.attn")
        query, key, value = attn.to_qkv(x).split([attn.q_dim, attn.kv_dim, attn.kv_dim], dim=-1)
        torch.testing.assert_close(query, F.linear(x, checkpoint[f"{prefix}.attn.to_q.weight"]))
        torch.testing.assert_close(key, F.linear(x, checkpoint[f"{prefix}.attn.to_k.weight"]))
        torch.testing.assert_close(value, F.linear(x, checkpoint[f"{prefix}.attn.to_v.weight"]))

        ff = model.get_submodule(f"{prefix}.ff")
        reference = F.linear(
            F.silu(F.linear(x, checkpoint[f"{prefix}.ff.gate.weight"]))
            * F.linear(x, checkpoint[f"{prefix}.ff.up.weight"]),
            checkpoint[f"{prefix}.ff.down.weight"],
        )
        torch.testing.assert_close(ff.down(ff.act(ff.gate_up_proj(x))), reference)


def test_packed_modules_mapping_matches_stacked_params_mapping():
    model = _tiny_model()
    model.load_weights(_diffusers_checkpoint(model).items())

    derived: dict[str, list[str]] = {}
    for param_name, shard_name, _ in model.stacked_params_mapping:
        derived.setdefault(param_name.strip("."), []).append(shard_name.strip("."))
    assert derived == Krea2Transformer2DModel.packed_modules_mapping
