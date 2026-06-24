# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structural tests for the Krea 2 transformer and utilities.

Verifies parameter naming, weight-name compatibility with diffusers
checkpoints, and correct shapes for the attention / text-fusion modules.
"""

import os

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_HIDDEN = 128
_HEADS = 4
_KV_HEADS = 2
_HEAD_DIM = _HIDDEN // _HEADS
_INTERMEDIATE = 256
_TEXT_DIM = 64
_TEXT_LAYERS = 3
_TEXT_HEADS = 2
_TEXT_KV_HEADS = 2
_TEXT_INTERMEDIATE = 128
_IN_CHANNELS = 16
_NUM_BLOCKS = 2


@pytest.fixture(autouse=True)
def _init_distributed():
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29503")
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="env://",
    )
    initialize_model_parallel()
    yield
    cleanup_dist_env_and_memory()


def _param_names(module) -> set[str]:
    return {name for name, _ in module.named_parameters()}


def test_krea2_rmsnorm_zero_centered():
    from vllm_omni.diffusion.models.krea2.krea2_transformer import Krea2RMSNorm

    norm = Krea2RMSNorm(32)
    assert torch.allclose(norm.weight, torch.zeros(32))
    x = torch.randn(2, 4, 32)
    out = norm(x)
    assert out.shape == x.shape


def test_krea2_attention_separate_qkv():
    from vllm_omni.diffusion.models.krea2.krea2_transformer import Krea2Attention

    attn = Krea2Attention(_HIDDEN, _HEADS, _KV_HEADS)
    params = _param_names(attn)

    assert "to_q.weight" in params
    assert "to_k.weight" in params
    assert "to_v.weight" in params
    assert "to_gate.weight" in params
    assert "norm_q.weight" in params
    assert "norm_k.weight" in params
    assert "to_out.0.weight" in params
    assert "to_q.bias" not in params
    assert "to_k.bias" not in params


def test_krea2_swiglu_no_bias():
    from vllm_omni.diffusion.models.krea2.krea2_transformer import Krea2SwiGLU

    ff = Krea2SwiGLU(_HIDDEN, _INTERMEDIATE)
    params = _param_names(ff)

    assert "gate.weight" in params
    assert "up.weight" in params
    assert "down.weight" in params
    assert "gate.bias" not in params


def test_krea2_text_fusion_block_structure():
    from vllm_omni.diffusion.models.krea2.krea2_transformer import Krea2TextFusionBlock

    block = Krea2TextFusionBlock(
        _TEXT_DIM, _TEXT_HEADS, _TEXT_KV_HEADS, _TEXT_INTERMEDIATE, eps=1e-5
    )
    params = _param_names(block)

    assert "norm1.weight" in params
    assert "norm2.weight" in params
    assert "attn.to_q.weight" in params
    assert "ff.gate.weight" in params


def test_krea2_text_fusion_forward():
    from vllm_omni.diffusion.models.krea2.krea2_transformer import Krea2TextFusion

    fusion = Krea2TextFusion(
        num_text_layers=_TEXT_LAYERS,
        dim=_TEXT_DIM,
        num_heads=_TEXT_HEADS,
        num_kv_heads=_TEXT_KV_HEADS,
        intermediate_size=_TEXT_INTERMEDIATE,
        num_layerwise_blocks=1,
        num_refiner_blocks=1,
        eps=1e-5,
    )
    B, S = 2, 8
    x = torch.randn(B, S, _TEXT_LAYERS, _TEXT_DIM)
    out = fusion(x)
    assert out.shape == (B, S, _TEXT_DIM)

    params = _param_names(fusion)
    assert "projector.weight" in params
    assert "layerwise_blocks.0.norm1.weight" in params
    assert "refiner_blocks.0.norm1.weight" in params


def test_krea2_transformer_block_params():
    from vllm_omni.diffusion.models.krea2.krea2_transformer import Krea2TransformerBlock

    block = Krea2TransformerBlock(
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        num_heads=_HEADS,
        num_kv_heads=_KV_HEADS,
        norm_eps=1e-5,
    )
    params = _param_names(block)

    assert "scale_shift_table" in params
    assert "norm1.weight" in params
    assert "norm2.weight" in params
    assert "attn.to_q.weight" in params
    assert "ff.gate.weight" in params


def test_krea2_final_layer_has_bias():
    from vllm_omni.diffusion.models.krea2.krea2_transformer import Krea2FinalLayer

    final = Krea2FinalLayer(_HIDDEN, _IN_CHANNELS, eps=1e-5)
    params = _param_names(final)

    assert "linear.weight" in params
    assert "linear.bias" in params
    assert "scale_shift_table" in params
    assert "norm.weight" in params


def test_krea2_transformer_full_param_names():
    """All parameter names in the full model should match diffusers checkpoint keys."""
    from vllm_omni.diffusion.models.krea2.krea2_transformer import Krea2Transformer2DModel

    model = Krea2Transformer2DModel(
        in_channels=_IN_CHANNELS,
        num_layers=_NUM_BLOCKS,
        attention_head_dim=_HEAD_DIM,
        num_attention_heads=_HEADS,
        num_key_value_heads=_KV_HEADS,
        intermediate_size=_INTERMEDIATE,
        timestep_embed_dim=32,
        text_hidden_dim=_TEXT_DIM,
        num_text_layers=_TEXT_LAYERS,
        text_num_attention_heads=_TEXT_HEADS,
        text_num_key_value_heads=_TEXT_KV_HEADS,
        text_intermediate_size=_TEXT_INTERMEDIATE,
        num_layerwise_text_blocks=1,
        num_refiner_text_blocks=1,
        axes_dims_rope=(8, 12, 12),
        rope_theta=1000.0,
    )
    params = _param_names(model)

    expected_prefixes = [
        "img_in.weight",
        "img_in.bias",
        "time_embed.linear_1.weight",
        "time_embed.linear_1.bias",
        "time_embed.linear_2.weight",
        "time_embed.linear_2.bias",
        "time_mod_proj.weight",
        "time_mod_proj.bias",
        "text_fusion.layerwise_blocks.0.norm1.weight",
        "text_fusion.layerwise_blocks.0.attn.to_q.weight",
        "text_fusion.layerwise_blocks.0.attn.to_out.0.weight",
        "text_fusion.layerwise_blocks.0.ff.gate.weight",
        "text_fusion.projector.weight",
        "text_fusion.refiner_blocks.0.norm1.weight",
        "txt_in.norm.weight",
        "txt_in.linear_1.weight",
        "txt_in.linear_1.bias",
        "txt_in.linear_2.weight",
        "txt_in.linear_2.bias",
        "transformer_blocks.0.scale_shift_table",
        "transformer_blocks.0.norm1.weight",
        "transformer_blocks.0.attn.to_q.weight",
        "transformer_blocks.0.attn.to_k.weight",
        "transformer_blocks.0.attn.to_v.weight",
        "transformer_blocks.0.attn.to_gate.weight",
        "transformer_blocks.0.attn.norm_q.weight",
        "transformer_blocks.0.attn.norm_k.weight",
        "transformer_blocks.0.attn.to_out.0.weight",
        "transformer_blocks.0.ff.gate.weight",
        "transformer_blocks.0.ff.up.weight",
        "transformer_blocks.0.ff.down.weight",
        "final_layer.scale_shift_table",
        "final_layer.norm.weight",
        "final_layer.linear.weight",
        "final_layer.linear.bias",
    ]
    for name in expected_prefixes:
        assert name in params, f"Expected param '{name}' not found in model. Got: {sorted(params)}"


def test_preprocess_pack_unpack_roundtrip():
    from vllm_omni.diffusion.models.krea2.preprocess_krea2 import (
        pack_latents,
        unpack_latents,
    )

    B, C, H, W = 2, 4, 8, 8
    latents = torch.randn(B, C, H, W)
    packed = pack_latents(latents, B, C, H, W, patch_size=2)
    assert packed.shape == (B, (H // 2) * (W // 2), C * 4)

    unpacked = unpack_latents(packed, H * 8, W * 8, vae_scale_factor=8, patch_size=2)
    assert unpacked.shape == (B, C, 1, H, W)


def test_prepare_position_ids_shape():
    from vllm_omni.diffusion.models.krea2.preprocess_krea2 import prepare_position_ids

    text_len = 10
    gh, gw = 4, 6
    ids = prepare_position_ids(text_len, gh, gw, device=torch.device("cpu"))
    assert ids.shape == (text_len + gh * gw, 3)
    assert torch.all(ids[:text_len] == 0)


def test_registry_has_krea2():
    from vllm_omni.diffusion.registry import _DIFFUSION_MODELS, _DIFFUSION_POST_PROCESS_FUNCS

    assert "Krea2Pipeline" in _DIFFUSION_MODELS
    assert "Krea2Pipeline" in _DIFFUSION_POST_PROCESS_FUNCS
