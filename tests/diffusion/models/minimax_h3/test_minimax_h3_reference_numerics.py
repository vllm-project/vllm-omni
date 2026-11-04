# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_blocks as blocks
from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3
from vllm_omni.diffusion.models.minimax_h3.numerics import MiniMaxH3RMSNorm

from .test_minimax_h3_quantization import _FakeAttention, _FakeLinear, _small_od_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def reference_config(monkeypatch):
    class Attention(_FakeAttention):
        attn_backend = SimpleNamespace(get_name=lambda: "SDPA")

    for name in ("ColumnParallelLinear", "RowParallelLinear"):
        monkeypatch.setattr(h3, name, _FakeLinear)
    for name in ("ColumnParallelLinear", "RowParallelLinear", "MergedColumnParallelLinear", "QKVParallelLinear"):
        monkeypatch.setattr(blocks, name, _FakeLinear)
    monkeypatch.setattr(blocks, "Attention", Attention)
    monkeypatch.setattr(h3, "get_tensor_model_parallel_world_size", lambda: 1)
    config = _small_od_config()
    config.cache_config = {"minimax_h3_adaln_offload": False}
    config.additional_config = {"minimax_h3_reference_numerics": True}
    config.parallel_config.ring_degree = config.parallel_config.allgather_degree = 1
    config.enforce_eager = True
    config.cache_backend = "none"
    return config


def test_reference_policy_covers_refiner_dit_heads_and_lora(reference_config):
    model = h3.MiniMaxH3DiTModel(reference_config)
    classes = (MiniMaxH3RMSNorm, blocks.MiniMaxH3Attention, blocks.MiniMaxH3MLP, blocks.MiniMaxH3DiTBlock)
    for module in model.modules():
        if isinstance(module, classes):
            assert module.reference_numerics
        if isinstance(module, _FakeLinear):
            assert module.diffusion_lora_fuse_addmm is False
    assert model.final_layer.reference_numerics
    reference_config.additional_config = {}
    optimized = h3.MiniMaxH3DiTModel(reference_config)
    assert set(model.state_dict()) == set(optimized.state_dict())
    assert not optimized.blocks[0].reference_numerics
    assert not optimized.token_refiner.final_norm.reference_numerics


@pytest.mark.parametrize("value", ["true", 1, None])
def test_reference_mode_requires_boolean(reference_config, value):
    reference_config.additional_config["minimax_h3_reference_numerics"] = value
    with pytest.raises(ValueError, match="must be a boolean"):
        h3.MiniMaxH3DiTModel(reference_config)


@pytest.mark.parametrize("setting,value", [("enforce_eager", False), ("cache_backend", "tea_cache")])
def test_reference_mode_rejects_arithmetic_changing_execution(reference_config, setting, value):
    setattr(reference_config, setting, value)
    with pytest.raises(ValueError, match="enforce_eager=True"):
        h3.MiniMaxH3DiTModel(reference_config)


def test_reference_mode_rejects_different_attention_kernel(reference_config, monkeypatch):
    monkeypatch.setattr(blocks.Attention, "attn_backend", SimpleNamespace(get_name=lambda: "CUDNN_ATTN"))
    with pytest.raises(ValueError, match="TORCH_SDPA"):
        h3.MiniMaxH3DiTModel(reference_config)


@pytest.mark.parametrize("degree", ["ulysses_degree", "ring_degree", "allgather_degree"])
def test_reference_mode_rejects_sequence_parallelism(reference_config, degree):
    setattr(reference_config.parallel_config, degree, 2)
    with pytest.raises(ValueError, match="TP=SP=1"):
        h3.MiniMaxH3DiTModel(reference_config)


def test_reference_gate_retains_bf16_product_rounding():
    from vllm_omni.diffusion.models.minimax_h3.numerics import reference_gate

    # The product rounds before cancellation against the residual.
    hidden = torch.tensor([[1.015625]], dtype=torch.bfloat16)
    gate = torch.tensor([[1.015625]], dtype=torch.bfloat16)
    residual = torch.tensor([[-1.0]], dtype=torch.bfloat16)
    actual = reference_gate(residual, gate, hidden, torch.tensor([0]))
    expected = residual + (gate * hidden)
    fused = (residual.float() + gate.float() * hidden.float()).bfloat16()
    assert not torch.equal(expected, fused)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
