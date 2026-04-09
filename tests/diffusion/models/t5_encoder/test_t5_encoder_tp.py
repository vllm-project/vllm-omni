# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from transformers import T5Config
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

from vllm_omni.diffusion.models.t5_encoder.t5_encoder import (
    T5EncoderModel,
    T5SelfAttention,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_T5_MODULE = "vllm_omni.diffusion.models.t5_encoder.t5_encoder"

SMALL_T5_CONFIG = dict(
    d_model=64,
    d_kv=8,
    d_ff=128,
    num_heads=8,
    num_layers=2,
    vocab_size=256,
    relative_attention_num_buckets=32,
    relative_attention_max_distance=128,
    is_gated_act=True,
    dense_act_fn="gelu_new",
    layer_norm_epsilon=1e-6,
    feed_forward_proj="gated-gelu",
)


@pytest.fixture(scope="module")
def t5_config() -> T5Config:
    return T5Config(**SMALL_T5_CONFIG)


@pytest.fixture(scope="function", autouse=True)
def setup_tp_group(monkeypatch, mocker):
    """Set up TP=2, rank=0, VllmConfig, and mock activation for all tests."""
    device_config = DeviceConfig(device="cpu")

    # TP world size
    monkeypatch.setattr("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(f"{_T5_MODULE}.get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(
        "vllm.model_executor.layers.vocab_parallel_embedding.get_tensor_model_parallel_world_size",
        lambda: 2,
    )

    monkeypatch.setattr(f"{_T5_MODULE}.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        "vllm.model_executor.layers.vocab_parallel_embedding.get_tensor_model_parallel_rank",
        lambda: 0,
    )

    # TP group
    mock_tp_group = mocker.MagicMock()
    mock_tp_group.world_size = 2
    mocker.patch("vllm.distributed.parallel_state.get_tp_group", return_value=mock_tp_group)

    monkeypatch.setattr(f"{_T5_MODULE}.get_act_fn", lambda _: torch.nn.GELU())

    with set_current_vllm_config(VllmConfig(device_config=device_config)):
        yield


class TestRelativePositionBiasTPSlicing:
    """Verify compute_bias slices heads correctly per TP rank."""

    def test_compute_bias_shape(self, t5_config):
        attn = T5SelfAttention(t5_config, has_relative_attention_bias=True)

        seq_len = 6
        bias = attn.compute_bias(seq_len, seq_len, device=torch.device("cpu"))

        local_heads = t5_config.num_heads // 2
        assert bias.shape == (1, local_heads, seq_len, seq_len)

    def test_all_ranks_cover_all_heads(self, t5_config, monkeypatch):
        seq_len = 4

        biases = []
        ref_weight = None
        for rank in range(2):
            monkeypatch.setattr(f"{_T5_MODULE}.get_tensor_model_parallel_rank", lambda r=rank: r)
            attn = T5SelfAttention(t5_config, has_relative_attention_bias=True)
            if rank > 0:
                attn.relative_attention_bias.weight.data.copy_(ref_weight)
            else:
                ref_weight = attn.relative_attention_bias.weight.data.clone()
            biases.append(attn.compute_bias(seq_len, seq_len, device=torch.device("cpu")))

        full_bias = torch.cat(biases, dim=1)
        assert full_bias.shape == (1, t5_config.num_heads, seq_len, seq_len)


class TestT5EncoderModelWeightLoading:
    """Test weight loading at the top-level T5EncoderModel."""

    def test_model_instantiation(self, t5_config):
        model = T5EncoderModel(t5_config, prefix="text_encoder")

        assert model.config is t5_config
        assert model.encoder is not None
        assert len(model.encoder.block) == t5_config.num_layers

    def test_embedding_shape(self, t5_config):
        model = T5EncoderModel(t5_config, prefix="text_encoder")

        assert model.shared.embedding_dim == t5_config.d_model

    def test_embed_input_ids(self, t5_config, monkeypatch):
        # Verify method and output shape
        model = T5EncoderModel(t5_config, prefix="text_encoder")

        # Mock all-reduce to be identity (no actual TP communication)
        monkeypatch.setattr(
            "vllm.model_executor.layers.vocab_parallel_embedding.tensor_model_parallel_all_reduce",
            lambda x: x,
        )

        input_ids = torch.randint(0, t5_config.vocab_size, (2, 8))
        embeddings = model.embed_input_ids(input_ids)

        assert embeddings.shape == (2, 8, t5_config.d_model)

    def test_qkv_weights_loaded_through_blocks(self):
        # Verify that HF-style separate Q/K/V weights can be loaded through
        # the block hierarchy
        config = T5Config(**{**SMALL_T5_CONFIG, "num_layers": 1})
        model = T5EncoderModel(config, prefix="text_encoder")

        inner_dim = config.num_heads * config.d_kv
        prefix = "encoder.block.0.layer.0.SelfAttention."

        loaded = model.load_weights(
            [
                (prefix + "q.weight", torch.randn(inner_dim, config.d_model)),
                (prefix + "k.weight", torch.randn(inner_dim, config.d_model)),
                (prefix + "v.weight", torch.randn(inner_dim, config.d_model)),
            ]
        )

        assert len(loaded) > 0
        attn = model.encoder.block[0].layer[0].SelfAttention
        expected_qkv_dim = 3 * (config.num_heads // 2) * config.d_kv
        assert attn.qkv_proj.weight.shape == (expected_qkv_dim, config.d_model)


class TestTPConstraints:
    """Verify that invalid TP configurations raise clear errors."""

    def test_num_heads_not_divisible_by_tp(self):
        config = T5Config(**{**SMALL_T5_CONFIG, "num_heads": 7})
        with pytest.raises(AssertionError, match=r"num_heads.*must be divisible by tp_size"):
            T5SelfAttention(config)

    def test_num_heads_divisible_by_tp(self, t5_config):
        attn = T5SelfAttention(t5_config)
        assert attn.n_heads_per_partition == 4
