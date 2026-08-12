# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from transformers import T5Config, UMT5Config, UMT5EncoderModel
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


@pytest.fixture(scope="module")
def umt5_config() -> UMT5Config:
    return UMT5Config(**SMALL_T5_CONFIG)


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


class TestRelativePositionBiasReuse:
    def test_t5_reuses_first_layer_position_bias(self, t5_config):
        model = T5EncoderModel(t5_config)

        attentions = [block.layer[0].SelfAttention for block in model.encoder.block]
        assert attentions[0].has_relative_attention_bias
        assert attentions[0].reuse_position_bias
        assert not attentions[1].has_relative_attention_bias
        assert attentions[1].reuse_position_bias

    def test_umt5_computes_position_bias_per_layer(self, umt5_config):
        model = T5EncoderModel(umt5_config)

        attentions = [block.layer[0].SelfAttention for block in model.encoder.block]
        assert all(attention.has_relative_attention_bias for attention in attentions)
        assert all(not attention.reuse_position_bias for attention in attentions)

    @pytest.mark.parametrize(
        ("config_cls", "expected_compute_calls"),
        [(T5Config, 0), (UMT5Config, 1)],
    )
    def test_forward_respects_position_bias_reuse(
        self,
        config_cls,
        expected_compute_calls,
        mocker,
    ):
        config = config_cls(**{**SMALL_T5_CONFIG, "num_layers": 1})
        attention = T5SelfAttention(config, has_relative_attention_bias=True)
        seq_len = 4
        local_heads = config.num_heads // 2
        hidden_states = torch.randn(1, seq_len, config.d_model)
        supplied_bias = torch.randn(1, local_heads, seq_len, seq_len)
        computed_bias = torch.randn_like(supplied_bias)

        mocker.patch.object(
            attention.qkv_proj,
            "forward",
            return_value=(
                torch.randn(1, seq_len, 3 * local_heads * config.d_kv),
                None,
            ),
        )
        mocker.patch.object(
            attention.o,
            "forward",
            side_effect=lambda value: value,
        )
        compute_bias = mocker.patch.object(
            attention,
            "compute_bias",
            return_value=computed_bias,
        )

        _, returned_bias = attention(hidden_states, position_bias=supplied_bias)

        assert compute_bias.call_count == expected_compute_calls
        expected_bias = computed_bias if expected_compute_calls else supplied_bias
        assert returned_bias is expected_bias

    def test_umt5_checkpoint_and_output_parity_tp1(self, umt5_config, monkeypatch):
        """Native UMT5 loading should match Transformers for a small checkpoint."""
        # The module fixture initializes a TP=2 test environment.  Use TP=1
        # here to cover the pipeline's non-sharded path as well.
        monkeypatch.setattr(f"{_T5_MODULE}.get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(f"{_T5_MODULE}.get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(
            "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
            lambda: 1,
        )
        monkeypatch.setattr(
            "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
            lambda: 0,
        )
        monkeypatch.setattr(
            "vllm.model_executor.layers.vocab_parallel_embedding.get_tensor_model_parallel_world_size",
            lambda: 1,
        )
        monkeypatch.setattr(
            "vllm.model_executor.layers.vocab_parallel_embedding.get_tensor_model_parallel_rank",
            lambda: 0,
        )
        monkeypatch.setattr(
            "vllm.model_executor.layers.linear.tensor_model_parallel_all_reduce",
            lambda value: value,
        )
        monkeypatch.setattr(
            "vllm.model_executor.layers.vocab_parallel_embedding.tensor_model_parallel_all_reduce",
            lambda value: value,
        )
        monkeypatch.setattr(f"{_T5_MODULE}.get_act_fn", lambda _: torch.nn.GELU(approximate="tanh"))

        torch.manual_seed(1234)
        hf_model = UMT5EncoderModel(umt5_config).eval()
        native_model = T5EncoderModel(umt5_config, prefix="text_encoder").eval()
        native_model.load_weights(list(hf_model.state_dict().items()))

        input_ids = torch.tensor([[1, 7, 13, 0, 0], [4, 9, 2, 8, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
        with torch.no_grad():
            expected = hf_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            actual = native_model(input_ids, attention_mask)[0]

        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


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
