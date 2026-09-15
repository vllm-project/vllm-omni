# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for T5EncoderModel prefix handling and weight loading fix."""

import pytest
import torch
from transformers import T5Config
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_omni.diffusion.models.t5_encoder.t5_encoder import T5EncoderModel
from vllm_omni.diffusion.models.utils import init_parameters

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


_SMALL_T5_CONFIG = dict(
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

_T5_MODULE = "vllm_omni.diffusion.models.t5_encoder.t5_encoder"


@pytest.fixture
def t5_config() -> T5Config:
    return T5Config(**_SMALL_T5_CONFIG)


@pytest.fixture(scope="function", autouse=True)
def setup_vllm_config(monkeypatch, mocker):
    """Set up VllmConfig and TP=2 mocks for tests."""
    device_config = DeviceConfig(device="cpu")

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

    mock_tp_group = mocker.MagicMock()
    mock_tp_group.world_size = 2
    mocker.patch("vllm.distributed.parallel_state.get_tp_group", return_value=mock_tp_group)

    monkeypatch.setattr(f"{_T5_MODULE}.get_act_fn", lambda _: torch.nn.GELU())

    with set_current_vllm_config(VllmConfig(device_config=device_config)):
        yield


class TestInitParameters:
    def test_materializes_meta_parameter_without_dropping_parameter_subclass(self):
        class CustomParameter(torch.nn.Parameter):
            @property
            def weight_loader(self):
                return self._weight_loader

        module = torch.nn.Module()

        def weight_loader(*args):
            return None

        param = CustomParameter(torch.empty((2, 3), device="meta"), requires_grad=False)
        param._weight_loader = weight_loader
        param.tp_rank = 0
        module.register_parameter("weight", param)

        init_parameters(module, dtype=torch.float16, device=torch.device("cpu"))

        assert isinstance(module.weight, CustomParameter)
        assert module.weight.device == torch.device("cpu")
        assert module.weight.dtype == torch.float16
        assert module.weight.weight_loader is weight_loader
        assert module.weight.tp_rank == 0


class TestT5EncoderModelPrefixHandling:
    """Test that T5EncoderModel correctly handles prefix attribute."""

    def test_prefix_stored_in_model(self, t5_config):
        """Test that prefix is stored in the model when provided."""
        prefix = "text_encoder"
        model = T5EncoderModel(t5_config, prefix=prefix)
        assert hasattr(model, "prefix")
        assert model.prefix == prefix

    def test_prefix_empty_by_default(self, t5_config):
        """Test that prefix defaults to empty string when not provided."""
        model = T5EncoderModel(t5_config)
        assert hasattr(model, "prefix")
        assert model.prefix == ""

    def test_quant_config_receives_component_prefixed_linears(self, mocker):
        """Test that quantized T5 linears receive fully qualified prefixes."""
        config = T5Config(**{**_SMALL_T5_CONFIG, "num_layers": 1})
        quant_config = mocker.MagicMock()
        quant_config.get_quant_method.side_effect = lambda layer, prefix: UnquantizedLinearMethod()

        T5EncoderModel(config, prefix="text_encoder", quant_config=quant_config)

        prefixes = {call.kwargs["prefix"] for call in quant_config.get_quant_method.call_args_list}
        assert {
            "text_encoder.encoder.block.0.layer.0.SelfAttention.qkv_proj",
            "text_encoder.encoder.block.0.layer.0.SelfAttention.o",
            "text_encoder.encoder.block.0.layer.1.DenseReluDense.wi",
            "text_encoder.encoder.block.0.layer.1.DenseReluDense.wo",
        }.issubset(prefixes)


class TestT5EncoderModelWeightLoadingWithPrefix:
    """Test weight loading with prefix handling."""

    def test_load_weights_with_prefix(self, t5_config):
        """Test that weights without prefix are loaded when model has prefix."""
        config = T5Config(**{**_SMALL_T5_CONFIG, "num_layers": 1})
        model = T5EncoderModel(config, prefix="text_encoder")

        inner_dim = config.num_heads * config.d_kv

        weights = [
            ("encoder.block.0.layer.0.SelfAttention.q.weight", torch.randn(inner_dim, config.d_model)),
            ("encoder.block.0.layer.0.SelfAttention.k.weight", torch.randn(inner_dim, config.d_model)),
            ("encoder.block.0.layer.0.SelfAttention.v.weight", torch.randn(inner_dim, config.d_model)),
        ]

        loaded = model.load_weights(weights)
        assert len(loaded) > 0

    def test_load_weights_embed_tokens_shared_sync(self, t5_config):
        """Test that embed_tokens and shared weights are synced."""
        model = T5EncoderModel(t5_config, prefix="text_encoder")

        d_model = t5_config.d_model
        vocab_size = t5_config.vocab_size

        embed_weight = torch.randn(vocab_size, d_model)
        weights = [
            ("encoder.embed_tokens.weight", embed_weight.clone()),
        ]

        model.load_weights(weights)

        shared_param = model.shared.weight
        embed_param = model.encoder.embed_tokens.weight

        assert torch.allclose(shared_param, embed_param), (
            "shared and embed_tokens should have the same weights after loading"
        )

    def test_load_weights_shared_without_prefix(self, t5_config):
        """Test shared.weight is recognized without relying on dot context."""
        model = T5EncoderModel(t5_config, prefix="text_encoder")

        shared_weight = torch.randn(t5_config.vocab_size, t5_config.d_model)
        loaded = model.load_weights([("shared.weight", shared_weight)])

        assert "shared.weight" in loaded
        assert torch.allclose(model.shared.weight, model.encoder.embed_tokens.weight)

    def test_unmatched_weights_are_not_reported_loaded(self, t5_config):
        """Test that skipped checkpoint weights are not added to loaded_params."""
        model = T5EncoderModel(t5_config, prefix="text_encoder")

        loaded = model.load_weights(
            [
                (
                    "text_encoder.encoder.block.0.layer.0.SelfAttention.missing.weight",
                    torch.randn(t5_config.d_model, t5_config.d_model),
                ),
            ]
        )

        assert loaded == set()


class TestT5EncoderModelWeightLoadingWithoutPrefix:
    """Test weight loading without prefix."""

    def test_load_weights_without_prefix(self, t5_config):
        """Test that weights without prefix are loaded correctly."""
        config = T5Config(**{**_SMALL_T5_CONFIG, "num_layers": 1})
        model = T5EncoderModel(config)

        inner_dim = config.num_heads * config.d_kv

        weights = [
            ("encoder.block.0.layer.0.SelfAttention.q.weight", torch.randn(inner_dim, config.d_model)),
        ]

        loaded = model.load_weights(weights)
        assert len(loaded) > 0
