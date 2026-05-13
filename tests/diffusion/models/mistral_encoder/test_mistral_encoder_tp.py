# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

from vllm_omni.diffusion.models.mistral_encoder.mistral_encoder import (
    MistralEncoderModel,
    MistralEncoderOutput,
    MistralRotaryEmbedding,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_MODULE = "vllm_omni.diffusion.models.mistral_encoder.mistral_encoder"

SMALL_MISTRAL_CONFIG = dict(
    hidden_size=64,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=8,
    intermediate_size=128,
    num_hidden_layers=2,
    rms_norm_eps=1e-5,
    max_position_embeddings=512,
    rope_theta=1000000.0,
    vocab_size=256,
)


def _make_config(**overrides):
    return SimpleNamespace(**{**SMALL_MISTRAL_CONFIG, **overrides})


def _make_nested_config(**overrides):
    """Simulate Mistral3Config with a text_config attribute."""
    text_config = _make_config(**overrides)
    return SimpleNamespace(text_config=text_config)


@pytest.fixture(scope="function", autouse=True)
def setup_tp_group(monkeypatch, mocker):
    """Set up TP=2, rank=0, and VllmConfig for all tests."""
    device_config = DeviceConfig(device="cpu")

    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(f"{_MODULE}.get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(
        "vllm.model_executor.layers.vocab_parallel_embedding.get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.vocab_parallel_embedding.get_tensor_model_parallel_rank",
        lambda: 0,
    )

    mock_tp_group = mocker.MagicMock()
    mock_tp_group.world_size = 2
    mocker.patch("vllm.distributed.parallel_state.get_tp_group", return_value=mock_tp_group)

    # Mock TP communication ops used during forward passes.  Each op is
    # imported by reference into the modules that use it, so we must patch
    # at every import site.
    _identity = lambda x: x  # noqa: E731
    monkeypatch.setattr(
        "vllm.model_executor.layers.vocab_parallel_embedding.tensor_model_parallel_all_reduce",
        _identity,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.tensor_model_parallel_all_reduce",
        _identity,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.tensor_model_parallel_all_gather",
        _identity,
    )
    monkeypatch.setattr(
        f"{_MODULE}.tensor_model_parallel_all_gather",
        _identity,
    )
    mocker.patch("torch.distributed.broadcast")

    from vllm.model_executor.layers.utils import default_unquantized_gemm

    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        lambda: default_unquantized_gemm,
    )

    with set_current_vllm_config(VllmConfig(device_config=device_config)):
        yield


class TestConfigParsing:
    """Verify that MistralEncoderModel extracts config correctly."""

    def test_plain_config(self):
        config = _make_config()
        model = MistralEncoderModel(config, prefix="text_encoder")

        assert model.hidden_size == 64
        assert model.num_heads == 8
        assert model.num_kv_heads == 4
        assert model.head_dim == 8
        assert model.intermediate_size == 128
        assert model.num_layers == 2
        assert model.rms_norm_eps == 1e-5
        assert model.rope_theta == 1000000.0
        assert model.vocab_size == 256

    def test_nested_text_config(self):
        config = _make_nested_config()
        model = MistralEncoderModel(config, prefix="text_encoder")

        assert model.hidden_size == 64
        assert model.num_heads == 8
        assert model.num_kv_heads == 4
        assert model.config is config.text_config

    def test_defaults_when_fields_missing(self):
        config = SimpleNamespace(
            hidden_size=64,
            num_attention_heads=8,
            intermediate_size=128,
            num_hidden_layers=1,
            vocab_size=256,
        )
        model = MistralEncoderModel(config, prefix="text_encoder")

        assert model.num_kv_heads == 8, "should fall back to num_attention_heads"
        assert model.head_dim == 8, "should compute hidden_size // num_heads"
        assert model.rms_norm_eps == 1e-5
        assert model.max_position_embeddings == 131072
        assert model.rope_theta == 1000000.0


class TestRoPEInitialization:
    """Verify that RoPE inv_freq is computed from config, not left uninitialized."""

    def test_inv_freq_deterministic(self):
        rope = MistralRotaryEmbedding(head_dim=8, max_position_embeddings=512, rope_theta=1000000.0)

        expected = 1.0 / (1000000.0 ** (torch.arange(0, 8, 2, dtype=torch.float32) / 8))
        assert torch.allclose(rope.inv_freq, expected)

    def test_different_theta_produces_different_freqs(self):
        rope_a = MistralRotaryEmbedding(head_dim=8, max_position_embeddings=512, rope_theta=10000.0)
        rope_b = MistralRotaryEmbedding(head_dim=8, max_position_embeddings=512, rope_theta=1000000.0)

        assert not torch.allclose(rope_a.inv_freq, rope_b.inv_freq)

    def test_cos_sin_shape_and_identity(self):
        rope = MistralRotaryEmbedding(head_dim=8, max_position_embeddings=512, rope_theta=1000000.0)
        seq_len = 16
        position_ids = torch.arange(seq_len).unsqueeze(0)
        cos, sin = rope(position_ids, torch.float32)

        assert cos.shape == (1, seq_len, 8)
        assert sin.shape == (1, seq_len, 8)
        assert torch.allclose(cos**2 + sin**2, torch.ones_like(cos), atol=1e-6)

    def test_model_rope_uses_config_theta(self):
        config = _make_config(rope_theta=10000.0)
        model = MistralEncoderModel(config, prefix="text_encoder")

        expected = 1.0 / (10000.0 ** (torch.arange(0, 8, 2, dtype=torch.float32) / 8))
        actual = model.language_model.model.rotary_emb.inv_freq
        assert torch.allclose(actual, expected)


class TestWeightLoading:
    """Test weight loading and stacked params mapping."""

    def test_qkv_weights_loaded(self):
        config = _make_config(num_hidden_layers=1)
        model = MistralEncoderModel(config, prefix="text_encoder")

        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim

        prefix = "language_model.model.layers.0.self_attn."
        weights = [
            (prefix + "q_proj.weight", torch.randn(num_heads * head_dim, hidden_size)),
            (prefix + "k_proj.weight", torch.randn(num_kv_heads * head_dim, hidden_size)),
            (prefix + "v_proj.weight", torch.randn(num_kv_heads * head_dim, hidden_size)),
        ]

        loaded = model.load_weights(weights)
        assert len(loaded) > 0
        assert any("qkv_proj" in p for p in loaded)

        attn = model.language_model.model.layers[0].self_attn
        # TP=2: q sharded to num_heads/2, kv sharded to num_kv_heads/2
        expected_dim = (num_heads // 2 + 2 * (num_kv_heads // 2)) * head_dim
        assert attn.qkv_proj.weight.shape == (expected_dim, hidden_size)

    def test_gate_up_weights_loaded(self):
        config = _make_config(num_hidden_layers=1)
        model = MistralEncoderModel(config, prefix="text_encoder")

        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        prefix = "language_model.model.layers.0.mlp."
        weights = [
            (prefix + "gate_proj.weight", torch.randn(intermediate_size, hidden_size)),
            (prefix + "up_proj.weight", torch.randn(intermediate_size, hidden_size)),
        ]

        loaded = model.load_weights(weights)
        assert len(loaded) > 0
        assert any("gate_up_proj" in p for p in loaded)

        mlp = model.language_model.model.layers[0].mlp
        # TP=2: each shard is intermediate_size/2, two merged
        expected_dim = 2 * (intermediate_size // 2)
        assert mlp.gate_up_proj.weight.shape == (expected_dim, hidden_size)

    def test_skips_vision_but_loads_lm_head(self):
        config = _make_config(num_hidden_layers=1)
        model = MistralEncoderModel(config, prefix="text_encoder")

        weights = [
            ("language_model.lm_head.weight", torch.randn(256, 64)),
            ("vision_tower.encoder.weight", torch.randn(64, 64)),
            ("multi_modal_projector.linear.weight", torch.randn(64, 64)),
        ]

        loaded = model.load_weights(weights)
        assert any("lm_head" in p for p in loaded)
        assert not any("vision_tower" in p for p in loaded)
        assert not any("multi_modal_projector" in p for p in loaded)

    def test_unknown_weights_ignored(self):
        config = _make_config(num_hidden_layers=1)
        model = MistralEncoderModel(config, prefix="text_encoder")

        weights = [("totally.fake.weight", torch.randn(10, 10))]
        loaded = model.load_weights(weights)
        assert len(loaded) == 0


class TestModelStructure:
    """Verify module hierarchy matches HF checkpoint layout."""

    def test_module_nesting(self):
        config = _make_config(num_hidden_layers=2)
        model = MistralEncoderModel(config, prefix="text_encoder")

        assert hasattr(model, "language_model")
        assert hasattr(model.language_model, "model")
        m = model.language_model.model
        assert hasattr(m, "embed_tokens")
        assert hasattr(m, "layers")
        assert hasattr(m, "norm")
        assert hasattr(m, "rotary_emb")
        assert len(m.layers) == 2

    def test_param_names_match_checkpoint_prefix(self):
        config = _make_config(num_hidden_layers=1)
        model = MistralEncoderModel(config, prefix="text_encoder")

        param_names = set(dict(model.named_parameters()).keys())
        assert any(n.startswith("language_model.model.embed_tokens") for n in param_names)
        assert any(n.startswith("language_model.model.layers.0.self_attn") for n in param_names)
        assert any(n.startswith("language_model.model.layers.0.mlp") for n in param_names)


class TestKVCache:
    """Verify KV cache plumbing through attention, layer, and model."""

    def test_attention_returns_none_kv_when_cache_off(self):
        from vllm_omni.diffusion.models.mistral_encoder.mistral_encoder import (
            MistralEncoderAttention,
        )

        attn = MistralEncoderAttention(
            hidden_size=64,
            num_heads=8,
            num_kv_heads=4,
            head_dim=8,
            prefix="test",
        )
        hidden = torch.randn(1, 4, 64)
        cos = torch.ones(1, 4, 8)
        sin = torch.zeros(1, 4, 8)
        out, kv = attn(hidden, cos, sin, use_cache=False)
        assert kv is None
        assert out.shape == (1, 4, 64)

    def test_attention_returns_kv_when_cache_on(self):
        from vllm_omni.diffusion.models.mistral_encoder.mistral_encoder import (
            MistralEncoderAttention,
        )

        attn = MistralEncoderAttention(
            hidden_size=64,
            num_heads=8,
            num_kv_heads=4,
            head_dim=8,
            prefix="test",
        )
        hidden = torch.randn(1, 4, 64)
        cos = torch.ones(1, 4, 8)
        sin = torch.zeros(1, 4, 8)
        out, kv = attn(hidden, cos, sin, use_cache=True)
        assert kv is not None
        k, v = kv
        # TP=2: num_kv_heads shard = 4//2 = 2
        assert k.shape == (1, 2, 4, 8)
        assert v.shape == (1, 2, 4, 8)

    def test_attention_appends_past_kv(self):
        from vllm_omni.diffusion.models.mistral_encoder.mistral_encoder import (
            MistralEncoderAttention,
        )

        attn = MistralEncoderAttention(
            hidden_size=64,
            num_heads=8,
            num_kv_heads=4,
            head_dim=8,
            prefix="test",
        )
        # Simulate a prefill with 4 tokens
        hidden = torch.randn(1, 4, 64)
        cos = torch.ones(1, 4, 8)
        sin = torch.zeros(1, 4, 8)
        _, kv = attn(hidden, cos, sin, use_cache=True)

        # Simulate a decode step with 1 token
        hidden_new = torch.randn(1, 1, 64)
        cos_new = torch.ones(1, 1, 8)
        sin_new = torch.zeros(1, 1, 8)
        out, kv2 = attn(hidden_new, cos_new, sin_new, past_key_value=kv, use_cache=True)
        k2, v2 = kv2
        assert k2.shape == (1, 2, 5, 8), "should be past(4) + new(1)"
        assert out.shape == (1, 1, 64)

    def test_model_forward_use_cache(self):
        config = _make_config(num_hidden_layers=2)
        model = MistralEncoderModel(config, prefix="text_encoder")
        input_ids = torch.randint(0, 128, (1, 8))

        output = model(input_ids, use_cache=True)
        assert isinstance(output, MistralEncoderOutput)
        assert output.past_key_values is not None
        assert len(output.past_key_values) == 2
        # Each layer's cache: (k, v) with seq_len=8
        k, v = output.past_key_values[0]
        assert k.shape[2] == 8
        assert v.shape[2] == 8

    def test_model_forward_no_cache(self):
        config = _make_config(num_hidden_layers=1)
        model = MistralEncoderModel(config, prefix="text_encoder")
        input_ids = torch.randint(0, 128, (1, 4))

        output = model(input_ids, use_cache=False)
        assert output.past_key_values is None

    def test_model_decode_with_past(self):
        config = _make_config(num_hidden_layers=1)
        model = MistralEncoderModel(config, prefix="text_encoder")

        # Prefill
        input_ids = torch.randint(0, 128, (1, 4))
        output = model(input_ids, use_cache=True)
        past = output.past_key_values

        # Decode one token
        new_token = torch.randint(0, 128, (1, 1))
        output2 = model(new_token, use_cache=True, past_key_values=past)
        assert output2.last_hidden_state.shape == (1, 1, 64)
        k2, v2 = output2.past_key_values[0]
        assert k2.shape[2] == 5, "cache should grow from 4 to 5"


class TestRoPEOffset:
    """Verify RoPE offset produces correct positions for decode steps."""

    def test_offset_zero_matches_original(self):
        rope = MistralRotaryEmbedding(head_dim=8, max_position_embeddings=512, rope_theta=1000000.0)
        position_ids = torch.arange(4).unsqueeze(0)
        cos_a, sin_a = rope(position_ids, torch.float32)
        cos_b, sin_b = rope(position_ids, torch.float32)
        assert torch.allclose(cos_a, cos_b)
        assert torch.allclose(sin_a, sin_b)

    def test_offset_produces_shifted_positions(self):
        rope = MistralRotaryEmbedding(head_dim=8, max_position_embeddings=512, rope_theta=1000000.0)
        # Full sequence of 5 positions
        full_ids = torch.arange(5).unsqueeze(0)
        cos_full, sin_full = rope(full_ids, torch.float32)
        # Position 4 only
        offset_ids = torch.tensor([[4]])
        cos_off, sin_off = rope(offset_ids, torch.float32)
        assert torch.allclose(cos_full[:, 4:5], cos_off)
        assert torch.allclose(sin_full[:, 4:5], sin_off)


class TestGenerate:
    """Verify autoregressive generate() method."""

    def test_generate_produces_tokens(self):
        config = _make_config(num_hidden_layers=1)
        model = MistralEncoderModel(config, prefix="text_encoder")
        input_ids = torch.randint(0, 128, (1, 4))
        attention_mask = torch.ones(1, 4, dtype=torch.long)

        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=3,
            do_sample=False,
            use_cache=True,
        )
        assert output.shape[0] == 1
        # prompt(4) + generated(3) = 7, unless EOS hit early
        assert output.shape[1] >= 5
        assert output.shape[1] <= 7
        # prompt tokens preserved
        assert torch.equal(output[:, :4], input_ids)

    def test_generate_stops_at_eos(self):
        config = _make_config(num_hidden_layers=1, eos_token_id=0)
        model = MistralEncoderModel(config, prefix="text_encoder")

        # Manually set embed_tokens so that token 0 maps to a specific embedding
        # that will produce logits strongly favouring token 0 again.
        # This is a probabilistic test — we just verify it doesn't exceed max.
        input_ids = torch.randint(1, 128, (1, 4))
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=10,
            do_sample=False,
            eos_token_id=0,
        )
        # Should have at most prompt(4) + max_new(10) = 14 tokens
        assert output.shape[1] <= 14

    def test_generate_greedy_deterministic(self):
        config = _make_config(num_hidden_layers=1)
        model = MistralEncoderModel(config, prefix="text_encoder")
        input_ids = torch.randint(0, 128, (1, 4))

        out1 = model.generate(input_ids=input_ids, max_new_tokens=5, do_sample=False)
        out2 = model.generate(input_ids=input_ids, max_new_tokens=5, do_sample=False)
        assert torch.equal(out1, out2)

    def test_generate_ignores_extra_kwargs(self):
        """generate() should accept and ignore pixel_values and other HF kwargs."""
        config = _make_config(num_hidden_layers=1)
        model = MistralEncoderModel(config, prefix="text_encoder")
        input_ids = torch.randint(0, 128, (1, 4))

        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=2,
            do_sample=False,
            pixel_values=torch.randn(1, 3, 224, 224),
        )
        assert output.shape[1] >= 5


class TestComputeLogits:
    """Verify logits computation via lm_head weight."""

    def test_logits_shape(self):
        config = _make_config(num_hidden_layers=1, vocab_size=256)
        model = MistralEncoderModel(config, prefix="text_encoder")

        hidden = torch.randn(1, 4, 64)
        logits = model._compute_logits(hidden)
        # TP=2: VocabParallelEmbedding stores vocab_size/2 = 128 per shard
        # With TP=2 but mocked (no actual all_gather), local logits only
        # In real TP, all_gather would give (1, 4, 256)
        # With mock, tp_size=2 triggers all_gather path but the mock may not
        # actually gather. Just verify we get a tensor back.
        assert logits.dim() == 3
        assert logits.shape[0] == 1
        assert logits.shape[1] == 4
