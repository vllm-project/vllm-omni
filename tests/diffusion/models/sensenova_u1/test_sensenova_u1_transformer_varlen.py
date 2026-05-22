# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for varlen forward methods in sensenova_u1_transformer.py.

Tests the three-layer varlen encapsulation:
  Attention.forward_gen_varlen → DecoderLayer._forward_gen_varlen → Model.forward_varlen

All tests are CPU-only and do not require model weights.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

HIDDEN_DIM = 32
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = HIDDEN_DIM // NUM_HEADS  # 8
NUM_LAYERS = 2


# ============================================================
# Stubs
# ============================================================


class _IdentityLinear(nn.Module):
    def forward(self, x):
        return x, None


class _IdentityNorm(nn.Module):
    def forward(self, x):
        return x


class _IdentityMLP(nn.Module):
    def forward(self, x):
        return x


class _MockCacheLayer:
    def __init__(self, prefix_len: int, num_kv_heads: int, head_dim: int, seed: int = 0):
        self.flash_prefix_len = prefix_len
        gen = torch.Generator().manual_seed(seed)
        max_len = prefix_len + 64
        self.flash_k_cache = torch.randn(1, max_len, num_kv_heads, head_dim, generator=gen)
        self.flash_v_cache = torch.randn(1, max_len, num_kv_heads, head_dim, generator=gen)


class _MockDynamicCache:
    def __init__(self, num_layers: int, prefix_lens: list[int], num_kv_heads: int, head_dim: int):
        self.layers = [
            _MockCacheLayer(prefix_lens[i], num_kv_heads, head_dim, seed=i)
            for i in range(num_layers)
        ]


def _make_attention():
    """Create a SenseNovaU1Attention stub without loading weights."""
    from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import (
        SenseNovaU1Attention,
    )

    attn = object.__new__(SenseNovaU1Attention)
    nn.Module.__init__(attn)
    attn.hidden_size = HIDDEN_DIM
    attn.num_heads = NUM_HEADS
    attn.num_kv_heads = NUM_KV_HEADS
    attn.head_dim = HEAD_DIM
    attn.scaling = 1.0 / (HEAD_DIM ** 0.5)
    attn.layer_idx = 0
    attn.o_proj_mot_gen = _IdentityLinear()
    attn.qkv_proj_mot_gen = _IdentityLinear()
    attn.q_norm_mot_gen = _IdentityNorm()
    attn.k_norm_mot_gen = _IdentityNorm()
    attn.q_norm_hw_mot_gen = _IdentityNorm()
    attn.k_norm_hw_mot_gen = _IdentityNorm()
    return attn


def _make_decoder_layer():
    """Create a SenseNovaU1DecoderLayer stub."""
    from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import (
        SenseNovaU1DecoderLayer,
    )

    layer = object.__new__(SenseNovaU1DecoderLayer)
    nn.Module.__init__(layer)
    layer.self_attn = _make_attention()
    layer.input_layernorm_mot_gen = _IdentityNorm()
    layer.post_attention_layernorm_mot_gen = _IdentityNorm()
    layer.mlp_mot_gen = _IdentityMLP()
    return layer


# ============================================================
# Attention.forward_gen_varlen tests
# ============================================================


class TestAttentionForwardGenVarlen:

    def test_output_shape_single_request(self) -> None:
        attn = _make_attention()
        total_S = 8
        hidden = torch.randn(1, total_S, HIDDEN_DIM)
        indexes = torch.zeros(3, total_S, dtype=torch.long)
        prefix_k = torch.randn(4, NUM_KV_HEADS, HEAD_DIM)
        prefix_v = torch.randn(4, NUM_KV_HEADS, HEAD_DIM)
        cu_q = torch.tensor([0, total_S], dtype=torch.int32)
        cu_k = torch.tensor([0, 4 + total_S], dtype=torch.int32)

        def fake_project_and_rope(hs, idx, *args):
            B, S, D = hs.shape
            q = torch.randn(B, NUM_HEADS, S, HEAD_DIM)
            k = torch.randn(B, NUM_KV_HEADS, S, HEAD_DIM)
            v = torch.randn(B, NUM_KV_HEADS, S, HEAD_DIM)
            return q, k, v

        attn._project_and_rope = fake_project_and_rope

        def fake_flash_attn(q, k, v, cu_sq, cu_sk, max_sq, max_sk, **kwargs):
            assert q.shape == (total_S, NUM_HEADS, HEAD_DIM)
            assert k.shape[0] == 4 + total_S
            assert kwargs.get("causal") is False
            return torch.randn(total_S, NUM_HEADS, HEAD_DIM)

        with patch("flash_attn.flash_attn_varlen_func", side_effect=fake_flash_attn):
            out = attn.forward_gen_varlen(
                hidden, indexes, [(prefix_k, prefix_v)],
                cu_q, cu_k, total_S, 4 + total_S,
            )

        assert out.shape == (1, total_S, HIDDEN_DIM)

    def test_packs_kv_correctly_multi_request(self) -> None:
        attn = _make_attention()
        q_lens = [6, 10]
        prefix_lens = [3, 5]
        total_S = sum(q_lens)
        hidden = torch.randn(1, total_S, HIDDEN_DIM)
        indexes = torch.zeros(3, total_S, dtype=torch.long)
        prefix_kv = [
            (torch.ones(prefix_lens[i], NUM_KV_HEADS, HEAD_DIM) * (i + 1),
             torch.ones(prefix_lens[i], NUM_KV_HEADS, HEAD_DIM) * (i + 1))
            for i in range(2)
        ]
        cu_q = torch.tensor([0, q_lens[0], total_S], dtype=torch.int32)
        cu_k = torch.tensor(
            [0, prefix_lens[0] + q_lens[0], prefix_lens[0] + q_lens[0] + prefix_lens[1] + q_lens[1]],
            dtype=torch.int32,
        )

        packed_k_shapes = []

        def fake_project_and_rope(hs, idx, *args):
            B, S, D = hs.shape
            q = torch.randn(B, NUM_HEADS, S, HEAD_DIM)
            k = torch.randn(B, NUM_KV_HEADS, S, HEAD_DIM)
            v = torch.randn(B, NUM_KV_HEADS, S, HEAD_DIM)
            return q, k, v

        attn._project_and_rope = fake_project_and_rope

        def fake_flash_attn(q, k, v, cu_sq, cu_sk, max_sq, max_sk, **kwargs):
            packed_k_shapes.append(k.shape)
            expected_total_kv = sum(p + q for p, q in zip(prefix_lens, q_lens))
            assert k.shape[0] == expected_total_kv
            return torch.randn(total_S, NUM_HEADS, HEAD_DIM)

        with patch("flash_attn.flash_attn_varlen_func", side_effect=fake_flash_attn):
            out = attn.forward_gen_varlen(
                hidden, indexes, prefix_kv,
                cu_q, cu_k, max(q_lens), max(p + q for p, q in zip(prefix_lens, q_lens)),
            )

        assert out.shape == (1, total_S, HIDDEN_DIM)
        assert len(packed_k_shapes) == 1
        assert packed_k_shapes[0][0] == sum(prefix_lens) + total_S

    def test_flash_attn_called_with_causal_false(self) -> None:
        attn = _make_attention()
        total_S = 4
        hidden = torch.randn(1, total_S, HIDDEN_DIM)
        indexes = torch.zeros(3, total_S, dtype=torch.long)
        cu_q = torch.tensor([0, total_S], dtype=torch.int32)
        cu_k = torch.tensor([0, total_S], dtype=torch.int32)

        attn._project_and_rope = lambda hs, idx, *args: (
            torch.randn(1, NUM_HEADS, total_S, HEAD_DIM),
            torch.randn(1, NUM_KV_HEADS, total_S, HEAD_DIM),
            torch.randn(1, NUM_KV_HEADS, total_S, HEAD_DIM),
        )

        captured = {}

        def fake_flash_attn(q, k, v, cu_sq, cu_sk, max_sq, max_sk, **kwargs):
            captured.update(kwargs)
            return torch.randn(total_S, NUM_HEADS, HEAD_DIM)

        prefix_kv = [(torch.empty(0, NUM_KV_HEADS, HEAD_DIM), torch.empty(0, NUM_KV_HEADS, HEAD_DIM))]

        with patch("flash_attn.flash_attn_varlen_func", side_effect=fake_flash_attn):
            attn.forward_gen_varlen(hidden, indexes, prefix_kv, cu_q, cu_k, total_S, total_S)

        assert captured["causal"] is False
        assert "softmax_scale" in captured


# ============================================================
# DecoderLayer._forward_gen_varlen tests
# ============================================================


class TestDecoderLayerForwardGenVarlen:

    def test_residual_connections(self) -> None:
        layer = _make_decoder_layer()
        total_S = 6
        hidden = torch.ones(1, total_S, HIDDEN_DIM) * 2.0
        indexes = torch.zeros(3, total_S, dtype=torch.long)
        cu_q = torch.tensor([0, total_S], dtype=torch.int32)
        cu_k = torch.tensor([0, total_S], dtype=torch.int32)
        prefix_kv = [(torch.empty(0, NUM_KV_HEADS, HEAD_DIM), torch.empty(0, NUM_KV_HEADS, HEAD_DIM))]

        attn_return = torch.ones(1, total_S, HIDDEN_DIM) * 3.0
        layer.self_attn.forward_gen_varlen = MagicMock(return_value=attn_return)

        out = layer._forward_gen_varlen(
            hidden, indexes, prefix_kv, cu_q, cu_k, total_S, total_S,
        )

        # residual1 = hidden(2) + attn(3) = 5
        # mlp is identity, so mlp(norm(5)) = 5
        # residual2 = 5 + 5 = 10
        expected = torch.ones(1, total_S, HIDDEN_DIM) * 10.0
        torch.testing.assert_close(out, expected)

    def test_calls_attn_with_correct_args(self) -> None:
        layer = _make_decoder_layer()
        total_S = 4
        hidden = torch.randn(1, total_S, HIDDEN_DIM)
        indexes = torch.zeros(3, total_S, dtype=torch.long)
        cu_q = torch.tensor([0, total_S], dtype=torch.int32)
        cu_k = torch.tensor([0, 2 + total_S], dtype=torch.int32)
        prefix_kv = [(torch.randn(2, NUM_KV_HEADS, HEAD_DIM), torch.randn(2, NUM_KV_HEADS, HEAD_DIM))]

        layer.self_attn.forward_gen_varlen = MagicMock(return_value=torch.zeros(1, total_S, HIDDEN_DIM))

        layer._forward_gen_varlen(hidden, indexes, prefix_kv, cu_q, cu_k, total_S, 2 + total_S)

        layer.self_attn.forward_gen_varlen.assert_called_once()
        call_args = layer.self_attn.forward_gen_varlen.call_args
        assert call_args[0][1] is indexes
        assert call_args[0][2] is prefix_kv


# ============================================================
# Model.forward_varlen tests
# ============================================================


class TestModelForwardVarlen:

    def _make_model(self, num_layers=NUM_LAYERS):
        from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import (
            SenseNovaU1Model,
        )

        model = object.__new__(SenseNovaU1Model)
        nn.Module.__init__(model)
        model.layers = nn.ModuleList([_make_decoder_layer() for _ in range(num_layers)])
        model.norm_mot_gen = _IdentityNorm()
        return model

    def test_iterates_all_layers(self) -> None:
        model = self._make_model(num_layers=3)
        total_S = 4
        prefix_lens = [2, 3, 1]
        caches = [
            _MockDynamicCache(3, prefix_lens, NUM_KV_HEADS, HEAD_DIM)
            for _ in range(1)
        ]

        call_count = {"n": 0}
        original_varlen = type(model.layers[0])._forward_gen_varlen

        def counting_forward(self_layer, *args, **kwargs):
            call_count["n"] += 1
            return args[0]  # pass through hidden_states

        for layer in model.layers:
            layer._forward_gen_varlen = lambda *a, _f=counting_forward, _l=layer, **kw: _f(_l, *a, **kw)

        hidden = torch.randn(1, total_S, HIDDEN_DIM)
        indexes = torch.zeros(3, total_S, dtype=torch.long)
        cu_q = torch.tensor([0, total_S], dtype=torch.int32)
        cu_k = torch.tensor([0, prefix_lens[0] + total_S], dtype=torch.int32)

        model.forward_varlen(hidden, indexes, caches, cu_q, cu_k, total_S, prefix_lens[0] + total_S)
        assert call_count["n"] == 3

    def test_extracts_prefix_from_cache(self) -> None:
        model = self._make_model(num_layers=1)
        total_S = 4
        prefix_len = 5
        cache = _MockDynamicCache(1, [prefix_len], NUM_KV_HEADS, HEAD_DIM)

        captured_prefix_kv = []

        def capturing_forward(hidden_states, indexes, prefix_kv_list, *args, **kwargs):
            captured_prefix_kv.append(prefix_kv_list)
            return hidden_states

        model.layers[0]._forward_gen_varlen = capturing_forward

        hidden = torch.randn(1, total_S, HIDDEN_DIM)
        indexes = torch.zeros(3, total_S, dtype=torch.long)
        cu_q = torch.tensor([0, total_S], dtype=torch.int32)
        cu_k = torch.tensor([0, prefix_len + total_S], dtype=torch.int32)

        model.forward_varlen(hidden, indexes, [cache], cu_q, cu_k, total_S, prefix_len + total_S)

        assert len(captured_prefix_kv) == 1
        kv_list = captured_prefix_kv[0]
        assert len(kv_list) == 1
        pk, pv = kv_list[0]
        assert pk.shape == (prefix_len, NUM_KV_HEADS, HEAD_DIM)
        torch.testing.assert_close(pk, cache.layers[0].flash_k_cache[0, :prefix_len])

    def test_applies_final_norm(self) -> None:
        model = self._make_model(num_layers=1)
        norm_called = {"called": False}

        class TrackingNorm(nn.Module):
            def forward(self, x):
                norm_called["called"] = True
                return x * 2.0

        model.norm_mot_gen = TrackingNorm()

        for layer in model.layers:
            layer._forward_gen_varlen = lambda hs, *a, **kw: hs

        hidden = torch.ones(1, 4, HIDDEN_DIM)
        indexes = torch.zeros(3, 4, dtype=torch.long)
        cache = _MockDynamicCache(1, [2], NUM_KV_HEADS, HEAD_DIM)
        cu_q = torch.tensor([0, 4], dtype=torch.int32)
        cu_k = torch.tensor([0, 6], dtype=torch.int32)

        out = model.forward_varlen(hidden, indexes, [cache], cu_q, cu_k, 4, 6)
        assert norm_called["called"]
        torch.testing.assert_close(out.last_hidden_state, hidden * 2.0)

    def test_multi_request_prefix_extraction(self) -> None:
        model = self._make_model(num_layers=1)
        prefix_lens = [3, 7]
        caches = [
            _MockDynamicCache(1, [prefix_lens[0]], NUM_KV_HEADS, HEAD_DIM),
            _MockDynamicCache(1, [prefix_lens[1]], NUM_KV_HEADS, HEAD_DIM),
        ]

        captured = []

        def capturing_forward(hidden_states, indexes, prefix_kv_list, *args, **kwargs):
            captured.append([(pk.shape[0], pv.shape[0]) for pk, pv in prefix_kv_list])
            return hidden_states

        model.layers[0]._forward_gen_varlen = capturing_forward

        q_lens = [4, 6]
        total_S = sum(q_lens)
        hidden = torch.randn(1, total_S, HIDDEN_DIM)
        indexes = torch.zeros(3, total_S, dtype=torch.long)
        cu_q = torch.tensor([0, q_lens[0], total_S], dtype=torch.int32)
        cu_k = torch.tensor(
            [0, prefix_lens[0] + q_lens[0], prefix_lens[0] + q_lens[0] + prefix_lens[1] + q_lens[1]],
            dtype=torch.int32,
        )

        model.forward_varlen(hidden, indexes, caches, cu_q, cu_k, max(q_lens), max(p + q for p, q in zip(prefix_lens, q_lens)))

        assert captured[0] == [(3, 3), (7, 7)]


# ============================================================
# ForCausalLM.forward_varlen tests
# ============================================================


class TestForCausalLMForwardVarlen:

    def test_delegates_to_model(self) -> None:
        from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import (
            SenseNovaU1CausalLMOutput,
            SenseNovaU1ForCausalLM,
            SenseNovaU1ModelOutput,
        )

        causal_lm = object.__new__(SenseNovaU1ForCausalLM)
        nn.Module.__init__(causal_lm)

        expected_hidden = torch.randn(1, 10, HIDDEN_DIM)

        mock_model = MagicMock()
        mock_model.forward_varlen.return_value = SenseNovaU1ModelOutput(
            last_hidden_state=expected_hidden,
        )
        causal_lm.model = mock_model

        cu_q = torch.tensor([0, 10], dtype=torch.int32)
        cu_k = torch.tensor([0, 15], dtype=torch.int32)
        inputs = torch.randn(1, 10, HIDDEN_DIM)
        indexes = torch.zeros(3, 10, dtype=torch.long)

        result = causal_lm.forward_varlen(inputs, indexes, [], cu_q, cu_k, 10, 15)

        assert isinstance(result, SenseNovaU1CausalLMOutput)
        torch.testing.assert_close(result.hidden_states, expected_hidden)
        mock_model.forward_varlen.assert_called_once_with(
            inputs, indexes, [], cu_q, cu_k, 10, 15,
        )
