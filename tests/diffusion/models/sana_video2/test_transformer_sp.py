# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import math
from types import MethodType

import pytest
import torch

from vllm_omni.diffusion.models.sana_video2.blocks import GatedLinearAttention, GatedSoftmaxAttention
from vllm_omni.diffusion.models.sana_video2.transformer_sana_video2 import (
    SanaVideo2TransformerConfig,
    SanaVideo2TransformerModel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _Group:
    def __init__(self, world_size=2, rank=0):
        self.world_size = world_size
        self.rank_in_group = rank
        self.ulysses_world_size = world_size
        self.ring_world_size = 1
        self.ulysses_group = object()
        self.calls = 0
        self.total = None
        self.partials = []

    def all_reduce(self, value):
        self.calls += 1
        self.partials.append(value.clone())
        return value if self.total is None else self.total.clone()

    def all_gather(self, value, dim=0, separate_tensors=True):
        self.calls += 1
        assert dim == 0 and separate_tensors
        return [value.clone() for _ in range(self.world_size)]


def _model(**overrides):
    config = dict(
        in_channels=2,
        hidden_size=24,
        depth=2,
        num_heads=4,
        caption_channels=8,
        model_max_length=4,
        linear_head_dim=6,
        softmax_head_dim=6,
        softmax_ratio=0.5,
        attn_res_block_size=2,
    )
    config.update(overrides)
    return SanaVideo2TransformerModel(SanaVideo2TransformerConfig(**config)).eval()


def _manual_rope_channel_first(x, freqs):
    batch, heads, head_dim, tokens = x.shape
    pairs = x.permute(0, 1, 3, 2).to(torch.float64).reshape(batch, heads, tokens, head_dim // 2, 2)
    real = pairs[..., 0] * freqs.real - pairs[..., 1] * freqs.imag
    imaginary = pairs[..., 0] * freqs.imag + pairs[..., 1] * freqs.real
    return (
        torch.stack((real, imaginary), dim=-1).reshape(batch, heads, tokens, head_dim).permute(0, 1, 3, 2).to(x.dtype)
    )


def _linear_dense_oracle(attn, x, freqs):
    batch, tokens, channels = x.shape
    q, k, v = attn.qkv(x).reshape(batch, tokens, 3, channels).unbind(2)
    q = attn.q_norm(q).transpose(-1, -2).reshape(batch, attn.heads, attn.dim, tokens)
    k = attn.k_norm(k).transpose(-1, -2).reshape(batch, attn.heads, attn.dim, tokens)
    v = v.transpose(-1, -2).reshape(batch, attn.heads, attn.dim, tokens)
    q = _manual_rope_channel_first(q, freqs).float()
    k = _manual_rope_channel_first(k, freqs).float()
    beta = torch.sigmoid(attn.beta_proj(x)).transpose(1, 2).unsqueeze(2)
    k = (k * beta).float()
    scores = torch.einsum("bhdt,bhds->bhts", k, q)
    output = torch.einsum("bhdt,bhts->bhds", v.float(), scores).to(x.dtype)
    output = attn.o_norm(output).reshape(batch, channels, tokens).permute(0, 2, 1)
    return attn.proj(output * torch.sigmoid(attn.output_gate(x)))


def _softmax_dense_oracle(attn, x):
    batch, tokens, channels = x.shape
    q, k, v = attn.qkv(x).reshape(batch, tokens, 3, channels).unbind(2)
    q = attn.q_norm(q).reshape(batch, tokens, attn.heads, attn.dim).transpose(1, 2).float()
    k = attn.k_norm(k).reshape(batch, tokens, attn.heads, attn.dim).transpose(1, 2).float()
    v = v.reshape(batch, tokens, attn.heads, attn.dim).transpose(1, 2).float()
    weights = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(attn.dim), dim=-1)
    output = torch.matmul(weights, v).transpose(1, 2).reshape(batch, tokens, channels).to(x.dtype)
    return attn.proj(output * torch.sigmoid(attn.output_gate(x)))


def test_sp1_forward_and_state_dict_are_unchanged():
    torch.manual_seed(3)
    model = _model()
    x = torch.randn(2, 2, 2, 1, 3)
    y = torch.randn(2, 3, 8)
    before = tuple(model.state_dict())
    baseline = model(x, torch.tensor([8, 6]), y)
    model.set_sequence_parallel(None)
    torch.testing.assert_close(model(x, torch.tensor([8, 6]), y), baseline, rtol=0, atol=0)
    assert tuple(model.state_dict()) == before


def test_linear_attention_sums_full_precision_state_across_uneven_token_shards():
    torch.manual_seed(5)
    attn = GatedLinearAttention(dim=24, head_dim=6).eval()
    x = torch.randn(2, 9, 24)
    phase = torch.randn(1, 1, 9, 3, dtype=torch.float64)
    freqs = torch.polar(torch.ones_like(phase), phase)
    ref = _linear_dense_oracle(attn, x, freqs)
    torch.testing.assert_close(attn(x, rotary_emb=freqs), ref, rtol=2e-5, atol=2e-5)
    group = _Group()
    attn.set_sequence_parallel(group)
    attn(x[:, :5], rotary_emb=freqs[:, :, :5])
    attn(x[:, 5:], rotary_emb=freqs[:, :, 5:])
    assert all(part.dtype == torch.float32 for part in group.partials)
    assert all(part.shape == (2, 4, 6, 6) for part in group.partials)
    group.total = sum(group.partials)
    group.partials.clear()
    out = torch.cat([attn(x[:, :5], rotary_emb=freqs[:, :, :5]), attn(x[:, 5:], rotary_emb=freqs[:, :, 5:])], dim=1)
    torch.testing.assert_close(out, ref, rtol=3e-5, atol=3e-5)


def test_softmax_attention_uses_public_ulysses_hooks_after_fp32_cast():
    torch.manual_seed(7)
    attn = GatedSoftmaxAttention(dim=24, head_dim=6).eval()
    x = torch.randn(1, 4, 24)
    baseline = attn(x)
    torch.testing.assert_close(baseline, _softmax_dense_oracle(attn, x), rtol=2e-5, atol=2e-5)

    class Spy:
        def __init__(self):
            self.calls = []

        def pre_attention(self, q, k, v, metadata):
            assert metadata is None
            assert q.dtype == k.dtype == v.dtype == torch.float32
            self.calls.append("pre")
            return q, k, v, metadata, object()

        def post_attention(self, out, ctx):
            assert out.dtype == torch.float32
            self.calls.append("post")
            return out

    spy = Spy()
    attn.set_sequence_parallel(spy)
    torch.testing.assert_close(attn(x), baseline, rtol=0, atol=0)
    assert spy.calls == ["pre", "post"]


@pytest.mark.parametrize("rank, expected_frames", [(0, [0, 0, 0, 1, 1]), (1, [1, 2, 2, 2])])
def test_ti2v_uneven_shard_uses_global_frame_mapping_and_rope(rank, expected_frames, monkeypatch):
    torch.manual_seed(11)
    monkeypatch.setattr("vllm_omni.diffusion.forward_context.get_ulysses_mode", lambda **_: "advanced_uaa")
    model = _model()
    group = _Group(rank=rank)
    model.set_sequence_parallel(group)
    recorded = []

    def attn_forward(self, hidden, y, t, mask=None, rotary_emb=None):
        recorded.append((t.clone(), rotary_emb.clone(), hidden.shape[1]))
        return torch.zeros_like(hidden)

    def mlp_forward(self, hidden, t):
        return torch.zeros_like(hidden)

    for block in model.blocks:
        block.forward_attn_sublayer = MethodType(attn_forward, block)
        block.forward_mlp_sublayer = MethodType(mlp_forward, block)

    final_t = []

    def final_forward(self, hidden, t):
        final_t.append(t.clone())
        return hidden.new_zeros(hidden.shape[0], hidden.shape[1], 2)

    model.final_layer.forward = MethodType(final_forward, model.final_layer)
    x = torch.randn(1, 2, 3, 1, 3)
    y = torch.randn(1, 2, 8)
    time = torch.tensor([2, 4, 8]).reshape(1, 1, 3, 1, 1)
    output = model(x, time, y)
    assert output.shape == x.shape
    expected = torch.tensor(expected_frames)
    embedded = model.t_embedder(time.flatten().float()).reshape(1, 1, 3, -1)
    expected_t = embedded.index_select(2, expected)
    expected_t0 = model.t_block(embedded).index_select(2, expected)
    assert len(recorded) == 2
    for (t, rope, tokens), kind in zip(recorded, model.block_attention_types):
        torch.testing.assert_close(t, expected_t0)
        start = 0 if rank == 0 else 5
        expected_rope = getattr(model, f"rope_{kind}")((3, 1, 3), x.device)[:, :, start : start + tokens]
        torch.testing.assert_close(rope, expected_rope)
    torch.testing.assert_close(final_t[0], expected_t)


def test_short_sequence_rejected_before_collectives():
    model = _model()
    group = _Group(world_size=4)
    model.set_sequence_parallel(group)
    with pytest.raises(ValueError, match="tokens.*SP"):
        model(torch.randn(1, 2, 1, 1, 3), torch.tensor([1]), torch.randn(1, 2, 8))
    assert group.calls == 0


def test_strict_mode_rejects_uneven_tokens_and_heads_before_collectives():
    model = _model()
    group = _Group()
    model.set_sequence_parallel(group)
    with pytest.raises(ValueError, match="strict.*token"):
        model(torch.randn(1, 2, 1, 1, 3), torch.tensor([1]), torch.randn(1, 2, 8))
    assert group.calls == 0

    model = _model(softmax_head_dim=8)
    model.set_sequence_parallel(group)
    with pytest.raises(ValueError, match="strict.*head"):
        model(torch.randn(1, 2, 1, 1, 4), torch.tensor([1]), torch.randn(1, 2, 8))
    assert group.calls == 0
