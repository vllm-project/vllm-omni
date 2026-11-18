# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused Drama 3 tower / sampling kernels against eager references (CUDA only)."""

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_marks

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "H100"}, num_cards=1)]

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DIM, HEADS, KV_HEADS, HEAD_DIM, ROTARY, INTER, EPS = 2048, 16, 2, 256, 64, 6144, 1e-6


def _gemma(x, w, eps=EPS):
    x32 = x.float()
    out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * (w.float() + 1.0)
    return out.to(x.dtype)


def _interleaved_rope(x, positions, base=1e7):
    inv = 1.0 / base ** (torch.arange(0, ROTARY, 2, device=x.device, dtype=torch.float32) / ROTARY)
    ang = positions.float()[:, None] * inv
    cos, sin = ang.cos()[:, None], ang.sin()[:, None]
    even, odd = x[..., 0:ROTARY:2].float(), x[..., 1:ROTARY:2].float()
    rot = torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1).flatten(-2)
    return torch.cat((rot.to(x.dtype), x[..., ROTARY:]), -1)


def _causal(q, k, v):
    # q [T, H, D], k/v [T, KV, D] for one sequence.
    rep = HEADS // KV_HEADS
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), is_causal=True)
    return out.transpose(0, 1).reshape(q.shape[0], -1)


def _weights(device):
    g = torch.Generator(device="cpu").manual_seed(0)

    def rnd(*shape, scale=0.02):
        return (torch.randn(*shape, generator=g) * scale).to(device=device, dtype=torch.bfloat16)

    return {
        "emb": rnd(4096, DIM, scale=0.5),
        "enorm": rnd(DIM, scale=0.1),
        "hnorm": rnd(DIM, scale=0.1),
        "eh": rnd(DIM, 2 * DIM),
        "in_norm": rnd(DIM, scale=0.1),
        "qkv": rnd(2 * HEADS * HEAD_DIM + 2 * KV_HEADS * HEAD_DIM, DIM),
        "q_norm": rnd(HEAD_DIM, scale=0.1),
        "k_norm": rnd(HEAD_DIM, scale=0.1),
        "o": rnd(DIM, HEADS * HEAD_DIM),
        "post_norm": rnd(DIM, scale=0.1),
        "gate_up": rnd(2 * INTER, DIM),
        "down": rnd(DIM, INTER),
        "final_norm": rnd(DIM, scale=0.1),
    }


def _reference_layer(w, hidden, code, mask, extra, positions):
    emb = w["emb"][code] * mask[:, None].to(torch.bfloat16)
    if extra is not None:
        emb = emb + extra
    mixed = torch.cat((_gemma(emb, w["enorm"]), _gemma(hidden, w["hnorm"])), -1) @ w["eh"].T
    x = _gemma(mixed, w["in_norm"])
    q_size, kv_size = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
    q, gate, k, v = (x @ w["qkv"].T).split([q_size, q_size, kv_size, kv_size], -1)
    t = x.shape[0]
    q = _interleaved_rope(_gemma(q.reshape(t, HEADS, HEAD_DIM), w["q_norm"]), positions)
    k = _interleaved_rope(_gemma(k.reshape(t, KV_HEADS, HEAD_DIM), w["k_norm"]), positions)
    attn = _causal(q, k, v.reshape(t, KV_HEADS, HEAD_DIM))
    attn_out = (attn * torch.sigmoid(gate)) @ w["o"].T
    residual = (attn_out.float() + mixed.float()).to(torch.bfloat16)
    ff_in = _gemma((attn_out.float() + mixed.float()), w["post_norm"]).to(torch.bfloat16)
    gate_p, up = (ff_in @ w["gate_up"].T).chunk(2, -1)
    ff = (F.silu(gate_p) * up) @ w["down"].T
    return _gemma(ff.float() + residual.float(), w["final_norm"]).to(torch.bfloat16)


@cuda
def test_fused_tower_layer_matches_reference():
    from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate

    from vllm_omni.model_executor.models.fish_omni.codebook_tower import interleaved_to_neox
    from vllm_omni.model_executor.models.fish_omni.kernels import gemma_rms_norm, sigmoid_gate, tower_mix_norm

    device = torch.device("cuda")
    w = _weights(device)
    t = 37
    g = torch.Generator(device="cpu").manual_seed(1)
    hidden = torch.randn(t, DIM, generator=g).to(device, torch.bfloat16)
    extra = (torch.randn(t, DIM, generator=g) * 0.3).to(device, torch.bfloat16)
    code = torch.randint(0, 4096, (t,), generator=g).to(device)
    mask = torch.rand(t, generator=g).to(device) > 0.3
    positions = torch.arange(1000, 1000 + t, device=device)
    expected = _reference_layer(w, hidden, code, mask, extra, positions)

    # Fused path with the finalize_weights() layout.
    order = interleaved_to_neox(HEAD_DIM, ROTARY).to(device)
    q_size, kv_size = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
    q_w, gate_w, k_w, v_w = w["qkv"].split([q_size, q_size, kv_size, kv_size], 0)
    q_w = q_w.reshape(HEADS, HEAD_DIM, -1)[:, order]
    k_w = k_w.reshape(KV_HEADS, HEAD_DIM, -1)[:, order]
    qkv_w = torch.cat(
        (
            torch.stack((q_w, gate_w.reshape(HEADS, HEAD_DIM, -1)), 1).reshape(2 * q_size, -1),
            k_w.reshape(kv_size, -1),
            v_w,
        )
    )
    inv = 1.0 / 1e7 ** (torch.arange(0, ROTARY, 2, device=device, dtype=torch.float32) / ROTARY)
    ang = torch.arange(4096, device=device, dtype=torch.float32)[:, None] * inv
    cache = torch.cat((ang.cos(), ang.sin()), -1)
    mixed = tower_mix_norm(hidden, code, mask, w["emb"], extra, w["enorm"], w["hnorm"], EPS) @ w["eh"].T
    qkv = gemma_rms_norm(mixed, w["in_norm"], EPS) @ qkv_w.T
    q_gate, k, v = qkv.split([2 * q_size, kv_size, kv_size], -1)
    q, k, gate = fused_qk_rmsnorm_rope_gate(
        q_gate,
        k,
        w["q_norm"][order],
        w["k_norm"][order],
        cache,
        positions,
        EPS,
        HEADS,
        KV_HEADS,
        HEAD_DIM,
        ROTARY,
        norm_beta=1.0,
    )
    attn = _causal(q.reshape(t, HEADS, HEAD_DIM), k.reshape(t, KV_HEADS, HEAD_DIM), v.reshape(t, KV_HEADS, HEAD_DIM))
    ff_in, residual = gemma_rms_norm(sigmoid_gate(attn, gate) @ w["o"].T, w["post_norm"], EPS, residual=mixed)
    gate_p, up = (ff_in @ w["gate_up"].T).chunk(2, -1)
    actual, _ = gemma_rms_norm((F.silu(gate_p) * up) @ w["down"].T, w["final_norm"], EPS, residual=residual)

    cos = F.cosine_similarity(actual.float(), expected.float(), dim=-1)
    assert cos.min() > 0.9995, cos.min()
    assert (actual.float() - expected.float()).abs().max() < 0.1


def _reference_top_k_top_p(logits, temperature, top_k, top_p):
    """Exact fp32 top-k / top-p sampling (top-p over the whole distribution, rank 0 kept)."""
    scaled = logits.float() / temperature.clamp(min=1e-5).unsqueeze(-1)
    sorted_logits, sorted_ids = torch.sort(scaled, dim=-1, descending=True)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    ranks = torch.arange(sorted_logits.shape[-1], device=logits.device).unsqueeze(0)
    dropped = (cumulative > top_p.unsqueeze(-1)) | (ranks >= top_k.unsqueeze(-1))
    dropped[:, 0] = False
    probabilities = torch.softmax(sorted_logits.masked_fill(dropped, float("-inf")), dim=-1)
    choice = torch.argmax(probabilities / torch.empty_like(probabilities).exponential_(), dim=-1, keepdim=True)
    return torch.gather(sorted_ids, -1, choice).squeeze(-1)


@cuda
def test_fused_draw_matches_reference_distribution():
    from vllm_omni.model_executor.models.fish_omni.kernels import draw_top_k_top_p

    device = torch.device("cuda")
    vocab, rows = 4096, 4096
    g = torch.Generator(device="cpu").manual_seed(3)
    base = (torch.randn(vocab, generator=g) * 3).to(device)
    logits = base.expand(rows, vocab).contiguous().to(torch.bfloat16)
    temperature = torch.full((rows,), 0.8, device=device)
    top_k = torch.full((rows,), 30, dtype=torch.int64, device=device)
    top_p = torch.full((rows,), 0.9, device=device)
    fused = draw_top_k_top_p(logits, temperature, top_k, top_p, torch.rand(rows, 64, device=device), 64)
    reference = _reference_top_k_top_p(logits, temperature, top_k, top_p)
    fused_hist = torch.bincount(fused, minlength=vocab).float() / rows
    reference_hist = torch.bincount(reference, minlength=vocab).float() / rows
    # Same support, and total variation within sampling noise.
    assert set(fused.unique().tolist()) <= set(torch.topk(logits[0].float(), 30).indices.tolist())
    assert 0.5 * (fused_hist - reference_hist).abs().sum() < 0.05


@cuda
def test_fused_draw_is_greedy_at_zero_temperature():
    from vllm_omni.model_executor.models.fish_omni.kernels import draw_top_k_top_p

    device = torch.device("cuda")
    logits = torch.randn(8, 4097, device=device)
    ids = torch.arange(4097, device=device) + 100
    out = draw_top_k_top_p(
        logits,
        torch.zeros(8, device=device),
        torch.full((8,), 30, dtype=torch.int64, device=device),
        torch.full((8,), 0.9, device=device),
        torch.rand(8, 64, device=device),
        64,
        id_map=ids,
    )
    assert torch.equal(out, logits.argmax(-1) + 100)


@cuda
def test_code_table_matches_mix_projection():
    """``T[c] + W_h hnorm(h)`` (load-time table) matches ``eh_proj([enorm(e_c); hnorm(h)])``."""
    from vllm_omni.model_executor.models.fish_omni.kernels import gemma_rms_norm, gemma_rms_norm_table

    device = torch.device("cuda")
    w = _weights(device)
    t = 257
    g = torch.Generator(device="cpu").manual_seed(5)
    hidden = torch.randn(t, DIM, generator=g).to(device, torch.bfloat16)
    code = torch.randint(0, 4096, (t,), generator=g).to(device)
    mask = torch.rand(t, generator=g).to(device) > 0.3

    emb = w["emb"][code] * mask[:, None].to(torch.bfloat16)
    mixed_ref = torch.cat((_gemma(emb, w["enorm"]), _gemma(hidden, w["hnorm"])), -1) @ w["eh"].T
    normed_ref = _gemma(mixed_ref, w["in_norm"])

    table = (gemma_rms_norm(w["emb"], w["enorm"], EPS).double() @ w["eh"][:, :DIM].double().T).to(torch.bfloat16)
    partial = gemma_rms_norm(hidden, w["hnorm"], EPS) @ w["eh"][:, DIM:].T
    normed, mixed = gemma_rms_norm_table(partial, code, mask, table, w["in_norm"], EPS)

    for actual, expected in ((mixed, mixed_ref), (normed, normed_ref)):
        cos = F.cosine_similarity(actual.float(), expected.float(), dim=-1)
        assert cos.min() > 0.9999, cos.min()
        rel = (actual.float() - expected.float()).norm() / expected.float().norm()
        assert rel < 1e-2, rel
