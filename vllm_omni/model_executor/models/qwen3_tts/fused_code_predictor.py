# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803 - Triton constexpr parameters use kernel-style capitals.
"""Fused residual-codebook predictor for Qwen3-TTS (CUDA, BF16).

Same computation as the frame-local-KV predictor with per-call top-k Gumbel
sampling, restructured to issue few kernels per residual step, since at serving
batch sizes each of the 15 dependent steps is launch- and latency-bound:

- Per layer: RMSNorm, one QKV GEMM, one attention kernel (q/k RMSNorm, RoPE,
  this call's K/V write and attention over the frame's keys), O projection
  accumulated into the residual stream (``addmm_``), RMSNorm, gate/up GEMM,
  SiLU-mul, down projection accumulated into the stream.
- The embedding of each sampled code followed by ``small_to_mtp_projection``
  is linear, so it folds at load into one ``[vocab, hidden]`` table per
  codebook; the sampling kernel writes the next step's input row directly.
- Sampling finds the top-k threshold by bisection over the BF16 order keys
  (exact k-th largest, ties kept) instead of a sort.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, tldevice, triton

from .tokenizer_12hz.streaming_decoder import _silu_mul_kernel


@triton.jit
def _rms_rows_kernel(x_ptr, x_stride, w_ptr, out_ptr, eps, N: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + row * x_stride + offs).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    dt = out_ptr.dtype.element_ty
    y = (x * tl.rsqrt(var + eps)).to(dt).to(tl.float32)
    w = tl.load(w_ptr + offs).to(tl.float32)
    tl.store(out_ptr + row * N + offs, (w * y).to(dt))


@triton.jit
def _bf(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _head_norm_rope(x1, x2, w1, w2, c1, c2, s1, s2, eps, HD: tl.constexpr):
    """HF RMSNorm (``weight * normed.to(bf16)``) then RoPE in BF16 elementwise steps, on the two halves."""
    var = (tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)) / HD
    r = tl.rsqrt(var + eps)
    n1 = _bf(w1 * _bf(x1 * r))
    n2 = _bf(w2 * _bf(x2 * r))
    # rotate_half(n) = [-n2, n1]
    o1 = _bf(_bf(n1 * c1) + _bf(-n2 * s1))
    o2 = _bf(_bf(n2 * c2) + _bf(n1 * s2))
    return o1, o2


@triton.jit
def _cp_attention_kernel(
    qkv_ptr,  # [B * NQ, (H + 2 * KVH) * HD]
    kc_ptr,  # [Bmax, KVH, MAXP, HD]
    vc_ptr,
    qw_ptr,
    kw_ptr,
    cos_ptr,  # [MAXP, HD] bf16
    sin_ptr,
    out_ptr,  # [B * NQ, H * HD]
    first,
    eps,
    scale,
    NQ: tl.constexpr,
    H: tl.constexpr,
    KVH: tl.constexpr,
    HD: tl.constexpr,
    MAXP: tl.constexpr,
    BK: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    qi = tl.program_id(2)
    kvh = h // (H // KVH)
    W: tl.constexpr = (H + 2 * KVH) * HD
    HALF: tl.constexpr = HD // 2
    d = tl.arange(0, HALF)
    p = first + qi
    row = (b * NQ + qi).to(tl.int64)
    qw1 = tl.load(qw_ptr + d).to(tl.float32)
    qw2 = tl.load(qw_ptr + HALF + d).to(tl.float32)
    kw1 = tl.load(kw_ptr + d).to(tl.float32)
    kw2 = tl.load(kw_ptr + HALF + d).to(tl.float32)
    qb = qkv_ptr + row * W + h * HD
    q1, q2 = _head_norm_rope(
        tl.load(qb + d).to(tl.float32), tl.load(qb + HALF + d).to(tl.float32), qw1, qw2,
        tl.load(cos_ptr + p * HD + d).to(tl.float32), tl.load(cos_ptr + p * HD + HALF + d).to(tl.float32),
        tl.load(sin_ptr + p * HD + d).to(tl.float32), tl.load(sin_ptr + p * HD + HALF + d).to(tl.float32),
        eps, HD,
    )  # fmt: skip
    j = tl.arange(0, BK)
    cache = ((b * KVH + kvh) * MAXP).to(tl.int64) * HD
    old = (j < first)[:, None]
    kp = kc_ptr + cache + j[:, None] * HD + d[None, :]
    vp = vc_ptr + cache + j[:, None] * HD + d[None, :]
    k1 = tl.load(kp, mask=old, other=0.0).to(tl.float32)
    k2 = tl.load(kp + HALF, mask=old, other=0.0).to(tl.float32)
    v1 = tl.load(vp, mask=old, other=0.0).to(tl.float32)
    v2 = tl.load(vp + HALF, mask=old, other=0.0).to(tl.float32)
    for n in tl.static_range(NQ):
        # this call's position first + n, computed from its own QKV row
        on = first + n
        kb = qkv_ptr + (b * NQ + n).to(tl.int64) * W + (H + kvh) * HD
        vb = qkv_ptr + (b * NQ + n).to(tl.int64) * W + (H + KVH + kvh) * HD
        kn1, kn2 = _head_norm_rope(
            tl.load(kb + d).to(tl.float32), tl.load(kb + HALF + d).to(tl.float32), kw1, kw2,
            tl.load(cos_ptr + on * HD + d).to(tl.float32), tl.load(cos_ptr + on * HD + HALF + d).to(tl.float32),
            tl.load(sin_ptr + on * HD + d).to(tl.float32), tl.load(sin_ptr + on * HD + HALF + d).to(tl.float32),
            eps, HD,
        )  # fmt: skip
        vn1 = tl.load(vb + d).to(tl.float32)
        vn2 = tl.load(vb + HALF + d).to(tl.float32)
        sel = (j == on)[:, None]
        k1 = tl.where(sel, kn1[None, :], k1)
        k2 = tl.where(sel, kn2[None, :], k2)
        v1 = tl.where(sel, vn1[None, :], v1)
        v2 = tl.where(sel, vn2[None, :], v2)
        wm = (d < HALF) & (qi == n) & (h % (H // KVH) == 0)
        dtc = kc_ptr.dtype.element_ty
        tl.store(kc_ptr + cache + on * HD + d, kn1.to(dtc), mask=wm)
        tl.store(kc_ptr + cache + on * HD + HALF + d, kn2.to(dtc), mask=wm)
        tl.store(vc_ptr + cache + on * HD + d, vn1.to(dtc), mask=wm)
        tl.store(vc_ptr + cache + on * HD + HALF + d, vn2.to(dtc), mask=wm)
    km = j <= p
    s = (tl.sum(k1 * q1[None, :], axis=1) + tl.sum(k2 * q2[None, :], axis=1)) * scale
    s = tl.where(km, s, -float("inf"))
    e = tl.exp(s - tl.max(s, axis=0))
    e = tl.where(km, e, 0.0)
    w = e / tl.sum(e, axis=0)
    ob = out_ptr + row * (H * HD) + h * HD
    dto = out_ptr.dtype.element_ty
    tl.store(ob + d, tl.sum(v1 * w[:, None], axis=0).to(dto))
    tl.store(ob + HALF + d, tl.sum(v2 * w[:, None], axis=0).to(dto))


@triton.jit
def _cp_sample_kernel(
    logits_ptr,  # [B, V] bf16
    u_ptr,  # uniforms, row stride u_stride, contiguous over V
    u_stride,
    codes_ptr,  # [B, G] int64
    step,
    G: tl.constexpr,
    table_ptr,  # [V, HID] next-step input table, or dummy
    next_ptr,  # [B, HID]
    inv_temperature,
    V: tl.constexpr,
    TOPK: tl.constexpr,
    HID: tl.constexpr,
    HAS_NEXT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    idx = tl.arange(0, V)
    value = tl.load(logits_ptr + row * V + idx)
    scaled = (value.to(tl.float32) * inv_temperature).to(value.dtype)
    if TOPK > 0:
        # order-preserving 16-bit keys of the BF16 values
        bits = scaled.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        # Top-k keeps all numeric ties at the cutoff, including both zeros.
        bits = tl.where((bits & 0x7FFF) == 0, 0, bits)
        key = tl.where(bits >= 0x8000, 0xFFFF - bits, bits | 0x8000)
        the = 0
        for bit in tl.static_range(15, -1, -1):
            cand = the | (1 << bit)
            cnt = tl.sum((key >= cand).to(tl.int32), axis=0)
            the = tl.where(cnt >= TOPK, cand, the)
        sf = tl.where(key >= the, scaled.to(tl.float32), -float("inf"))
    else:
        sf = scaled.to(tl.float32)
    u = tl.load(u_ptr + row * u_stride + idx)
    score = sf - tldevice.log(-tldevice.log(u))
    max_score = tl.max(score, 0)
    chosen = tl.min(tl.where(score == max_score, idx, V), 0)
    first_nan = tl.min(tl.where(score != score, idx, V), 0)
    code = tl.where(first_nan < V, first_nan, chosen)
    tl.store(codes_ptr + row * G + step, code.to(tl.int64))
    if HAS_NEXT:
        c = tl.arange(0, HID)
        tl.store(next_ptr + row * HID + c, tl.load(table_ptr + code.to(tl.int64) * HID + c))


class FusedCodePredictor:
    """Execution object built from a loaded ``Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM``."""

    def __init__(self, predictor, max_batch: int) -> None:
        model = predictor.model
        weight = next(model.parameters())
        dev, dt = weight.device, weight.dtype
        if dt != torch.bfloat16:
            raise ValueError("fused code predictor requires BF16 weights")
        cfg = predictor.config
        self.groups = int(cfg.num_code_groups)
        self.vocab = int(cfg.vocab_size)
        self.hidden = int(cfg.hidden_size)
        attn0 = model.layers[0].self_attn
        self.heads, self.kv_heads, self.head_dim = attn0.num_heads, attn0.num_kv_heads, attn0.head_dim
        self.scale = attn0.scaling
        self.eps = float(model.norm.variance_epsilon)
        self.max_pos = attn0.max_seq
        self.max_batch = max_batch
        f32 = lambda t: t.detach().to(device=dev, dtype=torch.float32)  # noqa: E731
        c = lambda t: t.to(dt).contiguous()  # noqa: E731
        proj = predictor.small_to_mtp_projection
        if isinstance(proj, torch.nn.Identity):
            pw, pb = None, None
        else:
            pw, pb = f32(proj.weight), f32(proj.bias)
        self.proj_wt = c(pw.t()) if pw is not None else None
        self.proj_b = c(pb) if pb is not None else None
        tables = []
        for emb in model.codec_embedding:
            e = f32(emb.weight)
            tables.append(c(e @ pw.t() + pb) if pw is not None else c(e))
        self.tables = tables
        self.layers = []
        for layer in model.layers:
            a = layer.self_attn
            self.layers.append(
                dict(
                    ln1=c(f32(layer.input_layernorm.weight)),
                    ln2=c(f32(layer.post_attention_layernorm.weight)),
                    eps1=float(layer.input_layernorm.variance_epsilon),
                    eps2=float(layer.post_attention_layernorm.variance_epsilon),
                    qkv=c(f32(a.qkv_proj.weight).t()),
                    qn=c(f32(a.q_norm.weight)),
                    kn=c(f32(a.k_norm.weight)),
                    head_eps=float(a.q_norm.variance_epsilon),
                    o=c(f32(a.o_proj.weight).t()),
                    gu=c(f32(layer.mlp.gate_up_proj.weight).t()),
                    down=c(f32(layer.mlp.down_proj.weight).t()),
                )
            )
            if a.qkv_proj.bias is not None:
                raise ValueError("fused code predictor assumes bias-free attention")
        self.norm_w = c(f32(model.norm.weight))
        self.lm_heads = [c(f32(h.weight).t()) for h in predictor.lm_head]
        rot = model.rotary_emb
        self.cos = rot.cos_cached.to(device=dev, dtype=dt).contiguous()
        self.sin = rot.sin_cached.to(device=dev, dtype=dt).contiguous()
        kv = (len(self.layers), 2, max_batch, self.kv_heads, self.max_pos, self.head_dim)
        self.cache = torch.zeros(kv, device=dev, dtype=dt)
        self._dummy = torch.zeros(1, device=dev, dtype=dt)

    def _layer(self, x: torch.Tensor, li: int, B: int, nq: int, first: int) -> None:
        L = self.layers[li]
        rows = x.shape[0]
        h = torch.empty_like(x)
        _rms_rows_kernel[(rows,)](x, x.stride(0), L["ln1"], h, L["eps1"], N=self.hidden)
        qkv = torch.mm(h, L["qkv"])
        att = torch.empty(rows, self.heads * self.head_dim, device=x.device, dtype=x.dtype)
        _cp_attention_kernel[(B, self.heads, nq)](
            qkv, self.cache[li, 0], self.cache[li, 1], L["qn"], L["kn"], self.cos, self.sin, att,
            first, L["head_eps"], self.scale,
            NQ=nq, H=self.heads, KVH=self.kv_heads, HD=self.head_dim, MAXP=self.max_pos,
            BK=triton.next_power_of_2(self.max_pos), num_warps=1,
        )  # fmt: skip
        x.addmm_(att, L["o"])
        _rms_rows_kernel[(rows,)](x, x.stride(0), L["ln2"], h, L["eps2"], N=self.hidden)
        gu = torch.mm(h, L["gu"])
        inter = gu.shape[1] // 2
        a = torch.empty(rows, inter, device=x.device, dtype=x.dtype)
        _silu_mul_kernel[(rows, triton.cdiv(inter, 1024))](gu, a, N=inter, BN=1024)
        x.addmm_(a, L["down"])

    @torch.inference_mode()
    def __call__(
        self,
        layer0_code: torch.Tensor,
        layer0_embed: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        inv_temperature: float,
        top_k: int,
        sample_uniforms: torch.Tensor,
    ) -> torch.Tensor:
        B = int(layer0_code.shape[0])
        if B > self.max_batch:
            raise ValueError("fused code predictor batch exceeds its capacity")
        dev, dt = layer0_embed.device, self.norm_w.dtype
        inp = torch.stack([last_talker_hidden.reshape(B, -1), layer0_embed.reshape(B, -1)], 1).to(dt)
        inp = inp.reshape(2 * B, -1)
        x = torch.addmm(self.proj_b, inp, self.proj_wt) if self.proj_wt is not None else inp.clone()
        codes = torch.empty(B, self.groups, device=dev, dtype=torch.int64)
        codes[:, 0] = layer0_code.reshape(B)
        u = sample_uniforms
        nq, first = 2, 0
        for step in range(1, self.groups):
            for li in range(len(self.layers)):
                self._layer(x, li, B, nq, first)
            last = x[nq - 1 :: nq] if nq > 1 else x
            hf = torch.empty(B, self.hidden, device=dev, dtype=dt)
            _rms_rows_kernel[(B,)](last, last.stride(0), self.norm_w, hf, self.eps, N=self.hidden)
            logits = torch.mm(hf, self.lm_heads[step - 1])
            has_next = step < self.groups - 1
            nxt = torch.empty(B, self.hidden, device=dev, dtype=dt) if has_next else self._dummy
            us = u[:, step - 1]
            _cp_sample_kernel[(B,)](
                logits, us, us.stride(0), codes, step, self.groups,
                self.tables[step - 1] if has_next else self._dummy, nxt, inv_temperature,
                V=self.vocab, TOPK=top_k, HID=self.hidden, HAS_NEXT=has_next, num_warps=4,
            )  # fmt: skip
            x = nxt
            first = step + 1
            nq = 1
        return codes
