# SPDX-License-Identifier: Apache-2.0
"""Overlap the Ulysses forward all-to-all with the QKV projection (issue #4690, 1.1).

HunyuanImage3 image self-attention runs, per layer:
    qkv_proj (one fused GEMM) -> RoPE(q,k) -> KV-cache reuse / GQA repeat / joint
    -> forward all-to-all (scatter heads, gather seq) -> attention -> reverse all-to-all

This module overlaps it with the projection GEMM by splitting the fused qkv_proj
into Q/K/V and issuing each tensor's forward all-to-all *asynchronously*, so that
``a2a(Q)`` overlaps ``proj(K)`` and ``a2a(K)`` overlaps ``proj(V)`` (V has no RoPE,
so it is naturally deferrable).

Correctness is preserved by construction: same math as the baseline path, only the
projection is split and the all-to-all are issued async. Only the strict
sequence-parallel image-gen steady-state / first-step paths are handled here; every
other case falls back to the original ``image_attn`` (see ``can_overlap``).

Enabled by ``VLLM_OMNI_ULYSSES_... `` -> ``VLLM_OMNI_HUNYUAN_OVERLAP_QKV``.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
import torch.nn.functional as F

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

logger = logging.getLogger(__name__)

# diagnostic counters (dumped via _log_once on first occurrence of each outcome)
_diag_logged: set[str] = set()


def _diag(msg: str) -> None:
    """Log each distinct overlap outcome once, at WARNING so it is visible."""
    if msg not in _diag_logged:
        _diag_logged.add(msg)
        logger.warning("[overlap_qkv_a2a] %s", msg)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA head expansion (local copy of hunyuan_image3_transformer.repeat_kv to
    avoid a circular import). (B, S, n_kv, D) -> (B, S, n_kv*n_rep, D)."""
    if n_rep == 1:
        return x
    b, s, n_kv, d = x.shape
    return x[:, :, :, None, :].expand(b, s, n_kv, n_rep, d).reshape(b, s, n_kv * n_rep, d)


# ---------------------------------------------------------------------------
# Async forward all-to-all (mirrors comm.all_to_all_4D scatter_idx=2,gather_idx=1)
# split into "reshape-in + launch" and "wait + reshape-out" so the collective can
# overlap with the next tensor's projection.
# ---------------------------------------------------------------------------
def _fwd_a2a_launch(pg: dist.ProcessGroup, x: torch.Tensor, ws: int):
    """(B, S_local, H, D) -> launch a2a. Returns (work, out_buf, post_fn).

    post_fn(out_buf) -> (B, S_global, H/ws, D). ``x`` and ``out_buf`` must stay alive
    until work.wait().
    """
    bs, shard_seqlen, hc, hs = x.shape
    seqlen = shard_seqlen * ws
    shard_hc = hc // ws
    inp = x.reshape(bs, shard_seqlen, ws, shard_hc, hs).transpose(0, 2).contiguous()
    out = torch.empty_like(inp)
    work = dist.all_to_all_single(out, inp, group=pg, async_op=True)

    def post(o: torch.Tensor) -> torch.Tensor:
        o = o.reshape(seqlen, bs, shard_hc, hs)
        return o.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)

    return work, out, inp, post


def _rev_a2a(pg: dist.ProcessGroup, x: torch.Tensor, ws: int) -> torch.Tensor:
    """(B, S_global, H_local, D) -> (B, S_local, H, D) (sync; mirrors all_to_all_4D 1,2)."""
    bs, seqlen, shard_hc, hs = x.shape
    hc = shard_hc * ws
    shard_seqlen = seqlen // ws
    inp = (
        x.reshape(bs, ws, shard_seqlen, shard_hc, hs)
        .transpose(0, 3)
        .transpose(0, 1)
        .contiguous()
        .reshape(ws, shard_hc, shard_seqlen, bs, hs)
    )
    out = torch.empty_like(inp)
    dist.all_to_all_single(out, inp, group=pg)
    out = out.reshape(hc, shard_seqlen, bs, hs)
    return out.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)


def _split_qkv_weight(qkv_proj, q_size: int, kv_size: int):
    """Return (Wq, Wk, Wv) views of the fused QKVParallelLinear weight, or None if
    not a plain unquantized weight (then caller falls back to the fused GEMM)."""
    w = getattr(qkv_proj, "weight", None)
    if w is None or w.dim() != 2 or w.shape[0] != q_size + 2 * kv_size:
        return None
    return w[:q_size], w[q_size : q_size + kv_size], w[q_size + kv_size :]


def can_overlap(mgr, kwargs) -> bool:
    """Only the strict-mode SP image-gen *steady-state* path is handled here.

    first_step (prompt KV caching + prompt/image seq split) and uncond_cfg_prefill
    have extra seq-splitting that interacts with the early-launched Q all-to-all, so
    they fall back to the original image_attn.
    """
    if mgr.sp_size <= 1:
        _diag("SKIP: sp_size<=1")
        return False
    if kwargs.get("mode", "gen_text") != "gen_image":
        _diag("SKIP: mode!=gen_image")
        return False
    if kwargs.get("uncond_cfg_prefill", False):
        _diag("SKIP: uncond_cfg_prefill")
        return False
    if kwargs.get("first_step"):
        _diag("SKIP: first_step")
        return False
    return True


def _rope_one(emb, t: torch.Tensor, n_heads: int, cos, sin, bs: int, q_len: int) -> torch.Tensor:
    """Apply HunYuan 2D RoPE to a single tensor (q or k), matching HunYuanRotary2DEmbedder."""
    t = t.reshape(bs, q_len, n_heads, emb.head_dim)
    t = emb.rope(t.to(torch.float32), cos, sin)
    return t.reshape(bs * q_len, n_heads * emb.head_dim).to(torch.bfloat16)


def image_self_attention_overlap(attn, hidden_states, attention_mask, custom_pos_emb, **kwargs):
    """Overlapped image self-attention (issue #4690 1.1). `attn` is a HunYuanAttention.

    Splits the fused qkv_proj into Q/K/V and issues each tensor's forward all-to-all
    asynchronously so a2a(Q) overlaps proj(K) and a2a(K) overlaps proj(V). Reuses the
    KV-cache helpers, attention backend and reverse-a2a (post_attention) of the
    baseline path; only the projection split + forward a2a + joint slice are new.
    Falls back to the original image_attn on any unsupported/edge case.
    """
    from vllm_omni.diffusion.attention.parallel.ulysses import _UlyssesCtx

    mgr = attn.image_attn
    if not can_overlap(mgr, kwargs):
        return None  # unsupported case -> caller falls back to image_attn
    Wsplit = _split_qkv_weight(attn.qkv_proj, attn.q_size, attn.kv_size)
    if Wsplit is None:
        _diag(
            f"SKIP: cannot split qkv_proj weight (type={type(attn.qkv_proj).__name__}, "
            f"weight={getattr(getattr(attn.qkv_proj, 'weight', None), 'shape', None)})"
        )
        return None
    if getattr(attn.qkv_proj, "bias", None) is not None:
        _diag("SKIP: qkv_proj has bias (unsupported)")
        return None
    Wq, Wk, Wv = Wsplit

    strategy = mgr.attn._get_active_parallel_strategy()
    if not hasattr(strategy, "_ulysses_pg"):
        _diag(f"SKIP: strategy {type(strategy).__name__} has no _ulysses_pg")
        return None
    pg = strategy._ulysses_pg
    ws = strategy._sp_group.ulysses_world_size
    rank = strategy._sp_group.ulysses_rank

    bsz, q_len, hidden_size = hidden_states.size()
    x = hidden_states.reshape(-1, hidden_size)

    first_step = kwargs.get("first_step")
    query_lens = kwargs.get("query_lens")
    seq_lens = kwargs.get("seq_lens")
    shard_image_size = kwargs.get("shard_image_size")
    bs = len(query_lens)
    qn = query_lens[0]
    seq_len = seq_lens[0]
    nH, nKV, hd = attn.num_heads, attn.num_kv_heads, attn.head_dim
    repeat_num = nH // nKV

    emb = attn.image_rope2d_emb
    cos, sin = emb._prepare_cos_sin(custom_pos_emb, first_step, x.device)

    # ---- Q: project -> rope -> qk-norm -> reshape (B,qn,H,D) -> launch forward a2a (async) ----
    # qk-norm (RMSNorm over head_dim) is applied AFTER rope, matching the baseline
    # HunYuanAttention.forward order (rope -> query_layernorm/key_layernorm).
    _diag(f"ACTIVE: overlap path running (ws={ws}, nH={nH}, nKV={nKV}, qn={qn}, qk_norm={attn.use_qk_norm})")
    q = F.linear(x, Wq)
    q = _rope_one(emb, q, nH, cos, sin, bs, qn).reshape(bs, qn, nH, hd)
    if attn.use_qk_norm:
        q = attn.query_layernorm(q.reshape(-1, nH, hd).contiguous()).reshape(bs, qn, nH, hd)
    work_q, buf_q, in_q, post_q = _fwd_a2a_launch(pg, q, ws)

    # ---- K/V: project (overlaps a2a(Q)) -> rope(K) -> qk-norm(K) -> KV-cache reuse -> repeat ----
    k = F.linear(x, Wk)
    k = _rope_one(emb, k, nKV, cos, sin, bs, qn).reshape(bs, qn, nKV, hd)
    if attn.use_qk_norm:
        k = attn.key_layernorm(k.reshape(-1, nKV, hd).contiguous()).reshape(bs, qn, nKV, hd)
    v = F.linear(x, Wv).reshape(bs, qn, nKV, hd)

    # steady-state SP image step: joint = cached prompt KV; image query/key/value as-is
    joint_k, joint_v = mgr._reuse_prompt_kv(k, v, seq_len, bs, shard_image_size)
    joint_q = q[:, :0, :, :]

    k = _repeat_kv(k, repeat_num)
    v = _repeat_kv(v, repeat_num)
    joint_k = _repeat_kv(joint_k, repeat_num)
    joint_v = _repeat_kv(joint_v, repeat_num)

    work_k, buf_k, in_k, post_k = _fwd_a2a_launch(pg, k, ws)  # overlaps proj(V) above already done; launch now
    work_v, buf_v, in_v, post_v = _fwd_a2a_launch(pg, v, ws)

    # slice joint heads for this rank (front strategy), matching pre_attention
    jh_q = joint_q.shape[-2] // ws
    jh_kv = joint_k.shape[-2] // ws
    joint_q = joint_q[..., jh_q * rank : jh_q * (rank + 1), :]
    joint_k = joint_k[..., jh_kv * rank : jh_kv * (rank + 1), :]
    joint_v = joint_v[..., jh_kv * rank : jh_kv * (rank + 1), :]
    joint_len = joint_q.shape[1]

    # wait + reshape-out (Q's a2a has been overlapping the K/V projection + repeat)
    work_q.wait()
    q = post_q(buf_q)
    work_k.wait()
    k = post_k(buf_k)
    work_v.wait()
    v = post_v(buf_v)

    # concat joint front (matches pre_attention front strategy)
    q = torch.cat([joint_q, q], dim=1)
    k = torch.cat([joint_k, k], dim=1)
    v = torch.cat([joint_v, v], dim=1)

    attn_md = AttentionMetadata(attn_mask=attention_mask, full_attn_spans=kwargs.get("full_attn_spans", None))
    if attn_md.attn_mask is not None and attn_md.attn_mask.ndim == 2:
        if attn_md.attn_mask.shape[1] != q.shape[1]:
            jmask = torch.ones([q.shape[0], q.shape[1] - attn_md.attn_mask.shape[1]], dtype=torch.bool, device=q.device)
            attn_md.attn_mask = torch.cat([jmask, attn_md.attn_mask], dim=1)
        attn_md.attn_mask = attn_md.attn_mask.bool().contiguous()

    out = mgr.attn._run_local_attention(q, k, v, attn_md)

    ctx = _UlyssesCtx(
        name="ulysses",
        ulysses_pg=pg,
        scatter_idx=strategy._scatter_idx,
        gather_idx=strategy._gather_idx,
        use_sync=strategy._use_sync,
        joint_len=joint_len,
        joint_strategy="front",
    )
    out = strategy.post_attention(out, ctx)  # reverse a2a (image) + allgather (joint), recombine
    return out.reshape(bs * qn, nH, hd)
