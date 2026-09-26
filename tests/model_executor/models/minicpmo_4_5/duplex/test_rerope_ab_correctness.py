# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A/B Correctness Regression Suite for MiniCPM-o 4.5 Stage-0 KV Window:

Compares in-place Re-RoPE + Block-Table Compaction against clean re-prefill ground truth:
1. Retained K after Re-RoPE vs. K from clean re-prefill (FP32 & BF16).
2. Next-step logits & greedy token generation rollout.
3. Prefix KV cache invariance (sink tokens remain bitwise untouched).
4. V cache invariance (values never receive RoPE phase rotation).
5. Block-boundary alignment & repeated compactions (compound rotations without drift).
6. Basic/context modes watermark policies, reference-audio sink, and camera MRoPE validation.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

try:
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_kv import (
        DUPLEX_WINDOW_BLOCK_SIZE,
        assert_uniform_position_shift,
        rotate_keys,
    )
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_plan import (
        DuplexWindowGeometry,
        plan_position_reanchor,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    import pathlib
    import sys
    import types

    def _make_pkg(name: str) -> types.ModuleType:
        if name in sys.modules:
            return sys.modules[name]
        m = types.ModuleType(name)
        m.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = m
        return m

    vllm = _make_pkg("vllm")
    vllm.__version__ = "0.7.0"  # type: ignore[attr-defined]
    vllm.__version_tuple__ = (0, 7, 0)  # type: ignore[attr-defined]
    v1 = _make_pkg("vllm.v1")
    spec_reg = _make_pkg("vllm.v1.kv_cache_spec_registry")
    spec_reg.register_kv_cache_spec = lambda *a, **k: (lambda cls: cls)  # type: ignore[attr-defined]
    kv_if = _make_pkg("vllm.v1.kv_cache_interface")

    class MockSpec:
        pass

    kv_if.KVCacheSpec = MockSpec  # type: ignore[attr-defined]
    kv_if.SlidingWindowSpec = MockSpec  # type: ignore[attr-defined]

    vo = _make_pkg("vllm_omni")
    vo_exp = _make_pkg("vllm_omni.experimental")
    vo_ad = _make_pkg("vllm_omni.experimental.ar_diffusion")
    vo_kc = _make_pkg("vllm_omni.experimental.ar_diffusion.kv_cache")
    vo_pg = _make_pkg("vllm_omni.experimental.ar_diffusion.kv_cache.paged")

    class MockChunkSpec:
        pass

    class MockChunkManager:
        def __init__(self, *a, **k):
            pass

        def reanchor_block_table(self, *a, **k):
            pass

    vo_pg.ChunkWindowSpec = MockChunkSpec  # type: ignore[attr-defined]
    vo_pg.ChunkWindowManager = MockChunkManager  # type: ignore[attr-defined]

    def compute_slot_mapping(block_ids, positions, block_size):
        p = positions.to(dtype=torch.long)
        t = torch.tensor(block_ids, dtype=torch.long, device=p.device)
        return t[torch.div(p, block_size, rounding_mode="floor")] * block_size + (p % block_size)

    vo_pg.compute_slot_mapping = compute_slot_mapping  # type: ignore[attr-defined]

    _make_pkg("vllm_omni.model_executor")
    _make_pkg("vllm_omni.model_executor.models")
    _make_pkg("vllm_omni.model_executor.models.minicpmo_4_5")
    _make_pkg("vllm_omni.model_executor.models.minicpmo_4_5.duplex")

    repo_root = pathlib.Path(__file__).resolve().parent
    while repo_root.name and not (repo_root / "vllm_omni").is_dir():
        repo_root = repo_root.parent

    def _load_module(name: str, rel_path: str) -> types.ModuleType:
        spec = importlib.util.spec_from_file_location(name, repo_root / rel_path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    _wp = _load_module(
        "vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_plan",
        "vllm_omni/model_executor/models/minicpmo_4_5/duplex/window_plan.py",
    )
    _wk = _load_module(
        "vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_kv",
        "vllm_omni/model_executor/models/minicpmo_4_5/duplex/window_kv.py",
    )

    DUPLEX_WINDOW_BLOCK_SIZE = _wk.DUPLEX_WINDOW_BLOCK_SIZE
    assert_uniform_position_shift = _wk.assert_uniform_position_shift
    rotate_keys = _wk.rotate_keys
    DuplexWindowGeometry = _wp.DuplexWindowGeometry
    plan_position_reanchor = _wp.plan_position_reanchor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

HEAD_DIM = 128
NUM_HEADS = 8
NUM_KV_HEADS = 2
HIDDEN_DIM = NUM_HEADS * HEAD_DIM
VOCAB_SIZE = 1000
BLOCK_SIZE = DUPLEX_WINDOW_BLOCK_SIZE  # 16


def _get_inv_freq(head_dim: int = HEAD_DIM, base: float = 1000000.0) -> torch.Tensor:
    return 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))


def _forward_rope(x: torch.Tensor, pos: int | torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    if isinstance(pos, int):
        pos_t = torch.tensor([pos], dtype=torch.float32, device=x.device)
    else:
        pos_t = pos.to(dtype=torch.float32, device=x.device)
    angle = torch.outer(pos_t, inv_freq.to(dtype=torch.float32)).unsqueeze(1)
    cos = torch.cos(angle).to(dtype=x.dtype)
    sin = torch.sin(angle).to(dtype=x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def _attention_decode(
    query_token: torch.Tensor,
    query_pos: int,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    w_q: torch.Tensor,
    w_o: torch.Tensor,
    lm_head: torch.Tensor,
    embedding: torch.Tensor,
    inv_freq: torch.Tensor,
) -> torch.Tensor:
    q_x = embedding[query_token]
    q_raw = (q_x @ w_q.T).view(-1, NUM_HEADS, HEAD_DIM)
    q = _forward_rope(q_raw, query_pos, inv_freq)

    num_rep = NUM_HEADS // NUM_KV_HEADS
    k_exp = k_cache.repeat_interleave(num_rep, dim=1)
    v_exp = v_cache.repeat_interleave(num_rep, dim=1)

    scores = torch.einsum("qhd,khd->hqk", q.float(), k_exp.float()) / math.sqrt(HEAD_DIM)
    probs = F.softmax(scores, dim=-1).to(k_cache.dtype)
    attn_out = torch.einsum("hqk,khd->qhd", probs, v_exp).reshape(-1, HIDDEN_DIM)
    out = attn_out @ w_o.T
    logits = out @ lm_head.T
    return logits


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_single_trim_ab_correctness(dtype: torch.dtype):
    """Criteria 1, 2, 3, 4: Retained K, Next-step Logits/Tokens, Prefix KV, Tail V invariance."""
    torch.manual_seed(42)
    inv_freq = _get_inv_freq(HEAD_DIM)

    W_q = (torch.randn(NUM_HEADS * HEAD_DIM, HIDDEN_DIM) * 0.02).to(dtype)
    W_k = (torch.randn(NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM) * 0.02).to(dtype)
    W_v = (torch.randn(NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM) * 0.02).to(dtype)
    W_o = (torch.randn(HIDDEN_DIM, NUM_HEADS * HEAD_DIM) * 0.02).to(dtype)
    lm_head = (torch.randn(VOCAB_SIZE, HIDDEN_DIM) * 0.02).to(dtype)
    embedding = (torch.randn(VOCAB_SIZE, HIDDEN_DIM) * 0.02).to(dtype)

    def compute_kv(tokens: torch.Tensor, positions: torch.Tensor):
        x = embedding[tokens]
        k = (x @ W_k.T).view(-1, NUM_KV_HEADS, HEAD_DIM)
        v = (x @ W_v.T).view(-1, NUM_KV_HEADS, HEAD_DIM)
        return _forward_rope(k, positions, inv_freq), v

    PREFIX_LEN, GAP_LEN, TAIL_LEN = 32, 32, 48
    TOTAL_LEN = PREFIX_LEN + GAP_LEN + TAIL_LEN
    tokens = torch.randint(0, VOCAB_SIZE, (TOTAL_LEN,))

    k_init, v_init = compute_kv(tokens, torch.arange(TOTAL_LEN))

    # Method A: In-Place Re-RoPE (rotate survivors by -GAP_LEN)
    k_tail_A = rotate_keys(k_init[PREFIX_LEN + GAP_LEN :], GAP_LEN, inv_freq)
    v_tail_A = v_init[PREFIX_LEN + GAP_LEN :]
    k_A = torch.cat([k_init[:PREFIX_LEN], k_tail_A], dim=0)
    v_A = torch.cat([v_init[:PREFIX_LEN], v_tail_A], dim=0)

    # Method B: Clean Re-prefill Ground Truth
    kept_tokens = torch.cat([tokens[:PREFIX_LEN], tokens[PREFIX_LEN + GAP_LEN :]], dim=0)
    k_B, v_B = compute_kv(kept_tokens, torch.arange(len(kept_tokens)))

    # Invariance assertions:
    # 1. Prefix K and V are strictly untouched (sink)
    assert torch.equal(k_A[:PREFIX_LEN], k_init[:PREFIX_LEN]), "Prefix K must be bitwise untouched"
    assert torch.equal(v_A[:PREFIX_LEN], v_init[:PREFIX_LEN]), "Prefix V must be bitwise untouched"
    # 2. Tail V is never rotated
    assert torch.equal(v_tail_A, v_init[PREFIX_LEN + GAP_LEN :]), "Tail V must not be modified by Re-RoPE"

    # 3. Retained K matches clean re-prefill
    diff_k = (k_tail_A.float() - k_B[PREFIX_LEN:].float()).abs().max().item()
    tol_k = 1e-5 if dtype == torch.float32 else 0.035
    assert diff_k < tol_k, f"Retained K diff {diff_k} exceeds tolerance {tol_k}"

    cos_sim = (
        F.cosine_similarity(
            k_tail_A.reshape(-1, HEAD_DIM).float(),
            k_B[PREFIX_LEN:].reshape(-1, HEAD_DIM).float(),
            dim=-1,
        )
        .mean()
        .item()
    )
    assert cos_sim > 0.9999, f"Cosine similarity {cos_sim} too low"

    # 4. Next-step logits & greedy token match
    q_tok = torch.randint(0, VOCAB_SIZE, (1,))
    next_pos = len(kept_tokens)
    log_A = _attention_decode(q_tok, next_pos, k_A, v_A, W_q, W_o, lm_head, embedding, inv_freq)
    log_B = _attention_decode(q_tok, next_pos, k_B, v_B, W_q, W_o, lm_head, embedding, inv_freq)

    tok_A = log_A.argmax(dim=-1).item()
    tok_B = log_B.argmax(dim=-1).item()
    assert tok_A == tok_B, f"Greedy next-step token mismatch: {tok_A} != {tok_B}"
    if dtype == torch.float32:
        diff_log = (log_A.float() - log_B.float()).abs().max().item()
        assert diff_log < 1e-4, f"Logits diff {diff_log} exceeds tolerance 1e-4"


def test_repeated_consecutive_trims_and_greedy_rollout():
    """Criteria 5: Verify compound rotations across 3 consecutive trims and 5-step rollout."""
    torch.manual_seed(123)
    inv_freq = _get_inv_freq(HEAD_DIM)

    W_q = torch.randn(NUM_HEADS * HEAD_DIM, HIDDEN_DIM) * 0.02
    W_k = torch.randn(NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM) * 0.02
    W_v = torch.randn(NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM) * 0.02
    W_o = torch.randn(HIDDEN_DIM, NUM_HEADS * HEAD_DIM) * 0.02
    lm_head = torch.randn(VOCAB_SIZE, HIDDEN_DIM) * 0.02
    embedding = torch.randn(VOCAB_SIZE, HIDDEN_DIM) * 0.02

    def compute_kv(tokens: torch.Tensor, positions: torch.Tensor):
        x = embedding[tokens]
        k = (x @ W_k.T).view(-1, NUM_KV_HEADS, HEAD_DIM)
        v = (x @ W_v.T).view(-1, NUM_KV_HEADS, HEAD_DIM)
        return _forward_rope(k, positions, inv_freq), v

    PREFIX_LEN, CHUNK = 32, 16
    t0 = torch.randint(0, VOCAB_SIZE, (PREFIX_LEN,))
    t1 = torch.randint(0, VOCAB_SIZE, (CHUNK,))
    t2 = torch.randint(0, VOCAB_SIZE, (CHUNK,))
    t3 = torch.randint(0, VOCAB_SIZE, (CHUNK,))

    # Prefill & Trim 1 (drop t1)
    k_p, v_p = compute_kv(torch.cat([t0, t1, t2, t3]), torch.arange(PREFIX_LEN + 3 * CHUNK))
    k1 = torch.cat([k_p[:PREFIX_LEN], rotate_keys(k_p[PREFIX_LEN + CHUNK :], CHUNK, inv_freq)], dim=0)
    v1 = torch.cat([v_p[:PREFIX_LEN], v_p[PREFIX_LEN + CHUNK :]], dim=0)

    # Append 1: t4, t5
    t4 = torch.randint(0, VOCAB_SIZE, (CHUNK,))
    t5 = torch.randint(0, VOCAB_SIZE, (CHUNK,))
    k_s1, v_s1 = compute_kv(torch.cat([t4, t5]), torch.arange(64, 64 + 2 * CHUNK))
    k2 = torch.cat([k1, k_s1], dim=0)
    v2 = torch.cat([v1, v_s1], dim=0)

    # Trim 2 (drop t2): t3 receives its 2nd rotation
    k3 = torch.cat([k2[:PREFIX_LEN], rotate_keys(k2[PREFIX_LEN + CHUNK :], CHUNK, inv_freq)], dim=0)
    v3 = torch.cat([v2[:PREFIX_LEN], v2[PREFIX_LEN + CHUNK :]], dim=0)

    # Append 2: t6
    t6 = torch.randint(0, VOCAB_SIZE, (CHUNK,))
    k_s2, v_s2 = compute_kv(t6, torch.arange(80, 80 + CHUNK))
    k4 = torch.cat([k3, k_s2], dim=0)
    v4 = torch.cat([v3, v_s2], dim=0)

    # Trim 3 (drop t3): t4 receives its 2nd rotation, t5 its 1st
    k_A_final = torch.cat([k4[:PREFIX_LEN], rotate_keys(k4[PREFIX_LEN + CHUNK :], CHUNK, inv_freq)], dim=0)
    v_A_final = torch.cat([v4[:PREFIX_LEN], v4[PREFIX_LEN + CHUNK :]], dim=0)

    # Ground Truth: Clean re-prefill of surviving sequence [t0, t4, t5, t6]
    kept_final = torch.cat([t0, t4, t5, t6])
    k_B_final, v_B_final = compute_kv(kept_final, torch.arange(len(kept_final)))

    diff_compound = (k_A_final - k_B_final).abs().max().item()
    assert diff_compound < 1e-5, f"Compound rotation diff {diff_compound} exceeds 1e-5"
    assert torch.allclose(v_A_final, v_B_final, atol=1e-6), (
        f"Compound V diff {(v_A_final - v_B_final).abs().max().item()} exceeds 1e-6"
    )

    # Multi-step autoregressive rollout (5 steps greedy decode)
    cur_A_tok = torch.tensor([42])
    cur_B_tok = torch.tensor([42])
    curr_k_A, curr_v_A = k_A_final.clone(), v_A_final.clone()
    curr_k_B, curr_v_B = k_B_final.clone(), v_B_final.clone()

    rollout_matches = []
    for step in range(5):
        cur_pos = len(kept_final) + step
        l_A = _attention_decode(cur_A_tok, cur_pos, curr_k_A, curr_v_A, W_q, W_o, lm_head, embedding, inv_freq)
        l_B = _attention_decode(cur_B_tok, cur_pos, curr_k_B, curr_v_B, W_q, W_o, lm_head, embedding, inv_freq)
        tok_A = l_A.argmax(dim=-1)
        tok_B = l_B.argmax(dim=-1)
        rollout_matches.append(tok_A.item() == tok_B.item())

        k_step_A, v_step_A = compute_kv(tok_A, torch.tensor([cur_pos]))
        curr_k_A = torch.cat([curr_k_A, k_step_A], dim=0)
        curr_v_A = torch.cat([curr_v_A, v_step_A], dim=0)

        k_step_B, v_step_B = compute_kv(tok_B, torch.tensor([cur_pos]))
        curr_k_B = torch.cat([curr_k_B, k_step_B], dim=0)
        curr_v_B = torch.cat([curr_v_B, v_step_B], dim=0)

        cur_A_tok = tok_A
        cur_B_tok = tok_B

    assert all(rollout_matches), f"Greedy rollout diverged at step {rollout_matches.index(False)}"


def test_block_boundary_and_fractional_tail():
    """Criteria 5: Test non-multiple block sizes and fractional tails."""
    torch.manual_seed(99)
    inv_freq = _get_inv_freq(HEAD_DIM)

    W_k = torch.randn(NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM) * 0.02
    W_v = torch.randn(NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM) * 0.02
    embedding = torch.randn(VOCAB_SIZE, HIDDEN_DIM) * 0.02

    def compute_kv(tokens: torch.Tensor, positions: torch.Tensor):
        x = embedding[tokens]
        k = (x @ W_k.T).view(-1, NUM_KV_HEADS, HEAD_DIM)
        v = (x @ W_v.T).view(-1, NUM_KV_HEADS, HEAD_DIM)
        return _forward_rope(k, positions, inv_freq), v

    PREFIX_LEN, GAP_LEN, PARTIAL_TAIL = 32, 16, 37  # 37 is not a multiple of 16
    tokens = torch.randint(0, VOCAB_SIZE, (PREFIX_LEN + GAP_LEN + PARTIAL_TAIL,))
    k_init, _ = compute_kv(tokens, torch.arange(len(tokens)))

    k_tail_A = rotate_keys(k_init[PREFIX_LEN + GAP_LEN :], GAP_LEN, inv_freq)
    k_A = torch.cat([k_init[:PREFIX_LEN], k_tail_A], dim=0)

    kept = torch.cat([tokens[:PREFIX_LEN], tokens[PREFIX_LEN + GAP_LEN :]], dim=0)
    k_B, _ = compute_kv(kept, torch.arange(len(kept)))

    diff = (k_A - k_B).abs().max().item()
    assert diff < 1e-5, f"Fractional tail diff {diff} exceeds 1e-5"


def test_basic_vs_context_mode_watermarks():
    """Criteria 6: Test watermark planner under basic and context session modes."""
    # Basic Mode: fixed watermarks (8000 high, 6000 low)
    geom_basic = DuplexWindowGeometry(
        prefix_tokens=96,
        window_tokens=6000,
        block_size=16,
        max_model_len=40960,
        high_watermark_tokens=8000,
    )
    # Below high watermark -> No trim
    assert plan_position_reanchor(geom_basic, computed_tokens=7900, pending_tokens=12) is None
    # Exceed high watermark (trigger = 96 + 8000 = 8096) -> Trims to low watermark
    plan_basic = plan_position_reanchor(geom_basic, computed_tokens=8100, pending_tokens=12)
    assert plan_basic is not None
    assert plan_basic.delta % 16 == 0
    assert plan_basic.sink_blocks == 6

    # Context Mode: dynamic unit history (e.g. max_units=24 -> 288 tokens + prefix)
    max_units = 24
    ctx_low = 96 + max_units * 12  # 384 tokens
    ctx_high = ctx_low + 500  # 884 tokens
    geom_context = DuplexWindowGeometry(
        prefix_tokens=96,
        window_tokens=ctx_low,
        block_size=16,
        max_model_len=40960,
        high_watermark_tokens=ctx_high,
    )
    # Exceed high watermark (trigger = 96 + 884 = 980) -> Trims
    plan_ctx = plan_position_reanchor(geom_context, computed_tokens=980, pending_tokens=12)
    assert plan_ctx is not None
    assert plan_ctx.delta % 16 == 0
    assert plan_ctx.sink_blocks == 6


def test_reference_audio_and_camera_mrope_paths():
    """Criteria 6: Reference-audio permanence in sink and camera non-uniform MRoPE rejection."""
    # Case A: Reference audio tokens (0..96) always live in sink blocks 0..5
    geom = DuplexWindowGeometry(
        prefix_tokens=96,
        window_tokens=6000,
        block_size=16,
        max_model_len=40960,
        high_watermark_tokens=8000,
    )
    plan = plan_position_reanchor(geom, computed_tokens=8100, pending_tokens=12)
    assert plan is not None
    assert plan.sink_blocks * 16 == 96
    assert plan.moved_from > 96  # Reference audio is strictly never rotated

    # Case B: Camera frame in retained tail (MRoPE non-uniform grid coordinates)
    mrope_pos_camera = torch.arange(200).unsqueeze(0).repeat(3, 1)
    mrope_pos_camera[1, 120:] += 4
    mrope_pos_camera[2, 120:] += 8
    with pytest.raises(RuntimeError, match="duplex re-anchor needs one position row"):
        assert_uniform_position_shift(mrope_pos_camera, moved_from=96)

    # Case C: Streaming audio/text units (MRoPE multi-row uniform advance)
    mrope_pos_audio = torch.arange(200).unsqueeze(0).repeat(3, 1)
    assert_uniform_position_shift(mrope_pos_audio, moved_from=96)
