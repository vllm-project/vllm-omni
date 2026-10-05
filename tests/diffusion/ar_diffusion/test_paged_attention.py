# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for AR-Diffusion paged self-attention contexts."""

from __future__ import annotations

import os
import subprocess
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec
from vllm_omni.experimental.ar_diffusion.kv_cache import (
    ARDiffusionKVCache,
    ARDiffusionKVConfig,
    ARDiffusionPagedLayerContext,
    ARDiffusionPagedLayerInputs,
    ar_diffusion_paged_attention,
    paged_write_attn,
)
from vllm_omni.experimental.ar_diffusion.kv_cache import paged_attention as paged_attention_module
from vllm_omni.experimental.ar_diffusion.kv_cache.config import KV_GATHER_ENV
from vllm_omni.experimental.ar_diffusion.kv_cache.paged import ChunkWindowManager
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


BLOCK = 16
N_HEADS = 4
HEAD_DIM = 64
POS = "positive"
NEG = "negative"


def make_state(
    *,
    num_layers=1,
    window_chunks=2,
    chunk_size=BLOCK,
    sink_chunks=0,
    reset_at_boundary=False,
    dtype=torch.float32,
    device=torch.device("cpu"),
    reuse_history_staging=False,
):
    """Build a cache. ``chunk_size`` defaults to the block size, but the two are
    independent -- the shipped 832x480 gives 1560 tokens per frame against
    16-token blocks, so a frame is 97.5 blocks."""
    cfg = ARDiffusionKVConfig(
        enable=True,
        chunk_size=chunk_size,
        window_chunks=window_chunks,
        sink_chunks=sink_chunks,
        reset_at_boundary=reset_at_boundary,
        reuse_history_staging=reuse_history_staging,
    )
    kv = ARDiffusionKVCache(
        cfg,
        num_layers=num_layers,
        num_kv_heads=N_HEADS,
        head_size=HEAD_DIM,
        dtype=dtype,
        block_size=BLOCK,
        max_model_len=4096,
        available_bytes=1 << 26,
        kv_branches=(ARDiffusionKVBranchSpec(POS, 0), ARDiffusionKVBranchSpec(NEG, 1)),
        session_capacity=2,
        frames_per_block=2,
        max_scratch_tokens_per_branch=BLOCK,
        device=device,
    )
    pos = kv.begin_request("r-pos")
    neg = kv.begin_request("r-neg")
    return kv, ARDiffusionKVState(kv, "s1", {POS: pos, NEG: neg}, num_layers=num_layers)


def _dense_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    scores = torch.einsum("bqhd,bkhd->bhqk", query.float(), key.float()) * (HEAD_DIM**-0.5)
    probs = torch.softmax(scores, dim=-1).to(value.dtype)
    return torch.einsum("bhqk,bkhd->bqhd", probs, value)


def _gpu_flash_attn_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    if torch.version.hip is not None:
        for module in ("aiter", "flash_attn"):
            try:
                imported = __import__(module, fromlist=["flash_attn_varlen_func"])
                if getattr(imported, "flash_attn_varlen_func", None) is not None:
                    return True
            except ImportError:
                pass
        return False
    try:
        spec = find_spec("vllm.vllm_flash_attn")
        if spec is None or spec.origin is None:
            return True
        fa2_so = Path(spec.origin).parent / "_vllm_fa2_C.abi3.so"
        linked = subprocess.check_output(["ldd", str(fa2_so)], text=True, timeout=5)
    except Exception:
        return True
    if "libcudart.so.13" not in linked:
        return True
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            timeout=5,
        )
        driver_major = int(out.splitlines()[0].split(".")[0])
    except Exception:
        return True
    return driver_major >= 580


def _require_gpu_flash_attn() -> None:
    """Skip optional local runs, but never let a dedicated GPU lane pass vacuously."""
    if _gpu_flash_attn_usable():
        return

    reason = "usable GPU FlashAttention is required"
    if os.environ.get("VLLM_OMNI_AR_FA_REQUIRED"):
        pytest.fail(f"{reason} when VLLM_OMNI_AR_FA_REQUIRED is set")
    pytest.skip(reason)


def _commit_video_span(
    kv: ARDiffusionKVCache,
    st: ARDiffusionKVState,
    *,
    kv_branch: str,
    n_chunks: int,
    dtype: torch.dtype,
    device: torch.device,
    chunk_size: int = BLOCK,
) -> tuple[torch.Tensor, torch.Tensor]:
    span = n_chunks * chunk_size
    ctx = st.get_kv_caches(kv_branch, seq_len=span, commit_current=True)[0].forward_ctx
    ctx.ensure_video_slots(device)
    k = torch.randn(1, span, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(1, span, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    kv._k_pools[0][ctx.current_video_slot_mapping] = k[0]
    kv._v_pools[0][ctx.current_video_slot_mapping] = v[0]
    st.commit_paged_context(kv_branch)
    return k, v


@pytest.mark.cpu
def test_paged_context_allocates_lazily_and_commits_after_forward():
    _, st = make_state()

    contexts = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=True)
    ctx = contexts[0].forward_ctx
    assert isinstance(contexts[0], ARDiffusionPagedLayerContext)
    assert st.adapter(POS).completed_chunks == 0
    assert ctx.current_video_slot_mapping is None

    ctx.ensure_video_slots(torch.device("cpu"))
    assert st.adapter(POS).completed_chunks == 0
    assert len(ctx.current_video_block_ids) == 1

    st.commit_paged_context(POS)
    assert st.adapter(POS).completed_chunks == 1
    assert st._committed[POS] == BLOCK


@pytest.mark.cpu
@pytest.mark.parametrize("chunk_size", [BLOCK, 2 * BLOCK, 3 * BLOCK])
def test_frame_causal_refresh_groups_and_commits_all_blocks_in_each_frame(chunk_size):
    kv, st = make_state(chunk_size=chunk_size)
    try:
        ctx = st.get_kv_caches(
            POS,
            seq_len=2 * chunk_size,
            commit_current=True,
            extra_visible_tokens=chunk_size,
            frame_causal=True,
        )[0].forward_ctx
        ctx.prepare(device=torch.device("cpu"), action_len=3, query_len=2 * chunk_size)
        assert ctx.block_table.shape[0] == 2
        assert ctx.query_start_loc.tolist() == [0, chunk_size, 2 * chunk_size]
        assert ctx.seq_lens.tolist() == [chunk_size + 3, 2 * chunk_size + 3]
        assert ctx.max_query_len == chunk_size
        assert ctx.action_scratch_block_ids == kv.scratch_block_ids(POS, 2 * chunk_size // BLOCK, 1)
        keys = torch.randn(2 * chunk_size, N_HEADS, HEAD_DIM)
        values = torch.randn_like(keys)
        kv._k_pools[0][ctx.current_video_slot_mapping] = keys
        kv._v_pools[0][ctx.current_video_slot_mapping] = values
        assert st.adapter(POS).completed_chunks == 0
        st.commit_paged_context(POS)
        assert st.adapter(POS).completed_chunks == 2
        blocks = kv.window_block_ids(st.adapter(POS))
        torch.testing.assert_close(kv.key_cache(0)[blocks].flatten(0, 1), keys)
        torch.testing.assert_close(kv.value_cache(0)[blocks].flatten(0, 1), values)
    finally:
        st.close()


@pytest.mark.cpu
def test_scratch_video_and_action_blocks_do_not_commit():
    kv, st = make_state()

    ctx = st.get_kv_caches(POS, seq_len=2 * BLOCK, commit_current=False)[0].forward_ctx
    ctx.ensure_video_slots(torch.device("cpu"))
    ctx.ensure_action_slots(3, torch.device("cpu"))

    assert ctx.current_video_block_ids == kv.scratch_block_ids(POS, 0, 2)
    assert ctx.action_scratch_block_ids == kv.scratch_block_ids(POS, 2, 1)
    st.commit_paged_context(POS)
    assert st.adapter(POS).completed_chunks == 0
    assert st._committed[POS] == 0


@pytest.mark.cpu
def test_pipeline_kv_get_paged_path_has_no_gather_backend():
    kv, st = make_state()
    assert not hasattr(kv, "gather_window_all_layers")

    from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline

    pipeline = DreamZeroPipeline.__new__(DreamZeroPipeline)
    pipeline._ar_diffusion_kv_state = st
    contexts = pipeline._kv_get(MagicMock(), False, seq_len=BLOCK, update_kv_cache=False)

    assert len(contexts) == 1
    assert isinstance(contexts[0], ARDiffusionPagedLayerContext)


@pytest.mark.parametrize("history_chunks", [0, 1, 3])
@pytest.mark.parametrize("action_len", [0, 3])
@pytest.mark.parametrize("commit_current", [False, True])
@pytest.mark.cpu
def test_paged_attention_matches_dense_reference_cpu(history_chunks, action_len, commit_current):
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32
    kv, st = make_state(dtype=dtype, device=device, window_chunks=2)

    history_k_parts: list[torch.Tensor] = []
    history_v_parts: list[torch.Tensor] = []
    if history_chunks:
        k, v = _commit_video_span(
            kv,
            st,
            kv_branch=POS,
            n_chunks=history_chunks,
            dtype=dtype,
            device=device,
        )
        history_k_parts.append(k)
        history_v_parts.append(v)

    layer_ctx = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=commit_current)[0]
    ctx = layer_ctx.forward_ctx
    ctx.ensure_video_slots(device)
    current_k = torch.randn(1, BLOCK, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    current_v = torch.randn(1, BLOCK, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    kv._k_pools[0][ctx.current_video_slot_mapping] = current_k[0]
    kv._v_pools[0][ctx.current_video_slot_mapping] = current_v[0]

    action_k = action_v = None
    if action_len:
        ctx.ensure_action_slots(action_len, device)
        action_k = torch.randn(1, action_len, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
        action_v = torch.randn(1, action_len, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
        kv._k_pools[0][ctx.action_slot_mapping] = action_k[0]
        kv._v_pools[0][ctx.action_slot_mapping] = action_v[0]

    query = torch.randn(1, BLOCK + action_len, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    block_table, query_start_loc, seq_lens, max_query_len, max_seq_len = ctx.build_block_table(
        action_len=action_len,
        query_len=query.shape[1],
        device=device,
    )
    paged = ar_diffusion_paged_attention(
        query,
        kv.key_cache(0),
        kv.value_cache(0),
        block_table=block_table,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        softmax_scale=HEAD_DIM**-0.5,
        causal=False,
    )

    if history_k_parts:
        history_k = torch.cat(history_k_parts, dim=1)
        history_v = torch.cat(history_v_parts, dim=1)
    else:
        history_k = torch.empty(1, 0, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
        history_v = torch.empty(1, 0, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    new_k = torch.cat([history_k, current_k], dim=1)[:, -kv.spec.sliding_window :]
    new_v = torch.cat([history_v, current_v], dim=1)[:, -kv.spec.sliding_window :]
    if action_len:
        new_k = torch.cat([new_k, action_k], dim=1)
        new_v = torch.cat([new_v, action_v], dim=1)
    ref = _dense_attention(query, new_k, new_v)

    torch.testing.assert_close(paged, ref, rtol=1e-5, atol=1e-5)

    ctx.prepare(device=device, action_len=action_len, query_len=query.shape[1])
    fused = paged_write_attn(
        layer_ctx.to_layer_inputs(),
        query[0],
        current_k[0],
        current_v[0],
        action_k[0] if action_len else None,
        action_v[0] if action_len else None,
        HEAD_DIM**-0.5,
    ).unsqueeze(0)
    torch.testing.assert_close(fused, ref, rtol=1e-5, atol=1e-5)

    before = st.adapter(POS).completed_chunks
    st.commit_paged_context(POS)
    assert st.adapter(POS).completed_chunks == before + (1 if commit_current else 0)


def test_extra_visible_tokens_keeps_the_history_window_in_addition_to_current():
    torch.manual_seed(0)
    device = torch.device("cpu")
    kv, st = make_state(window_chunks=2, device=device)
    committed = [
        _commit_video_span(
            kv,
            st,
            kv_branch=POS,
            n_chunks=1,
            dtype=torch.float32,
            device=device,
        )
        for _ in range(3)
    ]

    layer_ctx = st.get_kv_caches(
        POS,
        seq_len=BLOCK,
        commit_current=False,
        extra_visible_tokens=BLOCK,
    )[0]
    current_k = torch.randn(BLOCK, N_HEADS, HEAD_DIM)
    current_v = torch.randn_like(current_k)
    text_k = torch.randn(3, N_HEADS, HEAD_DIM)
    text_v = torch.randn_like(text_k)
    query = torch.randn(BLOCK, N_HEADS, HEAD_DIM)
    layer_ctx.forward_ctx.prepare(device=device, action_len=text_k.shape[0], query_len=query.shape[0])

    paged = paged_write_attn(
        layer_ctx.to_layer_inputs(),
        query,
        current_k,
        current_v,
        text_k,
        text_v,
        HEAD_DIM**-0.5,
    ).unsqueeze(0)
    dense_k = torch.cat([committed[-2][0], committed[-1][0], current_k.unsqueeze(0), text_k.unsqueeze(0)], dim=1)
    dense_v = torch.cat([committed[-2][1], committed[-1][1], current_v.unsqueeze(0), text_v.unsqueeze(0)], dim=1)

    torch.testing.assert_close(paged, _dense_attention(query.unsqueeze(0), dense_k, dense_v))


@pytest.mark.parametrize("history_chunks", [0, 1, 2, 3])
@pytest.mark.parametrize("window_chunks", [2, 4])
@pytest.mark.cpu
def test_staged_reuse_refreshes_the_current_blocks_at_their_live_offset(monkeypatch, history_chunks, window_chunks):
    """A second probe of the same AR block restages its current K/V where the table actually holds it.

    The padded table always ends in at least one action-capacity block, and while the window is still
    growing in unused window capacity too, so "the last current_blocks entries" is padding. The refresh
    has to start right after the visible history: empty (window growing), partial, full, and after a slide.
    """
    monkeypatch.setenv(KV_GATHER_ENV, "1")
    device = torch.device("cpu")
    dtype = torch.float32
    kv, st = make_state(dtype=dtype, device=device, window_chunks=window_chunks, reuse_history_staging=True)
    for buffer in kv.history_staging[0]:
        buffer.fill_(-7)
    if history_chunks:
        _commit_video_span(kv, st, kv_branch=POS, n_chunks=history_chunks, dtype=dtype, device=device)

    ctx = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=False)[0].forward_ctx
    ctx.max_video_tokens = 2 * BLOCK
    ctx.ensure_video_slots(device)
    key_cache, value_cache = kv.key_cache(0), kv.value_cache(0)

    def probe(step: int):
        # Each probe of the block writes new current K/V, then stages.
        kv._k_pools[0][ctx.current_video_slot_mapping] = torch.full((BLOCK, N_HEADS, HEAD_DIM), float(step))
        kv._v_pools[0][ctx.current_video_slot_mapping] = torch.full((BLOCK, N_HEADS, HEAD_DIM), -float(step))
        # Store the metadata on the context the way prepare() does.
        (ctx.block_table, ctx.query_start_loc, ctx.seq_lens, ctx.max_query_len, ctx.max_seq_len) = (
            ctx.build_block_table(action_len=0, query_len=BLOCK, device=device)
        )
        block_table, max_seq_len = ctx.block_table, ctx.max_seq_len
        ctx._prepare_history_staging(0)
        n_blocks = max_seq_len // BLOCK
        block_ids = block_table[0, :n_blocks].to(torch.long)
        stage_k, stage_v = ctx.history_staging(0)
        # The custom op narrows the manager-owned buffers before calling this helper.
        stage_k, stage_v = stage_k[:max_seq_len], stage_v[:max_seq_len]
        paged_attention_module._stage_window(
            stage_k,
            stage_v,
            key_cache,
            value_cache,
            block_ids,
            n_blocks,
            BLOCK,
            first_block=ctx.stage_first_block if ctx.reuse_history else 0,
        )
        full_k = key_cache.index_select(0, block_ids).reshape(n_blocks * BLOCK, N_HEADS, HEAD_DIM)
        full_v = value_cache.index_select(0, block_ids).reshape(n_blocks * BLOCK, N_HEADS, HEAD_DIM)
        for buffer in kv.history_staging[0]:
            assert (buffer[max_seq_len:] == -7).all()
        return stage_k, stage_v, full_k, full_v, n_blocks

    stage_k, stage_v, full_k, full_v, n_blocks = probe(1)
    assert ctx.reuse_history is False
    assert torch.equal(stage_k, full_k) and torch.equal(stage_v, full_v)

    stage_k, stage_v, full_k, full_v, n_blocks = probe(2)
    assert ctx.reuse_history is True
    visible_history_blocks = min(history_chunks, 2 - 1)  # window of 2 blocks minus the current one
    assert ctx.stage_first_block == visible_history_blocks
    # The padded table is wider than the live window, so the end-of-table guess is not the offset.
    assert n_blocks - 1 != ctx.stage_first_block
    assert torch.equal(stage_k, full_k) and torch.equal(stage_v, full_v)


@pytest.mark.parametrize("gather_enabled", [False, True])
@pytest.mark.cpu
def test_staging_is_only_allocated_for_the_gather_path(monkeypatch, gather_enabled):
    if gather_enabled:
        monkeypatch.setenv(KV_GATHER_ENV, "1")
    else:
        monkeypatch.delenv(KV_GATHER_ENV, raising=False)
    device = torch.device("cpu")
    kv, st = make_state(device=device, reuse_history_staging=True)
    # The manager owns the pairs and allocates them only when their consumer is on.
    assert bool(kv.history_staging) is gather_enabled
    ctx = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=False)[0].forward_ctx
    ctx.ensure_video_slots(device)
    (ctx.block_table, ctx.query_start_loc, ctx.seq_lens, ctx.max_query_len, ctx.max_seq_len) = ctx.build_block_table(
        action_len=0, query_len=BLOCK, device=device
    )
    ctx._prepare_history_staging(0)
    assert ctx.staging_enabled is gather_enabled
    stage_k, _ = ctx.history_staging(0)
    assert (stage_k is not None) is gather_enabled


@pytest.mark.parametrize("window_chunks", [2, 4])
@pytest.mark.cpu
def test_layer_inputs_preserve_static_staging_tensors(monkeypatch, window_chunks):
    """Compiled inputs must retain the manager's static-address annotations, including with spare capacity."""
    monkeypatch.setenv(KV_GATHER_ENV, "1")
    device = torch.device("cpu")
    kv, st = make_state(device=device, window_chunks=window_chunks, reuse_history_staging=True)
    stage_key, stage_value = kv.history_staging[0]
    for _ in range(2):
        ctx = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=False)[0].forward_ctx
        ctx.max_video_tokens = 2 * BLOCK
        ctx.prepare(device, action_len=0, query_len=BLOCK)
        inputs = ctx.layer_inputs(0)
        assert inputs.stage_key is stage_key
        assert inputs.stage_value is stage_value
        assert inputs.max_seq_len == 3 * BLOCK
        assert stage_key.shape[0] == (window_chunks + 1) * BLOCK


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("history_chunks", [1, 3])
@pytest.mark.parametrize("action_len", [0, 3])
@pytest.mark.parametrize("commit_current", [False, True])
def test_paged_attention_matches_dense_reference_gpu(history_chunks, action_len, commit_current):
    _require_gpu_flash_attn()
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float16
    kv, st = make_state(dtype=dtype, device=device, window_chunks=2)

    history_k, history_v = _commit_video_span(
        kv,
        st,
        kv_branch=POS,
        n_chunks=history_chunks,
        dtype=dtype,
        device=device,
    )

    layer_ctx = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=commit_current)[0]
    ctx = layer_ctx.forward_ctx
    current_k = torch.randn(1, BLOCK, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    current_v = torch.randn(1, BLOCK, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    action_k = action_v = None
    if action_len:
        action_k = torch.randn(1, action_len, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
        action_v = torch.randn(1, action_len, N_HEADS, HEAD_DIM, dtype=dtype, device=device)

    query = torch.randn(1, BLOCK + action_len, N_HEADS, HEAD_DIM, dtype=dtype, device=device)

    # The production path: once-per-forward host prep, then the fused
    # write+attend custom op consuming the NamedTuple payload.
    ctx.prepare(device=device, action_len=action_len, query_len=query.shape[1])
    inputs = layer_ctx.to_layer_inputs()
    assert isinstance(inputs, ARDiffusionPagedLayerInputs)
    paged = paged_write_attn(
        inputs,
        query[0],
        current_k[0],
        current_v[0],
        action_k[0] if action_k is not None else None,
        action_v[0] if action_v is not None else None,
        HEAD_DIM**-0.5,
    ).unsqueeze(0)

    # Direct python-fn call on the same (already written) pools must be
    # bit-exact: identical kernel, identical inputs.
    direct = ar_diffusion_paged_attention(
        query,
        kv.key_cache(0),
        kv.value_cache(0),
        block_table=ctx.block_table,
        query_start_loc=ctx.query_start_loc,
        seq_lens=ctx.seq_lens,
        max_query_len=ctx.max_query_len,
        max_seq_len=ctx.max_seq_len,
        softmax_scale=HEAD_DIM**-0.5,
        causal=False,
    )
    assert torch.equal(paged, direct)

    new_k = torch.cat([history_k, current_k], dim=1)[:, -kv.spec.sliding_window :]
    new_v = torch.cat([history_v, current_v], dim=1)[:, -kv.spec.sliding_window :]
    if action_len:
        new_k = torch.cat([new_k, action_k], dim=1)
        new_v = torch.cat([new_v, action_v], dim=1)
    ref = _dense_attention(query, new_k, new_v)

    torch.testing.assert_close(paged, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.cpu
def test_block_table_padded_to_fixed_width():
    """Shapes must be constant across window growth: only values change."""
    device = torch.device("cpu")
    kv, st = make_state(window_chunks=2)

    ctx1 = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=True)[0].forward_ctx
    ctx1.prepare(device=device, action_len=0, query_len=BLOCK)
    st.commit_paged_context(POS)

    ctx2 = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=True)[0].forward_ctx
    ctx2.prepare(device=device, action_len=0, query_len=BLOCK)
    st.commit_paged_context(POS)

    # 1-block vs 2-block visible history: same table width, same max_seq_len.
    assert ctx1.block_table.shape == ctx2.block_table.shape
    assert ctx1.max_seq_len == ctx2.max_seq_len
    expected_width = kv.spec.sliding_window // kv.block_size + 1
    assert ctx1.block_table.shape == (1, expected_width)
    # Real lengths live in seq_lens, not the padded table.
    assert int(ctx1.seq_lens[0]) == BLOCK
    assert int(ctx2.seq_lens[0]) == 2 * BLOCK


@pytest.mark.cpu
def test_prepare_is_idempotent_and_layers_share_metadata():
    device = torch.device("cpu")
    kv, st = make_state(num_layers=2)
    contexts = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=False)
    fctx = contexts[0].forward_ctx
    fctx.prepare(device=device, action_len=0, query_len=BLOCK)
    table = fctx.block_table
    fctx.prepare(device=device, action_len=0, query_len=BLOCK)
    assert fctx.block_table is table  # memoized, not rebuilt

    i0, i1 = contexts[0].to_layer_inputs(), contexts[1].to_layer_inputs()
    # 0-dim tensors (NOT python ints) so dynamo doesn't install per-layer
    # value guards on the shared block code object.
    assert isinstance(i0.layer_idx, torch.Tensor) and int(i0.layer_idx) == 0
    assert isinstance(i1.layer_idx, torch.Tensor) and int(i1.layer_idx) == 1
    # All layers share the same metadata tensor objects.
    assert i0.block_table is i1.block_table
    assert i0.seq_lens is i1.seq_lens
    assert i0.video_slots is i1.video_slots
    assert i0.key_pool is kv._k_pools[0]
    assert i0.value_pool is kv._v_pools[0]
    assert i1.key_pool is kv._k_pools[1]
    assert i1.value_pool is kv._v_pools[1]


@pytest.mark.cpu
def test_layer_inputs_before_prepare_raises():
    _, st = make_state()
    layer_ctx = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=False)[0]
    with pytest.raises(RuntimeError, match="before prepare"):
        layer_ctx.to_layer_inputs()


@pytest.mark.cpu
def test_custom_op_registration_idempotent():
    import importlib
    import sys

    assert hasattr(torch.ops.vllm_omni, "ar_diffusion_paged_write_attn")
    mod = "vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention"
    saved = sys.modules.pop(mod)
    try:
        importlib.import_module(mod)  # re-registration must not raise
    finally:
        sys.modules[mod] = saved
    assert hasattr(torch.ops.vllm_omni, "ar_diffusion_paged_write_attn")


@pytest.mark.cpu
def test_custom_op_mutable_arguments_cannot_be_elided_as_defaults():
    # Older PyTorch ADInplaceOrView handlers index positional mutable inputs
    # directly. Default-valued trailing inputs can disappear before that handler.
    schema = torch.ops.vllm_omni.ar_diffusion_paged_write_attn.default._schema
    mutable = [arg for arg in schema.arguments if arg.alias_info is not None and arg.alias_info.is_write]
    assert {arg.name for arg in mutable} == {"key_pool", "value_pool", "stage_key", "stage_value"}
    assert all(not arg.has_default_value() for arg in mutable)


@pytest.mark.parametrize("reuse_history_staging", [False, True])
@pytest.mark.cpu
def test_custom_op_compiles_fullgraph_without_recompile_on_value_change(monkeypatch, reuse_history_staging):
    """The op must trace as one opaque node: fullgraph OK, and changed tensor
    VALUES (new slots / block ids) must not trigger recompilation."""
    import torch._dynamo

    device = torch.device("cpu")
    monkeypatch.setenv(KV_GATHER_ENV, "1")
    kv, st = make_state(num_layers=2, window_chunks=2, reuse_history_staging=reuse_history_staging)
    if reuse_history_staging:
        # Hold the host-side staging offset fixed while table/slot values change.
        _commit_video_span(kv, st, kv_branch=POS, n_chunks=2, dtype=torch.float32, device=device)

    def run_one_forward(commit):
        contexts = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=commit)
        fctx = contexts[0].forward_ctx
        fctx.prepare(device=device, action_len=0, query_len=BLOCK)
        q = torch.randn(BLOCK, N_HEADS, HEAD_DIM)
        k = torch.randn(BLOCK, N_HEADS, HEAD_DIM)
        v = torch.randn(BLOCK, N_HEADS, HEAD_DIM)
        # Both layers use one compiled function; the tensor-valued layer index
        # must not specialize it.
        for layer_ctx in contexts:
            out = compiled(layer_ctx.to_layer_inputs(), q, k, v)
        st.commit_paged_context(POS)
        return out

    torch._dynamo.reset()
    try:
        from torch._dynamo.testing import CompileCounter

        counter = CompileCounter()

        def fn(inputs, q, k, v):
            return paged_write_attn(inputs, q, k, v, None, None, HEAD_DIM**-0.5) * 1.0

        compiled = torch.compile(fn, backend=counter, fullgraph=True)

        run_one_forward(commit=True)  # history grows between calls ->
        run_one_forward(commit=True)  # block-table VALUES change, shapes don't
        run_one_forward(commit=False)

        assert counter.frame_count == 1, f"recompiled: frame_count={counter.frame_count}"
    finally:
        # Leave a clean dynamo state for later suites in the same pytest process
        # (e.g. model_executor transformers models).
        torch._dynamo.reset()


# ── a frame that is not a whole number of blocks ────────────────────────────

# 24 tokens per chunk against 16-token blocks leaves 8 over, the same remainder
# the shipped 832x480 produces (1560 % 16 == 8). Every fixture above holds
# chunk_size == BLOCK, so nothing there can reach this path.
RAGGED_CHUNK = 24


@pytest.mark.parametrize("commit_current", [False, True])
@pytest.mark.parametrize("action_len", [0, 3])
@pytest.mark.cpu
def test_a_ragged_chunk_still_matches_the_dense_reference(commit_current, action_len):
    """Attention reads a sequence, not a set of blocks.

    With one committed chunk of 24 tokens the history stops 8 slots into its
    last block. The committing path writes straight after it and is fine. The
    scratch path used to restart at slot zero of a fresh region, which left
    those 8 slots unwritten -- and since the kernel consumes the block table as
    one contiguous run, it read them as if they were tokens and shifted every
    token after them by 8. Shapes stayed correct throughout, so only the values
    showed it.
    """
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32
    # window_chunks=2 with one chunk of history keeps the whole sequence
    # resident, so the window does not also round up here; that case is
    # test_a_ragged_window_keeps_the_block_table_a_fixed_shape below.
    kv, st = make_state(dtype=dtype, device=device, window_chunks=2, chunk_size=RAGGED_CHUNK)

    history_k, history_v = _commit_video_span(
        kv, st, kv_branch=POS, n_chunks=1, dtype=dtype, device=device, chunk_size=RAGGED_CHUNK
    )

    ctx = st.get_kv_caches(POS, seq_len=RAGGED_CHUNK, commit_current=commit_current)[0].forward_ctx
    ctx.ensure_video_slots(device)
    assert ctx.start_offset == RAGGED_CHUNK % BLOCK == 8

    current_k = torch.randn(1, RAGGED_CHUNK, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    current_v = torch.randn(1, RAGGED_CHUNK, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    kv._k_pools[0][ctx.current_video_slot_mapping] = current_k[0]
    kv._v_pools[0][ctx.current_video_slot_mapping] = current_v[0]

    action_k = action_v = None
    if action_len:
        ctx.ensure_action_slots(action_len, device)
        action_k = torch.randn(1, action_len, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
        action_v = torch.randn(1, action_len, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
        kv._k_pools[0][ctx.action_slot_mapping] = action_k[0]
        kv._v_pools[0][ctx.action_slot_mapping] = action_v[0]

    query = torch.randn(1, RAGGED_CHUNK + action_len, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    block_table, query_start_loc, seq_lens, max_query_len, max_seq_len = ctx.build_block_table(
        action_len=action_len, query_len=query.shape[1], device=device
    )
    paged = ar_diffusion_paged_attention(
        query,
        kv.key_cache(0),
        kv.value_cache(0),
        block_table=block_table,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        softmax_scale=HEAD_DIM**-0.5,
        causal=False,
    )

    new_k = torch.cat([history_k, current_k], dim=1)
    new_v = torch.cat([history_v, current_v], dim=1)
    if action_len:
        new_k = torch.cat([new_k, action_k], dim=1)
        new_v = torch.cat([new_v, action_v], dim=1)
    ref = _dense_attention(query, new_k, new_v)

    torch.testing.assert_close(paged, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.cpu
def test_a_ragged_scratch_chunk_is_physically_continuous_with_its_history():
    """The write targets themselves must leave no hole.

    This is the property the values above depend on, asserted directly so a
    regression names the cause rather than showing a numeric mismatch.
    """
    device = torch.device("cpu")
    kv, st = make_state(device=device, window_chunks=2, chunk_size=RAGGED_CHUNK)
    _commit_video_span(kv, st, kv_branch=POS, n_chunks=1, dtype=torch.float32, device=device, chunk_size=RAGGED_CHUNK)

    ctx = st.get_kv_caches(POS, seq_len=RAGGED_CHUNK, commit_current=False)[0].forward_ctx
    ctx.ensure_video_slots(device)

    history_tail_block = ctx.history_block_ids[-1]
    # The chunk starts inside the history's own last block, not at a fresh one.
    assert ctx.current_video_block_ids[0] == history_tail_block
    first_slot = int(ctx.current_video_slot_mapping[0])
    assert first_slot == history_tail_block * BLOCK + 8

    # And the action region must not be charged for that managed block.
    ctx.ensure_action_slots(3, device)
    assert ctx._scratch_blocks_used == len(ctx.current_video_block_ids) - 1
    assert ctx.action_scratch_block_ids[0] not in ctx.current_video_block_ids


@pytest.mark.cpu
def test_a_ragged_window_keeps_the_block_table_a_fixed_shape():
    """The table's shape must not track how full the window is.

    With a sink the visible window is 2*24 + 24 = 72 tokens, which is 4.5
    blocks. Deriving the width by flooring that would understate the capacity,
    the max() against the live block count would take over, and the shape would
    change as the window filled -- recompiling the graph. The same floor would
    also let max_seq_len come out under the kv_len actually passed.
    """
    device = torch.device("cpu")
    kv, st = make_state(device=device, window_chunks=2, sink_chunks=1, chunk_size=RAGGED_CHUNK)
    # The window is deliberately not a whole number of blocks.
    max_video_tokens = kv.spec.sliding_window + kv.spec.sink_chunks * kv.spec.chunk_size
    assert max_video_tokens % BLOCK != 0

    widths: set[int] = set()
    kv_lens: list[int] = []
    for _ in range(6):
        ctx = st.get_kv_caches(POS, seq_len=RAGGED_CHUNK, commit_current=True)[0].forward_ctx
        ctx.ensure_video_slots(device)
        block_table, _, seq_lens, _, max_seq_len = ctx.build_block_table(
            action_len=0, query_len=RAGGED_CHUNK, device=device
        )
        widths.add(int(block_table.shape[1]))
        # A bound handed to the kernel has to actually bound the sequence.
        assert int(seq_lens[0]) <= max_seq_len
        kv_lens.append(int(seq_lens[0]))
        st.commit_paged_context(POS)

    assert len(widths) == 1, f"block table width varied across ticks: {sorted(widths)}"
    # The window has to have actually filled, or none of the above was tested.
    assert max(kv_lens) > min(kv_lens)
    assert max(kv_lens) <= ctx.max_video_blocks * BLOCK
    # Whole blocks are read, so each of the two boundaries -- the sink's end and
    # the recent window's start -- can contribute up to BLOCK - 1 tokens the
    # window no longer needs.
    assert max(kv_lens) <= max_video_tokens + 2 * (BLOCK - 1)


def _poison_pools(kv: ARDiffusionKVCache) -> None:
    for pool in (*kv._k_pools, *kv._v_pools):
        pool.fill_(float("nan"))


def _write_prepared_chunk(
    kv: ARDiffusionKVCache, ctx, *, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write fresh K/V for a prepared chunk, poisoning the rest of its blocks first.

    Blocks are handed out whole and a reused block keeps whatever it held, so a
    slot read without having been written would not look unusual. Every slot of
    the chunk's blocks that this chunk does not write is set to NaN first --
    except the leading slots of a block the history still occupies -- so reading
    one turns the attention output NaN.
    """
    mapping = ctx.current_video_slot_mapping
    first_slot = int(mapping[0])
    for index, block in enumerate(ctx.current_video_block_ids):
        start = first_slot if index == 0 else block * BLOCK
        kv._k_pools[0][start : (block + 1) * BLOCK] = float("nan")
        kv._v_pools[0][start : (block + 1) * BLOCK] = float("nan")
    k = torch.randn(ctx.seq_len, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(ctx.seq_len, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    kv._k_pools[0][mapping] = k
    kv._v_pools[0][mapping] = v
    return k, v


def _window_positions(end: int, *, sink: int, window: int, written, history: int, resident: set[int]) -> list[int]:
    """Token positions attention must read at ``end``.

    Every token of the current chunk, plus every written history token in a
    block that is still resident and holds a sink token or one of the window's
    most recent tokens.
    """
    start = end - window
    blocks = {p // BLOCK for p in range(min(sink, end))} | {p // BLOCK for p in range(max(start, 0), end)}
    return [p for p in sorted(written) if p >= history or (p // BLOCK in blocks and p // BLOCK in resident)]


def _model_block_indices(kv, adapter) -> dict[int, int]:
    """Each resident block's index in model positions, by block id.

    Compaction removes the evicted gap after the sink's blocks and shifts the
    rest of the table down by it, so a table index past the sink is
    ``compacted_tokens`` behind the position it holds.
    """
    block_size = kv.block_size
    sink_blocks = -(-(kv.spec.sink_chunks * kv.spec.chunk_size) // block_size)
    shift = adapter.compacted_tokens // block_size
    return {
        int(block): index if index < sink_blocks else index + shift
        for index, block in enumerate(kv.block_table(adapter))
        if block != kv.null_block_id
    }


@pytest.mark.parametrize(
    ("sink_chunks", "window_chunks", "reset_at_boundary"),
    [(1, 2, False), (0, 2, False), (1, 3, False), (2, 2, True), (0, 2, True)],
)
@pytest.mark.cpu
def test_attention_after_eviction_reads_every_kept_token_and_nothing_unwritten(
    sink_chunks, window_chunks, reset_at_boundary
):
    """Once the window slides, the table must still cover the window exactly.

    The sink and the recent window are separate token ranges, and each can
    straddle a block edge on its own: 24-token chunks against 16-token blocks
    put a boundary 8 tokens past an edge on alternating chunks. Counting blocks
    off the ends of the resident list dropped the block holding the window's
    first tokens whenever both boundaries straddled, and the returned length
    ran past the last written slot. Neither changes a shape, so this compares
    values against dense attention over the tokens the window keeps -- after
    eviction, with every unwritten slot poisoned, on both the scratch and the
    committing forward of every chunk.

    The reset case keeps only the sink across a boundary, so the recent window
    reaches back over blocks that are no longer resident. Those must be skipped
    by position rather than read.

    Eight chunks are enough for compaction to drop the evicted gap from the
    table, after which storage positions run behind model positions. Removing
    a gap that is not a whole number of chunks moves where eviction snaps, and
    in the reset case the window then keeps tokens a boundary should have
    dropped.
    """
    torch.manual_seed(0)
    device, dtype = torch.device("cpu"), torch.float32
    kv, st = make_state(
        device=device,
        window_chunks=window_chunks,
        sink_chunks=sink_chunks,
        reset_at_boundary=reset_at_boundary,
        chunk_size=RAGGED_CHUNK,
    )
    _poison_pools(kv)
    sink, window = sink_chunks * RAGGED_CHUNK, int(kv.spec.sliding_window)
    committed: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    evicted = compacted = False

    for chunk in range(8):
        history = int(st.adapter(POS).absolute_num_computed_tokens)
        end = history + RAGGED_CHUNK
        for commit_current in (False, True):
            ctx = st.get_kv_caches(POS, seq_len=RAGGED_CHUNK, commit_current=commit_current)[0].forward_ctx
            ctx.ensure_video_slots(device)
            k, v = _write_prepared_chunk(kv, ctx, dtype=dtype, device=device)
            written = dict(committed)
            written.update({history + i: (k[i], v[i]) for i in range(RAGGED_CHUNK)})

            query = torch.randn(1, RAGGED_CHUNK, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
            block_table, query_start_loc, seq_lens, max_query_len, max_seq_len = ctx.build_block_table(
                action_len=0, query_len=RAGGED_CHUNK, device=device
            )
            resident = set(_model_block_indices(kv, st.adapter(POS)).values())
            positions = _window_positions(
                end, sink=sink, window=window, written=written, history=history, resident=resident
            )
            label = f"chunk {chunk}, commit_current={commit_current}"
            assert int(seq_lens[0]) == len(positions), f"{label}: kv_len does not match the written slots it covers"
            paged = ar_diffusion_paged_attention(
                query,
                kv.key_cache(0),
                kv.value_cache(0),
                block_table=block_table,
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                max_query_len=max_query_len,
                max_seq_len=max_seq_len,
                softmax_scale=HEAD_DIM**-0.5,
                causal=False,
            )
            ref = _dense_attention(
                query,
                torch.stack([written[p][0] for p in positions]).unsqueeze(0),
                torch.stack([written[p][1] for p in positions]).unsqueeze(0),
            )
            torch.testing.assert_close(paged, ref, rtol=1e-5, atol=1e-5, msg=lambda m, label=label: f"{label}: {m}")
        committed = written
        st.commit_paged_context(POS)
        compacted = compacted or st.adapter(POS).compacted_tokens > 0
        evicted = evicted or compacted or kv.null_block_id in kv.block_table(st.adapter(POS))

    assert evicted, "the window never slid, so nothing after eviction was tested"
    assert compacted, "the table was never compacted, so storage and model positions never diverged"


@pytest.mark.parametrize(
    ("sink_chunks", "window_chunks", "reset_at_boundary"),
    [(1, 2, False), (0, 2, False), (1, 3, False), (2, 2, True), (0, 2, True)],
)
@pytest.mark.cpu
def test_compaction_does_not_change_which_tokens_stay_resident(
    monkeypatch, sink_chunks, window_chunks, reset_at_boundary
):
    """Compaction renumbers the block table; it must not change what eviction keeps.

    Storage positions run ``compacted_tokens`` behind model positions, and
    eviction snaps to chunk boundaries counted in storage positions. Removing a
    gap that is not a whole number of chunks moves every later snap, so eviction
    keeps a different set of tokens -- in the reset case, tokens a boundary
    should have dropped, which the window then reads. Attention checked against
    what is resident cannot see that, so the oracle here is the same rollout
    with compaction switched off.
    """

    def resident_blocks_per_tick(compact: bool) -> tuple[list[set[int]], int]:
        with monkeypatch.context() as patch:
            if not compact:
                patch.setattr(ChunkWindowManager, "compact_block_table", lambda self, request_id: 0)
            kv, st = make_state(
                window_chunks=window_chunks,
                sink_chunks=sink_chunks,
                reset_at_boundary=reset_at_boundary,
                chunk_size=RAGGED_CHUNK,
            )
            ticks = []
            for _ in range(12):
                ctx = st.get_kv_caches(POS, seq_len=RAGGED_CHUNK, commit_current=True)[0].forward_ctx
                ctx.ensure_video_slots(torch.device("cpu"))
                ticks.append(set(_model_block_indices(kv, st.adapter(POS)).values()))
                st.commit_paged_context(POS)
            return ticks, st.adapter(POS).compacted_tokens

    expected, _ = resident_blocks_per_tick(compact=False)
    actual, compacted_tokens = resident_blocks_per_tick(compact=True)
    assert compacted_tokens > 0, "the table was never compacted, so nothing was tested"
    for tick, (want, got) in enumerate(zip(expected, actual)):
        assert got == want, f"tick {tick}: compaction kept blocks {sorted(got - want)} and dropped {sorted(want - got)}"


@pytest.mark.cpu
def test_history_staging_holds_a_ragged_window_and_restages_it_whole(monkeypatch):
    """The staging buffers must fit the table width this module pads to.

    A sink or window that is not a whole number of blocks spans one more block
    than its token count suggests, so sizing staging from tokens left it short
    and the first prepared forward refused to run. And because such a chunk
    shares its first block with the history's tail, the history a probe sees is
    not left untouched by the next probe, so it is restaged rather than reused.
    """
    monkeypatch.setenv(KV_GATHER_ENV, "1")
    device = torch.device("cpu")
    kv, st = make_state(
        device=device, window_chunks=2, sink_chunks=1, chunk_size=RAGGED_CHUNK, reuse_history_staging=True
    )
    capacity = int(kv.history_staging[0][0].shape[0])
    for chunk in range(6):
        for probe in range(2):
            ctx = st.get_kv_caches(POS, seq_len=RAGGED_CHUNK, commit_current=False)[0].forward_ctx
            ctx.prepare(device=device, action_len=0, query_len=RAGGED_CHUNK)
            assert int(ctx.max_seq_len) <= capacity
            assert not ctx.reuse_history, f"chunk {chunk}, probe {probe}: a ragged chunk reused the staged history"
        ctx = st.get_kv_caches(POS, seq_len=RAGGED_CHUNK, commit_current=True)[0].forward_ctx
        ctx.prepare(device=device, action_len=0, query_len=RAGGED_CHUNK)
        st.commit_paged_context(POS)


@pytest.mark.cpu
@pytest.mark.parametrize("frame_tokens", [24, 394, 924])
@pytest.mark.parametrize("commit_current,frame_causal", [(False, False), (True, False), (True, True)])
def test_partial_video_pages_with_text_match_dense(frame_tokens, commit_current, frame_causal):
    from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import paged_write_attn

    torch.manual_seed(7)
    kv, st = make_state(chunk_size=frame_tokens, window_chunks=8)
    try:
        history_k, history_v = _commit_video_span(
            kv,
            st,
            kv_branch=POS,
            n_chunks=1,
            dtype=torch.float32,
            device=torch.device("cpu"),
            chunk_size=frame_tokens,
        )
        count = 2 * frame_tokens
        ctx = st.get_kv_caches(
            POS,
            seq_len=count,
            commit_current=commit_current,
            frame_causal=frame_causal,
            extra_visible_tokens=frame_tokens,
        )[0].forward_ctx
        ctx.prepare(torch.device("cpu"), action_len=13, query_len=count)
        q, k, v = [torch.randn(count, N_HEADS, HEAD_DIM) for _ in range(3)]
        kt, vt = [torch.randn(13, N_HEADS, HEAD_DIM) for _ in range(2)]
        inputs = ctx.layer_inputs(0)
        actual = paged_write_attn(inputs, q, k, v, kt, vt, HEAD_DIM**-0.5, framewise_attention=frame_causal)
        for frame in range(2 if frame_causal else 1):
            end = (frame + 1) * frame_tokens if frame_causal else count
            start = frame * frame_tokens if frame_causal else 0
            keys = torch.cat([history_k.flatten(0, 1), k[:end], kt]).unsqueeze(0)
            values = torch.cat([history_v.flatten(0, 1), v[:end], vt]).unsqueeze(0)
            expected = _dense_attention(q[start:end].unsqueeze(0), keys, values)[0]
            torch.testing.assert_close(actual[start:end], expected)
        if frame_causal:
            poisoned = v.clone()
            poisoned[frame_tokens:] = 1000
            again = paged_write_attn(inputs, q, k, poisoned, kt, vt, HEAD_DIM**-0.5, framewise_attention=True)
            torch.testing.assert_close(again[:frame_tokens], actual[:frame_tokens], rtol=0, atol=0)
            paged_write_attn(inputs, q, k, v, kt, vt, HEAD_DIM**-0.5, framewise_attention=True)
        st.commit_paged_context(POS)
        if commit_current:
            table = kv.block_table(st.adapter(POS))
            from vllm_omni.experimental.ar_diffusion.kv_cache.paged import compute_slot_mapping

            slots = compute_slot_mapping(table, torch.arange(3 * frame_tokens), BLOCK)
            torch.testing.assert_close(kv._k_pools[0][slots], torch.cat([history_k.flatten(0, 1), k]))
            torch.testing.assert_close(kv._v_pools[0][slots], torch.cat([history_v.flatten(0, 1), v]))
    finally:
        st.close()


@pytest.mark.cpu
def test_the_shipped_geometry_reads_exactly_the_window_on_every_tick():
    """832x480 with the checkpoint's own sink and window, slot by slot.

    A 9-frame sink and a 9-frame window are each 8 tokens past a 16-token edge
    but 28080 tokens together -- a whole number of blocks -- so a width taken
    from their sum was one block short on every tick once the window had
    filled. Twelve ticks of three frames, checking every slot the kernel reads
    against the token that slot is supposed to hold. No attention is computed:
    at this size the bookkeeping is the whole question.
    """
    from vllm_omni.experimental.ar_diffusion.runner import paging_block_size

    frame, frames_per_tick = (480 // 16) * (832 // 16), 3
    block = paging_block_size(frame)
    seq_len = frames_per_tick * frame
    cfg = ARDiffusionKVConfig(enable=True, chunk_size=frame, window_chunks=9, sink_chunks=9)
    kv = ARDiffusionKVCache(
        cfg,
        num_layers=1,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float16,
        block_size=block,
        max_model_len=1 << 20,
        available_bytes=1 << 27,
        kv_branches=(ARDiffusionKVBranchSpec(POS, 0),),
        session_capacity=1,
        frames_per_block=frames_per_tick,
        max_scratch_tokens_per_branch=seq_len,
        device=torch.device("cpu"),
    )
    st = ARDiffusionKVState(kv, "s1", {POS: kv.begin_request("r-pos")}, num_layers=1)
    sink, window = 9 * frame, int(kv.spec.sliding_window)
    holds: dict[int, int] = {}  # slot -> token position last written there
    widths: set[int] = set()
    filled = compacted = False

    for tick in range(12):
        history = int(st.adapter(POS).absolute_num_computed_tokens)
        end = history + seq_len
        ctx = st.get_kv_caches(POS, seq_len=seq_len, commit_current=True)[0].forward_ctx
        ctx.ensure_video_slots(torch.device("cpu"))
        for offset, slot in enumerate(ctx.current_video_slot_mapping.tolist()):
            holds[slot] = history + offset
        table, _, seq_lens, _, _ = ctx.build_block_table(action_len=0, query_len=seq_len, device=torch.device("cpu"))
        widths.add(int(table.shape[1]))
        kv_len = int(seq_lens[0])

        position_of_block = {b: i * block for b, i in _model_block_indices(kv, st.adapter(POS)).items()}
        read: list[tuple[int, int]] = []
        for b in table[0].tolist()[: -(-kv_len // block)]:
            assert b in position_of_block, f"tick {tick}: read block {b}, which holds no token of this session"
            read.extend((b * block + o, position_of_block[b] + o) for o in range(block))
        read = read[:kv_len]
        wrong = [(slot, want) for slot, want in read if holds.get(slot) != want]
        assert not wrong, f"tick {tick}: {len(wrong)} slots read hold a token other than the one their position implies"

        start = end - window
        kept = set(range(min(sink, end))) | set(range(max(start, 0), end))
        missed = kept - {want for _, want in read}
        assert not missed, f"tick {tick}: {len(missed)} tokens the window keeps were not read"

        filled = filled or end > sink + window
        st.commit_paged_context(POS)
        compacted = compacted or st.adapter(POS).compacted_tokens > 0

    assert filled, "the window never filled, so the case that fails was never reached"
    assert compacted, "the table was never compacted, so storage and model positions never diverged"
    assert len(widths) == 1, f"block table width varied across ticks: {sorted(widths)}"


class _CountedBlockIds:
    """Wraps a vLLM block manager's ``get_blocks`` and counts the block ids read.

    Every path to a request's block ids goes through ``get_blocks``, so a copy of
    the whole table is counted in full even if only a few entries are indexed.
    """

    def __init__(self, manager):
        self.reads = 0
        self._get_blocks = manager.get_blocks

    def __call__(self, request_id):
        from vllm.v1.core.kv_cache_manager import KVCacheBlocks

        blocks = self._get_blocks(request_id)
        return KVCacheBlocks(tuple([_CountedBlock(block, self) for block in group] for group in blocks.blocks))


class _CountedBlock:
    def __init__(self, block, counter: _CountedBlockIds):
        self._block = block
        self._counter = counter

    @property
    def block_id(self) -> int:
        self._counter.reads += 1
        return self._block.block_id


@pytest.mark.cpu
def test_reading_the_window_touches_only_the_blocks_it_can_keep(monkeypatch):
    """However long the block table is, reading the window must not cost more.

    Evicted positions stay in the table as null entries until compaction drops
    them a whole number of chunks at a time -- up to two frames' worth of blocks
    at 832x480 -- so the table can still be far longer than the window. Only the
    sink's blocks and the recent window's blocks can hold a kept token, so those
    are all the read may touch. Compaction is switched off here so the table
    grows to many times the window, and reads are counted where the ids leave
    vLLM's block manager.
    """
    monkeypatch.setattr(ChunkWindowManager, "compact_block_table", lambda self, request_id: 0)
    device = torch.device("cpu")
    kv, st = make_state(device=device, window_chunks=2, sink_chunks=1, chunk_size=RAGGED_CHUNK)
    reads: list[int] = []
    table_lengths: list[int] = []
    for _ in range(60):
        ctx = st.get_kv_caches(POS, seq_len=RAGGED_CHUNK, commit_current=True)[0].forward_ctx
        ctx.ensure_video_slots(device)
        counted = _CountedBlockIds(kv.manager)
        kv.manager.get_blocks = counted
        try:
            ctx.build_block_table(action_len=0, query_len=RAGGED_CHUNK, device=device)
        finally:
            del kv.manager.get_blocks
        reads.append(counted.reads)
        table_lengths.append(len(kv.block_table(st.adapter(POS))))
        st.commit_paged_context(POS)

    assert table_lengths[-1] > 10 * ctx.max_video_blocks, "the table never outgrew the window, so nothing was tested"
    assert max(reads) <= ctx.max_video_blocks, (
        f"read {max(reads)} table entries for a {ctx.max_video_blocks}-block window"
    )


@pytest.mark.cpu
def test_the_checkpoints_own_default_resolution_can_build_a_cache():
    """832x480 is what LingBot World v2 ships as its default, and it could not run.

    A frame-sized block made the resolution a kernel-compatibility question:
    1560 tokens per frame is not a multiple of 16, so FlashAttention's paged
    kernel rejected it and the default resolution had no realtime path at all.
    """
    from vllm_omni.experimental.ar_diffusion.runner import paging_block_size

    tokens_per_frame = (480 // 16) * (832 // 16)
    assert tokens_per_frame == 1560
    assert tokens_per_frame % 16 == 8, "the whole point is that a frame is not a legal block"

    block_size = paging_block_size(tokens_per_frame)
    cfg = ARDiffusionKVConfig(enable=True, chunk_size=tokens_per_frame, window_chunks=2)
    kv = ARDiffusionKVCache(
        cfg,
        num_layers=1,
        num_kv_heads=N_HEADS,
        head_size=HEAD_DIM,
        dtype=torch.float32,
        block_size=block_size,
        max_model_len=1 << 16,
        available_bytes=1 << 28,
        kv_branches=(ARDiffusionKVBranchSpec(POS, 0),),
        session_capacity=1,
        frames_per_block=2,
        max_scratch_tokens_per_branch=block_size,
        device=torch.device("cpu"),
    )

    assert kv.block_size % 16 == 0
    # The eviction unit is still the frame; only the paging unit changed.
    assert kv.spec.chunk_size == tokens_per_frame
    assert kv.blocks_per_frame == -(-tokens_per_frame // block_size) == 98


@pytest.mark.cpu
def test_the_shipped_resolution_geometry_matches_dense_attention():
    """The real number, not a scaled-down stand-in: 1560 tokens per frame.

    This is the case the issue reported failing at ~4e-02 on the scratch path.
    It runs on CPU because the reference is dense attention, not a kernel.
    """
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32
    tokens_per_frame = (480 // 16) * (832 // 16)
    assert tokens_per_frame == 1560

    kv, st = make_state(dtype=dtype, device=device, window_chunks=2, chunk_size=tokens_per_frame)
    history_k, history_v = _commit_video_span(
        kv, st, kv_branch=POS, n_chunks=1, dtype=dtype, device=device, chunk_size=tokens_per_frame
    )

    # The non-committing path is the one that was wrong: four of the five
    # forwards per generated block take it.
    ctx = st.get_kv_caches(POS, seq_len=tokens_per_frame, commit_current=False)[0].forward_ctx
    ctx.ensure_video_slots(device)
    assert ctx.start_offset == 8, "1560 % 16 == 8 is the whole reason this case exists"

    current_k = torch.randn(1, tokens_per_frame, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    current_v = torch.randn(1, tokens_per_frame, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    kv._k_pools[0][ctx.current_video_slot_mapping] = current_k[0]
    kv._v_pools[0][ctx.current_video_slot_mapping] = current_v[0]

    query = torch.randn(1, tokens_per_frame, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    block_table, query_start_loc, seq_lens, max_query_len, max_seq_len = ctx.build_block_table(
        action_len=0, query_len=tokens_per_frame, device=device
    )
    paged = ar_diffusion_paged_attention(
        query,
        kv.key_cache(0),
        kv.value_cache(0),
        block_table=block_table,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        softmax_scale=HEAD_DIM**-0.5,
        causal=False,
    )
    ref = _dense_attention(
        query,
        torch.cat([history_k, current_k], dim=1),
        torch.cat([history_v, current_v], dim=1),
    )
    torch.testing.assert_close(paged, ref, rtol=1e-5, atol=1e-5)


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("commit_current", [False, True])
def test_the_shipped_resolution_geometry_on_the_real_kernel(commit_current):
    """The shipped 832x480 geometry through FlashAttention's paged kernel.

    The CPU test above proves the addressing is right against a dense
    reference. This one proves the real kernel agrees, at the real 1560
    tokens per frame, through the production path -- host prep followed by
    the fused write+attend op, which is what a forward actually calls.
    """
    _require_gpu_flash_attn()
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float16
    tokens_per_frame = (480 // 16) * (832 // 16)
    assert tokens_per_frame == 1560

    kv, st = make_state(dtype=dtype, device=device, window_chunks=2, chunk_size=tokens_per_frame)
    history_k, history_v = _commit_video_span(
        kv, st, kv_branch=POS, n_chunks=1, dtype=dtype, device=device, chunk_size=tokens_per_frame
    )

    layer_ctx = st.get_kv_caches(POS, seq_len=tokens_per_frame, commit_current=commit_current)[0]
    ctx = layer_ctx.forward_ctx
    current_k = torch.randn(1, tokens_per_frame, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    current_v = torch.randn(1, tokens_per_frame, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    query = torch.randn(1, tokens_per_frame, N_HEADS, HEAD_DIM, dtype=dtype, device=device)

    ctx.prepare(device=device, action_len=0, query_len=tokens_per_frame)
    assert ctx.start_offset == 8, "1560 % 16 == 8 is the whole reason this case exists"
    assert kv.block_size == 16

    inputs = layer_ctx.to_layer_inputs()
    paged = paged_write_attn(inputs, query[0], current_k[0], current_v[0], None, None, HEAD_DIM**-0.5).unsqueeze(0)

    new_k = torch.cat([history_k, current_k], dim=1)[:, -kv.spec.sliding_window :]
    new_v = torch.cat([history_v, current_v], dim=1)[:, -kv.spec.sliding_window :]
    ref = _dense_attention(query, new_k, new_v)

    torch.testing.assert_close(paged, ref, rtol=2e-2, atol=2e-2)


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("history_chunks", [0, 1, 3])
@pytest.mark.parametrize("action_len", [0, 3])
def test_contiguous_kv_gather_path_matches_paged_path_gpu(monkeypatch, history_chunks, action_len):
    """VLLM_OMNI_AR_DIFFUSION_KV_GATHER=1 gathers the visible blocks and runs varlen FA3 without a block table.

    Covers an empty, partial and full window plus a partially filled action block:
    the gather reads the tail-padding null block, so its (zeroed) rows must be masked.
    """
    _require_gpu_flash_attn()

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float16
    kv, st = make_state(dtype=dtype, device=device, window_chunks=2)
    if history_chunks:
        _commit_video_span(kv, st, kv_branch=POS, n_chunks=history_chunks, dtype=dtype, device=device)

    layer_ctx = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=False)[0]
    ctx = layer_ctx.forward_ctx
    current_k = torch.randn(1, BLOCK, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    current_v = torch.randn(1, BLOCK, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    action_k = action_v = None
    if action_len:
        action_k = torch.randn(1, action_len, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
        action_v = torch.randn(1, action_len, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    query = torch.randn(1, BLOCK + action_len, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    ctx.prepare(device=device, action_len=action_len, query_len=query.shape[1])
    inputs = layer_ctx.to_layer_inputs()

    def run() -> torch.Tensor:
        return paged_write_attn(
            inputs,
            query[0],
            current_k[0],
            current_v[0],
            action_k[0] if action_k is not None else None,
            action_v[0] if action_v is not None else None,
            HEAD_DIM**-0.5,
        ).unsqueeze(0)

    monkeypatch.delenv(KV_GATHER_ENV, raising=False)
    paged = run()
    monkeypatch.setenv(KV_GATHER_ENV, "1")
    gathered = run()
    assert torch.isfinite(gathered).all()
    # Same kernel family on the same K/V: only accumulation order differs.
    torch.testing.assert_close(gathered, paged, rtol=2e-3, atol=2e-3)


@pytest.mark.cpu
@pytest.mark.parametrize("sink", [0, 1])
@pytest.mark.parametrize("reset", [False, True])
def test_ragged_batched_refresh_matches_sequential_after_eviction(sink, reset):
    from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import paged_write_attn

    torch.manual_seed(11)
    frame = 24
    states = [
        make_state(chunk_size=frame, window_chunks=2, sink_chunks=sink, reset_at_boundary=reset)[1] for _ in range(2)
    ]
    try:
        for tick in range(10):
            q, k, v = [torch.randn(2 * frame, N_HEADS, HEAD_DIM) for _ in range(3)]
            kt, vt = [torch.randn(3, N_HEADS, HEAD_DIM) for _ in range(2)]
            outputs = []
            for state, batched in zip(states, [True, False]):
                parts = []
                for start in range(0, 2 * frame, 2 * frame if batched else frame):
                    end = 2 * frame if batched else start + frame
                    ctx = state.get_kv_caches(
                        POS, seq_len=end - start, commit_current=True, frame_causal=batched, extra_visible_tokens=frame
                    )[0].forward_ctx
                    ctx.prepare(torch.device("cpu"), action_len=3, query_len=end - start)
                    parts.append(
                        paged_write_attn(
                            ctx.layer_inputs(0),
                            q[start:end],
                            k[start:end],
                            v[start:end],
                            kt,
                            vt,
                            HEAD_DIM**-0.5,
                            framewise_attention=batched,
                        )
                    )
                    state.commit_paged_context(POS)
                outputs.append(torch.cat(parts))
            torch.testing.assert_close(outputs[0], outputs[1], msg=f"tick {tick}")
    finally:
        for state in states:
            state.close()
