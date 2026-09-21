# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for AR-Diffusion paged self-attention contexts."""

from __future__ import annotations

import subprocess
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec
from vllm_omni.experimental.ar_diffusion.kv_cache import (
    ARDiffusionKVCache,
    ARDiffusionKVConfig,
    ARDiffusionPagedLayerContext,
    ARDiffusionPagedLayerInputs,
    ar_diffusion_paged_attention,
    paged_write_attn,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


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

    ctx = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=commit_current)[0].forward_ctx
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

    before = st.adapter(POS).completed_chunks
    st.commit_paged_context(POS)
    assert st.adapter(POS).completed_chunks == before + (1 if commit_current else 0)


@pytest.mark.skipif(not _gpu_flash_attn_usable(), reason="usable GPU FlashAttention is required")
@pytest.mark.parametrize("history_chunks", [1, 3])
@pytest.mark.parametrize("action_len", [0, 3])
@pytest.mark.parametrize("commit_current", [False, True])
def test_paged_attention_matches_dense_reference_gpu(history_chunks, action_len, commit_current):
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
        action_k[0] if action_len else None,
        action_v[0] if action_len else None,
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


def test_layer_inputs_before_prepare_raises():
    _, st = make_state()
    layer_ctx = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=False)[0]
    with pytest.raises(RuntimeError, match="before prepare"):
        layer_ctx.to_layer_inputs()


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


def test_custom_op_compiles_fullgraph_without_recompile_on_value_change():
    """The op must trace as one opaque node: fullgraph OK, and changed tensor
    VALUES (new slots / block ids) must not trigger recompilation."""
    import torch._dynamo

    device = torch.device("cpu")
    kv, st = make_state(num_layers=2, window_chunks=2)

    def run_one_forward(commit):
        contexts = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=commit)
        fctx = contexts[0].forward_ctx
        fctx.prepare(device=device, action_len=0, query_len=BLOCK)
        q = torch.randn(BLOCK, N_HEADS, HEAD_DIM)
        k = torch.randn(BLOCK, N_HEADS, HEAD_DIM)
        v = torch.randn(BLOCK, N_HEADS, HEAD_DIM)
        # Both layers through ONE compiled fn: layer_idx is a tensor, so a
        # different layer must NOT recompile (all 40 DiT blocks share the
        # block-forward code object in production).
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


@pytest.mark.parametrize(
    ("sink_chunks", "window_chunks", "reset_at_boundary"),
    [(1, 2, False), (0, 2, False), (1, 3, False), (2, 2, True)],
)
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
    evicted = False

    for chunk in range(8):
        history = int(st.adapter(POS).num_computed_tokens)
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
            resident = {
                index for index, block in enumerate(kv.block_table(st.adapter(POS))) if block != kv.null_block_id
            }
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
        evicted = evicted or kv.null_block_id in kv.block_table(st.adapter(POS))

    assert evicted, "the window never slid, so nothing after eviction was tested"


@pytest.mark.parametrize("commit_current", [False, True])
def test_action_tokens_after_a_partly_written_video_block_are_refused(commit_current):
    """Action K/V follows the video blocks in the table, and the kernel reads it as one run.

    A 24-token chunk from an empty history ends 8 slots into its second 16-token
    block, so the run would read those 8 unwritten slots as the first action
    tokens and never reach the last ones. Every shape stays right, which is why
    this is refused rather than computed.
    """
    _, st = make_state(window_chunks=2, chunk_size=RAGGED_CHUNK)
    ctx = st.get_kv_caches(POS, seq_len=RAGGED_CHUNK, commit_current=commit_current)[0].forward_ctx
    with pytest.raises(ValueError, match="partly written video block"):
        ctx.build_block_table(action_len=3, query_len=RAGGED_CHUNK + 3, device=torch.device("cpu"))


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
    filled = False

    for tick in range(12):
        history = int(st.adapter(POS).num_computed_tokens)
        end = history + seq_len
        ctx = st.get_kv_caches(POS, seq_len=seq_len, commit_current=True)[0].forward_ctx
        ctx.ensure_video_slots(torch.device("cpu"))
        for offset, slot in enumerate(ctx.current_video_slot_mapping.tolist()):
            holds[slot] = history + offset
        table, _, seq_lens, _, _ = ctx.build_block_table(action_len=0, query_len=seq_len, device=torch.device("cpu"))
        widths.add(int(table.shape[1]))
        kv_len = int(seq_lens[0])

        position_of_block = {
            b: i * block for i, b in enumerate(kv.block_table(st.adapter(POS))) if b != kv.null_block_id
        }
        read = []
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

    assert filled, "the window never filled, so the case that fails was never reached"
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


def test_reading_the_window_touches_only_the_blocks_it_can_keep():
    """The block table grows with the session; reading the window must not.

    Every evicted position stays in the table as a null entry, so walking the
    whole table costs more on every tick of a long session. Only the sink's
    blocks and the recent window's blocks can hold a kept token, so those are
    all the read may touch -- checked on a session long enough that the table is
    many times the window, and counted where the ids leave vLLM's block manager.
    """
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


@pytest.mark.skipif(not _gpu_flash_attn_usable(), reason="usable GPU FlashAttention is required")
@pytest.mark.parametrize("commit_current", [False, True])
def test_the_shipped_resolution_geometry_on_the_real_kernel(commit_current):
    """The shipped 832x480 geometry through FlashAttention's paged kernel.

    The CPU test above proves the addressing is right against a dense
    reference. This one proves the real kernel agrees, at the real 1560
    tokens per frame, through the production path -- host prep followed by
    the fused write+attend op, which is what a forward actually calls.
    """
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
