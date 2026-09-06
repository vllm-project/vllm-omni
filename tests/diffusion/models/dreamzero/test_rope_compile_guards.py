# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Guard-stability tests for DreamZero's hoisted RoPE path.

Hoisting the RoPE table out of the compiled DiT block removes 79 redundant rebuilds per
forward. It does not, on its own, stop the block being *retraced* per query length. The
deployed geometry runs three: 880 (first prefill chunk), 1760 (later prefill) and 1785
(diffuse, 2x880 video + 24 action + 1 state), and Dynamo specializes on each. Three
things together collapse them to one graph, and this module covers all three:

* ``_materialize_block_freqs`` returns a fresh allocation rather than a view, so the
  table can carry a ``mark_dynamic`` hint at all, and so both branches of
  ``causal_rope_action_freqs`` hand the block the same flavour of tensor;
* ``paged_write_attn`` derives ``max_query_len`` from ``query.shape[0]``, keeping the
  varlen bound a symbolic int instead of a Dynamo value guard;
* the block's compilation count does not grow with sequence length or AR chunk position.

The first two are cheap, CPU-only and run by default. The third builds a real
``CausalWanAttentionBlock`` and compiles it with a counting backend, which is slow and
hardware-sensitive, so it is opt-in. Run it on the target accelerator when touching this
path::

    VLLM_OMNI_RUN_DREAMZERO_COMPILE_GUARD_TEST=1 \
        pytest -sv tests/diffusion/models/dreamzero/test_rope_compile_guards.py

It needs no engine, no KV pool and no distributed launcher: TP=1 is initialized in
process and the attention kernel is stubbed, so what remains under test is exactly the
RoPE/freqs plumbing and the block signature.
"""

import os

import pytest
import torch
import torch._dynamo

from vllm_omni.diffusion.models.dreamzero.causal_wan_model import (
    _mark_seq_dynamic,
    causal_rope_action_freqs,
    rope_params,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_HEAVY_ENV = "VLLM_OMNI_RUN_DREAMZERO_COMPILE_GUARD_TEST"

# num_frame_per_block=1 in the deployed config, so action_state_index == chunk_pos - 1.
NUM_FRAME_PER_BLOCK = 1
NUM_ACTION = 24
NUM_STATE = 1
ARL = NUM_ACTION + NUM_STATE  # 25 action/state registers
DIM = 256  # small, but structurally identical to the deployed block
HEADS = 4
HEAD_DIM = DIM // HEADS
VIDEO_LENS = (880, 1760)  # prefill geometries, no action registers
CHUNK_POSITIONS = (1, 3, 5, 7, 9, 11, 13, 15)  # AR positions, including unseen ones


def _action_state_index(chunk_pos: int) -> int:
    """The same expression ``_forward_blocks`` evaluates before the block loop."""
    return max(0, (chunk_pos - 1) // NUM_FRAME_PER_BLOCK)


def _video_freqs(video_len: int, head_dim: int = HEAD_DIM, device: str = "cpu") -> torch.Tensor:
    """Video RoPE table shaped the way ``_create_freqs`` hands it down.

    Built by slicing ``rope_params`` rather than from ``randn``: the real table has unit
    modulus, and a random complex stand-in would carry magnitudes that blow bf16 up to
    NaN and hide numerical regressions.
    """
    return rope_params(1024 * 10, head_dim)[:video_len].reshape(video_len, 1, -1).to(device)


def _build_table(video_len: int, with_action: bool, chunk_pos: int = 1, device: str = "cpu"):
    freqs_action = rope_params(1024 * 10, HEAD_DIM).to(device)
    freqs_state = rope_params(1024, HEAD_DIM).to(device)
    return causal_rope_action_freqs(
        _video_freqs(video_len, device=device),
        freqs_action,
        freqs_state,
        ARL if with_action else None,
        NUM_ACTION,
        NUM_STATE,
        _action_state_index(chunk_pos),
    )


# ── The table must be markable, and the same flavour on both branches ────────────


@pytest.mark.parametrize("with_action", [False, True])
def test_block_freqs_is_a_fresh_allocation(with_action: bool) -> None:
    """Both branches return an allocation, not a view.

    This is what makes the ``mark_dynamic`` hint in ``_forward_blocks`` land. On a view,
    ``mark_dynamic`` can raise, ``_mark_seq_dynamic`` swallows it by design, and the
    result is one graph per query length with no error to show for it. It is also what
    makes the two branches' dispatch key sets agree: the no-action branch would
    otherwise view the table handed down from ``_create_freqs`` while the action branch
    views a local ``torch.cat``, and Dynamo guards on that difference.
    """
    video_len = 1760
    table = _build_table(video_len, with_action)

    assert table._base is None, "block freqs is a view; the mark_dynamic hint will not land"
    assert table.shape == (video_len + (ARL if with_action else 0), 1, HEAD_DIM // 2, 2)
    assert table.dtype == torch.float32  # rope_params narrows to complex64
    assert table.is_contiguous()

    # Called directly, not through _mark_seq_dynamic, so a failure is not swallowed.
    torch._dynamo.mark_dynamic(table, 0)


@pytest.mark.parametrize("with_action", [False, True])
def test_materializing_the_table_does_not_change_its_values(with_action: bool) -> None:
    """The allocation is a copy, so it must be bit-for-bit the table it replaced."""
    video_len = 880
    chunk_pos = 5
    table = _build_table(video_len, with_action, chunk_pos)

    freqs = _video_freqs(video_len)
    if with_action:
        idx = _action_state_index(chunk_pos)
        action = rope_params(1024 * 10, HEAD_DIM)[idx * NUM_ACTION : (idx + 1) * NUM_ACTION]
        state = rope_params(1024, HEAD_DIM)[idx * NUM_STATE : (idx + 1) * NUM_STATE]
        tail = torch.cat([action, state], dim=0).view(ARL, 1, -1)
        freqs = torch.cat([freqs, tail], dim=0)
    expected = torch.view_as_real(freqs)

    assert torch.equal(table, expected)


def test_mark_seq_dynamic_swallows_failures() -> None:
    """The hint helper must never be able to fail a request.

    ``None`` and a double mark are the two cases that reach it in practice: the second
    happens whenever a tensor is marked twice across retries.
    """
    _mark_seq_dynamic(None, 1)  # must not raise

    t = torch.zeros(4, 8)
    _mark_seq_dynamic(t, 1)
    _mark_seq_dynamic(t, 1)  # already marked; swallowed


# ── The varlen bound must come from the query tensor, not from a Python int ──────


def test_paged_write_attn_derives_max_query_len_from_query(monkeypatch) -> None:
    """``inputs.max_query_len`` must not reach the op; ``query.shape[0]`` must.

    A Python int here is a Dynamo value guard, so forwarding it re-specializes the graph
    per query length even when the query's own sequence dim is marked dynamic. The
    sentinel below is deliberately a value no real bound could take, so a regression
    shows up as the sentinel arriving rather than as a silent numeric coincidence.
    """
    from vllm_omni.experimental.ar_diffusion.kv_cache import paged_attention as pa

    sentinel = -12345
    query_len, heads, head_dim = 1785, 4, 8
    block_size = 16
    captured: dict[str, object] = {}

    # Positional layout of vllm_omni::ar_diffusion_paged_write_attn: query, k_curr,
    # v_curr, k_act, v_act, key_pool, value_pool, block_size, video_slots,
    # action_slots, block_table, query_start_loc, seq_lens, max_query_len, max_seq_len,
    # softmax_scale.
    _MAX_QUERY_LEN_ARG = 13

    def spy(*args):
        captured["max_query_len"] = args[_MAX_QUERY_LEN_ARG]
        captured["all_args"] = args
        return torch.empty_like(args[0])

    monkeypatch.setattr(torch.ops.vllm_omni, "ar_diffusion_paged_write_attn", spy)

    query = torch.zeros(query_len, heads, head_dim)
    key = torch.zeros(query_len, heads, head_dim)
    value = torch.zeros(query_len, heads, head_dim)
    inputs = pa.ARDiffusionPagedLayerInputs(
        layer_idx=torch.tensor(0),
        key_pool=torch.zeros(block_size, heads, head_dim),
        value_pool=torch.zeros(block_size, heads, head_dim),
        block_size=block_size,
        seq_len=1760,
        video_slots=torch.zeros(1, dtype=torch.int64),
        action_slots=torch.zeros(1, dtype=torch.int64),
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        query_start_loc=torch.tensor([0, query_len], dtype=torch.int32),
        seq_lens=torch.tensor([query_len], dtype=torch.int32),
        max_query_len=sentinel,
        max_seq_len=18480,
    )

    pa.paged_write_attn(inputs, query, key, value, None, None, 1.0)

    assert captured["max_query_len"] == query_len
    # Compare only the int arguments: `sentinel in args` would run == against the
    # tensor arguments and raise on the multi-element result.
    int_args = [a for a in captured["all_args"] if isinstance(a, int)]
    assert sentinel not in int_args, "inputs.max_query_len still reaches the op"


# ── The acceptance test: compilation count must not grow with AR geometry ────────


def _init_single_rank_tp():
    """TP=1 group for the block's Column/RowParallelLinear layers.

    ``initialize_model_parallel()`` reads ``get_current_vllm_config()``, so the whole
    thing has to happen inside a ``set_current_vllm_config()`` context. The context is
    returned so the caller keeps it open for the lifetime of the block: the TP layers
    read the config lazily.
    """
    import torch.distributed as dist
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    cfg_ctx = set_current_vllm_config(VllmConfig())
    cfg_ctx.__enter__()
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29777")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        init_distributed_environment(
            world_size=1, rank=0, distributed_init_method="env://", local_rank=0, backend="gloo"
        )
    ensure_model_parallel_initialized(1, 1)
    return cfg_ctx


def _stub_attention(blk) -> None:
    """Replace the attention *call* with a traceable stand-in.

    Both self-attention paths bottom out in an op Dynamo cannot trace under fake
    tensors: the paged path needs a live KV pool and slot registry (an engine), and the
    non-paged fallback calls ``torch.ops._vllm_fa2_C.varlen_fwd``, which has no meta
    kernel. Neither is what this test is about. Everything the RoPE hoist touches stays
    real; only the kernel is substituted, by a shape-correct elementwise stand-in.
    """
    import types

    def fake_attn(self, q, k, v, *a, **kw):
        return q * 1.0  # (B, L, H, D) -> same shape, cheap and traceable

    for holder in (blk.self_attn, blk.cross_attn):
        inner = getattr(holder, "attn", None)
        if inner is not None:
            inner.forward = types.MethodType(fake_attn, inner)


def _make_inputs(video_len: int, with_action: bool, device: str, dtype: torch.dtype):
    g = torch.Generator(device="cpu").manual_seed(video_len)
    q_len = video_len + (ARL if with_action else 0)
    x = torch.randn(1, q_len, DIM, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    e = torch.randn(1, q_len, 6, DIM, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    ctx = torch.randn(1, 32, DIM, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    # Non-paged fallback KV: stacked (2, B, L, H, D) with an existing window.
    kv = torch.randn(2, 1, video_len, HEADS, HEAD_DIM, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    return x, e, ctx, kv


@pytest.mark.skipif(
    os.environ.get(_HEAVY_ENV, "0").strip().lower() not in ("1", "true", "yes", "on"),
    reason=f"compiles a real DiT block; opt in with {_HEAVY_ENV}=1",
)
def test_compilation_count_does_not_grow_with_ar_geometry() -> None:
    """Two graphs across 10 geometry/position cases, and two is the minimum.

    Prefill forwards carry no action registers (``action_register_length is None``,
    q=880/1760) while diffuse forwards carry 25 (q=1785). The
    ``if action_register_length is not None`` branch in the block splits it into two
    genuinely different dataflows -- the action branch slices off an action tail, RoPEs
    it separately and re-concatenates -- so specializing on that is correct and
    desirable: one graph per real code path, reused across every sequence length and
    every AR chunk position.

    What must hold, and what this asserts, is that the count does not grow with chunk
    position or sequence length. Before the hoist and these hints, the same sweep
    produced a new graph per ``(length, chunk position)`` pair.
    """
    from torch._inductor.compile_fx import compile_fx

    from vllm_omni.diffusion.models.dreamzero.causal_wan_model import CausalWanAttentionBlock

    device = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    if device == "cpu" and torch.cuda.is_available():
        device = "cuda"
    dtype = torch.bfloat16

    cfg_ctx = _init_single_rank_tp()  # keep open: TP layers read the config lazily
    try:
        torch.manual_seed(1234)
        blk = (
            CausalWanAttentionBlock(
                cross_attn_type="t2v_cross_attn",
                dim=DIM,
                ffn_dim=DIM * 2,
                num_heads=HEADS,
                frame_seqlen=220,
                local_attn_size=-1,
                sink_size=0,
                num_frame_per_block=NUM_FRAME_PER_BLOCK,
                qk_norm=True,
                cross_attn_norm=True,
                eps=1e-6,
                num_action_per_block=NUM_ACTION,
                num_state_per_block=NUM_STATE,
            )
            .to(device=device, dtype=dtype)
            .eval()
        )
        _stub_attention(blk)

        n_compiles = 0

        def counting_backend(gm, example_inputs, **kwargs):
            nonlocal n_compiles
            n_compiles += 1
            return compile_fx(gm, example_inputs, **kwargs)

        # Exactly setup_compile()'s DiT kwargs, plus a counting backend.
        blk.forward = torch.compile(blk.forward, backend=counting_backend, fullgraph=True, dynamic=False)

        cases = [(video_len, False, 0) for video_len in VIDEO_LENS]
        cases += [(1760, True, chunk_pos) for chunk_pos in CHUNK_POSITIONS]

        with torch.inference_mode():
            for video_len, with_action, chunk_pos in cases:
                x, e, ctx, kv = _make_inputs(video_len, with_action, device, dtype)
                arl = ARL if with_action else None
                table = _build_table(video_len, with_action, max(chunk_pos, 1), device=device)

                _mark_seq_dynamic(x, 1)
                _mark_seq_dynamic(e, 1)
                _mark_seq_dynamic(table, 0)
                # Harness only. This exercises the non-paged fallback, where kv_cache is
                # a (2, B, L, H, D) tensor whose L grows with the window. The deployed
                # path passes ARDiffusionPagedLayerInputs, whose block_table is padded to
                # a fixed width by design, so there is nothing to mark there.
                _mark_seq_dynamic(kv, 2)

                out, _ = blk(
                    x=x,
                    e=e,
                    freqs=table,
                    context=ctx,
                    action_register_length=arl,
                    kv_cache=kv,
                    crossattn_cache=None,
                )
                assert out.shape == x.shape
                assert not torch.isnan(out.float()).any()

        assert n_compiles == 2, (
            f"expected 2 graphs (prefill-no-action + diffuse-with-action), got {n_compiles} "
            f"across {len(VIDEO_LENS)} sequence lengths x {len(CHUNK_POSITIONS)} AR chunk positions"
        )
    finally:
        cfg_ctx.__exit__(None, None, None)
