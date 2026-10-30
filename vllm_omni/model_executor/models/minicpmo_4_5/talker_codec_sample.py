# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Framework boundaries for Talker codec sampling.

Production stochastic sampling uses a logits-filter boundary implemented in
graph-capturable tensor ops: stop-token censoring, repetition penalty,
frequency/presence penalties, temperature, min-p, top-k, then top-p over the
top-k-filtered distribution keeping at least one candidate, while native NPU
operators keep softmax and ``torch.multinomial`` semantics (including
Generator state) unchanged. ``logit_bias`` and ``allowed_token_ids`` are NOT
implemented here -- this filter has no reader for them -- so when the K-step
stream is armed the scheduler rejects requests that set either field
(finished as FINISHED_ERROR at admission) instead of silently sampling a
different codec distribution than a single-frame deployment would.

Order and defaults follow the engine's single-frame path so that enabling the
multi-frame (K-step) decode does not change the sampling semantics:
``vllm/v1/sample/sampler.py`` for the temperature/min-p/top-k/top-p order,
``logits_processor/builtin.py`` for the censoring, bias and min-p processors and
``model_executor/layers/utils.py`` for the penalties. Every added knob is a
no-op at its default value.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

VOCAB_SIZE = 6562
HISTORY_WINDOW = 16

# Engine steps 6a+7b run as one Triton kernel on NPU (adopted 2026-10-17 after a
# 200-case bit-exact verification). Two independent controls:
#   - ``talker_codec_fused_prelude: false`` in the stage connector's ``extra``
#     block forces the eager chain (accuracy triage, e.g. after a CANN/torch_npu
#     upgrade). The key is a no-op off NPU or without Triton.
#   - ``_FUSED_PRELUDE_OK`` is a process-wide, permanent latch: one failed
#     compile or launch turns it off and every later step takes the eager chain.
_FUSED_PRELUDE_OK = True
_FUSED_PRELUDE_WARMED = False
_PRELUDE_BLOCK = 1024

# Engine step 7d (top-k + top-p) runs as one Triton kernel on NPU when both
# filters are active -- the stochastic tail right after the fused prelude.
# Same two controls as the prelude: ``talker_codec_fused_tail: false`` in the
# stage connector's ``extra`` block forces the eager chain (accuracy triage),
# and ``_TAIL_FUSION_OK`` is a process-wide permanent latch fed by the first
# failed compile/launch. The eager pair is launch-overhead-bound at this
# vocab (per-op fixed cost dominates: 6562-wide and 25-wide eager ops time
# the same), so the fusion removes launches, not arithmetic.
_TAIL_FUSION_OK = True
_TAIL_FUSION_WARMED = False
_TAIL_BLOCK = 1024
# The pairwise pass is single-block; wide candidate sets stay eager.
_TAIL_MAX_CANDIDATES = 64

try:  # Triton is NPU-only; other backends keep the eager prologue.
    import triton
    import triton.language as tl

    @triton.jit
    def _rep_pen_temp_kernel(
        x_ptr,
        hist_ptr,
        hlen_ptr,
        pen_ptr,
        temp_ptr,
        out_ptr,
        vocab_size: tl.constexpr,
        hist_len: tl.constexpr,
        block_size: tl.constexpr,
    ):
        """One pass over the vocab: history-count rep penalty, then /temp.

        Bit-for-bit twin of the eager pair `_repetition_penalty_scaling`
        plus `logits / temperature`: integer counts, the same exp2/log2 pow
        form the eager torch.pow compiles to, and the same op order
        (sign-dependent penalty first, temperature division last).
        """
        pid = tl.program_id(0)
        offs = pid * block_size + tl.arange(0, block_size)
        mask = offs < vocab_size
        pen = tl.load(pen_ptr)
        temp = tl.load(temp_ptr)
        hlen = tl.load(hlen_ptr)
        count = tl.zeros([block_size], dtype=tl.int32)
        for h in tl.static_range(hist_len):
            tok = tl.load(hist_ptr + h).to(tl.int32)
            count += tl.where((offs == tok) & (h < hlen), 1, 0)
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        alpha = tl.exp2(tl.log2(pen) * count.to(tl.float32))
        x = tl.where(x < 0, x * alpha, x / alpha)
        x = x / temp
        tl.store(out_ptr + offs, x, mask=mask)

    @triton.jit
    def _topk_topp_mask_kernel(
        logits_ptr,
        vals_ptr,
        idx_ptr,
        out_ptr,
        top_p,
        K: tl.constexpr,  # noqa: N803 - Triton constexpr naming convention
        KP: tl.constexpr,  # noqa: N803 - Triton constexpr naming convention
        VOCAB: tl.constexpr,  # noqa: N803 - Triton constexpr naming convention
        BLOCK: tl.constexpr,  # noqa: N803 - Triton constexpr naming convention
    ):
        """Single-block fused top-k + top-p mask over one B=1 logits row.

        Semantics equal the eager chain it replaces (top-k ``masked_fill_``
        then the ascending-cumsum top-p): candidate ``i`` is removed iff the
        prob mass of strictly-smaller candidates plus its own prob stays
        ``<= 1 - top_p`` -- the ascending ``cumsum <= 1 - top_p`` predicate
        evaluated on the top-k-filtered distribution. The largest candidate
        is never removed (total mass 1 > 1 - top_p for top_p < 1), so the
        eager guard ``remove[..., -1:] = False`` holds by construction.
        Ties among finite values cannot arise from continuous model logits;
        ``-inf`` top-k padding candidates carry zero prob and sort before every
        finite value, so both paths always remove them.
        """
        offs = tl.arange(0, KP)
        cand = offs < K
        v = tl.load(vals_ptr + offs, mask=cand, other=float("-inf"))
        mx = tl.max(v, axis=0)
        e = tl.where(cand, tl.exp(v - mx), 0.0)
        q = e / tl.sum(e, axis=0)
        # The ascending order must be decided on the raw logits, not on q:
        # softmax is monotonic so v_j < v_i iff q_j < q_i, but tl.exp and the
        # eager softmax exp can differ by ~1e-7 relative -- enough to flip the
        # order of near-equal neighbours on near-flat rows, which shifts the
        # ascending prefix by one candidate. The raw logits carry no
        # computation and therefore no implementation drift.
        vi = tl.reshape(v, (KP, 1))
        vj = tl.reshape(v, (1, KP))
        qj = tl.reshape(q, (1, KP))
        smaller = tl.sum(tl.where(vj < vi, qj, 0.0), axis=1)
        keep = cand & ((smaller + q) > (1.0 - top_p))
        gids = tl.load(idx_ptr + offs, mask=cand, other=0)
        for start in range(0, VOCAB, BLOCK):
            o = start + tl.arange(0, BLOCK)
            tl.store(
                out_ptr + o,
                tl.full((BLOCK,), float("-inf"), tl.float32),
                mask=o < VOCAB,
            )
        tl.store(out_ptr + gids, v, mask=keep)

    _HAS_TRITON = True
except Exception:  # pragma: no cover - triton missing/CPU
    _HAS_TRITON = False


@dataclass(frozen=True)
class TalkerCodecDeviceState:
    history: torch.Tensor
    history_len: torch.Tensor
    step: torch.Tensor
    max_tokens: torch.Tensor
    finished: torch.Tensor
    # Whole-stream occurrence counts, one entry per codec id, or ``None`` when
    # the request set neither frequency_penalty nor presence_penalty. The engine
    # derives both penalties from the request's entire prompt+output sequence
    # (``model_executor/layers/utils.py:73-78``), which the 16-frame ``history``
    # window cannot express; this tensor is rebuilt from the host-side record at
    # every chunk boundary and advanced in place per frame. It is mutated rather
    # than re-allocated because a fresh VOCAB_SIZE-wide tensor per frame would
    # be an allocation per decode step.
    full_bins: torch.Tensor | None = None
    # Per-frame scratch for the repetition-penalty histogram over the history
    # window, (1, VOCAB_SIZE) float32, allocated once per request/segment and
    # zeroed in place each frame -- same rationale as full_bins: a fresh
    # VOCAB_SIZE-wide tensor per frame would be an allocation per decode step.
    rep_counts: torch.Tensor | None = None


@dataclass(frozen=True)
class TalkerCodecSampleResult:
    sampled_token: torch.Tensor
    state: TalkerCodecDeviceState
    emit: torch.Tensor


def make_device_state(
    codes: torch.Tensor,
    *,
    step: int,
    max_tokens: int,
    finished: bool,
    full_bins: torch.Tensor | None = None,
) -> TalkerCodecDeviceState:
    """Create one persistent B=1 state at request/segment setup time."""
    recent = codes.reshape(-1)[-HISTORY_WINDOW:].to(dtype=torch.int32)
    history = torch.zeros((1, HISTORY_WINDOW), dtype=torch.int32, device=codes.device)
    if recent.numel():
        history[0, : recent.numel()].copy_(recent)
    return TalkerCodecDeviceState(
        history=history,
        history_len=torch.tensor([recent.numel()], dtype=torch.int32, device=codes.device),
        step=torch.tensor([step], dtype=torch.int32, device=codes.device),
        max_tokens=torch.tensor([max_tokens], dtype=torch.int32, device=codes.device),
        finished=torch.tensor([finished], dtype=torch.bool, device=codes.device),
        full_bins=full_bins,
        rep_counts=torch.zeros((1, VOCAB_SIZE), dtype=torch.float32, device=codes.device),
    )


def _stop_id_tuple(eos_token_id: int, min_tokens_stop_ids: tuple[int, ...]) -> tuple[int, ...]:
    """The censored stop set, codec EOS first and de-duplicated in a stable order.

    Taken from the engine's ``all_stop_token_ids`` (``sampling_params.py:537``,
    ``:676``, ``:688``). An empty ``min_tokens_stop_ids`` keeps the
    codec-EOS-only behaviour, which is what the K-step path used before this
    knob existed.
    """
    return tuple(dict.fromkeys((int(eos_token_id), *min_tokens_stop_ids)))


# The censoring step runs once per codec frame and the K-step loop calls it
# once per frame per decode step, so rebuilding the index tensor from the
# Python tuple with torch.tensor(...) -- a blocking H2D copy -- is the
# dominant single host cost of this path under py-spy sampling. stop_ids is
# an immutable tuple of ints and the tensor is only ever consumed as an index
# (index_select/index_copy read it, never write it), so one tensor per
# (stop set, device) can be shared across frames and requests; the persistent
# storage also gives graph capture a stable address.
_STOP_INDEX_CACHE: dict[tuple[tuple[int, ...], str], torch.Tensor] = {}


def _stop_index_tensor(stop_ids: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """One device index tensor per (stop set, device), built on first use."""
    key = (stop_ids, str(device))
    cached = _STOP_INDEX_CACHE.get(key)
    if cached is None:
        cached = torch.tensor(stop_ids, dtype=torch.long, device=device)
        _STOP_INDEX_CACHE[key] = cached
    return cached


def _censor_stop_logits(
    logits: torch.Tensor,
    state: TalkerCodecDeviceState,
    min_tokens: torch.Tensor,
    *,
    stop_ids: tuple[int, ...],
    eos_window_masked: bool,
) -> torch.Tensor:
    """Censor every stop id while the request is below its ``min_tokens`` floor.

    The engine's ``MinTokensLogitsProcessor`` masks ``all_stop_token_ids``
    (eos + ``stop_token_ids`` + extra eos ids) and lifts the mask by a spec-aware
    counter (``builtin.py:196-207`` / ``:296-373``); here the counter is the
    state's frame step, which the single-frame path uses as well
    (``minicpmo_4_5_omni_tts.py``'s ``mask_eos_rows``). ``eos_window_masked``
    overrides the floor for the duplex turn-end drain window, which the
    ``step < min_tokens`` criterion cannot express.
    """
    stop_index = _stop_index_tensor(stop_ids, logits.device)
    stop_logits = logits.index_select(-1, stop_index)
    if eos_window_masked:
        censored = torch.full_like(stop_logits, float("-inf"))
    else:
        censored = torch.where(
            (state.step < min_tokens).reshape(1, 1),
            torch.full_like(stop_logits, float("-inf")),
            stop_logits,
        )
    return logits.index_copy(-1, stop_index, censored)


def _filter_codec_logits(
    logits: torch.Tensor,
    state: TalkerCodecDeviceState,
    min_tokens: torch.Tensor,
    *,
    stop_ids: tuple[int, ...],
    eos_window_masked: bool,
) -> torch.Tensor:
    """The stop censoring both sampling paths share.

    The engine blanks every ``all_stop_token_ids`` entry before its penalties
    (``sampler.py:393-405``), so the K-step path does the same here.
    """
    return _censor_stop_logits(
        logits,
        state,
        min_tokens,
        stop_ids=stop_ids,
        eos_window_masked=eos_window_masked,
    )


def _repetition_penalty_scaling(
    logits: torch.Tensor,
    state: TalkerCodecDeviceState,
    repetition_penalty: torch.Tensor,
) -> torch.Tensor:
    """Repetition penalty over the codec history window (engine step 6a).

    ``alpha = penalty ** frequency`` with the sign-dependent branch, matching the
    model's single-frame ``_apply_batched_repetition_penalty``.
    """
    positions = torch.arange(HISTORY_WINDOW, dtype=torch.int32, device=logits.device).reshape(1, -1)
    valid = positions < state.history_len.reshape(1, 1)
    safe_tokens = torch.where(valid, state.history, torch.zeros_like(state.history)).to(torch.long)
    if state.rep_counts is not None:
        # Review P3-6: reuse the per-request histogram buffer instead of
        # allocating a VOCAB_SIZE-wide tensor every frame. zero_() re-runs
        # under graph replay, so no stale count survives a replay.
        counts = state.rep_counts.zero_()
    else:
        counts = torch.zeros((1, VOCAB_SIZE), dtype=torch.float32, device=logits.device)
    counts.scatter_add_(1, safe_tokens, valid.to(torch.float32))
    alpha = torch.pow(repetition_penalty.reshape(1, 1), counts)
    return torch.where(logits < 0, logits * alpha, logits / alpha)


def _apply_stream_penalties(
    logits: torch.Tensor,
    state: TalkerCodecDeviceState,
    frequency_penalty: float,
    presence_penalty: float,
) -> torch.Tensor:
    """Whole-stream frequency and presence penalties (engine steps 6b/6c).

    Straight from ``model_executor/layers/utils.py:86-87``:

        logits -= frequency_penalties * output_bin_counts
        logits -= presence_penalties * output_mask

    ``output_bin_counts`` is the occurrence count over the whole prompt+output
    sequence and ``output_mask`` is its ``> 0`` image; both come from the
    request's stream record. Skipped entirely when the request set neither, or
    when the record is absent (nothing to subtract from).
    """
    if state.full_bins is None:
        return logits
    if not frequency_penalty and not presence_penalty:
        return logits
    bins = state.full_bins.reshape(1, -1)
    if frequency_penalty:
        logits = logits - bins * frequency_penalty
    if presence_penalty:
        logits = logits - (bins > 0).to(logits.dtype) * presence_penalty
    return logits


def _rep_pen_temp_fused(
    logits: torch.Tensor,
    state: TalkerCodecDeviceState,
    repetition_penalty: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor | None:
    """Fused rep-penalty + temperature prologue on NPU; None if unsupported."""
    global _FUSED_PRELUDE_OK
    if not _HAS_TRITON or logits.device.type != "npu":
        return None
    vocab = logits.shape[-1]
    flat = logits.reshape(1, vocab)
    out = torch.empty_like(flat)
    grid = (triton.cdiv(vocab, _PRELUDE_BLOCK),)
    try:
        _rep_pen_temp_kernel[grid](
            flat,
            state.history,
            state.history_len,
            repetition_penalty.reshape(1, 1),
            temperature.reshape(1, 1),
            out,
            vocab_size=vocab,
            hist_len=HISTORY_WINDOW,
            block_size=_PRELUDE_BLOCK,
            num_warps=4,
        )
    except Exception as exc:
        # Compile or launch failure. The latch is process-wide and permanent: every
        # later step in this process takes the eager chain. Failures inside an
        # already captured graph replay never reach this handler.
        _FUSED_PRELUDE_OK = False
        logger.warning("fused talker codec prelude disabled after a kernel failure: %s", exc)
        return None
    return out.reshape(logits.shape)


def warmup_fused_prelude(device: torch.device, enabled: bool = True) -> None:
    """Compile the fused kernel once, before any graph capture.

    The first launch pays the Triton compile; running it here settles
    ``_FUSED_PRELUDE_OK`` outside capture, so a compile failure downgrades the
    process instead of poisoning a capture that just saw a failed launch.
    Idempotent, and a no-op off NPU, without Triton, or when disabled -- an
    eager-only configuration must never touch the kernel, because after a
    CANN/torch_npu upgrade a broken kernel can fault the device instead of
    raising.
    """
    global _FUSED_PRELUDE_WARMED
    if not enabled or _FUSED_PRELUDE_WARMED or not _HAS_TRITON or device.type != "npu":
        return
    _FUSED_PRELUDE_WARMED = True
    state = TalkerCodecDeviceState(
        history=torch.zeros(HISTORY_WINDOW, dtype=torch.long, device=device),
        history_len=torch.zeros(1, dtype=torch.int32, device=device),
        step=torch.zeros(1, dtype=torch.int32, device=device),
        max_tokens=torch.ones(1, dtype=torch.int32, device=device),
        finished=torch.zeros(1, dtype=torch.bool, device=device),
    )
    logits = torch.zeros(1, VOCAB_SIZE, device=device)
    _rep_pen_temp_fused(
        logits,
        state,
        torch.ones(1, device=device),
        torch.ones(1, device=device),
    )


def _topk_topp_fused(
    logits: torch.Tensor,
    top_k: int,
    top_p: float,
) -> torch.Tensor | None:
    """Fused top-k + top-p mask on NPU; ``None`` falls back to the eager chain.

    One ``torch.topk`` for the candidate set, one Triton kernel for both
    masks. The returned row is what the eager top-k/top-p pair would produce:
    kept candidates keep their original logit, everything else is ``-inf``.
    The caller's softmax + ``torch.multinomial`` input is therefore unchanged
    in shape, dtype and RNG stream.
    """
    global _TAIL_FUSION_OK
    if not _TAIL_FUSION_OK or logits.device.type != "npu":
        return None
    keep = min(int(logits.shape[-1]), int(top_k))
    if keep > _TAIL_MAX_CANDIDATES:
        return None
    vals, idx = torch.topk(logits, keep, dim=-1)
    out = torch.empty_like(logits)
    kp = 1 << (keep - 1).bit_length()
    try:
        _topk_topp_mask_kernel[(1,)](
            logits,
            vals,
            idx,
            out,
            float(top_p),
            K=keep,
            KP=kp,
            VOCAB=int(logits.shape[-1]),
            BLOCK=_TAIL_BLOCK,
            num_warps=8,
        )
    except Exception as exc:
        # Same latch contract as the prelude: process-wide, permanent, and a
        # failure inside an already captured graph replay never lands here.
        _TAIL_FUSION_OK = False
        logger.warning("fused talker codec tail disabled after a kernel failure: %s", exc)
        return None
    return out


def warmup_fused_tail(device: torch.device, enabled: bool = True) -> None:
    """Compile the fused tail kernel once, before any graph capture.

    Same contract as :func:`warmup_fused_prelude`: running it here settles
    ``_TAIL_FUSION_OK`` outside capture, so a compile failure downgrades the
    process instead of poisoning a capture that just saw a failed launch.
    Idempotent, and a no-op off NPU, without Triton, or when disabled.
    """
    global _TAIL_FUSION_WARMED
    if not enabled or _TAIL_FUSION_WARMED or not _HAS_TRITON or device.type != "npu":
        return
    _TAIL_FUSION_WARMED = True
    _topk_topp_fused(torch.zeros(1, VOCAB_SIZE, device=device), 25, 0.85)


def prepare_codec_logits(
    raw_logits: torch.Tensor,
    state: TalkerCodecDeviceState,
    min_tokens: torch.Tensor,
    temperature: torch.Tensor,
    repetition_penalty: torch.Tensor,
    *,
    eos_token_id: int,
    top_k: int,
    top_p: float,
    min_p: float = 0.0,
    min_tokens_stop_ids: tuple[int, ...] = (),
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    eos_window_masked: bool = False,
    fused_prelude: bool = True,
    fused_tail: bool = True,
) -> torch.Tensor:
    """Prepare and filter logits without changing native NPU RNG semantics.

    ``min_p`` mirrors the engine's argmax-invariant ``MinPLogitsProcessor``,
    which runs after temperature and before top-k/top-p (``builtin.py:102-113``).
    ``min_tokens_stop_ids`` widens the censored stop set beyond the codec EOS;
    ``frequency_penalty``/``presence_penalty`` subtract the whole-stream counts.
    ``fused_prelude=False`` (stage connector ``extra`` key
    ``talker_codec_fused_prelude``) forces the eager chain for accuracy triage;
    ``fused_tail=False`` (``talker_codec_fused_tail``) does the same for the
    top-k/top-p pair.
    Every one of them is a no-op at its default value.
    """
    stop_ids = _stop_id_tuple(eos_token_id, min_tokens_stop_ids)
    logits = raw_logits.float()
    # Engine step 3-4: stop censoring, before any penalty.
    logits = _filter_codec_logits(
        logits,
        state,
        min_tokens,
        stop_ids=stop_ids,
        eos_window_masked=eos_window_masked,
    )
    # Engine step 6a + 7b: repetition penalty then temperature. On NPU the
    # pair runs as one fused Triton kernel when the whole-stream penalties
    # are no-ops (they sit between the two in the eager order, so non-zero
    # penalty requests fall back to the eager chain below).
    fused = (
        _rep_pen_temp_fused(logits, state, repetition_penalty, temperature)
        if (fused_prelude and _FUSED_PRELUDE_OK and not frequency_penalty and not presence_penalty)
        else None
    )
    if fused is not None:
        logits = fused
    else:
        # Engine step 6a: repetition penalty over the codec history window.
        logits = _repetition_penalty_scaling(logits, state, repetition_penalty)
        # Engine steps 6b/6c: frequency and presence penalties over the whole stream.
        logits = _apply_stream_penalties(logits, state, frequency_penalty, presence_penalty)
        # Engine step 7b: temperature is applied after the penalties. The division
        # also gives the in-place masks below a tensor of their own.
        logits = logits / temperature.reshape(1, 1)
    # Engine step 7c: min_p. The engine runs it as an argmax-invariant
    # processor, i.e. after temperature and before top-k/top-p, and the greedy
    # branch never sees it -- which is why greedy_codec_sample has no min_p.
    if min_p > 0.0:
        probabilities = torch.softmax(logits, dim=-1)
        min_p_threshold = probabilities.amax(dim=-1, keepdim=True) * min_p
        logits.masked_fill_(probabilities < min_p_threshold, float("-inf"))
    # Engine step 7d: top-k then top-p over the filtered distribution. On NPU
    # the pair runs as one Triton kernel when both filters are active (the
    # deployment pins top_k=25/top_p=0.85) -- the eager pair is dominated by
    # per-launch overhead at this vocab. Either path leaves the caller's
    # softmax + multinomial input identical in shape, dtype and RNG stream.
    tail = _topk_topp_fused(logits, top_k, top_p) if (fused_tail and top_k > 0 and 0.0 < float(top_p) < 1.0) else None
    if tail is not None:
        return tail
    # Engine step 7d: top-k first (topk_topp_sampler.py:392).
    if top_k > 0:
        keep = min(VOCAB_SIZE, int(top_k))
        threshold = torch.topk(logits, keep, dim=-1).values[..., -1, None]
        logits.masked_fill_(logits < threshold, float("-inf"))
    # Engine step 7d: top-p runs over the top-k-filtered distribution and keeps
    # at least one candidate (topk_topp_sampler.py:404/:415).
    if 0.0 < float(top_p) < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=False, dim=-1)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative_probs <= (1.0 - float(top_p))
        remove[..., -1:] = False
        remove = remove.scatter(-1, sorted_indices, remove)
        logits.masked_fill_(remove, float("-inf"))
    return logits


def _advance_full_bins(
    state: TalkerCodecDeviceState,
    sampled: torch.Tensor,
    emit: torch.Tensor,
) -> None:
    """Add one occurrence for the emitted id, in place.

    In-place on purpose: re-allocating a VOCAB_SIZE-wide tensor every frame would
    be an allocation per decode step. The tensor belongs to the request's state
    record, which is replaced every frame, so mutating it cannot be observed by
    an earlier state.
    """
    if state.full_bins is None:
        return
    state.full_bins.scatter_add_(
        0,
        sampled.reshape(-1).to(torch.long),
        emit.reshape(-1).to(state.full_bins.dtype),
    )


def advance_codec_device_state(
    state: TalkerCodecDeviceState,
    sampled_token: torch.Tensor,
    *,
    eos_token_id: int,
    ignore_eos: bool = False,
    stop_token_ids: tuple[int, ...] = (),
) -> TalkerCodecDeviceState:
    """Advance persistent stochastic state with graph-capturable tensor ops.

    ``stop_token_ids`` are the request's extra codec stop ids -- the
    ``stop_token_ids`` analogue for the self-sampled codec stream. They are
    request-level constants, so the Python loop unrolls into a fixed set of
    comparisons at capture time, the same way the pinned top-k/top-p values
    are baked into the per-request sampling graph.
    """
    sampled = sampled_token.reshape(1).to(torch.int32)
    is_eos, is_stop = _classify_stop(
        sampled,
        state,
        eos_token_id=eos_token_id,
        ignore_eos=ignore_eos,
        stop_token_ids=stop_token_ids,
    )
    next_state, _emit, _finished = _advance_classified(state, sampled, is_eos, is_stop)
    return next_state


def _classify_stop(
    sampled: torch.Tensor,
    state: TalkerCodecDeviceState,
    *,
    eos_token_id: int,
    ignore_eos: bool,
    stop_token_ids: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """The eos / stop-id masks shared by every advance entry point."""
    if ignore_eos:
        # ``ignore_eos`` blanks the engine's ``_eos_token_id``
        # (vllm/sampling_params.py:670-671), so a sampled codec EOS is just
        # another frame and must not finish the request.
        is_eos = torch.zeros_like(state.finished)
    else:
        is_eos = sampled == int(eos_token_id)
    is_stop = torch.zeros_like(state.finished)
    for token_id in stop_token_ids:
        is_stop = is_stop | (sampled == int(token_id))
    return is_eos, is_stop


def _advance_classified(
    state: TalkerCodecDeviceState,
    sampled: torch.Tensor,
    is_eos: torch.Tensor,
    is_stop: torch.Tensor,
) -> tuple[TalkerCodecDeviceState, torch.Tensor, torch.Tensor]:
    """Single transition source: next state plus the emit/finished decision.

    Both entry points (``advance_codec_device_state`` and
    ``codec_sample_result``) route through here, so the eos/stop/limit
    transition and its emit mask are computed exactly once per frame
    (review P3-6: they used to be computed twice -- once in
    ``codec_sample_result`` and again inside ``advance_codec_device_state``).
    """
    active = ~state.finished
    next_step = torch.where(active, state.step + 1, state.step)
    reached_limit = active & (next_step >= state.max_tokens)
    finished = state.finished | is_eos | is_stop | reached_limit
    emit = active & (~is_eos) & (~is_stop) & (~reached_limit)
    shifted = torch.cat([state.history[:, 1:], sampled.reshape(1, 1)], dim=1)
    append_at = state.history_len.clamp(min=0, max=HISTORY_WINDOW - 1).to(torch.long).reshape(1, 1)
    appended = state.history.scatter(1, append_at, sampled.reshape(1, 1))
    candidate = torch.where((state.history_len >= HISTORY_WINDOW).reshape(1, 1), shifted, appended)
    history = torch.where(emit.reshape(1, 1), candidate, state.history)
    history_len = torch.clamp(state.history_len + emit.to(torch.int32), max=HISTORY_WINDOW)
    _advance_full_bins(state, sampled, emit)
    next_state = TalkerCodecDeviceState(
        history, history_len, next_step, state.max_tokens, finished, state.full_bins, state.rep_counts
    )
    return next_state, emit, finished


def codec_sample_result(
    state: TalkerCodecDeviceState,
    sampled_token: torch.Tensor,
    *,
    eos_token_id: int,
    ignore_eos: bool = False,
    stop_token_ids: tuple[int, ...] = (),
) -> TalkerCodecSampleResult:
    """Advance device state and retain the device-side emit decision."""
    sampled = sampled_token.reshape(1).to(torch.int32)
    is_eos, is_stop = _classify_stop(
        sampled,
        state,
        eos_token_id=eos_token_id,
        ignore_eos=ignore_eos,
        stop_token_ids=stop_token_ids,
    )
    # One transition for state + emit (review P3-6): the emit mask here used
    # to be recomputed independently of the one inside
    # ``advance_codec_device_state``. Semantics unchanged: for an active
    # request ``~reached_limit`` == ``next_step < max_tokens``, and for an
    # already-finished one both forms emit False.
    next_state, emit, _finished = _advance_classified(state, sampled, is_eos, is_stop)
    return TalkerCodecSampleResult(sampled_token=sampled, state=next_state, emit=emit)


def greedy_codec_sample(
    raw_logits: torch.Tensor,
    state: TalkerCodecDeviceState,
    min_tokens: torch.Tensor,
    repetition_penalty: torch.Tensor,
    *,
    eos_token_id: int,
    min_tokens_stop_ids: tuple[int, ...] = (),
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    ignore_eos: bool = False,
    eos_window_masked: bool = False,
    stop_token_ids: tuple[int, ...] = (),
) -> TalkerCodecSampleResult:
    """Greedy codec sample in graph-capturable tensor ops.

    The engine's greedy branch returns the argmax after the censoring masks, the
    logit bias and the penalties, and skips temperature, min-p, top-k and top-p
    entirely (``vllm/v1/sample/sampler.py:30-60``), so no top-k mask belongs on
    this path -- masking candidates below the argmax cannot change the argmax
    itself. Every other filter is the same one the random path uses.

    ``eos_window_masked`` masks the codec EOS for this frame regardless of the
    step counter -- see prepare_codec_logits.
    """
    stop_ids = _stop_id_tuple(eos_token_id, min_tokens_stop_ids)
    penalized = raw_logits.float()
    # Engine step 3-4: same stop censoring as the random path.
    penalized = _filter_codec_logits(
        penalized,
        state,
        min_tokens,
        stop_ids=stop_ids,
        eos_window_masked=eos_window_masked,
    )
    # Engine steps 6a/6b/6c: same penalties as the random path.
    penalized = _repetition_penalty_scaling(penalized, state, repetition_penalty)
    penalized = _apply_stream_penalties(penalized, state, frequency_penalty, presence_penalty)
    sampled = torch.argmax(penalized, dim=-1).to(torch.int32)
    # The state advance / emit / finished transition lives in exactly one
    # place: codec_sample_result. The sampled id is already in [0, vocab), so
    # the random and greedy paths share it verbatim (RNG untouched: the
    # multinomial call lives in the caller, not here).
    return codec_sample_result(
        state,
        sampled,
        eos_token_id=eos_token_id,
        ignore_eos=ignore_eos,
        stop_token_ids=stop_token_ids,
    )
