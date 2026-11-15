# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Several codec frames inside one stage-1 ``execute_model``.

One MiniCPM-o Talker decode step is almost all vLLM host work wrapped around a
small device forward: a 190M-parameter model that emits exactly one codec
frame, but pays scheduler, metadata, sampling and IPC costs per step. Over the
~118 frames of a typical request that host work is the largest single term
after TTFT.

None of it is per *frame*. The scheduler step, input preparation, attention
metadata, sampling, output assembly and engine-core IPC are per *step*, and a
step could just as well carry K frames. This module runs them.

## How a frame advances without the host

The step is scheduled with K query positions per request (stage 1's
``speculative_config`` in the deploy YAML). vLLM prepares all K of them --
positions, KV slots, block table, sequence lengths -- and the captured graph is
a K-token uniform decode. What vLLM cannot do is fill in the *inputs*: frame
k+1's embedding is the codec token sampled at frame k, so the K positions are
sequential, not speculative.

So the loop replays the same captured K-token graph K times. Before replay k it
writes frame k's embedding into row k of the persistent ``inputs_embeds``
buffer the graph captured; after replay k it samples frame k from row k of the
hidden states. Rows above k still hold stale embeddings during replay k, and
their KV is stale too -- but the attention bias is causal, so row k never reads
them, and the next replay overwrites them with the real thing. After the K'th
replay every row and every KV entry is what a K-step decode would have written.

A replay of the 20-layer forward is 0.83 ms of device and microseconds of host,
so K replays cost what K frames cost and the per-step host work is paid once.

Nothing in the loop touches the host: the embedding lookup, the codec sample
(the captured codec-sampling step), the stop row and the emitted delta are all device
tensors enqueued in order on one stream, and the runner's existing coalesced
D2H carries them out at the end of the step.

## What makes it correct

* **Stop is still per frame.** ``make_omni_output`` evaluates the codec EOS and
  the frame ceiling on every frame exactly as it does today. A request that ends
  at frame j reports its stop row there; frames j+1..K-1 take the "already
  finished" early return, emit an empty delta and report a stop row too. vLLM's
  rejection sampler sees the first stop and truncates the request to j+1
  tokens, which is the same sequence a one-frame-per-step run would produce.
  The ceiling is not only the stage-resolved codec budget: the runner also
  passes each request's remaining output budget
  (``SamplingParams.max_tokens - num_output_tokens``) and the codec merge clamps
  it into ``state["max_tokens"]``, so a request limit that falls inside a
  K-frame step ends the loop where the engine stops accepting ids instead of
  emitting frames past it (PR #7929 review).
* **The emitted codes are unchanged.** The extra frames after a stop contribute
  nothing: their deltas are empty or the ``-1`` sentinel the connector already
  drops (``_codec_scalars``). So the audio is expected to be bit-identical, not
  merely close -- which is a far stronger check than the CER gate.

Off when stage 1's deploy config carries no speculative_config.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import torch
from vllm.logger import init_logger

from vllm_omni.model_executor.models.interfaces import supports_multi_frame_decode
from vllm_omni.model_executor.models.minicpmo_4_5.talker_kstep_contract import (
    CONTINUE_TOKEN_ID,
    STOP_ROW_WIDTH,
)
from vllm_omni.model_executor.models.minicpmo_4_5.talker_kstep_contract import (
    STOP_TOKEN_ID as STOP_TOKEN_ID,
)


class NarrowStep(Protocol):
    spans: list[tuple[int, int]]

    def select(self, frame: int) -> None: ...


logger = init_logger(__name__)

_LOGGED_ENGAGE = False


def neutralize_kstep_min_tokens(logitsprocs: Any) -> None:
    """Stop vLLM's ``min_tokens`` layer from censoring the stop signal.

    The yaml ``min_tokens: 50`` is enforced by ``MinTokensLogitsProcessor``
    writing -inf into the stop-token logits. It unmasks once
    ``len(output_token_ids) >= min_tokens``, a counter owned by vLLM's spec
    bookkeeping; while that counter lags, the request's only stop signal stays
    masked and ``finished_reason`` can only be ``length`` (observed on stage 1:
    all length, zero stop).

    The real minimum-length guard lives elsewhere:
    ``talker_codec_sample.greedy_codec_sample`` masks the codec EOS itself via
    ``state.step < min_tokens``, using the model's own frame counter. This vLLM
    layer is therefore redundant and, under multi-frame decode, also swallows
    the stop signal -- clear its list and leave the minimum to the model-side
    guard.

    ``min_toks`` is re-registered by ``update_state`` every step, so this must
    be called every step (the caller places it right before sampling). Never
    raises.
    """
    try:
        for proc in getattr(logitsprocs, "non_argmax_invariant", None) or []:
            if not type(proc).__name__.endswith("MinTokensLogitsProcessor"):
                # Suffix rather than equality: the processor can arrive
                # subclassed or wrapped (a platform patch, a profiling shim),
                # and a mismatch here is silent -- the mask stays on and the
                # only stop signal a K-step request has is censored forever.
                continue
            min_toks = getattr(proc, "min_toks", None)
            if isinstance(min_toks, dict) and min_toks:
                min_toks.clear()
    except Exception as exc:
        logger.warning("[minicpmo] K-step min_tokens neutralization skipped: %r", exc)


_LOGGED_BLOCKS: set[str] = set()
_LOGGED_FOLDS: set[str] = set()


def _log_block_once(reason: str) -> None:
    """Say each distinct blocking reason once; silence is the worse failure.

    A set, not a single slot: the blocking conditions and the per-step folds
    alternate at concurrency, and comparing against only the *last* reason
    logs on every alternation. The reason strings are a fixed set in this
    module, so the set stays bounded.
    """
    if reason in _LOGGED_BLOCKS:
        return
    _LOGGED_BLOCKS.add(reason)
    logger.info("[minicpmo] multi-frame Talker decode not engaged: %s", reason)


def _log_fold_once(reason: str) -> None:
    """Say each distinct fold reason once, at INFO.

    Separate from ``_log_block_once`` on purpose: a fold is an expected
    per-step outcome on an *engaged* loop, so it wants its own wording, and
    sharing one global would let a fold reason clobber the blocking reason
    that explains why the loop never engaged at all.
    """
    if reason in _LOGGED_FOLDS:
        return
    _LOGGED_FOLDS.add(reason)
    logger.info("[minicpmo] multi-frame Talker decode folded this step: %s", reason)


def applies(model: Any, model_kwargs_extra: dict[str, Any]) -> int:
    """Frames this step should run, or 0 when the loop must not engage.

    Returns the uniform per-request query length of a pure decode step over a
    Talker that can build its own decode embedding. Anything else -- a prefill,
    a mixed step, a batch whose requests were scheduled different token counts
    -- returns 0 and the caller takes the ordinary single-forward path.

    Every refusal says so once. A step that schedules several tokens per
    request and then takes the single-forward path is not merely slow: that
    path samples one frame from the *last* row of each span and reports one
    stop row for a step that needs one per position, so it corrupts the codec
    stream and the scheduler's accounting alike. `_model_forward` turns those
    into an error rather than letting them run.
    """
    if not supports_multi_frame_decode(model):
        # Every other stage: not a refusal, just not this model.
        return 0
    spans = model_kwargs_extra.get("request_token_spans")
    infos = model_kwargs_extra.get("model_intermediate_buffer")
    if not spans or not infos or len(spans) != len(infos):
        return _block("no request_token_spans/model_intermediate_buffer for this step")
    frames = int(spans[0][1]) - int(spans[0][0])
    if frames <= 1:
        return 0
    for start, end in spans:
        if int(end) - int(start) != frames:
            # A mixed step (one request prefilling, another decoding) has no
            # single frame count, and the captured graph is not a uniform
            # decode either. Dump the spans so a real-world refusal can be
            # attributed to an exact scheduling state instead of guessed at.
            logger.info(
                "[minicpmo] refusing non-uniform spans: frames=%d spans=%s",
                frames,
                [(int(s), int(e)) for s, e in spans],
            )
            return _block("request token spans are not uniform")
    for info in infos:
        if not isinstance(info, dict):
            return _block("a request carries no intermediate buffer")
        if bool(info.get("_omni_is_prefill", False)):
            return 0
        state = info.get("audio_state")
        if not isinstance(state, dict):
            return _block("a request carries no audio_state")
        if int(state.get("step", 0)) <= 0:
            # The request has not sampled a codec token yet, so there is no
            # previous code for frame 0 to embed and this is not a decode.
            return _block("a request has not sampled a codec token yet")
    return frames


def is_multi_token_decode(model: Any, model_kwargs_extra: dict[str, Any]) -> bool:
    """True when this step schedules several tokens for a request that is decoding.

    Scoped to the Talker. Stage 0 drafts the Thinker's text with n-grams and so
    schedules multi-token decode steps of its own, which vLLM handles perfectly
    well -- it is only the Talker's one-frame-per-position sampler that cannot.
    """
    if not supports_multi_frame_decode(model):
        return False
    spans = model_kwargs_extra.get("request_token_spans")
    infos = model_kwargs_extra.get("model_intermediate_buffer")
    if not spans or not infos or len(spans) != len(infos):
        return False
    for (start, end), info in zip(spans, infos):
        if int(end) - int(start) <= 1:
            continue
        if isinstance(info, dict) and not bool(info.get("_omni_is_prefill", False)):
            return True
    return False


def _block(reason: str) -> int:
    _log_block_once(reason)
    return 0


def run(
    *,
    model: Any,
    run_model: Callable[[], Any],
    after_forward: Callable[[], None],
    inputs_embeds: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    frames: int,
    model_kwargs: dict[str, Any],
    model_kwargs_extra: dict[str, Any],
    narrow: NarrowStep | None = None,
    keep_cuda_narrow_on_decline: bool = False,
) -> Any:
    """Run ``frames`` codec frames and return the merged step output.

    ``after_forward`` is the runner's post-forward graph bookkeeping. It is
    called after every replay because that is where it sits in the single-frame
    path; under static-shape decode it is two pointer comparisons, and the frames of
    one step share their sequence lengths so there is nothing for it to rebind.
    """
    global _LOGGED_ENGAGE
    spans = model_kwargs_extra["request_token_spans"]
    infos = model_kwargs_extra["model_intermediate_buffer"]

    from vllm_omni.worker.talker_cuda_graph import CUDANarrowStep, run_cuda_fused_step

    if isinstance(narrow, CUDANarrowStep) and narrow.inputs_embeds.is_cuda:
        fused = run_cuda_fused_step(
            narrow,
            model=model,
            model_kwargs={**model_kwargs, **model_kwargs_extra},
            intermediate_tensors=model_kwargs.get("intermediate_tensors"),
        )
        if fused is not None:
            output, stops = fused
            model.set_batch_stop_logits(stops.reshape(-1, stops.shape[-1]))
            after_forward()
            return output
        if not keep_cuda_narrow_on_decline:
            narrow.restore()
            narrow = None

    frame_outputs = []
    frame_stop_logits = []
    frame_hidden = []
    for frame in range(frames):
        if narrow is not None:
            # Frame 0's embeddings are already in the buffer, but at the rows a
            # K-query step laid them out on: request r's is at row r*K, and the
            # one-query graph reads row r. r <= r*K, so compacting in request
            # order never overwrites a row it has still to read.
            narrow.select(frame)
            if frame == 0:
                if not getattr(narrow, "frame_zero_prepared", False):
                    _compact_frame_zero_embeddings(inputs_embeds, spans)
            else:
                target_embeds = getattr(narrow, "inputs_embeds", inputs_embeds)
                _write_frame_embeddings(model, target_embeds, input_ids, spans, infos, frame, narrow=narrow)
        elif frame > 0:
            _write_frame_embeddings(model, inputs_embeds, input_ids, spans, infos, frame)
        if narrow is not None and hasattr(narrow, "run_model"):
            hidden = narrow.run_model(run_model, capture=False)
        else:
            hidden = run_model()
        after_forward()
        frame_kwargs = dict(model_kwargs_extra)
        frame_kwargs["request_token_spans"] = (
            list(narrow.spans)
            if narrow is not None
            else [(int(start) + frame, int(start) + frame + 1) for start, _ in spans]
        )
        if narrow is not None:
            # The graph writes one row per request and the next frame overwrites
            # it, so this frame's hidden state has to be kept before the replay
            # that follows.
            frame_hidden.append(hidden.clone())
        output = model.make_omni_output(hidden, **model_kwargs, **frame_kwargs)
        stop_logits = model.take_batch_stop_logits()
        if stop_logits is None:
            raise RuntimeError("MiniCPM-o multi-frame decode expects stop rows from every frame")
        frame_outputs.append(output)
        frame_stop_logits.append(stop_logits)

    if not _LOGGED_ENGAGE:
        _LOGGED_ENGAGE = True
        logger.info("[minicpmo] multi-frame Talker decode engaged: %d codec frames per step", frames)
    merged = model.merge_frame_outputs(frame_outputs, frame_stop_logits)
    if narrow is not None and frame_hidden:
        # (frames, rows, hidden) -> (rows * frames, hidden), request-major --
        # the layout `logits_indices` reads. OmniOutput is a NamedTuple.
        merged = merged._replace(
            text_hidden_states=torch.stack(frame_hidden, dim=1).reshape(-1, frame_hidden[0].shape[-1])
        )
    return merged


def _compact_frame_zero_embeddings(inputs_embeds: torch.Tensor | None, spans) -> None:
    """Move each request's frame-0 embedding to the row the narrow graph reads."""
    if inputs_embeds is None:
        raise RuntimeError("MiniCPM-o narrow Talker replay requires an inputs_embeds buffer")
    for row, (start, _) in enumerate(spans):
        if int(start) != row:
            inputs_embeds[row : row + 1].copy_(inputs_embeds[int(start) : int(start) + 1])


def _write_frame_embeddings(
    model: Any,
    inputs_embeds: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    spans: list[tuple[int, int]],
    infos: list[dict[str, Any]],
    frame: int,
    narrow: NarrowStep | None = None,
) -> None:
    """Put frame ``frame``'s decode embedding into the row the graph reads.

    ``model.preprocess`` is the same call the runner's ``_preprocess`` makes for
    frame 0; it reads the codec token the previous frame wrote into
    ``info["audio_codes"]["current"]`` and returns its embedding. Writing into
    ``inputs_embeds`` in place is what makes the next replay see it -- the
    buffer is the one the graph captured, and vLLM does not copy inputs into
    captured graphs (``cudagraph_copy_inputs`` is False).
    """
    if inputs_embeds is None:
        raise RuntimeError("MiniCPM-o multi-frame decode requires an inputs_embeds buffer")
    for index, ((start, _), info) in enumerate(zip(spans, infos)):
        # The token to embed always sits at the step's own row; only where the
        # embedding goes changes when the narrow graph is the one replaying.
        source = int(start) + frame
        row = index if narrow is not None else source
        row_ids = input_ids[source : source + 1] if input_ids is not None else None
        if row_ids is None:
            raise RuntimeError("MiniCPM-o multi-frame decode requires input_ids for preprocess")
        _, embeds, _ = model.preprocess(input_ids=row_ids, input_embeds=None, **info)
        inputs_embeds[row : row + 1].copy_(embeds[:1])


def drafts_this_step(runner: Any) -> int:
    """Frames the *next* step should be scheduled for, or 0 to stay generic.

    The count travels to the scheduler as speculative tokens because that is
    vLLM V1's only way of saying "this request advanced by more than one token",
    which is what grows the block table and reserves the KV slots the loop
    writes into. Nothing about it is speculative: the drafts are always
    `continue`, and the frames are generated sequentially rather than verified.
    """
    if not supports_multi_frame_decode(getattr(runner, "model", None)):
        return 0
    num_spec = int(getattr(runner, "num_spec_tokens", 0) or 0)
    if num_spec <= 0:
        return 0
    return num_spec + 1


def constant_drafts(
    valid_sampled_token_ids: Any,
    frames: int,
    num_reqs: int,
    scheduled_draft_counts: list[int] | None = None,
    fold_short_rows: bool = False,
) -> list[list[int]]:
    """`continue` repeated -- for the whole batch, or for none of it.

    The drafts are how the next step gets K query positions per request, and
    the multi-frame loop can only run a step whose requests all have the *same*
    span: it replays one captured uniform-decode graph and samples one codec
    frame per position. So drafting per request is not an option.

    ``scheduled_draft_counts[i]`` is how many draft positions the scheduler
    actually gave request *i* in the step that produced row *i* -- the
    authoritative split between the two one-token shapes that must not be
    confused:

    * a row shorter than the step's own schedule (1 sampled +
      ``scheduled`` drafts) on a step that *did* carry drafts is a stop
      row firing mid-span: the rejection sampler truncated it to the
      frames it accepted, so its next-step schedule is j+1 wide against
      the neighbours' K -- non-uniform, fold the batch. A row exactly as
      wide as the step scheduled is *not* a stop even below ``frames``:
      the scheduler itself shortened the span (the uniform rewrite in
      ``omni_ar_scheduler``), and this step must emit drafts or the
      scheduler's booked frames never reach stage 2 and the requests
      starve;
    * a row of length 1 on a step that carried *no* drafts is the expected
      shape of a prefill result or of an ordinary one-frame step (an
      intentional single-frame fallback, or the step right after such a
      fallback). Folding here would starve the K-step loop forever: the next
      step schedules one token again, produces another one-token row, and no
      drafts are ever emitted (Buildkite three-tier bench: 31/32 requests
      failed on exactly this stall). A draftless step is precisely where the
      K-step span starts or resumes, so those rows draft.

    Mixed batches -- some requests drafted, some not -- cannot share a span
    and fold. So: every request drafts, or nobody does.

    ``fold_short_rows`` gates the stop-truncated-row folding above. The
    caller (``npu_ar_model_runner.propose``) is the only place that can tell
    a *live* short row -- a request that already ran multi-frame and fired
    its codec stop mid-span, which must fold or the neighbours schedule
    against a dead speaker -- from a row that merely looks short because its
    step's drafts were scheduler padding: that request re-drafted is what
    keeps the K-step alive across the padding step. Passing the flag False
    keeps the pre-split behaviour of never folding on row length.

    When ``scheduled_draft_counts`` is None the caller predates the scheduler
    split and every row is assumed to sit on a drafted step.
    """
    for index in range(num_reqs):
        sampled = None
        if isinstance(valid_sampled_token_ids, list) and index < len(valid_sampled_token_ids):
            sampled = valid_sampled_token_ids[index]
        if not sampled:
            _log_fold_once("a request in this batch sampled nothing; no drafts this step")
            return [[] for _ in range(num_reqs)]
        scheduled = scheduled_draft_counts[index] if scheduled_draft_counts is not None else frames - 1
        if scheduled > 0:
            # Width test against the step's own schedule, not against
            # ``frames``: a row of exactly ``scheduled + 1`` sampled ids is a
            # step the scheduler itself shortened to a uniform sub-``frames``
            # span (the non-uniform rewrite) -- folding it starves the
            # pipeline (the scheduler booked ``scheduled`` frames the runner
            # then never emits; stage 2 waits 600s and errors the requests,
            # observed 10-09: rewrite to a uniform 3-frame batch, rows len 4
            # against frames=8, folded, 4 requests dead). Only a row shorter
            # than the step scheduled is a real codec stop.
            if fold_short_rows and isinstance(sampled, list) and len(sampled) < scheduled + 1:
                _log_fold_once("a request stopped mid-step, short of its drafts; no drafts this step")
                return [[] for _ in range(num_reqs)]
    # Draft every row instead of folding the batch: a folded step still pays
    # the K-position reservation while emitting one frame, and the batch
    # re-drafts next step, so it alternates forever. A row without a draft
    # slot this step gets one next step, converging on uniform spans.
    return [[CONTINUE_TOKEN_ID] * (frames - 1) for _ in range(num_reqs)]


def ensure_stop_token_vocab(runner: Any, logits: Any) -> None:
    """Let vLLM's rejection-sampler output survive the Talker's narrow head.

    `RejectionSampler.parse_output` keeps a sampled token only while it is both
    not the placeholder and `< vocab_size`, and stage 1's `input_batch` reports
    a vocab size of **0** -- the Talker's vLLM-level head is the two-wide
    continue/stop row `compute_logits` builds, not a tokenizer vocabulary, and
    nothing had ever needed the number. With 0 the mask is empty, every
    accepted token is dropped, and the request comes back having generated
    nothing: the scheduler then never rolls back the tokens it advanced for the
    step and schedules a negative count two steps later.

    A single-frame step never reached this. It returns one token per request,
    which `_bookkeeping_sync` takes down the `max_gen_len == 1` branch, and that
    branch does not consult `vocab_size` at all.

    Raising it back to the width of the row the model actually emits is the
    whole fix -- and that width is `STOP_ROW_WIDTH` (two), *not* the width of
    whatever tensor the caller happens to be holding. The Talker's vLLM-level
    head is the two-wide continue/stop row `compute_logits` builds, so two is
    what every `vocab_size` reader has to see.

    The width has to be the row the model emits, not the width of whatever
    tensor the caller happens to be holding: `InputBatch.add_request` stores
    `top_k = vocab_size` as its "no top-k" sentinel, which only holds for 0 and
    2. A hidden width (768) instead passes the `parse_output` filter by accident
    but leaves `top_k_reqs` non-empty, and the Ascend `enable_reduce_sample`
    branch then treats the two-column distribution as a 768-way one -- the stop
    row does not survive into the request's token list and the request runs to
    `max_tokens`.
    """
    if logits is None or not supports_multi_frame_decode(runner.model):
        return
    batch = getattr(runner, "input_batch", None)
    width = STOP_ROW_WIDTH
    if batch is None or int(getattr(batch, "vocab_size", 0) or 0) >= width:
        return
    logger.info(
        "[minicpmo] Talker input batch reported vocab_size=%s; raising it to the "
        "%d-wide stop row so parse_output keeps the accepted tokens",
        getattr(batch, "vocab_size", None),
        width,
    )
    batch.vocab_size = width
