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

import os
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
    """Delegate the minimum codec length to the model's frame counter.

    Clear the runner's re-registered minimum-token mask each step so it
    cannot suppress the continue/stop control row. The model still masks
    codec EOS while state.step is below min_tokens."""
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
    """Return the uniform query width for a Talker-only decode step.

    Prefill, mixed and uneven batches return zero. The caller must reject
    multi-token rows when the frame loop declines; sampling only the last
    row would corrupt codec output and scheduler accounting."""
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
    frame_forward: Callable[[int, Callable[[], Any]], torch.Tensor] | None = None,
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
    terminal_tail = None
    if (
        narrow is None
        and inputs_embeds is not None
        and inputs_embeds.device.type == "npu"
        and os.getenv("MINICPMO_NPU_KSTEP_SKIP_TERMINAL_TAIL") == "1"
    ):
        terminal_tail = getattr(model, "terminal_kstep_tail", None)
    for frame in range(frames):
        if frame > 0 and terminal_tail is not None:
            tail = terminal_tail(frame_outputs[-1].text_hidden_states, infos)
            if tail is not None:
                output, stops = tail
                frame_outputs.extend([output] * (frames - frame))
                frame_stop_logits.extend([stops] * (frames - frame))
                if frame_forward is not None:
                    frame_hidden.extend([output.text_hidden_states.clone()] * (frames - frame))
                logger.info_once("[minicpmo] MRv2 skips fully stopped Talker graph replays")
                break
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
        if frame_forward is not None:
            hidden = frame_forward(frame, run_model)
        elif narrow is not None and hasattr(narrow, "run_model"):
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
        if frame_forward is not None:
            frame_kwargs["request_token_spans"] = [(i, i + 1) for i in range(len(spans))]
        if narrow is not None or frame_forward is not None:
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
    if frame_hidden:
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
    if narrow is None and input_ids is not None:
        write_batch = getattr(model, "write_kstep_frame_embeddings", None)
        if write_batch is not None and write_batch(inputs_embeds, spans, infos, frame):
            return
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
    """Propose the same continuation width for every request, or none.

    Compare accepted rows with their actual scheduled draft counts to
    distinguish an early stop from a normal one-token step. After a stop
    or incompatible remaining context, clear the entire batch until a
    uniform K-step can resume. Never draft for individual rows."""
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
    """Use the model's control-row width when parsing accepted tokens.

    A zero vocabulary drops every accepted control token and prevents
    scheduler rollback. Update the runner and sampler metadata to the
    continue/stop head width without changing codec sampling."""
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
