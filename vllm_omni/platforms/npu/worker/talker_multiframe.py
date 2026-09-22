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
  the token limit on every frame exactly as it does today. A request that ends
  at frame j reports its stop row there; frames j+1..K-1 take the "already
  finished" early return, emit an empty delta and report a stop row too. vLLM's
  rejection sampler sees the first stop and truncates the request to j+1
  tokens, which is the same sequence a one-frame-per-step run would produce.
* **The emitted codes are unchanged.** The extra frames after a stop contribute
  nothing: their deltas are empty or the ``-1`` sentinel the connector already
  drops (``_codec_scalars``). So the audio is expected to be bit-identical, not
  merely close -- which is a far stronger check than the CER gate.

Off when stage 1's deploy config carries no speculative_config.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import init_logger

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


_LOGGED_BLOCK: str | None = None
_LOGGED_NARROW = False
_LOGGED_NARROW_BLOCK: str | None = None

# SoC families whose one-query capture is known to fault. The 910_93 part
# trips a vector-core error (acl 507035) inside the packaged codec operator
# while that graph is being captured, so the narrow path stays off there
# unless an operator opts in explicitly -- see narrow_replay_enabled.
# "ascend910c" guards the device-name fallback: on some A3 containers the
# driver reports the part as "Ascend910C" rather than its 910_93 SoC code,
# and an unblocked unknown would let the capture brick the deployment.
# "ascend910b" is fail-safe rather than observed: no K-step attempt there
# ever reached a capture (all 8 combos faulted earlier, inside the reject
# kernel), so the one-query capture is unverified on that family.
_NARROW_BLOCKED_SOC_PREFIXES = ("ascend910_93", "ascend910c", "ascend910b")
_NARROW_SOC_LOGGED: str | None = None


def _narrow_soc_allows() -> bool:
    """Whether this part may capture the one-query graph by default.

    Judged from the part: the CANN-exported SoC code first, the device name as
    the fallback for a worker launched without them.
    """
    global _NARROW_SOC_LOGGED

    name = ""
    for variable in ("SOC_VERSION", "ASCEND_SOC_VERSION"):
        value = os.environ.get(variable, "").strip().lower()
        if value:
            name = value
            break
    if not name:
        try:
            import torch_npu

            name = str(torch_npu.npu.get_device_name(torch_npu.npu.current_device())).lower()
        except Exception:
            name = ""
    allows = True if not name else not name.startswith(_NARROW_BLOCKED_SOC_PREFIXES)
    if _NARROW_SOC_LOGGED != name:
        _NARROW_SOC_LOGGED = name
        logger.info(
            "[minicpmo] narrow replay SoC gate: name='%s' -> %s",
            name or "<unidentified>",
            "allowed" if allows else "blocked",
        )
    return allows


def _log_block_once(reason: str) -> None:
    """Say once why the loop never engaged; silence is the worse failure."""
    global _LOGGED_BLOCK
    if _LOGGED_BLOCK == reason:
        return
    _LOGGED_BLOCK = reason
    logger.info("[minicpmo] multi-frame Talker decode not engaged: %s", reason)


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
    if not getattr(model, "supports_multi_frame_decode", False):
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
    if not getattr(model, "supports_multi_frame_decode", False):
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


@dataclass
class NarrowStep:
    """What one codec frame has to change before the one-query graph replays.

    A K-query graph replay costs 1.33-1.47 ms of device time against the
    one-query graph's 0.83 ms, and it is a step change rather than a per-row
    cost -- a two-query graph costs the same as a four-query one. The loop only
    ever reads one row of the wide graph, so it replays the narrow one instead
    and moves the five things a frame changes:

    * ``inputs_embeds[r]`` -- request r's embedding for this frame
    * ``positions[r]``     -- its position, +frame
    * ``slot_mapping[r]``  -- the KV slot vLLM reserved for this frame
    * ``seq_lens[r]``      -- its length, +frame+1, in the buffer the graph reads
    * the block table      -- unchanged for the whole step

    The step's own K-wide layouts are cloned at ``begin`` because the narrow
    views alias them: ``slot_mapping[:rows]`` is the first row of
    ``slot_mapping.view(rows, K)``.
    """

    rows: int
    frames: int
    inputs_embeds: Any
    positions: Any
    slot_mapping: Any
    seq_lens: Any
    step_slots: Any
    step_positions: Any
    step_seq_lens: Any
    spans: list[tuple[int, int]]
    saved_descriptor: Any = None
    saved_runtime_mode: Any = None

    def select(self, frame: int) -> None:
        """Put frame ``frame``'s inputs where the captured replay reads them.

        Three copies and nothing else: everything a frame needs was computed
        once, at the start of the step, because these run between two graph
        replays and an allocation there costs more than the values do.
        """
        rows = self.rows
        self.slot_mapping[:rows].copy_(self.step_slots[:, frame])
        self.positions[:rows].copy_(self.step_positions[:, frame])
        self.seq_lens[:rows].copy_(self.step_seq_lens[:, frame])


def narrow_replay_enabled(runner: Any) -> bool:
    """Whether this worker should capture and replay the one-query graph.

    Off on a part whose one-query capture is known to fault
    (``_NARROW_BLOCKED_SOC_PREFIXES``), so a capture can never brick a
    deployment; those parts keep the K-query captures and the wide replay.
    """
    if not _narrow_soc_allows():
        return False
    model = getattr(runner, "model", None)
    if not getattr(model, "supports_multi_frame_decode", False):
        return False
    return int(getattr(runner, "num_spec_tokens", 0) or 0) > 0


def begin_narrow_step(
    *,
    runner: Any,
    forward_context: Any,
    frames: int,
    positions: Any,
    inputs_embeds: Any,
    spans: list[tuple[int, int]],
) -> NarrowStep | None:
    """Point the forward context at the one-query graph, or decline.

    Declines -- and the caller falls back to the wide replay -- whenever the
    narrow graph for this row count is not captured, which is what makes this
    safe to land before every path that reaches it has been exercised.
    """
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor

    # The KV geometry required by narrow replay lives in
    # platforms/npu/attention/fixed_kv_decode, which may be absent. Import
    # defensively: when missing, decline as documented and let the caller fall
    # back to wide replay; multi-frame decode and the stop chain are unaffected.
    try:
        from vllm_omni.platforms.npu.attention import fixed_kv_decode
    except Exception:
        fixed_kv_decode = None

    if fixed_kv_decode is None:
        return _decline("the fixed-KV decode backend is not present in this build")
    if not narrow_replay_enabled(runner) or inputs_embeds is None or positions is None:
        return None
    rows = len(spans)
    if rows <= 0:
        return None
    seq_lens = fixed_kv_decode.captured_seq_lens(rows)
    if seq_lens is None:
        return _decline(f"no one-query graph captured for {rows} rows")
    metadata = getattr(forward_context, "attn_metadata", None)
    if isinstance(metadata, dict):
        metadata = next(iter(metadata.values()), None)
    step_slot_mapping = getattr(metadata, "slot_mapping", None)
    live_seq_lens = getattr(metadata, "seq_lens_device", None)
    graph_slot_mapping = fixed_kv_decode.captured_slot_mapping(rows)
    if step_slot_mapping is None or live_seq_lens is None or graph_slot_mapping is None:
        return _decline("this step's attention metadata carries no slot mapping")
    if step_slot_mapping.shape[0] < rows * frames:
        return _decline("slot mapping is shorter than the step's own frames")

    narrow = NarrowStep(
        rows=rows,
        frames=frames,
        inputs_embeds=inputs_embeds,
        positions=positions,
        slot_mapping=graph_slot_mapping,
        seq_lens=seq_lens,
        step_slots=step_slot_mapping[: rows * frames].view(rows, frames).clone(),
        step_positions=positions[: rows * frames].view(rows, frames).clone(),
        # `seq_lens` already counts all K of this step's frames, and by frame k
        # only k + 1 have been written -- so row r, column k is
        # `live - frames + k + 1`.
        step_seq_lens=(
            live_seq_lens[:rows, None]
            - frames
            + torch.arange(1, frames + 1, device=live_seq_lens.device, dtype=live_seq_lens.dtype)
        ),
        spans=[(index, index + 1) for index in range(rows)],
    )
    descriptor = getattr(forward_context, "batch_descriptor", None)
    narrow.saved_descriptor = descriptor
    if descriptor is not None:
        # `uniform` too: a failed dispatch clears it on the descriptor it hands
        # back, and the captured key has it set. What this step is has not
        # changed -- it is one token per request against a graph captured for
        # exactly that -- only which graph replays it.
        forward_context.batch_descriptor = dataclasses.replace(descriptor, num_tokens=rows, num_reqs=rows, uniform=True)
    else:
        forward_context.batch_descriptor = BatchDescriptor(num_tokens=rows, num_reqs=rows, uniform=True)
    # The dispatcher builds its lookup descriptor from its own query length --
    # 4 at runtime, which is right for the step -- so it finds no key for the
    # narrow capture and hands back NONE, and the wrapper then runs the model
    # uncaptured without saying so. This step *is* a full-graph uniform decode;
    # only which graph it replays changed, which is what this function decides.
    narrow.saved_runtime_mode = getattr(forward_context, "cudagraph_runtime_mode", None)
    forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL
    global _LOGGED_NARROW
    if not _LOGGED_NARROW:
        _LOGGED_NARROW = True
        logger.info(
            "[minicpmo] multi-frame Talker decode replays the one-query graph: "
            "%d rows, descriptor %s, dispatched mode was %s",
            rows,
            forward_context.batch_descriptor,
            narrow.saved_runtime_mode,
        )
    return narrow


def end_narrow_step(forward_context: Any, narrow: NarrowStep | None) -> None:
    if narrow is None:
        return
    forward_context.batch_descriptor = narrow.saved_descriptor
    if narrow.saved_runtime_mode is not None:
        forward_context.cudagraph_runtime_mode = narrow.saved_runtime_mode


def _decline(reason: str) -> None:
    global _LOGGED_NARROW_BLOCK
    if _LOGGED_NARROW_BLOCK != reason:
        _LOGGED_NARROW_BLOCK = reason
        logger.info("[minicpmo] one-query Talker replay not engaged: %s", reason)
    return None


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
) -> Any:
    """Run ``frames`` codec frames and return the merged step output.

    ``after_forward`` is the runner's post-forward graph bookkeeping. It is
    called after every replay because that is where it sits in the single-frame
    path; under fixed-KV decode it is two pointer comparisons, and the frames of
    one step share their sequence lengths so there is nothing for it to rebind.
    """
    global _LOGGED_ENGAGE
    spans = model_kwargs_extra["request_token_spans"]
    infos = model_kwargs_extra["model_intermediate_buffer"]

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
                _compact_frame_zero_embeddings(inputs_embeds, spans)
            else:
                _write_frame_embeddings(model, inputs_embeds, input_ids, spans, infos, frame, narrow=narrow)
        elif frame > 0:
            _write_frame_embeddings(model, inputs_embeds, input_ids, spans, infos, frame)
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


# The Talker's vLLM-level vocabulary is two tokens wide: `compute_logits`
# returns one-hot stop rows ([0, -inf] / [-inf, 0]) and `sample` takes their
# argmax, so 0 means "keep going" and 1 is the stop token the deploy config
# names in `stop_token_ids`.
CONTINUE_TOKEN_ID = 0

# The id that same row emits to mean stop: the stop column wins the argmax,
# `parse_output` keeps ids below the two-wide vocab, and vLLM's `check_stop`
# only fires if this id is in the request's `stop_token_ids` -- which stage 1's
# pipeline constraints derive from this constant.
STOP_TOKEN_ID = 1

# Width of the Talker vLLM-level head: the two-wide continue/stop row.
# input_batch.vocab_size must report the same, because InputBatch stores
# top_k = vocab_size as its no-top-k sentinel and that holds only for 0 or 2.
STOP_ROW_WIDTH = 2


def drafts_this_step(runner: Any) -> int:
    """Frames the *next* step should be scheduled for, or 0 to stay generic.

    The count travels to the scheduler as speculative tokens because that is
    vLLM V1's only way of saying "this request advanced by more than one token",
    which is what grows the block table and reserves the KV slots the loop
    writes into. Nothing about it is speculative: the drafts are always
    `continue`, and the frames are generated sequentially rather than verified.
    """
    if not getattr(getattr(runner, "model", None), "supports_multi_frame_decode", False):
        return 0
    num_spec = int(getattr(runner, "num_spec_tokens", 0) or 0)
    if num_spec <= 0:
        return 0
    return num_spec + 1


def constant_drafts(
    valid_sampled_token_ids: Any,
    frames: int,
    num_reqs: int,
) -> list[list[int]]:
    """`continue` repeated -- for the whole batch, or for none of it.

    The drafts are how the next step gets K query positions per request, and
    the multi-frame loop can only run a step whose requests all have the *same*
    span: it replays one captured uniform-decode graph and samples one codec
    frame per position. So drafting per request is not an option. A request
    that sampled nothing this step (discarded, or finished on this step's stop
    token) would be scheduled one token while its neighbours got K, and
    `applies` would refuse the resulting step -- which `_model_forward` turns
    into a fatal error rather than let the single-forward path sample one frame
    from the last row of a K-row span.

    So: every request drafts, or nobody does. A step with no drafts is an
    ordinary one-frame-per-request decode, and the frames resume on the step
    after it.
    """
    for index in range(num_reqs):
        sampled = None
        if isinstance(valid_sampled_token_ids, list) and index < len(valid_sampled_token_ids):
            sampled = valid_sampled_token_ids[index]
        if not sampled:
            _log_block_once("a request in this batch sampled nothing; no drafts this step")
            return [[] for _ in range(num_reqs)]
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
    if logits is None or not getattr(runner.model, "supports_multi_frame_decode", False):
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
