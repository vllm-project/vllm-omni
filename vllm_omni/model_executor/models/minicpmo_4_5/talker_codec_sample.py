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
    def _cuda_codec_prelude_kernel(
        x_ptr,
        hist_ptr,
        step_ptr,
        min_ptr,
        stop_ptr,
        pen_ptr,
        temp_ptr,
        out_ptr,
        eos_id: tl.constexpr,
        vocab: tl.constexpr,
        block: tl.constexpr,
        hist: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.program_id(1) * block + tl.arange(0, block)
        valid = offsets < vocab
        count = tl.zeros((block,), tl.int32)
        for index in tl.static_range(hist):
            token = tl.load(hist_ptr + row * hist + index)
            count += ((token >= 0) & (offsets == token)).to(tl.int32)
        scores = tl.load(x_ptr + row * vocab + offsets, valid, other=0.0)
        stop = tl.load(stop_ptr + row * vocab + offsets, valid, other=False)
        censor = tl.load(step_ptr + row) < tl.load(min_ptr + row)
        scores = tl.where(censor & (stop | (offsets == eos_id)), -float("inf"), scores)
        penalty = tl.load(pen_ptr + row)
        alpha = tl.exp2(tl.log2(penalty) * count.to(tl.float32))
        scores = tl.where(scores < 0, scores * alpha, scores / alpha)
        temperature = tl.load(temp_ptr + row)
        scores = scores / tl.where(temperature == 0, 1.0, temperature)
        tl.store(out_ptr + row * vocab + offsets, scores, valid)

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


@dataclass
class TalkerCudaKStepState:
    """Fixed-address batch workspace; request binding and host commit stay outside capture."""

    history: torch.Tensor
    steps: torch.Tensor
    finished: torch.Tensor
    last_tokens: torch.Tensor
    temperatures: torch.Tensor
    penalties: torch.Tensor
    min_tokens: torch.Tensor
    max_tokens: torch.Tensor
    top_k: torch.Tensor
    top_p: torch.Tensor
    stop_mask: torch.Tensor
    uniforms: torch.Tensor
    tokens: torch.Tensor
    emits: torch.Tensor
    stops: torch.Tensor
    row_ids: torch.Tensor
    prelude_scores: torch.Tensor

    @classmethod
    def allocate(cls, rows: int, frames: int, device: torch.device) -> TalkerCudaKStepState:
        def integers(*shape: int) -> torch.Tensor:
            return torch.zeros(shape, dtype=torch.long, device=device)

        def floats(*shape: int) -> torch.Tensor:
            return torch.ones(shape, dtype=torch.float32, device=device)

        return cls(
            history=torch.full((rows, HISTORY_WINDOW), -1, dtype=torch.long, device=device),
            steps=integers(rows),
            finished=torch.zeros(rows, dtype=torch.bool, device=device),
            last_tokens=integers(rows),
            temperatures=floats(rows),
            penalties=floats(rows),
            min_tokens=integers(rows),
            max_tokens=integers(rows),
            top_k=integers(rows),
            top_p=floats(rows),
            stop_mask=torch.zeros((rows, VOCAB_SIZE), dtype=torch.bool, device=device),
            uniforms=floats(frames, rows),
            tokens=integers(frames, rows),
            emits=torch.zeros((frames, rows), dtype=torch.bool, device=device),
            stops=torch.zeros((frames, rows), dtype=torch.bool, device=device),
            row_ids=torch.arange(rows, device=device),
            # Deterministic buffer: snapshot/restore compares it bit-for-bit.
            prelude_scores=torch.zeros((rows, VOCAB_SIZE), dtype=torch.float32, device=device),
        )

    def __post_init__(self) -> None:
        # EOS id the native prelude was compiled/lauched for; None means eager.
        self._native_prelude_eos: int | None = None

    def _tensor_fields(self) -> dict[str, torch.Tensor]:
        return {name: value for name, value in vars(self).items() if isinstance(value, torch.Tensor)}

    def _launch_native_prelude(self, logits: torch.Tensor, eos_token_id: int) -> torch.Tensor:
        _cuda_codec_prelude_kernel[(self.history.shape[0], triton.cdiv(VOCAB_SIZE, _PRELUDE_BLOCK))](
            logits,
            self.history,
            self.steps,
            self.min_tokens,
            self.stop_mask,
            self.penalties,
            self.temperatures,
            self.prelude_scores,
            eos_id=eos_token_id,
            vocab=VOCAB_SIZE,
            block=_PRELUDE_BLOCK,
            hist=HISTORY_WINDOW,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return self.prelude_scores

    def warmup_native_prelude(self, eos_token_id: int, enabled: bool = True) -> bool:
        """Compile/launch outside capture; failure selects eager for this workspace."""
        if self.history.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("CUDA codec prelude warmup must run outside capture")
        if enabled and self._native_prelude_eos == eos_token_id:
            return True
        self._native_prelude_eos = None
        if not enabled or not _HAS_TRITON or not self.history.is_cuda:
            return False
        try:
            self._launch_native_prelude(torch.zeros_like(self.prelude_scores), eos_token_id)
            torch.cuda.current_stream(self.history.device).synchronize()
        except Exception as exc:
            logger.warning("CUDA codec prelude warmup failed; using eager: %s", exc)
            return False
        self._native_prelude_eos = eos_token_id
        return True

    def snapshot(self) -> dict[str, torch.Tensor]:
        return {name: tensor.clone() for name, tensor in self._tensor_fields().items()}

    def restore(self, snapshot: dict[str, torch.Tensor]) -> None:
        tensors = self._tensor_fields()
        if tensors.keys() != snapshot.keys():
            raise ValueError("CUDA K-step snapshot fields do not match workspace")
        for name, tensor in tensors.items():
            saved = snapshot[name]
            if tensor.shape != saved.shape or tensor.dtype != saved.dtype or tensor.device != saved.device:
                raise ValueError(f"CUDA K-step snapshot layout mismatch: {name}")
        for name, tensor in tensors.items():
            tensor.copy_(snapshot[name])

    def storage_key(self) -> tuple:
        return tuple(
            (name, tensor.data_ptr(), tuple(tensor.shape), tensor.stride(), tensor.dtype, tensor.device)
            for name, tensor in self._tensor_fields().items()
        )

    def sample(self, logits: torch.Tensor, frame: int, eos_token_id: int) -> torch.Tensor:
        """Sample one frame with tensor-only EOS, history and feedback transitions."""
        scores = logits.float().clone()
        # History has only 16 entries; scatter avoids a rows x vocab x history temporary.
        seen = torch.zeros_like(scores, dtype=torch.long)
        seen.scatter_add_(1, self.history.clamp_min(0), (self.history >= 0).long())
        alpha = self.penalties[:, None].pow(seen)
        scores = torch.where(scores < 0, scores * alpha, scores / alpha)
        mask_eos = self.steps < self.min_tokens
        scores.masked_fill_(mask_eos[:, None] & self.stop_mask, float("-inf"))
        scores[:, eos_token_id] = scores[:, eos_token_id].masked_fill(mask_eos, float("-inf"))
        scores = scores / self.temperatures.clamp_min(1e-6)[:, None]
        sorted_scores, indices = scores.sort(dim=-1, descending=True)
        ranks = torch.arange(scores.shape[-1], device=scores.device)[None, :]
        sorted_scores = sorted_scores.masked_fill(
            (self.top_k[:, None] > 0) & (ranks >= self.top_k[:, None]), float("-inf")
        )
        sorted_probs = sorted_scores.softmax(-1)
        remove = (sorted_probs.cumsum(-1) - sorted_probs) > self.top_p[:, None]
        sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
        cdf = sorted_scores.softmax(-1).cumsum(-1)
        choice = (cdf < self.uniforms[frame, :, None]).sum(-1).clamp_max(scores.shape[-1] - 1)
        random_tokens = indices.gather(1, choice[:, None]).squeeze(1)
        sampled = torch.where(self.temperatures == 0, scores.argmax(-1), random_tokens)
        return self.advance(sampled, frame, eos_token_id)

    def sample_native(
        self,
        logits: torch.Tensor,
        frame: int,
        eos_token_id: int,
        sampling: tuple,
        generators: tuple,
    ) -> torch.Tensor:
        fused_scores = None
        if (
            logits.is_cuda
            and getattr(self, "_native_prelude_eos", None) == eos_token_id
            and logits.dtype == torch.float32
            and logits.is_contiguous()
            and logits.shape == self.prelude_scores.shape
        ):
            # Do not catch a launch failure here: capture must be aborted by the caller.
            fused_scores = self._launch_native_prelude(logits, eos_token_id)
        history = self.history
        lengths = (history >= 0).sum(dim=1)
        if fused_scores is None:
            positions = torch.arange(HISTORY_WINDOW, device=logits.device)[None, :] + HISTORY_WINDOW - lengths[:, None]
            history = history.gather(1, positions.remainder(HISTORY_WINDOW)).clamp_min(0)
        groups: dict[tuple, list[int]] = {}
        for row, (temperature, top_k, top_p, stop_ids) in enumerate(sampling):
            key = (
                bool(temperature),
                int(top_k) if temperature else 0,
                float(top_p) if temperature else 1.0,
                tuple(stop_ids),
            )
            groups.setdefault(key, []).append(row)

        filtered = [None] * len(generators)
        for (stochastic, top_k, top_p, stop_ids), rows in groups.items():
            if len(rows) == len(sampling):

                def select(tensor):
                    return tensor
            else:
                index = torch.empty(len(rows), dtype=torch.long, device=logits.device)
                for local_row, row in enumerate(rows):
                    index[local_row].fill_(row)

                def select(tensor):
                    return tensor.index_select(0, index)

            group_state = TalkerCodecDeviceState(
                history=select(history),
                history_len=select(lengths),
                step=select(self.steps),
                max_tokens=select(self.max_tokens),
                finished=select(self.finished),
            )
            if fused_scores is not None:
                scores = _truncate_codec_logits(select(fused_scores), top_k=top_k, top_p=top_p)
            else:
                scores = prepare_codec_logits(
                    select(logits),
                    group_state,
                    select(self.min_tokens),
                    select(self.temperatures) if stochastic else self.temperatures.new_ones(len(rows)),
                    select(self.penalties),
                    eos_token_id=eos_token_id,
                    top_k=top_k,
                    top_p=top_p,
                    min_tokens_stop_ids=stop_ids,
                    fused_prelude=False,
                )
            for local_row, row in enumerate(rows):
                filtered[row] = scores[local_row : local_row + 1]

        sampled = []
        for row, (config, generator) in enumerate(zip(sampling, generators)):
            temperature = config[0]
            scores = filtered[row]
            token = (
                torch.multinomial(scores.softmax(-1), 1, generator=generator).reshape(1)
                if temperature
                else scores.argmax(dim=-1)
            )
            sampled.append(token)
        return self.advance(torch.cat(sampled), frame, eos_token_id)

    def advance(self, sampled: torch.Tensor, frame: int, eos_token_id: int) -> torch.Tensor:
        was_finished = self.finished.clone()
        next_step = self.steps + 1
        stop = (sampled == eos_token_id) | self.stop_mask[self.row_ids, sampled] | (next_step >= self.max_tokens)
        emit = (
            ~was_finished
            & (sampled != eos_token_id)
            & ~self.stop_mask[self.row_ids, sampled]
            & (next_step < self.max_tokens)
        )
        self.tokens[frame].copy_(sampled)
        self.emits[frame].copy_(emit)
        self.finished.logical_or_(stop)
        self.stops[frame].copy_(self.finished)
        self.steps.copy_(torch.where(was_finished, self.steps, next_step))
        self.last_tokens.copy_(torch.where(emit, sampled, self.last_tokens))
        next_history = torch.cat((self.history[:, 1:], sampled[:, None]), dim=1)
        self.history.copy_(torch.where(emit[:, None], next_history, self.history))
        return self.last_tokens


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
    if logits.is_cuda and torch.cuda.is_current_stream_capturing():
        stop_index = torch.empty(len(stop_ids), dtype=torch.long, device=logits.device)
        for index, token in enumerate(stop_ids):
            stop_index[index].fill_(token)
    else:
        stop_index = _stop_index_tensor(stop_ids, logits.device)
    stop_logits = logits.index_select(-1, stop_index)
    if eos_window_masked:
        censored = torch.full_like(stop_logits, float("-inf"))
    else:
        censored = torch.where(
            (state.step < min_tokens).reshape(-1, 1),
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
    valid = positions < state.history_len.reshape(-1, 1)
    safe_tokens = torch.where(valid, state.history, torch.zeros_like(state.history)).to(torch.long)
    if state.rep_counts is not None:
        # Review P3-6: reuse the per-request histogram buffer instead of
        # allocating a VOCAB_SIZE-wide tensor every frame. zero_() re-runs
        # under graph replay, so no stale count survives a replay.
        counts = state.rep_counts.zero_()
    else:
        counts = torch.zeros_like(logits, dtype=torch.float32)
    counts.scatter_add_(1, safe_tokens, valid.to(torch.float32))
    alpha = torch.pow(repetition_penalty.reshape(-1, 1), counts)
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
) -> torch.Tensor:
    """Prepare and filter logits without changing native NPU RNG semantics.

    ``min_p`` mirrors the engine's argmax-invariant ``MinPLogitsProcessor``,
    which runs after temperature and before top-k/top-p (``builtin.py:102-113``).
    ``min_tokens_stop_ids`` widens the censored stop set beyond the codec EOS;
    ``frequency_penalty``/``presence_penalty`` subtract the whole-stream counts.
    ``fused_prelude=False`` (stage connector ``extra`` key
    ``talker_codec_fused_prelude``) forces the eager chain for accuracy triage.
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
        logits = logits / temperature.reshape(-1, 1)
    # Engine step 7c: min_p. The engine runs it as an argmax-invariant
    # processor, i.e. after temperature and before top-k/top-p, and the greedy
    # branch never sees it -- which is why greedy_codec_sample has no min_p.
    if min_p > 0.0:
        probabilities = torch.softmax(logits, dim=-1)
        min_p_threshold = probabilities.amax(dim=-1, keepdim=True) * min_p
        logits.masked_fill_(probabilities < min_p_threshold, float("-inf"))
    return _truncate_codec_logits(logits, top_k=top_k, top_p=top_p)


def _truncate_codec_logits(logits: torch.Tensor, *, top_k: int, top_p: float) -> torch.Tensor:
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
