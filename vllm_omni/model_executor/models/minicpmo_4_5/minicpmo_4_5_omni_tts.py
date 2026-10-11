# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from:
# https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/main/modeling_minicpmo.py
"""MiniCPM-o 4.5 native autoregressive Talker.

Pipeline:
  1. Receive thinker hidden_states + full token IDs via additional_information
  2. Extract tts_bos..tts_eos region
  3. Build condition: emb_text(tokens) + projector_semantic(hidden) (hidden_text_merge)
  4. Project last hidden through head_code; vLLM Sampler picks the codec id
  5. Next decode embeds that id with emb_code and emits it to Code2Wav
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from functools import cached_property
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, tldevice, triton
from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.logits_processor import LogitsContext, LogitsProcessor
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
from vllm.v1.worker.gpu.states import RequestState

from vllm_omni.data_entry_keys import flatten_payload
from vllm_omni.engine.duplex.intermediate import get_tts_handoff
from vllm_omni.model_executor.models.minicpmo_4_5 import (
    MINICPMO45_DUPLEX_CODEC_TOKENS_PER_CHUNK,
    MINICPMO45_DUPLEX_TURN_END_CODEC_TOKENS,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.output_snapshot import RequestOutputSnapshot
from vllm_omni.platforms import current_omni_platform
from vllm_omni.utils.device_copy import index_to_device, to_device_nonblocking
from vllm_omni.utils.mm_outputs import partition_flat_payload
from vllm_omni.worker_v2.decode_burst import BurstOutputBatch, decode_burst_steps
from vllm_omni.worker_v2.omni_sampler import OmniSampler, OmniSamplingOutput, StandardSample

logger = init_logger(__name__)

_REPETITION_PENALTY_CHUNK_SIZE = 16
# ``past_window`` of MiniCPMTTS's codec repetition penalty: both generate() and
# generate_chunk() build it through gen_logits(), which hardcodes
# CustomRepetitionPenaltyLogitsProcessorRepeat(penalty, num_code, 16).
_CODEC_PENALTY_WINDOW = 16
# MiniCPMTTS.generate's max_new_token. The Talker context bounds this further;
# without it a request that never samples codec EOS keeps emitting frames for
# twice as long as upstream would, which is audible as a long silent tail.
_OFFLINE_CODEC_MAX_NEW_TOKENS = 2048
# Native duplex Talker must finish after one MiniCPMTTS.generate_chunk:
# 25 codec frames (``codec_chunk_frames``) plus the terminating sample.
# Without this, the single-vocab Sampler keeps the stage-1 request alive
# until codec EOS / 4096 and Thinker never starts the next model turn.
_DUPLEX_CODEC_TOKENS_PER_CHUNK = MINICPMO45_DUPLEX_CODEC_TOKENS_PER_CHUNK
_DUPLEX_TURN_END_CODEC_TOKENS = MINICPMO45_DUPLEX_TURN_END_CODEC_TOKENS
#: Frames the Talker forwards per generate_chunk before its cadence EOS.
_DUPLEX_CODEC_FRAMES_PER_CHUNK = MINICPMO45_DUPLEX_CODEC_TOKENS_PER_CHUNK - 1
#: On a turn-end chunk the Talker's EOS is masked for this many steps after
#: each 25-frame boundary: the model emits a cadence EOS there whether or not
#: its text is spoken, while a genuine end of text shows up as an EOS anywhere
#: else in the window.
_DUPLEX_TURN_END_BOUNDARY_MASK_STEPS = 5
#: Per-request native duplex output metadata, constant for one condition.
_DUPLEX_OUTPUT_META_KEYS = ("native_duplex", "duplex_epoch", "duplex_turn_id", "llm_output_text_utf8", "turn_end")


def _native_duplex_chunk_budget(meta: Mapping[str, Any] | None) -> tuple[int, int]:
    """Return ``(max_tokens, min_tokens)`` for one native-duplex Talker request."""
    turn_start = isinstance(meta, Mapping) and bool(meta.get("turn_start"))
    turn_end = isinstance(meta, Mapping) and bool(meta.get("turn_end"))
    if turn_end:
        # The turn-end chunk drains the text the Talker still owes: no floor
        # (an early EOS means the text is spoken) and a multi-unit ceiling.
        return _DUPLEX_TURN_END_CODEC_TOKENS, 0
    ceiling = _DUPLEX_CODEC_TOKENS_PER_CHUNK
    return ceiling, 0 if turn_start else ceiling


def _turn_end_boundary_eos_masked(step: int) -> bool:
    """Whether a turn-end chunk masks codec EOS at ``step`` forwarded frames.

    The Talker emits a cadence EOS after every 25 frames regardless of the text
    left, so a turn-end chunk ignores EOS in a short window after each boundary
    and lets the model continue; EOS elsewhere ends the chunk as usual.
    """
    if step < _DUPLEX_CODEC_FRAMES_PER_CHUNK:
        return False
    return step % _DUPLEX_CODEC_FRAMES_PER_CHUNK < _DUPLEX_TURN_END_BOUNDARY_MASK_STEPS


def blank_scheduler_prompt_for_penalties(
    prompt_token_ids: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """Return a penalty prompt whose every position is the pad id (``vocab_size``).

    No Talker prompt position is codec history: prefill conditioning arrives as
    embeddings from ``preprocess`` and decode embeds sampled ids with
    ``emb_code``. The scheduler ids are placeholders (``llm2tts`` fills them
    with ``0``, or with thinker token ids on the non-handoff path), so scoring
    them would tax unrelated codec tokens.
    """
    return torch.full_like(prompt_token_ids, int(vocab_size))


def _restore_weight_norm_weight(weight_g: torch.Tensor, weight_v: torch.Tensor) -> torch.Tensor:
    """Materialize ``weight_norm(..., dim=0)`` checkpoint parameters."""
    return torch._weight_norm(weight_v, weight_g, dim=0)


def _apply_batched_repetition_penalty(
    logits: torch.Tensor,
    histories: Sequence[torch.Tensor],
    *,
    penalty: float | torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """Apply request-local frequency penalties to a batch of codec logits.

    ``penalty`` may be a scalar or one value per row, mirroring upstream's
    per-request ``sampling_params.repetition_penalty``.
    """
    if logits.ndim != 2:
        raise ValueError(f"batched codec logits must be 2D, got shape {tuple(logits.shape)}")
    batch_size, vocab_size = logits.shape
    if len(histories) != batch_size:
        raise ValueError(f"expected {batch_size} codec histories, got {len(histories)}")
    if batch_size == 0:
        return logits

    penalties = torch.as_tensor(penalty, device=logits.device, dtype=logits.dtype).reshape(-1)
    if penalties.numel() == 1:
        penalties = penalties.expand(batch_size)
    elif penalties.numel() != batch_size:
        raise ValueError(f"expected 1 or {batch_size} codec repetition penalties, got {penalties.numel()}")
    penalized = logits.clone()
    for start in range(0, batch_size, _REPETITION_PENALTY_CHUNK_SIZE):
        end = min(start + _REPETITION_PENALTY_CHUNK_SIZE, batch_size)
        chunk_logits = logits[start:end]
        chunk_histories = histories[start:end]
        # A spare column absorbs padding in fixed-size device histories.
        history_device = "cpu" if all(history.device.type == "cpu" for history in chunk_histories) else logits.device
        width = vocab_size if history_device == "cpu" else vocab_size + 1
        encoded_rows: list[torch.Tensor] = []
        for local_row, history in enumerate(chunk_histories):
            recent = to_device_nonblocking(history.reshape(-1)[-window_size:].long(), history_device)
            if recent.numel() > 0:
                encoded_rows.append(recent + local_row * width)
        if not encoded_rows:
            continue
        encoded = encoded_rows[0] if len(encoded_rows) == 1 else torch.cat(encoded_rows)
        encoded = to_device_nonblocking(encoded, logits.device)
        frequencies = torch.zeros((end - start) * width, dtype=torch.long, device=logits.device)
        frequencies.scatter_add_(0, encoded, torch.ones_like(encoded))
        frequencies = frequencies.reshape(end - start, width)[:, :vocab_size]
        alpha = torch.pow(penalties[start:end].unsqueeze(1), frequencies.to(dtype=logits.dtype))
        penalized[start:end] = torch.where(chunk_logits < 0, chunk_logits * alpha, chunk_logits / alpha)

    return penalized


@triton.jit
def _codec_window_penalty_kernel(
    logits,
    logits_row_stride: tl.constexpr,
    logits_col_stride: tl.constexpr,
    slots,
    token_ids,
    token_row_stride: tl.constexpr,
    token_col_stride: tl.constexpr,
    total_len,
    prompt_len,
    penalties,
    prefix_history,
    prefix_row_stride: tl.constexpr,
    prefix_col_stride: tl.constexpr,
    vocab_size: tl.constexpr,
    window_size: tl.constexpr,
    has_prefix: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(slots + row)
    penalty = tl.load(penalties + slot)
    if penalty == 1.0:
        return
    end = tl.load(total_len + slot)
    prompt = tl.load(prompt_len + slot)
    offsets = tl.arange(0, triton.next_power_of_2(window_size))
    positions = end - window_size + offsets
    valid = (offsets < window_size) & (positions >= tl.maximum(end - window_size, prompt))
    history = tl.load(
        token_ids + slot * token_row_stride + positions * token_col_stride,
        mask=valid,
        other=-1,
    )
    if has_prefix:
        relative = positions - prompt
        seed_valid = (offsets < window_size) & (relative < 0) & (relative >= -window_size)
        seeds = tl.load(
            prefix_history + slot * prefix_row_stride + (relative + window_size) * prefix_col_stride,
            mask=seed_valid,
            other=-1,
        )
        seed_valid &= seeds >= 0
        history = tl.where(seed_valid, seeds, history)
        valid |= seed_valid
    tokens = tl.program_id(1) * block_size + tl.arange(0, block_size)
    frequencies = tl.sum(((history[:, None] == tokens[None, :]) & valid[:, None]).to(tl.int32), axis=0)
    values = tl.load(
        logits + row * logits_row_stride + tokens * logits_col_stride,
        mask=tokens < vocab_size,
        other=0,
    )
    dtype = values.dtype
    # Match the existing helper's casts before pow and before multiplication.
    alpha = tldevice.pow(penalty.to(dtype).to(tl.float32), frequencies.to(tl.float32)).to(dtype).to(tl.float32)
    values = values.to(tl.float32)
    output = tl.where(values < 0, values * alpha, tldevice.div_rn(values, alpha))
    tl.store(logits + row * logits_row_stride + tokens * logits_col_stride, output, mask=tokens < vocab_size)


def _apply_codec_window_penalty_gpu(
    logits: torch.Tensor,
    slots: torch.Tensor,
    all_token_ids: torch.Tensor,
    total_len: torch.Tensor,
    prompt_len: torch.Tensor,
    penalty: torch.Tensor,
    *,
    window_size: int,
    prefix_history: torch.Tensor | None = None,
) -> None:
    """In place: ``_apply_batched_repetition_penalty`` on the runner's device token history.

    Model Runner V2 keeps every request's sampled ids (``all_token_ids``) and
    lengths on the device, so the 16-frame window is gathered there instead of
    being rebuilt on the host each step. CUDA fuses the window gather, counts
    and scaling, avoiding a full-row history copy and temporary vocabulary
    tensors. Other devices retain the Torch path. Row ``r`` is scored over its last
    ``window_size`` output ids, ``[max(total_len - window, prompt_len),
    total_len)``: the codes sampled so far, exactly the history the V1 path
    builds in ``make_omni_output``. It uses the V1 helper's frequency-based
    arithmetic; fused float32 pow can differ by a few rounding bits.
    """
    num_rows, vocab_size = logits.shape
    if num_rows == 0:
        return
    if logits.is_cuda and current_omni_platform.is_cuda():
        prefix = all_token_ids if prefix_history is None else prefix_history
        _codec_window_penalty_kernel[(num_rows, triton.cdiv(vocab_size, 256))](
            logits,
            *logits.stride(),
            slots,
            all_token_ids,
            *all_token_ids.stride(),
            total_len,
            prompt_len,
            penalty,
            prefix,
            *prefix.stride(),
            vocab_size,
            window_size,
            prefix_history is not None,
            256,
        )
        return
    slots = slots.long()
    end = total_len.index_select(0, slots).long()
    start = torch.maximum(end - window_size, prompt_len.index_select(0, slots).long())
    offsets = torch.arange(window_size, device=logits.device, dtype=torch.long)
    positions = end.unsqueeze(1) - window_size + offsets.unsqueeze(0)
    valid = positions >= start.unsqueeze(1)
    rows = all_token_ids.index_select(0, slots)
    history = rows.gather(1, positions.clamp_min(0)).long()
    if prefix_history is not None:
        relative = positions - prompt_len.index_select(0, slots).long().unsqueeze(1)
        seed_positions = (relative + window_size).clamp(0, window_size - 1)
        seeds = prefix_history.index_select(0, slots).gather(1, seed_positions)
        seed_valid = (relative < 0) & (relative >= -window_size) & (seeds >= 0)
        history = torch.where(seed_valid, seeds, history)
        valid = valid | seed_valid
    # Invalid positions count into a spare column that is dropped below.
    history = torch.where(valid & (history >= 0) & (history < vocab_size), history, vocab_size)
    frequencies = torch.zeros((num_rows, vocab_size + 1), dtype=torch.long, device=logits.device)
    frequencies.scatter_add_(1, history, torch.ones_like(history))
    alpha = torch.pow(
        penalty.index_select(0, slots).to(dtype=logits.dtype).unsqueeze(1),
        frequencies[:, :vocab_size].to(dtype=logits.dtype),
    )
    logits.copy_(torch.where(logits < 0, logits * alpha, logits / alpha))


_DECODE_OUTPUT_BLOCK = 128


# One compiled variant: no pointer-alignment or row-count specializations
# that could trigger a JIT compile on a live step after the startup warmup.
@triton.jit(
    do_not_specialize=[
        "token_ids",
        "slot_ids",
        "seq_lens",
        "prompt_len",
        "empty_speech",
        "controls",
        "codes_out",
        "frame_valid_out",
        "forced_out",
        "mask_out",
        "num_reqs",
        "num_tokens",
    ]
)
def _mrv2_decode_output_kernel(
    token_ids,
    slot_ids,
    seq_lens,
    prompt_len,
    empty_speech,
    controls,
    codes_out,
    frame_valid_out,
    forced_out,
    mask_out,
    num_reqs,
    num_tokens,
    eos_id: tl.constexpr,
    context_limit: tl.constexpr,
    max_new_minus_one: tl.constexpr,
    frames_per_chunk: tl.constexpr,
    mask_steps: tl.constexpr,
    native: tl.constexpr,
    block: tl.constexpr,
):
    """``make_omni_output_mrv2``'s decode-only device math in one launch.

    Row ``r < num_reqs`` is request ``r``'s single token (``token r``). Every
    value is an exact integer/boolean expression of the eager Torch path.
    """
    rows = tl.program_id(0) * block + tl.arange(0, block)
    token_mask = rows < num_tokens
    req_mask = rows < num_reqs
    token = tl.load(token_ids + rows, mask=token_mask, other=0).to(tl.int64)
    tl.store(codes_out + rows, token, mask=token_mask)
    slot = tl.load(slot_ids + rows, mask=req_mask, other=0).to(tl.int64)
    prompt = tl.load(prompt_len + slot, mask=req_mask, other=0).to(tl.int64)
    step = tl.load(seq_lens + rows, mask=req_mask, other=0).to(tl.int64) - prompt
    empty = tl.load(empty_speech + slot, mask=req_mask, other=0) != 0
    decode = step > 0
    ended = (decode & (token == eos_id)) | empty
    valid = decode & (~ended)
    limit = tl.minimum(tl.maximum(context_limit - prompt, 0), max_new_minus_one)
    if native:
        control_limit = tl.load(controls + slot * 3, mask=req_mask, other=-1)
        min_steps = tl.load(controls + slot * 3 + 1, mask=req_mask, other=-1)
        turn_end_drain = tl.load(controls + slot * 3 + 2, mask=req_mask, other=-1)
        limit = tl.where(control_limit >= 0, control_limit, limit)
        # ``step >= frames_per_chunk > 0``: C and Python remainders agree.
        boundary = (step >= frames_per_chunk) & ((step % frames_per_chunk) < mask_steps)
        mask_eos = (step < min_steps) | ((turn_end_drain == 1) & boundary)
    forced = ended | (step >= limit)
    tl.store(frame_valid_out + rows, valid & req_mask, mask=token_mask)
    tl.store(forced_out + rows, forced, mask=req_mask)
    if native:
        tl.store(mask_out + rows, mask_eos & (~forced), mask=req_mask)


def _mrv2_decode_output_gpu(
    token_ids: torch.Tensor,
    slot_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    prompt_len: torch.Tensor,
    empty_speech: torch.Tensor,
    controls: torch.Tensor | None,
    *,
    num_reqs: int,
    eos_id: int,
    context_limit: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """(codes [T, 1] int64, frame_valid [T], forced_eos [R], mask_eos [R] | None)."""
    num_tokens = int(token_ids.shape[0])
    device = token_ids.device
    codes = torch.empty((num_tokens, 1), dtype=torch.long, device=device)
    frame_valid = torch.empty(num_tokens, dtype=torch.bool, device=device)
    forced = torch.empty(num_reqs, dtype=torch.bool, device=device)
    native = controls is not None
    mask = torch.empty(num_reqs, dtype=torch.bool, device=device) if native else None
    _mrv2_decode_output_kernel[(triton.cdiv(max(num_tokens, 1), _DECODE_OUTPUT_BLOCK),)](
        token_ids,
        slot_ids,
        seq_lens,
        prompt_len,
        empty_speech,
        controls if native else slot_ids,
        codes,
        frame_valid,
        forced,
        mask if native else forced,
        num_reqs,
        num_tokens,
        eos_id=int(eos_id),
        context_limit=int(context_limit),
        max_new_minus_one=_OFFLINE_CODEC_MAX_NEW_TOKENS - 1,
        frames_per_chunk=_DUPLEX_CODEC_FRAMES_PER_CHUNK,
        mask_steps=_DUPLEX_TURN_END_BOUNDARY_MASK_STEPS,
        native=native,
        block=_DECODE_OUTPUT_BLOCK,
    )
    return codes, frame_valid, forced, mask


class _CodecWindowPenaltiesState(LogitsProcessor):
    """MRv2 sampler penalties for the MiniCPM-o Talker.

    Replaces the sampler's presence-based repetition penalty (whole prompt and
    output) with MiniCPMTTS's frequency penalty over the last 16 codes, scored
    on the device. Frequency and presence penalties, if a request sets them,
    still go through the upstream state, which keeps the output bin counts.

    The MRv2 sampler runs the processors in ``sampler.logits_processors``, so
    ``_install_mrv2_talker_sampler`` puts this state in the stock penalty
    state's slot of that pipeline.
    """

    def __init__(self, base: PenaltiesState, *, window_size: int) -> None:
        from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor

        self.base = base
        self.req_states = base.req_states
        self.window_size = int(window_size)
        max_num_reqs = int(self.req_states.max_num_reqs)
        self.repetition_penalty = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.repetition_penalty.np.fill(1.0)
        self.repetition_penalty.copy_to_uva()
        self.use_window = np.zeros(max_num_reqs, dtype=bool)
        self.use_penalty = np.zeros(max_num_reqs, dtype=bool)
        self.prefix_history = torch.full(
            (max_num_reqs, self.window_size), -1, dtype=torch.long, device=self.req_states.device
        )

    @property
    def output_bin_counts(self) -> torch.Tensor:
        return self.base.output_bin_counts

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        self.prefix_history[req_idx].fill_(-1)
        repetition = float(getattr(sampling_params, "repetition_penalty", 1.0))
        self.repetition_penalty.np[req_idx] = repetition
        self.use_window[req_idx] = repetition != 1.0
        base_applies = self.base.add_request(
            req_idx,
            SimpleNamespace(
                repetition_penalty=1.0,
                frequency_penalty=float(getattr(sampling_params, "frequency_penalty", 0.0)),
                presence_penalty=float(getattr(sampling_params, "presence_penalty", 0.0)),
            ),
        )
        self.use_penalty[req_idx] = bool(self.use_window[req_idx] or base_applies)
        return bool(self.use_penalty[req_idx])

    def apply_staged_writes(self) -> None:
        self.repetition_penalty.copy_to_uva()
        self.base.apply_staged_writes()

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        if np.any(self.use_window[ctx.idx_mapping_np]):
            _apply_codec_window_penalty_gpu(
                logits,
                ctx.expanded_idx_mapping,
                self.req_states.all_token_ids.gpu,
                self.req_states.total_len.gpu,
                self.req_states.prompt_len.gpu,
                self.repetition_penalty.gpu,
                window_size=self.window_size,
                prefix_history=self.prefix_history,
            )
        return self.base.apply(logits, ctx)


def _mrv2_duplex_output_partition(
    outputs: dict[str, Any],
    output_meta: dict[str, list[torch.Tensor]],
    entries: Sequence[tuple[Any, ...]],
    token_axis_sizes: set[int],
) -> dict[str, Any] | RequestOutputSnapshot:
    """Per-request payloads for one native duplex Talker step, built directly.

    ``outputs`` is the host copy of ``make_omni_output_mrv2``'s payload, whose
    ``meta.finished`` is one ``[num_reqs]`` tensor; ``output_meta`` holds the
    step's per-request metadata tensors. The result equals what
    ``OmniARModelRunner._build_async_chunk_outputs_from_mm`` builds from the
    per-request list layout: token-axis views for codes and validity, owned
    clones of every per-request scalar/text tensor, and client entries that
    share the inter-stage tensors. Any other layout gets the list layout back
    and keeps the runner's generic partition.
    """
    num_rows = len(entries)
    codes = outputs.get("codes")
    meta = outputs.get("meta")
    finished = meta.get("finished") if isinstance(meta, dict) else outputs.get("meta.finished")
    audio = codes.get("audio") if isinstance(codes, dict) else None
    valid = meta.get("codec_frame_valid") if isinstance(meta, dict) else None
    if (
        isinstance(codes, dict)
        and isinstance(meta, dict)
        and list(outputs) == ["codes", "meta"]
        and list(codes) == ["audio"]
        and list(meta) == ["codec_frame_valid", "finished"]
        and isinstance(audio, torch.Tensor)
        and isinstance(valid, torch.Tensor)
        and isinstance(finished, torch.Tensor)
        and audio.dim() > 0
        and audio.shape[0] in token_axis_sizes
        and valid.dim() > 0
        and valid.shape[0] in token_axis_sizes
        and tuple(finished.shape) == (num_rows,)
        and all(len(column) == num_rows for column in output_meta.values())
    ):
        keys = ["codes.audio", "meta.codec_frame_valid", "meta.finished"]
        keys += [f"meta.{key}" for key in _DUPLEX_OUTPUT_META_KEYS]
        inter_template, client_template = partition_flat_payload(dict.fromkeys(keys))
        inter_keys, client_keys = list(inter_template), list(client_template)
        finished_rows = finished.unbind()
        columns = [output_meta[key] for key in _DUPLEX_OUTPUT_META_KEYS]
        inter_stage: list[dict[str, Any] | None] = []
        client: list[dict[str, Any] | None] = []
        for i, entry in enumerate(entries):
            start, end = entry[0], entry[0] + entry[1]
            values = [audio[start:end].contiguous(), valid[start:end].contiguous(), finished_rows[i].clone()]
            values += [column[i].clone() for column in columns]
            flat = dict(zip(keys, values))
            inter_stage.append({key: flat[key] for key in inter_keys} or None)
            client.append({key: flat[key] for key in client_keys} or None)
        return RequestOutputSnapshot(inter_stage=inter_stage, client=client)
    # Generic layout: per-request lists, as the runner's partition expects.
    if isinstance(meta, dict):
        if isinstance(finished, torch.Tensor):
            meta["finished"] = list(finished.unbind())
        for key in _DUPLEX_OUTPUT_META_KEYS:
            meta[key] = list(output_meta[key])
    else:
        if isinstance(finished, torch.Tensor):
            outputs["meta.finished"] = list(finished.unbind())
        for key in _DUPLEX_OUTPUT_META_KEYS:
            outputs[f"meta.{key}"] = list(output_meta[key])
    return outputs


def _install_mrv2_talker_sampler(sampler: Any, talker: "MiniCPMO45OmniTTSForConditionalGeneration") -> Any:
    """Wrap the runner's MRv2 sampler with codec penalties and EOS control.

    The upstream sampler applies ``min_tokens`` (codec EOS is a stage stop
    token), temperature and top-k/top-p on the device. Seeded native duplex
    rows retain V1's request-local draw to preserve codec onset behavior.
    The Talker adds its 16-frame codec penalty and overrides rows the model
    terminates (``_mrv2_forced_eos``) with codec EOS after sampling, exactly
    where V1's ``_force_eos_on_sampled_ids`` does.
    """

    stock = sampler.penalties_state
    if isinstance(stock, _CodecWindowPenaltiesState):
        # A second install would nest the window state and score it twice.
        raise RuntimeError("MiniCPM-o Talker: MRv2 sampler already carries the codec window penalty")
    processors = sampler.logits_processors
    slot = next((i for i, processor in enumerate(processors) if processor is stock), None)
    if slot is None:
        raise RuntimeError("MiniCPM-o Talker: MRv2 sampler has no penalty stage in logits_processors")
    window = _CodecWindowPenaltiesState(stock, window_size=_CODEC_PENALTY_WINDOW)
    # The sampler applies (and registers requests with) the list entries;
    # ``penalties_state`` is still read by the runner for output bin counts.
    processors[slot] = window
    sampler.penalties_state = window
    talker._mrv2_empty_speech = torch.zeros(
        int(sampler.req_states.max_num_reqs), dtype=torch.bool, device=sampler.req_states.device
    )
    logger.info(
        "MiniCPM-o Talker: MRv2 sampler with device-side %d-frame codec penalty and EOS control",
        _CODEC_PENALTY_WINDOW,
    )
    talker._mrv2_penalties = sampler.penalties_state
    if current_omni_platform.is_cuda() and sampler.req_states.device.type == "cuda":
        from vllm_omni.model_executor.models.minicpmo_4_5.duplex.mrv2 import MiniCPMO45SeededCodecSampler

        sampler = MiniCPMO45SeededCodecSampler(sampler, talker)
        talker._mrv2_seeded_codec_sampler = sampler
    return MiniCPMO45TalkerSampler(sampler, talker)


class MiniCPMO45TalkerSampler(OmniSampler):
    omni_static_staged_writes = True

    def __init__(self, base_sampler, talker):
        super().__init__(base_sampler)
        self.talker = talker

    def __call__(self, logits: torch.Tensor, input_batch: Any) -> Any:
        forced = self.talker.take_mrv2_forced_eos(input_batch, self.req_states, logits.shape[0])
        masked = getattr(self.talker, "_mrv2_mask_eos", None)
        if forced is not None and masked is not None and masked.shape[0] == logits.shape[0]:
            from vllm_omni.model_executor.models.minicpmo_4_5.duplex.mrv2 import SeededCodecDecodeGraphs

            graphs = getattr(getattr(self.talker, "_mrv2_seeded_codec_sampler", None), "decode_graphs", None)
            if isinstance(graphs, SeededCodecDecodeGraphs):
                # Same kernels (mask, sampling, forced EOS) replayed as one graph.
                output = graphs.try_sample(logits, input_batch, masked, forced)
                if output is not None:
                    return output
            logits[:, int(self.talker._codec_eos_id)].masked_fill_(masked, float("-inf"))
        output = self.base_sampler(logits, input_batch)
        if forced is not None:
            sampled = output.sampled_token_ids
            sampled.masked_fill_(forced.view(-1, *([1] * (sampled.ndim - 1))), int(self.talker._codec_eos_id))
        return output

    def sample_step(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        req_states: RequestState,
        grammar_output: GrammarOutput | None,
        standard_sample: StandardSample,
    ) -> OmniSamplingOutput:
        result = super().sample_step(hidden_states, input_batch, req_states, grammar_output, standard_sample)
        infos = getattr(self.talker, "_mrv2_output_infos", ())
        if any(info.get("native_duplex") is True for info in infos):
            # Code2Wav consumes codec ids and control metadata only.
            result.include_hidden_states = False
            result.finalize_multimodal = self.talker.mrv2_codec_history_finalizer(input_batch, infos)
        return result


class _TalkerDecodeBurst:
    """``DecodeBurst`` over native duplex codec rows (see worker_v2.decode_burst).

    A row continues until it draws codec EOS, sampled or forced; the
    scheduler stops it at that token. Every further step of the burst still
    draws for it, so each request's generator resumes after its last kept
    draw once the host knows how many were kept.
    """

    def __init__(
        self,
        talker: "MiniCPMO45OmniTTSForConditionalGeneration",
        steps: int,
        generators: list[torch.Generator | None],
    ) -> None:
        self.talker = talker
        self.steps = steps
        self._generators = generators
        # Per step: each row's generator offset after that step's draw.
        self._offsets: list[list[int]] = []

    def continues(self, sampled_token_ids: torch.Tensor) -> torch.Tensor:
        return sampled_token_ids[:, 0] != int(self.talker._codec_eos_id)

    def record_step(self) -> None:
        self._offsets.append([0 if generator is None else generator.get_offset() for generator in self._generators])

    def merge_outputs(self, outputs: list[dict[str, Any]], live: list[torch.Tensor]) -> dict[str, Any]:
        """``make_omni_output_mrv2`` payloads, each request's steps contiguous.

        Steps after a row ended forward its EOS id, which is never a valid
        frame; ``finished`` is the row's value at its last kept step.
        """
        num_reqs = int(live[0].shape[0])
        codes = torch.stack([output["codes"]["audio"][:num_reqs, 0] for output in outputs], dim=1)
        valid = torch.stack(
            [output["meta"]["codec_frame_valid"][:num_reqs] & keep for output, keep in zip(outputs, live)], dim=1
        )
        finished = outputs[0]["meta"]["finished"]
        for output, keep in zip(outputs[1:], live[1:]):
            finished = torch.where(keep, output["meta"]["finished"], finished)
        return {
            "codes": {"audio": codes.reshape(-1, 1)},
            "meta": {"codec_frame_valid": valid.reshape(-1), "finished": finished},
        }

    def finalizer(
        self, batch: BurstOutputBatch
    ) -> Callable[[dict[str, Any], list[int]], dict[str, Any] | RequestOutputSnapshot]:
        return self.talker._codec_history_finalizer(
            batch, self.talker._mrv2_output_infos, self.talker._mrv2_last_output_meta
        )

    def finish(self, num_sampled_np: np.ndarray, copy_event: torch.cuda.Event) -> None:
        self.talker._mrv2_seeded_codec_sampler.defer_rng_rewind(
            self._generators, self._offsets, num_sampled_np, copy_event
        )


class _MiniCPMTTSProjector(nn.Module):
    """Checkpoint-compatible hidden-state projector used by MiniCPMTTS."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(hidden_states)))


class MiniCPMO45OmniTTSForConditionalGeneration(nn.Module, SupportsPP):
    """Runner-owned MiniCPM-o 4.5 Talker that emits codec tokens only."""

    requires_request_sample_eligibility = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import MiniCPMOConfig

        config: MiniCPMOConfig = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        self._force_eos_rows: list[bool] | torch.Tensor | None = None
        self._mask_eos_rows: list[bool] | torch.Tensor | None = None
        self._pending_force_eos_rows: list[bool] | torch.Tensor | None = None
        self._penalty_histories: list[torch.Tensor] | torch.Tensor | None = None
        # Owned decode input IDs, keyed by request ID. CUDA IDs remain on
        # device through EOS control, penalty scoring and output construction.
        self._decode_codec_ids: dict[str, tuple[torch.Tensor, int]] = {}
        self._request_audio_states: dict[str, dict[str, Any]] = {}
        # Mirrors upstream TTSStreamingGenerator._chunk_info: one committed
        # condition plus, during a rollover, one immutable recompute recipe.
        self._request_condition_states: dict[str, dict[str, Any]] = {}
        self._deferred_cleanup_ids: set[str] = set()

        tts_config = getattr(config, "tts_config", None)
        if tts_config is None and getattr(config, "model_type", None) == "minicpmtts":
            tts_config = config
        if tts_config is not None:
            self._tts_config = tts_config
            self._tts_bos_id = getattr(tts_config, "audio_bos_token_id", 151687)
            self._text_eos_id = getattr(tts_config, "text_eos_token_id", 151692)
            self._num_audio_tokens = getattr(tts_config, "num_audio_tokens", 6562)
            self._codec_eos_id = int(getattr(tts_config, "eos_token_id", self._num_audio_tokens - 1))
            self._hidden_size = getattr(tts_config, "hidden_size", 768)
            self._normalize = getattr(tts_config, "normalize_projected_hidden", True)
        else:
            self._tts_config = None
            self._codec_eos_id = 0

        self.has_preprocess = True
        self.has_postprocess = False
        # Same-step codes travel through make_omni_output from the previous
        # sampled id (decode preprocess embeds that id via emb_code). They are
        # CUDA output deltas stay on device until the runner's output copy;
        # intermediate-buffer updates contain only empty CPU placeholders.
        self._init_native_talker(prefix)
        # Model Runner V2 keeps the sampled id, codec history and EOS state on
        # the GPU (see make_omni_output_mrv2 and mrv2_custom_sampler).
        self._use_v2_model_runner = bool(getattr(vllm_config.model_config, "use_v2_model_runner", False))
        # Per request slot: the Thinker handed over an empty condition.
        self._mrv2_empty_speech: torch.Tensor | None = None
        # Rows the sampler must force to codec EOS, computed with this step's output.
        self._mrv2_forced_eos: torch.Tensor | None = None
        self._mrv2_decode_rows_logged = False
        # This step's per-request duplex metadata, handed from
        # make_omni_output_mrv2 to the codec history finalizer.
        self._mrv2_output_meta: tuple[list[dict[str, Any]], dict[str, list[torch.Tensor]]] | None = None
        # The metadata the latest finalizer consumed; a decode burst's merged
        # output reuses it (it is constant across the burst's steps).
        self._mrv2_last_output_meta: tuple[list[dict[str, Any]], dict[str, list[torch.Tensor]]] | None = None
        # The model-owned codec limit also sizes scheduler KV lookahead.
        self._mrv2_decode_burst_steps = decode_burst_steps(vllm_config)

    #: Model Runner V2 output leaves are allocated by every
    #: ``make_omni_output_mrv2`` call (codes via a dtype-converting copy,
    #: validity and forced EOS from this step's device ops) and nothing writes
    #: them afterwards, so the runner copies them to the host without a
    #: snapshot. Per-request duplex metadata stays out of the device payload;
    #: the codec history finalizer attaches it on the host.
    mm_outputs_fresh_per_step = True

    def _init_native_talker(self, prefix: str) -> None:
        if self._tts_config is None:
            raise ValueError("MiniCPM-o continuous Talker requires tts_config")
        cfg = self._tts_config
        if int(getattr(cfg, "num_vq", 1)) != 1:
            raise ValueError(
                "MiniCPM-o continuous Talker currently requires num_vq=1; "
                f"checkpoint reports {getattr(cfg, 'num_vq', None)}"
            )
        llama_config = LlamaConfig(
            vocab_size=32000,
            hidden_size=int(cfg.hidden_size),
            intermediate_size=int(cfg.intermediate_size),
            num_hidden_layers=int(cfg.num_hidden_layers),
            num_attention_heads=int(cfg.num_attention_heads),
            num_key_value_heads=int(cfg.num_key_value_heads),
            hidden_act=getattr(cfg, "hidden_act", "silu"),
            max_position_embeddings=int(cfg.max_position_embeddings),
            rms_norm_eps=float(getattr(cfg, "rms_norm_eps", 1e-6)),
            tie_word_embeddings=False,
        )
        talker_config = self.vllm_config.with_hf_config(llama_config, architectures=["LlamaForCausalLM"])
        talker_config.model_config.hf_text_config = llama_config
        self.tts_model = LlamaModel(
            vllm_config=talker_config,
            prefix=maybe_prefix(prefix, "tts_obj.model"),
        )
        self.emb_text = nn.Embedding(int(cfg.num_text_tokens), int(cfg.hidden_size))
        self.projector_semantic = _MiniCPMTTSProjector(int(cfg.llm_dim), int(cfg.hidden_size))
        self.emb_code = nn.ModuleList(
            [nn.Embedding(int(cfg.num_audio_tokens), int(cfg.hidden_size)) for _ in range(int(cfg.num_vq))]
        )
        self.head_code = nn.ModuleList(
            [nn.Linear(int(cfg.hidden_size), int(cfg.num_audio_tokens), bias=False) for _ in range(int(cfg.num_vq))]
        )
        self.make_empty_intermediate_tensors = self.tts_model.make_empty_intermediate_tensors

    def _text_embedding_row(self, token_id: int) -> torch.Tensor:
        """``emb_text`` of one constant id as a weight-row view: no index upload or gather."""
        weight = self.emb_text.weight
        token_id = int(token_id)
        if not 0 <= token_id < weight.shape[0]:
            return self.emb_text(index_to_device([token_id], weight.device))
        return weight[token_id : token_id + 1]

    def _boundary_embeddings(self) -> torch.Tensor:
        """Embed the ``<text_eos><audio_bos>`` tail every condition ends with."""
        return torch.cat([self._text_embedding_row(self._text_eos_id), self._text_embedding_row(self._tts_bos_id)])

    def _build_condition_embeddings(
        self,
        tts_token_ids: torch.Tensor,
        tts_hidden_states: torch.Tensor,
        *,
        native_duplex: bool = False,
    ) -> torch.Tensor:
        if tts_token_ids.numel() == 0 or tts_hidden_states.numel() == 0:
            # The thinker can legally emit an empty speech segment (<|tts_bos|>
            # immediately followed by a boundary token) when it decides not to
            # speak. Condition on the boundary tokens alone, which matches the
            # 2-token scheduler prompt the stage bridge builds for an empty
            # handoff.
            return self._boundary_embeddings()
        device = self.emb_text.weight.device
        dtype = self.emb_text.weight.dtype
        # Pinned, non-blocking H2D: a pageable copy would stall every row in the step.
        token_ids = to_device_nonblocking(tts_token_ids, device).to(dtype=torch.long).reshape(-1)
        hidden = to_device_nonblocking(tts_hidden_states, device).to(dtype=dtype)
        if hidden.shape[0] != token_ids.shape[0] and token_ids.shape[0] != 1:
            raise ValueError(
                "MiniCPM-o Talker condition length mismatch: "
                f"token_ids={token_ids.shape[0]} hidden_states={hidden.shape[0]}"
            )
        text_embeds = self.emb_text(token_ids)
        hidden_embeds = self.projector_semantic(hidden)
        if self._normalize:
            hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)
        audio_bos = self._text_embedding_row(self._tts_bos_id)
        condition = text_embeds + hidden_embeds
        if native_duplex:
            # Match MiniCPMTTS.generate_chunk's streaming condition.
            return torch.cat([condition, audio_bos], dim=0)
        return torch.cat([condition, self._boundary_embeddings()], dim=0)

    def _build_streaming_recompute_embeddings(
        self,
        current_condition: torch.Tensor,
        *,
        request_id: str,
        info_dict: Mapping[str, Any],
        meta: Mapping[str, Any],
    ) -> torch.Tensor:
        """Return the official one-previous-chunk sliding-recompute window."""
        condition_seq = meta.get("streaming_condition_seq")
        if not isinstance(condition_seq, int) or isinstance(condition_seq, bool):
            if meta.get("streaming_prompt_recompute") is True:
                raise ValueError("streaming prompt recompute is missing streaming_condition_seq")
            # Direct model tests and non-connector callers do not participate in
            # the persistent async-chunk lifecycle, so they need no window state.
            return current_condition

        turn_start = bool(meta.get("turn_start"))
        recompute = meta.get("streaming_prompt_recompute") is True
        states = self._request_condition_states
        state = states.get(request_id)
        if turn_start:
            if recompute:
                raise ValueError("streaming prompt recompute cannot cross a native duplex turn boundary")
            states[request_id] = {
                "condition_seq": condition_seq,
                "condition": current_condition.detach().clone(),
                "base_recent_codes": (),
            }
            return current_condition

        if state is None:
            if recompute:
                raise ValueError("streaming prompt recompute is missing the previous Talker condition")
            states[request_id] = {
                "condition_seq": condition_seq,
                "condition": current_condition.detach().clone(),
                "base_recent_codes": (),
            }
            return current_condition

        previous_seq = state.get("condition_seq")
        if not isinstance(previous_seq, int) or condition_seq < previous_seq:
            raise ValueError(
                f"stale native duplex Talker condition sequence: current={condition_seq}, previous={previous_seq}"
            )
        if condition_seq > previous_seq + 1:
            raise ValueError(
                f"native duplex Talker skipped a condition sequence: current={condition_seq}, previous={previous_seq}"
            )

        if not recompute:
            if condition_seq > previous_seq:
                attention_type = getattr(self._tts_config, "attention_type", "full_attention")
                if attention_type == "sliding_recompute":
                    raise ValueError(
                        "a native duplex Talker condition advanced without its streaming recompute marker: "
                        f"current={condition_seq}, previous={previous_seq}"
                    )
                audio_state = self._request_audio_states.get(request_id)
                recent_codes = audio_state.get("recent_codes") if isinstance(audio_state, dict) else None
                if isinstance(recent_codes, torch.Tensor):
                    base_recent_codes = recent_codes.detach().clone()
                elif isinstance(recent_codes, list):
                    base_recent_codes = tuple(int(code_id) for code_id in recent_codes[-_CODEC_PENALTY_WINDOW:])
                else:
                    base_recent_codes = state.get("base_recent_codes")
                    if not isinstance(base_recent_codes, (tuple, torch.Tensor)):
                        raise ValueError("streaming Talker condition lost its frozen codec history")
                states[request_id] = {
                    "condition_seq": condition_seq,
                    "condition": current_condition.detach().clone(),
                    "base_recent_codes": base_recent_codes,
                }
                return current_condition
            if "active_embeddings" in state:
                raise ValueError("an active streaming recompute was replayed without its recompute marker")
            return current_condition

        if condition_seq == previous_seq:
            active_embeddings = state.get("active_embeddings")
            if not isinstance(active_embeddings, torch.Tensor):
                raise ValueError("streaming prompt window lost its cached recompute embeddings")
            return active_embeddings

        if condition_seq != previous_seq + 1:
            raise ValueError(
                "streaming prompt recompute skipped a Talker condition: "
                f"previous={previous_seq}, current={condition_seq}"
            )

        previous_condition = state.get("condition")
        if not isinstance(previous_condition, torch.Tensor):
            raise ValueError("streaming prompt recompute lost the previous Talker condition")

        ids = info_dict.get("ids")
        previous_codes = ids.get("streaming_prompt_previous_codes") if isinstance(ids, Mapping) else None
        if isinstance(previous_codes, torch.Tensor):
            code_ids = previous_codes.to(device=self.emb_code[0].weight.device, dtype=torch.long).reshape(-1)
        elif isinstance(previous_codes, (list, tuple)):
            code_ids = torch.as_tensor(previous_codes, device=self.emb_code[0].weight.device, dtype=torch.long)
        else:
            raise ValueError("streaming prompt recompute is missing confirmed codec ids")
        if code_ids.numel() > _DUPLEX_TURN_END_CODEC_TOKENS - 1:
            raise ValueError(f"streaming prompt recompute has too many codec ids: {code_ids.numel()}")
        if code_ids.numel() and bool(((code_ids < 0) | (code_ids >= self._codec_eos_id)).any()):
            raise ValueError("streaming prompt recompute codec ids include an invalid or terminal token")

        parts = [previous_condition]
        if code_ids.numel():
            parts.append(self.emb_code[0](code_ids))
        parts.append(current_condition)
        full_embeddings = torch.cat(parts, dim=0)
        previous_base_codes = state.get("base_recent_codes")
        if not isinstance(previous_base_codes, (tuple, torch.Tensor)):
            raise ValueError("streaming Talker condition lost its frozen codec history")
        states[request_id] = {
            "condition_seq": condition_seq,
            "condition": current_condition.detach().clone(),
            "active_embeddings": full_embeddings.detach().clone(),
            # Official generate_with_buffer keeps all_generated_tokens across
            # sliding recomputes, so the first sample in this chunk still sees
            # the previous chunk's repetition-penalty window.
            "base_recent_codes": (
                torch.cat([previous_base_codes, code_ids])[-_CODEC_PENALTY_WINDOW:]
                if isinstance(previous_base_codes, torch.Tensor)
                else (*previous_base_codes, *(int(code_id) for code_id in code_ids.tolist()))[-_CODEC_PENALTY_WINDOW:]
            ),
        }
        return full_embeddings

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build request-local prefill/decode embeddings for the vLLM runner."""
        del input_embeds
        # A FULL CUDA graph replay never calls forward(), so release finished
        # requests here too; this runs on every step before the graph.
        if getattr(self, "_deferred_cleanup_ids", None):
            self._flush_deferred_cleanup()
        span_len = int(input_ids.shape[0])
        is_prefill = bool(info_dict.get("_omni_is_prefill", False))
        state = info_dict.get("audio_state")
        first_call = not isinstance(state, dict)
        # V1 preprocess uses request_id; MRv2's slot buffer uses req_id.
        # Streaming conditions and codec history belong to that request,
        # including when other sessions prefill in the same batch.
        request_id = str(info_dict.get("request_id") or info_dict.get("req_id", "0"))

        if is_prefill or first_call:
            token_ids, hidden_states = get_tts_handoff(info_dict)
            # get_tts_handoff unpacks the bridge's float32 bytes; legacy
            # callers may still hand over nested lists. Normalize both
            # before validating/building the Talker condition.
            if isinstance(token_ids, (list, tuple)):
                token_ids = torch.as_tensor(token_ids, dtype=torch.long)
            if isinstance(hidden_states, (list, tuple)):
                hidden_states = torch.as_tensor(hidden_states, dtype=torch.float32)
            if not isinstance(token_ids, torch.Tensor) or not isinstance(hidden_states, torch.Tensor):
                available = sorted(key for key in info_dict if not key.startswith("_"))
                raise ValueError(
                    "MiniCPM-o Talker requires tensor tts_token_ids and "
                    "tts_hidden_states conditioning; "
                    f"received token_ids={type(token_ids).__name__}, "
                    f"hidden_states={type(hidden_states).__name__}, "
                    f"available_keys={available}"
                )
            # An empty condition means the thinker chose not to speak: finish the
            # request up front so it emits zero audio codes instead of killing
            # the stage engine.
            empty_condition = token_ids.numel() == 0 or hidden_states.numel() == 0
            if empty_condition:
                logger.warning_once(
                    "MiniCPM-o Talker received an empty condition (request %s); this request produces no audio.",
                    info_dict.get("request_id"),
                )
            native_duplex = bool(info_dict.get("native_duplex", False))
            meta = info_dict.get("meta")
            full_embeds = self._build_condition_embeddings(
                token_ids,
                hidden_states,
                native_duplex=native_duplex,
            )
            if native_duplex:
                full_embeds = self._build_streaming_recompute_embeddings(
                    full_embeds,
                    request_id=request_id,
                    info_dict=info_dict,
                    meta=meta if isinstance(meta, Mapping) else {},
                )
            retained_codes: list[int] = []
            condition_seq = meta.get("streaming_condition_seq") if isinstance(meta, Mapping) else None
            if native_duplex and isinstance(condition_seq, int) and not isinstance(condition_seq, bool):
                condition_state = self._request_condition_states.get(request_id)
                base_recent_codes = (
                    condition_state.get("base_recent_codes") if isinstance(condition_state, dict) else None
                )
                if not isinstance(base_recent_codes, (tuple, torch.Tensor)):
                    raise ValueError("streaming Talker condition lost its frozen codec history")
                retained_codes = (
                    base_recent_codes if isinstance(base_recent_codes, torch.Tensor) else list(base_recent_codes)
                )
            offset = int(info_dict.get("_omni_num_computed_tokens", 0))
            # The handoff rebuilds only the tail-aligned Talker condition.
            # Materialize zero-token embeddings for any scheduler prompt
            # prefix so chunked prefill can slice from a non-zero offset.
            prompt_len = info_dict.get("_omni_prompt_len")
            target_len = int(prompt_len) if prompt_len is not None else offset + span_len
            if native_duplex and isinstance(meta, Mapping) and meta.get("streaming_prompt_recompute") is True:
                if target_len != full_embeds.shape[0]:
                    raise ValueError(
                        "streaming prompt recompute length mismatch: "
                        f"scheduler={target_len}, model={full_embeds.shape[0]}"
                    )
            prefix_len = target_len - full_embeds.shape[0]
            if prefix_len > 0:
                placeholder_ids = torch.zeros(
                    prefix_len,
                    dtype=torch.long,
                    device=self.emb_text.weight.device,
                )
                full_embeds = torch.cat([self.emb_text(placeholder_ids), full_embeds], dim=0)
            embeds = full_embeds[offset : offset + span_len]
            if embeds.shape[0] != span_len:
                raise ValueError(
                    "MiniCPM-o Talker prefill span exceeds condition: "
                    f"request_id={info_dict.get('request_id')} offset={offset} "
                    f"span={span_len} condition={full_embeds.shape[0]} "
                    f"tts_ids={token_ids.shape[0]} tts_hidden={hidden_states.shape[0]} "
                    f"prompt_len={info_dict.get('_omni_prompt_len')}"
                )
            if native_duplex:
                max_tokens, min_tokens = _native_duplex_chunk_budget(meta if isinstance(meta, Mapping) else None)
            else:
                # MiniCPMTTS.generate()'s max_new_token, clamped to what the
                # Talker context can still hold. Sampler min_tokens (upstream's
                # min_new_token=50) comes from the deploy YAML.
                remaining = int(self._tts_config.max_position_embeddings) - target_len
                max_tokens = max(min(_OFFLINE_CODEC_MAX_NEW_TOKENS, remaining), 1)
                min_tokens = None
            state: dict[str, Any] = {
                "finished": empty_condition,
                "step": 0,
                "max_tokens": max_tokens,
                "min_tokens": min_tokens,
                "turn_end_drain": bool(native_duplex and isinstance(meta, Mapping) and bool(meta.get("turn_end"))),
            }
            if isinstance(retained_codes, torch.Tensor) or retained_codes:
                state["recent_codes"] = retained_codes
            request_states = getattr(self, "_request_audio_states", None)
            if request_states is None:
                request_states = {}
                self._request_audio_states = request_states
            request_states[request_id] = state
            empty_codes = torch.empty(0, dtype=torch.long, device="cpu")
            return (
                input_ids,
                embeds,
                {
                    "audio_state": state,
                    # Prefill has no previous codec id. vLLM samples the first
                    # code after this forward; the next decode emits it.
                    "codes": {"audio": empty_codes},
                },
            )

        stored = self._request_audio_states.get(request_id)
        if isinstance(stored, dict):
            state = stored
        if input_ids.device.type in ("cuda", "npu") and isinstance(state, dict) and "_gpu_slot" in state:
            # A scalar fallback after batched decode must use the same device
            # state, rather than reviving the stale host EOS/history fields.
            row_info = dict(info_dict, audio_state=state, request_id=request_id)
            _, embeds, updates = self.preprocess_decode_batch(input_ids=input_ids[-1:], req_infos=[row_info])
            return input_ids, embeds, updates[0]
        if isinstance(state, dict) and state.get("finished"):
            # An empty speech segment can still be scheduled until EOS is
            # eligible. The sampler is forced to EOS; any shape-correct
            # embedding is enough for these leftover decode rows.
            weight = self.emb_code[0].weight
            empty_codes = torch.empty(0, dtype=torch.long, device="cpu")
            return input_ids, weight.new_zeros((span_len, weight.shape[1])), {"codes": {"audio": empty_codes}}

        # Decode: vLLM's previous sampled codec id is this step's input.
        # Embed it with the codec table (not the 32k Llama embed_tokens) and
        # hand the same id to make_omni_output so Code2Wav sees it this step.
        # The runner sends one-token decode rows through preprocess_decode_batch,
        # which keeps EOS detection and penalty state on the device.
        code = input_ids.to(device=self.emb_code[0].weight.device, dtype=torch.long).reshape(-1)[-1:]
        embeds = self.emb_code[0](code)
        code_id = int(code.item())
        if code_id == int(self._codec_eos_id):
            if isinstance(state, dict):
                state["finished"] = True
            elif stored is None:
                self._request_audio_states[request_id] = {"finished": True, "step": 0}
            delta = torch.empty(0, dtype=torch.long, device="cpu")
        else:
            # The runner and connector consume CPU IDs; reuse the scalar
            # already read for EOS instead of copying the same token again.
            delta = torch.tensor([[code_id]], dtype=torch.long, device="cpu")
        return input_ids, embeds, {"codes": {"audio": delta}}

    def _decode_codec_id_map(self) -> dict[str, tuple[torch.Tensor, int]]:
        pending = getattr(self, "_decode_codec_ids", None)
        if pending is None:
            pending = {}
            self._decode_codec_ids = pending
        return pending

    def preprocess_decode_batch(
        self,
        *,
        input_ids: torch.Tensor,
        req_infos: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        """Embed every one-token decode row at once, without reading ids on the host.

        CUDA rows retain an owned device snapshot. ``make_omni_output`` updates
        EOS, frame counts and penalty history on the device; the runner copies
        codec payloads to the host together with its asynchronous output.
        CPU rows retain the scalar implementation's semantics.
        """
        num_rows = len(req_infos)
        if input_ids.numel() != num_rows:
            raise ValueError(f"MiniCPM-o Talker batched decode needs one id per row: {input_ids.numel()} != {num_rows}")
        weight = self.emb_code[0].weight
        codes = input_ids.reshape(-1).to(device=weight.device, dtype=torch.long)
        embeds = self.emb_code[0](codes)
        # The runner and connector consume CPU codec ids.
        empty_codes = torch.empty(0, dtype=torch.long, device="cpu")
        request_states = self._request_audio_states
        updates: list[dict[str, Any]] = []
        live_rows: list[tuple[int, str]] = []
        zero_rows: list[int] = []
        for row, info in enumerate(req_infos):
            request_id = str(info.get("request_id", "0"))
            state = info.get("audio_state")
            if info.get("_omni_is_prefill", False) or not isinstance(state, dict):
                # A row without Talker state builds its condition like a prefill.
                _, row_embeds, update = self.preprocess(input_ids=input_ids[row : row + 1], input_embeds=None, **info)
                embeds[row : row + 1] = row_embeds
                updates.append(update)
                continue
            stored = request_states.get(request_id)
            if isinstance(stored, dict):
                state = stored
            if state.get("finished"):
                # Leftover decode of a finished request: shape-correct zeros.
                zero_rows.append(row)
                updates.append({"codes": {"audio": empty_codes}})
                continue
            live_rows.append((row, request_id))
            # Output construction owns the codec snapshot. An empty buffer
            # update avoids retaining the preceding step's delta.
            updates.append({"codes": {"audio": empty_codes}})
        if zero_rows:
            embeds.index_fill_(0, index_to_device(zero_rows, embeds.device), 0.0)
        codec_state = getattr(self, "_device_codec_state", None)
        if codec_state is not None and codes.device.type in ("cuda", "npu"):
            slots = [request_states.get(request_id, {}).get("_gpu_slot", -1) for _, request_id in live_rows]
            if slots and all(slot >= 0 for slot in slots):
                codec_state.mask_embeddings(embeds, slots, [row for row, _ in live_rows])
            elif slots:
                initialized = [(row, slot) for (row, _), slot in zip(live_rows, slots) if slot >= 0]
                if initialized:
                    codec_state.mask_embeddings(
                        embeds, [slot for _, slot in initialized], [row for row, _ in initialized]
                    )
        if live_rows:
            owned_ids = codes.detach().clone()
            pending = self._decode_codec_id_map()
            for row, request_id in live_rows:
                pending[request_id] = (owned_ids, row)
        return input_ids, embeds, updates

    # ------------------------------------------------------------------
    # Model Runner V2
    # ------------------------------------------------------------------
    #
    # Under MRv2 a decode row's input id is the previous step's sample, which
    # the runner gathers on the device (``last_sampled_tokens``) and embeds
    # into its static input buffer with ``embed_input_ids`` (``emb_code``).
    # That is the whole decode preprocess, so decode rows skip the per-row
    # hook. Like V1's batched CUDA path, codec deltas, EOS detection, frame
    # counts and penalty histories stay on the device. MRv2 derives these
    # directly from runner state rather than the model-owned V1 codec slots.

    #: Decode rows need no per-row preprocess beyond the runner's embedding.
    mrv2_decode_preprocess_is_identity = True

    @property
    def logits_vocab_size(self) -> int:
        return int(self._num_audio_tokens)

    def preprocess_decode_batch_mrv2(
        self,
        *,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        req_infos: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        """Decode rows are ``emb_code`` of their input id, already in ``input_embeds``."""
        if getattr(self, "_deferred_cleanup_ids", None):
            self._flush_deferred_cleanup()
        num_rows = len(req_infos)
        empty = input_embeds.new_empty((num_rows, 0))
        return input_ids, input_embeds, empty, empty, [{} for _ in range(num_rows)]

    def mrv2_custom_sampler(self, sampler: Any) -> tuple[Any, None]:
        return _install_mrv2_talker_sampler(sampler, self), None

    def capture_auxiliary_graphs(self) -> None:
        """Warm the CUDA seeded sampler before the worker reports readiness."""
        self._warmup_mrv2_decode_output()
        sampler = getattr(self, "_mrv2_seeded_codec_sampler", None)
        if sampler is not None:
            sampler.warmup()

    @torch.inference_mode()
    def _warmup_mrv2_decode_output(self) -> None:
        """Compile the fused decode-output kernel (both layouts) on disposable inputs."""
        empty_speech = getattr(self, "_mrv2_empty_speech", None)
        tts_config = getattr(self, "_tts_config", None)
        if (
            empty_speech is None
            or tts_config is None
            or empty_speech.device.type != "cuda"
            or not current_omni_platform.is_cuda()
        ):
            return
        device = empty_speech.device
        slots = empty_speech.shape[0]
        ids = torch.zeros(1, dtype=torch.int32, device=device)
        lengths = torch.zeros(slots, dtype=torch.int32, device=device)
        controls = torch.full((slots, 3), -1, dtype=torch.long, device=device)
        for native_controls in (None, controls):
            _mrv2_decode_output_gpu(
                ids,
                ids,
                ids,
                lengths,
                torch.zeros_like(empty_speech),
                native_controls,
                num_reqs=1,
                eos_id=int(self._codec_eos_id),
                context_limit=int(tts_config.max_position_embeddings) - 1,
            )

    def make_omni_output_mrv2(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        *,
        input_batch: Any,
        req_states: Any,
        model_intermediate_buffer: list[dict[str, Any]],
    ) -> OmniOutput:
        """Device-side ``make_omni_output`` for Model Runner V2.

        Emits every token row's input id as ``codes.audio`` plus
        ``meta.codec_frame_valid``: true only for a decode row whose input is a
        codec id of a request that has not ended. That id is the code V1 hands
        to Code2Wav on this step; the stage payload builder drops the other
        rows on the host, after the asynchronous output copy. Rows the model
        terminates this step (empty condition, codec EOS already sampled, or
        the offline length cap) are recorded for the sampler.
        """
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        hidden = model_outputs
        empty_speech = self._mrv2_empty_speech
        if empty_speech is None:
            raise RuntimeError("MiniCPM-o Talker MRv2 output requires its MRv2 sampler (mrv2_custom_sampler)")
        num_reqs = int(input_batch.num_reqs)
        device = hidden.device
        self._mrv2_output_infos = model_intermediate_buffer
        native = any(info.get("native_duplex") is True for info in model_intermediate_buffer)
        if native and getattr(self, "_mrv2_codec_controls", None) is None:
            self._mrv2_codec_controls = torch.full((empty_speech.shape[0], 3), -1, dtype=torch.long, device=device)
            self._mrv2_metadata_by_slot = {}
        if input_batch.has_prefill:
            # Prefill rows only: record whether the Thinker handed over an
            # empty condition (preprocess marks such a request finished).
            slots: list[int] = []
            flags: list[bool] = []
            controls, histories = [], []
            for row in np.flatnonzero(input_batch.is_prefilling_np[:num_reqs]).tolist():
                info = model_intermediate_buffer[row]
                state = info.get("audio_state")
                flags.append(bool(state.get("finished")) if isinstance(state, dict) else False)
                slots.append(int(input_batch.idx_mapping_np[row]))
                if native:
                    if info.get("native_duplex") is True:
                        controls.append([state["max_tokens"] - 1, state["min_tokens"], int(state["turn_end_drain"])])
                        history = list(state.get("recent_codes", ()))[-_CODEC_PENALTY_WINDOW:]
                        histories.append([-1] * (_CODEC_PENALTY_WINDOW - len(history)) + history)
                    else:
                        controls.append([-1, -1, -1])
                        histories.append([-1] * _CODEC_PENALTY_WINDOW)
                    self._mrv2_metadata_by_slot[slots[-1]] = self._duplex_output_metadata([info])
            if slots:
                slot_tensor = index_to_device(slots, device)
                empty_speech.index_copy_(
                    0,
                    slot_tensor,
                    index_to_device(flags, device, dtype=torch.bool),
                )
                if native:
                    self._mrv2_codec_controls.index_copy_(
                        0, slot_tensor, to_device_nonblocking(torch.tensor(controls, dtype=torch.long), device)
                    )
                    self._mrv2_penalties.prefix_history.index_copy_(
                        0, slot_tensor, to_device_nonblocking(torch.tensor(histories, dtype=torch.long), device)
                    )
        num_tokens = int(hidden.shape[0])
        token_ids = input_batch.input_ids[:num_tokens]
        # index_select takes the runner's int32 indices directly, and the step
        # arithmetic stays exact in int32: no per-step widening copies.
        slot_ids = input_batch.idx_mapping[:num_reqs]
        one_token_rows = int(input_batch.query_start_loc_np[num_reqs]) == num_reqs
        if (
            one_token_rows
            and not input_batch.has_prefill
            and num_reqs > 0
            and device.type == "cuda"
            and current_omni_platform.is_cuda()
        ):
            # Steady decode: the ~25 elementwise launches below in one kernel.
            codes, frame_valid, forced, mask = _mrv2_decode_output_gpu(
                token_ids,
                slot_ids,
                input_batch.seq_lens[:num_reqs],
                req_states.prompt_len.gpu,
                empty_speech,
                self._mrv2_codec_controls if native else None,
                num_reqs=num_reqs,
                eos_id=int(self._codec_eos_id),
                context_limit=int(self._tts_config.max_position_embeddings) - 1,
            )
            self._mrv2_mask_eos = mask
            self._mrv2_forced_eos = forced
            return self._mrv2_omni_output(
                hidden, codes, frame_valid, forced, native, input_batch, model_intermediate_buffer
            )
        prompt_len = req_states.prompt_len.gpu.index_select(0, slot_ids)
        # Codes emitted before this step's input (V1's ``state["step"]``).
        step = input_batch.seq_lens[:num_reqs] - prompt_len
        # Every row schedules one token (plain decode): row i's last token is
        # token i, so its logits index needs no gather.
        if one_token_rows:
            last_rows = None
            last_ids = token_ids[:num_reqs]
        else:
            last_rows = input_batch.logits_indices[:num_reqs].long()
            last_ids = token_ids.index_select(0, last_rows)
        empty = empty_speech.index_select(0, slot_ids)
        decode = step > 0
        eos_input = decode & (last_ids == int(self._codec_eos_id))
        ended = eos_input | empty
        valid = decode & ~ended
        # MiniCPMTTS.generate's max_new_token, clamped to the Talker context
        # (see ``preprocess``); the sample after the last allowed code is EOS.
        # clamp(context - prompt, 1, max) - 1 == clamp(context - 1 - prompt, 0, max - 1).
        limit = torch.clamp(
            (int(self._tts_config.max_position_embeddings) - 1) - prompt_len,
            min=0,
            max=_OFFLINE_CODEC_MAX_NEW_TOKENS - 1,
        )
        self._mrv2_mask_eos = None
        if native:
            controls = self._mrv2_codec_controls.index_select(0, slot_ids)
            limit = torch.where(controls[:, 0] >= 0, controls[:, 0], limit)
            boundary = (step >= _DUPLEX_CODEC_FRAMES_PER_CHUNK) & (
                step.remainder(_DUPLEX_CODEC_FRAMES_PER_CHUNK) < _DUPLEX_TURN_END_BOUNDARY_MASK_STEPS
            )
            self._mrv2_mask_eos = (step < controls[:, 1]) | ((controls[:, 2] == 1) & boundary)
        self._mrv2_forced_eos = ended | (step >= limit)
        if native:
            self._mrv2_mask_eos &= ~self._mrv2_forced_eos
        if one_token_rows and num_tokens == num_reqs:
            # ``valid`` is this step's own tensor and is never written again.
            frame_valid = valid
        else:
            frame_valid = torch.zeros(num_tokens, dtype=torch.bool, device=device)
            if last_rows is None:
                frame_valid[:num_reqs].copy_(valid)
            else:
                frame_valid.index_copy_(0, last_rows, valid)
        # copy=True: the payload never aliases the runner's input buffer.
        codes = token_ids.to(dtype=torch.long, copy=True).reshape(num_tokens, 1)
        return self._mrv2_omni_output(
            hidden, codes, frame_valid, self._mrv2_forced_eos, native, input_batch, model_intermediate_buffer
        )

    def _mrv2_omni_output(
        self,
        hidden: torch.Tensor,
        codes: torch.Tensor,
        frame_valid: torch.Tensor,
        forced: torch.Tensor,
        native: bool,
        input_batch: Any,
        model_intermediate_buffer: list[dict[str, Any]],
    ) -> OmniOutput:
        if not self._mrv2_decode_rows_logged and not input_batch.has_prefill:
            self._mrv2_decode_rows_logged = True
            logger.info("MiniCPM-o Talker: MRv2 device-side codec output active (no host read of sampled ids)")
        meta = {"codec_frame_valid": frame_valid}
        self._mrv2_output_meta = None
        if native:
            # One device tensor (one D2H copy); the finalizer restores the
            # per-request rows on the host. The per-slot metadata never
            # enters the device payload: capture this step's tensors here.
            meta["finished"] = forced
            slot_meta = [self._mrv2_metadata_by_slot[int(slot)] for slot in input_batch.idx_mapping_np]
            self._mrv2_output_meta = (
                model_intermediate_buffer,
                {key: [entry[key][0] for entry in slot_meta] for key in _DUPLEX_OUTPUT_META_KEYS},
            )
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={"codes": {"audio": codes}, "meta": meta},
        )

    def mrv2_decode_burst(self, input_batch: InputBatch, req_states: RequestState) -> "_TalkerDecodeBurst | None":
        """Plan a decode burst (worker_v2.decode_burst) for an all-native-duplex decode batch.

        Called after the scheduled step sampled. The burst ends at the first
        step whose draw is a forced codec EOS for every row (the chunk
        ceiling) and stays within the request slots the scheduler reserved.
        """
        steps = self._mrv2_decode_burst_steps
        sampler = getattr(self, "_mrv2_seeded_codec_sampler", None)
        infos = getattr(self, "_mrv2_output_infos", ())
        num_reqs = int(input_batch.num_reqs)
        if (
            steps <= 1
            or sampler is None
            or sampler._rewind_async_lookahead
            or len(infos) != num_reqs
            or not all(info.get("native_duplex") is True for info in infos)
        ):
            return None
        num_computed = input_batch.num_computed_tokens_np[:num_reqs]
        # This step forwards the request's ``step``-th codec frame (the
        # device's ``seq_len - prompt_len``); the draw at
        # ``step == max_tokens - 1`` is the chunk's forced codec EOS.
        frame = num_computed + 1 - req_states.prompt_len.np[input_batch.idx_mapping_np[:num_reqs]]
        remaining = 0
        for row, info in enumerate(infos):
            state = self._request_audio_states.get(str(info.get("request_id") or info["req_id"]))
            if not isinstance(state, dict) or state.get("finished") or not isinstance(state.get("max_tokens"), int):
                return None
            remaining = max(remaining, int(state["max_tokens"]) - int(frame[row]))
        # Lookahead slots end at max_model_len.
        room = int(self.vllm_config.model_config.max_model_len) - int(num_computed.max())
        steps = min(steps, remaining, room)
        if steps <= 1:
            return None
        generators = [sampler._generators.get(request_id) for request_id in input_batch.req_ids[:num_reqs]]
        return _TalkerDecodeBurst(self, steps, generators)

    def take_mrv2_forced_eos(self, input_batch: Any, req_states: Any, num_rows: int) -> torch.Tensor | None:
        """This step's forced-EOS rows, once; ``None`` outside a model step (warmup)."""
        forced, self._mrv2_forced_eos = self._mrv2_forced_eos, None
        if forced is None or int(forced.shape[0]) != int(num_rows):
            return None
        return forced

    def mrv2_codec_history_finalizer(
        self, input_batch: InputBatch, infos: list[dict[str, Any]]
    ) -> Callable[[dict[str, Any], list[int]], dict[str, Any] | RequestOutputSnapshot]:
        """Update cross-condition history from the runner's existing CPU copy.

        GPU penalties/EOS do not read this history in decode. Capture state
        identities so delayed output cannot update a new condition or session.

        For native duplex steps it also restores the per-request output
        metadata ``make_omni_output_mrv2`` kept off the device payload, and
        returns the request partition the runner would otherwise build
        recursively (same keys, tensors and ownership).
        """
        stashed = getattr(self, "_mrv2_output_meta", None)
        self._mrv2_output_meta = None
        self._mrv2_last_output_meta = stashed
        return self._codec_history_finalizer(input_batch, infos, stashed)

    def _codec_history_finalizer(
        self,
        input_batch: InputBatch | BurstOutputBatch,
        infos: list[dict[str, Any]],
        stashed: tuple[list[dict[str, Any]], dict[str, list[torch.Tensor]]] | None,
    ) -> Callable[[dict[str, Any], list[int]], dict[str, Any] | RequestOutputSnapshot]:
        sampler = getattr(self, "_mrv2_seeded_codec_sampler", None)
        track_rng = sampler is not None and sampler._rewind_async_lookahead
        num_rows = len(infos)
        starts = np.asarray(input_batch.query_start_loc_np[:num_rows]).tolist()
        counts = np.asarray(input_batch.num_scheduled_tokens[:num_rows]).tolist()
        audio_states = self._request_audio_states
        entries = []
        for i, info in enumerate(infos):
            request_id = str(info.get("request_id") or info["req_id"])
            entries.append(
                (
                    starts[i],
                    counts[i],
                    request_id,
                    # Intermediate-buffer updates merge dict fields into a new
                    # container. Capture the model-owned state, whose identity
                    # fences delayed output against a successor condition.
                    audio_states.get(request_id),
                    bool(input_batch.is_prefilling_np[i]) if track_rng else False,
                    sampler.codec_rng_checkpoint(request_id) if track_rng else None,
                )
            )
        if stashed is not None and stashed[0] is not infos:
            # make_omni_output_mrv2 and this finalizer must describe one step;
            # without its metadata the payload cannot be routed per request.
            raise RuntimeError("MiniCPM-o Talker MRv2 output metadata belongs to a different batch")
        output_meta = stashed[1] if stashed is not None else None
        token_axis_sizes: set[int] = set()
        if output_meta is not None and num_rows == int(input_batch.num_reqs):
            # The token-axis sizes OmniAsyncOutput slices request payloads by.
            query_start_loc = input_batch.query_start_loc_np
            if query_start_loc.shape[0] > num_rows:
                total_tokens = int(query_start_loc[num_rows])
            else:
                total_tokens = int(np.asarray(input_batch.num_scheduled_tokens[:num_rows], dtype=np.int64).sum())
            token_axis_sizes = {total_tokens, int(getattr(input_batch, "num_tokens_after_padding", total_tokens))}
        eos_id = int(self._codec_eos_id)

        def finalize(outputs: dict[str, Any], num_sampled: list[int]) -> dict[str, Any] | RequestOutputSnapshot:
            payload = flatten_payload(outputs)
            audio, valid = payload.get("codes.audio"), payload.get("meta.codec_frame_valid")
            if not isinstance(audio, torch.Tensor) or not isinstance(valid, torch.Tensor):
                raise RuntimeError("MRv2 Talker output lost its codec rows or validity")
            # One codec id per token row: read all rows once, not per request.
            per_token = (
                audio.dim() > 0 and audio.numel() == audio.shape[0] == valid.numel() and valid.dtype == torch.bool
            )
            audio_ids: list[int] | None = None
            valid_rows: list[bool] = []
            for i, (start, count, request_id, state, prefill, checkpoint) in enumerate(entries):
                if not num_sampled[i] or not isinstance(state, dict):
                    continue
                if self._request_audio_states.get(request_id) is not state:
                    continue
                if per_token:
                    if audio_ids is None:
                        audio_ids = audio.reshape(-1).tolist()
                        valid_rows = valid.reshape(-1).tolist()
                    row_ids = audio_ids[start : start + count]
                    if checkpoint is not None and (prefill or eos_id not in row_ids):
                        sampler.commit_codec_rng(request_id, checkpoint)
                    codes = [code for code, keep in zip(row_ids, valid_rows[start : start + count]) if keep]
                else:
                    row_audio = audio[start : start + count].reshape(-1)
                    if checkpoint is not None and (prefill or eos_id not in row_audio.tolist()):
                        sampler.commit_codec_rng(request_id, checkpoint)
                    codes = row_audio[valid[start : start + count].reshape(-1)].tolist()
                if codes:
                    state["recent_codes"] = (list(state.get("recent_codes", ())) + codes)[-_CODEC_PENALTY_WINDOW:]
            if output_meta is None:
                return outputs
            return _mrv2_duplex_output_partition(outputs, output_meta, entries, token_axis_sizes)

        return finalize

    @staticmethod
    def _duplex_output_metadata(infos: list[dict[str, Any]]) -> dict[str, list[torch.Tensor]]:
        result = {key: [] for key in _DUPLEX_OUTPUT_META_KEYS}
        for info in infos:
            info = info if isinstance(info, dict) else {}
            native = info.get("native_duplex") is True
            duplex_info = info.get("duplex")
            if not isinstance(duplex_info, dict):
                duplex_info = {}
            epoch = duplex_info.get("epoch", -1)
            turn_id = duplex_info.get("turn_id", -1)
            if native and not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in (epoch, turn_id)
            ):
                raise RuntimeError(
                    "MiniCPM-o native duplex Talker requires non-negative integer "
                    f"epoch and turn_id, got epoch={epoch!r}, turn_id={turn_id!r}"
                )
            meta_info = info.get("meta")
            if not isinstance(meta_info, dict):
                meta_info = {}
            segment_text = meta_info.get("native_duplex_segment_text", "") if native else ""
            if not isinstance(segment_text, str):
                segment_text = ""
            turn_eos_id = meta_info.get("turn_eos_token_id")
            ids_info = info.get("ids")
            tts_ids = ids_info.get("tts") if native and isinstance(ids_info, dict) else None
            if isinstance(tts_ids, torch.Tensor):
                contains_turn_eos = isinstance(turn_eos_id, int) and bool(
                    torch.any(tts_ids.reshape(-1) == turn_eos_id).item()
                )
            elif isinstance(tts_ids, (list, tuple)):
                contains_turn_eos = isinstance(turn_eos_id, int) and turn_eos_id in tts_ids
            else:
                contains_turn_eos = False
            result["native_duplex"].append(torch.tensor(native, dtype=torch.bool))
            result["duplex_epoch"].append(torch.tensor(epoch if isinstance(epoch, int) else -1, dtype=torch.long))
            result["duplex_turn_id"].append(torch.tensor(turn_id if isinstance(turn_id, int) else -1, dtype=torch.long))
            result["llm_output_text_utf8"].append(
                torch.tensor(
                    list(segment_text.encode("utf-8")),
                    dtype=torch.uint8,
                )
            )
            result["turn_end"].append(torch.tensor(native and contains_turn_eos, dtype=torch.bool))
        return result

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        **kwargs: Any,
    ) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        hidden = model_outputs
        infos = kwargs.get("model_intermediate_buffer") or []
        spans = kwargs.get("request_token_spans")
        if spans is None or len(spans) != len(infos):
            raise RuntimeError("MiniCPM-o continuous Talker requires one request_token_span per request")
        emit_duplex_metadata = any(isinstance(info, dict) and info.get("native_duplex") is True for info in infos)

        empty_delta = torch.empty((0, 1), dtype=torch.long, device="cpu")
        codec_deltas = [empty_delta for _ in infos]
        terminal_flags = [torch.tensor(False, dtype=torch.bool) for _ in infos]
        force_eos_rows = [False] * len(infos)
        mask_eos_rows = [False] * len(infos)
        empty_history = torch.empty(0, dtype=torch.long, device="cpu")
        penalty_histories = [empty_history for _ in infos]
        pending_codec_ids = self._decode_codec_id_map()
        codec_eos_id = int(self._codec_eos_id)
        gpu_rows = []
        gpu_groups = []
        frame_valid = [torch.empty(0, dtype=torch.bool, device="cpu") for _ in infos]
        for index, info in enumerate(infos):
            if not isinstance(info, dict):
                continue
            request_id = str(info.get("request_id", index))
            state = self._request_audio_states.get(request_id)
            if not isinstance(state, dict):
                state = dict(info.get("audio_state", {}) or {})
                self._request_audio_states[request_id] = state
            codes = info.get("codes", {})
            audio = codes.get("audio") if isinstance(codes, Mapping) else None
            pending = pending_codec_ids.pop(request_id, None)
            if pending is not None:
                # Accelerator batched decode keeps the codec ID and its state on
                # device; CPU decode retains the scalar output contract.
                owned_ids, row = pending
                if owned_ids.device.type in ("cuda", "npu"):
                    # Device state is resolved in one batch below, without a host read.
                    gpu_rows.append((index, state, owned_ids[row]))
                    gpu_groups.append((owned_ids, row))
                    continue
                code_id = int(owned_ids[row])
                if code_id == codec_eos_id:
                    state["finished"] = True
                    audio = None
                else:
                    audio = torch.tensor([[code_id]], dtype=torch.long, device="cpu")
            if isinstance(state.get("recent_codes"), torch.Tensor):
                # Prefill after duplex rollover inherits a device penalty window.
                gpu_rows.append((index, state, None))
                continue
            empty_speech = bool(state.get("finished"))
            if isinstance(audio, torch.Tensor) and audio.numel() > 0:
                audio = audio.to(device="cpu", dtype=torch.long)
                codec_deltas[index] = audio.reshape(-1, 1)
                frame_valid[index] = torch.ones(audio.numel(), dtype=torch.bool, device="cpu")
                state["step"] = int(state.get("step", 0)) + 1
                # ``audio`` is the id sampled last step, i.e. exactly upstream's
                # ``new_tokens[:, 0:t]`` history for the logits computed below.
                recent = state.get("recent_codes")
                recent = (recent if isinstance(recent, list) else []) + audio.reshape(-1).tolist()
                state["recent_codes"] = recent[-_CODEC_PENALTY_WINDOW:]
            recent_codes = state.get("recent_codes")
            if recent_codes:
                penalty_histories[index] = torch.tensor(recent_codes, dtype=torch.long, device="cpu")
            max_tokens = state.get("max_tokens")
            min_tokens = state.get("min_tokens")
            step = int(state.get("step", 0))
            # Duplex: 26 samples include the terminating EOS, so force it after
            # 25 forwarded frames — the same cadence as generate_chunk.
            # Offline: force-stop at the remaining Talker context rather than
            # waiting for a sampled EOS.
            hit_chunk_limit = max_tokens is not None and step >= int(max_tokens) - 1
            chunk_done = empty_speech or hit_chunk_limit
            if hit_chunk_limit:
                # The next sampled id is forced EOS and will finish the
                # request; mark the chunk done on this step so the data
                # plane does not wait for a follow-up empty decode.
                state["finished"] = True
            force_eos_rows[index] = chunk_done
            mask_eos_rows[index] = not force_eos_rows[index] and (
                (min_tokens is not None and step < int(min_tokens))
                or (bool(state.get("turn_end_drain")) and _turn_end_boundary_eos_masked(step))
            )
            terminal_flags[index] = torch.tensor(chunk_done, dtype=torch.bool)

        if gpu_rows:
            device = hidden.device
            from .codec_state import CodecState

            codec_state = getattr(self, "_device_codec_state", None)
            if codec_state is None:
                capacity = getattr(
                    getattr(getattr(self, "vllm_config", None), "scheduler_config", None), "max_num_seqs", 4096
                )
                codec_state = self._device_codec_state = CodecState(device, self._num_audio_tokens, capacity=capacity)
            rows = [(str(infos[index].get("request_id", index)), state, code) for index, state, code in gpu_rows]
            if (
                len(gpu_groups) == len(gpu_rows)
                and all(group is gpu_groups[0][0] for group, _ in gpu_groups)
                and [row for _, row in gpu_groups] == list(range(len(gpu_rows)))
                and gpu_groups[0][0].numel() == len(gpu_rows)
            ):
                ids = gpu_groups[0][0]
            else:
                zero = torch.zeros((), dtype=torch.long, device=device)
                ids = torch.stack([code if code is not None else zero for _, _, code in gpu_rows]).long()
            histories, valid, done, masked = codec_state.update(
                rows,
                ids,
                eos=codec_eos_id,
                cadence=_DUPLEX_CODEC_FRAMES_PER_CHUNK,
                boundary=_DUPLEX_TURN_END_BOUNDARY_MASK_STEPS,
            )
            history_rows = histories.unbind()
            valid_rows = valid.split(1)
            done_rows = done.unbind()
            masked_rows = masked.unbind()
            id_rows = ids.reshape(-1, 1).split(1)
            for row, (index, state, code) in enumerate(gpu_rows):
                state["recent_codes"] = history_rows[row]
                if code is not None:
                    codec_deltas[index] = id_rows[row]
                    frame_valid[index] = valid_rows[row]
                terminal_flags[index] = done_rows[row]
                force_eos_rows[index] = done_rows[row]
                mask_eos_rows[index] = masked_rows[row]
                penalty_histories[index] = history_rows[row]

            if len(gpu_rows) == len(infos):
                force_eos_rows, mask_eos_rows = done, masked
                penalty_histories = histories
            else:
                # Upload CPU prefill flags once per batch, then scatter GPU
                # flags without reading them back or uploading each scalar.
                indices = index_to_device([index for index, _, _ in gpu_rows], device)

                def device_flags(flags, values):
                    host = [False if isinstance(flag, torch.Tensor) else flag for flag in flags]
                    return index_to_device(host, device, dtype=torch.bool).index_copy_(0, indices, values)

                force_eos_rows = device_flags(force_eos_rows, done)
                mask_eos_rows = device_flags(mask_eos_rows, masked)

        # Empty-speech rows, finished duplex chunks, and offline requests that
        # fill the remaining Talker context must sample codec EOS so the
        # scheduler releases the request. Mid-chunk rows mask EOS until
        # min_tokens.
        self._force_eos_rows = force_eos_rows
        self._mask_eos_rows = mask_eos_rows
        self._penalty_histories = penalty_histories
        meta_outputs = {"finished": terminal_flags}
        if gpu_rows:
            meta_outputs["codec_frame_valid"] = frame_valid
        if emit_duplex_metadata:
            meta_outputs.update(self._duplex_output_metadata(infos))
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={
                "codes": {"audio": codec_deltas},
                "meta": meta_outputs,
            },
        )

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        self._deferred_cleanup_ids.update(str(req_id) for req_id in finished_req_ids)

    def _flush_deferred_cleanup(self) -> None:
        seeded_sampler = getattr(self, "_mrv2_seeded_codec_sampler", None)
        if seeded_sampler is not None:
            seeded_sampler.on_requests_finished(self._deferred_cleanup_ids)
        request_audio_states = getattr(self, "_request_audio_states", {})
        request_condition_states = getattr(self, "_request_condition_states", {})
        decode_codec_ids = getattr(self, "_decode_codec_ids", {})
        for request_id in self._deferred_cleanup_ids:
            request_audio_states.pop(request_id, None)
            request_condition_states.pop(request_id, None)
            decode_codec_ids.pop(request_id, None)
            codec_state = getattr(self, "_device_codec_state", None)
            if codec_state is not None:
                codec_state.release(request_id)
        self._deferred_cleanup_ids.clear()

    def _dummy_hidden_states(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        """Shape-correct zero tensor for vllm KV cache profiling.

        vllm's gpu_model_runner._dummy_run takes forward()'s return value as
        ``hidden_states`` and does ``hidden_states[logit_indices_device]``;
        returning None on the dummy path crashes with
        ``TypeError: 'NoneType' object is not subscriptable``.
        """
        for ref in (input_ids, positions, inputs_embeds):
            if isinstance(ref, torch.Tensor):
                num_tokens = int(ref.shape[0]) if ref.ndim >= 1 else 1
                device = ref.device
                break
        else:
            num_tokens = 1
            device = current_omni_platform.get_torch_device()
        hidden_size = int(getattr(self, "_hidden_size", 768) or 768)
        return torch.zeros((num_tokens, hidden_size), device=device, dtype=torch.bfloat16)

    def forward(
        self,
        input_ids=None,
        positions=None,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        self._flush_deferred_cleanup()
        if input_ids is None and inputs_embeds is None:
            return self._dummy_hidden_states(input_ids, positions, inputs_embeds)
        return self.tts_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states, *args, **kwargs):
        if not isinstance(hidden_states, torch.Tensor):
            return None
        if hidden_states.numel() == 0:
            return hidden_states.new_empty((0, int(self._num_audio_tokens)))
        logits = self.head_code[0](hidden_states).float()
        force_eos = self._force_eos_rows
        mask_eos = self._mask_eos_rows
        self._force_eos_rows = None
        self._mask_eos_rows = None
        if (
            isinstance(force_eos, torch.Tensor)
            and isinstance(mask_eos, torch.Tensor)
            and len(force_eos) == logits.shape[0]
            and len(mask_eos) == logits.shape[0]
        ):
            from .codec_state import mask_logits

            self._pending_force_eos_rows = force_eos
            return mask_logits(logits, force_eos, mask_eos, int(self._codec_eos_id))
        need_force = (
            force_eos is not None
            and len(force_eos) == logits.shape[0]
            and (isinstance(force_eos, torch.Tensor) or any(force_eos))
        )
        need_mask = (
            mask_eos is not None
            and len(mask_eos) == logits.shape[0]
            and (isinstance(mask_eos, torch.Tensor) or any(mask_eos))
        )
        # sample() re-applies the decision on the sampled ids: vLLM's
        # MinTokensLogitsProcessor runs after this and would blank the codec EOS
        # we just forced (it is in the stage's ``stop_token_ids``), leaving an
        # all -inf row and a request that never releases.
        self._pending_force_eos_rows = force_eos if need_force else None
        # Row masks go up through pinned memory and are applied with where /
        # masked_fill: a pageable H2D or boolean-mask indexing blocks the host.
        if need_force or need_mask:
            logits = logits.clone()
            eos_id = int(self._codec_eos_id)
            if need_force:
                assert force_eos is not None
                forced = (
                    force_eos
                    if isinstance(force_eos, torch.Tensor)
                    else index_to_device(force_eos, logits.device, dtype=torch.bool)
                )
                eos_only = torch.full_like(logits[:1], float("-inf"))
                eos_only[:, eos_id] = 0.0
                logits = torch.where(forced.unsqueeze(1), eos_only, logits)
            if need_mask:
                assert mask_eos is not None
                masked = (
                    mask_eos
                    if isinstance(mask_eos, torch.Tensor)
                    else index_to_device(mask_eos, logits.device, dtype=torch.bool)
                )
                logits[:, eos_id].masked_fill_(masked, float("-inf"))
        return logits

    @cached_property
    def _codec_sampler(self) -> Sampler:
        """Reuse the V1 sampler, as Qwen3-Omni does; metadata remains step-owned."""
        return Sampler()

    def sample(self, logits, sampling_metadata, *, per_req_sampling_params: list[SamplingParams | None] | None = None):
        # Check the host SamplingParams rather than reading GPU penalty tensors.
        # Missing/incomplete context keeps the general sampler penalty path.
        skip_upstream_penalties = (
            isinstance(logits, torch.Tensor)
            and per_req_sampling_params is not None
            and len(per_req_sampling_params) == logits.shape[0]
            and all(
                isinstance(params, SamplingParams) and params.frequency_penalty == 0 and params.presence_penalty == 0
                for params in per_req_sampling_params
            )
        )
        logits, sampling_metadata = self._apply_codec_repetition_penalty(
            logits, sampling_metadata, skip_upstream_penalties=skip_upstream_penalties
        )
        prompt_ids = getattr(sampling_metadata, "prompt_token_ids", None)
        if (
            isinstance(logits, torch.Tensor)
            and isinstance(prompt_ids, torch.Tensor)
            and not getattr(sampling_metadata, "no_penalties", False)
        ):
            # Copy rather than mutate: the runner may hand us the input batch's
            # own persistent SamplingMetadata.
            sampling_metadata = replace(
                sampling_metadata,
                prompt_token_ids=blank_scheduler_prompt_for_penalties(prompt_ids, logits.shape[-1]),
            )
        force_eos = self._pending_force_eos_rows
        self._pending_force_eos_rows = None
        output = self._codec_sampler(logits, sampling_metadata)
        return self._force_eos_on_sampled_ids(output, force_eos)

    def _apply_codec_repetition_penalty(self, logits, sampling_metadata, *, skip_upstream_penalties: bool = False):
        """Score MiniCPMTTS.generate's windowed codec penalty, not vLLM's.

        Upstream taxes a code by ``penalty ** frequency`` over the last
        ``past_window`` frames only (``gen_logits`` builds
        ``CustomRepetitionPenaltyLogitsProcessorRepeat(penalty, num_code, 16)``).
        vLLM's is presence-based over the whole stream, so on a codec stream
        thousands of frames long every code ever sampled ends up taxed by the
        same flat factor while never-sampled codes stay untouched, and the tail
        of a long answer drifts off the speech manifold into near-silence.

        Runs before ``Sampler`` so the penalty lands ahead of top-k/top-p, as
        upstream does. Upstream scores it after dividing by temperature, but
        the penalty only rescales and preserves sign, so the two orders agree.
        """
        histories = self._penalty_histories
        self._penalty_histories = None
        penalties = getattr(sampling_metadata, "repetition_penalties", None)
        if (
            not isinstance(logits, torch.Tensor)
            or histories is None
            or len(histories) != logits.shape[0]
            or not isinstance(penalties, torch.Tensor)
            or getattr(sampling_metadata, "no_penalties", False)
        ):
            return logits, sampling_metadata
        if isinstance(histories, torch.Tensor) and histories.device.type in ("cuda", "npu"):
            from .codec_state import apply_window_penalty

            logits = apply_window_penalty(
                logits.contiguous(), histories, to_device_nonblocking(penalties.float().contiguous(), logits.device)
            )
        else:
            logits = _apply_batched_repetition_penalty(
                logits,
                histories,
                penalty=penalties.to(device=logits.device, dtype=logits.dtype),
                window_size=_CODEC_PENALTY_WINDOW,
            )
        # Neutralize the sampler's own pass so the penalty is scored once.
        if skip_upstream_penalties:
            # The codec window has been scored and no frequency/presence
            # penalties remain. Avoid packing the whole output history and
            # launching neutral penalty kernels (including ones/full_like).
            return logits, replace(sampling_metadata, no_penalties=True)
        return logits, replace(sampling_metadata, repetition_penalties=torch.ones_like(penalties))

    def _force_eos_on_sampled_ids(self, output: Any, force_eos: list[bool] | torch.Tensor | None) -> Any:
        """Overwrite sampled ids for rows the model terminated this step.

        The codec EOS is a stage ``stop_token_ids`` entry, so vLLM's
        ``min_tokens`` processor masks it for the first ``min_tokens`` steps.
        A row the model forced to EOS therefore reaches the sampler as all
        -inf and comes back as an arbitrary codec id, which keeps an
        already-finished request decoding until its length cap.
        """
        if force_eos is None or (not isinstance(force_eos, torch.Tensor) and not any(force_eos)):
            return output
        sampled = getattr(output, "sampled_token_ids", None)
        if not isinstance(sampled, torch.Tensor) or sampled.shape[0] != len(force_eos):
            return output
        rows = (
            force_eos
            if isinstance(force_eos, torch.Tensor)
            else index_to_device(force_eos, sampled.device, dtype=torch.bool)
        )
        sampled.masked_fill_(rows.view(-1, *([1] * (sampled.ndim - 1))), int(self._codec_eos_id))
        return output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        return self._load_native_weights(weights)

    def _load_native_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded: set[str] = set()
        backbone_weights: list[tuple[str, torch.Tensor]] = []
        direct_params = dict(self.named_parameters())
        head_g = head_v = None

        for name, tensor in weights:
            if not name.startswith("tts."):
                continue
            stripped = name[len("tts.") :]
            if stripped.startswith("model."):
                backbone_weights.append((stripped[len("model.") :], tensor))
                continue
            if stripped == "head_code.0.parametrizations.weight.original0":
                head_g = tensor
                continue
            if stripped == "head_code.0.parametrizations.weight.original1":
                head_v = tensor
                continue
            target = stripped
            parameter = direct_params.get(target)
            if parameter is None:
                continue
            parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
            loaded.add(target)

        for name in self.tts_model.load_weights(backbone_weights):
            loaded.add(f"tts_model.{name}")

        if head_g is None or head_v is None:
            raise ValueError("MiniCPM-o checkpoint is missing weight-norm Talker head parameters")
        restored = _restore_weight_norm_weight(head_g, head_v)
        self.head_code[0].weight.data.copy_(
            restored.to(
                device=self.head_code[0].weight.device,
                dtype=self.head_code[0].weight.dtype,
            )
        )
        loaded.add("head_code.0.weight")
        return loaded

    def get_input_embeddings(self, input_ids, multimodal_embeddings=None, **kwargs):
        del multimodal_embeddings
        # Decode tokens live in the codec table. Prefill overwrites these
        # embeddings in preprocess with emb_text + projected thinker hidden.
        return self.emb_code[0](input_ids)

    def embed_input_ids(self, input_ids, **kwargs):
        return self.get_input_embeddings(input_ids, **kwargs)
