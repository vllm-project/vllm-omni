# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Merge a certified immutable feedback batch without per-view certificates.

No device values are read here. Every guard completes before request state is
written. Ordinary postprocess remains the fallback for unsupported batches.
"""

from __future__ import annotations

import weakref
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class OwnedFeedbackBatch:
    audio: weakref.ReferenceType
    snapshot: object
    phonemes: torch.Tensor | None
    ended: torch.Tensor | None
    layouts: tuple
    live_rows: frozenset


def certify_feedback_batch(audio, snapshot):
    """Called only immediately after the existing owning clones are created."""
    audio._easymagpie_owned_feedback_batch = OwnedFeedbackBatch(
        weakref.ref(audio),
        snapshot,
        snapshot.phonemes,
        snapshot.ended,
        tuple(_layout(value) for value in (audio, snapshot.phonemes, snapshot.ended)),
        snapshot.live_rows,
    )


def feedback_merge_status(runner):
    return {
        "enabled": bool(getattr(runner.get_model(), "_batched_feedback_merge_enabled", False)),
        "attempts": getattr(runner, "_feedback_merge_attempts", 0),
        "hits": getattr(runner, "_feedback_merge_hits", 0),
        "requests": getattr(runner, "_feedback_merge_requests", 0),
        "fallbacks": dict(getattr(runner, "_feedback_merge_fallbacks", {})),
    }


def try_merge_feedback(
    runner,
    hidden_states,
    multimodal_outputs,
    num_scheduled_tokens_np,
    *,
    combined_hidden_states=None,
    combined_multimodal_outputs=None,
    req_ids_filter=None,
    req_ids=None,
    query_start_loc_cpu=None,
):
    from .easymagpie import EasyMagpieTTSForConditionalGeneration

    # The pinned runner unwraps supported CUDA/ubatch wrappers without executing
    # model code. Keep the exact underlying EasyMagpie type check below.
    model = runner.get_model()
    if not getattr(model, "_batched_feedback_merge_enabled", False):
        return False
    runner._feedback_merge_attempts = getattr(runner, "_feedback_merge_attempts", 0) + 1

    def fallback(reason):
        if not hasattr(runner, "_feedback_merge_fallbacks"):
            runner._feedback_merge_fallbacks = Counter()
        runner._feedback_merge_fallbacks[reason] += 1
        return False

    if type(model) is not EasyMagpieTTSForConditionalGeneration or not getattr(
        model, "_batched_owned_feedback_enabled", False
    ):
        return fallback("model")
    if combined_hidden_states is not None or combined_multimodal_outputs is not None:
        return fallback("combined_payload")
    current_ids = runner.input_batch.req_ids
    ids = current_ids if req_ids is None else req_ids
    count = len(ids)
    if not count or list(ids) != list(current_ids) or len(set(ids)) != count:
        return fallback("request_order")
    # The pinned eager async hook supplies the full downstream set even when
    # every request participates. Partial filters retain generic semantics.
    if req_ids_filter is not None and set(req_ids_filter) != set(ids):
        return fallback("partial_filter")
    if (
        not isinstance(hidden_states, torch.Tensor)
        or hidden_states.ndim != 2
        or not hidden_states.is_contiguous()
        or hidden_states.storage_offset() != 0
        or hidden_states.numel() == 0
        or hidden_states.shape[0] < count
    ):
        return fallback("hidden_layout")
    counts = _cpu_values(num_scheduled_tokens_np)
    starts = _cpu_values(runner.query_start_loc.cpu if query_start_loc_cpu is None else query_start_loc_cpu)
    if counts is None or counts != [1] * count or starts is None or starts[: count + 1] != list(range(count + 1)):
        return fallback("token_spans")
    requests = [runner.requests.get(key) for key in ids]
    if any(request is None for request in requests):
        return fallback("missing_request")
    existing = [runner.model_intermediate_buffer.get(key) for key in ids]
    if any(value is not None and type(value) is not dict for value in existing):
        return fallback("buffer_type")
    audio = _audio_tensor(multimodal_outputs)
    certificate = getattr(audio, "_easymagpie_owned_feedback_batch", None)
    if not _valid_batch(audio, certificate, model, hidden_states):
        return fallback("snapshot")

    # split/unbind construct all views together; each view retains its original
    # immutable allocation. No clone, scalar read or CUDA operation is needed.
    audio_rows = audio[:count].split(1, dim=0)
    phoneme_rows = certificate.phonemes[:count].split(1, dim=0) if model.has_phoneme else None
    ended_rows = certificate.ended[:count].unbind(0) if certificate.ended is not None else None
    for index, (key, request, previous) in enumerate(zip(ids, requests, existing, strict=True)):
        info = previous if previous is not None else {}
        info["last_audio_codes"] = audio_rows[index]
        if phoneme_rows is not None:
            info["last_phoneme_token"] = phoneme_rows[index]
        if ended_rows is not None and index in certificate.live_rows:
            info["phoneme_ended"] = ended_rows[index]
        if previous is None:
            runner.model_intermediate_buffer[key] = info
        request.additional_information_cpu = info
    runner._feedback_merge_hits = getattr(runner, "_feedback_merge_hits", 0) + 1
    runner._feedback_merge_requests = getattr(runner, "_feedback_merge_requests", 0) + count
    return True


def _layout(value):
    if value is None:
        return None
    return (
        tuple(value.shape),
        tuple(value.stride()),
        value.storage_offset(),
        value.dtype,
        value.device,
        value.untyped_storage().data_ptr(),
    )


def _valid_batch(audio, certificate, model, hidden):
    if not isinstance(certificate, OwnedFeedbackBatch) or certificate.audio() is not audio:
        return False
    snapshot = getattr(audio, "_easymagpie_feedback_snapshot", None)
    if (
        snapshot is not certificate.snapshot
        or snapshot.audio() is not audio
        or snapshot.phonemes is not certificate.phonemes
        or snapshot.ended is not certificate.ended
        or snapshot.live_rows != certificate.live_rows
    ):
        return False
    batches = (audio, certificate.phonemes, certificate.ended)
    if tuple(_layout(value) for value in batches) != certificate.layouts:
        return False
    if any(value is not None and (value.requires_grad or not value.is_contiguous()) for value in batches):
        return False
    if (
        audio.dtype != torch.long
        or audio.device != hidden.device
        or audio.shape != (hidden.shape[0], model.num_codebooks)
    ):
        return False
    phonemes = certificate.phonemes
    if model.has_phoneme:
        if (
            phonemes is None
            or phonemes.dtype != torch.long
            or phonemes.device != audio.device
            or phonemes.shape != (audio.shape[0], model.arch.phoneme_stacking_factor)
        ):
            return False
    elif phonemes is not None:
        return False
    ended = certificate.ended
    return ended is None or (
        ended.dtype == torch.bool and ended.device == audio.device and ended.shape == (audio.shape[0],)
    )


def _cpu_values(value):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.ndim != 1 or value.dtype not in (torch.int32, torch.int64):
            return None
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist() if value.ndim == 1 and np.issubdtype(value.dtype, np.integer) else None
    if isinstance(value, (list, tuple)) and all(type(item) is int for item in value):
        return list(value)
    return None


def _audio_tensor(payload):
    if not isinstance(payload, dict):
        return None
    for key in ("audio_codes", "model_outputs"):
        value = payload.get(key)
        if value is not None:
            return value
    codes = payload.get("codes")
    return codes.get("audio") if isinstance(codes, dict) else payload.get("codes.audio")
