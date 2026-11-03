# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Batched owned feedback, certified only for this worker's private store hook.

Each returned view retains its immutable source batch. No request-slot lifetime
assumption is needed: source tensors never alias graph scratch or decode state.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class _OwnedView:
    key: str
    owner: torch.Tensor
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    offset: int


def _view(owner, index, key, *, scalar=False):
    value = (owner[index] if scalar else owner[index : index + 1]).detach()
    value._easymagpie_owned_feedback = _OwnedView(
        key, owner, tuple(value.shape), tuple(value.stride()), value.storage_offset()
    )
    return value


def is_owned_feedback(value, key):
    """Accept only a view explicitly issued from an owned snapshot batch."""
    if not isinstance(value, torch.Tensor):
        return False
    marker = getattr(value, "_easymagpie_owned_feedback", None)
    return (
        isinstance(marker, _OwnedView)
        and marker.key == key
        and tuple(value.shape) == marker.shape
        and tuple(value.stride()) == marker.stride
        and value.storage_offset() == marker.offset
        and value.dtype == marker.owner.dtype
        and value.device == marker.owner.device
        and value.untyped_storage().data_ptr() == marker.owner.untyped_storage().data_ptr()
    )


class FeedbackSnapshot:
    def __init__(self, model, audio):
        # A weak reference avoids audio -> snapshot -> audio reference cycles.
        # Issued audio views separately retain their original owned batch.
        self.audio = weakref.ref(audio)
        count = audio.shape[0]
        self.phonemes = model._dec_phoneme_tokens[:count].detach().clone() if model.has_phoneme else None
        self.ended = None
        self.live_rows = frozenset()
        state = getattr(model, "_device_decode_state", None)
        if state is not None and model.has_phoneme:
            self.ended = state.ended.index_select(0, state.row_slots[:count])
            self.live_rows = frozenset(state.row_slots_host)

    def outputs(self, audio, last):
        if self.audio() is not audio or not 0 <= last < audio.shape[0]:
            return None
        output = {"last_audio_codes": _view(audio, last, "last_audio_codes")}
        if self.phonemes is not None:
            output["last_phoneme_token"] = _view(self.phonemes, last, "last_phoneme_token")
        if self.ended is not None and last in self.live_rows:
            output["phoneme_ended"] = _view(self.ended, last, "phoneme_ended", scalar=True)
        return output


def attach_feedback_snapshot(model, audio):
    snapshot = audio._easymagpie_feedback_snapshot = FeedbackSnapshot(model, audio)
    if getattr(model, "_batched_feedback_merge_enabled", False):
        from .batched_feedback import certify_feedback_batch

        certify_feedback_batch(audio, snapshot)


def owned_feedback_outputs(audio, last):
    snapshot = getattr(audio, "_easymagpie_feedback_snapshot", None)
    return snapshot.outputs(audio, last) if isinstance(snapshot, FeedbackSnapshot) else None
