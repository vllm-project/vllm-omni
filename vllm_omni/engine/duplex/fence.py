# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Read-only control identity checks shared by dispatch and state transitions."""

from vllm_omni.engine.duplex.messages import DuplexFence


class DuplexFenceMismatchError(RuntimeError):
    def __init__(self, expected: DuplexFence, actual: DuplexFence) -> None:
        super().__init__(f"duplex fence mismatch: expected {expected!r}, got {actual!r}")
        self.expected = expected
        self.actual = actual


def validate_fence(current: DuplexFence, incoming: DuplexFence) -> None:
    if incoming.session_id != current.session_id or incoming.incarnation != current.incarnation:
        raise DuplexFenceMismatchError(current, incoming)
    if incoming.epoch < current.epoch or (
        incoming.epoch == current.epoch
        and (incoming.turn_id < current.turn_id or incoming.response_seq < current.response_seq)
    ):
        raise DuplexFenceMismatchError(current, incoming)


def validate_cancel_fences(current: DuplexFence, cancelled: DuplexFence, next_fence: DuplexFence) -> None:
    if cancelled.session_id != current.session_id or cancelled.incarnation != current.incarnation:
        raise DuplexFenceMismatchError(current, cancelled)
    if (
        next_fence.session_id != current.session_id
        or next_fence.incarnation != current.incarnation
        or next_fence.epoch <= cancelled.epoch
    ):
        raise DuplexFenceMismatchError(cancelled, next_fence)
    current_key = (current.epoch, current.turn_id, current.response_seq)
    cancelled_key = (cancelled.epoch, cancelled.turn_id, cancelled.response_seq)
    next_key = (next_fence.epoch, next_fence.turn_id, next_fence.response_seq)
    # A retry of an already-applied cancel may clean only its old fence. It
    # cannot use that stale identity to advance a newer, live epoch again.
    if cancelled_key > current_key or (cancelled_key < current_key and next_key > current_key):
        raise DuplexFenceMismatchError(current, cancelled)


def preemption_covers(control: DuplexFence, append: DuplexFence, *, close: bool) -> bool:
    if not close:
        return append == control
    return (
        append.session_id == control.session_id
        and append.incarnation == control.incarnation
        and (append.epoch, append.turn_id, append.response_seq)
        <= (control.epoch, control.turn_id, control.response_seq)
    )
