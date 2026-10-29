# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Small model-independent helpers for continuous-latent TTS decoders."""

from __future__ import annotations

from dataclasses import dataclass

STOP_REASON_CONTINUE = "continue"
STOP_REASON_STOP_HEAD = "stop_head"
STOP_REASON_MAX_DECODE_STEPS = "max_decode_steps"
STOP_REASON_CODES = {
    STOP_REASON_CONTINUE: 0,
    STOP_REASON_STOP_HEAD: 1,
    STOP_REASON_MAX_DECODE_STEPS: 2,
}
STOP_REASON_BY_CODE = {code: reason for reason, code in STOP_REASON_CODES.items()}


@dataclass(frozen=True)
class StopDecision:
    reason: str
    should_stop: bool
    forced: bool
    minimum_steps: int
    next_token_id: int


def resolve_stop_decision(
    *,
    emitted_steps: int,
    stop_probability: float,
    stop_threshold: float,
    minimum_steps: int,
    maximum_steps: int,
    continue_token_id: int,
    stop_token_id: int,
) -> StopDecision:
    """Resolve the token forced after an already-emitted latent patch.

    emitted_steps is one-based. This makes the ordering explicit and
    prevents callers from accidentally discarding the patch that caused the
    stop head to fire.
    """
    if minimum_steps < 1:
        raise ValueError("minimum_steps must be positive")
    if maximum_steps < minimum_steps:
        raise ValueError(f"maximum_steps={maximum_steps} is smaller than minimum_steps={minimum_steps}")
    if emitted_steps >= maximum_steps:
        return StopDecision(
            STOP_REASON_MAX_DECODE_STEPS,
            True,
            True,
            minimum_steps,
            stop_token_id,
        )
    if emitted_steps >= minimum_steps and stop_probability > stop_threshold:
        return StopDecision(
            STOP_REASON_STOP_HEAD,
            True,
            False,
            minimum_steps,
            stop_token_id,
        )
    return StopDecision(
        STOP_REASON_CONTINUE,
        False,
        False,
        minimum_steps,
        continue_token_id,
    )


__all__ = [
    "STOP_REASON_CONTINUE",
    "STOP_REASON_BY_CODE",
    "STOP_REASON_CODES",
    "STOP_REASON_MAX_DECODE_STEPS",
    "STOP_REASON_STOP_HEAD",
    "StopDecision",
    "resolve_stop_decision",
]
