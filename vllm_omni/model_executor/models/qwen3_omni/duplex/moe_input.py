# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Commit-only PCM buffer for Qwen3-Omni duplex (emit whole utterance on commit)."""

from __future__ import annotations

from vllm_omni.model_executor.models.qwen3_omni.duplex.turn_commit import (
    TurnCommitPcmAppendBuffer,
    TurnCommitPcmAppendReservation,
)


class Qwen3OmniPcmAppendReservation(TurnCommitPcmAppendReservation):
    pass


class Qwen3OmniPcmAppendBuffer(TurnCommitPcmAppendBuffer):
    def __init__(self) -> None:
        super().__init__(label="Qwen3-Omni duplex")


__all__ = ["Qwen3OmniPcmAppendBuffer", "Qwen3OmniPcmAppendReservation"]
