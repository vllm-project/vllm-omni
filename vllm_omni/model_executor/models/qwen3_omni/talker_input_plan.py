# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Metadata-only layout shared by Talker admission and embedding assembly."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TalkerPrefillPart:
    kind: Literal["user", "assistant"]
    start: int
    end: int

    @property
    def num_rows(self) -> int:
        return self.end - self.start if self.kind == "user" else 9


def read_talker_prefill_plan(
    value: object, *, sequence_length: int, embedding_rows: int, hidden_rows: int
) -> tuple[TalkerPrefillPart, ...]:
    """Read the producer's layout, without scanning tokens or planning again.

    This small wire record travels with the existing fixed payload. There is
    no request-ID keyed plan cache to invalidate on replacement or ID reuse.
    """
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("Talker prefill plan must contain retained parts")
    parts = []
    previous_end = 0
    for index, entry in enumerate(value):
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            raise ValueError("Talker prefill plan requires kind/start/end entries")
        kind, start, end = entry
        if kind not in ("user", "assistant") or type(start) is not int or type(end) is not int:
            raise ValueError("Talker prefill plan has invalid role or offsets")
        available = (
            min(sequence_length, embedding_rows, hidden_rows)
            if kind == "user"
            else min(sequence_length, embedding_rows)
        )
        if not previous_end <= start < end <= available:
            raise ValueError("Talker prefill plan overlaps or exceeds retained source rows")
        if kind == "assistant" and (index != len(value) - 1 or end - start < 3):
            raise ValueError("Talker prefill plan requires a final three-token assistant header")
        parts.append(TalkerPrefillPart(kind, start, end))
        previous_end = end
    return tuple(parts)


def plan_talker_prefill(
    prompt_ids: Sequence[int],
    sequence_length: int,
    *,
    im_start_token_id: int = 151644,
    system_token_id: int = 8948,
    user_token_id: int = 872,
    assistant_token_id: int = 77091,
    embedding_rows: int | None = None,
    hidden_rows: int | None = None,
) -> tuple[TalkerPrefillPart, ...]:
    """Retain users and the final assistant's nine-row bootstrap.

    Boundaries come from the prompt, not generated text that might itself
    contain a chat marker. The final span extends through the known sequence.
    No tensors, projections, request IDs or cache state belong in this plan.
    """
    if sequence_length < len(prompt_ids):
        raise ValueError("Thinker sequence ends before its prompt")
    embedding_rows = sequence_length if embedding_rows is None else embedding_rows
    hidden_rows = sequence_length if hidden_rows is None else hidden_rows
    if embedding_rows < 0 or hidden_rows < 0:
        raise ValueError("Thinker conditioning row counts must be non-negative")
    starts = [index for index, token in enumerate(prompt_ids) if token == im_start_token_id]
    parts = []
    for index, start in enumerate(starts):
        if start + 1 >= len(prompt_ids):
            raise ValueError("Thinker chat segment has no role token")
        end = starts[index + 1] if index + 1 < len(starts) else sequence_length
        role = prompt_ids[start + 1]
        if role == user_token_id:
            end = min(end, embedding_rows, hidden_rows)
            if end > start:
                parts.append(TalkerPrefillPart("user", start, end))
        elif role == assistant_token_id:
            if index == len(starts) - 1:
                end = min(end, embedding_rows)
                if end - start < 3:
                    raise ValueError("Talker assistant bootstrap requires a three-token header")
                parts.append(TalkerPrefillPart("assistant", start, end))
        elif role != system_token_id:
            raise ValueError("Expect assistant, user or system after the Thinker chat marker")
    return tuple(parts)


def talker_prefill_token_ids(
    plan: Sequence[TalkerPrefillPart],
    sequence_ids: Sequence[int],
    *,
    tts_pad_token_id: int,
    tts_bos_token_id: int,
) -> list[int]:
    """Identity rows, not indices into the Talker's codec embedding table.

    A missing first-text row uses PAD; its absence and the codec additions
    also belong in the complete conditioning descriptor, not in these IDs
    alone. The plan must use actual conditioning row counts after trimming.
    """
    token_ids: list[int] = []
    for part in plan:
        if part.end > len(sequence_ids):
            raise ValueError("Talker row plan exceeds the Thinker token sequence")
        if part.kind == "user":
            token_ids.extend(sequence_ids[part.start : part.end])
        else:
            token_ids.extend(sequence_ids[part.start : part.start + 3])
            token_ids.extend([tts_pad_token_id] * 4)
            token_ids.append(tts_bos_token_id)
            token_ids.append(sequence_ids[part.start + 3] if part.end > part.start + 3 else tts_pad_token_id)
    return token_ids
