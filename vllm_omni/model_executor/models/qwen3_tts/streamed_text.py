# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The Talker's view of each request's streamed text (see ``core/sched/text_stream.py``).

The scheduler hands a request its newly arrived text ids with each of its
scheduled steps, and the runner passes them to :meth:`StreamedTexts.receive`
before the step's preprocess. The Talker builds a streamed request's prompt
from the ids it has received, appends later ids to the request's trailing
text queue, and forbids codec EOS until the request has read its text END.
The scheduler only runs a step whose text has arrived, so the queue never
runs dry before the END.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import torch
from vllm.logger import init_logger

from vllm_omni.core.sched.output import TextStreamDelivery
from vllm_omni.core.sched.text_stream import end_read, text_row
from vllm_omni.utils.device_copy import index_to_device

logger = init_logger(__name__)


def lead_layout(
    prefix: torch.Tensor,
    text_ids: torch.Tensor,
    *,
    project: Callable[[torch.Tensor], torch.Tensor],
    ended: bool,
    lead: int,
    pad_row: torch.Tensor,
    bos_row: torch.Tensor,
    tts_eos_embed: torch.Tensor,
    tts_pad_embed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The streaming CustomVoice prompt with ``lead`` text ids before codec BOS.

    After ``prefix``: ``text[:lead]``, each + codec PAD (the END too when the
    lead holds all ``n`` ids and the END, ``lead = n + 1``); then
    ``text[lead]`` (the END at ``lead = n``, PAD past it) + codec BOS. The rest
    of the text trails, with its END once the text has ended; a text that has
    not ended trails no END, and the Talker appends what arrives later. With
    ``lead = 0`` and a whole text this is the stock streaming layout row for
    row: the BOS row and the trailing rows are projected in separate calls, as
    there. ``project`` maps ``[1, k]`` text ids to ``[1, k, hidden]`` rows.
    """
    n = int(text_ids.shape[1])
    if lead > n + 1 or (lead >= n and not ended):
        raise ValueError(f"text lead {lead} needs {lead + 1} text rows; the request has {n} text ids")

    def rows(ids: torch.Tensor) -> torch.Tensor:
        return prefix[:, :0] if ids.shape[1] == 0 else project(ids)

    lead_rows = rows(text_ids[:, : min(lead, n)])
    if lead == n + 1:
        lead_rows = torch.cat([lead_rows, tts_eos_embed], dim=1)
    if lead < n:
        bos_text = rows(text_ids[:, lead : lead + 1])
    else:
        bos_text = tts_eos_embed if lead == n else tts_pad_embed
    prompt = torch.cat([prefix, lead_rows + pad_row, bos_text + bos_row], dim=1)
    if lead >= n:
        return prompt, tts_pad_embed
    trailing = rows(text_ids[:, lead + 1 :])
    if ended:
        trailing = torch.cat([trailing, tts_eos_embed], dim=1)
    return prompt, trailing


@dataclass
class StreamedText:
    """One request's streamed text, as delivered so far."""

    prompt_ids: int
    token_ids: list[int] = field(default_factory=list)
    ended: bool = False
    # Ids already placed in the prompt or the trailing queue, and whether the END is.
    queued: int = 0
    end_queued: bool = False


class StreamedTexts:
    """Streamed text of the Talker's live requests, keyed by request id."""

    def __init__(self) -> None:
        self._texts: dict[str, StreamedText] = {}

    def __bool__(self) -> bool:
        return bool(self._texts)

    def __contains__(self, req_id: object) -> bool:
        return req_id in self._texts

    def receive(self, deliveries: Mapping[str, TextStreamDelivery]) -> None:
        """Record a step's deliveries. A delivery repeats no id the request already has."""
        for req_id, delivery in deliveries.items():
            text = self._texts.get(req_id)
            if text is None:
                text = self._texts[req_id] = StreamedText(prompt_ids=delivery.prompt_ids)
            have = len(text.token_ids)
            if delivery.start > have:
                raise RuntimeError(
                    f"Streamed text of request {req_id} skips ids {have} to {delivery.start - 1}; "
                    "the scheduler hands each id over once, in order."
                )
            text.token_ids.extend(delivery.token_ids[have - delivery.start :])
            text.ended = text.ended or delivery.ended

    def prompt_text(self, req_id: str | None, device: torch.device) -> tuple[torch.Tensor, bool] | None:
        """The ids for a streamed request's prompt, ``[1, n]``, and whether its text ended.

        Every id received so far goes into the prompt build (the prompt's own
        rows, then the trailing queue), so they count as queued.
        """
        text = self._texts.get(req_id) if req_id is not None else None
        if text is None:
            return None
        text.queued = len(text.token_ids)
        text.end_queued = text.ended
        return index_to_device(text.token_ids, device).reshape(1, -1), text.ended

    def extend_queue(
        self,
        req_id: str | None,
        tail: torch.Tensor | None,
        text_offset: int,
        *,
        device: torch.device,
        project: Callable[[torch.Tensor], torch.Tensor],
        end_row: Callable[[torch.device], torch.Tensor],
    ) -> tuple[torch.Tensor | None, int, bool]:
        """Append a streamed request's newly received text rows to its trailing queue.

        Returns the queue to read, its read offset and whether it changed (the
        caller then stores it). Other requests pass through unchanged.
        ``project`` maps ``[1, k]`` ids on ``device`` to ``[1, k, hidden]``
        rows; ``end_row`` gives the ``[1, 1, hidden]`` END row.
        """
        text = self._texts.get(req_id) if req_id is not None else None
        if text is None or (text.queued == len(text.token_ids) and text.end_queued == text.ended):
            return tail, text_offset, False
        rows: list[torch.Tensor] = []
        new_ids = text.token_ids[text.queued :]
        if new_ids:
            rows.append(project(index_to_device(new_ids, device).reshape(1, -1))[0])
        if text.ended and not text.end_queued:
            rows.append(end_row(device)[0])
        new_rows = torch.cat(rows, dim=0)
        if isinstance(tail, torch.Tensor) and tail.ndim == 2:
            new_rows = torch.cat([tail[text_offset:], new_rows.to(device=tail.device, dtype=tail.dtype)], dim=0)
        text.queued = len(text.token_ids)
        text.end_queued = text.ended
        return new_rows, 0, True

    def is_open(self, req_id: str | None) -> bool:
        """Whether a request streams text whose END has not arrived."""
        text = self._texts.get(req_id) if req_id is not None else None
        return text is not None and not text.ended

    def queue_open(self, req_id: str | None) -> bool:
        """Whether more rows will reach a streamed request's text queue (its END is not in it yet)."""
        text = self._texts.get(req_id) if req_id is not None else None
        return text is not None and not text.end_queued

    def forbid_early_eos(
        self,
        logits: torch.Tensor,
        req_ids: Sequence[str],
        frames: Sequence[int] | torch.Tensor,
        eos_token_id: int,
    ) -> torch.Tensor:
        """Forbid codec EOS to each streamed row that has not read its text END.

        Training never ends audio before the text. ``frames`` holds each row's
        count of sampled frames: per-row lengths on the host, or MRV2's
        per-row counts on the device (the comparison then stays on the device).
        """
        rows: list[int] = []
        texts: list[StreamedText] = []
        for row, req_id in enumerate(req_ids):
            text = self._texts.get(req_id)
            if text is not None:
                rows.append(row)
                texts.append(text)
        if not rows or not 0 <= eos_token_id < logits.shape[1]:
            return logits
        if isinstance(frames, torch.Tensor):
            device = logits.device
            rows_t = index_to_device(rows, device)
            spec = index_to_device(
                [value for text in texts for value in (text.prompt_ids, len(text.token_ids), int(text.ended))], device
            ).reshape(-1, 3)
            frame = frames.index_select(0, rows_t).to(torch.long)
            ban = (spec[:, 2] == 0) | (text_row(spec[:, 0], frame) < spec[:, 1])
            column = logits[rows_t, eos_token_id]
            logits[rows_t, eos_token_id] = torch.where(ban, torch.full_like(column, float("-inf")), column)
            return logits
        banned = [
            row
            for row, text in zip(rows, texts, strict=True)
            if not end_read(text.prompt_ids, int(frames[row]), len(text.token_ids), text.ended)
        ]
        if banned:
            logits[banned, eos_token_id] = float("-inf")
        return logits

    def release(self, req_ids: Iterable[str]) -> None:
        """Drop the streamed text of finished or aborted requests."""
        for req_id in req_ids:
            self._texts.pop(req_id, None)
