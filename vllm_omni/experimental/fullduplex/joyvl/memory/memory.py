# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm_omni.experimental.fullduplex.joyvl.bridges.backend import ModelBackend
from vllm_omni.experimental.fullduplex.joyvl.decision import prompts


@dataclass
class WorkingChunk:
    messages: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class MidTermSummary:
    chunk_index: int
    frame_range: str
    summary_text: str


@dataclass
class QAEntry:
    query: str
    query_time: str
    responses: list[tuple[str, str]] = field(default_factory=list)
    archived_in_chunk: int = 0


@dataclass
class SessionMemory:
    long_term_memory: str = ""
    long_term_blocks: list[str] = field(default_factory=list)
    mid_term_summaries: list[MidTermSummary] = field(default_factory=list)
    qa_history: list[QAEntry] = field(default_factory=list)


def build_memory_prefix(
    memory: SessionMemory,
    *,
    current_query: str | None,
    query_in_current_chunk: bool,
    keep_qa_history: bool,
    current_chunk_index: int,
) -> str:
    has_query = bool(current_query)
    sections: list[str] = []

    history_parts: list[str] = []
    if has_query and memory.long_term_memory:
        history_parts.append(memory.long_term_memory)
    for entry in memory.mid_term_summaries:
        history_parts.append(f"<{entry.frame_range}>\n{entry.summary_text}")
    if history_parts:
        sections.append(prompts.VIDEO_HISTORY_HEADER + "\n\n".join(history_parts))

    if keep_qa_history and has_query:
        qa_entries = [e for e in memory.qa_history if e.archived_in_chunk < current_chunk_index]
        if qa_entries:
            lines = []
            for idx, entry in enumerate(qa_entries, 1):
                parts = [f"#{idx} [{prompts.QA_QUERY_LABEL}@{entry.query_time or 'N/A'}] {entry.query}"]
                for resp_time, payload in entry.responses:
                    parts.append(f"[{prompts.QA_RESPONSE_LABEL}@{resp_time}] {payload}")
                lines.append("\n".join(parts))
            sections.append(prompts.QA_HISTORY_HEADER + "\n".join(lines))

    if has_query and not query_in_current_chunk:
        sections.append(prompts.USER_QUERY_HEADER + "\n" + current_query.strip())

    return "\n\n".join(sections)


def _sample_indices(n: int, budget: int) -> list[int]:
    if n == 0:
        return []
    if budget <= 0 or n <= budget:
        return list(range(n))
    if budget == 1:
        return [n // 2]
    selected: list[int] = []
    last_pos = -1
    for slot in range(budget):
        target = round(slot * (n - 1) / (budget - 1))
        target = max(target, last_pos + 1)
        target = min(target, n - (budget - slot))
        selected.append(target)
        last_pos = target
    return selected


def _resize_data_url(data_url: str, max_pixels: int) -> str:
    """Downscale a base64 image data URL so width*height <= max_pixels (preserving
    aspect ratio), mirroring the reference summarizer. No-op if already within budget,
    not a data URL, or PIL is unavailable."""
    if max_pixels <= 0 or not data_url.startswith("data:"):
        return data_url
    import base64
    import io
    import re

    match = re.match(r"data:image/\w+;base64,(.+)", data_url)
    if not match:
        return data_url
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(base64.b64decode(match.group(1))))
        w, h = img.size
        if w * h <= max_pixels:
            return data_url
        scale = (max_pixels / (w * h)) ** 0.5
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG")
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return data_url


class Summarizer:
    def __init__(
        self,
        backend: ModelBackend,
        *,
        key_frames_per_chunk: int = 0,
        mid_term_max_tokens: int = 4000,
        long_term_max_tokens: int = 4000,
        preferred_time_span: float = 10.0,
        max_pixels: int = 262144,
    ) -> None:
        self._backend = backend
        self._key_frames = key_frames_per_chunk
        self._mid_max_tokens = mid_term_max_tokens
        self._long_max_tokens = long_term_max_tokens
        self._preferred_time_span = preferred_time_span
        self._max_pixels = max_pixels

    async def summarize_chunk(
        self,
        chunk_index: int,
        frame_range: str,
        frames: list[tuple[str, str]],
    ) -> str:
        if not frames:
            return prompts.EMPTY_CHUNK_SUMMARY.format(frame_range=frame_range)

        picked = [frames[i] for i in _sample_indices(len(frames), self._key_frames)]
        prompt = prompts.MID_TERM_SUMMARY_PROMPT.format(
            chunk_index=chunk_index,
            frame_range=frame_range,
            length_instruction="",
            preferred_time_span=f"{self._preferred_time_span:g} seconds",
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for time_range, data_url in picked:
            content.append({"type": "text", "text": f"<{time_range}>"})
            content.append({"type": "image_url", "image_url": {"url": _resize_data_url(data_url, self._max_pixels)}})

        text, _ = await self._backend.generate(
            [{"role": "user", "content": content}],
            max_tokens=self._mid_max_tokens,
            temperature=0.8,
            top_p=0.9,
            extra_body={"top_k": 40, "repetition_penalty": 1.1, "presence_penalty": 1.0, "greedy": False},
        )
        return text.strip()

    async def compress_to_long_term(self, mid_terms: list[MidTermSummary]) -> str:
        """Compress a batch of mid-term summaries into a single long-term block.

        Returns one block; the caller (InteractionBrain) accumulates blocks and applies
        the sliding window, matching the reference long-term memory behavior.
        """
        if not mid_terms:
            return ""

        merged_range = f"{_range_start(mid_terms[0].frame_range)}-{_range_end(mid_terms[-1].frame_range)}"
        summaries_text = "\n\n".join(f"<{m.frame_range}>\n{m.summary_text}" for m in mid_terms)
        prompt = prompts.LONG_TERM_COMPRESS_PROMPT.format(
            merged_range=merged_range,
            summaries_text=summaries_text,
            length_instruction="",
        )
        compressed, _ = await self._backend.generate(
            [{"role": "user", "content": prompt}],
            max_tokens=self._long_max_tokens,
            temperature=0.3,
            top_p=0.7,
            extra_body={"top_k": 30, "repetition_penalty": 1.1, "presence_penalty": 0.5, "greedy": False},
        )
        return f"<{merged_range}>\n{compressed.strip()}"

    async def aclose(self) -> None:
        await self._backend.aclose()


def _range_start(frame_range: str) -> str:
    for sep in (" ~ ", "-"):
        if sep in frame_range:
            return frame_range.split(sep, 1)[0].strip()
    return frame_range.strip()


def _range_end(frame_range: str) -> str:
    for sep in (" ~ ", "-"):
        if sep in frame_range:
            return frame_range.split(sep, 1)[-1].strip()
    return frame_range.strip()
