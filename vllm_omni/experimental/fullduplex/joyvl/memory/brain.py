# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
from typing import Any

from vllm_omni.experimental.fullduplex.joyvl.bridges.delegation import DelegationBridge
from vllm_omni.experimental.fullduplex.joyvl.decision.output_parser import Action, ParsedAction
from vllm_omni.experimental.fullduplex.joyvl.memory.memory import (
    MidTermSummary,
    QAEntry,
    SessionMemory,
    Summarizer,
    build_memory_prefix,
)


class InteractionBrain:
    def __init__(
        self,
        *,
        summarizer: Summarizer | None = None,
        delegation: DelegationBridge | None = None,
        chunk_frames: int = 16,
        long_term_every_n_chunks: int = 5,
        long_term_window: int = 15,
        keep_qa_history: bool = True,
        frame_seconds: float = 1.0,
        enable_delegation: bool = True,
    ) -> None:
        self._summarizer = summarizer
        self._delegation = delegation
        self._chunk_frames = chunk_frames
        self._long_term_every_n = long_term_every_n_chunks
        self._long_term_window = long_term_window
        self._keep_qa_history = keep_qa_history
        self._frame_seconds = frame_seconds
        self._enable_delegation = enable_delegation
        self.reset()

    def reset(self) -> None:
        # Cancel any in-flight/finished-but-unpolled delegations so the shared bridge
        # does not accumulate abandoned tasks when a session is reset or evicted.
        if self._delegation is not None:
            for task in getattr(self, "_pending_delegations", []):
                self._delegation.cancel(task["task_id"])
        self.memory = SessionMemory()
        self.current_query: str | None = None
        self.query_time: str | None = None
        self.query_in_current_chunk = False
        self.frame_index = 0
        self.chunk_index = 1
        self._chunk_frame_count = 0
        self.working_frames: list[tuple[str, str]] = []
        self.last_response_text = ""
        self.response_records: list[tuple[str, str]] = []
        self._pending_delegations: list[dict[str, str]] = []
        self._consolidate_lock = asyncio.Lock()

    def now(self) -> str:
        return f"{self.frame_index * self._frame_seconds:.1f} seconds"

    def last_frame_time(self) -> str:
        return f"{max(0, self.frame_index - 1) * self._frame_seconds:.1f} seconds"

    def tick(self, n: int = 1) -> None:
        self.frame_index += n
        self._chunk_frame_count += n

    def observe(self, time_range: str, data_url: str) -> None:
        self.tick()
        self.working_frames.append((time_range, data_url))

    def take_working_frames(self) -> list[tuple[str, str]]:
        frames = self.working_frames
        self.working_frames = []
        return frames

    def should_flush(self) -> bool:
        return self._chunk_frames > 0 and self._chunk_frame_count >= self._chunk_frames

    def update_query(self, query: str | None) -> bool:
        q = (query or "").strip()
        if not q or q == self.current_query:
            return False
        if self.current_query is not None:
            self.archive_query()
        self.current_query = q
        self.query_time = self.now()
        self.query_in_current_chunk = True
        return True

    def archive_query(self) -> None:
        if self.current_query and self.response_records:
            self.memory.qa_history.append(
                QAEntry(
                    self.current_query,
                    self.query_time or "",
                    list(self.response_records),
                    self.chunk_index,
                )
            )
        self.response_records = []

    def record_response(self, text: str) -> None:
        if text and self.current_query:
            self.response_records.append((self.last_frame_time(), text))

    def build_prefix(self) -> str:
        return build_memory_prefix(
            self.memory,
            current_query=self.current_query,
            query_in_current_chunk=self.query_in_current_chunk,
            keep_qa_history=self._keep_qa_history,
            current_chunk_index=self.chunk_index,
        )

    def close_chunk(self) -> int:
        self.archive_query()
        closed = self.chunk_index
        self.chunk_index += 1
        self._chunk_frame_count = 0
        self.query_in_current_chunk = False
        return closed

    async def consolidate(self, chunk_index: int, frames: list[tuple[str, str]]) -> None:
        if self._summarizer is None or not frames:
            return
        frame_range = f"{frames[0][0]}-{frames[-1][0]}"
        async with self._consolidate_lock:
            summary = await self._summarizer.summarize_chunk(chunk_index, frame_range, frames)
            self.memory.mid_term_summaries.append(MidTermSummary(chunk_index, frame_range, summary))
            if len(self.memory.mid_term_summaries) >= self._long_term_every_n:
                block = await self._summarizer.compress_to_long_term(self.memory.mid_term_summaries)
                if block:
                    self.memory.long_term_blocks.append(block)
                    window = self._long_term_window
                    if window > 0 and len(self.memory.long_term_blocks) > window:
                        # drop oldest blocks so long-term memory stays bounded over hours
                        del self.memory.long_term_blocks[: len(self.memory.long_term_blocks) - window]
                    self.memory.long_term_memory = "\n\n".join(self.memory.long_term_blocks)
                self.memory.mid_term_summaries.clear()

    async def flush(self, frames: list[tuple[str, str]]) -> None:
        await self.consolidate(self.close_chunk(), frames)

    async def submit_delegation(self, action: ParsedAction, frames: list[tuple[str, str]]) -> dict[str, Any] | None:
        if action.action is not Action.DELEGATE or self._delegation is None or not self._enable_delegation:
            return None
        task_id = await self._delegation.submit(action.delegated_question or "", action.text, frames)
        self._pending_delegations.append({"task_id": task_id, "question": action.delegated_question or ""})
        return {"task_id": task_id, "status": "submitted", "question": action.delegated_question}

    async def fold_delegations(self) -> dict[str, Any] | None:
        if not self._pending_delegations or self._delegation is None:
            return None
        folded: dict[str, Any] | None = None
        still_pending: list[dict[str, str]] = []
        for task in self._pending_delegations:
            result = await self._delegation.poll(task["task_id"])
            if result.is_ready:
                self.memory.qa_history.append(
                    QAEntry(
                        f"[delegated] {task['question']}",
                        self.query_time or "",
                        [("", result.digest)],
                        self.chunk_index,
                    )
                )
                folded = {
                    "task_id": result.task_id,
                    "status": "ready",
                    "question": task["question"],
                    "digest": result.digest,
                }
                if result.media:
                    folded["media"] = result.media
            elif result.status == "error":
                folded = {"task_id": result.task_id, "status": "error", "question": task["question"]}
            else:
                still_pending.append(task)
        self._pending_delegations = still_pending
        return folded
