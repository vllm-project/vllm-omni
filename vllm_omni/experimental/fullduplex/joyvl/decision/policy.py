# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from vllm_omni.experimental.fullduplex.joyvl.bridges.delegation import DelegationBridge
from vllm_omni.experimental.fullduplex.joyvl.decision.output_parser import Action, ParsedAction, parse_action
from vllm_omni.experimental.fullduplex.joyvl.decision.prompts import SYSTEM_PROMPTS, USER_QUERY_HEADER
from vllm_omni.experimental.fullduplex.joyvl.memory.brain import InteractionBrain
from vllm_omni.experimental.fullduplex.joyvl.memory.memory import Summarizer


def sample_frames(frames: list[str], num_frames: int) -> list[str]:
    n = len(frames)
    if num_frames <= 0 or n <= num_frames:
        return list(frames)
    stride = max(1, n // num_frames)
    idx = [i * stride for i in range(num_frames - 1)] + [n - 1]
    return [frames[i] for i in idx]


class JoyVLPolicy:
    def __init__(
        self,
        *,
        persona: str = "default",
        system_prompt: str | None = None,
        num_frames: int = 4,
        summarizer: Summarizer | None = None,
        delegation: DelegationBridge | None = None,
        chunk_frames: int = 16,
        long_term_every_n_chunks: int = 5,
        long_term_window: int = 15,
        keep_qa_history: bool = True,
        frame_seconds: float = 1.0,
        enable_delegation: bool = True,
        response_dedup_threshold: float = 1.0,
    ) -> None:
        self.response_dedup_threshold = response_dedup_threshold
        self.brain = InteractionBrain(
            summarizer=summarizer,
            delegation=delegation,
            chunk_frames=chunk_frames,
            long_term_every_n_chunks=long_term_every_n_chunks,
            long_term_window=long_term_window,
            keep_qa_history=keep_qa_history,
            frame_seconds=frame_seconds,
            enable_delegation=enable_delegation,
        )
        self.system_prompt = system_prompt or SYSTEM_PROMPTS.get(persona, SYSTEM_PROMPTS["default"])
        self.num_frames = num_frames

    def tick(self, n: int = 1) -> None:
        self.brain.tick(n)

    def observe(self, time_range: str, data_url: str) -> None:
        self.brain.observe(time_range, data_url)

    @property
    def working_frames(self) -> list[tuple[str, str]]:
        return self.brain.working_frames

    def take_working_frames(self) -> list[tuple[str, str]]:
        return self.brain.take_working_frames()

    def set_query(self, query: str | None) -> bool:
        return self.brain.update_query(query)

    def should_respond(self) -> bool:
        return True

    def user_content(self, frame_parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        prefix = self.brain.build_prefix()
        if prefix:
            content.append({"type": "text", "text": prefix})
        content.extend(frame_parts)
        if self.brain.current_query and self.brain.query_in_current_chunk:
            content.append({"type": "text", "text": f"{USER_QUERY_HEADER}\n{self.brain.current_query}"})
        return content

    def build_messages(self, frame_parts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        user = {"role": "user", "content": self.user_content(frame_parts)}
        return [{"role": "system", "content": self.system_prompt}, user], user

    def _is_repeat(self, text: str) -> bool:
        last = self.brain.last_response_text
        if not last:
            return False
        if text == last:
            return True
        if self.response_dedup_threshold >= 1.0:
            return False
        similarity = SequenceMatcher(None, text.strip().lower(), last.strip().lower()).ratio()
        return similarity >= self.response_dedup_threshold

    def commit(self, response_text: str) -> ParsedAction:
        action = parse_action(response_text)
        if action.action is Action.SILENCE:
            self.brain.last_response_text = ""
            return action
        if action.action is Action.RESPONSE and action.text and self._is_repeat(action.text):
            return ParsedAction(Action.SILENCE, raw="</silence>")
        self.brain.last_response_text = action.text
        if self.brain.current_query:
            self.brain.record_response(action.text)
        return action

    async def fold_delegations(self) -> dict[str, Any] | None:
        return await self.brain.fold_delegations()

    async def submit_if_delegate(
        self, action: ParsedAction, frames: list[tuple[str, str]] | None = None
    ) -> dict | None:
        return await self.brain.submit_delegation(action, frames or [])

    def needs_flush(self) -> bool:
        return self.brain.should_flush()

    def close_chunk(self) -> int:
        return self.brain.close_chunk()

    async def consolidate(self, chunk_index: int, frames: list[tuple[str, str]] | None = None) -> None:
        await self.brain.consolidate(chunk_index, frames or [])

    async def flush(self, frames: list[tuple[str, str]] | None = None) -> None:
        await self.brain.flush(frames or [])
