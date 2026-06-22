# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from vllm_omni.experimental.fullduplex.joyvl.bridges.backend import ModelBackend
from vllm_omni.experimental.fullduplex.joyvl.bridges.delegation import DelegationBridge
from vllm_omni.experimental.fullduplex.joyvl.decision.output_parser import Action, ParsedAction
from vllm_omni.experimental.fullduplex.joyvl.decision.policy import JoyVLPolicy
from vllm_omni.experimental.fullduplex.joyvl.decision.prompts import SYSTEM_PROMPTS, USER_QUERY_HEADER
from vllm_omni.experimental.fullduplex.joyvl.memory.memory import Summarizer, WorkingChunk
from vllm_omni.experimental.fullduplex.joyvl.serving.config import InteractionConfig

logger = logging.getLogger("vllm_omni.experimental.fullduplex.joyvl")


@dataclass
class StepResult:
    action: ParsedAction
    chunk_index: int
    frame_index: int
    inference_skipped: bool
    latency_ms: float
    long_term_memory: str
    mid_term_summaries: list[dict[str, Any]] = field(default_factory=list)
    delegation: dict[str, Any] | None = None


class InteractionSession:
    def __init__(
        self,
        session_id: str,
        config: InteractionConfig,
        backend: ModelBackend,
        summarizer: Summarizer | None = None,
        delegation: DelegationBridge | None = None,
    ) -> None:
        self.session_id = session_id
        self.config = config
        self._backend = backend
        self._policy = JoyVLPolicy(
            persona=config.persona,
            summarizer=summarizer,
            delegation=delegation,
            chunk_frames=config.chunk_frames,
            long_term_every_n_chunks=config.long_term_every_n_chunks,
            keep_qa_history=config.keep_qa_history,
            frame_seconds=config.frame_seconds,
            enable_delegation=config.enable_delegation,
            response_dedup_threshold=config.response_dedup_threshold,
        )
        self.chunk = WorkingChunk()
        self._system_prompt = config.system_prompt
        self.last_access = time.monotonic()
        self._consolidating: set[asyncio.Task] = set()

    def set_persona(self, persona: str) -> bool:
        prompt = SYSTEM_PROMPTS.get(persona)
        if prompt is None:
            return False
        self._system_prompt = prompt
        return True

    async def step(self, frames: list[str], query: str | None = None, t: float | None = None) -> StepResult:
        self.last_access = time.monotonic()
        started = time.perf_counter()
        policy = self._policy
        brain = policy.brain

        base = t if t is not None else brain.frame_index * self.config.frame_seconds
        time_ranges = [f"{base + i * self.config.frame_seconds:.1f}s" for i in range(len(frames))]

        delegation_info = await policy.fold_delegations()

        if policy.needs_flush():
            self._spawn_consolidation(policy.close_chunk(), policy.take_working_frames())
            self.chunk = WorkingChunk()

        query_is_fresh = policy.set_query(query)

        for tr, url in zip(time_ranges, frames):
            policy.observe(tr, url)
        self.chunk.messages.append(self._frame_message(time_ranges, frames, include_query=query_is_fresh))

        if self.config.force_silence_before_query and not brain.current_query:
            action = ParsedAction(Action.SILENCE, raw="</silence>")
            skipped = True
        else:
            action = policy.commit(await self._infer())
            skipped = False
        self.chunk.messages.append({"role": "assistant", "content": action.raw or "</silence>"})

        submitted = await policy.submit_if_delegate(action, list(policy.working_frames))
        if submitted:
            delegation_info = submitted

        return StepResult(
            action=action,
            chunk_index=brain.chunk_index,
            frame_index=brain.frame_index,
            inference_skipped=skipped,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            long_term_memory=brain.memory.long_term_memory,
            mid_term_summaries=[
                {"chunk_index": m.chunk_index, "frame_range": m.frame_range, "summary_text": m.summary_text}
                for m in brain.memory.mid_term_summaries
            ],
            delegation=delegation_info,
        )

    async def reset(self) -> None:
        await self._cancel_consolidation()
        self._policy.brain.reset()
        self.chunk = WorkingChunk()

    async def aclose(self) -> None:
        await self._cancel_consolidation()

    async def _cancel_consolidation(self) -> None:
        tasks = list(self._consolidating)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._consolidating.clear()

    def _spawn_consolidation(self, chunk_index: int, frames: list[tuple[str, str]]) -> None:
        task = asyncio.create_task(self._policy.consolidate(chunk_index, frames))
        self._consolidating.add(task)
        task.add_done_callback(self._on_consolidation_done)

    def _on_consolidation_done(self, task: asyncio.Task) -> None:
        self._consolidating.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.warning("memory consolidation failed for session %s: %s", self.session_id, task.exception())

    async def _infer(self) -> str:
        s = self.config.sampling
        extra_body = {
            "skip_special_tokens": False,
            "greedy": False,
            "top_k": s.top_k,
            "repetition_penalty": s.repetition_penalty,
            "presence_penalty": s.presence_penalty,
        }
        raw, _ = await self._backend.generate(
            self._build_api_messages(),
            max_tokens=s.max_tokens,
            temperature=s.temperature,
            top_p=s.top_p,
            extra_body=extra_body,
        )
        return raw

    def _build_api_messages(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": self._system_prompt}]
        chunk_messages = [dict(m) for m in self.chunk.messages]
        prefix = self._policy.brain.build_prefix()
        if prefix and chunk_messages:
            head = chunk_messages[0]
            head["content"] = [{"type": "text", "text": prefix}] + list(head["content"])
        messages.extend(chunk_messages)
        return messages

    def _frame_message(self, time_ranges: list[str], frames: list[str], include_query: bool) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        query = self._policy.brain.current_query
        if include_query and query:
            content.append({"type": "text", "text": f"{USER_QUERY_HEADER}\n{query.strip()}"})
        for tr, url in zip(time_ranges, frames):
            content.append({"type": "text", "text": f"<{tr}>"})
            content.append({"type": "image_url", "image_url": {"url": url}})
        return {"role": "user", "content": content}
