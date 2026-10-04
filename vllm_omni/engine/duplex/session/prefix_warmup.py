# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Prefill a session's pending camera frames before its turn commits.

While the user is still speaking, the next turn's prompt is known up to the
utterance itself: instructions, earlier turns and the camera frames sent since
the last answer. ``PrefixWarmer`` has the model plugin render that prefix and
prefills it on Stage0 alone, so with prefix caching the committed turn only
computes its audio.

At most one warmup runs per session. It follows the conversation: a newer
state replaces it, and it is cancelled (which aborts its request) as soon as a
turn starts, a response is still playing, or the session closes.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.logger import init_logger

from vllm_omni.engine.duplex.config import DuplexSessionState
from vllm_omni.engine.duplex.contracts import duplex_resource_request_id
from vllm_omni.engine.duplex.session import helpers

if TYPE_CHECKING:
    from vllm_omni.engine.duplex.session.context import DuplexSessionContext

logger = init_logger(__name__)


def pending_turn_has_images(history: Sequence[dict[str, object]]) -> bool:
    """Whether the unanswered user items carry an image and no committed audio yet."""
    has_image = False
    for message in reversed(history):
        if message.get("role") != "user":
            break
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            part_type = part.get("type") if isinstance(part, dict) else None
            if part_type == "audio_url":
                return False
            has_image = has_image or part_type == "image_url"
    return has_image


class PrefixWarmer:
    """Keeps at most one Stage0 prefix warmup in step with one session's conversation."""

    def __init__(self, ctx: DuplexSessionContext, *, enabled: bool) -> None:
        self._ctx = ctx
        self._enabled = enabled
        #: Conversation state of the warmup in flight or last completed.
        self._signature: tuple[object, ...] | None = None
        self._task: asyncio.Future[None] | None = None

    def refresh(self) -> None:
        """Start, keep or cancel the warmup so it matches the session as it is now."""
        if not self._enabled:
            return
        signature = self._eligible_signature()
        if signature is None:
            self.cancel()
            return
        if signature == self._signature:
            return
        self.cancel()
        self._signature = signature
        self._task = self._ctx.services.spawn(self._warm(), name=f"duplex-prefix-warmup-{self._ctx.session.session_id}")

    def cancel(self) -> None:
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            self._signature = None

    def _eligible_signature(self) -> tuple[object, ...] | None:
        ctx = self._ctx
        session = ctx.session
        if ctx.run.closing or session.state != DuplexSessionState.OPEN:
            return None
        if (
            helpers.response_in_progress(session, ctx.tasks)
            or ctx.tasks.append_tasks
            or ctx.run.stream_request_id is not None
            or ctx.model_state.committed_audio_payload is not None
        ):
            return None
        history = session.history
        if not pending_turn_has_images(history):
            return None
        # Content objects are shared by the history copies, so comparing them is mostly identity checks.
        return (session.config_generation, *((message.get("role"), message.get("content")) for message in history))

    async def _warm(self) -> None:
        ctx = self._ctx
        session = ctx.session
        request_id = duplex_resource_request_id(session.fence, f"warmup-{uuid.uuid4().hex[:12]}")
        try:
            plan = await ctx.plugin.prepare_prefix_warmup_plan(
                request_id=request_id,
                session_config={**session.config.as_dict(), "conversation": list(session.history)},
                runtime_config=dict(session.runtime_config),
                state=ctx.model_state,
            )
            if plan is None:
                return
            started = time.monotonic()
            finished = await ctx.stage_port.run_prefix_warmup(
                request_id=request_id,
                session_id=session.session_id,
                prompt=plan.prompt,
                sampling_params=ctx.manager.sampling_params_for(session)[0],
            )
            logger.debug(
                "duplex prefix warmup session=%s request=%s finished=%s elapsed_ms=%.1f",
                session.session_id,
                request_id,
                finished,
                (time.monotonic() - started) * 1000,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("duplex prefix warmup failed session=%s", session.session_id, exc_info=True)
