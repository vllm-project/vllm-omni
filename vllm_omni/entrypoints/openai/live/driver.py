# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``ResumableRequestDriver``: one resumable ``AsyncOmni`` request per session.

The engine consumes an async generator of ``StreamingInput``: the first chunk
opens a resumable request, every later chunk is a streaming update appended
after the previous segment stopped. The driver feeds that generator from a
queue and forwards every stage output to the handler. A resubmission aborts
the request (``aclose`` on the output generator) and opens a new epoch with a
fully re-rendered prompt.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from typing import Any

from vllm.engine.protocol import StreamingInput
from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind, SamplingParams

logger = init_logger(__name__)

OutputCallback = Callable[[int, Any], Awaitable[None]]
ErrorCallback = Callable[[int, BaseException], Awaitable[None]]


def streaming_params(params: Sequence[Any]) -> list[Any]:
    """Clone the per-stage defaults with DELTA outputs for streaming input."""
    result = []
    for sp in params:
        if isinstance(sp, SamplingParams):
            sp = sp.clone()
            sp.output_kind = RequestOutputKind.DELTA
            if sp.stop:
                raise ValueError("Live sessions do not support stop strings; use stop_token_ids")
            if sp.n != 1:
                raise ValueError("Live sessions require n == 1")
        result.append(sp)
    return result


def finish_reason_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value)).lower()


class ResumableRequestDriver:
    def __init__(
        self,
        engine: Any,
        session_id: str,
        sampling_params_list: Sequence[Any],
        on_output: OutputCallback,
        on_error: ErrorCallback,
    ) -> None:
        self._engine = engine
        self._session_id = session_id
        self.sampling_params_list = streaming_params(sampling_params_list)
        self._on_output = on_output
        self._on_error = on_error
        self.epoch = 0
        self._queue: asyncio.Queue[StreamingInput | None] | None = None
        self._task: asyncio.Task | None = None
        # Stage-0 bookkeeping of the current epoch.
        self.last_sampled_token: int | None = None

    @property
    def active(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def stage0_params(self) -> SamplingParams:
        return self.sampling_params_list[0]

    async def start(self, prompt: dict[str, Any], stage0_params: SamplingParams | None) -> int:
        """Open a new epoch with ``prompt`` as its first chunk; returns the epoch."""
        await self.abort()
        self.epoch += 1
        self.last_sampled_token = None
        self._queue = asyncio.Queue()
        logger.debug(
            "Live %s epoch %d start: %d prompt tokens",
            self._session_id,
            self.epoch,
            len(prompt.get("prompt_token_ids") or ()),
        )
        self._queue.put_nowait(StreamingInput(prompt=prompt, sampling_params=stage0_params))
        self._task = asyncio.create_task(self._run(self.epoch, self._queue), name=f"live-driver-{self.epoch}")
        return self.epoch

    def append(self, prompt: dict[str, Any], stage0_params: SamplingParams | None) -> None:
        if not self.active or self._queue is None:
            raise RuntimeError("no active resumable request")
        logger.debug("Live %s append: %d prompt tokens", self._session_id, len(prompt.get("prompt_token_ids") or ()))
        self._queue.put_nowait(StreamingInput(prompt=prompt, sampling_params=stage0_params))

    async def abort(self) -> None:
        """Abort the in-flight request (engine-side abort via ``aclose``)."""
        task, self._task, self._queue = self._task, None, None
        if task is None or task is asyncio.current_task():
            # Called from the request's own output/error callback: the task
            # is already unwinding and closes its generator in ``finally``.
            return
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    async def close(self) -> None:
        await self.abort()

    async def _inputs(self, queue: asyncio.Queue[StreamingInput | None]) -> AsyncGenerator[StreamingInput, None]:
        while True:
            item = await queue.get()
            if item is None:
                return
            yield item

    def _track(self, output: Any) -> None:
        if getattr(output, "stage_id", None) != 0 or not getattr(output, "outputs", None):
            return
        token_ids = list(getattr(output.outputs[0], "token_ids", None) or ())
        if token_ids:
            self.last_sampled_token = token_ids[-1]

    async def _run(self, epoch: int, queue: asyncio.Queue[StreamingInput | None]) -> None:
        request_id = f"live-{self._session_id}-e{epoch}"
        result_gen = self._engine.generate(
            prompt=self._inputs(queue),
            request_id=request_id,
            sampling_params_list=self.sampling_params_list,
        )
        try:
            async for output in result_gen:
                if logger.isEnabledFor(logging.DEBUG):
                    completions = getattr(output, "outputs", None) or ()
                    first = completions[0] if completions else None
                    logger.debug(
                        "Live %s output: stage=%s type=%s tokens=%s finish=%s finished=%s segment=%s mm=%s",
                        request_id,
                        getattr(output, "stage_id", None),
                        getattr(output, "final_output_type", None),
                        list(getattr(first, "token_ids", None) or ()),
                        getattr(first, "finish_reason", None),
                        getattr(output, "finished", None),
                        getattr(output, "segment_finished", None),
                        sorted((getattr(output, "multimodal_output", None) or {}).keys()),
                    )
                self._track(output)
                await self._on_output(epoch, output)
            logger.info("Live request %s ended", request_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Live request %s ended with an error: %s", request_id, exc)
            await self._on_error(epoch, exc)
        finally:
            # Runs AsyncOmni.generate's cleanup now: input-pump cancellation
            # and a shielded engine-side abort.
            with contextlib.suppress(Exception):
                await result_gen.aclose()
