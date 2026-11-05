# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run an application's tools for one duplex session.

This module depends on ``DuplexClientBase``, never the reverse: an application
can read the same events and call ``send_function_call_output`` / ``cancel_function_call``
itself. The server never executes a tool; each handler here runs in its own
task, so the application keeps streaming audio while a tool works.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping

from vllm.logger import init_logger

from vllm_omni.clients.duplex import DuplexClientBase, DuplexClientError, DuplexEvent

__all__ = ["DuplexToolRunner", "ToolHandler", "function_call_session_defaults"]

logger = init_logger(__name__)

#: Takes the decoded call arguments and returns the output the model will see.
ToolHandler = Callable[[dict[str, object]], Awaitable[str]]


def function_call_session_defaults() -> dict[str, object]:
    """``extra_body`` for a session whose pending tools survive an interruption.

    Pass it in ``SessionConfig``: the policy is fixed when the session opens, so
    a runner attached later cannot set it.
    """
    return {"function_call_interrupt_policy": "keep"}


class DuplexToolRunner:
    """Execute function calls from one client's event stream."""

    def __init__(self, client: DuplexClientBase, handlers: Mapping[str, ToolHandler]) -> None:
        self._client = client
        self._handlers = dict(handlers)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        #: Running calls the server ended. Anything sent for one would be refused.
        self._settled: set[str] = set()

    async def run(self) -> None:
        """Handle calls until the session ends; running tools are then cancelled."""
        try:
            async for event in self._client.events():
                self.handle_event(event)
        finally:
            tasks = list(self._tasks.values())
            self._tasks.clear()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def handle_event(self, event: DuplexEvent) -> None:
        """Start a tool for a new call, or stop one whose call has ended."""
        if event.type == "response.output_item.done":
            item = event.raw.get("item")
            if isinstance(item, dict) and item.get("type") == "function_call":
                self._start(item)
        elif event.type == "duplex.function_call.settled":
            details = event.raw.get("event")
            call_id = details.get("call_id") if isinstance(details, dict) else None
            if not isinstance(call_id, str):
                return
            task = self._tasks.pop(call_id, None)
            if task is not None:
                self._settled.add(call_id)
                task.cancel()

    def _start(self, item: Mapping[str, object]) -> None:
        call_id = item.get("call_id")
        if not isinstance(call_id, str) or not call_id or call_id in self._tasks:
            return
        task = asyncio.create_task(self._run_call(call_id, item.get("name"), item.get("arguments")))
        self._tasks[call_id] = task
        task.add_done_callback(lambda done: self._forget(call_id, done))

    def _forget(self, call_id: str, task: asyncio.Task[None]) -> None:
        self._settled.discard(call_id)
        if self._tasks.get(call_id) is task:
            del self._tasks[call_id]

    async def _run_call(self, call_id: str, name: object, arguments: object) -> None:
        try:
            handler = self._handlers.get(name) if isinstance(name, str) else None
            if handler is None:
                raise LookupError(f"no handler for tool {name!r}")
            decoded = json.loads(arguments) if isinstance(arguments, str) and arguments else {}
            if not isinstance(decoded, dict):
                raise TypeError("tool arguments must be a JSON object")
            output = await handler(decoded)
            if not isinstance(output, str):
                raise TypeError(f"tool {name!r} must return a string, got {type(output).__name__}")
        except asyncio.CancelledError:
            raise
        except Exception:
            # Report the failure, not its text: a failed call's output must not
            # reach the model.
            logger.exception("Tool call %s failed", call_id)
            if call_id not in self._settled:
                await self._finish(self._client.cancel_function_call(call_id, reason="failed"))
            return
        if call_id in self._settled:
            return
        await self._finish(self._client.send_function_call_output(call_id, output))
        if call_id in self._settled:
            return
        capabilities = self._client.session_info.get("capabilities")
        if (
            isinstance(capabilities, dict)
            and capabilities.get("function_call_output_continuation") == "client_response_create"
        ):
            await self._finish(self._client.send({"type": "response.create"}))

    @staticmethod
    async def _finish(send: Awaitable[object]) -> None:
        """A closed session has already ended the call, so a lost send is final."""
        try:
            await send
        except DuplexClientError as exc:
            logger.warning("Could not send a tool-call update: %s", exc)
