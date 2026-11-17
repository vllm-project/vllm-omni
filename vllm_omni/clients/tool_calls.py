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
        #: Every running tool, including a settled one still finishing its
        #: cancellation cleanup, so ``run()`` waits for that cleanup too.
        self._tasks: dict[str, asyncio.Task[None]] = {}
        #: Running calls the server ended. Anything sent for one would be refused.
        self._settled: set[str] = set()
        self.undelivered_results: dict[str, str] = {}
        self.delivery_errors: dict[str, DuplexClientError] = {}
        self.pending_continuations: set[str] = set()
        self.continuation_errors: dict[str, DuplexClientError] = {}
        self._active_responses: set[str] = set()
        self._playback: dict[str, tuple[float, float]] = {}
        self._ready = asyncio.Event()
        self._ready.set()
        self._continuation_lock = asyncio.Lock()
        self._awaiting_response = False

    async def run(self) -> None:
        """Handle calls until the session ends; running tools are then cancelled.

        Returns only after every tool has finished, including the cancellation
        cleanup of one whose call already settled.
        """
        try:
            async for event in self._client.events():
                self.handle_event(event)
        finally:
            tasks = dict(self._tasks)
            self._tasks.clear()
            for call_id, task in tasks.items():
                # A settled tool was already cancelled; a second cancel would
                # interrupt its cleanup.
                if call_id not in self._settled:
                    task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)

    def handle_event(self, event: DuplexEvent) -> None:
        """Start a tool for a new call, or stop one whose call has ended."""
        response_id = event.response_id
        if event.type == "response.created" and response_id:
            self._awaiting_response = False
            self._active_responses.add(response_id)
        elif event.type == "response.done" and response_id:
            self._active_responses.discard(response_id)
            response = event.raw.get("response", {})
            metadata = response.get("metadata", {}) if isinstance(response, dict) else {}
            if isinstance(response, dict) and response.get("status") in {"cancelled", "failed"}:
                self._playback.pop(response_id, None)
            else:
                self._note_playback(response_id, metadata)
        elif event.type == "playback.acknowledged":
            details = event.raw.get("event", {})
            item_id = details.get("item_id") if isinstance(details, dict) else None
            rid = details.get("response_id") if isinstance(details, dict) else None
            if not rid and isinstance(item_id, str) and item_id.startswith("item_"):
                rid = item_id[5:]
            if isinstance(rid, str):
                self._note_playback(rid, details)
        elif event.type == "output_audio_buffer.cleared" and response_id:
            self._playback.pop(response_id, None)
        self._update_readiness()
        if event.type == "response.output_item.done":
            item = event.raw.get("item")
            if isinstance(item, dict) and item.get("type") == "function_call":
                self._start(item)
        elif event.type == "duplex.function_call.settled":
            details = event.raw.get("event")
            call_id = details.get("call_id") if isinstance(details, dict) else None
            if not isinstance(call_id, str):
                return
            self.undelivered_results.pop(call_id, None)
            self.delivery_errors.pop(call_id, None)
            self.pending_continuations.discard(call_id)
            self.continuation_errors.pop(call_id, None)
            task = self._tasks.get(call_id)
            if task is not None and call_id not in self._settled:
                # The task stays in ``_tasks`` until it is done; ``_forget``
                # then removes it.
                self._settled.add(call_id)
                task.cancel()

    def _start(self, item: Mapping[str, object]) -> None:
        call_id = item.get("call_id")
        if (
            not isinstance(call_id, str)
            or not call_id
            or call_id in self._tasks
            or call_id in self.undelivered_results
            or call_id in self.pending_continuations
        ):
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
        await self._deliver_result(call_id, output)

    def _note_playback(self, response_id: str, details: object) -> None:
        if self._client.session_info.get("playback_commit_policy") != "ack_only" or not isinstance(details, dict):
            return
        playback = details.get("playback")
        if isinstance(playback, dict):
            sent, committed = self._playback.get(response_id, (0.0, 0.0))
            self._playback[response_id] = (
                max(sent, float(playback.get("sent_ms", 0))),
                max(committed, float(playback.get("committed_ms", 0))),
            )
            sent, committed = self._playback[response_id]
            if response_id not in self._active_responses and committed >= sent:
                self._playback.pop(response_id)

    async def retry_result(self, call_id: str) -> bool:
        """Resend an undelivered result after reconnecting, without rerunning its tool.

        Await this in an application task while ``run()`` consumes lifecycle events.
        A successful write is not a server acceptance acknowledgement.
        """
        return await self._retry_update(
            call_id, lambda: self._deliver_result(call_id, self.undelivered_results[call_id])
        )

    async def retry_continuation(self, call_id: str) -> bool:
        """Retry a pending follow-up without rerunning its handler or resending output.

        Await this while ``run()`` consumes lifecycle events. A failed write may
        have reached the server; the application decides when recovery is safe.
        """
        if call_id not in self.pending_continuations:
            raise KeyError(call_id)
        return await self._retry_update(call_id, lambda: self._continue(call_id))

    async def _retry_update(self, call_id: str, operation: Callable[[], Awaitable[bool]]) -> bool:
        if call_id in self._tasks:
            raise RuntimeError("The tool task has not finished yet")
        succeeded = False

        async def deliver() -> None:
            nonlocal succeeded
            succeeded = await operation()

        task = asyncio.create_task(deliver())
        self._tasks[call_id] = task
        task.add_done_callback(lambda done: self._forget(call_id, done))
        await task
        return succeeded

    async def _deliver_result(self, call_id: str, output: str) -> bool:
        self.undelivered_results[call_id] = output
        try:
            await self._client.send_function_call_output(call_id, output)
        except DuplexClientError as exc:
            self.delivery_errors[call_id] = exc
            logger.warning("Tool result %s was not delivered: %s", call_id, exc)
            return False
        self.undelivered_results.pop(call_id, None)
        self.delivery_errors.pop(call_id, None)
        if call_id in self._settled:
            return False
        capabilities = self._client.session_info.get("capabilities")
        if (
            isinstance(capabilities, dict)
            and capabilities.get("function_call_output_continuation") == "client_response_create"
        ):
            self.pending_continuations.add(call_id)
            return await self._continue(call_id)
        return True

    def _update_readiness(self) -> None:
        if (
            self._awaiting_response
            or self._active_responses
            or any(sent > committed for sent, committed in self._playback.values())
        ):
            self._ready.clear()
        else:
            self._ready.set()

    async def _continue(self, call_id: str) -> bool:
        async with self._continuation_lock:
            # A prior follow-up may already cover this result.
            if call_id not in self.pending_continuations:
                return True
            await self._ready.wait()
            if call_id in self._settled:
                return False
            covered = set(self.pending_continuations)
            # The reader can observe created/done before send() returns. Its
            # lifecycle updates must win over our pre-send reservation.
            self._awaiting_response = True
            self._ready.clear()
            written = False
            try:
                await self._client.send({"type": "response.create"})
                written = True
            except DuplexClientError as exc:
                for pending in covered & self.pending_continuations:
                    self.continuation_errors[pending] = exc
                logger.warning("Tool continuation was not delivered: %s", exc)
                return False
            finally:
                # On failure/cancellation roll back only the reservation;
                # observed active responses and playback remain authoritative.
                if not written:
                    self._awaiting_response = False
                    self._update_readiness()
            self.pending_continuations.difference_update(covered)
            for pending in covered:
                self.continuation_errors.pop(pending, None)
            return True

    @staticmethod
    async def _finish(send: Awaitable[object]) -> bool:
        """Log failures of best-effort cancellation updates."""
        try:
            await send
        except DuplexClientError as exc:
            logger.warning("Could not send a tool-call update: %s", exc)
            return False
        return True


def main() -> None:
    """Run application-supplied async tool handlers from the command line."""
    import argparse
    from importlib import import_module
    from pathlib import Path

    from vllm_omni.clients.duplex import DuplexClient, SessionConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tools-json", required=True, help="JSON file containing a list of tool declarations")
    parser.add_argument("--handler", action="append", required=True, help="NAME=MODULE:ASYNC_FUNCTION; repeat per tool")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--timeout", type=float, default=60, help="Seconds to consume events before closing")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    tools = json.loads(Path(args.tools_json).read_text(encoding="utf-8"))
    if not isinstance(tools, list):
        parser.error("--tools-json must contain a JSON list")
    handlers = {}
    for spec in args.handler:
        name, target = spec.split("=", 1)
        module, function = target.split(":", 1)
        handlers[name] = getattr(import_module(module), function)

    async def serve() -> None:
        config = SessionConfig(modalities=("text",), auto_response=False, extra_body=function_call_session_defaults())
        async with DuplexClient(args.url, model=args.model, config=config) as client:
            runner = DuplexToolRunner(client, handlers)

            async def print_events() -> None:
                async for event in client.events():
                    print(json.dumps(event.raw), flush=True)

            printer = asyncio.create_task(print_events())
            running = asyncio.create_task(runner.run())
            try:
                await asyncio.sleep(0)
                await client.send({"type": "session.update", "session": {"tools": tools}})
                await client.send(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": args.prompt}],
                        },
                    }
                )
                await client.send({"type": "response.create"})
                try:
                    await asyncio.wait_for(running, args.timeout)
                except TimeoutError:
                    pass
            finally:
                running.cancel()
                printer.cancel()
                await asyncio.gather(running, printer, return_exceptions=True)

    asyncio.run(serve())


if __name__ == "__main__":
    main()
