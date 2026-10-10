# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise the Lychee binding through the real Unified Duplex session core."""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
from dataclasses import dataclass, field
from typing import Any

from vllm.sampling_params import SamplingParams

from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import (
    DuplexStagePort,
    DuplexStageRequestContext,
    DuplexStageSubmission,
    DuplexStageSubmissionResult,
)
from vllm_omni.engine.duplex.delivery import DuplexOutputBuffer
from vllm_omni.engine.duplex.events import DuplexEvent
from vllm_omni.engine.duplex.messages import (
    CloseDuplexSessionMessage,
    DuplexControlResultMessage,
    DuplexSessionCommandMessage,
    DuplexSessionEventMessage,
    OpenDuplexSessionMessage,
)
from vllm_omni.engine.duplex.session.manager import DuplexSessionManager
from vllm_omni.engine.duplex.session.runner import DuplexSessionRunner
from vllm_omni.model_executor.models.lychee_fd.duplex.plugin import LycheeDuplexPlugin

SESSION_ID = "lychee-lifecycle-probe"
SYSTEM_PREFIX = [151666, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151665]


class LifecyclePlugin(LycheeDuplexPlugin):
    """Use a recorded tokenizer fixture at this CPU scheduler boundary."""

    async def prepare_runtime_config(self, config, *, model_config):
        runtime = await super().prepare_runtime_config(config, model_config=model_config)
        runtime["lychee_system_token_ids"] = list(SYSTEM_PREFIX)
        return runtime


class RecordingStagePort(DuplexStagePort):
    """The scheduler boundary used by the architecture probe."""

    def __init__(self) -> None:
        self.ensured: list[DuplexStageRequestContext] = []
        self.submissions: list[DuplexStageSubmission] = []
        self.cleanups: list[tuple[list[str], bool]] = []
        self.aborts: list[list[str]] = []

    @property
    def stage_count(self) -> int:
        return 1

    def sampling_defaults(self) -> tuple[object, ...]:
        return (SamplingParams(max_tokens=32, detokenize=False),)

    def ensure_request(self, context: DuplexStageRequestContext) -> None:
        self.ensured.append(context)

    async def submit(self, submission: DuplexStageSubmission) -> DuplexStageSubmissionResult:
        self.submissions.append(submission)
        return DuplexStageSubmissionResult(
            request_id=submission.context.request_id,
            stage_id=submission.context.stage_id,
            replica_id=0,
        )

    async def cleanup(self, request_ids: list[str], *, abort: bool = False) -> None:
        self.cleanups.append((list(request_ids), abort))

    async def abort_requests(self, request_ids: list[str]) -> None:
        self.aborts.append(list(request_ids))


@dataclass
class _Harness:
    manager: DuplexSessionManager
    port: RecordingStagePort
    output: DuplexOutputBuffer
    results: asyncio.Queue[Any]
    runner: DuplexSessionRunner
    control_output: asyncio.Queue[Any]
    events: list[DuplexEvent] = field(default_factory=list)

    def submit(self, command: commands.DuplexCommand) -> None:
        self.manager.dispatch(DuplexSessionCommandMessage(session_id=SESSION_ID, command=command))

    async def settle(self, *, idle_s: float = 0.03, timeout_s: float = 2.0) -> list[DuplexEvent]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        quiet_since: float | None = None
        collected: list[DuplexEvent] = []
        while loop.time() < deadline:
            drained = False
            while not self.control_output.empty():
                message = self.control_output.get_nowait()
                if isinstance(message, DuplexSessionEventMessage):
                    collected.append(message.event)
                    drained = True
            while True:
                try:
                    event = await asyncio.wait_for(self.output.get(), timeout=0.001)
                except asyncio.TimeoutError:
                    break
                if event is None:
                    break
                collected.append(event)
                drained = True
            busy = (
                drained
                or not self.runner._mailbox.empty()
                or any(not task.done() for task in self.runner.tasks.append_tasks)
                or any(not task.done() for task in self.runner._background_tasks)
            )
            now = loop.time()
            if busy:
                quiet_since = None
            elif quiet_since is None:
                quiet_since = now
            elif now - quiet_since >= idle_s:
                break
            await asyncio.sleep(0.005)
        self.events.extend(collected)
        return collected

    async def run(self, command: commands.DuplexCommand) -> list[DuplexEvent]:
        self.submit(command)
        return await self.settle()


def _append(samples: int) -> commands.AppendAudio:
    audio = struct.pack(f"<{samples}f", *([0.05] * samples))
    return commands.AppendAudio(
        audio=audio,
        format="pcm_f32le",
        sample_rate_hz=16000,
        is_speech=True,
    )


def _event_types(events: list[DuplexEvent]) -> list[str]:
    return [event.type for event in events]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


async def _open() -> _Harness:
    port = RecordingStagePort()
    output_sink: asyncio.Queue[Any] = asyncio.Queue()
    output = DuplexOutputBuffer(max_bytes=1024 * 1024, max_events=256)
    results: asyncio.Queue[Any] = asyncio.Queue()
    manager = DuplexSessionManager(
        plugin=LifecyclePlugin(lambda *args: None),
        stage_port=port,
        output_sink=output_sink,
        result_sink=results,
        runtime_config=DuplexSessionRuntimeConfig(max_sessions=1),
        model_config=None,
    )
    config = DuplexSessionConfig(
        model="lychee-fd",
        modalities=["text"],
        instructions="",
        extra_body={"auto_response": True},
    )
    await manager.handle(
        OpenDuplexSessionMessage(control_id="open", session_id=SESSION_ID, session_config=config, output_buffer=output)
    )
    result = await asyncio.wait_for(results.get(), timeout=2.0)
    _require(isinstance(result, DuplexControlResultMessage) and result.ok, f"open failed: {result!r}")
    harness = _Harness(manager, port, output, results, manager.runners[SESSION_ID], output_sink)
    await harness.settle()
    return harness


async def run_probe() -> dict[str, object]:
    """Run open -> append/park -> append/submit -> cancel -> close."""

    harness = await _open()
    try:
        open_events = _event_types(harness.events)
        _require(open_events == ["session.created", "session.updated"], f"unexpected open events: {open_events}")
        _require(len(harness.port.ensured) == 1, "open did not reserve exactly one Stage0 request")
        stage0_reserved_on_open = len(harness.port.ensured)

        await harness.run(_append(3200))
        parked_submissions = len(harness.port.submissions)
        parked_bytes = harness.runner.model_state.audio_buffer.pending_byte_count
        _require(parked_submissions == 0, "a partial 200 ms chunk submitted work instead of parking")
        _require(parked_bytes == 3200 * 4, "partial audio was not retained")

        await harness.run(_append(3200))
        _require(len(harness.port.submissions) == 1, "one complete 400 ms window did not submit exactly once")
        submission = harness.port.submissions[0]
        prompt_token_ids = list(submission.prompt["prompt_token_ids"])
        duplex = submission.prompt["model_intermediate_buffer"]["duplex"]
        ledger = dict(duplex["payload"]["lychee_audio_ledger"])
        _require(prompt_token_ids == SYSTEM_PREFIX + [158_358], "initial prefill lost the recorded system prompt")
        _require(
            submission.context.stage_sampling_params.max_tokens == 9,
            "tick zero is known; first window must sample ticks one through nine",
        )
        _require(duplex["scheduler_token_budget"] == 9, "initial 400 ms window did not budget nine new ticks")
        _require(ledger["consumable_tick_end"] == 10, "audio ledger did not expose ten consumable ticks")

        await harness.run(commands.CancelInput())
        cancel_epoch = harness.runner.session.epoch
        _require(cancel_epoch == 1, "cancel did not advance the session epoch")
        _require(bool(harness.port.aborts), "cancel did not abort the old Stage0 request")

        await harness.manager.handle(
            CloseDuplexSessionMessage(control_id="close", session_id=SESSION_ID, reason="probe_complete")
        )
        close_result = await asyncio.wait_for(harness.results.get(), timeout=2.0)
        close_events = _event_types(await harness.settle())
        _require(close_result.ok, f"close failed: {close_result!r}")
        _require(close_events == ["session.closed"], f"unexpected close events: {close_events}")
        _require(SESSION_ID not in harness.manager.runners, "closed session remained admitted")

        return {
            "passed": True,
            "session_id": SESSION_ID,
            "open_events": open_events,
            "stage0_reserved": stage0_reserved_on_open,
            "stage0_reserved_after_cancel": len(harness.port.ensured),
            "parked_submissions": parked_submissions,
            "parked_bytes": parked_bytes,
            "whole_window_submissions": len(harness.port.submissions),
            "prompt_token_ids": prompt_token_ids,
            "audio_ledger": ledger,
            "cancel_epoch": cancel_epoch,
            "aborted_request_ids": harness.port.aborts,
            "close_events": close_events,
        }
    finally:
        await harness.manager.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    report = asyncio.run(run_probe())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
