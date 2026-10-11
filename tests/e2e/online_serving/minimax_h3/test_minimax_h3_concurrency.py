# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""High-concurrency serving coverage for the H3 transport paths.

The PR changes the A2A/all-reduce transport, so a single-request smoke test cannot
show that the shipped path works when the server is loaded. These cases fire a
fixed number of simultaneous ``/v1/videos/sync`` requests and check the request
accounting (submitted == completed, nothing failed, nothing aborted), that every
returned video decodes, and that the accelerated transport was actually the one
that ran rather than silently falling back to bf16.

Two things make the last point enforceable instead of assumed:

* the transport modules log the mode they engaged (``transport mode -> int8`` /
  ``TP all-reduce mode -> int8``) and latch a failure, so a fallback is visible in
  the server's own output, which these cases read back from the server's capture
  file (``OmniServer(log_capture=True)`` -- the subprocess cannot be observed
  through ``capfd`` because it is spawned before the capture window opens);
* ``qkv_batch`` additionally exercises the fused exchange.

The levels are the ones a maintainer needs for a load comparison (C1/C8/C16/C32);
each is a separate server boot, so run the subset you need rather than the whole
matrix in a constrained CI lane.
"""

from __future__ import annotations

import concurrent.futures
import statistics
import threading
import time
from dataclasses import dataclass

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OpenAIClientHandler

from ._common import (
    CONCURRENCY_SEED,
    assert_h3_video,
    concurrency_params,
    concurrency_request,
    concurrency_task,
    post_sync,
)

pytestmark = [pytest.mark.advanced_model, pytest.mark.diffusion]

# The measured run used four SM120 cards at TP2 x USP2.
FOUR_CARD_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=4)

LEVELS = [1, 8, 16, 32]
# The task-type cases check that the request path works, not how it scales, so they
# fire a single request. Load behaviour is the accounting case's job.
TASK_LEVEL = 1
# Every task type this PR serves. Ref2VA and FL2VA are separate partitions, so each
# needs its own model root (see _common.partition_model).
TASKS = ["ref2va", "fl2va", "t2va"]
WIDTH = 1344
HEIGHT = 768


@dataclass
class Outcome:
    """Terminal state of one submitted request."""

    index: int
    seconds: float
    completed: bool
    aborted: bool
    error: str | None
    body: bytes | None


def _one(
    server: OpenAIClientHandler,
    index: int,
    barrier: threading.Barrier,
    task: str,
    model: str,
) -> Outcome:
    """Fire one request, released with every other request at the same instant."""
    barrier.wait()
    start = time.perf_counter()
    try:
        form, files = concurrency_request(task, CONCURRENCY_SEED + index, model=model)
        body = post_sync(server, form, files=files)
    except concurrent.futures.CancelledError:
        return Outcome(index, time.perf_counter() - start, False, True, "cancelled", None)
    except Exception as exc:  # noqa: BLE001 - recorded, asserted below
        # Surface the server's own message: an httpx 400 alone says nothing about which
        # request field the model rejected.
        detail = repr(exc)
        response = getattr(exc, "response", None)
        if response is not None:
            error_body = getattr(response, "text", "") or ""
            detail = f"{detail} body={error_body[:400]}"
        return Outcome(index, time.perf_counter() - start, False, False, detail, None)
    return Outcome(index, time.perf_counter() - start, True, False, None, body)


def _fire(server: OpenAIClientHandler, count: int, task: str, model: str) -> list[Outcome]:
    """Submit *count* requests together and wait for every one of them."""
    barrier = threading.Barrier(count)
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(_one, server, i, barrier, task, model) for i in range(count)]
        return [future.result() for future in futures]


def _assert_wave(outcomes: list[Outcome], count: int) -> list[Outcome]:
    """Assert every submitted request was accounted for, completed, and decodes."""
    assert len(outcomes) == count, "every submitted request must reach a terminal state"
    failed = [o for o in outcomes if not o.completed and not o.aborted]
    aborted = [o for o in outcomes if o.aborted]
    completed = [o for o in outcomes if o.completed]
    assert not failed, f"{len(failed)} request(s) failed: {[o.error for o in failed][:3]}"
    assert not aborted, f"{len(aborted)} request(s) aborted"
    assert len(completed) == count, "submitted requests must all complete"
    for outcome in completed:
        assert outcome.body is not None
        assert_h3_video(outcome.body, width=WIDTH, height=HEIGHT)
    return completed


def _report_accounting(task: str, wire: str, outcomes: list[Outcome]) -> None:
    """Print the request accounting only.

    These cases check that a task type is served, not how fast it is. The first
    request after a boot carries one-time compile work, so a latency printed here
    would read as a performance figure it is not.
    """
    print(
        f"[task-types] task={task} wire={wire} submitted={len(outcomes)} "
        f"completed={sum(o.completed for o in outcomes)} "
        f"failed={sum(not o.completed and not o.aborted for o in outcomes)} "
        f"aborted={sum(o.aborted for o in outcomes)}"
    )


def _assert_transport_engaged(log_text: str, wire: str) -> None:
    """Fail unless the server log shows the requested transport running."""
    assert log_text, "the server log was not captured; cannot prove which transport ran"
    assert f"transport mode -> {wire}" in log_text, (
        f"the {wire} all-to-all never engaged; the server log did not contain it"
    )
    if wire == "int8":
        assert "TP all-reduce mode -> int8" in log_text, "the int8 all-reduce never engaged"
        assert "int8 transport FAILED" not in log_text, "the int8 transport raised and fell back mid-run"
        assert "int8 AR FAILED" not in log_text, "the int8 all-reduce raised and fell back mid-run"


def _report(level: int, wire: str, task: str, outcomes: list[Outcome], wall: float) -> None:
    """Print the numbers a maintainer needs to compare two arms."""
    latencies = sorted(o.seconds for o in outcomes if o.completed)
    mean = statistics.fmean(latencies) if latencies else float("nan")
    median = statistics.median(latencies) if latencies else float("nan")
    p95 = latencies[min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1))))] if latencies else float("nan")
    print(
        f"[concurrency] task={task} wire={wire} C={level} submitted={len(outcomes)} "
        f"completed={sum(o.completed for o in outcomes)} "
        f"failed={sum(not o.completed and not o.aborted for o in outcomes)} "
        f"aborted={sum(o.aborted for o in outcomes)} "
        f"throughput={len(latencies) / wall if wall else 0:.4f} req/s "
        f"latency_mean={mean:.2f}s median={median:.2f}s p95={p95:.2f}s wall={wall:.2f}s"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(
            concurrency_params(wire="int8"),
            id="int8",
            marks=FOUR_CARD_MARKS,
        ),
        pytest.param(
            concurrency_params(wire="bf16"),
            id="bf16",
            marks=FOUR_CARD_MARKS,
        ),
    ],
    indirect=True,
)
def test_minimax_h3_concurrency_accounting(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
    level: int,
) -> None:
    """Fire ``level`` simultaneous requests and check the full accounting."""
    # Taken from the server that was actually launched, so the expectation can never
    # drift from the arm under test.
    wire = (omni_server.env_dict or {})["H3_A2A_WIRE"]
    task = concurrency_task(omni_server)
    started = time.perf_counter()
    outcomes = _fire(openai_client, level, task, omni_server.model)
    wall = time.perf_counter() - started
    _report(level, wire, task, outcomes, wall)

    _assert_wave(outcomes, level)
    _assert_transport_engaged(omni_server.read_captured_log(), wire)


@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(
            concurrency_params(wire="int8", task=task),
            id=f"{task}_int8",
            marks=FOUR_CARD_MARKS,
        )
        for task in TASKS
    ],
    indirect=True,
)
def test_minimax_h3_every_task_type_under_concurrency(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
) -> None:
    """Every task type the PR serves.

    The transport change and the request path are shared, so a case that only
    exercised ref2va would not cover t2va's reference-free path or fl2va's
    keyframe path.
    """
    task = concurrency_task(omni_server)
    outcomes = _fire(openai_client, TASK_LEVEL, task, omni_server.model)
    _report_accounting(task, (omni_server.env_dict or {})["H3_A2A_WIRE"], outcomes)
    _assert_wave(outcomes, TASK_LEVEL)
    _assert_transport_engaged(omni_server.read_captured_log(), (omni_server.env_dict or {})["H3_A2A_WIRE"])


@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(
            concurrency_params(wire="int8", qkv_batch=True),
            id="int8_fused_qkv",
            marks=FOUR_CARD_MARKS,
        )
    ],
    indirect=True,
)
def test_minimax_h3_fused_qkv_exchange_under_concurrency(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
) -> None:
    """The fused QKV exchange, enabled and exercised with simultaneous requests."""
    task = concurrency_task(omni_server)
    outcomes = _fire(openai_client, TASK_LEVEL, task, omni_server.model)
    _assert_wave(outcomes, TASK_LEVEL)
    _assert_transport_engaged(omni_server.read_captured_log(), "int8")
    assert (omni_server.env_dict or {}).get("H3_A2A_QKV_BATCH") == "1", "the fused case must run with batching enabled"
