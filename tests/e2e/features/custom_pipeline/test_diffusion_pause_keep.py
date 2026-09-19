# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Lifecycle test for ``AsyncOmni.pause_generation(mode="keep")`` on a diffusion stage.

Per round, request A is started, request B is queued behind it, and the pause
is issued while A is still running. The pause must return only after A has
been delivered, B must stay queued through ``sleep`` / ``wake_up``, and B may
start only after ``resume_generation``.

A second test follows the order the docs recommend for a weight update: sleep
is issued as soon as the pause returns, without awaiting the request that was
running, which must still be delivered intact.

Both single-GPU executor backends are covered explicitly: ``uni`` (in-process
worker) and ``mp`` (worker process with the asynchronous D2H output thread).

Usage:
    pytest tests/e2e/features/custom_pipeline/test_diffusion_pause_keep.py -v -s
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from contextlib import ExitStack

import pytest

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.data import is_diffusion_request_started_output
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = "tiny-random/Qwen-Image"
CUSTOM_PIPELINE_CLASS = "tests.e2e.features.helpers.custom_pipeline.QwenImagePipelineWithLogProbForTest"
WORKER_EXTENSION_CLASS = "tests.e2e.features.helpers.custom_pipeline.vLLMOmniColocateWorkerExtensionForTest"

ROUNDS = 10
# Enough denoising steps that the pause lands while A is still executing.
NUM_INFERENCE_STEPS = 40


def _sampling_params() -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=0.0,
        height=256,
        width=256,
        seed=42,
        emit_request_lifecycle=True,
    )


async def _generate(engine: AsyncOmni, request_id: str, events: dict[str, dict[str, float]]):
    final = None
    async for output in engine.generate(
        prompt={"prompt_ids": list(range(50))},
        request_id=request_id,
        sampling_params_list=[_sampling_params()],
        output_modalities=["image"],
    ):
        if is_diffusion_request_started_output(output):
            events[request_id]["started"] = time.monotonic()
            continue
        final = output
    events[request_id]["finished"] = time.monotonic()
    return final


async def _wait_for(predicate, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        await asyncio.sleep(0.005)


def _all_true(results) -> bool:
    flat = []
    stack = list(results)
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
        else:
            flat.append(item)
    return bool(flat) and all(item is True for item in flat)


async def _pause_round(engine: AsyncOmni, events: dict[str, dict[str, float]], round_index: int) -> float:
    """A runs, B queues, keep-pause lands while A runs; B starts only after resume.

    Returns the pause latency.
    """
    a_id, b_id = f"a{round_index}", f"b{round_index}"
    a = asyncio.create_task(_generate(engine, a_id, events))
    await _wait_for(lambda: "started" in events[a_id])
    b = asyncio.create_task(_generate(engine, b_id, events))
    await asyncio.sleep(0.05)

    assert not a.done(), "A finished before the pause was issued; raise NUM_INFERENCE_STEPS"
    pause_start = time.monotonic()
    await engine.pause_generation(mode="keep", clear_cache=False)
    pause_latency = time.monotonic() - pause_start
    assert await engine.is_paused()

    a_output = await asyncio.wait_for(a, 60.0)
    assert a_output.images, "A must be delivered, not aborted, by a keep pause"
    assert "started" not in events[b_id], "B must not start while paused"

    # Control RPCs and the sleep cycle stay available on the paused stage.
    assert await engine.list_loras() == []
    await engine.sleep(level=1)
    assert await engine.is_sleeping()
    await engine.wake_up()
    assert not await engine.is_sleeping()
    assert await engine.is_paused()
    assert not b.done() and "started" not in events[b_id]

    resume_at = time.monotonic()
    await engine.resume_generation()
    b_output = await asyncio.wait_for(b, 60.0)
    assert b_output.images, "B must run after resume"
    assert events[b_id]["started"] >= resume_at
    return pause_latency


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["uni", "mp"])
async def test_keep_pause_sleep_wake_resume_lifecycle(backend: str):
    with ExitStack() as after:
        engine = AsyncOmni(
            model=MODEL,
            custom_pipeline_args={"pipeline_class": CUSTOM_PIPELINE_CLASS},
            worker_extension_cls=WORKER_EXTENSION_CLASS,
            enforce_eager=True,
            enable_sleep_mode=True,
            distributed_executor_backend=backend,
        )
        after.callback(engine.shutdown)
        events: dict[str, dict[str, float]] = defaultdict(dict)

        pause_latencies = [await _pause_round(engine, events, round_index) for round_index in range(ROUNDS)]

        assert not engine.request_states, "every request must be finalized"
        print(
            f"[{backend}] rounds={ROUNDS} steps={NUM_INFERENCE_STEPS} "
            f"pause_latency_s min/median/max="
            f"{min(pause_latencies):.3f}/{sorted(pause_latencies)[ROUNDS // 2]:.3f}/{max(pause_latencies):.3f}"
        )


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["uni", "mp"])
async def test_sleep_right_after_keep_pause_keeps_the_running_batch(backend: str):
    """The documented weight-update order: sleep is issued as soon as the pause
    returns, with the request that was running still un-awaited.
    """
    with ExitStack() as after:
        engine = AsyncOmni(
            model=MODEL,
            custom_pipeline_args={"pipeline_class": CUSTOM_PIPELINE_CLASS},
            worker_extension_cls=WORKER_EXTENSION_CLASS,
            enforce_eager=True,
            enable_sleep_mode=True,
            distributed_executor_backend=backend,
        )
        after.callback(engine.shutdown)
        events: dict[str, dict[str, float]] = defaultdict(dict)

        a = asyncio.create_task(_generate(engine, "a", events))
        await _wait_for(lambda: "started" in events["a"])
        await asyncio.sleep(0.05)
        assert not a.done(), "A finished before the pause was issued; raise NUM_INFERENCE_STEPS"

        await engine.pause_generation(mode="keep", clear_cache=False)
        await engine.sleep(level=1)

        a_output = await asyncio.wait_for(a, 60.0)
        assert a_output.images, "A must survive a sleep issued right after the pause ACK"

        await engine.wake_up()
        await engine.resume_generation()
        b_output = await asyncio.wait_for(_generate(engine, "b", events), 120.0)
        assert b_output.images, "the stage must generate again after wake and resume"


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=2)
@pytest.mark.asyncio
async def test_keep_pause_two_ranks_ack_and_rank_failure():
    """Every rank runs the barrier before the ACK; a barrier failure on a
    non-zero rank fails the pause without aborting or dropping anything.
    """
    with ExitStack() as after:
        engine = AsyncOmni(
            model=MODEL,
            custom_pipeline_args={"pipeline_class": CUSTOM_PIPELINE_CLASS},
            worker_extension_cls=WORKER_EXTENSION_CLASS,
            enforce_eager=True,
            enable_sleep_mode=True,
            num_gpus=2,
            parallel_config={"ulysses_degree": 2, "ulysses_mode": "advanced_uaa"},
        )
        after.callback(engine.shutdown)
        events: dict[str, dict[str, float]] = defaultdict(dict)
        assert _all_true(await engine.collective_rpc(method="start_counting_synchronize_device"))

        pause_latencies = [await _pause_round(engine, events, round_index) for round_index in range(3)]
        # The result is the all-rank AND, so True means both ranks ran the barrier.
        assert _all_true(await engine.collective_rpc(method="synchronize_device_seen"))

        assert _all_true(await engine.collective_rpc(method="set_synchronize_device_failure", args=(1, True)))
        a = asyncio.create_task(_generate(engine, "a-fail", events))
        await _wait_for(lambda: "started" in events["a-fail"])
        with pytest.raises(RuntimeError, match="pause_scheduler failed"):
            await engine.pause_generation(mode="keep", clear_cache=False)
        assert await engine.is_paused()
        a_output = await asyncio.wait_for(a, 60.0)
        assert a_output.images, "a failed barrier must not abort the running batch"

        assert _all_true(await engine.collective_rpc(method="set_synchronize_device_failure", args=(1, False)))
        await engine.resume_generation()
        assert not await engine.is_paused()
        recovered = await asyncio.wait_for(_generate(engine, "after-recovery", events), 60.0)
        assert recovered.images

        assert not engine.request_states
        print(f"[sp2] rounds=3 pause_latency_s min/max={min(pause_latencies):.3f}/{max(pause_latencies):.3f}")
