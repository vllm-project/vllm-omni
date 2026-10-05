# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Two-rank ``pause_generation(mode="keep")``: every rank runs the pause barrier
before the ACK, and a barrier failure on a non-zero rank fails the pause
without aborting or dropping the running batch.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from contextlib import ExitStack

import pytest

from tests.e2e.features.custom_pipeline.test_diffusion_pause_keep import (
    CUSTOM_PIPELINE_CLASS,
    MODEL,
    WORKER_EXTENSION_CLASS,
    _generate,
    _pause_round,
    _wait_for,
)
from tests.helpers.mark import hardware_test
from vllm_omni.entrypoints.async_omni import AsyncOmni


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


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=2)
@pytest.mark.asyncio
async def test_keep_pause_two_ranks_ack_and_rank_failure():
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
