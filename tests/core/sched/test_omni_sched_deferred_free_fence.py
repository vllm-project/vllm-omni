# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""v0.28 deferred KV-block free fence in the Omni schedulers (#6606 review).

Upstream v0.28 (defer_block_free) advances ``sched_step_seq`` for every
non-empty scheduled step and, at the top of ``update_from_output()``,
advances ``processed_step_seq`` and drains the deferred frees. The Omni
schedulers each carried only half of that contract:

- ``OmniARScheduler.schedule()`` delegates to ``super().schedule()`` (fence
  advances) but its reimplemented ``update_from_output()`` never advanced
  ``processed_step_seq`` nor drained -> deferred blocks accumulate without
  bound under async/PP KV-consumer configurations.
- ``OmniGenerationScheduler`` reimplements both: the fast-path
  ``schedule()`` never advanced ``sched_step_seq`` (so finished requests
  freed blocks an in-flight GPU step could still be writing) and its
  ``update_from_output()`` never drained.

These tests pin the restored contract at the call boundary; the drain and
free logic itself is upstream-tested.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# Imports must run in this order: vllm_omni applies patches to vllm.v1.request
# before Request / RequestStatus are bound in this module.
# isort: off
import vllm_omni  # noqa: F401 - import for side effects (patch vLLM)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler

# isort: on

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_update_sched(scheduler_cls) -> MagicMock:
    """MagicMock scheduler driving the real ``update_from_output`` with an
    empty batch; only the fence state carries real values."""
    sched = MagicMock()
    sched.requests = {}
    sched.perf_metrics = None
    sched.chunk_transfer_adapter = None
    sched.defer_block_free = True
    sched.processed_step_seq = 0
    return sched


def _empty_frame(*, total_num_scheduled_tokens: int):
    scheduler_output = MagicMock(spec=SchedulerOutput)
    scheduler_output.num_scheduled_tokens = {}
    scheduler_output.total_num_scheduled_tokens = total_num_scheduled_tokens
    model_runner_output = MagicMock(spec=ModelRunnerOutput)
    model_runner_output.sampled_token_ids = []
    model_runner_output.logprobs = None
    model_runner_output.prompt_logprobs_dict = {}
    model_runner_output.pooler_output = None
    model_runner_output.num_nans_in_logits = None
    model_runner_output.kv_connector_output = None
    model_runner_output.cudagraph_stats = None
    return scheduler_output, model_runner_output


@pytest.mark.parametrize("scheduler_cls", [OmniARScheduler, OmniGenerationScheduler])
def test_update_from_output_advances_fence_and_drains(scheduler_cls) -> None:
    sched = _make_update_sched(scheduler_cls)
    scheduler_output, model_runner_output = _empty_frame(total_num_scheduled_tokens=3)

    scheduler_cls.update_from_output(sched, scheduler_output, model_runner_output)

    assert sched.processed_step_seq == 1
    sched._drain_deferred_frees.assert_called_once_with()


@pytest.mark.parametrize("scheduler_cls", [OmniARScheduler, OmniGenerationScheduler])
def test_update_from_output_zero_token_step_leaves_fence(scheduler_cls) -> None:
    """0-token steps do not advance the fence upstream (their seq was never
    advanced in schedule()); draining on them would run ahead of the GPU."""
    sched = _make_update_sched(scheduler_cls)
    scheduler_output, model_runner_output = _empty_frame(total_num_scheduled_tokens=0)

    scheduler_cls.update_from_output(sched, scheduler_output, model_runner_output)

    assert sched.processed_step_seq == 0
    sched._drain_deferred_frees.assert_not_called()


@pytest.mark.parametrize("scheduler_cls", [OmniARScheduler, OmniGenerationScheduler])
def test_update_from_output_defer_disabled_leaves_fence(scheduler_cls) -> None:
    sched = _make_update_sched(scheduler_cls)
    sched.defer_block_free = False
    scheduler_output, model_runner_output = _empty_frame(total_num_scheduled_tokens=3)

    scheduler_cls.update_from_output(sched, scheduler_output, model_runner_output)

    assert sched.processed_step_seq == 0
    sched._drain_deferred_frees.assert_not_called()


def _make_fast_path_sched(*, defer_block_free: bool) -> tuple[MagicMock, Request]:
    """MagicMock scheduler that drives the real generation fast-path
    ``schedule()`` over one running one-shot request."""
    request = Request(
        request_id="req-fence",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        arrival_time=100.0,
        block_hasher=None,
    )
    request.status = RequestStatus.RUNNING

    sched = MagicMock()
    sched.max_num_scheduled_tokens = 8
    sched._pause_state = PauseState.UNPAUSED
    sched.requests = {request.request_id: request}
    sched.running = [request]
    sched.waiting = []  # falsy: the waiting loop never runs
    sched.policy = SchedulingPolicy.FCFS
    sched.chunk_transfer_adapter = None
    sched.scheduler_config = MagicMock(enable_chunked_prefill=True)
    sched.num_lookahead_tokens = 0
    sched.log_stats = False
    sched.kv_cache_config = MagicMock(kv_cache_groups=[])
    sched.use_v2_model_runner = False
    sched.prev_step_scheduled_req_ids = set()
    sched.needs_kv_cache_zeroing = False
    sched.connector = None
    sched.ec_connector = None
    sched.defer_block_free = defer_block_free
    sched.sched_step_seq = 0
    # last_sched_seq stamping order: _update_after_schedule must observe the
    # already-advanced fence value, as upstream schedule() guarantees.
    sched.seq_at_update_after_schedule = None

    def _capture(_scheduler_output):
        sched.seq_at_update_after_schedule = sched.sched_step_seq

    sched._update_after_schedule.side_effect = _capture
    return sched, request


def test_fast_path_schedule_advances_fence_before_update_after_schedule() -> None:
    sched, request = _make_fast_path_sched(defer_block_free=True)

    OmniGenerationScheduler.schedule(sched)

    assert sched.sched_step_seq == 1
    assert sched.seq_at_update_after_schedule == 1


def test_fast_path_schedule_defer_disabled_leaves_fence() -> None:
    sched, request = _make_fast_path_sched(defer_block_free=False)

    OmniGenerationScheduler.schedule(sched)

    assert sched.sched_step_seq == 0
