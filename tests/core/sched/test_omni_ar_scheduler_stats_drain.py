# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The last SchedulerStats of a stage must report the KV cache it freed last.

A finished request can keep its blocks after it leaves the queues: a stage that
hands its KV cache to the next stage holds them until the runner acks the
extraction, and an upstream KV connector holds them until the worker reports
them sent. The engine keeps stepping until then, so the step that frees them is
the stage's last step. Its stats must be taken after the free and must not be
throttled; otherwise ``vllm:kv_cache_usage_perc`` stays above zero while
nothing is allocated.

The scheduler, KV allocator and transfer bookkeeping are real; only the model
configuration and runner output are synthetic.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config import CacheConfig, SchedulerConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.sampling_params import SamplingParams
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_omni.core.sched import omni_scheduler_mixin
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.outputs import OmniModelRunnerOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_BLOCK_SIZE = 4
_NUM_BLOCKS = 16


def _scheduler(*, hands_kv_to_next_stage: bool) -> OmniARScheduler:
    config = VllmConfig()
    # Avoid downloading model weights/config; all scheduler/cache objects are real.
    config.model_config = SimpleNamespace(
        multimodal_config=None,
        uses_mrope=False,
        is_encoder_decoder=False,
        max_model_len=64,
        runner_type="generate",
        is_diffusion=False,
        supports_multimodal_inputs=False,
        return_sampling_mask=False,
        recompute_preemption="fail",
        omni_kv_config={"need_send_cache": hands_kv_to_next_stage},
    )
    config.scheduler_config = SchedulerConfig(
        max_model_len=64,
        max_num_seqs=2,
        max_num_batched_tokens=64,
        is_encoder_decoder=False,
    )
    config.cache_config = CacheConfig(block_size=_BLOCK_SIZE, enable_prefix_caching=False)
    config.cache_config.num_gpu_blocks = _NUM_BLOCKS
    register_all_kvcache_specs(config)
    kv_config = KVCacheConfig(
        num_blocks=_NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(block_size=_BLOCK_SIZE, num_kv_heads=1, head_size=8, dtype=torch.float16),
            )
        ],
    )
    return OmniARScheduler(config, kv_config, Mock(spec=StructuredOutputManager), _BLOCK_SIZE, log_stats=True)


def _connector() -> Mock:
    """An upstream KV connector that holds a finished request's blocks until they are sent."""
    connector = Mock(spec=KVConnectorBase_V1)
    connector.requires_kv_delivery = False
    connector.has_pending_block_frees.return_value = False
    connector.has_pending_push_work.return_value = False
    connector.get_num_new_matched_tokens.return_value = (0, False)
    connector.request_finished.return_value = (True, None)
    connector.build_connector_meta.return_value = None
    connector.take_events.return_value = ()
    connector.get_kv_connector_stats.return_value = None
    return connector


def _output(
    request: Request | None = None,
    *,
    kv_extracted_req_ids: list[str] | None = None,
    kv_connector_output: KVConnectorOutput | None = None,
) -> OmniModelRunnerOutput:
    ids = [] if request is None else [request.request_id]
    return OmniModelRunnerOutput(
        req_ids=ids,
        req_id_to_index={rid: i for i, rid in enumerate(ids)},
        sampled_token_ids=[] if request is None else [[1]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        kv_connector_output=kv_connector_output,
        kv_extracted_req_ids=kv_extracted_req_ids,
    )


def _run_until_finished(scheduler: OmniARScheduler, clock: list[float], request_id: str) -> Request:
    request = Request(request_id, list(range(8)), SamplingParams(max_tokens=3), None)
    scheduler.add_request(request)

    # Step 0: prefill. The first report is never throttled.
    prefill = _stats(scheduler.update_from_output(scheduler.schedule(), _output(request)))
    assert prefill is not None and prefill.num_running_reqs == 1

    # Step 1: an ordinary decode step inside the window stays throttled.
    clock[0] += 0.1
    assert _stats(scheduler.update_from_output(scheduler.schedule(), _output(request))) is None

    # Step 2: the request finishes but keeps its blocks, so the engine keeps stepping.
    clock[0] += 0.1
    scheduler.update_from_output(scheduler.schedule(), _output(request))
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert scheduler.has_requests()
    assert scheduler.kv_cache_manager.usage > 0
    return request


def _assert_idle_report(scheduler: OmniARScheduler, last) -> None:
    assert not scheduler.has_requests()
    assert scheduler.kv_cache_manager.usage == 0
    assert last is not None
    assert (last.num_running_reqs, last.num_waiting_reqs) == (0, 0)
    assert last.kv_cache_usage == 0


def _stats(outputs):
    reported = [out.scheduler_stats for out in outputs.values() if out.scheduler_stats is not None]
    assert len(reported) <= 1
    return reported[0] if reported else None


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    now = [1000.0]
    monkeypatch.setattr(omni_scheduler_mixin, "time", SimpleNamespace(monotonic=lambda: now[0]))
    return now


# 0.1 s: the freeing step falls inside the throttle window.
# 1.5 s: it falls outside, so it is reported either way and must not count the freed blocks.
@pytest.mark.parametrize("free_delay_s", [0.1, 1.5])
def test_last_step_reports_kv_freed_after_stage_handoff(clock, free_delay_s):
    scheduler = _scheduler(hands_kv_to_next_stage=True)
    request = _run_until_finished(scheduler, clock, "req-handoff")
    assert request.request_id in scheduler.waiting_for_transfer_free

    # Step 3: the runner acks the extraction and the blocks are freed.
    clock[0] += free_delay_s
    handoff = scheduler.schedule()
    assert request.request_id in scheduler.active_kv_transfers
    outputs = scheduler.update_from_output(handoff, _output(kv_extracted_req_ids=[request.request_id]))
    _assert_idle_report(scheduler, _stats(outputs))


@pytest.mark.parametrize("free_delay_s", [0.1, 1.5])
def test_last_step_reports_kv_freed_after_connector_send(clock, free_delay_s):
    scheduler = _scheduler(hands_kv_to_next_stage=False)
    scheduler.connector = _connector()
    request = _run_until_finished(scheduler, clock, "req-send")
    # Nothing is unfinished; only the connector's hold keeps the engine stepping.
    assert not scheduler.has_unfinished_requests()

    # Step 3: the worker reports the blocks sent and the scheduler frees them.
    clock[0] += free_delay_s
    sent = KVConnectorOutput(finished_sending={request.request_id})
    outputs = scheduler.update_from_output(scheduler.schedule(), _output(kv_connector_output=sent))
    _assert_idle_report(scheduler, _stats(outputs))
