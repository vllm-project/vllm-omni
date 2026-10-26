# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Capacity preemption through the production scheduler and real KV allocator.

Only model configuration, runner output and connector I/O are synthetic. No
schedule, finish, allocation or terminal-output method is replaced.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config import CacheConfig, SchedulerConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.sampling_params import SamplingParams
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.engine import FinishReason
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _scheduler(mode="fail", requires_kv_delivery=False, auxiliary=False, block_size=16, num_blocks=2):
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
        recompute_preemption=mode,
    )
    config.scheduler_config = SchedulerConfig(
        max_model_len=64,
        max_num_seqs=2,
        max_num_batched_tokens=64,
        is_encoder_decoder=False,
    )
    config.aux_output_config.enable_return_routed_experts = auxiliary
    config.cache_config = CacheConfig(block_size=block_size, enable_prefix_caching=False)
    # By default: one null block and one usable block, enough only for prefill.
    config.cache_config.num_gpu_blocks = num_blocks
    # Match EngineCore initialization before constructing the scheduler. Custom
    # specs registered during test collection can suppress lazy built-in registration.
    register_all_kvcache_specs(config)
    kv_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(block_size=block_size, num_kv_heads=1, head_size=8, dtype=torch.float16),
            )
        ],
    )
    scheduler = OmniARScheduler(config, kv_config, Mock(spec=StructuredOutputManager), block_size)
    if requires_kv_delivery:
        connector = Mock(spec=KVConnectorBase_V1)
        connector.requires_kv_delivery = True
        connector.has_pending_block_frees.return_value = False
        connector.get_num_new_matched_tokens.return_value = (0, False)
        connector.request_finished.return_value = (False, None)
        connector.build_connector_meta.return_value = None
        connector.take_events.return_value = ()
        connector.get_kv_connector_stats.return_value = None
        scheduler.connector = connector
        scheduler.requires_kv_delivery = connector.requires_kv_delivery
    return scheduler


def _output(request=None):
    ids = [] if request is None else [request.request_id]
    return ModelRunnerOutput(
        req_ids=ids,
        req_id_to_index={rid: i for i, rid in enumerate(ids)},
        sampled_token_ids=[] if request is None else [[1]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _prefill(scheduler, prompt_len=16):
    request = Request("req-preempt", list(range(prompt_len)), SamplingParams(max_tokens=32), None, client_index=3)
    scheduler.add_request(request)
    scheduled = scheduler.schedule()
    assert scheduled.num_scheduled_tokens == {request.request_id: prompt_len}
    scheduler.update_from_output(scheduled, _output(request))
    assert request.status == RequestStatus.RUNNING
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 0
    return request


@pytest.mark.parametrize("requires_kv_delivery", [False, True])
@pytest.mark.parametrize("auxiliary", [False, True])
def test_capacity_preemption_emits_one_error_and_frees_state(requires_kv_delivery, auxiliary):
    scheduler = _scheduler(requires_kv_delivery=requires_kv_delivery, auxiliary=auxiliary)
    request = _prefill(scheduler)
    scheduled = scheduler.schedule()
    outputs = scheduler.update_from_output(scheduled, _output())

    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.request_id not in scheduler.requests
    assert not scheduler.running and not scheduler.waiting and not scheduler.kv_holding_waiting
    assert request not in scheduler.deferred_waiting
    assert request not in scheduler._inflight_prefills
    assert not any(scheduler.kv_cache_manager.get_block_ids(request.request_id))
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 1
    assert scheduled.finished_req_ids == {request.request_id}
    assert scheduled.num_scheduled_tokens == {}
    [terminal] = outputs[3].outputs
    assert terminal.finish_reason == FinishReason.ERROR
    assert terminal.new_token_ids == []
    assert "cannot replay" in terminal.stop_reason and "truncated" in terminal.stop_reason
    assert outputs[3].finished_requests == {request.request_id}
    if requires_kv_delivery:
        scheduler.connector.request_finished.assert_called_once()

    if auxiliary:
        assert request.request_id not in scheduler.aux_output_connector._sent_hash_counts
        assert scheduled.aux_output_connector_metadata.finished_requests == (request.request_id,)

    # A late repeat and subsequent empty engine step must not emit ERROR/ABORT twice.
    scheduler._preempt_request(request, 1.0)
    next_step = scheduler.schedule()
    if auxiliary:
        assert next_step.aux_output_connector_metadata.finished_requests == ()
    assert scheduler.update_from_output(next_step, _output()) == {}


@pytest.mark.parametrize("requires_kv_delivery", [False, True])
def test_allow_mode_capacity_preemption_requeues(requires_kv_delivery):
    scheduler = _scheduler("allow", requires_kv_delivery)
    request = _prefill(scheduler)
    scheduled = scheduler.schedule()
    assert request.status == RequestStatus.PREEMPTED
    assert request in scheduler.waiting
    assert request.num_computed_tokens == 0
    assert request.drop_stale_output is requires_kv_delivery
    assert scheduler.update_from_output(scheduled, _output()) == {}


@pytest.mark.parametrize("requires_kv_delivery", [False, True])
def test_explicit_prefix_cache_reset_preserves_resume(requires_kv_delivery):
    scheduler = _scheduler(requires_kv_delivery=requires_kv_delivery)
    # The prompt plus its first sampled token must still fit for reset replay.
    request = _prefill(scheduler, prompt_len=15)
    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    assert request.status == RequestStatus.PREEMPTED
    assert request in scheduler.waiting
    assert request.drop_stale_output
    assert not scheduler._resetting_prefix_cache
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 1
    # The resumed prefill fills the only block, then capacity must fail again:
    # the reset exception cannot leak into a subsequent scheduler iteration.
    scheduled = scheduler.schedule()
    scheduler.update_from_output(scheduled, _output(request))
    outputs = scheduler.update_from_output(scheduler.schedule(), _output())
    [terminal] = outputs[3].outputs
    assert terminal.finish_reason == FinishReason.ERROR


def test_prefix_cache_reset_scope_is_cleared_on_error(monkeypatch):
    scheduler = _scheduler()
    _prefill(scheduler)
    monkeypatch.setattr(scheduler.kv_cache_manager, "reset_prefix_cache", lambda: False)
    with pytest.raises(RuntimeError, match="Failed to reset KV cache"):
        scheduler.reset_prefix_cache(reset_running_requests=True)
    assert not scheduler._resetting_prefix_cache
