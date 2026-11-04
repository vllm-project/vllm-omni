# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Resident turn boundaries through the real async scheduler and KV allocator."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

# isort: off
import vllm_omni  # noqa: F401 - apply Request patches before importing vLLM types
from vllm.config import CacheConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager
from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler
# isort: on

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _scheduler():
    config = VllmConfig()
    # Only model configuration and worker results are synthetic; scheduling,
    # block allocation, hashing, cancellation and output accounting are real.
    config.model_config = SimpleNamespace(
        multimodal_config=None,
        uses_mrope=False,
        is_encoder_decoder=False,
        max_model_len=128,
        runner_type="generate",
        is_diffusion=False,
        supports_multimodal_inputs=False,
        return_sampling_mask=False,
        stage_id=0,
        session_mode="duplex",
    )
    config.scheduler_config = SchedulerConfig(
        max_model_len=128,
        max_num_seqs=4,
        max_num_batched_tokens=128,
        is_encoder_decoder=False,
        async_scheduling=True,
    )
    config.cache_config = CacheConfig(block_size=16, enable_prefix_caching=True)
    config.cache_config.num_gpu_blocks = 32
    register_all_kvcache_specs(config)
    kv_config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16),
            )
        ],
    )
    return OmniARAsyncScheduler(config, kv_config, Mock(spec=StructuredOutputManager), 16)


def _request(tokens, request_id="resident"):
    hash_fn = get_hash_fn_by_name("sha256")
    init_none_hash(hash_fn)
    request = Request(
        request_id,
        tokens,
        SamplingParams(max_tokens=16, stop_token_ids=[999]),
        None,
        block_hasher=get_request_block_hasher(16, hash_fn),
        resumable=True,
    )
    request.resume_from_full_prompt = True
    return request


def _output(request, token):
    return ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[token]],
    )


def _stop_with_in_flight_decode(scheduler, request):
    scheduler.add_request(request)
    prefill = scheduler.schedule()
    decode = scheduler.schedule()
    scheduler.update_from_output(prefill, _output(request, 33))
    lookahead = scheduler.schedule()
    result = scheduler.update_from_output(decode, _output(request, 999))
    assert result[request.client_index].outputs[0].is_segment_finished
    assert request.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert request.num_computed_tokens == 34
    assert request.num_in_flight_tokens == request.num_stale_output_tokens == 1
    assert request.num_output_placeholders == 0
    return lookahead


@pytest.mark.parametrize("change", ["append", "edit", "shrink"])
def test_async_continuation_drains_then_reinstalls_worker(change):
    scheduler = _scheduler()
    request = _request(list(range(33)))
    lookahead = _stop_with_in_flight_decode(scheduler, request)
    previous_blocks = scheduler.kv_cache_manager.get_block_ids(request.request_id)[0][:2]
    tokens = list(range(50 if change != "shrink" else 16))
    if change == "edit":
        tokens[0] = 123
    scheduler.add_request(_request(tokens))
    assert request.num_computed_tokens == (32 if change == "append" else 0)

    # A different session can prefill while the continued one is fenced.
    other = _request([101, 102, 103], request_id="other")
    scheduler.add_request(other)
    other_step = scheduler.schedule()
    assert request.request_id not in other_step.num_scheduled_tokens
    assert request.request_id not in other_step.finished_req_ids
    assert other_step.num_scheduled_tokens[other.request_id] == 3
    assert not scheduler.update_from_output(lookahead, _output(request, 777))
    assert request.num_stale_output_tokens == request.num_in_flight_tokens == 0
    assert request.num_output_tokens == 0
    scheduler.update_from_output(other_step, _output(other, 999))

    resumed = scheduler.schedule()
    [arrival] = resumed.scheduled_new_reqs
    assert arrival.req_id == request.request_id
    assert arrival.num_computed_tokens == (32 if change == "append" else 0)
    assert request.request_id in resumed.finished_req_ids
    if change == "append":
        assert scheduler.kv_cache_manager.get_block_ids(request.request_id)[0][:2] == previous_blocks
    result = scheduler.update_from_output(resumed, _output(request, 888))
    assert result[request.client_index].outputs[0].new_token_ids == [888]
    assert list(request.output_token_ids) == [888]
    assert request.num_output_placeholders == request.num_in_flight_tokens == 0
    assert request.prompt_token_ids == tokens


def test_cancel_async_continuation_while_old_output_drains():
    scheduler = _scheduler()
    request = _request(list(range(33)))
    lookahead = _stop_with_in_flight_decode(scheduler, request)
    scheduler.add_request(_request(list(range(50))))
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    result = scheduler.update_from_output(lookahead, _output(request, 777))
    assert all(not output.new_token_ids for batch in result.values() for output in batch.outputs)
    assert request.request_id not in scheduler.requests
    assert not scheduler.kv_cache_manager.get_block_ids(request.request_id)[0]
    assert not scheduler.waiting and not scheduler.kv_holding_waiting and not scheduler.running
    assert scheduler.num_waiting_for_streaming_input == 0


def test_direct_turn_delta_reuses_resident_blocks_without_snapshot_hashes():
    scheduler = _scheduler()
    request = _request(list(range(33)))
    lookahead = _stop_with_in_flight_decode(scheduler, request)
    blocks = scheduler.kv_cache_manager.get_block_ids(request.request_id)[0][:2]
    update = _request([999, 100, 101])
    update.append_prefix_length = 34
    # These are hashes of the suffix alone and cannot match the resident prefix.
    update.block_hashes.clear()
    scheduler.add_request(update)
    assert request.prompt_token_ids == list(range(33)) + [33, 999, 100, 101]
    assert request.num_computed_tokens == 32
    assert scheduler.num_waiting_for_streaming_input == 0
    assert not scheduler.update_from_output(lookahead, _output(request, 777))
    resumed = scheduler.schedule()
    assert request.request_id in resumed.finished_req_ids
    assert scheduler.kv_cache_manager.get_block_ids(request.request_id)[0][:2] == blocks
    scheduler.update_from_output(resumed, _output(request, 888))
    assert list(request.output_token_ids) == [888]


@pytest.mark.parametrize("missing", [False, True])
def test_stale_turn_delta_fails_only_its_request(missing):
    scheduler = _scheduler()
    if not missing:
        resident = _request(list(range(33)))
        _stop_with_in_flight_decode(scheduler, resident)
    update = _request([999, 100])
    update.append_prefix_length = 123
    scheduler.add_request(update)
    assert "resident" not in scheduler.requests
    assert "resident" in scheduler._streaming_context_overflow
    scheduler.add_request(_request([10, 11], request_id="other"))
    assert scheduler.schedule().num_scheduled_tokens["other"] == 2
