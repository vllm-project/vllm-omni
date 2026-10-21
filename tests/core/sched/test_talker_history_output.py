# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Scheduler contract for request-local Talker replay.

The scheduler publishes generated KV blocks with owner-private hashes only. It
does not receive, validate, serialize, or replay RVQ codec decisions.
"""

import pytest
import torch
from transformers import GPT2Config
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import RepetitionDetectionParams, SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from tests.helpers.fixtures import ipc
from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler, OmniARScheduler
from vllm_omni.core.sched.output import OmniNewRequestData
from vllm_omni.model_executor.models.qwen3_omni.talker_history import get_talker_request_block_hasher
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.request import OmniRequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
executor_roundtrip = ipc.executor_roundtrip


@pytest.mark.parametrize("min_tokens", [0, 4])
@pytest.mark.parametrize("tokens", [[7, 7], [7, 7, 7], [7, 7, 7, 7], [9], [8], [1, 2, 3, 4, 5, 6]])
def test_native_stop_order_is_unchanged(min_tokens, tokens):
    params = SamplingParams(
        max_tokens=6,
        min_tokens=min_tokens,
        stop_token_ids=[8],
        repetition_detection=RepetitionDetectionParams(max_pattern_size=1, min_count=3),
    )
    params._eos_token_id = 9
    request = OmniRequest(request_id="probe", prompt_token_ids=[100, 101], sampling_params=params, pooling_params=None)
    request.append_output_token_ids(tokens)
    assert check_stop(request, 64) == request.is_finished()


@pytest.fixture(params=[OmniARScheduler, OmniARAsyncScheduler])
def scheduler(request, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    path = tmp_path / "model"
    GPT2Config(
        n_embd=32,
        n_layer=1,
        n_head=1,
        n_positions=64,
        vocab_size=128,
        bos_token_id=1,
        eos_token_id=2,
        architectures=["GPT2LMHeadModel"],
    ).save_pretrained(path)
    model = ModelConfig(model=str(path), dtype="float16", max_model_len=64, skip_tokenizer_init=True)
    cache = CacheConfig(block_size=4, enable_prefix_caching=True)
    cache.num_gpu_blocks = 24
    config = VllmConfig(
        model_config=model,
        cache_config=cache,
        parallel_config=ParallelConfig(),
        device_config=DeviceConfig(device="cpu"),
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=16,
            max_model_len=64,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
            watermark=0.0,
            async_scheduling=request.param is OmniARAsyncScheduler,
        ),
    )
    register_all_kvcache_specs(config)
    init_none_hash(sha256)
    kv_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=4,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    return request.param(
        vllm_config=config,
        kv_cache_config=kv_config,
        block_size=4,
        hash_block_size=4,
        log_stats=False,
        structured_output_manager=StructuredOutputManager(config),
    )


def _request(req_id):
    return OmniRequest(
        request_id=req_id,
        prompt_token_ids=[100, 101, 102],
        sampling_params=SamplingParams(max_tokens=16, ignore_eos=True),
        pooling_params=None,
        block_hasher=get_talker_request_block_hasher(4, sha256),
        cache_salt="fixed-conditioning",
    )


def _output(scheduled, *, token=7):
    ids = list(scheduled.num_scheduled_tokens)
    return OmniModelRunnerOutput(
        req_ids=ids,
        req_id_to_index={rid: i for i, rid in enumerate(ids)},
        sampled_token_ids=[[token] for _ in ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def test_generated_publication_needs_no_scheduler_codec_history(scheduler, executor_roundtrip):
    request = _request("producer")
    scheduler.add_request(request)
    for _ in range(6):
        scheduled = scheduler.schedule()
        output = executor_roundtrip(_output(scheduled))
        scheduler.update_from_output(scheduled, output)
        assert not hasattr(request, "talker_codec_inputs")
        assert request.num_in_flight_tokens == 0
    assert request.num_tokens == 9
    assert len(request.block_hashes) == 2
    assert scheduler.kv_cache_manager.get_computed_blocks(request)[1] == 8


def test_replacement_stale_output_does_not_append_or_publish(scheduler):
    request = _request("replaced")
    scheduler.add_request(request)
    scheduled = scheduler.schedule()
    output = _output(scheduled)
    old_owner = request._omni_prefix_cache_owner
    scheduler._reset_streaming_session_replacement_state(request)
    scheduler._accept_prefix_cache_replacement(request)
    assert request._omni_prefix_cache_owner != old_owner
    scheduler.update_from_output(scheduled, output)
    assert request.num_output_tokens == 0
    assert request.num_in_flight_tokens == request.num_stale_output_tokens == 0


def test_preemption_resume_carries_no_talker_history_wire(scheduler, executor_roundtrip):
    request = _request("resume")
    scheduler.add_request(request)
    for _ in range(6):
        scheduled = scheduler.schedule()
        assert not hasattr(scheduled.scheduled_cached_reqs, "talker_codec_inputs")
        scheduler.update_from_output(scheduled, _output(scheduled))
    scheduler.running.remove(request)
    scheduler._preempt_request(request, 0.0)
    resumed = scheduler.schedule()
    cached = resumed.scheduled_cached_reqs
    assert cached.resumed_req_ids == {"resume"}
    assert cached.num_computed_tokens == [8]
    assert not hasattr(executor_roundtrip(cached), "talker_codec_inputs")
    scheduler.update_from_output(resumed, _output(resumed))
    assert request.num_output_tokens == 7

    new = OmniNewRequestData.from_request(request, scheduler.kv_cache_manager.get_blocks("resume").get_block_ids())
    assert not hasattr(new, "talker_codec_inputs")
    assert not hasattr(executor_roundtrip(new), "talker_codec_inputs")


@pytest.mark.parametrize("drop_stale", [False, True])
def test_preemption_delivers_inflight_output_unless_native_drop_mode_is_requested(scheduler, drop_stale):
    request = _request("inflight-resume")
    scheduler.add_request(request)
    for _ in range(6):
        scheduled = scheduler.schedule()
        scheduler.update_from_output(scheduled, _output(scheduled))
    scheduled = scheduler.schedule()
    output = _output(scheduled)
    codes = torch.tensor([[7, 2, 3, 4]])
    output.multimodal_outputs = [{"codes.audio": codes}]
    scheduler.running.remove(request)
    scheduler._preempt_request(request, 0.0, drop_stale_output=drop_stale)
    result = scheduler.update_from_output(scheduled, output)
    emitted = [item for batch in result.values() for item in batch.outputs]
    if drop_stale:
        assert emitted == []
        assert request.num_output_tokens == 6
    else:
        assert len(emitted) == 1
        torch.testing.assert_close(emitted[0].multimodal_output["codes.audio"], codes)
        assert emitted[0].new_token_ids == [7]
        assert request.num_output_tokens == 7
    resumed = scheduler.schedule()
    assert resumed.scheduled_cached_reqs.resumed_req_ids == {request.request_id}
    assert not hasattr(resumed.scheduled_cached_reqs, "talker_codec_inputs")


@pytest.mark.parametrize("terminal_kind", ["eos", "stop", "length", "model_length"])
@pytest.mark.parametrize("output_count", [1, 2, 5])
def test_terminal_primary_uses_native_stop_without_codec_marker(scheduler, terminal_kind, output_count, monkeypatch):
    request = _request("terminal")
    if terminal_kind == "eos":
        request.sampling_params._eos_token_id = 9
    elif terminal_kind == "stop":
        request.sampling_params.stop_token_ids = [9]
    elif terminal_kind == "length":
        request.max_tokens = output_count
    else:
        scheduler.max_model_len = 3 + output_count
    scheduler.add_request(request)
    for _ in range(output_count - 1):
        scheduled = scheduler.schedule()
        scheduler.update_from_output(scheduled, _output(scheduled))
    frontier = request.num_tokens
    cache_blocks = scheduler.kv_cache_manager.cache_blocks

    def publish(req, computed):
        assert computed <= frontier
        return cache_blocks(req, computed)

    monkeypatch.setattr(scheduler.kv_cache_manager, "cache_blocks", publish)
    scheduled = scheduler.schedule()
    scheduler.update_from_output(scheduled, _output(scheduled, token=9))
    assert request.output_token_ids[-1] == 9
    assert request.num_output_tokens == output_count
    assert request.status == (
        RequestStatus.FINISHED_STOPPED if terminal_kind in ("eos", "stop") else RequestStatus.FINISHED_LENGTH_CAPPED
    )
    assert "terminal" not in scheduler.requests
    # A hash may be computed at append time, but native KV publication must
    # exclude the unexecuted terminal. The extra token permits lookup through
    # that block, rather than hiding it behind native's leave-one-token rule.
    probe = _request("terminal-lookup")
    probe._omni_prefix_cache_owner = request._omni_prefix_cache_owner
    probe.append_output_token_ids([*request.output_token_ids, 11])
    assert scheduler.kv_cache_manager.get_computed_blocks(probe)[1] == frontier // 4 * 4
