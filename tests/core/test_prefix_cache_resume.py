# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regression for prefix delivery across real vLLM preemption."""

from types import SimpleNamespace

import pytest
import torch
from transformers import GPT2Config
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_omni.core.prefix_cache.adapter import PrefixCacheSchedulerAdapter
from vllm_omni.core.prefix_cache.group_view import FullAttentionGroupView
from vllm_omni.core.prefix_cache.interface import HIDDEN_KEY, PrefixCacheConfig
from vllm_omni.core.prefix_cache.manager import OmniPrefixCacheManager

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

BLOCK_SIZE = 4
NUM_BLOCKS = 16
STEP_BUDGET = 8
PROMPT = list(range(10, 30))


def _scheduler(model_path: str) -> Scheduler:
    model = ModelConfig(model=model_path, dtype="float16", max_model_len=64, skip_tokenizer_init=True)
    scheduler = SchedulerConfig(
        max_num_seqs=4,
        max_num_batched_tokens=STEP_BUDGET,
        max_model_len=64,
        enable_chunked_prefill=True,
        is_encoder_decoder=model.is_encoder_decoder,
        watermark=0.0,
    )
    cache = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=True,
    )
    config = VllmConfig(
        model_config=model,
        scheduler_config=scheduler,
        cache_config=cache,
        parallel_config=ParallelConfig(),
        device_config=DeviceConfig(device="cpu"),
    )
    spec = FullAttentionSpec(block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32)
    kv_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
    )
    cache.num_gpu_blocks = NUM_BLOCKS
    register_all_kvcache_specs(config)
    instance = Scheduler(
        vllm_config=config,
        kv_cache_config=kv_config,
        block_size=BLOCK_SIZE,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(config),
    )
    instance.use_v2_model_runner = False
    return instance


def _request(req_id: str) -> Request:
    sampling = SamplingParams(ignore_eos=True, max_tokens=8)
    sampling.update_from_generation_config({}, 2)
    return Request(
        request_id=req_id,
        prompt_token_ids=PROMPT.copy(),
        sampling_params=sampling,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


class _CPUCacheRun:
    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
        self.adapter = PrefixCacheSchedulerAdapter()
        self.manager = OmniPrefixCacheManager(
            PrefixCacheConfig(
                num_blocks=NUM_BLOCKS,
                block_size=BLOCK_SIZE,
                staging_capacity_tokens=STEP_BUDGET,
            ),
            eager=True,
        )

    def close(self):
        self.manager.shutdown()

    def step(self):
        output = self.scheduler.schedule()
        req_ids = list(output.num_scheduled_tokens)
        starts = {data.req_id: data.num_computed_tokens for data in output.scheduled_new_reqs}
        starts.update(zip(output.scheduled_cached_reqs.req_ids, output.scheduled_cached_reqs.num_computed_tokens))
        block_ids = {
            req_id: self.scheduler.kv_cache_manager.get_blocks(req_id).get_block_ids()[0] for req_id in req_ids
        }
        width = max((len(ids) for ids in block_ids.values()), default=1)
        block_table = torch.zeros((len(req_ids), width), dtype=torch.int32)
        for index, req_id in enumerate(req_ids):
            block_table[index, : len(block_ids[req_id])] = torch.tensor(block_ids[req_id], dtype=torch.int32)
        input_batch = SimpleNamespace(
            req_ids=req_ids,
            req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
            num_computed_tokens_cpu=[starts[req_id] for req_id in req_ids],
            block_table=[SimpleNamespace(block_table=SimpleNamespace(cpu=block_table))],
        )
        view = FullAttentionGroupView(input_batch, BLOCK_SIZE)
        layout = self.adapter.build_write_layout(view, num_scheduled_tokens=output.num_scheduled_tokens)
        self.manager.new_step_starts(self.adapter.translate_step(output))
        positions = [
            pos
            for req_id in req_ids
            for pos in range(starts[req_id], starts[req_id] + output.num_scheduled_tokens[req_id])
        ]
        rows = torch.tensor(positions, dtype=torch.float32).unsqueeze(1)
        step_id = self.manager.save_outputs(
            rows,
            {},
            num_tokens_unpadded=len(positions),
            num_tokens_padded=len(positions),
            write_layout=layout,
        )
        raw = self.manager.materialize(step_id, req_ids)
        delivery = self.manager.delivery_view(raw, req_ids)
        delivered = delivery.hidden_states
        self.manager.ack_delivery(delivery)
        self.scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
                sampled_token_ids=[[] for _ in req_ids],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
        return output, delivered, layout


@pytest.fixture
def cpu_run(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    model_path = tmp_path / "model"
    GPT2Config(
        n_embd=32,
        n_layer=1,
        n_head=1,
        n_positions=64,
        vocab_size=100,
        bos_token_id=1,
        eos_token_id=2,
        architectures=["GPT2LMHeadModel"],
    ).save_pretrained(model_path)
    init_none_hash(sha256)
    run = _CPUCacheRun(_scheduler(str(model_path)))
    yield run
    run.close()


def _positions(delivered, req_id):
    return [int(row) for row in delivered[req_id][:, 0].tolist()]


def _preempt(run, request):
    run.scheduler.running.remove(request)
    run.scheduler._preempt_request(request, 0.0)
    assert request.status == RequestStatus.PREEMPTED


def _resume_hit_end(output, req_id):
    for data in output.scheduled_new_reqs:
        if data.req_id == req_id:
            return data.num_computed_tokens
    index = output.scheduled_cached_reqs.req_ids.index(req_id)
    assert req_id in output.scheduled_cached_reqs.resumed_req_ids
    return output.scheduled_cached_reqs.num_computed_tokens[index]


def test_growing_hit_delivers_only_missing_absolute_rows(cpu_run):
    run = cpu_run
    victim = _request("victim")
    run.scheduler.add_request(victim)
    first, delivered, _ = run.step()
    assert first.num_scheduled_tokens == {"victim": 8}
    assert _positions(delivered, "victim") == list(range(8))

    _preempt(run, victim)
    helper = _request("helper")
    run.scheduler.add_request(helper)
    run.scheduler.waiting.remove_request(victim)
    run.scheduler.waiting.add_request(victim)
    middle, delivered, _ = run.step()
    assert middle.scheduled_new_reqs[0].req_id == "helper"
    assert middle.scheduled_new_reqs[0].num_computed_tokens == 8
    assert _positions(delivered, "helper") == list(range(16))

    run.scheduler.finish_requests("helper", RequestStatus.FINISHED_ABORTED)
    resumed, delivered, _ = run.step()
    assert _resume_hit_end(resumed, "victim") == 16
    assert _positions(delivered, "victim") == list(range(8, 20))


def test_shrinking_hit_saves_replay_without_redelivering_it(cpu_run):
    run = cpu_run
    victim = _request("victim")
    run.scheduler.add_request(victim)
    _, delivered, _ = run.step()
    assert _positions(delivered, "victim") == list(range(8))
    _, delivered, _ = run.step()
    assert _positions(delivered, "victim") == list(range(8, 16))

    blocks = run.scheduler.kv_cache_manager.get_blocks("victim").get_block_ids()[0]
    assert len(blocks) == 4
    _preempt(run, victim)
    run.scheduler.kv_cache_manager.block_pool.evict_blocks(set(blocks[2:]))
    resumed, delivered, layout = run.step()
    assert _resume_hit_end(resumed, "victim") == 8
    assert _positions(delivered, "victim") == []

    replay_slots = layout.slots_cpu
    assert run.manager._pool.rows(HIDDEN_KEY, replay_slots)[:, 0].tolist() == list(range(8, 16))
    _, delivered, _ = run.step()
    assert _positions(delivered, "victim") == list(range(16, 20))
