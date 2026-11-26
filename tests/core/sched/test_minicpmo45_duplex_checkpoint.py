# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from array import array
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_omni.model_executor.models.minicpmo_4_5.duplex import checkpoint
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import CHECKPOINT_KEY, REPLAY_CONTEXT_KEY

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _buffer(turn_id, *, epoch=0, seq=1, marker=None):
    return {
        "duplex": {
            "data_plane": True,
            "epoch": epoch,
            "seq": seq,
            "turn_id": turn_id,
            "session_id": "session",
            "runtime_config": {"duplex_first_append_context_tokens": 8},
            CHECKPOINT_KEY: marker,
        }
    }


@pytest.fixture
def saved_checkpoint(request, monkeypatch):
    epoch = getattr(request, "param", 0)
    monkeypatch.setattr(checkpoint, "monotonic", lambda: 0)
    pool = BlockPool(256, enable_caching=False, hash_block_size=16)
    manager = FullAttentionManager(
        FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16),
        block_pool=pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=16,
    )
    scheduler = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            coordinator=SimpleNamespace(single_type_managers=(manager,)),
            enable_caching=False,
            block_pool=pool,
        ),
        connector=None,
        processed_step_seq=2,
        max_model_len=4096,
    )
    cache = checkpoint.MiniCPMO45DuplexCheckpoints(scheduler)
    source = SimpleNamespace(
        request_id="old",
        model_intermediate_buffer=_buffer(1, epoch=epoch),
        num_prompt_tokens=16,
        num_tokens=24,
        num_preemptions=0,
        num_computed_tokens=40,
        num_in_flight_tokens=4,
        last_sched_seq=2,
        _all_token_ids=list(range(40)),
    )
    target = SimpleNamespace(
        request_id="new",
        model_intermediate_buffer=_buffer(3, epoch=epoch + 1, marker={"history": "completed", "interrupted_turn": 2}),
        prompt_token_ids=[99] * 35,
        _all_token_ids=array("q", [99] * 35),
        num_computed_tokens=0,
        num_prompt_tokens=35,
    )
    target.model_intermediate_buffer["duplex"].update(
        {
            "scheduler_token_id": 99,
            "duplex_replay_token_budget": 16,
            REPLAY_CONTEXT_KEY: [{"token_budget": 12}, {"token_budget": 4}],
        }
    )
    update = SimpleNamespace(
        model_intermediate_buffer=_buffer(2, epoch=epoch, seq=2, marker={"history": "completed", "completed_turn": 1}),
        prompt_token_ids=[99] * 14,
    )
    cache.record_boundary(source, update, confirmed_end=23)
    source.model_intermediate_buffer = update.model_intermediate_buffer
    blocks = pool.get_new_blocks(3)
    manager.req_to_blocks[source.request_id] = blocks.copy()
    cache.retain_before_abort(source)
    manager.free(source.request_id)
    return cache, scheduler, manager, blocks, target


@pytest.mark.parametrize("saved_checkpoint", [0, 2], indirect=True)
def test_stage0_checkpoint_transfers_only_the_confirmed_prefix(saved_checkpoint):
    cache, _, manager, (first, tail, cancelled), target = saved_checkpoint
    assert (first.ref_cnt, tail.ref_cnt, cancelled.ref_cnt) == (1, 1, 0)

    cache.restore(target)
    assert manager.req_to_blocks[target.request_id] == [first, tail]
    assert target.num_computed_tokens == 23
    assert target.prompt_token_ids == list(range(23)) + [99] * 17
    assert target.model_intermediate_buffer["duplex"][CHECKPOINT_KEY] == {"reused_tokens": 23, "terminator": 23}
    assert target.model_intermediate_buffer["duplex"][REPLAY_CONTEXT_KEY] == [{"token_budget": 4}]
    manager.free(target.request_id)
    assert (first.ref_cnt, tail.ref_cnt, cancelled.ref_cnt) == (0, 0, 0)


@pytest.mark.parametrize("miss", ["expired", "in_flight", "edited_history", "different_session", "cache_pressure"])
def test_checkpoint_miss_preserves_the_full_replay(saved_checkpoint, miss, monkeypatch):
    cache, scheduler, manager, blocks, target = saved_checkpoint
    duplex = target.model_intermediate_buffer["duplex"]
    pressure = []
    if miss == "expired":
        monkeypatch.setattr(checkpoint, "monotonic", lambda: 16)
    elif miss == "in_flight":
        scheduler.processed_step_seq = 1
    elif miss == "edited_history":
        duplex[CHECKPOINT_KEY]["history"] = "edited"
    elif miss == "different_session":
        duplex["session_id"] = "other"
    elif miss == "cache_pressure":
        pool = manager.block_pool
        pressure = pool.get_new_blocks(pool.get_num_free_blocks() - 1)
    cache.restore(target)
    assert target.num_computed_tokens == 0
    assert target.prompt_token_ids == [99] * 35
    assert duplex[REPLAY_CONTEXT_KEY] == [{"token_budget": 12}, {"token_budget": 4}]
    manager.block_pool.free_blocks(pressure)
    monkeypatch.setattr(checkpoint, "monotonic", lambda: 16)
    cache.sweep()
    assert all(block.ref_cnt == 0 for block in blocks)
