# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Auxiliary capture runs after eval and profiling does not start a producer."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.ming_flash_omni.ming_flash_omni_talker import (
    MingFlashOmniTalkerForConditionalGeneration as Talker,
)
from vllm_omni.model_executor.models.ming_flash_omni.talker_batch import MingTalkerBatchEngine, PackedQwen2, _capture

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("error", [RuntimeError("operator failure"), torch.OutOfMemoryError("out of memory")])
def test_graph_capture_failure_is_not_retried(error, mocker):
    mocker.patch("torch.accelerator.empty_cache")
    graph = mocker.patch("torch.cuda.CUDAGraph")
    mocker.patch("torch.cuda.graph", return_value=nullcontext())
    body = mocker.Mock(side_effect=error)

    with pytest.raises(type(error)) as caught:
        _capture(body, pool=None, stream=None)

    assert caught.value is error
    body.assert_called_once()
    graph.assert_called_once()


def test_capture_requires_eval_and_release_closes_idle_runtime(mocker):
    engine = mocker.Mock()
    model = SimpleNamespace(
        training=True,
        device=torch.device("cuda"),
        _batch_rows=2,
        _batch_engine=None,
        _batch_vae=None,
        _stream_jobs={},
        _build_batch_engine=mocker.Mock(return_value=engine),
    )
    with pytest.raises(RuntimeError, match="eval"):
        Talker.capture_auxiliary_graphs(model)
    model._build_batch_engine.assert_not_called()
    model.training = False
    Talker.capture_auxiliary_graphs(model)
    Talker.capture_auxiliary_graphs(model)
    model._build_batch_engine.assert_called_once()
    engine.start.assert_not_called()
    model._stream_jobs["active"] = object()
    with pytest.raises(RuntimeError, match="active requests"):
        Talker.release_auxiliary_graphs(model)
    model._stream_jobs.clear()
    Talker.release_auxiliary_graphs(model)
    engine.close.assert_called_once()
    assert model._batch_engine is None and model._batch_vae is None


def test_idle_engine_start_is_idempotent_and_close_joins_thread(mocker):
    from threading import Condition

    engine = object.__new__(MingTalkerBatchEngine)
    engine.closed = False
    engine.thread = None
    engine.cond = Condition()
    engine.stream = mocker.Mock()

    def idle():
        with engine.cond:
            engine.cond.wait_for(lambda: engine.closed)

    engine._run = idle
    engine.start()
    thread = engine.thread
    assert thread is not None
    engine.start()
    assert engine.thread is thread
    engine.close()
    assert not thread.is_alive()
    engine.stream.synchronize.assert_called_once()
    with pytest.raises(RuntimeError, match="closed"):
        engine.start()


@pytest.mark.parametrize("live_rows", [(0, 7), (0, 3, 7), (2, 4, 6, 7)])
def test_compaction_moves_request_state_and_swaps_kv_ownership(live_rows):
    engine = object.__new__(MingTalkerBatchEngine)
    engine.device = torch.device("cpu")
    engine.buckets = [1, 2, 4, 8]
    engine.hs = torch.arange(8 * 4).reshape(8, 1, 4).float()
    engine.his_lat = torch.arange(8 * 6).reshape(8, 2, 3).float()
    engine.positions = torch.arange(8) + 10
    engine.advance = torch.ones(8, dtype=torch.long)
    engine.sde_args = torch.arange(3 * 8).reshape(3, 8, 1, 1).float()
    engine.llm = SimpleNamespace(block_table=torch.arange(9 * 2).reshape(9, 2))
    requests = {row: SimpleNamespace(row=row, num_steps=20, generator=object()) for row in live_rows}
    engine.rows = requests.copy()
    engine.free_rows = [row for row in range(8) if row not in requests]
    states = {name: getattr(engine, name).clone() for name in ("hs", "his_lat", "positions", "advance", "sde_args")}
    original_table = engine.llm.block_table.clone()
    table_address = engine.llm.block_table.data_ptr()
    generators = {row: request.generator for row, request in requests.items()}

    engine._compact_rows()

    bucket = next(b for b in engine.buckets if b >= len(requests))
    assert max(engine.rows) < bucket
    assert engine.llm.block_table.data_ptr() == table_address
    assert sorted(engine.llm.block_table[:, 0].tolist()) == original_table[:, 0].tolist()
    torch.testing.assert_close(engine.llm.block_table[-1], original_table[-1])
    for old_row, request in requests.items():
        if old_row < bucket:
            assert request.row == old_row
        assert engine.rows[request.row] is request
        assert request.generator is generators[old_row]
        for name in ("hs", "his_lat", "positions", "advance"):
            torch.testing.assert_close(getattr(engine, name)[request.row], states[name][old_row])
        torch.testing.assert_close(engine.sde_args[:, request.row], states["sde_args"][:, old_row])
        torch.testing.assert_close(engine.llm.block_table[request.row], original_table[old_row])
    for row in live_rows:
        if row not in engine.rows:
            assert engine.positions[row] == 0 and engine.advance[row] == 0
    assert engine.free_rows == [row for row in range(8) if row not in engine.rows]

    # Completion after relocation releases the new logical row exactly once.
    for request in requests.values():
        engine._release(request)
        engine._release(request)
    assert engine.rows == {}
    assert sorted(engine.free_rows) == list(range(8))


def test_compaction_keeps_slots_when_the_graph_bucket_cannot_shrink():
    engine = object.__new__(MingTalkerBatchEngine)
    engine.buckets = [1, 2, 4, 8]
    engine.rows = {row: SimpleNamespace(row=row) for row in (0, 1, 3)}
    # No device buffers are required on the unchanged-bucket fast path.
    engine._compact_rows()
    assert list(engine.rows) == [0, 1, 3]


def test_packed_kv_writes_follow_the_relocated_attention_blocks(mocker):
    packed = object.__new__(PackedQwen2)
    packed.ring_blocks = packed.window = None
    packed.max_len = packed.BLOCK
    packed.num_heads = packed.num_kv_heads = 1
    packed.head_dim = 2
    packed._fa_version = 3
    packed.block_table = torch.tensor([[1], [0], [2]], dtype=torch.int32)
    packed.k_cache = [torch.zeros(3, packed.BLOCK, 1, 2)]
    packed.v_cache = [torch.zeros_like(packed.k_cache[0])]
    packed.model = SimpleNamespace(
        rotary_emb=lambda embeds, positions: (torch.ones(1, 1, 2), torch.zeros(1, 1, 2)),
        layers=[SimpleNamespace(self_attn=SimpleNamespace(scaling=1.0))],
        norm=lambda hidden: hidden,
    )
    packed._qkv = mocker.Mock(
        side_effect=lambda layer, hidden, *args: (hidden[:, None], hidden[:, None], hidden[:, None] * 2)
    )
    packed._out = mocker.Mock(side_effect=lambda layer, hidden, output: output.flatten(1))

    def attention(query, keys, values, *args, block_table, seqused_k, **kwargs):
        return values[block_table[:, 0].long(), seqused_k.long() - 1]

    packed._attn = mocker.Mock(side_effect=attention)
    embeds = torch.tensor([[3.0, 7.0]])
    output = packed.forward(
        embeds,
        torch.tensor([3]),
        torch.tensor([0]),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32),
        packed.block_table[:1],
        1,
    )
    torch.testing.assert_close(packed.k_cache[0][1, 3, 0], embeds[0])
    assert not packed.k_cache[0][0].count_nonzero()
    torch.testing.assert_close(output, embeds * 2)
