# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA scatter/gather and fixed-address replay with the native KV allocator."""

from functools import partial

import pytest
import torch

from tests.diffusion.ar_diffusion.test_segmented_kv import append as _append
from tests.diffusion.ar_diffusion.test_segmented_kv import make_cache, make_state, rows
from tests.helpers.mark import hardware_marks

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    *hardware_marks(res={"cuda": ["H100", "B200"]}, num_cards=1),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


@pytest.fixture
def cuda_state():
    # Keep the unindexed device: allocations resolve it to the current GPU.
    cache = make_cache(device=torch.device("cuda"), dtype=torch.bfloat16)
    free = cache.manager.block_pool.get_num_free_blocks()
    owner = make_state(cache)
    for pool in cache._k_pools + cache._v_pools:
        pool.fill_(float("nan"))
    yield owner.segmented("main")
    owner.close()
    assert cache.manager.block_pool.get_num_free_blocks() == free


_rows = partial(rows, device="cuda", dtype=torch.bfloat16)


@pytest.mark.parametrize("lengths", [(15, 3), (17, 5)])
def test_cuda_partial_pages_exclude_poisoned_padding(cuda_state, lengths):
    state = cuda_state
    free = state.cache.manager.block_pool.get_num_free_blocks()
    _append(state, lengths, offset=2)
    assert free - state.cache.manager.block_pool.get_num_free_blocks() == sum((n + 15) // 16 for n in lengths)
    state.prepare_layer_views(4, 2)
    for layer in range(state.cache.num_layers):
        media, condition = _rows(4, 5), _rows(2, -3)
        key, value = state.assemble(layer, condition, -condition, media, -media)
        expected = torch.cat((_rows(sum(lengths), 2 + layer * 100), media, condition))
        torch.testing.assert_close(key, expected, rtol=0, atol=0)
        torch.testing.assert_close(value, -expected, rtol=0, atol=0)
        assert torch.isfinite(key).all() and torch.isfinite(value).all()
        query = _rows(3, 0.1).transpose(0, 1)
        actual = torch.nn.functional.scaled_dot_product_attention(query, key.transpose(0, 1), value.transpose(0, 1))
        reference = torch.nn.functional.scaled_dot_product_attention(
            query, expected.transpose(0, 1), -expected.transpose(0, 1)
        )
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_cuda_partial_commit_rollback_preserves_history(cuda_state):
    state = cuda_state
    old = _append(state, (17, 5), offset=7)
    free = state.cache.manager.block_pool.get_num_free_blocks()
    state.prepare_layer_views(6, 2)
    pending = state.begin_commit((2, 4))
    state.assemble(0, _rows(2), _rows(2), _rows(6, 99), _rows(6, 99))
    state.stage(0)
    with pytest.raises(RuntimeError, match="missing transformer layers"):
        state.commit(retain=pending)
    state.rollback()
    assert state.handles == old
    assert state.cache.manager.block_pool.get_num_free_blocks() == free
    state.prepare_layer_views(2, 0)
    for layer in range(state.cache.num_layers):
        key, value = state.assemble(layer, _rows(0), _rows(0), _rows(2), _rows(2))
        expected = _rows(22, 7 + 100 * layer)
        torch.testing.assert_close(key[:22], expected, rtol=0, atol=0)
        torch.testing.assert_close(value[:22], -expected, rtol=0, atol=0)


def test_cuda_graph_replays_assembled_views_with_stable_independent_storage(cuda_state):
    state = cuda_state
    handles = _append(state, (17, 5), offset=3)
    state.prepare_layer_views(4, 2)
    staging_key, staging_value = state._staging().layers[0]
    addresses = (staging_key.data_ptr(), staging_value.data_ptr())
    assert staging_key.untyped_storage().data_ptr() != staging_value.untyped_storage().data_ptr()
    assert state.cache._k_pools[0].untyped_storage().data_ptr() != state.cache._v_pools[0].untyped_storage().data_ptr()
    media, condition = _rows(4), _rows(2)
    media_value, condition_value = -media, -condition
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        state.assemble(0, condition, condition_value, media, media_value)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        key, value = state.assemble(0, condition, condition_value, media, media_value)
        captured_key, captured_value = key.clone(), value.clone()

    history = _rows(22, 3)
    for step in range(3):
        if step == 2:
            # A same-length history revision must be visible to an existing graph.
            state.retain(handles[::-1])
            history = torch.cat((history[17:], history[:17]))
        state.prepare_layer_views(4, 2)
        media.copy_(_rows(4, 2 + step))
        condition.copy_(_rows(2, -3 - step))
        media_value.copy_(-media)
        condition_value.copy_(-condition)
        graph.replay()
        expected = torch.cat((history, media, condition))
        torch.testing.assert_close(captured_key, expected, rtol=0, atol=0)
        torch.testing.assert_close(captured_value, -expected, rtol=0, atol=0)
        assert addresses == tuple(tensor.data_ptr() for tensor in state._staging().layers[0])
    assert state.history_gathers == 2
