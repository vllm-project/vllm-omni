# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.worker.ubatch_utils import UBatchSlice

from vllm_omni.worker.v1_compat.misc import split_attn_metadata

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _metadata(*, positions=True):
    starts = torch.tensor([0, 3, 5, 9], dtype=torch.int32)
    seq_lens = torch.tensor([8, 5, 10], dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=starts.clone(),
        query_start_loc_cpu=starts.clone(),
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens.clone(),
        num_reqs=3,
        num_actual_tokens=9,
        max_query_len=4,
        max_seq_len=32,
        block_table_tensor=torch.arange(12, dtype=torch.int32).reshape(3, 4),
        slot_mapping=torch.arange(9, dtype=torch.int64) + 100,
        positions=torch.arange(9, dtype=torch.int64) + 20 if positions else None,
    )


def test_split_metadata_spans_a_request_without_mutating_the_batch():
    metadata = _metadata()
    originals = {
        name: getattr(metadata, name).clone()
        for name in (
            "query_start_loc",
            "query_start_loc_cpu",
            "seq_lens",
            "seq_lens_cpu_upper_bound",
            "block_table_tensor",
            "slot_mapping",
            "positions",
        )
    }
    slices = [UBatchSlice(slice(0, 2), slice(0, 4)), UBatchSlice(slice(1, 3), slice(4, 9))]

    first, second = split_attn_metadata(slices, metadata)

    for result, query_starts, seq_lens, slots, block_rows, max_query_len in (
        (first, [0, 3, 4], [8, 4], slice(0, 4), slice(0, 2), 3),
        (second, [0, 1, 5], [5, 10], slice(4, 9), slice(1, 3), 4),
    ):
        assert result.query_start_loc.tolist() == query_starts
        assert result.query_start_loc_cpu.tolist() == query_starts
        assert result.seq_lens.tolist() == seq_lens
        assert result.seq_lens_cpu_upper_bound.tolist() == seq_lens
        assert result.num_reqs == 2
        assert result.num_actual_tokens == slots.stop - slots.start
        assert result.max_query_len == max_query_len
        assert result.max_seq_len == 32
        torch.testing.assert_close(result.block_table_tensor, metadata.block_table_tensor[block_rows])
        torch.testing.assert_close(result.slot_mapping, metadata.slot_mapping[slots])
        torch.testing.assert_close(result.positions, metadata.positions[slots])
    for name, original in originals.items():
        torch.testing.assert_close(getattr(metadata, name), original)


@pytest.mark.parametrize("positions", [False, True])
def test_split_inside_one_request_keeps_only_its_tokens(positions):
    metadata = _metadata(positions=positions)

    (result,) = split_attn_metadata([UBatchSlice(slice(0, 1), slice(1, 2))], metadata)

    assert result.query_start_loc.tolist() == [0, 1]
    assert result.query_start_loc_cpu.tolist() == [0, 1]
    assert result.seq_lens.tolist() == [7]
    assert result.seq_lens_cpu_upper_bound.tolist() == [7]
    assert result.num_reqs == result.num_actual_tokens == result.max_query_len == 1
    assert result.max_seq_len == 32
    assert result.slot_mapping.tolist() == [101]
    assert result.positions.tolist() == [21] if positions else result.positions is None


def test_split_preserves_graph_padding_slots():
    metadata = _metadata()
    metadata.slot_mapping = torch.cat((metadata.slot_mapping, torch.tensor([-1, -1])))
    metadata.positions = torch.cat((metadata.positions, torch.tensor([0, 0])))

    (result,) = split_attn_metadata([UBatchSlice(slice(0, 3), slice(0, 11))], metadata)

    assert result.query_start_loc.tolist() == [0, 3, 5, 9]
    assert result.seq_lens.tolist() == [8, 5, 10]
    assert result.num_actual_tokens == 11
    assert result.slot_mapping.tolist() == [*range(100, 109), -1, -1]
    assert result.positions.tolist() == [*range(20, 29), 0, 0]
    assert result.max_seq_len == 32


@pytest.mark.parametrize(
    ("request_slice", "token_slice", "message"),
    [
        (slice(0, 0), slice(0, 1), "is empty"),
        (slice(0, 1), slice(0, 0), "is empty"),
        (slice(1, 2), slice(0, 2), "Token slice start outside of first request"),
    ],
)
def test_split_rejects_invalid_microbatches(request_slice, token_slice, message):
    with pytest.raises(AssertionError, match=message):
        split_attn_metadata([UBatchSlice(request_slice, token_slice)], _metadata())
