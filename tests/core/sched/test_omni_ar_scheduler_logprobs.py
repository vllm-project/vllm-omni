"""Tests for the AR sampled-token logprob contract."""

from __future__ import annotations

import numpy as np
import pytest
from vllm.v1.outputs import LogprobsLists

from vllm_omni.core.sched.omni_ar_scheduler import _slice_sampled_logprobs

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _logprobs(
    token_ids: list[list[int]],
    values: list[list[float]],
    *,
    cumulative_rows: list[int] | None = None,
) -> LogprobsLists:
    return LogprobsLists(
        logprob_token_ids=np.asarray(token_ids, dtype=np.int32),
        logprobs=np.asarray(values, dtype=np.float32),
        sampled_token_ranks=np.zeros(len(token_ids), dtype=np.int32),
        cu_num_generated_tokens=cumulative_rows,
    )


def test_slices_requested_rows_and_preserves_sampled_token_values() -> None:
    all_logprobs = _logprobs(
        [[3, 8], [4, 9], [11, 2], [12, 1]],
        [[-0.3, -1.0], [-0.4, -1.1], [-0.11, -2.0], [-0.12, -2.1]],
        cumulative_rows=[0, 2],
    )

    sliced = _slice_sampled_logprobs(all_logprobs, req_index=1, sampled_token_ids=[11, 12])

    np.testing.assert_array_equal(sliced.logprob_token_ids[:, 0], [11, 12])
    np.testing.assert_allclose(sliced.logprobs[:, 0], [-0.11, -0.12])


def test_rejects_missing_logprobs() -> None:
    with pytest.raises(RuntimeError, match="model runner returned none"):
        _slice_sampled_logprobs(None, req_index=0, sampled_token_ids=[7])


def test_rejects_row_count_mismatch() -> None:
    logprobs = _logprobs([[7, 1]], [[-0.7, -1.0]])

    with pytest.raises(RuntimeError, match="row count does not match"):
        _slice_sampled_logprobs(logprobs, req_index=0, sampled_token_ids=[7, 8])


def test_rejects_sampled_token_mismatch() -> None:
    logprobs = _logprobs([[9, 7]], [[-0.9, -1.0]])

    with pytest.raises(RuntimeError, match="generated_token=7 logprob_token=9"):
        _slice_sampled_logprobs(logprobs, req_index=0, sampled_token_ids=[7])


def test_rejects_non_finite_sampled_logprob() -> None:
    logprobs = _logprobs([[7, 1]], [[float("nan"), -1.0]])

    with pytest.raises(RuntimeError, match="non-finite values at rows \\[0\\]"):
        _slice_sampled_logprobs(logprobs, req_index=0, sampled_token_ids=[7])
