# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""A pooling stage's request output must not break the stage-pool metrics.

vLLM's ``OutputProcessor._new_request_output`` unwraps the one-element
list before building ``PoolingRequestOutput``, so a pooling stage's
``request_output.outputs`` is a single ``PoolingOutput`` rather than a list of
completions. The stage-pool metrics used to iterate it as a sequence and
raised ``TypeError: 'PoolingOutput' object is not iterable``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.outputs import PoolingOutput, PoolingRequestOutput

from vllm_omni.engine.stage_pool import StagePool

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _pool(final_output_type: str = "audio") -> StagePool:
    client = SimpleNamespace(stage_type="llm", final_output_type=final_output_type)
    return StagePool(0, [client])  # type: ignore[arg-type]


def _pooling_request_output(data: torch.Tensor) -> PoolingRequestOutput:
    """A pooling stage's request output, built from the upstream classes."""
    return PoolingRequestOutput(
        request_id="req-1",
        outputs=PoolingOutput(data=data),
        prompt_token_ids=[1, 2],
        num_cached_tokens=0,
        finished=True,
    )


def test_has_non_empty_output_accepts_single_pooling_output() -> None:
    pool = _pool()

    assert pool.has_non_empty_output(_pooling_request_output(torch.zeros(2, 2))) is True


def test_has_non_empty_output_rejects_empty_pooling_output() -> None:
    pool = _pool()

    assert pool.has_non_empty_output(_pooling_request_output(torch.zeros(0, 2))) is False


def test_has_non_empty_output_accepts_scalar_pooling_output() -> None:
    # A pooled score can be a 0-d tensor: its shape is empty, so it is not an
    # empty value.
    pool = _pool()

    assert pool.has_non_empty_output(_pooling_request_output(torch.zeros(()))) is True


def test_has_non_empty_output_reads_generator_backed_outputs_once() -> None:
    # The pooled check must not consume ``outputs`` before the text check reads
    # them: a generator-backed container would be exhausted by then and the
    # completion would be reported as empty.
    pool = _pool(final_output_type="text")
    completion = SimpleNamespace(text="hello", token_ids=[1, 2])
    request_output = SimpleNamespace(outputs=iter([completion]))

    assert pool.has_non_empty_output(request_output) is True


def test_record_output_timestamps_records_populated_pooling_output() -> None:
    pool = _pool()
    assert pool.collect_stage_metrics is True

    request_output = _pooling_request_output(torch.zeros(2, 2))
    output_ts = 1000.0
    pool.record_output_timestamps([request_output], output_ts=output_ts)

    metrics = pool.build_stage_metrics(
        [request_output],
        submit_ts=output_ts,
        request_timestamp=output_ts - 0.5,
        replica_id=0,
    )

    assert metrics.serving_time_to_first_output_ms == pytest.approx(500.0)


def test_record_output_timestamps_skips_empty_pooling_output() -> None:
    pool = _pool()
    assert pool.collect_stage_metrics is True

    request_output = _pooling_request_output(torch.zeros(0, 2))
    output_ts = 1000.0
    pool.record_output_timestamps([request_output], output_ts=output_ts)

    metrics = pool.build_stage_metrics(
        [request_output],
        submit_ts=output_ts,
        request_timestamp=output_ts - 0.5,
        replica_id=0,
    )

    assert metrics.serving_time_to_first_output_ms == 0.0
