# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from vllm.outputs import PoolingOutput, PoolingRequestOutput

from vllm_omni.engine.stage_pool import StagePool

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("data", [[], [0.1, 0.9]])
@pytest.mark.parametrize("output_type", [None, "text"])
def test_pooling_output_records_timestamps_without_audio(data, output_type) -> None:
    pool = StagePool(1, SimpleNamespace(final_output_type=output_type))
    output = PoolingRequestOutput(
        request_id="judge",
        outputs=PoolingOutput(torch.tensor(data)),
        prompt_token_ids=[1],
        num_cached_tokens=0,
        finished=True,
    )
    pool.record_output_timestamps([output], output_ts=10.0)
    pool.record_output_timestamps([output], output_ts=11.0)
    assert pool._output_timestamps_by_request == {"judge": [10.0, 11.0]}
    assert pool._non_empty_first_output_timestamps_by_request == ({"judge": 10.0} if data else {})
    assert pool._audio_frames_by_request == {}
    assert pool._audio_sample_rate_by_request == {}


@pytest.mark.parametrize("collect_metrics", [False, True])
@pytest.mark.parametrize("output_type", [None, "text"])
def test_pooling_output_builds_metrics_without_completion_fields(collect_metrics, output_type) -> None:
    pool = StagePool(1, SimpleNamespace(final_output_type=output_type))
    pool.collect_stage_metrics = collect_metrics
    output = PoolingRequestOutput(
        request_id="judge",
        outputs=PoolingOutput(torch.tensor([0.1, 0.9])),
        prompt_token_ids=[1],
        num_cached_tokens=0,
        finished=True,
    )
    pool.record_output_timestamps([output], output_ts=10.0)
    metrics = pool.build_stage_metrics([output], submit_ts=9.0, request_timestamp=8.0, replica_id=0)
    assert metrics.finish_reason is None
    assert metrics.num_tokens_out == 0
    assert metrics.audio_generated_frames == 0
    if collect_metrics:
        assert metrics.serving_time_to_first_output_ms == 2000.0
        assert pool._output_timestamps_by_request == {}
        assert pool._non_empty_first_output_timestamps_by_request == {}
