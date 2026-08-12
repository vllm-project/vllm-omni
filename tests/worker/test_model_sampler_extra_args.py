# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``GPUARModelRunner._build_model_sampler_extra_args``.

Custom model samplers that set ``model_sampler_wants_extra_args = True``
(e.g. HunyuanImage3's per-request ``ar_task_mode``, #6088) receive one
``SamplingParams.extra_args`` entry per batch row, in ``input_batch.req_ids``
order — the same order ``_build_model_sampler_output_token_ids`` uses. Pure CPU.
"""

from types import SimpleNamespace

import pytest

from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _runner(*, req_ids, requests):
    runner = object.__new__(GPUARModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=req_ids)
    runner.requests = requests
    return runner


def _state(extra_args):
    return SimpleNamespace(sampling_params=SimpleNamespace(extra_args=extra_args))


def test_row_order_follows_req_ids():
    runner = _runner(
        req_ids=["b", "a"],
        requests={
            "a": _state({"ar_task_mode": "comprehension"}),
            "b": _state(None),
        },
    )
    assert runner._build_model_sampler_extra_args() == [None, {"ar_task_mode": "comprehension"}]


def test_missing_request_state_yields_none():
    runner = _runner(req_ids=["gone"], requests={})
    assert runner._build_model_sampler_extra_args() == [None]


def test_request_without_sampling_params_yields_none():
    runner = _runner(req_ids=["r0"], requests={"r0": SimpleNamespace(sampling_params=None)})
    assert runner._build_model_sampler_extra_args() == [None]


def test_empty_batch():
    runner = _runner(req_ids=[], requests={})
    assert runner._build_model_sampler_extra_args() == []
