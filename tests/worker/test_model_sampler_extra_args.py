# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for per-request model sampler arguments.

Custom model samplers that set ``model_sampler_wants_extra_args = True``
(e.g. HunyuanImage3's per-request ``ar_task_mode``, #6088) receive one
``SamplingParams.extra_args`` entry per batch row, in ``input_batch.req_ids``
order — the same order ``_build_model_sampler_output_token_ids`` uses. Pure CPU.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.worker.sampling_utils import build_model_sampler_extra_args, call_model_sampler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _state(extra_args):
    return SimpleNamespace(sampling_params=SimpleNamespace(extra_args=extra_args))


def test_row_order_follows_req_ids():
    input_batch = SimpleNamespace(req_ids=["b", "a"])
    requests = {
        "a": _state({"ar_task_mode": "comprehension"}),
        "b": _state(None),
    }
    assert build_model_sampler_extra_args(input_batch, requests) == [None, {"ar_task_mode": "comprehension"}]


def test_missing_request_state_yields_none():
    assert build_model_sampler_extra_args(SimpleNamespace(req_ids=["gone"]), {}) == [None]


def test_request_without_sampling_params_yields_none():
    input_batch = SimpleNamespace(req_ids=["r0"])
    assert build_model_sampler_extra_args(input_batch, {"r0": SimpleNamespace(sampling_params=None)}) == [None]


def test_empty_batch():
    assert build_model_sampler_extra_args(SimpleNamespace(req_ids=[]), {}) == []


def test_opted_in_sampler_receives_per_request_extra_args(mocker):
    model = SimpleNamespace(model_sampler_wants_extra_args=True)
    model_sample = mocker.Mock(return_value="sampled")
    input_batch = SimpleNamespace(req_ids=["a", "b"])
    logits = torch.empty(0)
    metadata = SimpleNamespace()
    requests = {
        "a": _state(None),
        "b": _state({"ar_task_mode": "comprehension"}),
    }

    result = call_model_sampler(
        model,
        model_sample,
        logits,
        metadata,
        input_batch=input_batch,
        requests=requests,
    )

    assert result == "sampled"
    model_sample.assert_called_once_with(
        logits,
        metadata,
        per_req_extra_args=[None, {"ar_task_mode": "comprehension"}],
    )


def test_non_opted_in_sampler_keeps_two_argument_contract(mocker):
    model = SimpleNamespace(model_sampler_wants_extra_args=False)
    model_sample = mocker.Mock(return_value="sampled")
    logits = torch.empty(0)
    metadata = SimpleNamespace()

    result = call_model_sampler(
        model,
        model_sample,
        logits,
        metadata,
        input_batch=SimpleNamespace(req_ids=["a"]),
        requests={"a": _state({"ar_task_mode": "comprehension"})},
    )

    assert result == "sampled"
    model_sample.assert_called_once_with(logits, metadata)


@pytest.mark.parametrize(
    ("presence", "frequency", "expected"),
    [(set(), set(), True), ({"r0"}, set(), False), (set(), {"r1"}, False), (None, set(), False)],
)
def test_penalty_fast_path_requires_known_empty_cpu_sets(mocker, presence, frequency, expected):
    model = SimpleNamespace(model_sampler_wants_penalty_flags=True)
    sample = mocker.Mock()
    logits, metadata = torch.zeros(2, 8), SimpleNamespace()
    batch = SimpleNamespace(presence_penalties_reqs=presence, frequency_penalties_reqs=frequency)
    call_model_sampler(model, sample, logits, metadata, input_batch=batch, requests={})
    sample.assert_called_once_with(logits, metadata, skip_standard_penalties=expected)


def test_penalty_flags_follow_batch_changes(mocker):
    model = SimpleNamespace(model_sampler_wants_penalty_flags=True, model_sampler_wants_extra_args=True)
    sample = mocker.Mock()
    batch = SimpleNamespace(req_ids=["r"], presence_penalties_reqs=set(), frequency_penalties_reqs=set())
    for enabled in (False, True, False):
        batch.frequency_penalties_reqs = {"r"} if enabled else set()
        call_model_sampler(model, sample, None, None, input_batch=batch, requests={})
        sample.assert_called_with(None, None, per_req_extra_args=[None], skip_standard_penalties=not enabled)
