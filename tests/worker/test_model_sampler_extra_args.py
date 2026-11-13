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
from vllm.sampling_params import SamplingParams

from vllm_omni.worker import sampling_utils
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


@pytest.mark.parametrize("with_extra_args", [False, True])
def test_opted_in_sampler_receives_host_sampling_params_in_row_order(mocker, with_extra_args):
    params = SamplingParams(repetition_penalty=1.05, extra_args={"ar_task_mode": "generation"})
    model = SimpleNamespace(model_sampler_wants_sampling_params=True, model_sampler_wants_extra_args=with_extra_args)
    model_sample = mocker.Mock(return_value="sampled")
    logits = torch.zeros(3, 8)
    metadata = SimpleNamespace()
    requests = {"a": SimpleNamespace(sampling_params=params), "b": SimpleNamespace(sampling_params=None)}
    result = call_model_sampler(
        model,
        model_sample,
        logits,
        metadata,
        input_batch=SimpleNamespace(req_ids=["b", "missing", "a"]),
        requests=requests,
    )
    kwargs: dict[str, object] = {"per_req_sampling_params": [None, None, params]}
    if with_extra_args:
        kwargs["per_req_extra_args"] = [None, None, params.extra_args]
    model_sample.assert_called_once_with(logits, metadata, **kwargs)
    assert result == "sampled"


@pytest.mark.parametrize("opted_in", [False, True])
def test_unsupported_deferred_rng_is_logged_without_changing_contract(mocker, opted_in):
    model = SimpleNamespace(model_sampler_wants_sampling_params=opted_in)
    model_sample = mocker.Mock(return_value="sampled")
    debug = mocker.patch.object(sampling_utils.logger, "debug_once")
    logits = torch.empty(0)
    metadata = SimpleNamespace()
    result = call_model_sampler(
        model, model_sample, logits, metadata, input_batch=SimpleNamespace(req_ids=[]), requests={}, defer_rng=True
    )
    kwargs = {"per_req_sampling_params": []} if opted_in else {}
    model_sample.assert_called_once_with(logits, metadata, **kwargs)
    debug.assert_called_once()
    assert result == "sampled"


def test_supported_deferred_rng_is_forwarded(mocker):
    model = SimpleNamespace(model_sampler_wants_sampling_params=True, model_sampler_can_defer_rng=True)
    model_sample = mocker.Mock(return_value="sampled")
    debug = mocker.patch.object(sampling_utils.logger, "debug_once")
    logits = torch.empty(0)
    metadata = SimpleNamespace()
    result = call_model_sampler(
        model, model_sample, logits, metadata, input_batch=SimpleNamespace(req_ids=[]), requests={}, defer_rng=True
    )
    model_sample.assert_called_once_with(logits, metadata, per_req_sampling_params=[], defer_rng=True)
    debug.assert_not_called()
    assert result == "sampled"
