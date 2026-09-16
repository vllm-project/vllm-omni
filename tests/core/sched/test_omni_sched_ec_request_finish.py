# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""v0.28 EC request-finish contract in the Omni schedulers (#6606 review).

Upstream v0.28 ``_free_request()`` fires ``ec_connector.request_finished()``
BEFORE the encoder cache is freed (so the connector can inspect per-request
state), honors its delayed-free result, and returns the generated
``ec_transfer_params``; ``update_from_output()`` then carries them into the
``EngineCoreOutput``. ``OmniARScheduler`` previously skipped the hook and
always returned ``None``; both Omni schedulers discarded the second tuple
item, so a remote encoder-cache handle could never reach the frontend.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# Imports must run in this order: vllm_omni applies patches to vllm.v1.request
# before Request / RequestStatus are bound in this module.
# isort: off
import vllm_omni  # noqa: F401 - import for side effects (patch vLLM)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler

# isort: on

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

EC_PARAMS = {"remote_encoder_cache_handle": "ec-1"}


def _make_free_request_sched(*, ec_connector) -> tuple[OmniARScheduler, MagicMock]:
    """Minimal real-`_free_request` harness; `order` records the EC hook
    firing relative to the encoder-cache free."""
    order = MagicMock()
    sched = OmniARScheduler.__new__(OmniARScheduler)
    sched._omits_kv_transfer_cache = {}
    sched._connector_finished = lambda request: (False, None)
    sched.ec_connector = ec_connector
    if ec_connector is not None:
        order.attach_mock(ec_connector.request_finished, "request_finished")
    sched.encoder_cache_manager = MagicMock()
    order.attach_mock(sched.encoder_cache_manager.free, "encoder_cache_free")
    sched.finished_req_ids = set()
    sched._new_prompt_len_snapshot = {}
    sched.finished_req_ids_dict = None
    sched._should_transfer_kv_for_request = lambda req_id: False
    sched._free_blocks = MagicMock()
    sched._free_input_coordinator_request = MagicMock()
    sched.chunk_transfer_adapter = None
    return sched, order


class _FakeFinishedRequest:
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id

    def is_finished(self) -> bool:
        return True


def test_free_request_fires_ec_hook_before_encoder_cache_free() -> None:
    ec_connector = MagicMock()
    ec_connector.request_finished.return_value = (False, EC_PARAMS)
    sched, order = _make_free_request_sched(ec_connector=ec_connector)
    request = _FakeFinishedRequest("req-ec")

    kv_params, ec_params = sched._free_request(request)

    assert ec_params == EC_PARAMS
    ec_connector.request_finished.assert_called_once_with(request)
    hook_calls = [name for name, *_ in order.mock_calls]
    assert hook_calls.index("request_finished") < hook_calls.index("encoder_cache_free")
    sched._free_blocks.assert_called_once_with(request)


def test_free_request_honors_ec_delayed_free() -> None:
    ec_connector = MagicMock()
    ec_connector.request_finished.return_value = (True, EC_PARAMS)
    sched, _ = _make_free_request_sched(ec_connector=ec_connector)
    request = _FakeFinishedRequest("req-ec-delay")

    kv_params, ec_params = sched._free_request(request)

    assert ec_params == EC_PARAMS
    sched._free_blocks.assert_not_called()


def test_free_request_without_ec_connector_returns_none() -> None:
    sched, _ = _make_free_request_sched(ec_connector=None)
    request = _FakeFinishedRequest("req-no-ec")

    kv_params, ec_params = sched._free_request(request)

    assert ec_params is None
    sched._free_blocks.assert_called_once_with(request)


def _make_finishing_request() -> Request:
    request = Request(
        request_id="req-ec-finish",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        arrival_time=100.0,
        block_hasher=None,
    )
    request.status = RequestStatus.RUNNING
    return request


def _make_finish_sched(request: Request) -> MagicMock:
    """MagicMock scheduler driving the real ``update_from_output`` through a
    finishing request; ``_free_request`` reports EC transfer params."""
    sched = MagicMock()
    sched.requests = {request.request_id: request}
    sched.perf_metrics = None
    sched.defer_block_free = False
    sched.structured_output_manager.should_advance.return_value = False
    sched._process_kv_transfer_trigger.return_value = False
    sched._maybe_decode_pooling_output.return_value = None
    sched._handle_stopped_request.return_value = True
    sched._free_request.return_value = (None, EC_PARAMS)
    sched.kv_cache_manager.estimate_cached_tokens.return_value = 0
    sched.chunk_transfer_adapter = None
    sched.finished_req_ids_dict = None
    sched._new_prompt_len_snapshot = {}
    return sched


def test_ar_update_from_output_carries_ec_params_to_engine_output() -> None:
    request = _make_finishing_request()
    sched = _make_finish_sched(request)
    sched._update_request_with_output.return_value = ([42], True)

    engine_core_outputs = OmniARScheduler.update_from_output(sched, *_finish_frame_parts(request))

    (eco,) = engine_core_outputs[request.client_index].outputs
    assert eco.ec_transfer_params == EC_PARAMS
    sched._free_request.assert_called_once_with(request)


def _finish_frame_parts(request: Request):
    scheduler_output = MagicMock(spec=SchedulerOutput)
    scheduler_output.num_scheduled_tokens = {request.request_id: 1}
    scheduler_output.total_num_scheduled_tokens = 1
    scheduler_output.scheduled_spec_decode_tokens = {}
    scheduler_output.num_invalid_spec_tokens = 0

    model_runner_output = MagicMock(spec=ModelRunnerOutput)
    model_runner_output.sampled_token_ids = [[42]]
    model_runner_output.logprobs = None
    model_runner_output.prompt_logprobs_dict = {}
    model_runner_output.pooler_output = None
    model_runner_output.num_nans_in_logits = None
    model_runner_output.kv_connector_output = None
    model_runner_output.cudagraph_stats = None
    model_runner_output.req_id_to_index = {request.request_id: 0}
    model_runner_output.routed_experts = None
    return scheduler_output, model_runner_output


def test_generation_update_from_output_carries_ec_params_to_engine_output() -> None:
    request = _make_finishing_request()
    # One-shot generation finish: the whole prompt is computed.
    request.num_computed_tokens = len(request.prompt_token_ids)
    sched = _make_finish_sched(request)

    engine_core_outputs = OmniGenerationScheduler.update_from_output(sched, *_finish_frame_parts(request))

    (eco,) = engine_core_outputs[request.client_index].outputs
    assert eco.ec_transfer_params == EC_PARAMS
    sched._free_request.assert_called_once_with(request)
