# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for StageEngineCoreClient.check_health()."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm import SamplingParams
from vllm.v1.engine.core_client import AsyncMPClient, DPLBAsyncMPClient
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.engine.stage_engine_core_client import (
    DPLBStageEngineCoreClient,
    StageEngineCoreClient,
    StageEngineCoreClientBase,
)
from vllm_omni.engine.stage_init_utils import StageMetadata

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_client(*, engine_dead=False):
    client = object.__new__(StageEngineCoreClient)
    client.stage_id = 0
    client.resources = SimpleNamespace(engine_dead=engine_dead)
    return client


def test_check_health_passes_when_alive():
    client = _make_client(engine_dead=False)
    client.check_health()  # no exception


def test_check_health_raises_when_resources_engine_dead():
    client = _make_client(engine_dead=True)
    with pytest.raises(EngineDeadError, match="engine core is dead"):
        client.check_health()


@pytest.mark.parametrize("dp_size, external_lb", [(1, False), (2, False), (2, True)])
@pytest.mark.parametrize("renderer", [None, object()])
def test_stage_client_forwards_renderer_to_the_actual_upstream_constructor(dp_size, external_lb, renderer):
    config = SimpleNamespace(
        model_config=None,
        parallel_config=SimpleNamespace(data_parallel_size=dp_size, data_parallel_external_lb=external_lb),
    )
    metadata = StageMetadata(
        stage_id=0,
        stage_type="llm",
        engine_output_type="text",
        is_comprehension=True,
        requires_multimodal_data=False,
        engine_input_source=[],
        final_output=True,
        final_output_type="text",
        default_sampling_params=SamplingParams(),
        custom_process_input_func=None,
        model_stage=None,
        runtime_cfg=None,
    )
    internal_lb = dp_size > 1 and not external_lb
    parent = DPLBAsyncMPClient if internal_lb else AsyncMPClient
    # Keep Omni construction/factory routing real; intercept only upstream's
    # subprocess/socket boundary, using its current signature via autospec.
    with patch.object(parent, "__init__", autospec=True, return_value=None) as init:
        addresses = {"input_address": "tcp://192.0.2.1:1"}
        client = StageEngineCoreClientBase.make_async_mp_client(
            config, object, metadata=metadata, client_addresses=addresses, renderer=renderer
        )
        assert isinstance(client, DPLBStageEngineCoreClient if internal_lb else StageEngineCoreClient)
        init.assert_called_once_with(
            client,
            config,
            object,
            log_stats=False,
            client_addresses=addresses,
            client_count=1,
            client_index=0,
            renderer=renderer,
        )
