# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for StageEngineCoreClient.check_health()."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.distributed.omni_connectors.utils.config import (
    ConnectorSpec,
    StageConnectorPlan,
    StageConnectorSpec,
)
from vllm_omni.distributed.omni_connectors.utils.kv_utils import kv_zmq_port
from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClient

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


def test_payload_sender_info_uses_bound_replica_port():
    client = _make_client()
    client.stage_id = 1
    client.replica_id = 2
    client._stage_connector_plan = StageConnectorPlan(
        outbound=StageConnectorSpec(
            1,
            2,
            ConnectorSpec(
                "MooncakeTransferEngineConnector",
                {"host": "10.0.0.8", "zmq_port": 51000},
            ),
        )
    )

    endpoint = client.get_payload_sender_info()

    assert endpoint is not None
    assert endpoint.host == "10.0.0.8"
    assert endpoint.zmq_port == kv_zmq_port(51000, 1, replica_id=2)
