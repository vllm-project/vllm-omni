# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import pytest

from vllm_omni.distributed.omni_connectors.factory import (
    OmniConnectorFactory,
    StageConnectorSet,
)
from vllm_omni.distributed.omni_connectors.utils.config import (
    ConnectorSpec,
    OmniTransferConfig,
    StageConnectorPlan,
    StageConnectorSpec,
)
from vllm_omni.distributed.omni_connectors.utils.initialization import (
    resolve_stage_connector_plan,
)
from vllm_omni.distributed.omni_connectors.utils.kv_utils import kv_zmq_port

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _RecordingConnector:
    spec: ConnectorSpec
    close_count: int = 0

    def close(self) -> None:
        self.close_count += 1


@pytest.fixture
def created_connectors(monkeypatch):
    created: list[_RecordingConnector] = []

    def create(_cls, spec: ConnectorSpec):
        connector = _RecordingConnector(spec)
        created.append(connector)
        return connector

    monkeypatch.setattr(OmniConnectorFactory, "create_connector", classmethod(create))
    return created


def _edge(source: int, target: int, name: str, **extra) -> StageConnectorSpec:
    return StageConnectorSpec(source, target, ConnectorSpec(name, extra))


def test_plan_resolves_only_edges_attached_to_the_stage():
    config = OmniTransferConfig(
        connectors={
            ("0", "1"): ConnectorSpec("MooncakeTransferEngineConnector", {"zmq_port": 50051}),
            ("1", "2"): ConnectorSpec("SharedMemoryConnector"),
        }
    )

    first = resolve_stage_connector_plan(config, 0)
    middle = resolve_stage_connector_plan(config, 1)
    last = resolve_stage_connector_plan(config, 2)

    assert first.inbound is None
    assert first.outbound is not None
    assert middle.inbound is not None and middle.outbound is not None
    assert last.inbound is not None
    assert last.outbound is None


def test_absent_config_preserves_legacy_shm_but_explicit_empty_config_does_not():
    legacy = resolve_stage_connector_plan(None, 0)
    explicit = resolve_stage_connector_plan(OmniTransferConfig(), 0)

    assert legacy.inbound is not None and legacy.outbound is not None
    assert legacy.inbound.spec.name == legacy.outbound.spec.name == "SharedMemoryConnector"
    assert explicit == StageConnectorPlan()


def test_legacy_shm_plan_shares_one_dual_instance(created_connectors):
    connectors = OmniConnectorFactory.create_stage_connectors(
        resolve_stage_connector_plan(None, 0),
        stage_id=0,
    )

    assert connectors.receive is connectors.send
    assert len(created_connectors) == 1
    assert created_connectors[0].spec.extra["role"] == "dual"


@pytest.mark.parametrize(
    "connectors",
    [
        {
            ("0", "1"): ConnectorSpec("SharedMemoryConnector"),
            ("2", "1"): ConnectorSpec("SharedMemoryConnector"),
        },
        {
            ("1", "2"): ConnectorSpec("SharedMemoryConnector"),
            ("1", "3"): ConnectorSpec("SharedMemoryConnector"),
        },
    ],
)
def test_fan_in_and_fan_out_are_rejected(connectors):
    with pytest.raises(ValueError, match="Fan-"):
        resolve_stage_connector_plan(OmniTransferConfig(connectors), 1)


def test_compatible_edges_share_one_dual_instance(created_connectors):
    plan = StageConnectorPlan(
        inbound=_edge(
            0,
            1,
            "MooncakeTransferEngineConnector",
            host="auto",
            zmq_port=50051,
            sender_zmq_port=50051,
        ),
        outbound=_edge(
            1,
            2,
            "MooncakeTransferEngineConnector",
            host="auto",
            zmq_port=50052,
        ),
    )

    connectors = OmniConnectorFactory.create_stage_connectors(
        plan,
        stage_id=1,
        local_rank=2,
        replica_id=1,
    )

    assert connectors.receive is connectors.send
    assert len(created_connectors) == 1
    extra = created_connectors[0].spec.extra
    assert extra["role"] == "dual"
    assert extra["zmq_port"] == kv_zmq_port(50052, 1, local_rank=2, replica_id=1)
    # The upstream endpoint must not inherit this worker's rank offset.
    assert extra["sender_zmq_port"] == 50051


@pytest.mark.parametrize(
    "name",
    ["MooncakeStoreConnector", "YuanrongConnector", "MooncakeConnector"],
)
def test_compatible_store_edges_share_one_instance(created_connectors, name):
    plan = StageConnectorPlan(
        inbound=_edge(0, 1, name, host="store-host"),
        outbound=_edge(1, 2, name, host="store-host"),
    )

    connectors = OmniConnectorFactory.create_stage_connectors(plan, stage_id=1)

    assert connectors.receive is connectors.send
    assert len(created_connectors) == 1


def test_dual_merge_preserves_directional_rank_mappings(created_connectors):
    recv_mapping = {"from_tp": 4, "to_tp": 2}
    send_mapping = {"from_tp": 2, "to_tp": 1}
    plan = StageConnectorPlan(
        inbound=_edge(
            0,
            1,
            "MooncakeTransferEngineConnector",
            rank_mapping=recv_mapping,
        ),
        outbound=_edge(
            1,
            2,
            "MooncakeTransferEngineConnector",
            rank_mapping=send_mapping,
        ),
    )

    OmniConnectorFactory.create_stage_connectors(plan, stage_id=1)

    extra = created_connectors[0].spec.extra
    assert "rank_mapping" not in extra
    assert extra["recv_rank_mapping"] == recv_mapping
    assert extra["send_rank_mapping"] == send_mapping


def test_dual_merge_preserves_shared_rank_mapping(created_connectors):
    mapping = {"from_tp": 2, "to_tp": 2}
    plan = StageConnectorPlan(
        inbound=_edge(0, 1, "SharedMemoryConnector", rank_mapping=mapping),
        outbound=_edge(1, 2, "SharedMemoryConnector", rank_mapping=mapping),
    )

    OmniConnectorFactory.create_stage_connectors(plan, stage_id=1)

    extra = created_connectors[0].spec.extra
    assert extra["rank_mapping"] == mapping
    assert "recv_rank_mapping" not in extra
    assert "send_rank_mapping" not in extra


@pytest.mark.parametrize(
    ("inbound", "outbound"),
    [
        (
            _edge(0, 1, "MooncakeTransferEngineConnector", host="10.0.0.1"),
            _edge(1, 2, "MooncakeTransferEngineConnector", host="10.0.0.2"),
        ),
        (
            _edge(0, 1, "MooncakeTransferEngineConnector"),
            _edge(1, 2, "SharedMemoryConnector"),
        ),
    ],
)
def test_incompatible_edges_keep_two_directional_instances(
    created_connectors,
    inbound,
    outbound,
):
    connectors = OmniConnectorFactory.create_stage_connectors(
        StageConnectorPlan(inbound, outbound),
        stage_id=1,
    )

    assert connectors.receive is not connectors.send
    assert len(created_connectors) == 2
    assert created_connectors[0].spec.extra["role"] == "receiver"
    assert created_connectors[1].spec.extra["role"] == "sender"


def test_direction_is_derived_from_the_edge(created_connectors):
    plan = StageConnectorPlan(
        inbound=_edge(0, 1, "SharedMemoryConnector", role="sender"),
        outbound=_edge(1, 2, "MooncakeStoreConnector", role="receiver"),
    )

    OmniConnectorFactory.create_stage_connectors(plan, stage_id=1)

    assert created_connectors[0].spec.extra["role"] == "receiver"
    assert created_connectors[1].spec.extra["role"] == "sender"


@pytest.mark.parametrize(
    ("plan", "has_receive", "has_send"),
    [
        (StageConnectorPlan(inbound=_edge(0, 1, "SharedMemoryConnector")), True, False),
        (StageConnectorPlan(outbound=_edge(1, 2, "SharedMemoryConnector")), False, True),
    ],
)
def test_one_way_plan_creates_only_its_direction(
    created_connectors,
    plan,
    has_receive,
    has_send,
):
    connectors = OmniConnectorFactory.create_stage_connectors(plan, stage_id=1)

    assert (connectors.receive is not None) is has_receive
    assert (connectors.send is not None) is has_send
    assert len(created_connectors) == 1


def test_close_deduplicates_a_shared_dual_instance():
    connector = _RecordingConnector(ConnectorSpec("test"))
    StageConnectorSet(receive=connector, send=connector).close()
    assert connector.close_count == 1
