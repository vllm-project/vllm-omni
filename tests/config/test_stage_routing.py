# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest

from vllm_omni.config.stage_config import PipelineConfig, StagePipelineConfig
from vllm_omni.config.stage_routing import StageRouting
from vllm_omni.distributed.omni_connectors.utils.config import (
    ConnectorSpec,
    OmniTransferConfig,
    get_stage_connector_peer,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("num_stages", [1, 2, 5])
def test_default_route_preserves_sequential_prefixes(num_stages):
    route = StageRouting.from_transitions(num_stages)
    assert route.stage_order == tuple(range(num_stages))
    for final in range(num_stages):
        assert route.path_to(final) == tuple(range(final + 1))
        assert route.next_stage(final, final) is None


@pytest.mark.parametrize(
    ("transitions", "order"),
    [(((0, 2),), (0, 2)), (((2, 1), (0, 2)), (0, 2, 1)), ((), (0,))],
)
def test_explicit_route_has_no_implicit_edges(transitions, order):
    route = StageRouting.from_transitions(3, transitions)
    assert route.stage_order == order
    assert route.next_stage(order[-1], order[-1]) is None


@pytest.mark.parametrize(
    ("transitions", "error"),
    [
        (((0, 3),), "invalid stage"),
        (((-1, 1),), "invalid stage"),
        (((0, True),), "invalid stage"),
        (((0, 0),), "Self transition"),
        (((0, 1), (1, 0)), "cycle"),
        (((0, 1), (0, 2)), "branching"),
        (((0, 1), (0, 1)), "duplicate"),
        (((0, 2), (1, 2)), "multiple incoming"),
        (((1, 2),), "unreachable"),
    ],
)
def test_invalid_topology_is_rejected(transitions, error):
    with pytest.raises(ValueError, match=error):
        StageRouting.from_transitions(3, transitions)


def test_unreachable_request_endpoint_is_rejected():
    route = StageRouting.from_transitions(3, ((0, 2),))
    with pytest.raises(ValueError, match="not reachable"):
        route.path_to(1)
    assert route.next_stage(1, 2) is None


@pytest.mark.parametrize("transitions", [[(0, 2)], ([0, 2],), ((0,),), ((0, 1, 2),)])
def test_mutable_or_malformed_transition_configuration_is_rejected(transitions):
    with pytest.raises(ValueError, match="immutable tuple"):
        StageRouting.from_transitions(3, transitions)


def test_pipeline_accepts_skipped_stage_with_satisfied_dependencies():
    pipeline = PipelineConfig(
        model_type="test",
        stages=(
            StagePipelineConfig(0, "entry"),
            StagePipelineConfig(1, "inactive", input_sources=(0,)),
            StagePipelineConfig(2, "output", input_sources=(0,), final_output=True),
        ),
        stage_transitions=((0, 2),),
    )
    assert pipeline.get_validation_errors() == []


@pytest.mark.parametrize("sources", [(1,), (2,)])
def test_pipeline_rejects_unsatisfied_route_dependency(sources):
    with pytest.raises(ValueError, match="must precede"):
        PipelineConfig(
            model_type="test",
            stages=(
                StagePipelineConfig(0, "entry"),
                StagePipelineConfig(1, "inactive"),
                StagePipelineConfig(2, "output", input_sources=sources, final_output=True),
            ),
            stage_transitions=((0, 2),),
        )


def test_pipeline_rejects_route_without_terminal_output():
    with pytest.raises(ValueError, match="last stage"):
        PipelineConfig(
            model_type="test",
            stages=(StagePipelineConfig(0, "entry"), StagePipelineConfig(1, "output", final_output=True)),
            stage_transitions=(),
        )


@pytest.mark.parametrize("stage_id", [3, 1.0, True])
def test_pipeline_rejects_ids_that_cannot_index_stage_arrays(stage_id):
    with pytest.raises(ValueError, match="contiguous IDs"):
        PipelineConfig(
            model_type="test",
            stages=(StagePipelineConfig(0, "entry"), StagePipelineConfig(stage_id, "output", final_output=True)),
            stage_transitions=((0, 1),),
        )


@pytest.mark.parametrize(("stage_id", "source", "target"), [(0, None, 2), (2, 0, 1), (1, 2, None)])
def test_connector_projection_preserves_both_route_directions(stage_id, source, target):
    from vllm_omni.engine.stage_init_utils import get_stage_connector_spec

    route = StageRouting.from_transitions(3, ((0, 2), (2, 1)))
    spec = get_stage_connector_spec(None, stage_id, True, route)
    config = SimpleNamespace(stage_connector_config=spec)
    assert get_stage_connector_peer(config, stage_id, "from_stage") == source
    assert get_stage_connector_peer(config, stage_id, "to_stage") == target


def test_legacy_connector_endpoints_keep_numeric_defaults():
    config = SimpleNamespace(stage_connector_config={})
    assert get_stage_connector_peer(config, 1, "from_stage") == 0
    assert get_stage_connector_peer(config, 1, "to_stage") == 2


@pytest.mark.parametrize("value", ["", " ", 1.5])
def test_legacy_unset_connector_target_keeps_sequential_default(value):
    config = SimpleNamespace(stage_connector_config={"to_stage": value})
    assert get_stage_connector_peer(config, 1, "to_stage") == 2


def test_connector_projection_honors_default_without_mutating_deployment():
    from vllm_omni.engine.stage_init_utils import get_stage_connector_spec

    backend = ConnectorSpec("NixlConnector", {"zmq_port": 50051})
    config = OmniTransferConfig({("0", "1"): ConnectorSpec("SharedMemoryConnector")}, backend)
    original = dict(config.connectors)
    route = StageRouting.from_transitions(3, ((0, 2), (2, 1)))
    spec = get_stage_connector_spec(config, 2, True, route)
    assert spec["name"] == "NixlConnector"
    assert (spec["from_stage"], spec["to_stage"]) == (0, 1)
    assert spec["extra"]["outgoing"]["from_stage"] == 2
    assert spec["extra"]["outgoing"]["to_stage"] == 1
    assert config.connectors == original and config.default_connector is backend
    assert backend.extra == {"zmq_port": 50051}


def test_connector_projection_preserves_bridge_fed_stage():
    from vllm_omni.engine.stage_init_utils import get_stage_connector_spec

    config = OmniTransferConfig({("2", "1"): ConnectorSpec("SharedMemoryConnector")})
    route = StageRouting.from_transitions(3, ((0, 2), (2, 1)))
    spec = get_stage_connector_spec(config, 2, True, route)
    assert spec["extra"]["role"] == "sender"
    assert (spec["from_stage"], spec["to_stage"]) == (0, 1)


@pytest.mark.parametrize(
    "backends", [("SharedMemoryConnector", "NixlConnector"), ("NixlConnector", "SharedMemoryConnector")]
)
def test_middle_stage_rejects_incompatible_connector_backends(backends):
    from vllm_omni.engine.stage_init_utils import get_stage_connector_spec

    config = OmniTransferConfig({("0", "2"): ConnectorSpec(backends[0]), ("2", "1"): ConnectorSpec(backends[1])})
    route = StageRouting.from_transitions(3, ((0, 2), (2, 1)))
    with pytest.raises(ValueError, match="same connector backend"):
        get_stage_connector_spec(config, 2, True, route)


def test_inactive_stage_has_no_payload_peers():
    from vllm_omni.engine.stage_init_utils import get_stage_connector_spec

    spec = get_stage_connector_spec(None, 1, True, StageRouting.from_transitions(3, ((0, 2),)))
    assert spec == {"from_stage": None, "to_stage": None, "extra": {"role": "sender"}}
