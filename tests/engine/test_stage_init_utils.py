import pytest

from vllm_omni.distributed.omni_connectors.utils.initialization import load_omni_transfer_config
from vllm_omni.engine.stage_init_utils import get_stage_connector_spec

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_get_stage_connector_spec_prefers_outgoing_edge_for_async_chunk():
    transfer_config = load_omni_transfer_config(
        config_dict={
            "async_chunk": True,
            "stage_args": [
                {
                    "stage_id": 0,
                    "output_connectors": {
                        "to_stage_1": "connector_stage0_to_1",
                    },
                },
                {
                    "stage_id": 1,
                    "input_connectors": {
                        "from_stage_0": "connector_stage0_to_1",
                    },
                    "output_connectors": {
                        "to_stage_2": "connector_stage1_to_2",
                    },
                },
                {
                    "stage_id": 2,
                    "input_connectors": {
                        "from_stage_1": "connector_stage1_to_2",
                    },
                },
            ],
            "runtime": {
                "connectors": {
                    "connector_stage0_to_1": {
                        "name": "SharedMemoryConnector",
                        "extra": {
                            "tag": "incoming",
                        },
                    },
                    "connector_stage1_to_2": {
                        "name": "SharedMemoryConnector",
                        "extra": {
                            "tag": "outgoing",
                            "async_talker_greedy": True,
                        },
                    },
                },
                "edges": [
                    {"from": 0, "to": 1},
                    {"from": 1, "to": 2},
                ],
            },
        }
    )

    stage1_spec = get_stage_connector_spec(
        omni_transfer_config=transfer_config,
        stage_id=1,
        async_chunk=True,
    )
    stage2_spec = get_stage_connector_spec(
        omni_transfer_config=transfer_config,
        stage_id=2,
        async_chunk=True,
    )

    assert stage1_spec == {
        "name": "SharedMemoryConnector",
        "extra": {
            "tag": "outgoing",
            "async_talker_greedy": True,
        },
    }
    assert stage2_spec == {
        "name": "SharedMemoryConnector",
        "extra": {
            "tag": "outgoing",
            "async_talker_greedy": True,
        },
    }
