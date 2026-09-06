"""Connector progress must not depend on scheduling a model forward."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from vllm.v1.outputs import ECConnectorOutput, KVConnectorOutput, ModelRunnerOutput

from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("runner_cls", [OmniGPUModelRunner, OmniGenerationModelRunner])
def test_empty_scheduler_step_pumps_kv_and_ec_connectors(runner_cls):
    runner = object.__new__(runner_cls)
    # Admission is covered separately; this test exercises the idle-step return.
    for name in (
        "_prepare_native_data_plane",
        "finish_requests",
        "free_states",
        "add_requests",
        "update_requests",
        "_sync_native_data_plane_payloads",
        "_handle_async_chunk_updates",
        "_apply_block_table_staged_writes_if_available",
    ):
        setattr(runner, name, Mock())
    runner.block_tables = SimpleNamespace(apply_staged_writes=Mock())
    runner._omni_data_plane = None
    kv_output = KVConnectorOutput(finished_sending={"kv-request"})
    ec_output = ECConnectorOutput(finished_sending={"encoded-input"})
    idle_output = ModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[], kv_connector_output=kv_output)
    runner.kv_connector = SimpleNamespace(no_forward=Mock(return_value=idle_output))
    runner.ec_connector = SimpleNamespace(no_forward=Mock(return_value=SimpleNamespace(ec_connector_output=ec_output)))
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=0)

    output = runner.execute_model(scheduler_output)

    runner.kv_connector.no_forward.assert_called_once_with(scheduler_output)
    runner.ec_connector.no_forward.assert_called_once_with(scheduler_output)
    assert output.kv_connector_output is kv_output
    assert output.ec_connector_output is ec_output
