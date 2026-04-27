import asyncio
from types import SimpleNamespace

import pytest
from vllm import SamplingParams

from vllm_omni.engine.cfg_companion_tracker import CfgCompanionTracker
from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState
from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClient
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummySenderStage:
    def __init__(self, sender_info):
        self._sender_info = sender_info
        self.engine_outputs = None

    def set_engine_outputs(self, outputs):
        self.engine_outputs = outputs

    def get_kv_sender_info(self):
        return self._sender_info


class _DummyDiffusionStage:
    stage_type = "diffusion"
    custom_process_input_func = None

    def __init__(self, engine_input_source=None):
        self.engine_input_source = engine_input_source or [0]
        self.calls = []

    async def add_request_async(self, request_id, prompt, sampling_params, kv_sender_info=None):
        self.calls.append(
            {
                "request_id": request_id,
                "prompt": prompt,
                "sampling_params": sampling_params,
                "kv_sender_info": kv_sender_info,
            }
        )


def test_stage_engine_core_client_builds_kv_sender_info_from_tcp_address():
    client = object.__new__(StageEngineCoreClient)
    client.stage_id = 0
    client.client_addresses = {"input_address": "tcp://10.20.30.40:1234"}
    client._omni_kv_config = None
    client._kv_sender_info = None
    client._kv_sender_initialized = False
    client._kv_sender_host = client._resolve_contact_host()
    client._initialize_kv_sender_endpoint()

    assert client.get_kv_sender_info() == {
        "host": "10.20.30.40",
        "zmq_port": 50151,
    }


def test_stage_engine_core_client_falls_back_to_detected_ip_for_loopback(monkeypatch):
    client = object.__new__(StageEngineCoreClient)
    client.stage_id = 1
    client.client_addresses = {"input_address": "tcp://127.0.0.1:1234"}
    client._omni_kv_config = None
    client._kv_sender_info = None
    client._kv_sender_initialized = False
    monkeypatch.setattr(client, "_detect_local_ip", lambda: "192.168.0.12")
    client._kv_sender_host = client._resolve_contact_host()
    client._initialize_kv_sender_endpoint()

    assert client.get_kv_sender_info() == {
        "host": "192.168.0.12",
        "zmq_port": 50152,
    }


def test_stage_engine_core_client_uses_connector_config_for_sender_port():
    client = object.__new__(StageEngineCoreClient)
    client.stage_id = 3
    client.client_addresses = {"input_address": "tcp://10.20.30.40:1234"}
    client._kv_sender_info = None
    client._kv_sender_initialized = False
    client._omni_kv_config = {
        "omni_from_stage": "3",
        "connector_config": {
            "type": "MooncakeTransferEngineConnector",
            "role": "sender",
            "host": "10.20.30.99",
            "zmq_port": 51000,
        },
    }
    client._kv_sender_host = client._resolve_contact_host()
    client._initialize_kv_sender_endpoint()

    assert client.get_kv_sender_info() == {
        "host": "10.20.30.99",
        "zmq_port": 51103,
    }


def test_stage_engine_core_client_preserves_explicit_loopback_sender_host():
    client = object.__new__(StageEngineCoreClient)
    client.stage_id = 2
    client.client_addresses = {"input_address": "tcp://10.20.30.40:1234"}
    client._kv_sender_info = None
    client._kv_sender_initialized = False
    client._omni_kv_config = {
        "omni_from_stage": "2",
        "connector_config": {
            "type": "MooncakeTransferEngineConnector",
            "role": "sender",
            "host": "127.0.0.1",
            "zmq_port": 51000,
        },
    }
    client._kv_sender_host = client._resolve_contact_host()
    client._initialize_kv_sender_endpoint()

    assert client.get_kv_sender_info() == {
        "host": "127.0.0.1",
        "zmq_port": 51102,
    }


def test_forward_to_diffusion_attaches_kv_sender_info():
    orchestrator = object.__new__(Orchestrator)
    sender_stage = _DummySenderStage({"host": "10.0.0.2", "zmq_port": 50151})
    diffusion_stage = _DummyDiffusionStage(engine_input_source=[0])

    orchestrator.num_stages = 2
    orchestrator.stage_clients = [sender_stage, diffusion_stage]
    orchestrator._cfg_tracker = CfgCompanionTracker()
    orchestrator.stage_vllm_configs = [None, None]
    orchestrator.output_processors = [None, None]

    params = OmniDiffusionSamplingParams()
    req_state = OrchestratorRequestState(
        request_id="req-1",
        prompt={"prompt": "hello"},
        sampling_params_list=[SamplingParams(max_tokens=4), params],
        final_stage_id=1,
    )

    output = SimpleNamespace(request_id="req-1", finished=True)
    asyncio.run(Orchestrator._forward_to_next_stage(orchestrator, "req-1", 0, output, req_state))

    assert sender_stage.engine_outputs == [output]
    assert diffusion_stage.calls[0]["request_id"] == "req-1"
    assert diffusion_stage.calls[0]["kv_sender_info"] == {
        0: {"host": "10.0.0.2", "zmq_port": 50151},
    }
    assert req_state.stage_submit_ts[1] > 0


def test_forward_to_diffusion_uses_engine_input_source_for_kv_sender_info():
    orchestrator = object.__new__(Orchestrator)
    source_stage = _DummySenderStage({"host": "10.0.0.2", "zmq_port": 50151})
    previous_stage = _DummySenderStage({"host": "10.0.0.9", "zmq_port": 59999})
    diffusion_stage = _DummyDiffusionStage(engine_input_source=[0])

    orchestrator.num_stages = 3
    orchestrator.stage_clients = [source_stage, previous_stage, diffusion_stage]
    orchestrator._cfg_tracker = CfgCompanionTracker()
    orchestrator.stage_vllm_configs = [None, None, None]
    orchestrator.output_processors = [None, None, None]

    params = OmniDiffusionSamplingParams()
    req_state = OrchestratorRequestState(
        request_id="req-3",
        prompt={"prompt": "hello"},
        sampling_params_list=[SamplingParams(max_tokens=4), SamplingParams(max_tokens=4), params],
        final_stage_id=2,
    )

    output = SimpleNamespace(request_id="req-3", finished=True)
    asyncio.run(Orchestrator._forward_to_next_stage(orchestrator, "req-3", 1, output, req_state))

    assert previous_stage.engine_outputs == [output]
    assert diffusion_stage.calls[0]["kv_sender_info"] == {
        0: {"host": "10.0.0.2", "zmq_port": 50151},
    }


def test_forward_to_diffusion_returns_terminal_error_for_empty_custom_inputs():
    orchestrator = object.__new__(Orchestrator)
    sender_stage = _DummySenderStage({"host": "10.0.0.2", "zmq_port": 50151})
    diffusion_stage = _DummyDiffusionStage(engine_input_source=[0])
    diffusion_stage.custom_process_input_func = lambda *_args, **_kwargs: []

    class _AsyncQueue:
        def __init__(self):
            self.items = []

        async def put(self, item):
            self.items.append(item)

    orchestrator.num_stages = 2
    orchestrator.stage_clients = [sender_stage, diffusion_stage]
    orchestrator._cfg_tracker = CfgCompanionTracker()
    orchestrator.stage_vllm_configs = [None, None]
    orchestrator.output_processors = [None, None]
    orchestrator.output_async_queue = _AsyncQueue()
    orchestrator.request_states = {}
    orchestrator._pd_kv_params = {}

    params = OmniDiffusionSamplingParams()
    req_state = OrchestratorRequestState(
        request_id="req-empty",
        prompt={"prompt": "hello"},
        sampling_params_list=[SamplingParams(max_tokens=4), params],
        final_stage_id=1,
    )
    orchestrator.request_states["req-empty"] = req_state

    output = SimpleNamespace(request_id="req-empty", finished=True)
    asyncio.run(Orchestrator._forward_to_next_stage(orchestrator, "req-empty", 0, output, req_state))

    assert sender_stage.engine_outputs == [output]
    assert diffusion_stage.calls == []
    assert len(orchestrator.output_async_queue.items) == 1
    terminal_msg = orchestrator.output_async_queue.items[0]
    assert terminal_msg["type"] == "output"
    assert terminal_msg["request_id"] == "req-empty"
    assert terminal_msg["stage_id"] == 1
    assert terminal_msg["finished"] is True
    assert "produced no valid inputs" in terminal_msg["engine_outputs"].error
    assert "req-empty" not in orchestrator.request_states


def test_prewarm_diffusion_attaches_kv_sender_info():
    orchestrator = object.__new__(Orchestrator)
    sender_stage = _DummySenderStage({"host": "10.0.0.3", "zmq_port": 50151})
    diffusion_stage = _DummyDiffusionStage(engine_input_source=[0])

    orchestrator.stage_clients = [sender_stage, diffusion_stage]
    orchestrator.num_stages = 2

    req_state = OrchestratorRequestState(
        request_id="req-2",
        prompt={"prompt": "hello"},
        sampling_params_list=[SamplingParams(max_tokens=4), OmniDiffusionSamplingParams()],
        final_stage_id=1,
    )

    stage0_request = SimpleNamespace(prompt_token_ids=[1, 2, 3])
    asyncio.run(Orchestrator._prewarm_async_chunk_stages(orchestrator, "req-2", stage0_request, req_state))

    assert diffusion_stage.calls[0]["request_id"] == "req-2"
    assert diffusion_stage.calls[0]["kv_sender_info"] == {
        0: {"host": "10.0.0.3", "zmq_port": 50151},
    }
    assert req_state.stage_submit_ts[1] > 0
