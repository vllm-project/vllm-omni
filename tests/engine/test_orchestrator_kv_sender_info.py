from types import SimpleNamespace

import pytest
from vllm import SamplingParams

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
    client._kv_sender_host = client._resolve_contact_host()

    assert client.get_kv_sender_info() == {
        "host": "10.20.30.40",
        "zmq_port": 50151,
    }


def test_stage_engine_core_client_falls_back_to_detected_ip_for_loopback(monkeypatch):
    client = object.__new__(StageEngineCoreClient)
    client.stage_id = 1
    client.client_addresses = {"input_address": "tcp://127.0.0.1:1234"}
    monkeypatch.setattr(client, "_detect_local_ip", lambda: "192.168.0.12")
    client._kv_sender_host = client._resolve_contact_host()

    assert client.get_kv_sender_info() == {
        "host": "192.168.0.12",
        "zmq_port": 50152,
    }


@pytest.mark.asyncio
async def test_forward_to_diffusion_attaches_kv_sender_info():
    orchestrator = object.__new__(Orchestrator)
    sender_stage = _DummySenderStage({"host": "10.0.0.2", "zmq_port": 50151})
    diffusion_stage = _DummyDiffusionStage(engine_input_source=[0])

    orchestrator.num_stages = 2
    orchestrator.stage_clients = [sender_stage, diffusion_stage]
    orchestrator._companion_map = {}
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
    await Orchestrator._forward_to_next_stage(orchestrator, "req-1", 0, output, req_state)

    assert sender_stage.engine_outputs == [output]
    assert diffusion_stage.calls[0]["request_id"] == "req-1"
    assert diffusion_stage.calls[0]["kv_sender_info"] == {
        0: {"host": "10.0.0.2", "zmq_port": 50151},
    }
    assert req_state.stage_submit_ts[1] > 0


@pytest.mark.asyncio
async def test_prewarm_diffusion_attaches_kv_sender_info():
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
    await Orchestrator._prewarm_async_chunk_stages(orchestrator, "req-2", stage0_request, req_state)

    assert diffusion_stage.calls[0]["request_id"] == "req-2"
    assert diffusion_stage.calls[0]["kv_sender_info"] == {
        0: {"host": "10.0.0.3", "zmq_port": 50151},
    }
    assert req_state.stage_submit_ts[1] > 0
