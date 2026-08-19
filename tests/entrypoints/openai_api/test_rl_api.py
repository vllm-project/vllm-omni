# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_omni.entrypoints.openai.rl_api import router
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _acks():
    return [{"stage_id": 1, "success": True, "result": {"rank": 0}}]


@pytest.fixture
def engine(mocker):
    value = mocker.MagicMock()
    value.pause_generation = mocker.AsyncMock()
    value.resume_generation = mocker.AsyncMock()
    value.is_paused = mocker.AsyncMock(return_value=True)
    value.get_pause_status = mocker.AsyncMock(return_value={"is_paused": True, "barrier_complete": True})
    value.abort = mocker.AsyncMock()
    value.abort_all = mocker.AsyncMock(return_value=3)
    value.init_weight_transfer_engine = mocker.AsyncMock(return_value=_acks())
    value.start_weight_update = mocker.AsyncMock(return_value=_acks())
    value.start_draft_weight_update = mocker.AsyncMock(return_value=_acks())
    value.update_weights = mocker.AsyncMock(return_value=_acks())
    value.finish_weight_update = mocker.AsyncMock(return_value=_acks())
    value.get_weights_checksum = mocker.AsyncMock(return_value=_acks())
    value.add_lora_with_acks = mocker.AsyncMock(return_value=_acks())
    value.remove_lora_with_acks = mocker.AsyncMock(return_value=_acks())
    value.reset_prefix_cache = mocker.AsyncMock(return_value=True)
    value.reset_mm_cache = mocker.AsyncMock()
    value.reset_encoder_cache = mocker.AsyncMock()
    value.sleep = mocker.AsyncMock()
    value.wake_up = mocker.AsyncMock()
    value.get_sleep_status = mocker.AsyncMock(
        return_value={"is_sleeping": False, "stages": [{"stage_id": 1, "is_sleeping": False}]}
    )
    value.get_stage_topology.return_value = {
        "world_size": 6,
        "stages": [
            {"stage_id": 0, "world_size": 2},
            {"stage_id": 1, "world_size": 4},
        ],
    }
    value.stage_configs = [{"stage_type": "llm"}, {"stage_type": "diffusion"}]
    value.default_sampling_params_list = [
        SimpleNamespace(),
        OmniDiffusionSamplingParams(),
    ]
    return value


@pytest.fixture
def client(engine):
    app = FastAPI()
    app.include_router(router)
    app.state.engine_client = engine
    app.state.vllm_config = None
    return TestClient(app)


def test_lifecycle_and_abort_routes_are_vllm_compatible(client, engine):
    assert client.post("/pause?mode=wait&clear_cache=false").json() == {"status": "paused"}
    engine.pause_generation.assert_awaited_once_with(
        mode="wait",
        wait_for_inflight_requests=False,
        clear_cache=False,
    )
    assert client.get("/is_paused").json() == {
        "is_paused": True,
        "barrier_complete": True,
    }
    assert client.post("/resume").json() == {"status": "resumed"}

    response = client.post("/abort_requests", json={})
    assert response.json() == {"status": "aborted", "aborted": 3}
    engine.abort_all.assert_awaited_once()

    response = client.post("/abort_requests", json={"request_ids": ["a", "b"]})
    assert response.json() == {"status": "aborted", "aborted": 2}
    engine.abort.assert_awaited_once_with(["a", "b"])


def test_weight_update_routes_return_per_stage_acknowledgements(client, engine):
    init = client.post(
        "/init_weight_transfer_engine",
        json={"init_info": {"backend": "safetensors", "stage_ids": [1]}},
    )
    assert init.status_code == 200
    assert init.json()["acks"] == _acks()

    assert client.post("/start_weight_update").status_code == 200
    assert client.post("/update_weights", json={"update_info": {"path": "/tmp/policy.safetensors"}}).status_code == 200
    assert client.post("/finish_weight_update").status_code == 200
    assert client.post("/start_draft_weight_update").status_code == 200

    checksum = client.post(
        "/get_weights_checksum",
        json={"stage_ids": [1], "component": "transformer"},
    )
    assert checksum.status_code == 200
    assert checksum.json()["acks"][0]["stage_id"] == 1
    engine.get_weights_checksum.assert_awaited_once_with(
        stage_ids=[1],
        component="transformer",
    )


def test_topology_cache_and_sleep_routes_are_stage_aware(client, engine):
    assert client.get("/get_world_size?stage_id=1").json() == {
        "world_size": 4,
        "stages": [{"stage_id": 1, "world_size": 4}],
    }
    engine.get_stage_topology.assert_called_once_with(include_dp=True)

    assert client.post("/reset_prefix_cache?stage_ids=1&reset_external=true").status_code == 200
    engine.reset_prefix_cache.assert_awaited_once_with(
        reset_running_requests=False,
        reset_connector=True,
        stage_ids=[1],
    )
    assert client.post("/reset_mm_cache?stage_ids=1").status_code == 200
    assert client.post("/reset_encoder_cache?stage_ids=1").status_code == 200

    assert client.post("/sleep?level=1&stage_ids=1").status_code == 200
    assert engine.pause_generation.await_args.kwargs == {"mode": "abort", "clear_cache": True}
    engine.sleep.assert_awaited_once_with(stage_ids=[1], level=1, mode="abort")
    assert client.post("/wake_up?stage_ids=1&tags=weights").status_code == 200
    engine.wake_up.assert_awaited_once_with(stage_ids=[1], tags=["weights"])
    engine.resume_generation.assert_awaited_once()
    assert client.get("/is_sleeping").json()["stages"] == [{"stage_id": 1, "is_sleeping": False}]


def test_rollout_generate_returns_handles_without_tensor_payload(client, engine):
    handle = {
        "__tensor_shm__": True,
        "preserve_for_client": True,
        "kind": "shm",
        "name": "psm_policy",
        "shape": [1, 2],
        "numpy_dtype": "float32",
        "nbytes": 8,
        "torch_dtype": "torch.float32",
        "ownership": "client",
        "release": "unlink",
    }

    async def generate(**kwargs):
        params = kwargs["sampling_params_list"][1]
        assert kwargs["request_id"] == "rollout-1"
        assert params.return_trajectory_latents is True
        assert params.return_trajectory_handles is True
        yield OmniRequestOutput(
            request_id="rollout-1",
            stage_id=1,
            trajectory_latents=handle,
            trajectory_timesteps=handle,
            trajectory_log_probs=handle,
        )

    engine.generate = generate
    response = client.post(
        "/rollout/generate",
        json={"prompt": "a robot", "request_id": "rollout-1", "stage_id": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["transport"] == "shm"
    assert payload["handles"]["latents"]["name"] == "psm_policy"
    assert "__tensor_shm__" not in payload["handles"]["latents"]


def test_rollout_rejects_non_diffusion_stage(client):
    response = client.post(
        "/rollout/generate",
        json={"prompt": "a robot", "stage_id": 0},
    )
    assert response.status_code == 400


def test_lora_routes_require_stage_acknowledgements(client, engine):
    loaded = client.post(
        "/v1/load_lora_adapter",
        json={"lora_name": "policy", "lora_path": "/tmp/policy", "stage_ids": [1]},
    )
    assert loaded.status_code == 200
    lora_request = engine.add_lora_with_acks.await_args.args[0]
    assert lora_request.lora_name == "policy"
    assert engine.add_lora_with_acks.await_args.kwargs["stage_ids"] == [1]

    unloaded = client.post(
        "/v1/unload_lora_adapter",
        json={"lora_name": "policy", "lora_int_id": lora_request.lora_int_id, "stage_ids": [1]},
    )
    assert unloaded.status_code == 200
    engine.remove_lora_with_acks.assert_awaited_once_with(
        lora_request.lora_int_id,
        stage_ids=[1],
    )


def test_server_info_advertises_stage_aware_rl(client):
    response = client.get("/server_info")

    assert response.status_code == 200
    assert response.json()["omni_rl"]["stage_aware"] is True
