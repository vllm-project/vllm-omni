import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI, WebSocket
from omegaconf import OmegaConf
from starlette.testclient import TestClient

from vllm_omni.entrypoints.openpi import connection as openpi_connection
from vllm_omni.entrypoints.openpi import serving as openpi_serving

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

TEST_POLICY_SERVER_CONFIG = {
    "image_resolution": (180, 320),
    "n_external_cameras": 2,
    "needs_wrist_camera": True,
    "needs_stereo_camera": False,
    "needs_session_id": True,
    "action_space": "joint_position",
}


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _json_pack(obj):
    return json.dumps(obj, default=_json_default).encode()


def _json_unpack(data):
    return json.loads(data.decode())


def _engine_with_policy_config(policy_config=None):
    if policy_config is None:
        policy_config = TEST_POLICY_SERVER_CONFIG
    od_config = SimpleNamespace(model_config={"policy_server_config": policy_config})
    return SimpleNamespace(get_diffusion_od_config=lambda: od_config)


class RecordingEngine:
    def __init__(self):
        self.od_config = SimpleNamespace(model_config={"policy_server_config": TEST_POLICY_SERVER_CONFIG})
        self.generate_calls = []

    def get_diffusion_od_config(self):
        return self.od_config

    def generate(self, *, prompt, request_id, sampling_params_list):
        async def _generate():
            self.generate_calls.append(
                {
                    "prompt": prompt,
                    "request_id": request_id,
                    "sampling_params_list": sampling_params_list,
                }
            )
            yield SimpleNamespace(multimodal_output={"actions": [0.0]})

        return _generate()


class ConcurrentRecordingEngine(RecordingEngine):
    def __init__(self, *, expected_calls: int):
        super().__init__()
        self.expected_calls = expected_calls
        self.condition = threading.Condition()
        self.saw_overlap = False

    def _wait_for_expected_calls(self):
        with self.condition:
            completed = self.condition.wait_for(
                lambda: len(self.generate_calls) >= self.expected_calls,
                timeout=5.0,
            )
            self.saw_overlap = self.saw_overlap or completed

    def generate(self, *, prompt, request_id, sampling_params_list):
        async def _generate():
            with self.condition:
                self.generate_calls.append(
                    {
                        "prompt": prompt,
                        "request_id": request_id,
                        "sampling_params_list": sampling_params_list,
                    }
                )
                if len(self.generate_calls) >= self.expected_calls:
                    self.saw_overlap = True
                    self.condition.notify_all()

            await asyncio.to_thread(self._wait_for_expected_calls)
            yield SimpleNamespace(multimodal_output={"actions": [0.0]})

        return _generate()


def test_policy_server_config_reads_diffusion_model_config():
    policy_config = {
        "image_resolution": [64, 64],
        "n_external_cameras": 1,
        "custom_model_key": {"nested": True},
    }
    od_config = SimpleNamespace(model_config={"policy_server_config": policy_config})
    engine_client = SimpleNamespace(get_diffusion_od_config=lambda: od_config)

    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=engine_client)

    assert serving.policy_server_config.to_dict() == policy_config


def test_policy_server_config_reads_stage_config_model_config():
    policy_config = {"custom_model_key": "from-stage-config"}
    engine_client = SimpleNamespace(
        get_diffusion_od_config=lambda: None,
        stage_configs=[
            SimpleNamespace(
                stage_type="diffusion",
                engine_args=SimpleNamespace(model_config={"policy_server_config": policy_config}),
            )
        ],
    )

    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=engine_client)

    assert serving.policy_server_config.to_dict() == policy_config


def test_policy_server_config_reads_omegaconf_stage_config():
    engine_client = SimpleNamespace(
        get_diffusion_od_config=lambda: None,
        stage_configs=[
            SimpleNamespace(
                stage_type="diffusion",
                engine_args=SimpleNamespace(
                    model_config=OmegaConf.create({"policy_server_config": {"custom_model_key": "from-omegaconf"}})
                ),
            )
        ],
    )

    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=engine_client)

    assert serving.policy_server_config.to_dict() == {"custom_model_key": "from-omegaconf"}


def test_policy_server_config_is_required():
    od_config = SimpleNamespace(model_config={})
    engine_client = SimpleNamespace(get_diffusion_od_config=lambda: od_config)

    with pytest.raises(ValueError) as exc_info:
        openpi_serving.ServingRealtimeRobotOpenPI(engine_client=engine_client)

    assert "policy_server_config" in str(exc_info.value)


def test_create_policy_server_returns_none_without_policy_config():
    od_config = SimpleNamespace(model_config={})
    engine_client = SimpleNamespace(get_diffusion_od_config=lambda: od_config)

    serving = openpi_serving.ServingRealtimeRobotOpenPI.create_policy_server(
        engine_client=engine_client,
        model_name="generic-model",
    )

    assert serving is None


def test_policy_server_config_allows_explicit_empty_config():
    serving = openpi_serving.ServingRealtimeRobotOpenPI(
        engine_client=_engine_with_policy_config(policy_config={}),
        model_name="nvidia/Cosmos3-Nano-Policy-DROID",
    )

    assert serving.policy_server_config.to_dict() == {}


def test_policy_server_config_reads_engine_model_config():
    policy_config = {"custom_model_key": "custom-value"}
    engine_client = SimpleNamespace(model_config=SimpleNamespace(policy_server_config=policy_config))

    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=engine_client)

    assert serving.policy_server_config.to_dict() == policy_config


def test_build_request_uses_unique_engine_request_id_per_inference():
    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=_engine_with_policy_config())

    request_a = serving._build_request(
        {"prompt": "pick up the object"},
        session_id="session-a",
        reset=True,
    )
    request_b = serving._build_request(
        {"prompt": "pick up the object"},
        session_id="session-a",
        reset=False,
    )

    assert request_a.sampling_params.extra_args["reset"] is True
    assert request_b.sampling_params.extra_args["reset"] is False
    assert request_a.sampling_params.extra_args["session_id"] == "session-a"
    assert request_b.sampling_params.extra_args["session_id"] == "session-a"
    assert request_a.sampling_params.extra_args["robot_obs"]["prompt"] == "pick up the object"
    assert request_b.sampling_params.extra_args["robot_obs"]["prompt"] == "pick up the object"

    assert request_a.request_id == "robot-session-a-0"
    assert request_b.request_id == "robot-session-a-1"
    assert request_a.request_id != request_b.request_id


def test_infer_keeps_session_state_but_uses_unique_engine_request_ids():
    engine = RecordingEngine()
    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=engine)

    async def run_requests():
        await serving.infer({"prompt": "pick up the object"}, session_id="session-a", reset=True)
        await serving.infer({"prompt": "pick up the object"}, session_id="session-a", reset=False)

    asyncio.run(run_requests())

    assert [call["request_id"] for call in engine.generate_calls] == [
        "robot-session-a-0",
        "robot-session-a-1",
    ]
    assert engine.generate_calls[0]["request_id"] != engine.generate_calls[1]["request_id"]

    sampling_params_a = engine.generate_calls[0]["sampling_params_list"][0]
    sampling_params_b = engine.generate_calls[1]["sampling_params_list"][0]
    assert sampling_params_a.extra_args["session_id"] == "session-a"
    assert sampling_params_b.extra_args["session_id"] == "session-a"
    assert sampling_params_a.extra_args["reset"] is True
    assert sampling_params_b.extra_args["reset"] is False


def test_two_websocket_clients_without_session_id_do_not_conflict(monkeypatch):
    monkeypatch.setattr(openpi_connection, "_pack", _json_pack)
    monkeypatch.setattr(openpi_connection, "_unpack", _json_unpack)

    engine = ConcurrentRecordingEngine(expected_calls=2)
    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=engine)
    app = FastAPI()

    @app.websocket("/v1/realtime/robot/openpi")
    async def openpi_endpoint(websocket: WebSocket):
        connection = openpi_connection.RobotRealtimeConnection(websocket, serving)
        await connection.handle_connection()

    def run_client(prompt: str):
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime/robot/openpi") as websocket:
                metadata = _json_unpack(websocket.receive_bytes())
                assert metadata["needs_session_id"] is True

                websocket.send_bytes(_json_pack({"prompt": prompt}))
                actions = _json_unpack(websocket.receive_bytes())
                np.testing.assert_array_equal(
                    np.asarray(actions, dtype=np.float32),
                    np.asarray([0.0], dtype=np.float32),
                )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_client, "first client"),
            executor.submit(run_client, "second client"),
        ]
        for future in futures:
            future.result(timeout=10.0)

    request_ids = [call["request_id"] for call in engine.generate_calls]
    assert len(request_ids) == 2
    assert len(set(request_ids)) == 2
    assert all(request_id.startswith("robot-default-") for request_id in request_ids)
    assert engine.saw_overlap is True

    sampling_params = [call["sampling_params_list"][0] for call in engine.generate_calls]
    assert [params.extra_args["session_id"] for params in sampling_params] == ["default", "default"]
    assert [params.extra_args["reset"] for params in sampling_params] == [True, True]


def test_infer_extracts_actions_from_generic_multimodal_output():
    class FakeEngineClient:
        def get_diffusion_od_config(self):
            return SimpleNamespace(model_config={"policy_server_config": TEST_POLICY_SERVER_CONFIG})

        async def generate(self, **kwargs):
            self.generate_kwargs = kwargs
            yield SimpleNamespace(multimodal_output={"actions": [[1.0, 2.0, 3.0]]})

    engine_client = FakeEngineClient()
    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=engine_client)

    actions = asyncio.run(serving.infer({"prompt": "pick up"}, session_id="session-a", reset=True))

    np.testing.assert_allclose(actions, np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    assert engine_client.generate_kwargs["prompt"] == "pick up"
    assert engine_client.generate_kwargs["request_id"] == "robot-session-a-0"


def test_infer_preserves_dict_actions_from_multimodal_output():
    class FakeEngineClient:
        def get_diffusion_od_config(self):
            return SimpleNamespace(model_config={"policy_server_config": TEST_POLICY_SERVER_CONFIG})

        async def generate(self, **kwargs):
            self.generate_kwargs = kwargs
            yield SimpleNamespace(
                multimodal_output={
                    "actions": {
                        "left_arm": [[1.0, 2.0]],
                        "right_arm": np.array([[3.0, 4.0]], dtype=np.float64),
                    }
                }
            )

    engine_client = FakeEngineClient()
    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=engine_client)

    actions = asyncio.run(serving.infer({"prompt": "pick up"}, session_id="session-a", reset=True))

    assert isinstance(actions, dict)
    assert set(actions) == {"left_arm", "right_arm"}
    np.testing.assert_allclose(actions["left_arm"], np.array([[1.0, 2.0]], dtype=np.float32))
    np.testing.assert_allclose(actions["right_arm"], np.array([[3.0, 4.0]], dtype=np.float32))
    assert actions["left_arm"].dtype == np.float32
    assert actions["right_arm"].dtype == np.float32


def test_extract_actions_does_not_iterate_result_object():
    class IterableResult:
        multimodal_output = {"actions": [[1.0, 2.0, 3.0]]}

        def __iter__(self):
            raise AssertionError("result object should not be iterated")

    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=_engine_with_policy_config())

    actions = serving._extract_actions(IterableResult())

    np.testing.assert_allclose(actions, np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
