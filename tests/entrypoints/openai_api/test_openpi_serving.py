# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

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
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

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


def test_legacy_policy_request_defaults_key_no_longer_controls_serving():
    od_config = SimpleNamespace(
        model_config={
            "policy_server_config": {},
            "policy_request_defaults": "malformed legacy value",
        }
    )

    serving = openpi_serving.ServingRealtimeRobotOpenPI.create_policy_server(
        engine_client=SimpleNamespace(get_diffusion_od_config=lambda: od_config),
        model_name="generic-policy",
    )

    assert serving is not None


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


def test_build_request_clones_stage_defaults_before_protocol_fields():
    default_params = OmniDiffusionSamplingParams(
        extra_args={
            "format_prompt_as_json": True,
            "session_id": "configured-session-must-not-win",
            "reset": False,
            "nested": {"values": []},
        }
    )
    od_config = SimpleNamespace(model_config={"policy_server_config": {}})
    engine_client = SimpleNamespace(
        get_diffusion_od_config=lambda: od_config,
        default_sampling_params_list=[default_params],
    )
    serving = openpi_serving.ServingRealtimeRobotOpenPI(
        engine_client=engine_client,
        model_name="custom-policy",
    )

    request_a = serving._build_request(
        {"prompt": "pick up the object"},
        session_id="wire-session",
        reset=True,
    )
    request_b = serving._build_request(
        {"prompt": "pick up the object"},
        session_id="wire-session",
        reset=False,
    )

    assert request_a.sampling_params.extra_args["format_prompt_as_json"] is True
    assert request_a.sampling_params.extra_args["session_id"] == "wire-session"
    assert request_a.sampling_params.extra_args["reset"] is True
    request_a.sampling_params.extra_args["nested"]["values"].append("request-a")
    assert request_b.sampling_params.extra_args["nested"] == {"values": []}
    assert default_params.extra_args["nested"] == {"values": []}


def test_build_request_forwards_seed_to_sampling_params():
    """``seed`` in the inference message is the engine-level seed; omitted, the request auto-seeds."""
    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=_engine_with_policy_config())

    seeded = serving._build_request({"prompt": "pick up the object", "seed": 42}, session_id="s", reset=True)
    unseeded = serving._build_request({"prompt": "pick up the object"}, session_id="s", reset=False)

    assert seeded.sampling_params.seed == 42
    assert "seed" not in seeded.sampling_params.extra_args["robot_obs"]
    assert isinstance(unseeded.sampling_params.seed, int)


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


def test_infer_yields_event_loop_while_engine_is_running():
    class DelayedEngine(RecordingEngine):
        def generate(self, *, prompt, request_id, sampling_params_list):
            async def _generate():
                await asyncio.sleep(0.03)
                yield SimpleNamespace(multimodal_output={"actions": [0.0]})

            return _generate()

    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=DelayedEngine())

    async def run_test():
        infer_task = asyncio.create_task(serving.infer({"prompt": "pick"}, session_id="session-a", reset=True))
        ticks = 0
        while not infer_task.done():
            await asyncio.sleep(0.005)
            ticks += 1
        await infer_task
        return ticks

    assert asyncio.run(run_test()) >= 2


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


def test_drop_session_calls_dreamzero_close_hook():
    closed: list[str] = []
    pipeline = SimpleNamespace(close_ar_diffusion_session=lambda session_id: closed.append(session_id))
    engine_client = SimpleNamespace(
        model_config={"policy_server_config": TEST_POLICY_SERVER_CONFIG},
        model_runner=SimpleNamespace(pipeline=pipeline),
    )
    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=engine_client)

    serving.drop_session("rollout-1")

    assert closed == ["rollout-1"]


def _stage(role, *, stage_type="diffusion", model_config=None):
    return SimpleNamespace(
        stage_type=stage_type,
        stage_role=role,
        model_stage=role,
        engine_args=SimpleNamespace(model_config=model_config) if model_config is not None else None,
    )


def _disaggregated_engine(default_sampling_params_list=None, roles=("encode", "denoise", "decode")):
    """Engine double exposing a multi-stage diffusion topology like the deploy yaml."""
    stage_configs = [
        _stage(
            role,
            model_config={"policy_server_config": TEST_POLICY_SERVER_CONFIG} if index == 0 else None,
        )
        for index, role in enumerate(roles)
    ]
    engine = SimpleNamespace(
        get_diffusion_od_config=lambda: None,
        stage_configs=stage_configs,
        num_stages=len(stage_configs),
    )
    if default_sampling_params_list is not None:
        engine.default_sampling_params_list = default_sampling_params_list
    return engine


def test_single_stage_topology_still_sends_one_sampling_params():
    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=_engine_with_policy_config())

    request = serving._build_request({"prompt": "pick"}, session_id="s", reset=True)

    assert serving.stage_roles == ("full",)
    assert len(request.sampling_params_list) == 1
    assert request.sampling_params_list[0] is request.sampling_params
    assert request.sampling_params.extra_args["robot_obs"]["prompt"] == "pick"


def test_three_stage_topology_builds_one_sampling_params_per_stage():
    defaults = [
        OmniDiffusionSamplingParams(num_inference_steps=1, extra_args={"stage": "encode", "shared": {"v": []}}),
        OmniDiffusionSamplingParams(num_inference_steps=2, extra_args={"stage": "denoise"}),
        OmniDiffusionSamplingParams(num_inference_steps=3, extra_args={"stage": "decode"}),
    ]
    serving = openpi_serving.ServingRealtimeRobotOpenPI(
        engine_client=_disaggregated_engine(default_sampling_params_list=defaults)
    )

    request = serving._build_request({"prompt": "pick"}, session_id="wire", reset=True)
    params_list = request.sampling_params_list

    assert serving.stage_roles == ("encode", "denoise", "decode")
    assert len(params_list) == 3
    # Per-stage defaults survive: no stage is a copy of stage 0.
    assert [params.num_inference_steps for params in params_list] == [1, 2, 3]
    assert [params.extra_args["stage"] for params in params_list] == ["encode", "denoise", "decode"]
    # Independent objects and independent mutable containers.
    assert len({id(params) for params in params_list}) == 3
    assert len({id(params.extra_args) for params in params_list}) == 3
    params_list[0].extra_args["shared"]["v"].append("mutated")
    assert defaults[0].extra_args["shared"] == {"v": []}
    # The same normalized identity and lifecycle intent reaches every stage.
    assert [params.extra_args["session_id"] for params in params_list] == ["wire"] * 3
    assert [params.extra_args["reset"] for params in params_list] == [True] * 3
    assert [params.extra_args["close_session"] for params in params_list] == [False] * 3
    # Only the encoder-owning stage receives the raw observation.
    assert params_list[0].extra_args["robot_obs"]["prompt"] == "pick"
    assert "robot_obs" not in params_list[1].extra_args
    assert "robot_obs" not in params_list[2].extra_args


def test_three_stage_topology_shares_one_seed_across_stages():
    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=_disaggregated_engine())

    seeded = serving._build_request({"prompt": "pick", "seed": 7}, session_id="s", reset=True)
    unseeded = serving._build_request({"prompt": "pick"}, session_id="s", reset=False)

    assert [params.seed for params in seeded.sampling_params_list] == [7, 7, 7]
    seeds = {params.seed for params in unseeded.sampling_params_list}
    assert len(seeds) == 1
    assert isinstance(next(iter(seeds)), int)


def test_stage_defaults_are_not_mutated_between_requests():
    defaults = [OmniDiffusionSamplingParams(extra_args={"session_id": "configured", "reset": True}) for _ in range(3)]
    serving = openpi_serving.ServingRealtimeRobotOpenPI(
        engine_client=_disaggregated_engine(default_sampling_params_list=defaults)
    )

    serving._build_request({"prompt": "pick"}, session_id="wire", reset=False)

    for default_params in defaults:
        assert default_params.extra_args == {"session_id": "configured", "reset": True}
        assert default_params.seed is None


def test_typed_tick_overrides_flat_session_controls_on_every_stage():
    """A typed tick is authoritative; a typed False must beat a stale flat True."""
    from vllm_omni.experimental.ar_diffusion.tick_protocol import AR_DIFFUSION_TICK_KEY

    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=_disaggregated_engine())
    tick = {
        "session_id": "typed-session",
        "request_id": "tick-0",
        "chunk_index": 0,
        "reset": False,
        "close_session": True,
    }

    request = serving._build_request(
        {"prompt": "pick", AR_DIFFUSION_TICK_KEY: tick},
        session_id="flat-session",
        reset=True,
        close_session=False,
    )

    for params in request.sampling_params_list:
        assert params.extra_args["session_id"] == "typed-session"
        assert params.extra_args["reset"] is False
        assert params.extra_args["close_session"] is True
        # The typed tick itself is forwarded so the runner applies the same
        # precedence rather than re-deriving from the flat controls.
        assert params.extra_args[AR_DIFFUSION_TICK_KEY] == tick
    assert request.request_id.startswith("robot-typed-session-")


def test_non_diffusion_stage_keeps_its_own_params_and_gets_no_policy_controls():
    defaults = [
        OmniDiffusionSamplingParams(extra_args={"stage": "encode"}),
        OmniDiffusionSamplingParams(extra_args={"stage": "denoise"}),
        OmniDiffusionSamplingParams(extra_args={"stage": "decode"}),
        OmniDiffusionSamplingParams(extra_args={"stage": "llm"}),
    ]
    engine = _disaggregated_engine(default_sampling_params_list=defaults)
    engine.stage_configs = list(engine.stage_configs) + [_stage("llm", stage_type="llm")]
    engine.num_stages = 4
    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=engine)

    request = serving._build_request({"prompt": "pick"}, session_id="s", reset=True)

    assert serving.stage_roles == ("encode", "denoise", "decode", None)
    assert len(request.sampling_params_list) == 4
    assert request.sampling_params_list[3].extra_args == {"stage": "llm"}


def test_topology_without_an_encode_stage_is_rejected_with_an_actionable_error():
    serving = openpi_serving.ServingRealtimeRobotOpenPI(
        engine_client=_disaggregated_engine(roles=("denoise", "decode"))
    )

    with pytest.raises(ValueError) as exc_info:
        serving._build_request({"prompt": "pick"}, session_id="s", reset=True)

    message = str(exc_info.value)
    assert "no encode-capable diffusion stage" in message
    assert "stage_role" in message


def test_infer_sends_every_stage_params_to_generate():
    engine = RecordingEngine()
    engine.stage_configs = _disaggregated_engine().stage_configs
    engine.num_stages = 3
    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=engine)

    actions = asyncio.run(serving.infer({"prompt": "pick"}, session_id="session-a", reset=True))

    np.testing.assert_allclose(actions, np.asarray([0.0], dtype=np.float32))
    sent = engine.generate_calls[0]["sampling_params_list"]
    assert len(sent) == 3
    assert [params.extra_args["session_id"] for params in sent] == ["session-a"] * 3


def test_release_session_waits_for_the_last_connection_holding_it():
    closed: list[str] = []
    engine_client = SimpleNamespace(
        model_config={"policy_server_config": TEST_POLICY_SERVER_CONFIG},
        model_runner=SimpleNamespace(
            pipeline=SimpleNamespace(close_ar_diffusion_session=lambda session_id: closed.append(session_id))
        ),
    )
    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=engine_client)

    serving.acquire_session("shared")
    serving.acquire_session("shared")

    assert asyncio.run(serving.release_session("shared")) is False
    assert closed == []
    assert serving.session_refcount("shared") == 1

    assert asyncio.run(serving.release_session("shared")) is True
    assert closed == ["shared"]
    assert serving.session_refcount("shared") == 0


# -- coordinated remote close ------------------------------------------------


class FakeControlPlane:
    """An engine client that answers the orchestrator's lifecycle control RPC.

    Mirrors the real reply contract: ``True`` for a confirmed close and a
    ``{"supported": False, "error": ...}`` mapping for a failure.
    """

    def __init__(self, *, reply=True, roles=("encode", "denoise", "decode")):
        self.reply = reply
        self.calls: list[dict] = []
        self.stage_configs = [
            SimpleNamespace(
                stage_type="diffusion",
                stage_role=role,
                model_stage=role,
                coordinated_session_lifecycle=True,
                engine_args=(
                    SimpleNamespace(model_config={"policy_server_config": TEST_POLICY_SERVER_CONFIG})
                    if index == 0
                    else None
                ),
            )
            for index, role in enumerate(roles)
        ]
        self.num_stages = len(self.stage_configs)

    def get_diffusion_od_config(self):
        return None

    async def collective_rpc(self, *, method, args=(), timeout=None, **kwargs):
        self.calls.append({"method": method, "args": args, "timeout": timeout})
        if isinstance(self.reply, Exception):
            raise self.reply
        return [self.reply]


def test_close_operation_name_matches_the_orchestrator():
    """The name is duplicated to keep the orchestrator out of the serving imports."""
    from vllm_omni.engine import orchestrator

    assert openpi_serving.CLOSE_COORDINATED_SESSION == orchestrator.CLOSE_COORDINATED_SESSION


def test_a_coordinated_topology_closes_through_the_engine_control_plane():
    engine = FakeControlPlane()
    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=engine)

    assert serving.coordinated_session_lifecycle is True
    serving.acquire_session("A")
    assert asyncio.run(serving.release_session("A")) is True

    assert [call["method"] for call in engine.calls] == [openpi_serving.CLOSE_COORDINATED_SESSION]
    assert engine.calls[0]["args"] == ("A",)
    assert engine.calls[0]["timeout"] == serving.session_close_timeout_s


def test_a_coordinated_close_that_is_not_confirmed_raises_and_is_remembered():
    engine = FakeControlPlane(reply={"supported": False, "error": "denoise stage unreachable"})
    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=engine)
    serving.acquire_session("A")

    with pytest.raises(RuntimeError, match="denoise stage unreachable"):
        asyncio.run(serving.release_session("A"))

    # The unaccounted-for state is not forgotten just because the reference went.
    assert serving.unresolved_closes == frozenset({"A"})
    with pytest.raises(RuntimeError, match="unresolved failed close"):
        serving.acquire_session("A")


def test_a_coordinated_deployment_without_a_control_plane_is_an_error():
    engine = FakeControlPlane()
    del engine.collective_rpc
    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=engine)
    serving.acquire_session("A")

    with pytest.raises(RuntimeError, match="no collective_rpc"):
        asyncio.run(serving.release_session("A"))


def test_an_uncoordinated_policy_keeps_the_local_hook_and_tolerates_no_hook():
    closed: list[str] = []
    engine = SimpleNamespace(
        model_config={"policy_server_config": TEST_POLICY_SERVER_CONFIG},
        model_runner=SimpleNamespace(
            pipeline=SimpleNamespace(close_ar_diffusion_session=lambda session_id: closed.append(session_id))
        ),
    )
    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=engine)

    assert serving.coordinated_session_lifecycle is False
    serving.acquire_session("A")
    assert asyncio.run(serving.release_session("A")) is True
    assert closed == ["A"]

    # A stateless policy with no hook at all is still supported.
    stateless = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=_engine_with_policy_config())
    stateless.acquire_session("B")
    assert asyncio.run(stateless.release_session("B")) is True


def test_a_shared_session_is_only_closed_by_its_last_holder():
    engine = FakeControlPlane()
    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=engine)
    serving.acquire_session("shared")
    serving.acquire_session("shared")

    assert asyncio.run(serving.release_session("shared")) is False
    assert engine.calls == []

    assert asyncio.run(serving.release_session("shared")) is True
    assert [call["args"] for call in engine.calls] == [("shared",)]


def test_a_cancelled_release_leaves_the_session_unresolved():
    """An aborted await does not prove the remote close stopped."""
    entered = asyncio.Event()

    class HangingControlPlane(FakeControlPlane):
        async def collective_rpc(self, *, method, args=(), timeout=None, **kwargs):
            self.calls.append({"method": method, "args": args, "timeout": timeout})
            entered.set()
            await asyncio.sleep(10)

    engine = HangingControlPlane()
    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=engine)
    serving.acquire_session("A")

    async def scenario():
        task = asyncio.create_task(serving.release_session("A"))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    # The unresolved close is remembered and the id cannot be taken again.
    assert serving.unresolved_closes == frozenset({"A"})
    with pytest.raises(RuntimeError, match="unresolved failed close"):
        serving.acquire_session("A")


def test_a_lifecycle_error_on_the_final_output_is_surfaced_verbatim():
    """Not replaced by a complaint about the empty payload it arrived with."""

    class FailingEngine(RecordingEngine):
        def generate(self, *, prompt, request_id, sampling_params_list):
            async def _generate():
                yield SimpleNamespace(
                    error="Engine shut down before session lifecycle cleanup was confirmed",
                    error_type="session_lifecycle_error",
                    multimodal_output=None,
                )

            return _generate()

    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=FailingEngine())

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(serving.infer({"prompt": "pick"}, session_id="s", reset=True))

    message = str(exc_info.value)
    assert "session_lifecycle_error" in message
    assert "lifecycle cleanup was confirmed" in message
    assert "multimodal_output" not in message
