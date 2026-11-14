# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Positronic session lifecycle, isolation, authentication and rig-geometry contracts."""

import asyncio
import io
import threading
from collections.abc import Callable
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from fastapi import FastAPI
from PIL import Image
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.praxis1 import processing
from vllm_omni.entrypoints.openpi.serving import ServingRealtimeRobotOpenPI
from vllm_omni.entrypoints.positronic import codec, router
from vllm_omni.entrypoints.positronic.praxis import parse_session, session_metadata
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras.praxis1 import PraxisSessionOptions
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class RecordingEngine:
    prepare: Callable[[dict], processing.ChunkInputs] | None = None
    stage_configs: list[object] = []

    def __init__(self):
        self.default_sampling_params_list = [OmniDiffusionSamplingParams(num_inference_steps=2)]
        self.config = Mock(
            spec=OmniDiffusionConfig,
            model_config={
                "max_context_chunks": 1,
                "exec_horizon": 24,
                "control_mode": "droid_impedance",
                "policy_server_config": {
                    "policy": "praxis",
                    "model_id": "praxis-1-droid",
                    "supports_close_session": True,
                    "image_resolution": [234, 416],
                    "tile_size": [224, 416],
                    "fps": 15,
                    "video_antialias": True,
                },
            },
        )
        self.calls = []
        self.sessions = set()
        self.fail = False

    def get_diffusion_od_config(self):
        return self.config

    async def generate(self, *, prompt, request_id, sampling_params_list):
        params = sampling_params_list[0]
        extra = params.extra_args
        session_id = extra["session_id"]
        if extra.get("reserve_session"):
            available = len(self.sessions) < 4
            if available:
                self.sessions.add(session_id)
            yield Mock(spec=OmniRequestOutput, multimodal_output={"trajectory": {"session_available": available}})
            return
        self.calls.append(params)
        if extra.get("close_session"):
            self.sessions.discard(session_id)
            yield Mock(spec=OmniRequestOutput, multimodal_output={"actions": np.empty((0, 13), np.float32)})
            return
        self.sessions.add(session_id)
        if self.fail:
            raise RuntimeError("worker failed after allocating session")
        if self.prepare is not None:
            self.prepared = self.prepare(extra["robot_obs"])
        commands = [
            {
                "robot_command": {
                    "type": "cartesian_pos",
                    "pose": np.arange(12, dtype=np.float32),
                    "mode": {"controller": "test"},
                },
                "target_grip": 0.5,
                "timestamp": 0.0,
            }
        ]
        yield Mock(
            spec=OmniRequestOutput,
            multimodal_output={"actions": np.zeros((1, 13), np.float32), "trajectory": {"commands": commands}},
        )


@pytest.fixture
def server():
    engine = RecordingEngine()
    app = FastAPI()
    app.include_router(router)
    app.state.openai_serving_realtime_robot = ServingRealtimeRobotOpenPI(engine, model_name="praxis-1")
    with TestClient(app) as client:
        yield client, engine


def ready(ws):
    assert codec.unpack(ws.receive_bytes())["status"] == "loading"
    response = codec.unpack(ws.receive_bytes())
    assert response["status"] == "ready"
    return response["meta"]


def test_connections_isolate_sessions_and_cleanup_after_failure(server):
    client, engine = server
    with client.websocket_connect("/api/v1/session?rollout_id=same") as a:
        ready(a)
        with client.websocket_connect("/api/v1/session?rollout_id=same") as b:
            ready(b)
            for ws in (a, b):
                ws.send_bytes(codec.pack({"task": "move"}))
                assert "result" in codec.unpack(ws.receive_bytes())
            assert len(engine.sessions) == 2
        assert len(engine.sessions) == 1
        engine.fail = True
        a.send_bytes(codec.pack({"task": "move"}))
        assert "error" in codec.unpack(a.receive_bytes())
        with pytest.raises(WebSocketDisconnect):
            a.receive_bytes()
    assert not engine.sessions


@pytest.mark.parametrize("timing", ["false", "0", "off"])
def test_budget_and_disabled_timing(server, timing):
    client, engine = server
    with client.websocket_connect(
        "/api/v1/session?sample_seed=7&max_chunks_per_episode=1&return_server_timing=" + timing
    ) as ws:
        ready(ws)
        ws.send_bytes(codec.pack({"task": "move"}))
        assert "server_timing" not in codec.unpack(ws.receive_bytes())
        ws.send_bytes(codec.pack({"task": "move"}))
        assert "max_chunks_per_episode" in codec.unpack(ws.receive_bytes())["error"]
    assert not engine.sessions
    assert len(engine.calls) == 2  # one inference and one worker cleanup


@pytest.mark.parametrize("query", [{"context_mode": "stateless"}, {"max_context_chunks": "0"}, {"exec_horizon": "12"}])
def test_stateless_session_declares_no_temporal_stack(query):
    options, values = parse_session(query, {})
    meta = session_metadata(options, values, "praxis", ["jpeg"])
    assert meta["context_mode"] == "stateless"
    stack = meta["local_stack"]
    assert isinstance(stack, dict)
    assert "temporal_stack" not in [s["name"] for s in stack["seq"]]


def test_session_parameters_preserve_deploy_defaults_and_allow_matching_fixed_settings():
    options, values = parse_session(
        {
            "sample_seed": "7",
            "cfg_scale": "2.5",
            "num_steps": "4",
            "action_steps": "4",
            "solver": "euler",
            "stack_keys": ",".join(("image.wrist", "image.exterior", "robot_state.ee_pose", "grip")),
        },
        {"max_context_chunks": 3, "control_mode": "none"},
    )
    assert options == PraxisSessionOptions(num_steps=4, cfg_scale=2.5, max_context_chunks=3, control_mode="none")
    assert values["sample_seed"] == 7


def test_decoded_image_budget_is_enforced(monkeypatch):
    monkeypatch.setattr(codec, "MAX_IMAGE_BYTES", 1)
    with pytest.raises(ValueError, match="budget"):
        codec.unpack(codec.pack({"image": np.zeros((2, 2, 3), np.uint8)}))


def test_replicated_policy_is_refused_before_session_allocation(server):
    from vllm_omni.config.omni_config import VllmOmniDiffusionStageConfig
    from vllm_omni.config.stage_config import StagePipelineConfig

    client, engine = server
    stage = VllmOmniDiffusionStageConfig(stage_pipeline_config=StagePipelineConfig(stage_id=0, model_stage="diffusion"))
    stage.runtime_config.num_replicas = 2
    engine.stage_configs = [stage]
    with client.websocket_connect("/api/v1/session") as ws:
        assert "one worker replica" in codec.unpack(ws.receive_bytes())["error"]
    assert not engine.calls


@pytest.mark.parametrize("cli_keys,env_key", [(["cli-key"], "env-key"), (None, "env-key"), (["first", "second"], None)])
def test_api_keys_guard_model_listing_and_websockets(server, monkeypatch, cli_keys, env_key):
    from argparse import Namespace

    client, engine = server
    client.app.state.args = Namespace(api_key=cli_keys)
    if env_key:
        monkeypatch.setenv("VLLM_API_KEY", env_key)
    else:
        monkeypatch.delenv("VLLM_API_KEY", raising=False)
    assert client.get("/api/v1/models").status_code == 401
    assert client.get("/api/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
    for path in ("/api/v1/session", "/api/v1/session/praxis-1-droid"):
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(path):
                pytest.fail("unauthenticated socket accepted")
        assert exc.value.code == 1008
    token = cli_keys[-1] if cli_keys else env_key
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/api/v1/models", headers=headers).status_code == 200
    with client.websocket_connect("/api/v1/session", headers=headers) as ws:
        assert ready(ws)["policy"] == "praxis"
    if cli_keys and env_key:
        assert client.get("/api/v1/models", headers={"Authorization": f"Bearer {env_key}"}).status_code == 401
    assert all(p.extra_args.get("close_session") for p in engine.calls) and not engine.sessions


@pytest.mark.parametrize("stateless", [False, True])
def test_handshake_matches_validated_checkpoint_geometry(server, stateless):
    client, engine = server
    config = engine.config.model_config["policy_server_config"]
    config.update(video_antialias=True, image_resolution=[180, 320], tile_size=[176, 320], fps=10)
    client.app.state.openai_serving_realtime_robot.policy_server_config.values.update(config)
    with client.websocket_connect(
        "/api/v1/session?sample_seed=7&context_mode=" + ("stateless" if stateless else "observed")
    ) as ws:
        meta = ready(ws)
        assert meta["tile_size"] == config["tile_size"]
        assert (meta["cfg_scale"], meta["cfg_video_scale"], meta["t_shift"]) == (1.0, 4.0, 12.5)
        seq = meta["local_stack"]["seq"]
        names = [step["name"] for step in seq]
        restrict = next(step for step in seq if step["name"] == "restrict_image_size")
        assert restrict["args"] == dict(zip(("height", "width"), config["image_resolution"]))
        assert meta["action_fps"] == config["fps"]
        assert names.index("restrict_image_size") == 1
        if not stateless:
            offsets = next(step for step in seq if step["name"] == "temporal_stack")["args"]["offsets_sec"]
            np.testing.assert_array_equal(offsets[:-1], np.arange(-24, 0) / config["fps"] + 0.5 / config["fps"])
            assert offsets[-1] == 0


@pytest.mark.parametrize("timing", ["true", "1", "on"])
def test_per_call_sampler_metadata_matches_ready(server, timing):
    client, _ = server
    with client.websocket_connect("/api/v1/session?sample_seed=7&return_server_timing=" + timing) as ws:
        meta = ready(ws)
        ws.send_bytes(codec.pack({"task": "pick"}))
        response = codec.unpack(ws.receive_bytes())
        for key in ("cfg_scale", "cfg_video_scale", "t_shift", "solver", "cfg_batched", "num_evals"):
            assert response["server_timing"][key] == meta[key]
        assert meta["action_fps"] == 15


@pytest.mark.parametrize("pipeline_next", [False, True])
def test_disconnect_or_early_message_releases_worker_state(server, monkeypatch, pipeline_next):
    client, engine = server
    started, cancelled, released = threading.Event(), threading.Event(), threading.Event()
    original = engine.generate

    async def inflight(*, prompt, request_id, sampling_params_list):
        extra = sampling_params_list[0].extra_args
        if extra.get("close_session") or extra.get("reserve_session"):
            async for output in original(
                prompt=prompt, request_id=request_id, sampling_params_list=sampling_params_list
            ):
                yield output
            if extra.get("close_session"):
                released.set()
        else:
            engine.sessions.add(extra["session_id"])
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

    monkeypatch.setattr(engine, "generate", inflight)
    with client.websocket_connect("/api/v1/session?sample_seed=7") as ws:
        ready(ws)
        ws.send_bytes(codec.pack({"task": "move"}))
        assert started.wait(5)
        if pipeline_next:
            ws.send_bytes(codec.pack({"task": "move again"}))
            assert ws.receive()["code"] == 1008
        else:
            ws.close()
        assert cancelled.wait(5) and released.wait(5)
        assert client.get("/api/v1/models").status_code == 200
    assert not engine.sessions


def test_capacity_is_refused_before_ready_and_reusable(server):
    client, engine = server
    from contextlib import ExitStack

    with ExitStack() as sockets:
        for _ in range(4):
            ready(sockets.enter_context(client.websocket_connect("/api/v1/session")))
        with client.websocket_connect("/api/v1/session") as ws:
            assert codec.unpack(ws.receive_bytes())["status"] == "loading"
            assert "capacity" in codec.unpack(ws.receive_bytes())["error"]
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_bytes()
            assert exc.value.code == 1008
    assert not engine.sessions
    with client.websocket_connect("/api/v1/session") as ws:
        ready(ws)


@pytest.mark.parametrize("key", ["task", "prompt"])
def test_binary_prompt_limit_precedes_inference(server, key):
    client, engine = server
    with client.websocket_connect("/api/v1/session") as ws:
        ready(ws)
        ws.send_bytes(codec.pack({key: b"x" * 65537}))
        response = codec.unpack(ws.receive_bytes())
        assert response["status"] == "error" and response["error"].startswith("Invalid observation")
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_bytes()
        assert closed.value.code == 1008
    assert all(p.extra_args.get("close_session") for p in engine.calls) and not engine.sessions


@pytest.mark.parametrize("payload", [b"\xc1", codec.pack({"image.exterior": {b"__h264__": True}})])
def test_invalid_observations_are_refused_not_failed(server, payload, caplog):
    """Client errors close with 1008 and an error payload; engine failures keep 1011."""
    client, engine = server
    with client.websocket_connect("/api/v1/session") as ws:
        ready(ws)
        ws.send_bytes(payload)
        response = codec.unpack(ws.receive_bytes())
        assert response["status"] == "error" and response["error"].startswith("Invalid observation")
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_bytes()
        assert closed.value.code == 1008
    assert not any(record.levelname == "ERROR" for record in caplog.records)
    assert not engine.sessions


def test_refusal_reason_fits_a_close_frame(server):
    """Query values echo into refusals; the close reason must stay within 123 UTF-8 bytes."""
    client, _ = server
    with client.websocket_connect("/api/v1/session?solver=" + "%C3%A9" * 100) as ws:
        response = codec.unpack(ws.receive_bytes())
        assert response["status"] == "error" and "solver" in response["error"]
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_bytes()
        assert closed.value.code == 1008 and len(closed.value.reason.encode("utf-8")) <= 123


def test_client_camera_history_reaches_worker_preparation(server):
    client, engine = server
    encode = Mock(return_value=torch.zeros(1, 192, 1, 7, 26))
    stats = processing.NormStats(-np.ones(10), np.ones(10), np.zeros(10), np.ones(10))
    state = processing.ServeSession(encode, stats, "cpu", tile_size=(224, 416))
    engine.prepare = state.prepare

    def camera(offset):
        frames = []
        for i in range(25):
            buf = io.BytesIO()
            Image.fromarray(np.full((234, 416, 3), i + offset, np.uint8)).save(buf, format="JPEG", quality=90)
            frames.append(buf.getvalue())
        return {b"__jpeg__": True, b"frames": frames, b"ndim": 4}

    poses = np.tile([0.4, 0, 0.3, 1, 0, 0, 0], (25, 1)).astype(np.float32)
    poses[:, 0] += np.arange(25) * 0.001
    obs = {
        "image.exterior": camera(0),
        "image.wrist": camera(64),
        "image.exterior_2": {b"__jpeg__": True, b"frames": [b"invalid extra view"]},
        "robot_state.ee_pose": poses,
        "grip": np.linspace(0, 1, 25),
        "task": "pick",
        "descriptor": "praxis.sim",
    }
    with client.websocket_connect("/api/v1/session") as ws:
        ready(ws)
        ws.send_bytes(codec.pack(obs))
        response = codec.unpack(ws.receive_bytes())
        assert "result" in response and response["server_timing"]["image_codec"] == "jpeg"
    np.testing.assert_allclose(engine.prepared.pose_now[:3], poses[-1, :3])
    assert engine.prepared.pose_now[-1] == 1
    image = encode.call_args.args[0]
    assert torch.all(image[:, :, -1, :, :416] == 24) and torch.all(image[:, :, -1, :, 416:] == 88)
    assert not engine.sessions


def test_yam_handshake_and_all_three_jpeg_cameras(server):
    client, engine = server
    # Deployment defaults of the two-arm YAM checkpoint.
    defaults: dict[str, Any] = {
        "max_num_steps": 16,
        "session_idle_timeout": 300,
        "warmup": True,
        "max_sessions": 4,
        "max_context_chunks": 1,
        "exec_horizon": 24,
        "cfg_scale": 1.0,
        "cfg_video_scale": 1.0,
        "t_shift": 1.0,
        "control_mode": "none",
        "policy_server_config": {
            "policy": "praxis",
            "model_id": "praxis-1-yam",
            "needs_session_id": True,
            "supports_close_session": True,
            "image_resolution": [416, 416],
            "tile_size": [256, 416],
            "video_antialias": True,
            "action_space": "cartesian_position_rotation_matrix_gripper",
            "action_dim": 26,
            "action_horizon": 24,
            "first_action_horizon": 21,
            "fps": 15,
            "control_mode": "none",
            "embodiment": "yam_ee2",
            "pose_frame": "rig",
            "action_layout": "ee-20d [L: xyz3 rot6d6 grip1 | R: same], rel-to-anchor, grip abs",
            "wire_commands": ["robot_command.left", "robot_command.right", "target_grip.left", "target_grip.right"],
            "camera_keys": ["image.exterior", "image.wrist_left", "image.wrist_right"],
            "stack_keys": [
                "image.exterior",
                "image.wrist_left",
                "image.wrist_right",
                "robot_state.left.ee_pose",
                "robot_state.right.ee_pose",
                "grip.left",
                "grip.right",
            ],
        },
    }
    engine.config.model_config = defaults
    from vllm_omni.entrypoints.openpi.serving import PolicyServerConfig

    metadata = PolicyServerConfig.from_model_config(defaults).to_dict()
    assert metadata["camera_keys"] == ["image.exterior", "image.wrist_left", "image.wrist_right"]
    assert "needs_wrist_camera" not in metadata and "n_external_cameras" not in metadata
    client.app.state.openai_serving_realtime_robot.policy_server_config.values.update(defaults["policy_server_config"])
    pixels = np.zeros((260, 416, 3), np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="JPEG")
    marker = {b"__jpeg__": True, b"frames": [buf.getvalue()], b"ndim": 3}
    with client.websocket_connect("/api/v1/session?sample_seed=7") as ws:
        meta = ready(ws)
        assert meta["embodiment"] == "yam_ee2" and meta["pose_frame"] == "rig"
        assert (meta["cfg_video_scale"], meta["t_shift"], meta["control_mode"]) == (1.0, 1.0, "none")
        stack = next(s for s in meta["local_stack"]["seq"] if s["name"] == "temporal_stack")
        assert stack["args"]["keys"] == defaults["policy_server_config"]["stack_keys"]
        ws.send_bytes(codec.pack({k: marker for k in defaults["policy_server_config"]["camera_keys"]}))
        assert "result" in codec.unpack(ws.receive_bytes())
        obs = engine.calls[-1].extra_args["robot_obs"]
        assert all(obs[k].shape == (260, 416, 3) for k in defaults["policy_server_config"]["camera_keys"])
    with client.websocket_connect("/api/v1/session?control_mode=droid_impedance") as ws:
        assert "error" in codec.unpack(ws.receive_bytes())
