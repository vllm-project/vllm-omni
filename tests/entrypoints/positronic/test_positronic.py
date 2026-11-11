# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Positronic wire, session isolation and native-engine request contracts."""

import io
from collections.abc import Callable
from unittest.mock import Mock
from urllib.parse import urlencode

import msgspec
import numpy as np
import pytest
from fastapi import FastAPI
from PIL import Image
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.praxis1 import processing
from vllm_omni.entrypoints.openpi.serving import ServingRealtimeRobotOpenPI
from vllm_omni.entrypoints.positronic import codec, router
from vllm_omni.entrypoints.positronic.praxis import parse_session
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
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


@pytest.mark.parametrize("path", ["/api/v1/session", "/api/v1/session/praxis-1-droid", "/api/v1/session/praxis-1"])
@pytest.mark.parametrize("rollout", ["run1", '"run1"', "20261104"])
def test_routes_handshake_and_command_envelope(server, path, rollout):
    client, engine = server
    assert client.get("/api/v1/models").json() == {"models": ["praxis-1-droid"]}
    with client.websocket_connect(path + "?" + urlencode({"sample_seed": 7, "rollout_id": rollout})) as ws:
        meta = ready(ws)
        assert meta["rollout_id"] == rollout.strip('"') and meta["seed_source"] == "session_param"
        assert meta["sample_seed_base"] == 7 and meta["positronic_wire_ref"] == "d5f2a47"
        seq = meta["local_stack"]["seq"]
        assert [x["name"] for x in seq] == [
            "stop_on_fault",
            "restrict_image_size",
            "temporal_stack",
            "chunked_schedule",
        ]
        np.testing.assert_allclose(
            seq[2]["args"]["offsets_sec"], [0.0 if i == 0 else -i / 15 + 0.5 / 15 for i in range(24, -1, -1)]
        )
        assert seq[1]["args"] == {"width": 416, "height": 234}
        for i in range(2):
            ws.send_bytes(
                codec.pack(
                    {
                        "task": "move block",
                        "seed": 1000,
                        "session_id": "injected",
                        "sampling_params": {"num_inference_steps": 1000},
                    }
                )
            )
            answer = codec.unpack(ws.receive_bytes())
            assert answer["effective_prompt"] == "move block" and answer["prompt_original"] is None
            row = answer["result"][0]
            np.testing.assert_array_equal(row["robot_command"]["pose"], np.arange(12))
            assert row["target_grip"] == 0.5 and row["timestamp"] == 0.0
            assert "server_timing" in answer
            params = engine.calls[-1]
            assert params.seed == 7 and params.num_inference_steps == 2
            assert params.extra_args["reset"] is False
            assert params.extra_args["session_id"] != "injected"
            assert "sampling_params" not in params.extra_args["robot_obs"]
            assert not params.extra_args.get("return_diagnostics")
    assert not engine.sessions
    assert engine.calls[-1].extra_args["close_session"]


@pytest.mark.parametrize(
    "suffix", ["/unknown", "?smoothing=savgol", "?num_steps=0", "?pad_start=true", "?no_such_key=1"]
)
def test_refuses_unsupported_sessions_before_inference(server, suffix):
    client, engine = server
    with client.websocket_connect("/api/v1/session" + suffix) as ws:
        assert codec.unpack(ws.receive_bytes())["status"] == "error"
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_bytes()
        assert exc.value.code == 1008
    assert not engine.calls


def test_non_policy_deployment_does_not_advertise_positronic(server):
    client, engine = server
    engine.config.model_config["policy_server_config"].pop("policy")
    client.app.state.openai_serving_realtime_robot = ServingRealtimeRobotOpenPI(engine)
    assert client.get("/api/v1/models").json() == {"models": []}
    with client.websocket_connect("/api/v1/session") as ws:
        assert codec.unpack(ws.receive_bytes())["status"] == "error"


def test_binary_frame_requirement(server):
    client, engine = server
    with client.websocket_connect("/api/v1/session") as ws:
        ready(ws)
        ws.send_text("{}")
        assert codec.unpack(ws.receive_bytes())["status"] == "error"
    assert all(p.extra_args.get("close_session") for p in engine.calls) and not engine.sessions


@pytest.mark.parametrize(
    "query",
    [
        {"num_steps": "true"},
        {"num_steps": "100000000"},
        {"cfg_batched": "true"},
        {"cfg_scale": "NaN"},
        {"t_shift": "0"},
        {"max_context_chunks": "4"},
        {"max_chunks_per_episode": "0"},
        {"sample_seed": "-1"},
        {"history_frames": "23"},
        {"stack_keys": '["wrong"]'},
        {"context_mode": "rolling"},
        {"solver": "heun"},
        {"compile": "true"},
    ],
)
def test_rejects_unsupported_or_mistyped_settings(query):
    with pytest.raises(ValueError):
        parse_session(query, {})


@pytest.mark.parametrize("ndim", [3, 4])
def test_raw_and_jpeg_markers(ndim):
    pixels: np.ndarray = np.full((5, 7, 3), 100, np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="JPEG", quality=90)
    with Image.open(io.BytesIO(buf.getvalue())) as im:
        expected = np.asarray(im.convert("RGB"))
    marker = {b"__jpeg__": True, b"frames": [buf.getvalue()] * (2 if ndim == 4 else 1), b"ndim": ndim}
    result = codec.unpack(codec.pack({"image.exterior": marker, "raw": pixels}))
    np.testing.assert_array_equal(result["image.exterior"], np.stack([expected, expected]) if ndim == 4 else expected)
    np.testing.assert_array_equal(result["raw"], pixels)


def test_h264_is_refused_without_decoding():
    assert codec.IMAGE_CODECS == ("jpeg",)
    with pytest.raises(ValueError, match="H.264 decoding is unsupported"):
        codec.unpack(codec.pack({"image.exterior": {b"__h264__": True, b"data": b"invalid stream"}}))


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {b"bytes_key": 1},
        {"image": {b"__ndarray__": True, b"dtype": "O", b"shape": [1], b"data": b"12345678"}},
        {"image.exterior": {b"__jpeg__": True, b"frames": [b"x"] * 26, b"ndim": 4}},
        {"image.exterior": {b"__h264__": True, b"shape": [100, 5, 7, 3], b"ndim": 4, b"data": b"x"}},
    ],
)
def test_malformed_wire_is_rejected(payload):
    with pytest.raises((ValueError, TypeError)):
        codec.unpack(msgspec.msgpack.encode(payload))
