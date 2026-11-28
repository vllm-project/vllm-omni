# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Exercise the real HTTP bridge and transport against a deterministic vLLM API."""

import base64
import io
import json
import threading
from dataclasses import replace
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np
import pytest
import requests
from PIL import Image

from vllm_omni.entrypoints.cosmos3.action_policy_common import LiberoActionNormalizer
from vllm_omni.entrypoints.cosmos3.action_policy_http import ActionPolicyHTTPServer
from vllm_omni.entrypoints.cosmos3.libero_policy_inputs import prepare_libero_policy_inputs
from vllm_omni.entrypoints.cosmos3.libero_remote_policy import LiberoRemotePolicyService, RemotePolicyConfig
from vllm_omni.entrypoints.cosmos3.remote_action_stream import RemoteActionPolicyHTTPServer
from vllm_omni.entrypoints.cosmos3.vllm_omni_action import VllmOmniActionTransport


def request_item(value=30):
    image = np.full((256, 512, 3), value, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return {
        "image": base64.b64encode(buffer.getvalue()).decode(),
        "prompt": "pick up the object",
        "domain_name": "libero",
        "image_size": 256,
    }


@pytest.fixture
def bridge(tmp_path):
    state = SimpleNamespace(
        posts=[],
        deleted=[],
        jobs={},
        mode="completed",
        profile=True,
        stream_callbacks=True,
        confirm_early_stop=True,
        callback_urls=[],
        callback_finished={},
        release_final=threading.Event(),
    )
    state.release_final.set()

    class BackendHandler(BaseHTTPRequestHandler):
        def reply(self, data):
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            content_type = self.headers["Content-Type"]
            message = BytesParser(policy=policy.default).parsebytes(
                f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
                + self.rfile.read(int(self.headers["Content-Length"]))
            )
            fields = {
                part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
                for part in message.iter_parts()
            }
            pixels = np.asarray(Image.open(io.BytesIO(fields["input_reference"]))).copy()
            state.posts.append((fields, pixels))
            job_id = f"job-{len(state.posts)}"
            state.jobs[job_id] = float(pixels[0, 0, 0]) / 255
            finished = threading.Event()
            state.callback_finished[job_id] = finished
            self.reply({"id": job_id})
            extra = json.loads(fields["extra_params"])
            count = extra.get("early_stop_actions") or 16
            stream = extra.get("action_stream")
            if stream and state.stream_callbacks:
                callback_session = requests.Session()
                callback_session.trust_env = False
                state.callback_urls.append(stream["callback_url"])
                try:
                    final_step = 5 if count == 4 else 10
                    for step, indices in [(1, [0]), (final_step, list(range(1, count)))]:
                        response = callback_session.post(
                            stream["callback_url"],
                            json={
                                "type": "partial",
                                "step": step,
                                "total_steps": 10,
                                "action_indices": indices,
                                "actions": np.full((len(indices), 10), state.jobs[job_id]).tolist(),
                                "policy_profile": "libero10_native_nonstream_v1",
                                "action_stream_version": 1,
                            },
                            timeout=3,
                        )
                        response.raise_for_status()
                        if step == 1:
                            state.release_final.wait(timeout=3)
                finally:
                    callback_session.close()
            finished.set()

        def do_GET(self):
            if self.path == "/v1/models":
                self.reply({"data": [{"id": "edge-model"}]})
                return
            job_id = self.path.rsplit("/", 1)[-1]
            state.callback_finished[job_id].wait(timeout=5)
            value = state.jobs[job_id]
            fields = state.posts[int(job_id.split("-")[-1]) - 1][0]
            extra = json.loads(fields["extra_params"])
            count = extra.get("early_stop_actions") or 16
            action = {"data": np.full((count, 10), value).tolist(), "shape": [count, 10], "dtype": "float32"}
            if state.profile:
                action["policy_profile"] = extra["policy_profile"]
            if state.confirm_early_stop:
                action.update(
                    early_stop_actions=extra.get("early_stop_actions"),
                    denoise_steps=5 if count == 4 and extra["action_time_schedule"] == "has_euler" else 10,
                    requested_denoise_steps=10,
                )
            self.reply(
                {
                    "status": state.mode,
                    "action": action,
                    "error": "backend failed",
                    "inference_time_s": 0.0125,
                    "stage_durations": {"diffusion": 12.5},
                    "peak_memory_mb": 128,
                }
            )

        def do_DELETE(self):
            state.deleted.append(self.path)
            self.reply({})

        def log_message(self, *args):
            pass

    backend = ThreadingHTTPServer(("127.0.0.1", 0), BackendHandler)
    transport = VllmOmniActionTransport(f"http://127.0.0.1:{backend.server_port}", poll_interval=0, timeout=2)
    service = LiberoRemotePolicyService(transport, RemotePolicyConfig(dump_dir=tmp_path))
    server = RemoteActionPolicyHTTPServer(("127.0.0.1", 0), service)
    state.service = service
    threads = [threading.Thread(target=item.serve_forever, daemon=True) for item in (backend, server)]
    for thread in threads:
        thread.start()
    session = requests.Session()
    session.trust_env = False
    try:
        yield f"http://127.0.0.1:{server.server_port}", state, session, tmp_path
    finally:
        for item, thread in zip((server, backend), reversed(threads), strict=True):
            item.shutdown()
            item.server_close()
            thread.join(timeout=2)
        session.close()
        service.close()


def test_three_terminal_http_path_preserves_inputs_and_denormalizes_once(bridge):
    url, state, session, dumps = bridge
    assert session.get(url, timeout=3).status_code == 200
    info = session.get(url + "/info", timeout=3).json()
    assert info["backend"] == "vllm-omni" and not info["profile_confirmed"]
    assert info["measure_ttfa"] and "nonstream" in info["ttfa_scope"]
    assert info["timing_schema"] == "pi_libero_v1"
    response = session.post(url + "/predict", json=request_item(), timeout=3)
    response.raise_for_status()
    result = response.json()
    expected = LiberoActionNormalizer().denormalize(np.full((16, 10), 30 / 255, dtype=np.float32))
    np.testing.assert_array_equal(result["action"], expected)
    assert result["video"] == []
    assert result["timing_ms"]["ttfa"] == result["timing_ms"]["full_chunk"]
    assert result["timing_ms"]["remote_roundtrip"] <= result["timing_ms"]["full_chunk"]
    assert result["server_timing"]["infer_ms"] == result["timing_ms"]["full_chunk"]
    assert result["benchmark"]["policy_wall_ms"] == result["policy_timing"]["infer_ms"]
    assert result["policy_timing"]["infer_ms"] == result["server_timing"]["infer_ms"]
    assert result["policy_timing"]["backend_ms"] == result["timing_ms"]["backend_roundtrip"]
    assert result["policy_timing"]["backend_ms"] <= result["benchmark"]["policy_wall_ms"]
    assert result["performance"]["generated_actions"] == 16
    assert result["performance"]["worker_metrics"][0]["worker_inference_ms"] == 12.5
    assert result["performance"]["worker_metrics"][0]["stage_durations"] == {"diffusion": 12.5}
    assert len(result["performance"]["action_timings"]) == 16
    assert not result["performance"]["streaming"]
    fields, pixels = state.posts[0]
    expected_pixels, caption, frame_size = prepare_libero_policy_inputs(
        np.full((256, 512, 3), 30, dtype=np.uint8), "pick up the object", 256
    )
    np.testing.assert_array_equal(pixels, expected_pixels)
    assert fields["prompt"].decode() == caption
    assert json.loads(fields["extra_params"])["native_frame_size"] == frame_size
    assert len(state.deleted) == 1
    assert session.get(url + "/info", timeout=3).json()["profile_confirmed"]
    assert list(dumps.glob("request_*/action_000.json"))


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_synthetic_warmup_uses_real_request_path_without_advancing_seed(bridge, streaming, batch_size):
    url, state, session, _ = bridge
    state.service.cfg = replace(state.service.cfg, dump_dir=None, streaming_enabled=streaming, callback_base_url=url)
    rng_state = np.random.get_state()
    response = session.post(
        url + "/warmup", json={"requests": 2, "batch_size": batch_size, "streaming": streaming}, timeout=10
    )
    response.raise_for_status()
    warmup = response.json()
    assert warmup["requests"] == len(warmup["timings"]) == 2
    assert warmup["seed"] == state.service.transport.seed == 0
    assert len(state.posts) == len(state.deleted) == 2 * batch_size
    assert all(fields["seed"] == b"0" for fields, _ in state.posts)
    assert bool(state.callback_urls) == streaming
    assert session.get(url + "/info", timeout=3).json()["synthetic_warmup"]["completed_requests"] == 2
    current = np.random.get_state()
    np.testing.assert_array_equal(current[1], rng_state[1])
    assert current[2:] == rng_state[2:]


def test_batch_keeps_order_and_validates_all_items_before_submission(bridge):
    url, state, session, _ = bridge
    response = session.post(url + "/predict_batch", json={"items": [request_item(10), request_item(90)]}, timeout=3)
    response.raise_for_status()
    actions = response.json()["actions"]
    for action, value in zip(actions, (10, 90), strict=True):
        expected = LiberoActionNormalizer().denormalize(np.full((16, 10), value / 255, dtype=np.float32))
        np.testing.assert_array_equal(action, expected)
    assert len(state.deleted) == 2
    invalid = request_item()
    invalid["domain_name"] = "robocasa"
    response = session.post(url + "/predict_batch", json={"items": [request_item(), invalid]}, timeout=3)
    assert response.status_code == 400 and response.json()["actions"] == []
    assert len(state.posts) == 2


@pytest.mark.parametrize("profile,mode", [(False, "completed"), (True, "failed"), (True, "cancelled")])
def test_backend_failures_reach_evaluator_and_jobs_are_cleaned_up(bridge, profile, mode):
    url, state, session, dumps = bridge
    state.profile, state.mode = profile, mode
    response = session.post(url + "/predict", json=request_item(), timeout=3)
    assert response.status_code == 400
    assert response.json()["action"] == [] and response.json()["error"]
    assert len(state.deleted) == 1
    assert list(dumps.glob("*_ERROR/error.txt"))


def test_nonstream_backend_rejects_streaming_without_running_inference(bridge):
    url, state, session, _ = bridge
    response = session.post(url + "/predict_stream", json=request_item(), timeout=3)
    assert response.status_code == 400
    assert response.headers["Content-Type"] == "application/json"
    assert not state.posts


def test_constant_baseline_uses_same_transforms_and_disables_has_stream(bridge):
    url, state, session, _ = bridge
    state.service.transport.action_time_schedule = "constant_euler"
    state.service.transport.policy_profile = "libero10_native_constant_v1"
    item = request_item()
    response = session.post(url + "/predict", json=item, timeout=5)
    assert response.status_code == 200 and len(response.json()["action"]) == 16
    fields, pixels = state.posts[0]
    extra = json.loads(fields["extra_params"])
    assert extra["action_time_schedule"] == "constant_euler"
    assert extra["policy_profile"] == "libero10_native_constant_v1"
    assert "action_stream" not in extra
    expected, caption, size = prepare_libero_policy_inputs(
        np.full((256, 512, 3), 30, dtype=np.uint8), item["prompt"], 256
    )
    np.testing.assert_array_equal(pixels, expected)
    assert fields["prompt"].decode() == caption and extra["native_frame_size"] == size
    info = session.get(url + "/info", timeout=3).json()
    assert info["action_time_schedule"] == "constant_euler" and not info["streaming"]["supported"]
    response = session.post(url + "/predict_stream", json=item, timeout=3)
    assert response.status_code == 400 and len(state.posts) == 1


@pytest.mark.parametrize("endpoint", ["/predict", "/predict_batch", "/predict_stream"])
def test_shared_http_handler_preserves_native_service_and_stream_callback(endpoint):
    calls = []

    class NativeService:
        cfg = SimpleNamespace(dump_dir=None, http_400_on_error=False)

        def predict_distributed(self, req, *, is_batch, on_actions_ready):
            calls.append((is_batch, on_actions_ready is not None))
            if on_actions_ready is not None:
                on_actions_ready({"type": "partial", "action_indices": [0], "actions": [[1.0]]})
            return {"actions": [[[1.0]]]} if is_batch else {"action": [[1.0]], "video": []}

    server = ActionPolicyHTTPServer(("127.0.0.1", 0), NativeService())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.post(f"http://127.0.0.1:{server.server_port}{endpoint}", json={"items": [{}]}, timeout=3)
        assert response.status_code == 200
        if endpoint.endswith("_stream"):
            events = [json.loads(line) for line in response.text.splitlines()]
            assert [event["type"] for event in events] == ["partial", "final"]
            assert events[0]["request_id"] == events[1]["request_id"] == 1
            assert events[1]["action"] == [[1.0]]
        else:
            assert response.json() == (
                {"actions": [[[1.0]]]} if endpoint.endswith("_batch") else {"action": [[1.0]], "video": []}
            )
        assert calls == [(endpoint.endswith("_batch"), endpoint.endswith("_stream"))]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        session.close()
