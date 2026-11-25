# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import io
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import av
import numpy as np
import pytest
import torch
from fastapi import FastAPI
from starlette.testclient import TestClient

from vllm_omni.diffusion.models.invoker_render.interaction import ProxyQueue
from vllm_omni.diffusion.models.invoker_render.pipeline import InvokerRenderPipeline
from vllm_omni.diffusion.models.invoker_render.proxy import (
    BOOTSTRAP_KEY,
    PROXY_MODALITY,
    invoker_render_proxy_spec,
)
from vllm_omni.diffusion.models.invoker_render.schedule import Window
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.entrypoints.openai.serving_video_output_stream import OmniStreamingVideoOutputHandler
from vllm_omni.entrypoints.openai.video.proxy_ingress import ProxyIngress
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def header(seq: int, *, eos: bool = False) -> dict:
    start = seq * 17
    count = 5 if eos else 17
    return {
        "type": "session.proxy.append",
        "input_seq": seq,
        "frame_start": start,
        "num_frames": count,
        "eos": eos,
        "capture_ms": [index * 1000 / 24 + (index % 3) * 4 for index in range(start, start + count)],
    }


def test_credit_wait_and_unknown_length():
    proxy = ProxyIngress(invoker_render_proxy_spec((32, 32)))
    rgb = bytes(17 * 32 * 32 * 3)
    assert proxy.receive(header(0), rgb) is None
    assert not proxy.bootstrap_ready.is_set()
    with pytest.raises(ValueError, match="out of sequence"):
        proxy.receive(header(2), rgb)
    assert proxy.receive(header(1), rgb) is None
    assert proxy.bootstrap_ready.is_set()
    assert proxy.receive(header(2), rgb).input_seq == 2
    with pytest.raises(ValueError, match="queue is full"):
        proxy.receive(header(3), rgb)
    assert proxy.sequence.consume(34)
    assert proxy.receive(header(3, eos=True), bytes(5 * 32 * 32 * 3)).eos
    with pytest.raises(ValueError, match="follows EOS"):
        proxy.receive(header(4), rgb)
    state = StepRequestState(request_id="live", sampling=OmniDiffusionSamplingParams(), total_chunks=None)
    state.chunk_num_steps, state.chunk_index = 4, 500
    assert not state.request_denoise_completed
    state.stream_input_drained = True
    assert state.request_denoise_completed


@dataclass
class _FakeCausal:
    prepare: Callable[[object, Window], torch.Tensor]


@dataclass
class _FakeSession:
    windows: deque[Window]
    causal: _FakeCausal
    active: Window | None = None


@dataclass
class _FakeEngine:
    abort: Callable[..., object]
    submit_interaction_async: Callable[..., object] | None = field(default=None)


def test_resume_starts_each_window_at_first_timestep(monkeypatch):
    pipeline = object.__new__(InvokerRenderPipeline)
    torch.nn.Module.__init__(pipeline)
    cache = object()
    window = Window(9, 14, False)
    latents = torch.ones(1, 5, 2, 2)

    def prepare(active_cache, active_window):
        assert torch.is_inference_mode_enabled()
        assert active_cache is cache and active_window is window
        return latents

    monkeypatch.setattr(pipeline, "_cache", lambda: cache)
    monkeypatch.setattr(pipeline, "apply_interaction_at_chunk_boundary", lambda state: None)
    state = StepRequestState(
        request_id="live",
        sampling=OmniDiffusionSamplingParams(),
        timesteps=torch.arange(4),
        step_index=4,
        step_in_chunk=4,
        chunk_num_steps=4,
    )
    state.extra[PROXY_MODALITY] = ProxyQueue((32, 32))
    state.extra["invoker_session"] = _FakeSession(windows=deque([window]), causal=_FakeCausal(prepare=prepare))

    pipeline.ar_diffusion_resume_step(state, "interaction")

    assert state.current_timestep.item() == 0
    assert state.step_index == state.step_in_chunk == 0
    assert state.latents is latents


def test_websocket_uses_bootstrap_and_muxes_eos(monkeypatch):
    tail_received = asyncio.Event()

    async def submit(request_id, *, interaction, on_error):
        assert interaction["event_id"] == "proxy-2"
        assert interaction["event"]["multi_modal_data"][PROXY_MODALITY]["eos"]
        tail_received.set()

    async def abort(request_id, *, timeout):
        return None

    handler = OmniStreamingVideoOutputHandler(
        _FakeEngine(submit_interaction_async=submit, abort=abort),
        proxy_spec=invoker_render_proxy_spec((32, 32)),
        stall_timeout=4,
    )

    async def generate(prompt, gen_params, request_id, progress):
        assert [item["input_seq"] for item in prompt["multi_modal_data"][BOOTSTRAP_KEY]] == [0, 1]
        assert gen_params.fps == 24
        for chunk, (start, count, consumed) in enumerate(((0, 26, 34), (26, 0, 39), (26, 13, 39))):
            if chunk == 1:
                await asyncio.wait_for(tail_received.wait(), 2)
            yield OmniRequestOutput.from_diffusion(
                request_id=request_id,
                images=[np.full((count, 32, 32, 3), 30 + chunk * 80, dtype=np.uint8)],
                multimodal_output={
                    "metadata": {
                        "stream": {
                            "generation_chunk_index": chunk,
                            "source_frame_start": start,
                            "source_frame_count": count,
                            "source_frames_consumed": consumed,
                        }
                    }
                },
                finished=chunk == 2,
            )

    monkeypatch.setattr(handler, "_iter_generation_outputs", generate)
    app = FastAPI()
    app.add_api_websocket_route("/v1/realtime/video", handler.handle_session)
    with TestClient(app) as client, client.websocket_connect("/v1/realtime/video") as ws:
        ws.send_json({"type": "session.start", "prompt": "road", "seed": 42})
        assert ws.receive_json()["proxy"]["credit_frames"] == 56
        for seq in range(3):
            packet = header(seq, eos=seq == 2)
            ws.send_json(packet)
            ws.send_bytes(bytes(packet["num_frames"] * 32 * 32 * 3))
        output = bytearray()
        source = []
        last_credit = None
        while True:
            message = ws.receive_json()
            assert message["type"] != "error", message
            if message["type"] == "session.done":
                assert not message["stopped"]
                break
            if message["type"] == "session.proxy.credit":
                last_credit = message
            if message["type"] == "video.chunk_metadata":
                output.extend(ws.receive_bytes())
                if message["kind"] == "media":
                    source.append((message["source_frame_start"], message["source_frame_count"]))
        assert source == [(0, 26), (26, 0), (26, 13)]
        assert last_credit is not None
        assert last_credit["consumed_frames"] == 39
    with av.open(io.BytesIO(output)) as clip:
        assert len(list(clip.decode(video=0))) == 39


def test_websocket_can_stop_during_bootstrap():
    aborted = []

    async def abort(request_id, *, timeout):
        aborted.append(request_id)

    handler = OmniStreamingVideoOutputHandler(
        _FakeEngine(abort=abort),
        proxy_spec=invoker_render_proxy_spec((32, 32)),
        stall_timeout=4,
    )
    app = FastAPI()
    app.add_api_websocket_route("/v1/realtime/video", handler.handle_session)
    with TestClient(app) as client, client.websocket_connect("/v1/realtime/video") as ws:
        ws.send_json({"type": "session.start", "prompt": "road"})
        started = ws.receive_json()
        ws.send_json({"type": "session.stop"})
        stopped = ws.receive_json()
        assert stopped["type"] == "session.done" and stopped["stopped"]
        assert stopped["request_id"] == started["request_id"]
    assert aborted and set(aborted) == {started["request_id"]}
