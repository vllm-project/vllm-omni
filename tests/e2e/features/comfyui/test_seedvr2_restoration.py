# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
from fractions import Fraction
from io import BytesIO

import aiohttp
import av
import numpy as np
import pytest
from aiohttp import web
from comfy.model_management import InterruptProcessingException
from comfy_api.input import VideoInput
from comfyui_vllm_omni import nodes as omni_nodes
from comfyui_vllm_omni.nodes import VLLMOmniRestoreVideo
from comfyui_vllm_omni.utils import api_client
from comfyui_vllm_omni.utils import format as media_format

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _clip(frames=6, fps=Fraction(30000, 1001), channels=0):
    buffer = BytesIO()
    with av.open(buffer, "w", format="mp4") as container:
        video = container.add_stream("libx264", rate=fps)
        video.width = video.height = 64
        video.pix_fmt = "yuv420p"
        if channels:
            audio = container.add_stream("aac", rate=48000, layout="mono" if channels == 1 else "stereo")
        for index in range(frames):
            image = np.full((64, 64, 3), 20 + index * 30, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            container.mux(video.encode(frame))
        container.mux(video.encode(None))
        if channels:
            samples = round(frames / fps * 48000)
            tone = 0.1 * np.sin(np.arange(samples) * (2 * np.pi * 440 / 48000))
            frame = av.AudioFrame.from_ndarray(
                np.tile(tone.astype(np.float32), (channels, 1)), format="fltp", layout=audio.layout.name
            )
            frame.sample_rate = 48000
            frame.pts = 0
            container.mux(audio.encode(frame))
            container.mux(audio.encode(None))
    return buffer.getvalue()


@pytest.fixture
def decoded_components(monkeypatch):
    # Keep the real PyAV decoder; expose the data handed to ComfyUI's constructor.
    monkeypatch.setattr(media_format.InputImpl, "VideoFromComponents", lambda components: components)


@pytest.fixture
async def service(unused_tcp_port):
    state = {
        "submission": {"id": "restore-1", "status": "queued"},
        "submit_status": 200,
        "poll": [{"status": "in_progress"}, {"status": "completed"}],
        "poll_started": asyncio.Event(),
        "content": _clip(),
        "deleted": False,
        "delete_status": 200,
        "submit_delay": 0,
        "poll_delay": 0,
        "poll_status": 200,
        "content_delay": 0,
        "content_status": 200,
        "delete_delay": 0,
        "fields": [],
    }

    async def submit(request):
        reader = await request.multipart()
        async for field in reader:
            state["fields"].append(
                (field.name, field.filename, field.headers.get("Content-Type"), bytes(await field.read()))
            )
        await asyncio.sleep(state["submit_delay"])
        return web.json_response(state["submission"], status=state["submit_status"])

    async def poll(request):
        state["poll_started"].set()
        await asyncio.sleep(state["poll_delay"])
        records = state["poll"]
        return web.json_response(records.pop(0) if len(records) > 1 else records[0], status=state["poll_status"])

    async def content(request):
        await asyncio.sleep(state["content_delay"])
        if state["content_status"] != 200:
            return web.json_response({"detail": "Video content unavailable"}, status=state["content_status"])
        return web.Response(body=state["content"], content_type="video/mp4")

    async def delete(request):
        state["deleted"] = True
        await asyncio.sleep(state["delete_delay"])
        return web.json_response({"deleted": True}, status=state["delete_status"])

    app = web.Application()
    app.add_routes(
        [
            web.post("/v1/videos", submit),
            web.get("/v1/videos/restore-1", poll),
            web.get("/v1/videos/restore-1/content", content),
            web.delete("/v1/videos/restore-1", delete),
        ]
    )
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", unused_tcp_port).start()
    client = api_client.VLLMOmniClient(
        f"http://127.0.0.1:{unused_tcp_port}/v1", poll_interval=0.001, max_poll_duration=1
    )
    try:
        yield client, state
    finally:
        await runner.cleanup()


async def _restore(client, **overrides):
    params = dict(model="seedvr2-alias", video=VideoInput(b"source-video"), width=224, height=128, seed=7723)
    return await client.restore_video(**(params | overrides))


async def test_restoration_multipart_and_real_decode(service, monkeypatch):
    client, state = service
    formats = []

    def save(self, file, *, format="auto"):
        formats.append(format)
        file.write(b"source-video")

    monkeypatch.setattr(VideoInput, "save_to", save)
    monkeypatch.setattr(VideoInput, "get_frame_rate", lambda self: Fraction(30000, 1001))
    result = await _restore(client)
    fields = {name: value for name, _, _, value in state["fields"]}
    assert fields == {
        "model": b"seedvr2-alias",
        "prompt": b" ",
        "size": b"224x128",
        "num_inference_steps": b"1",
        "guidance_scale": b"1",
        "seed": b"7723",
        "input_references": b"source-video",
    }
    assert state["fields"][-1][1:3] == ("source.mp4", "video/mp4")
    assert formats == ["mp4"]
    with av.open(BytesIO(result._data)) as container:
        assert len(list(container.decode(video=0))) == 6
        assert container.streams.video[0].average_rate == Fraction(30000, 1001)
    assert state["deleted"]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"width": 0}, "positive multiples of 16"),
        ({"height": -16}, "positive multiples of 16"),
        ({"width": 225}, "positive multiples of 16"),
        ({"height": 129}, "positive multiples of 16"),
        ({"video": None}, "requires a VIDEO input"),
        ({"model": "  "}, "Model must not be empty"),
    ],
)
async def test_invalid_inputs_fail_before_upload(service, overrides, message):
    client, state = service
    with pytest.raises(ValueError, match=message):
        await _restore(client, **overrides)
    assert not state["fields"]


@pytest.mark.parametrize("status, exception", [(400, ValueError), (404, ValueError), (503, RuntimeError)])
async def test_service_errors_are_visible(service, status, exception):
    client, state = service
    state.update(submit_status=status, submission={"detail": "SeedVR2 capability unavailable or clip exceeds budget"})
    with pytest.raises(exception, match=f"status {status}.*SeedVR2"):
        await _restore(client)


@pytest.mark.parametrize(
    "record, message",
    [
        ({}, "missing job 'id'"),
        ({"id": "restore-1"}, "missing job 'status'"),
        ({"id": "restore-1", "status": "failed", "error": "clip exceeds configured pixel budget"}, "pixel budget"),
    ],
)
async def test_invalid_or_failed_job_is_reported(service, record, message):
    client, state = service
    state["submission"] = record
    with pytest.raises(RuntimeError, match=message):
        await _restore(client)
    assert state["deleted"] == ("id" in record)


async def test_timeout_deletes_running_job(service):
    client, state = service
    client.max_poll_duration = 0.02
    state["poll"] = [{"status": "in_progress"}]
    with pytest.raises(RuntimeError, match="Timed out waiting for video job restore-1"):
        await _restore(client)
    assert state["deleted"]


async def test_submission_timeout_is_reported(service):
    client, state = service
    client.timeout = aiohttp.ClientTimeout(total=0.02)
    state["submit_delay"] = 0.1
    with pytest.raises(RuntimeError, match="Timed out submitting video job"):
        await _restore(client)
    assert not state["deleted"]  # The server has not returned a job ID.


@pytest.mark.parametrize("phase", ["poll", "content"])
async def test_job_http_error_still_deletes_job(service, phase):
    client, state = service
    state[f"{phase}_status"] = 503
    with pytest.raises(RuntimeError, match="status 503"):
        await _restore(client)
    assert state["deleted"]


@pytest.mark.parametrize("phase", ["poll", "content"])
async def test_slow_poll_or_download_is_bounded(service, phase):
    client, state = service
    client.max_poll_duration = 0.02
    state[f"{phase}_delay"] = 0.1
    with pytest.raises(RuntimeError, match="Timed out waiting for video job"):
        await _restore(client)
    assert state["deleted"]


async def test_unresponsive_cleanup_does_not_block_timeout(service):
    client, state = service
    client.timeout = aiohttp.ClientTimeout(total=0.02)
    client.max_poll_duration = 0.02
    state["poll"] = [{"status": "in_progress"}]
    state["delete_delay"] = 1
    with pytest.raises(RuntimeError, match="Timed out waiting for video job"):
        await asyncio.wait_for(_restore(client), timeout=0.5)
    assert state["deleted"]


async def test_cancellation_deletes_running_job(service):
    client, state = service
    state["submit_delay"] = 0.1
    state["poll"] = [{"status": "in_progress"}]
    task = asyncio.create_task(_restore(client))
    await asyncio.wait_for(state["poll_started"].wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert state["deleted"]


@pytest.fixture
async def waiting_restoration(monkeypatch):
    """Keep the real node/client; substitute the HTTP boundary of a running job."""
    polled = asyncio.Event()
    delete_started = asyncio.Event()
    delete_release = asyncio.Event()
    delete_release.set()
    state = {"deleted": False}
    pending_response = asyncio.Event()

    async def request(session, url, verb="get", **kwargs):
        if verb == "post":
            return {"id": "restore-1", "status": "queued"}
        if verb == "delete":
            delete_started.set()
            await delete_release.wait()
            state["deleted"] = True
            return {"deleted": True}
        polled.set()
        await pending_response.wait()

    client = api_client.VLLMOmniClient("http://service/v1", timeout=1, poll_interval=0, max_poll_duration=5)
    monkeypatch.setattr(api_client, "url_json", request)
    monkeypatch.setattr(omni_nodes, "VLLMOmniClient", lambda *args, **kwargs: client)
    task = asyncio.create_task(
        VLLMOmniRestoreVideo().restore(VideoInput(), "http://service/v1", "seedvr2", 224, 128, 7723, 1)
    )
    try:
        await asyncio.wait_for(polled.wait(), timeout=1)
        yield task, state, delete_started, delete_release
    finally:
        delete_release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("wait_for_cleanup", [False, True])
async def test_comfyui_stop_cancels_restoration_and_deletes_job(waiting_restoration, monkeypatch, wait_for_cleanup):
    task, state, delete_started, delete_release = waiting_restoration
    if wait_for_cleanup:
        delete_release.clear()

    def interrupt():
        raise InterruptProcessingException()

    monkeypatch.setattr(omni_nodes, "processing_interrupted", lambda: True)
    monkeypatch.setattr(omni_nodes, "throw_exception_if_processing_interrupted", interrupt)
    if wait_for_cleanup:
        await asyncio.wait_for(delete_started.wait(), timeout=1)
        assert not task.done()
        delete_release.set()
    with pytest.raises(InterruptProcessingException):
        await asyncio.wait_for(task, timeout=1)
    assert state["deleted"]


async def test_parent_cancellation_waits_for_job_cleanup(waiting_restoration):
    task, state, _, _ = waiting_restoration
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)
    assert state["deleted"]


async def test_corrupt_response_still_deletes_job(service):
    client, state = service
    state["content"] = b"not an MP4"
    with pytest.raises(RuntimeError, match="Failed to remux restored video"):
        await _restore(client)
    assert state["deleted"]


async def test_delete_failure_does_not_discard_restored_video(service):
    client, state = service
    state["delete_status"] = 503
    result = await _restore(client)
    with av.open(BytesIO(result._data)) as container:
        assert len(list(container.decode(video=0))) == 6


async def test_unreachable_service():
    client = api_client.VLLMOmniClient("http://127.0.0.1:1/v1", timeout=1)
    with pytest.raises(RuntimeError, match="Network error connecting"):
        await _restore(client)


@pytest.mark.parametrize(
    "frames, fps, channels", [(5, Fraction(24), 0), (6, Fraction(30000, 1001), 1), (6, Fraction(24), 2)]
)
def test_decode_keeps_frame_count_fps_and_audio(decoded_components, frames, fps, channels):
    result = media_format.bytes_to_video(_clip(frames, fps, channels))
    assert result["images"].shape == (frames, 64, 64, 3)
    assert result["frame_rate"] == fps
    if channels:
        audio = result["audio"]
        assert audio["sample_rate"] == 48000
        assert audio["waveform"].shape[1] == channels
        assert audio["waveform"].abs().max() > 0.01
    else:
        assert result["audio"] is None


async def test_node_uses_configured_timeout_and_normalizes_url(monkeypatch):
    calls: dict[str, object] = {}

    async def restore(self, **kwargs):
        calls.update(url=self.base_url, timeout=self.timeout.total, poll_timeout=self.max_poll_duration, **kwargs)
        return "restored"

    monkeypatch.setattr(api_client.VLLMOmniClient, "restore_video", restore)
    node = VLLMOmniRestoreVideo()
    assert await node.restore(VideoInput(), " http://localhost:8098/v1/ ", " seedvr2 ", 224, 128, 7723, 600) == (
        "restored",
    )
    assert calls["url"] == "http://localhost:8098/v1"
    assert calls["model"] == "seedvr2"
    assert calls["timeout"] == calls["poll_timeout"] == 600
    with pytest.raises(ValueError, match="timeout must be positive"):
        await node.restore(VideoInput(), "http://localhost/v1", "seedvr2", 224, 128, 7723, 0)


@pytest.mark.parametrize("error", [ValueError("invalid dimensions"), RuntimeError("Timed out waiting for video job")])
async def test_node_preserves_client_errors(monkeypatch, error):
    async def restore(self, **kwargs):
        raise error

    monkeypatch.setattr(api_client.VLLMOmniClient, "restore_video", restore)
    with pytest.raises(type(error)) as raised:
        await VLLMOmniRestoreVideo().restore(VideoInput(), "http://service/v1", "seedvr2", 224, 128, 7723, 1)
    assert raised.value is error


@pytest.mark.parametrize("channels", [0, 1, 2])
def test_remux_preserves_encoded_streams_and_exact_source_fps(channels):
    payload = _clip(fps=Fraction(2997, 100), channels=channels)
    result = media_format.bytes_to_restored_video(payload, Fraction(30000, 1001))

    def packets(data):
        with av.open(BytesIO(data)) as container:
            video = container.streams.video[0]
            encoded = []
            audio_timing = []
            for packet in container.demux():
                if packet.dts is None:
                    continue
                encoded.append((packet.stream.type, bytes(packet)))
                if packet.stream.type == "audio":
                    audio_timing.append((packet.pts * packet.time_base, packet.duration * packet.time_base))
            return video.average_rate, encoded, audio_timing

    old_rate, old_packets, old_audio_timing = packets(payload)
    rate, new_packets, audio_timing = packets(result._data)
    assert old_rate == Fraction(2997, 100)
    assert rate == Fraction(30000, 1001)
    assert new_packets == old_packets
    assert audio_timing == old_audio_timing
