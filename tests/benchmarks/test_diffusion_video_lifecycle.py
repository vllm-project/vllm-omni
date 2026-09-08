# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest

from benchmarks.diffusion import backends

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]


class Response:
    def __init__(self, payload=None, status=200):
        self.payload = payload or {}
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def json(self):
        return self.payload

    async def text(self):
        return str(self.payload)

    async def read(self):
        return b"video"


class VideoSession:
    def __init__(self, clock, *, status="completed", elapsed=602, failure=None):
        self.clock = clock
        self.status = status
        self.elapsed = elapsed
        self.failure = failure
        self.deleted = []
        self.content_reads = 0

    def post(self, *args, **kwargs):
        if self.failure == "exception":
            raise TimeoutError()
        if self.failure == "post":
            return Response(status=500)
        if self.failure == "missing_id":
            return Response({"status": "queued"})
        return Response({"id": "video-1", "status": "queued"})

    def get(self, url):
        if url.endswith("/content"):
            self.content_reads += 1
            return Response(status=500 if self.failure == "content" else 200)
        self.clock.now += self.elapsed
        return Response({"status": self.status}, status=500 if self.failure == "poll" else 200)

    def delete(self, url):
        self.deleted.append(url)
        return Response()


@pytest.fixture
def video_env(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(backends, "time", SimpleNamespace(perf_counter=lambda: clock.now))

    async def sleep(_):
        clock.now += 2

    monkeypatch.setattr(backends, "asyncio", SimpleNamespace(sleep=sleep))
    return clock


def request():
    return backends.RequestFuncInput(prompt="a cat", api_url="http://test/v1/videos", model="test")


@pytest.mark.asyncio
async def test_completed_video_at_deadline_is_not_discarded(video_env, mocker):
    session = VideoSession(video_env)
    progress = mocker.Mock()
    output = await backends.async_request_v1_videos(request(), session, progress)
    assert output.success
    assert output.response_body == b"video"
    assert output.latency == 604
    assert session.content_reads == 1
    assert session.deleted == ["http://test/v1/videos/video-1"]
    progress.update.assert_called_once_with(1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["post", "missing_id", "poll", "content", "exception", "failed", "timeout", "references"]
)
async def test_failed_video_updates_progress(video_env, mocker, failure):
    status = {"failed": "failed", "timeout": "in_progress"}.get(failure, "completed")
    session = VideoSession(video_env, status=status, elapsed=602 if failure == "timeout" else 1, failure=failure)
    req = request()
    if failure == "references":
        req.image_paths = ["image.png"]
        req.video_paths = ["video.mp4"]
    progress = mocker.Mock()
    output = await backends.async_request_v1_videos(req, session, progress)
    assert not output.success
    assert output.error
    assert output.latency == video_env.now
    progress.update.assert_called_once_with(1)
    if failure not in {"post", "missing_id", "exception", "references"}:
        assert session.deleted == ["http://test/v1/videos/video-1"]


@pytest.mark.asyncio
async def test_video_timeout_can_exceed_ten_minutes(video_env, monkeypatch):
    session = VideoSession(video_env, status="in_progress", elapsed=602)
    original_get = session.get

    def get(url):
        if video_env.now > 600:
            session.status = "completed"
        return original_get(url)

    monkeypatch.setattr(session, "get", get)
    req = request()
    req.video_job_timeout = 1800
    output = await backends.async_request_v1_videos(req, session)
    assert output.success
    assert output.latency > 600
    assert session.content_reads == 1


@pytest.mark.asyncio
async def test_warmup_preserves_video_job_timeout(monkeypatch):
    import importlib
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks" / "diffusion"))
    benchmark = importlib.import_module("diffusion_benchmark_serving")
    req = request()
    req.video_job_timeout = 1800
    req.num_frames = 33
    req.num_inference_steps = 8
    args = SimpleNamespace(task="t2v", warmup_requests=1, warmup_concurrency=1, warmup_num_inference_steps=2)
    received = []
    session = object()

    async def send(warm_req, actual_session, pbar):
        assert actual_session is session
        assert pbar is None
        received.append(warm_req)
        return backends.RequestFuncOutput(success=True)

    pairs = await benchmark._run_warmups([req], args, session, send)
    assert len(received) == len(pairs) == 1
    assert received[0].video_job_timeout == 1800
    assert received[0].num_frames == 1
    assert received[0].num_inference_steps == 2
    assert req.num_frames == 33
    assert req.num_inference_steps == 8
    assert pairs[0][1].success
