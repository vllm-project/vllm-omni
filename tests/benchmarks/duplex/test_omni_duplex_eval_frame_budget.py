# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import io
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pybase64 as base64
import pytest
import requests
from PIL import Image

from vllm_omni.benchmarks.duplex import omni_duplex_eval_eval as eval_mod
from vllm_omni.benchmarks.duplex.omni_duplex_eval_dataset import DuplexSample
from vllm_omni.benchmarks.duplex.omni_duplex_eval_judge import DuplexJudge

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]

OVERFLOW = "The decoder prompt (length 93016) is longer than the maximum model length of 65536"


def _http_error(status: int, body: str) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    response._content = body.encode()
    return requests.HTTPError(f"{status} response from judge", response=response)


@pytest.mark.parametrize("status", [401, 403, 413, 429, 502, 503])
def test_classifier_rejects_unrelated_http_status(status: int) -> None:
    assert not eval_mod._is_prompt_too_long_error(_http_error(status, OVERFLOW))


@pytest.mark.parametrize("status", [400, 500])
def test_classifier_accepts_reported_prompt_overflow(status: int) -> None:
    assert eval_mod._is_prompt_too_long_error(_http_error(status, OVERFLOW))


@pytest.mark.parametrize("status", [400, 500])
def test_classifier_accepts_explicit_context_length_code(status: int) -> None:
    body = json.dumps({"error": {"code": "context_length_exceeded", "message": "Context is full"}})
    assert eval_mod._is_prompt_too_long_error(_http_error(status, body))


def test_classifier_does_not_match_unrelated_json_fields() -> None:
    body = json.dumps({"error": {"message": "Model worker failed"}, "debug": OVERFLOW})
    assert not eval_mod._is_prompt_too_long_error(_http_error(500, body))


def test_classifier_requires_an_http_response() -> None:
    assert not eval_mod._is_prompt_too_long_error(requests.HTTPError(OVERFLOW))


@pytest.mark.parametrize("count", [0, 1, 16])
@pytest.mark.parametrize("allow_reduction", [False, True])
def test_success_preserves_input_and_does_not_retry(count: int, allow_reduction: bool) -> None:
    frames = [bytes([i]) for i in range(count)]
    calls = []

    def call(selected: list[bytes]) -> str:
        calls.append(list(selected))
        return "score"

    result, selected, retries = eval_mod._judge_frames_with_overflow_retry(
        call, frames, allow_reduction=allow_reduction
    )
    assert (result, selected, retries) == ("score", frames, 0)
    assert calls == [frames]
    assert frames == [bytes([i]) for i in range(count)]


def test_subsampling_uses_original_timeline_and_preserves_endpoints() -> None:
    frames = [bytes([i]) for i in range(16)]
    calls = []

    def call(selected: list[bytes]) -> str:
        calls.append(list(selected))
        if len(selected) > 4:
            raise _http_error(500, OVERFLOW)
        return "score"

    result, selected, retries = eval_mod._judge_frames_with_overflow_retry(call, frames, allow_reduction=True)
    assert result == "score"
    assert [len(batch) for batch in calls] == [16, 8, 4]
    assert selected == [bytes([i]) for i in [0, 5, 10, 15]]
    assert retries == 2
    assert all(batch[0] == frames[0] and batch[-1] == frames[-1] for batch in calls)
    assert len(frames) == 16


@pytest.mark.parametrize("count", [0, 1, 2, 3, 16, 17])
def test_exhausted_retry_propagates_without_empty_fallback(count: int) -> None:
    frames = [bytes([i]) for i in range(count)]
    calls = []
    error = _http_error(500, OVERFLOW)

    def call(selected: list[bytes]) -> str:
        calls.append(list(selected))
        raise error

    with pytest.raises(requests.HTTPError) as caught:
        eval_mod._judge_frames_with_overflow_retry(call, frames, allow_reduction=True)
    assert caught.value is error
    expected_counts = [count]
    while expected_counts[-1] > 1:
        expected_counts.append((expected_counts[-1] + 1) // 2)
    assert [len(batch) for batch in calls] == expected_counts
    if count:
        assert calls[-1] == [frames[count // 2]]


@pytest.mark.parametrize("exception", [requests.ReadTimeout("timeout"), requests.ConnectionError("disconnected")])
def test_transport_failures_are_not_retried(exception: Exception) -> None:
    calls = []

    def call(selected: list[bytes]) -> str:
        calls.append(selected)
        raise exception

    with pytest.raises(type(exception)) as caught:
        eval_mod._judge_frames_with_overflow_retry(call, [b"a", b"b"], allow_reduction=True)
    assert caught.value is exception
    assert len(calls) == 1


@pytest.mark.parametrize("status", [401, 429, 500])
def test_unrelated_http_errors_are_not_retried(status: int) -> None:
    calls = []
    error = _http_error(status, "Model worker failed")

    def call(selected: list[bytes]) -> str:
        calls.append(selected)
        raise error

    with pytest.raises(requests.HTTPError) as caught:
        eval_mod._judge_frames_with_overflow_retry(call, [b"a", b"b"], allow_reduction=True)
    assert caught.value is error
    assert len(calls) == 1


@contextmanager
def _http_judge(limit: int = 4) -> Iterator[tuple[DuplexJudge, list[int]]]:
    """Real HTTP and production judge client; deterministic fixture, not a model."""
    counts: list[int] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.connection.settimeout(5)
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert self.path == "/v1/chat/completions"
            assert payload["model"] == "fixture-judge"
            assert payload["max_tokens"] == 1200
            parts = payload["messages"][-1]["content"]
            images = [part for part in parts if part["type"] == "image_url"]
            for part in images:
                data = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
                with Image.open(io.BytesIO(data)) as image:
                    assert image.size == (3734, 2100)
            counts.append(len(images))
            response: dict[str, object]
            if len(images) > limit:
                status = 500
                response = {"error": {"message": OVERFLOW}}
            else:
                status = 200
                score = json.dumps({"content_score": 4, "temporal_score": 4, "is_relevant": 1})
                response = {"choices": [{"message": {"content": score}}]}
            encoded = json.dumps(response).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args) -> None:
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        try:
            yield DuplexJudge(f"http://127.0.0.1:{server.server_port}", "fixture-judge", timeout=5), counts
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


def _sample_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, with_sentence: bool = True):
    response_path = tmp_path / "responses" / "sample.json"
    response_path.parent.mkdir()
    items = [{"sentence": "A person moves.", "start": 2.0, "end": 4.0}] if with_sentence else []
    response_path.write_text(json.dumps(items), encoding="utf-8")
    response_path.with_name("sample.meta.json").write_text('{"clock": "media"}', encoding="utf-8")
    score_path = tmp_path / "scores" / "RTD_OCR" / "sample.json"
    sample = DuplexSample(
        id="sample",
        split="RTD_OCR",
        family="rtd",
        task_type=None,
        video=tmp_path / "video.mp4",
        video_duration=48.0,
        question_text="What happened?",
        answer1="A person moves.",
    )
    with io.BytesIO() as buffer:
        Image.new("RGB", (3734, 2100), "white").save(buffer, format="JPEG")
        frame = buffer.getvalue()
    monkeypatch.setattr(eval_mod, "materialize_media", lambda *args, **kwargs: sample.video)
    monkeypatch.setattr(eval_mod, "_extract_frames", lambda *args, **kwargs: [frame] * 16)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    return sample, response_path, score_path


def test_real_http_temporal_and_content_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sample, response_path, score_path = _sample_inputs(tmp_path, monkeypatch)
    with _http_judge() as (judge, counts):
        result = eval_mod.evaluate_sample(
            sample, response_path, score_path, judge, judge_video_mode="frame-sample", judge_frame_overflow="reduce"
        )
    assert counts == [16, 8, 4, 16, 8, 4]
    assert result["judge_frame_overflow"] == "reduce"
    for row in [result["temporal"]["sentences"][0], result["content"]]:
        assert row["frame_count_initial"] == 16
        assert row["frame_count"] == 4
        assert row["frame_budget_retries"] == 2
    assert json.loads(score_path.read_text(encoding="utf-8")) == result
    summary = eval_mod.summarize_scores(tmp_path / "scores")
    assert summary["rtd"]["frame_reduction_samples"] == 1
    assert summary["rtd"]["judge_frame_overflow_policies"] == ["reduce"]


def test_default_policy_fails_without_changing_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sample, response_path, score_path = _sample_inputs(tmp_path, monkeypatch)
    with _http_judge() as (judge, counts), pytest.raises(requests.HTTPError):
        eval_mod.evaluate_sample(sample, response_path, score_path, judge, judge_video_mode="frame-sample")
    assert counts == [16]
    assert not score_path.exists()


def test_video_url_content_is_not_changed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sample, response_path, score_path = _sample_inputs(tmp_path, monkeypatch, with_sentence=False)
    with _http_judge() as (judge, counts):
        result = eval_mod.evaluate_sample(sample, response_path, score_path, judge, judge_frame_overflow="reduce")
    assert counts == [0]
    assert "frame_count" not in result["content"]


def test_success_keeps_frames_and_reports_zero_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sample, response_path, score_path = _sample_inputs(tmp_path, monkeypatch)
    with _http_judge(limit=16) as (judge, counts):
        result = eval_mod.evaluate_sample(
            sample, response_path, score_path, judge, judge_video_mode="frame-sample", judge_frame_overflow="reduce"
        )
    assert counts == [16, 16]
    assert result["content"]["frame_count"] == 16
    assert result["content"]["frame_budget_retries"] == 0
    assert eval_mod.summarize_scores(tmp_path / "scores")["rtd"]["frame_reduction_samples"] == 0
