# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Failed judge inputs survive export without changing requests or scores."""

import hashlib
import io
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests
from PIL import Image

from tests.dfx import conftest as dfx
from vllm_omni.benchmarks.duplex import omni_duplex_eval_eval as evaluator
from vllm_omni.benchmarks.duplex import omni_duplex_eval_judge as judge_module
from vllm_omni.benchmarks.duplex import omni_duplex_eval_judge_capture as capture
from vllm_omni.benchmarks.duplex.omni_duplex_eval_dataset import DuplexSample

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _reply(text: str, finish_reason: str = "stop", status: int = 200) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(
        {"choices": [{"message": {"content": text}, "finish_reason": finish_reason}]}
    ).encode()
    return response


@pytest.fixture
def temporal_call(tmp_path, monkeypatch):
    response = tmp_path / "response.json"
    response.write_text('[{"sentence": "The garlic is cut.", "start": 2, "end": 3}]')
    response.with_name("response.meta.json").write_text(
        json.dumps({"clock": "media", "pace": "realtime", "mix": "soundtrack"})
    )
    sample = DuplexSample.from_row(
        {
            "id": "362",
            "split": "RTD_world_knowledge",
            "question_text": "Describe the action.",
            "video": str(tmp_path / "video.mp4"),
            "video_duration": 10,
        }
    )
    encoded = io.BytesIO()
    Image.new("RGB", (2, 2), (10, 20, 30)).save(encoded, format="JPEG")
    frames = [encoded.getvalue(), encoded.getvalue()]
    monkeypatch.setattr(evaluator, "_extract_frames", lambda *args: frames)
    post = Mock(return_value=_reply("C picks up the garlic"))
    monkeypatch.setattr(judge_module.requests, "post", post)
    score = tmp_path / "run/evaluation/RTD_world_knowledge/362.json"
    judge = judge_module.DuplexJudge("http://endpoint-not-for-export.invalid", "judge", api_key="auth-not-for-export")

    def run():
        return evaluator.evaluate_sample(sample, response, score, judge)

    return run, score, post


def _manifests(score: Path) -> list[Path]:
    return sorted((score.parent / "judge_requests").glob("request-*.judge.jsonl"))


def _restore(path: Path) -> dict:
    record = json.loads(path.read_text())
    assert record["replayable"]
    body = record["request_body"]
    for ref in record["frame_refs"]:
        assert Path(ref["file"]).name == ref["file"]
        data_url = json.loads((path.parent / ref["file"]).read_text())["data_url"]
        assert hashlib.sha256(data_url.encode()).hexdigest() == ref["data_url_sha256"]
        parent = body
        for key in ref["path"][:-1]:
            parent = parent[key]
        parent[ref["path"][-1]] = data_url
    canonical = (json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    assert hashlib.sha256(canonical).hexdigest() == record["request_body_sha256"]
    return body


def test_failed_temporal_request_is_exact_deduplicated_and_secret_safe(temporal_call):
    run, score, post = temporal_call
    with pytest.raises(ValueError, match="Judge response contains no score fields"):
        run()
    assert not score.exists()
    trace = score.with_suffix(".judge.jsonl")
    assert trace.read_text() == '{"raw_response": "C picks up the garlic"}\n'
    manifests = _manifests(score)
    assert len(manifests) == 1
    body = _restore(manifests[0])
    assert body == post.call_args.kwargs["json"]
    assert set(body) == {"model", "messages", "temperature", "max_tokens"}
    assert body["temperature"] == 0.1 and body["max_tokens"] == 1200
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer auth-not-for-export"
    assert len(list(manifests[0].parent.glob("frame-*.judge.jsonl"))) == 1
    assert len(json.loads(manifests[0].read_text())["frame_refs"]) == 2
    for artifact in manifests[0].parent.iterdir():
        data = artifact.read_text()
        assert "auth-not-for-export" not in data
        assert "endpoint-not-for-export" not in data
        assert "Authorization" not in data and "api_key" not in data and "headers" not in data
    assert evaluator.summarize_scores(score.parent.parent)["samples"] == 0


def test_failed_request_survives_actual_dfx_export_and_temporary_cleanup(temporal_call, tmp_path):
    run, score, post = temporal_call
    with pytest.raises(ValueError, match="no score fields"):
        run()
    score_root = score.parent.parent
    media = score_root / ".media/362.mp4"
    media.parent.mkdir()
    media.write_bytes(b"temporary video must not be copied")
    published = tmp_path / "published/result_acc_probe.json"
    for _ in range(2):
        dfx._save_duplex_eval_diagnostics(str(score_root), published, response_root=str(score_root.parent))
    exported = published.parent / "diagnostics/result_acc_probe-run"
    restored = next(exported.rglob("request-*.judge.jsonl"))
    shutil.rmtree(score_root)
    assert _restore(restored) == post.call_args.kwargs["json"]
    assert not list(exported.rglob("*.mp4"))
    assert not list(exported.rglob("*.json"))
    assert (exported / "RTD_world_knowledge/362.judge.jsonl").read_text() == (
        '{"raw_response": "C picks up the garlic"}\n'
    )


def test_success_keeps_original_score_and_writes_no_request(temporal_call):
    run, score, post = temporal_call
    post.side_effect = [_reply('{"temporal_score": 2, "is_relevant": 1}'), _reply('{"content_score": 2.5}')]
    result = run()
    assert result["temporal"]["summary"]["avg_temporal_score"] == 2
    assert result["content"]["content_score"] == 2.5
    assert score.is_file() and not (score.parent / "judge_requests").exists()
    assert evaluator.summarize_scores(score.parent.parent)["samples"] == 1


@pytest.mark.parametrize("failure", ["timeout", "http", "incomplete"])
def test_request_retention_preserves_original_judge_exception(temporal_call, failure):
    run, score, post = temporal_call
    expected: type[Exception]
    if failure == "timeout":
        error = requests.Timeout("fixture timeout")
        post.side_effect = error
        expected = requests.Timeout
    elif failure == "http":
        post.return_value = _reply("Unavailable", status=503)
        expected = requests.HTTPError
    else:
        post.return_value = _reply('{"temporal_score": 3, "is_relevant": 1}', finish_reason="length")
        expected = judge_module.JudgeResponseError
    with pytest.raises(expected) as caught:
        run()
    if failure == "timeout":
        assert caught.value is error
    elif failure == "incomplete":
        row = json.loads(score.with_suffix(".judge.jsonl").read_text())
        assert row["finish_reason"] == "length" and row["raw_response"] == caught.value.raw_response
    assert not score.exists()
    assert _restore(_manifests(score)[0]) == post.call_args.kwargs["json"]
    assert evaluator.summarize_scores(score.parent.parent)["samples"] == 0


@pytest.mark.parametrize("limit,status", [("_MAX_REQUEST_BYTES", "request_byte_limit"), ("_MAX_FRAMES", "frame_limit")])
def test_capture_limit_is_explicit_and_preserves_original_failure(temporal_call, monkeypatch, limit, status):
    run, score, _ = temporal_call
    monkeypatch.setattr(capture, limit, 1)
    with pytest.raises(ValueError, match="no score fields"):
        run()
    row = json.loads(_manifests(score)[0].read_text())
    assert row["capture_status"] == status and not row["replayable"]
    assert "request_body" not in row
    assert not list(_manifests(score)[0].parent.glob("frame-*.judge.jsonl"))


def test_storage_limit_and_write_failure_do_not_replace_parse_error(temporal_call, monkeypatch, caplog):
    run, score, _ = temporal_call
    monkeypatch.setattr(capture, "_MAX_CAPTURE_BYTES", 1)
    with pytest.raises(ValueError, match="no score fields"):
        run()
    assert not _manifests(score)
    monkeypatch.setattr(capture, "_MAX_CAPTURE_BYTES", 16 * 1024 * 1024)

    def fail_write(*args):
        raise OSError("auth-not-for-export")

    monkeypatch.setattr(capture, "_atomic_write", fail_write)
    with pytest.raises(ValueError, match="no score fields"):
        run()
    assert "OSError" in caplog.text and "auth-not-for-export" not in caplog.text


def test_shared_judge_keeps_thread_requests_separate(tmp_path, monkeypatch):
    barrier = threading.Barrier(2)

    def post(*args, **kwargs):
        barrier.wait(timeout=5)
        return _reply("bare text")

    monkeypatch.setattr(judge_module.requests, "post", post)
    judge = judge_module.DuplexJudge("http://unused", "judge")

    def worker(name):
        trace = tmp_path / "split" / f"{name}.judge.jsonl"
        with capture.capture_judge_request() as captured:
            judge.temporal(name, [b"frame"])
            capture.save_failed_judge_request(captured, trace, 0)
        record = next(path for path in _manifests(trace) if f"request-{name}.judge-" in path.name)
        assert _restore(record)["messages"][1]["content"][0]["text"] == name

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(worker, ["first", "second"]))
    assert len(list((tmp_path / "split/judge_requests").glob("frame-*.judge.jsonl"))) == 1
