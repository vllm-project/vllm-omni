# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import json
import time

import pytest
import websockets
from PIL import Image

from vllm_omni.benchmarks.data_modules.long_video_dataset import load_video_questions
from vllm_omni.benchmarks.duplex.long_video_runner import PreparedVideo, prepare_video, replay_video
from vllm_omni.benchmarks.duplex.long_video_serving import (
    LongVideoSampleRequest,
    extract_choice,
    finalize_long_video,
)
from vllm_omni.benchmarks.patch.patch import MixRequestFuncOutput, get_samples
from vllm_omni.entrypoints.cli.benchmark.cli_args import preprocess_serve_args
from vllm_omni.entrypoints.cli.benchmark.serve import OmniBenchmarkServingSubcommand
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]


def _annotation(tmp_path):
    (tmp_path / "video.mp4").touch()
    row = {
        "id": 37,
        "task": "EPM",
        "video": "video.mp4",
        "realtime": 315,
        "question": "Where is the object?",
        "options": ["box", "table", "carpet", "floor"],
        "gt": 2,
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps([row]))
    return path, row


def test_cli_routes_real_annotations(tmp_path):
    annotation, _ = _annotation(tmp_path)
    parser = TrackingArgumentParser()
    OmniBenchmarkServingSubcommand.add_cli_args(parser)
    args = parser.parse_args(
        [
            "--model",
            "test",
            "--backend",
            "openai-realtime-duplex",
            "--endpoint",
            "/v1/realtime",
            "--dataset-name",
            "ovo-bench",
            "--dataset-path",
            str(annotation),
            "--long-video-media-root",
            str(tmp_path),
            "--long-video-ids",
            "37",
            "--num-prompts",
            "0",
            "--disable-shuffle",
            "--long-video-output-dir",
            str(tmp_path / "outputs"),
        ]
    )
    preprocess_serve_args(args)
    samples = get_samples(args, tokenizer=None)
    assert len(samples) == args.num_prompts == 1
    assert isinstance(samples[0], LongVideoSampleRequest)
    assert samples[0].case.id == "37"
    assert samples[0].case.end_s == 315
    assert samples[0].case.answer == "C"
    assert "gt" not in samples[0].case.prompt
    args.long_video_annotation_sha256 = "wrong-revision"
    with pytest.raises(ValueError, match="pinned dataset revision"):
        get_samples(args, tokenizer=None)


@pytest.mark.parametrize("change", [{"video": "../outside.mp4"}, {"realtime": float("nan")}, {"gt": 9}])
def test_invalid_annotation_fails_before_inference(tmp_path, change):
    annotation, row = _annotation(tmp_path)
    annotation.write_text(json.dumps([{**row, **change}]))
    with pytest.raises(ValueError):
        load_video_questions(annotation, tmp_path, dataset="ovo-bench")


def test_missing_or_forward_task_is_not_silently_dropped(tmp_path):
    annotation, row = _annotation(tmp_path)
    annotation.write_text(json.dumps([{**row, "task": "REC"}]))
    with pytest.raises(ValueError, match="Missing or unsupported"):
        load_video_questions(annotation, tmp_path, dataset="ovo-bench", ids=["37"])


def test_videomme_official_flat_rows(tmp_path):
    (tmp_path / "abc.mp4").touch()
    path = tmp_path / "mme.json"
    path.write_text(
        json.dumps(
            [
                {
                    "videoID": "abc",
                    "question_id": "322-2",
                    "task_type": "Temporal Reasoning",
                    "question": "Which order?",
                    "options": ["A. first", "B. second", "C. third", "D. fourth"],
                    "answer": "A",
                }
            ]
        )
    )
    (case,) = load_video_questions(path, tmp_path, dataset="videomme-streaming")
    assert case.end_s is None
    assert case.prompt.count("A. ") == 1
    assert case.id == "322-2"


@pytest.mark.parametrize("text,expected", [("</response> C", "C"), ("Answer: (B).", "B"), ("A or B", None), ("", None)])
def test_mcq_scoring(text, expected):
    assert extract_choice(text) == expected


def test_failed_and_invalid_answers_remain_in_denominator(tmp_path):
    annotation, _ = _annotation(tmp_path)
    (case,) = load_video_questions(annotation, tmp_path, dataset="ovo-bench")
    sample = LongVideoSampleRequest(prompt="", prompt_len=0, expected_output_len=0, case=case, output_dir=tmp_path)
    outputs = [
        MixRequestFuncOutput(success=True, generated_text="C"),
        MixRequestFuncOutput(success=True, generated_text="unsure"),
        MixRequestFuncOutput(success=False),
    ]
    report = finalize_long_video([sample] * 3, outputs)
    assert report["accuracy"] == 1 / 3
    assert report["completed"] == 2
    assert report["valid_answers"] == 1
    assert json.loads((tmp_path / "accuracy.json").read_text())["samples"] == 3


def test_prefix_decoder_never_exposes_future_frames(tmp_path):
    from dataclasses import replace

    import av

    annotation, _ = _annotation(tmp_path)
    (case,) = load_video_questions(annotation, tmp_path, dataset="ovo-bench")
    with av.open(str(case.video), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for index in range(20):
            frame = av.VideoFrame.from_image(Image.new("RGB", (32, 32), "red" if index < 10 else "blue"))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    directory = tmp_path / "frames"
    directory.mkdir()
    prepared = prepare_video(replace(case, end_s=1.0), directory, fps=4, max_edge=16)
    assert prepared.duration_s == 1
    assert len(prepared.frames) == 4
    for path in prepared.frames:
        with Image.open(path) as image:
            assert image.size == (16, 16)
            red, _, blue = image.getpixel((0, 0))
            assert red > 200 and blue < 20


def test_accuracy_file_failure_preserves_measurements(tmp_path, monkeypatch):
    from pathlib import Path

    annotation, _ = _annotation(tmp_path)
    (case,) = load_video_questions(annotation, tmp_path, dataset="ovo-bench")
    sample = LongVideoSampleRequest(prompt="", prompt_len=0, expected_output_len=0, case=case, output_dir=tmp_path)

    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", fail_write)
    report = finalize_long_video([sample], [MixRequestFuncOutput(success=True, generated_text="C")])
    assert report["accuracy"] == 1
    assert report["artifact_error"] == "disk full"


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("text_event", ["response.output_text.delta", "response.output_audio_transcript.delta"])
def test_real_websocket_replay_and_cleanup(tmp_path, fail, text_event):
    annotation, _ = _annotation(tmp_path)
    (case,) = load_video_questions(annotation, tmp_path, dataset="ovo-bench")
    frame = tmp_path / "frame.png"
    Image.new("RGB", (16, 16), "red").save(frame)
    prepared = PreparedVideo((frame, frame), 0.12, "test-digest")
    commands = []

    async def exercise():
        async def handler(socket):
            query = False
            count = 0
            async for raw in socket:
                event = json.loads(raw)
                commands.append((time.perf_counter(), event))
                kind = event["type"]
                if kind == "session.update":
                    await socket.send(
                        json.dumps(
                            {
                                "type": "session.created",
                                "session": {
                                    "id": "session-test",
                                    "capabilities": {"optional_input_modalities": ["video", "text"]},
                                },
                            }
                        )
                    )
                elif kind == "conversation.item.create":
                    query = True
                    assert "carpet" in event["item"]["content"][0]["text"]
                    assert "answer" not in event["item"]
                elif kind == "input_audio_buffer.commit":
                    count += 1
                    rid = f"response-{count}"
                    if fail:
                        await socket.send(json.dumps({"type": "error", "error": {"message": "broken"}}))
                        continue
                    for reply in [
                        {"type": "response.created", "response": {"id": rid}},
                        {"type": text_event, "response_id": rid, "delta": "C" if query else ""},
                        {"type": "response.done", "response": {"id": rid, "status": "completed"}},
                    ]:
                        await socket.send(json.dumps(reply))
                elif kind == "session.close":
                    await socket.send(json.dumps({"type": "session.closed"}))

        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            return await replay_video(
                case,
                prepared,
                url=f"http://127.0.0.1:{port}/v1/realtime",
                model="test",
                fps=20,
                timeout_s=1,
                extra_body={},
            )

    if fail:
        with pytest.raises(RuntimeError, match="broken"):
            asyncio.run(exercise())
    else:
        result = asyncio.run(exercise())
        assert result["text"] == "C"
        assert result["input_frames"] == 2
        assert 0 <= result["query_ttft_ms"] <= result["query_completion_ms"]
        appends = [when for when, event in commands if event["type"] == "input_audio_buffer.append"]
        question_at = next(when for when, event in commands if event["type"] == "conversation.item.create")
        assert question_at - appends[0] >= 0.1
    assert commands[-1][1]["type"] == "session.close"


def test_request_timeout_marks_failure_and_joins_cleanup(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from vllm_omni.benchmarks.duplex import long_video_runner
    from vllm_omni.benchmarks.duplex.long_video_serving import request_long_video

    annotation, _ = _annotation(tmp_path)
    (case,) = load_video_questions(annotation, tmp_path, dataset="ovo-bench")
    sample = LongVideoSampleRequest(prompt="", prompt_len=0, expected_output_len=0, case=case)
    request = SimpleNamespace(
        long_video_sample=sample, api_url="http://unused", model="test", model_name=None, extra_body={}
    )
    cleaned = []

    async def blocked(*args, **kwargs):
        try:
            await asyncio.Future()
        finally:
            cleaned.append(True)

    monkeypatch.setattr(long_video_runner, "run_video_question", blocked)
    output = asyncio.run(
        request_long_video(request, MixRequestFuncOutput(), headers={}, pbar=None, total_timeout_s=0.01)
    )
    assert not output.success
    assert output.timed_out
    assert "TimeoutError" in output.error
    assert cleaned == [True]


def test_accuracy_artifact_uses_standard_envelope(tmp_path, monkeypatch):
    from datetime import datetime

    from tests.e2e.accuracy.long_video_qa.validation import assert_accuracy_result, write_accuracy_result

    monkeypatch.setenv("ACC_BENCH_RESULT_DIR", str(tmp_path))
    item = dict(
        id="37",
        input_frames=315,
        input_duration_s=315,
        video_sha256="media",
        annotation_sha256="annotations",
        fps=1,
        max_edge=336,
        correct=True,
    )
    score = dict(
        protocol="continuous-visual-mcq-v1", samples=1, completed=1, valid_answers=1, accuracy=1.0, items=[item]
    )
    result = dict(
        model_id="jdopensource/JoyAI-VL-Interaction-Preview",
        Hardware="H800",
        branch="test",
        completed=1,
        failed=0,
        long_video_qa=score,
    )
    assert_accuracy_result(result, {"long_video_ids": ["37"]}, 1)
    path = write_accuracy_result(
        result,
        args=["--dataset-name", "ovo-bench", "--dataset-path", "annotations.json", "--max-concurrency", "8"],
        num_prompts=1,
        returncode=0,
        test_name="test_joyai_long_video_qa",
        name=None,
        when=datetime(2026, 11, 14, 12, 0, 0),
    )
    assert path.parent == tmp_path
    assert path.name == "result_acc_JoyAI-VL-Interaction-Preview_ovo-bench_20261114_120000.json"
    saved = json.loads(path.read_text())
    assert saved["Hardware"] == "H800"
    assert saved["max_concurrency"] == 8
    assert saved["accuracy"] == score
    assert saved["generation"] == {"completed": 1, "failed": 0}
    score["valid_answers"] = 0
    with pytest.raises(AssertionError, match="Missing or invalid"):
        assert_accuracy_result(result, {}, 1)
