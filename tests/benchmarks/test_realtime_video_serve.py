# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exercise the real serving lifecycle with a local WS protocol peer."""

import asyncio
import json
from pathlib import Path

import pytest
from websockets.asyncio.server import serve

from vllm_omni.benchmarks.realtime_video.serving import build_serve_args
from vllm_omni.benchmarks.realtime_video.workload import pixel_frames_in_chunk
from vllm_omni.entrypoints.cli.benchmark.serve import OmniBenchmarkServingSubcommand

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]


@pytest.mark.parametrize("failure", [None, "warmup", "measured", "save-video"])
def test_real_serve_entry_saves_video_sessions_and_failed_warmups(monkeypatch, tmp_path: Path, failure):
    """Warmup and measured raw streams traverse the actual scheduler/writer."""
    import vllm.benchmarks.serve as upstream

    def no_tokenizer(*args, **kwargs):
        pytest.fail("Realtime video must not initialize a text tokenizer")

    monkeypatch.setattr(upstream, "get_tokenizer", no_tokenizer)
    observed = []

    async def scenario():
        async def handler(websocket):
            payload = json.loads(await websocket.recv())
            count = len(payload["extra_params"]["camera_action_script"])
            observed.append(count)
            request_id = f"request-{len(observed)}"
            await websocket.send(json.dumps({"type": "video.start", "request_id": request_id, "format": "m4s"}))
            failing = (failure == "warmup" and len(observed) == 1) or (failure == "measured" and len(observed) == 3)
            for index in range(count):
                await websocket.send(
                    json.dumps(
                        {
                            "type": "video.chunk_metadata",
                            "request_id": request_id,
                            "kind": "media",
                            "num_frames": pixel_frames_in_chunk(index),
                            "byte_length": 1,
                            "generation_chunk_index": index,
                        }
                    )
                )
                await websocket.send(b"x")
            if failing:
                # Even a full set of chunks is invalid without a successful terminal event.
                await websocket.send(json.dumps({"type": "error", "message": "deliberate WS failure"}))
            else:
                await websocket.send(json.dumps({"type": "session.done", "stopped": False, "chunks": count}))

        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            args = build_serve_args(
                [
                    "--base-url",
                    f"ws://127.0.0.1:{port}/v1/realtime/video",
                    "--model",
                    "test-video",
                    "--num-prompts",
                    "2",
                    "--num-warmups",
                    "1",
                    "--realtime-video-num-chunks",
                    "10",
                    "--realtime-video-warmup-num-chunks",
                    "8",
                    "--realtime-video-image",
                    "data:image/png;base64,eA==",
                    "--realtime-video-ping-interval",
                    "0",
                    "--disable-tqdm",
                    "--save-result",
                    "--result-dir",
                    str(tmp_path),
                    "--result-filename",
                    "result.json",
                ]
            )
            if failure == "save-video":
                blocked_parent = tmp_path / "parent-file"
                blocked_parent.write_text("not a directory")
                args.realtime_video_save_video = str(blocked_parent / "video.mp4")
            if failure:
                with pytest.raises(RuntimeError, match="Realtime video benchmark failed"):
                    await asyncio.to_thread(OmniBenchmarkServingSubcommand.cmd, args)
            else:
                await asyncio.to_thread(OmniBenchmarkServingSubcommand.cmd, args)

    asyncio.run(asyncio.wait_for(scenario(), timeout=30))
    saved = json.loads((tmp_path / "result.json").read_text())
    video = saved["realtime_video"]
    assert saved["backend"] == "realtime-video"
    assert "total_output_tokens" not in saved and "mean_ttft_ms" not in saved and "itls" not in saved
    assert len(video["warmups"]) == 1
    warmup = video["warmups"][0]
    assert warmup["request_id"] == "request-1" and len(warmup["chunks"]) == 8
    assert warmup["measured"] is False
    assert video["passed"] is (failure is None)
    if failure == "warmup":
        assert observed == [8]
        assert video["sessions"] == [] and saved["completed"] == 0
        assert video["failed_warmups"] == 1 and warmup["success"] is False
    else:
        assert observed == [8, 10, 10]  # No hidden full-rollout ready request.
        assert len(video["sessions"]) == 2
        assert all(len(session["chunks"]) == 10 for session in video["sessions"])
        assert video["sessions"][0]["metrics"]["steady_chunks"] == 3
        assert video["sessions"][0]["metrics"]["total_frames"] == 117
        assert "ttfc_ms" in video["sessions"][0]
        assert saved["completed"] == (1 if failure == "measured" else 2)
        if failure == "measured":
            assert video["sessions"][-1]["success"] is False
    if failure == "save-video":
        assert all(session["success"] and session["metrics"] for session in video["sessions"])
        assert video["metrics"] and video["finalization_errors"]
        assert "save video:" in video["errors"][0]
    elif failure:
        assert video["errors"] and "deliberate WS failure" in video["errors"][0]


@pytest.mark.parametrize(
    "flags",
    [
        ["--max-concurrency", "2"],
        ["--request-rate", "1"],
        ["--probe-request-rate", "1"],
        ["--self-timed"],
        ["--ready-check-timeout-sec", "5"],
        ["--profile"],
        ["--ramp-up-strategy", "linear"],
        ["--plot-timeline"],
        ["--plot-dataset-stats"],
    ],
)
def test_realtime_video_rejects_modes_that_change_the_rollout_population(flags):
    with pytest.raises(ValueError, match="realtime-video"):
        build_serve_args(flags)


def test_realtime_video_defaults_are_safe_and_do_not_change_text_defaults():
    from vllm.benchmarks.serve import add_cli_args

    from vllm_omni.utils.tracking_parser import TrackingArgumentParser

    args = build_serve_args([])
    assert (args.num_prompts, args.num_warmups, args.max_concurrency) == (1, 0, 1)
    assert args.skip_tokenizer_init and args.ready_check_timeout_sec == 0
    parser = TrackingArgumentParser()
    OmniBenchmarkServingSubcommand.add_cli_args(parser)
    text_args = parser.parse_args([])
    baseline_parser = TrackingArgumentParser()
    add_cli_args(baseline_parser)
    baseline_args = baseline_parser.parse_args([])
    assert text_args.num_prompts == 1000
    assert text_args.ready_check_timeout_sec == baseline_args.ready_check_timeout_sec
    assert text_args.skip_tokenizer_init == baseline_args.skip_tokenizer_init
