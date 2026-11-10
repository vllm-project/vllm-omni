# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Long-run lifecycle contracts; scripted transport is not model qualification."""

import asyncio
import json
from contextlib import nullcontext

import pytest

from benchmarks.lingbot_world.regression import RegressionError, read_json
from tests.benchmarks.test_lingbot_world_realtime import _serve
from tests.dfx.lingbot_world import runner
from tests.e2e.online_serving import test_lingbot_world_v2_stepwise as lifecycle
from tests.helpers.runtime import OmniServer
from vllm_omni.benchmarks.realtime_video.client import BenchmarkError, SessionResult
from vllm_omni.benchmarks.realtime_video.workload import ChunkRecord

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("abrupt", [False, True])
@pytest.mark.parametrize("followup", ["valid", "truncated", "reused_id", "reordered", "stopped"])
def test_disconnect_probe_requires_complete_distinct_followup(tmp_path, monkeypatch, mocker, abrupt, followup):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"scripted image")
    monkeypatch.setattr(lifecycle, "_IMAGE_PATH_OVERRIDE", str(image))
    closed = []
    starts = []

    async def script(ws, payload):
        starts.append(payload)
        connection = len(starts)
        request_id = "first" if connection == 1 or followup == "reused_id" else "fresh"
        await ws.send(json.dumps({"type": "video.start", "request_id": request_id}))
        indices = [0] if connection == 1 or followup == "truncated" else [0, 1, 2]
        if connection == 2 and followup == "reordered":
            indices = [0, 2, 1]
        for index in indices:
            await ws.send(
                json.dumps(
                    {
                        "type": "video.chunk_metadata",
                        "kind": "media",
                        "request_id": request_id,
                        "generation_chunk_index": index,
                        "num_frames": 9 if index == 0 else 12,
                        "byte_length": 1,
                    }
                )
            )
            await ws.send(b"v")
        if connection == 1:
            await ws.wait_closed()
            closed.append(ws.close_code)
        else:
            await ws.send(
                json.dumps({"type": "session.done", "stopped": followup == "stopped", "chunks": len(indices)})
            )

    async def scenario():
        async with _serve(script) as url:
            port = int(url.split(":")[2].split("/")[0])
            server = mocker.Mock(spec=OmniServer, host="127.0.0.1", port=port, model="scripted")
            await asyncio.to_thread(
                lifecycle.test_lingbot_world_v2_stepwise_disconnect_releases_session, server, abrupt
            )

    with nullcontext() if followup == "valid" else pytest.raises(AssertionError):
        asyncio.run(asyncio.wait_for(scenario(), timeout=5))
    assert len(starts) == 2
    assert starts[0]["streaming_pacing"] == {"max_lead_ms": lifecycle._PACED_MAX_LEAD_MS}
    assert closed == [1006 if abrupt else 1000]


@pytest.mark.parametrize("terminal", ["error", "session.done", "empty_body"])
def test_disconnect_probe_cannot_pass_without_receiving_media(tmp_path, monkeypatch, terminal):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"scripted image")
    monkeypatch.setattr(lifecycle, "_IMAGE_PATH_OVERRIDE", str(image))

    async def script(ws, payload):
        if terminal == "empty_body":
            await ws.send(json.dumps({"type": "video.chunk_metadata", "kind": "media"}))
            await ws.send(b"")
        else:
            await ws.send(json.dumps({"type": terminal}))

    async def scenario():
        async with _serve(script) as url:
            await lifecycle._paced_disconnect_session(url, "scripted", abrupt=True)

    with pytest.raises(AssertionError):
        asyncio.run(asyncio.wait_for(scenario(), timeout=5))


def _mock_lifecycle(mocker, failure):
    names = [
        "test_lingbot_world_v2_stepwise_streams_one_chunk_per_block",
        "test_lingbot_world_v2_stepwise_paced_playback_bounds_lead",
        "test_lingbot_world_v2_stepwise_stop_during_paced_wait_releases_session",
        "test_lingbot_world_v2_stepwise_disconnect_releases_session",
    ]
    for index, name in enumerate(names):
        mocker.patch.object(
            lifecycle, name, side_effect=AssertionError("probe failed") if failure == index else None, create=True
        )


@pytest.mark.parametrize("failure", [None, 0, 1, 2, 3, "growth", "device"])
def test_lifecycle_retains_failed_and_unstarted_verdicts(tmp_path, mocker, failure):
    from vllm_omni.platforms import current_omni_platform as platform

    _mock_lifecycle(mocker, failure)
    mocker.patch.object(platform, "device", side_effect=lambda _: nullcontext())
    samples: list[tuple[int, int] | OSError] = [(900, 1000)]
    if failure == "device":
        samples.append(OSError("device lost"))
    elif failure == "growth":
        samples.append((700, 1000))
    else:
        samples.extend([(900, 1000)] * 5)
    mocker.patch.object(platform, "get_device_memory", side_effect=samples)
    destination = tmp_path / "lifecycle.json"
    server = mocker.Mock(spec=OmniServer)
    with nullcontext() if failure is None else pytest.raises((AssertionError, RegressionError, OSError)):
        runner.run_lifecycle_checks(server, tmp_path / "image.jpg", destination, dies=1, memory_budget=100)
    document = read_json(destination)
    rows = [
        document[name]
        for name in (
            "completion",
            "paced_playback",
            "stop_and_recover",
            "close_and_recover",
            "transport_loss_and_recover",
        )
    ]
    if failure is None:
        assert [row["status"] for row in rows] == ["passed"] * 5
    else:
        failed = next(index for index, row in enumerate(rows) if row["status"] == "failed")
        assert all(row["status"] == "passed" for row in rows[:failed])
        assert all(row["status"] == "not_run" for row in rows[failed + 1 :])
        if failure == "growth":
            assert rows[failed]["used_bytes"] == [300]


@pytest.mark.parametrize("bad_frames", [False, True])
def test_long_session_retains_partial_transport_and_validation_failures(tmp_path, mocker, bad_frames):
    from vllm_omni.platforms import current_omni_platform as platform

    mocker.patch.object(platform, "device", side_effect=lambda _: nullcontext())
    mocker.patch.object(platform, "get_device_memory", return_value=(900, 1000))
    result = SessionResult("failed", [ChunkRecord(0, 1, 1, 8, 99 if bad_frames else 9)], b"", 1, 0, 1, {}, [])

    async def session(*args, on_failure=None, on_chunk, **kwargs):
        on_chunk(result.records[0])
        if bad_frames:
            return result
        error = BenchmarkError("transport failed")
        if on_failure is not None:
            on_failure(result, error)
        raise error

    mocker.patch.object(runner, "run_session", side_effect=session)
    image = tmp_path / "image.jpg"
    image.write_bytes(b"scripted image")
    settings = {
        "num_chunks": 600,
        "warmup_chunks": 0,
        "dies": 1,
        "width": 832,
        "height": 480,
        "mux_fps": 16,
        "first_chunk_timeout_s": 1,
        "chunk_timeout_s": 1,
        "memory_growth_budget_bytes": 100,
        "cases": [{"prompt": "scene", "seed": 42}],
    }
    server = mocker.Mock(spec=OmniServer, host="localhost", port=8000, model="fixture")
    destination = tmp_path / "session.json"
    with pytest.raises((BenchmarkError, RegressionError)):
        runner.stability_session(server, settings, image, destination)
    trace = read_json(destination)
    assert trace["request_id"] == "failed" and len(trace["chunks"]) == 1
    assert trace["memory"] == [{"chunk": 0, "used_bytes": [100]}]
    assert trace["transport_complete"] is bad_frames
    assert trace["error"] == (None if bad_frames else "transport failed")


@pytest.mark.parametrize("failure", [None, "session", "lifecycle", "cleanup", "both"])
def test_stability_attempts_cleanup_after_failure_and_preserves_both_errors(tmp_path, mocker, failure):
    _mock_lifecycle(mocker, None)
    settings = {
        "num_chunks": 600,
        "measured_sessions": 2,
        "cases": [{"image": "scene"}],
        "output": tmp_path,
        "profile_path": tmp_path / "profile.json",
        "dies": 1,
        "startup_timeout_s": 1,
        "memory_growth_budget_bytes": 100,
    }
    settings["profile_path"].write_text("{}")
    mocker.patch.object(runner, "checked_assets", return_value=(tmp_path, {"scene": tmp_path / "image.jpg"}))
    mocker.patch.object(runner, "hardware_contract", return_value={"platform": "cuda"})
    mocker.patch.object(runner, "server_arguments", return_value=[])
    server = mocker.MagicMock(spec=OmniServer)
    mocker.patch("tests.helpers.runtime.OmniServer", return_value=server)
    sessions = mocker.patch.object(
        runner,
        "stability_session",
        side_effect=[
            {"request_id": "one", "memory": [{"used_bytes": [100]}]},
            {"request_id": "two", "memory": [{"used_bytes": [100]}]},
        ],
    )
    if failure == "session":
        sessions.side_effect = RegressionError("session failed")
    probes = mocker.patch.object(runner, "run_lifecycle_checks", create=True)
    if failure in {"lifecycle", "both"}:
        probes.side_effect = AssertionError("lifecycle failed")
    cleanup = mocker.patch.object(runner, "assert_device_cleanup")
    if failure in {"cleanup", "both"}:
        cleanup.side_effect = OSError("cleanup failed")
    report = runner.run_stability(settings)
    cleanup.assert_called_once_with(1, tmp_path / "cleanup.json")
    server.__exit__.assert_called_once()
    assert report["passed"] is (failure is None)
    assert len(report["errors"]) == (2 if failure == "both" else int(failure is not None))
    assert read_json(tmp_path / "stability.json") == report
