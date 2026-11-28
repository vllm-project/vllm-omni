# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Long-run lifecycle contracts; scripted transport is not model qualification."""

import asyncio
import json
import os
from contextlib import nullcontext
from pathlib import Path

import pytest

from tests.benchmarks.test_lingbot_world_realtime import _serve
from tests.e2e.online_serving import test_lingbot_world_v2_stepwise as lifecycle
from tests.helpers.runtime import OmniServer

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


@pytest.mark.parametrize("failure", [None, "stop", "close", "abort", "growth", "device"])
def test_stability_probe_retains_memory_and_failure_verdicts(tmp_path, mocker, failure):
    from tests.dfx.stability import lingbot_world as probes

    image = tmp_path / "image.jpg"
    image.write_bytes(b"fixture")
    original_image = lifecycle._IMAGE_PATH_OVERRIDE
    mocker.patch.object(probes.platform, "get_device_count", return_value=2)
    mocker.patch.object(probes.platform, "device", side_effect=lambda _: nullcontext())
    samples: list[tuple[int, int] | OSError] = [(900, 1000)] * 8
    if failure == "growth":
        samples[2] = (700, 1000)
    if failure == "device":
        samples[2] = OSError("device lost")
    mocker.patch.object(probes.platform, "get_device_memory", side_effect=samples)
    mocker.patch.object(
        lifecycle,
        "test_lingbot_world_v2_stepwise_stop_during_paced_wait_releases_session",
        side_effect=AssertionError("stop failed") if failure == "stop" else None,
    )
    mocker.patch.object(
        lifecycle,
        "test_lingbot_world_v2_stepwise_disconnect_releases_session",
        side_effect=[
            AssertionError("close failed") if failure == "close" else None,
            AssertionError("abort failed") if failure == "abort" else None,
        ],
    )
    destination = tmp_path / "lifecycle.json"
    with nullcontext() if failure is None else pytest.raises((AssertionError, OSError)):
        probes.run_lifecycle_checks("localhost", 8000, "model", image, destination, memory_budget=100)
    report = json.loads(destination.read_text())
    assert report["passed"] is (failure is None)
    assert report["baseline_bytes"] == [100, 100]
    rows = [
        report["cases"][name]["status"]
        for name in ("stop_and_recover", "close_and_recover", "transport_loss_and_recover")
    ]
    if failure is None:
        assert rows == ["passed"] * 3
    else:
        index = rows.index("failed")
        assert rows[:index] == ["passed"] * index
        assert rows[index + 1 :] == ["not_run"] * (2 - index)
        assert report["error"]
    assert lifecycle._IMAGE_PATH_OVERRIDE == original_image


@pytest.mark.parametrize("budget", [-1, True, 1.5])
def test_stability_probe_rejects_invalid_budget_before_serving(tmp_path, budget):
    from tests.dfx.stability.lingbot_world import run_lifecycle_checks

    with pytest.raises(ValueError, match="nonnegative integer"):
        run_lifecycle_checks(
            "localhost", 1, "model", tmp_path / "missing.jpg", tmp_path / "report.json", memory_budget=budget
        )


@pytest.mark.parametrize("failure", [None, "process", "missing", "malformed", "empty", "lifecycle"])
@pytest.mark.parametrize("enabled", [False, True])
def test_shared_stability_batch_retains_raw_results_and_gates_recovery(tmp_path, mocker, monkeypatch, failure, enabled):
    import subprocess

    from tests.dfx.stability import helpers
    from tests.dfx.stability import lingbot_world as probes

    image = tmp_path / "image.jpg"
    image.write_bytes(b"fixture")
    outputs = []
    monkeypatch.setenv("PYTHONPATH", "/inherited/compiler/modules")

    def run(command, **kwargs):
        assert kwargs["env"]["PYTHONPATH"] == os.pathsep.join((str(helpers.REPO_ROOT), "/inherited/compiler/modules"))
        assert "--lifecycle-checks" not in command
        assert "--lifecycle-memory-growth-budget-bytes" not in command
        output = Path(command[command.index("--output-json") + 1])
        outputs.append(output)
        output.with_name("benchmark-serve-result.json").write_text('{"realtime_video": {"passed": false}}')
        if failure != "missing":
            output.write_text(
                "invalid JSON"
                if failure == "malformed"
                else json.dumps(
                    {
                        "sessions": [] if failure == "empty" else [{"request_id": "one"}],
                        "metrics": {"wall_seconds": 12.0},
                    }
                )
            )
        return subprocess.CompletedProcess(
            command, 1 if failure == "process" else 0, "partial stdout", "partial stderr"
        )

    mocker.patch.object(helpers.subprocess, "run", side_effect=run)
    check = mocker.patch.object(
        probes,
        "run_lifecycle_checks",
        side_effect=AssertionError("recovery failed") if failure == "lifecycle" else None,
    )
    result = helpers._run_one_lingbot_world_batch(
        "localhost",
        8000,
        "model",
        {
            "image": str(image),
            "num_chunks": 40,
            **({"lifecycle_checks": True, "lifecycle_memory_growth_budget_bytes": 100} if enabled else {}),
        },
        1,
        None,
        1,
        str(tmp_path / "results"),
        0,
    )
    bad = failure in {"process", "missing", "malformed", "empty"} or failure == "lifecycle" and enabled
    assert result["failed"] == int(bad)
    assert result["completed"] == (0 if bad else 1)
    assert check.call_count == int(enabled and failure not in {"process", "missing", "malformed", "empty"})
    assert len(outputs) == 1
    output = outputs[0]
    assert output.with_name("benchmark-serve-result.json").is_file()
    assert output.with_name("client.stdout.log").read_text() == "partial stdout"
    assert output.with_name("client.stderr.log").read_text() == "partial stderr"
    assert output.exists() is (failure != "missing")


def test_lifecycle_enabled_only_in_cuda_stability_profile():
    root = Path(__file__).resolve().parents[2]
    cases = json.loads((root / "tests/dfx/stability/tests/test_lingbot_world.json").read_text())
    assert cases[0]["benchmark_params"][0]["lifecycle_checks"] is True
    assert cases[0]["benchmark_params"][0]["lifecycle_memory_growth_budget_bytes"] == 256 * 2**20
    assert "lifecycle_checks" not in cases[1]["benchmark_params"][0]


@pytest.mark.parametrize("failed", [False, True])
def test_shared_batch_retains_actual_client_raw_trace(tmp_path, monkeypatch, failed):
    from tests.dfx.stability import helpers
    from vllm_omni.benchmarks.realtime_video.workload import pixel_frames_in_chunk

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    async def script(ws, payload):
        chunks = len(payload["extra_params"]["camera_action_script"])
        await ws.send(json.dumps({"type": "video.start", "request_id": "scripted", "format": "m4s"}))
        for index in range(chunks):
            await ws.send(
                json.dumps(
                    {
                        "type": "video.chunk_metadata",
                        "kind": "media",
                        "request_id": "scripted",
                        "generation_chunk_index": index,
                        "num_frames": pixel_frames_in_chunk(index),
                        "byte_length": 1,
                    }
                )
            )
            await ws.send(b"v")
        await ws.send(
            json.dumps(
                {"type": "error", "message": "retained failure"}
                if failed
                else {"type": "session.done", "stopped": False, "chunks": chunks}
            )
        )

    async def scenario():
        async with _serve(script) as url:
            port = int(url.split(":")[2].split("/")[0])
            return await asyncio.to_thread(
                helpers._run_one_lingbot_world_batch,
                "127.0.0.1",
                port,
                "scripted",
                {"num_chunks": 8, "image": "data:image/png;base64,eA=="},
                1,
                None,
                1,
                str(tmp_path),
                0,
            )

    result = asyncio.run(asyncio.wait_for(scenario(), timeout=90))
    assert result["failed"] == int(failed)
    directories = list(tmp_path.glob("lingbot-world-batch-*"))
    assert len(directories) == 1
    raw = json.loads((directories[0] / "benchmark-serve-result.json").read_text())
    assert raw["realtime_video"]["passed"] is not failed
    assert (directories[0] / "client.stdout.log").is_file()
    assert (directories[0] / "command.json").is_file()
    assert (directories[0] / "benchmark.json").is_file() is not failed
