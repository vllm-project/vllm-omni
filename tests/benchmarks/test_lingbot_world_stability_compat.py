# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Protect legacy paths still consumed by the shared stability runner."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from websockets.asyncio.server import ServerConnection, serve

from vllm_omni.benchmarks.realtime_video.workload import pixel_frames_in_chunk
from vllm_omni.config.stage_config import load_deploy_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "benchmarks" / "lingbot_world"


@pytest.mark.parametrize("profile", ["usp4_compiled.yaml", "npu_usp4_eager.yaml"])
def test_upstream_stability_deploy_paths_remain_loadable(profile: str) -> None:
    """The upstream H800/A3 cases still load these bases before SP8 overrides."""
    deploy = load_deploy_config(LEGACY / "configs" / profile)
    assert deploy.pipeline == "lingbot_world"
    assert len(deploy.stages) == 1
    assert deploy.stages[0].step_execution is True
    assert deploy.stages[0].engine_extras["streaming_output"] is True


@pytest.mark.parametrize("failed", [False, True])
def test_upstream_stability_script_keeps_legacy_cli_and_result(tmp_path: Path, failed: bool) -> None:
    """Run the actual legacy subprocess against a local realtime-video peer."""
    output = tmp_path / "batch.json"

    async def scenario() -> None:
        async def handler(websocket: ServerConnection) -> None:
            payload = json.loads(await websocket.recv())
            assert len(payload["extra_params"]["camera_action_script"]) == 8
            await websocket.send(json.dumps({"type": "video.start", "request_id": "batch", "format": "m4s"}))
            for index in range(8):
                await websocket.send(
                    json.dumps(
                        {
                            "type": "video.chunk_metadata",
                            "request_id": "batch",
                            "kind": "media",
                            "num_frames": pixel_frames_in_chunk(index),
                            "byte_length": 1,
                            "generation_chunk_index": index,
                        }
                    )
                )
                await websocket.send(b"x")
            terminal: dict[str, str | bool | int] = (
                {"type": "error", "message": "injected stability failure"}
                if failed
                else {"type": "session.done", "stopped": False, "chunks": 8}
            )
            await websocket.send(json.dumps(terminal))

        async with serve(handler, "127.0.0.1", 0) as server:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(LEGACY / "benchmark_lingbot_world_realtime.py"),
                "--host",
                "127.0.0.1",
                "--port",
                str(server.sockets[0].getsockname()[1]),
                "--model",
                "stability-test",
                "--sessions",
                "1",
                "--output-json",
                str(output),
                "--image",
                "data:image/png;base64,eA==",
                "--num-chunks",
                "8",
                "--fps",
                "16",
                "--target-fps",
                "12",
                "--seed",
                "42",
                "--camera-pattern",
                "forward",
                "--flow-shift",
                "5.0",
                cwd=ROOT,
                env={
                    "PATH": os.environ["PATH"],
                    "PYTHONPATH": str(ROOT),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "CUDA_VISIBLE_DEVICES": "",
                    "OMP_NUM_THREADS": "1",
                },
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                stdout, _ = await asyncio.wait_for(process.communicate(), timeout=60)
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
            if failed:
                assert process.returncode != 0, stdout.decode()
                assert "injected stability failure" in stdout.decode()
                assert not output.exists()
                shared = json.loads((tmp_path / "batch-serve-result.json").read_text())
                assert shared["realtime_video"]["passed"] is False
            else:
                assert process.returncode == 0, stdout.decode()
                document = json.loads(output.read_text())
                # These are the exact fields consumed by _run_one_lingbot_world_batch.
                assert len(document["sessions"]) == 1
                assert document["metrics"]["wall_seconds"] > 0
                shared = json.loads((tmp_path / "batch-serve-result.json").read_text())
                assert shared["realtime_video"] == document

    asyncio.run(scenario())
