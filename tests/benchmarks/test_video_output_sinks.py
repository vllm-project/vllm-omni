# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Lossy encoding must not be reported as a failed lossless transport."""

from __future__ import annotations

import json
from multiprocessing import shared_memory

import pytest

from benchmarks.diffusion import bench_video_output_sinks as benchmark
from vllm_omni.entrypoints.openai.video_api_utils import encode_video_base64
from vllm_omni.entrypoints.openai.video_output_shm import export_video_frames_to_shm, release_video_frames

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]


@pytest.mark.parametrize("mode", ["base64", "shared_memory"])
def test_consumer_reports_decoding_separately_from_lossless_identity(tmp_path, capsys, mode: str) -> None:
    frames = benchmark._frames(5, 32, 48)
    handle = None
    if mode == "shared_memory":
        handle = export_video_frames_to_shm(frames, ttl_seconds=300)
        payload = {"handle": handle.model_dump(mode="json")}
    else:
        payload = {"b64_json": encode_video_base64(frames, fps=24), "shape": list(frames.shape)}
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(payload))
    try:
        benchmark._consumer(mode, str(path), benchmark._digest(frames))
        result = json.loads(capsys.readouterr().out.split("RESULT ")[-1])
        assert result["decoded_ok"] is True
        assert result["lossless"] is (True if mode == "shared_memory" else None)
        if handle is not None:
            with pytest.raises(FileNotFoundError):
                shared_memory.SharedMemory(name=handle.name)
    finally:
        if handle is not None:
            release_video_frames(handle)
