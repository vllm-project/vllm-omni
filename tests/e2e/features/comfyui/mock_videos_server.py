# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Minimal mock of vLLM-Omni's ``/v1/videos`` API for e2e tests.

Records every received multipart field to ``MOCK_STATE_FILE`` (JSON) so a test
can assert exactly what the client serialized, and returns a synthetic MP4.
"""

import asyncio
import json
import os
import subprocess
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import Response
from starlette.datastructures import UploadFile

app = FastAPI()
_JOBS: dict[str, dict] = {}
_STATE_FILE = os.environ.get("MOCK_STATE_FILE")


def _record(fields: dict) -> None:
    if _STATE_FILE:
        with open(_STATE_FILE, "w") as f:
            json.dump(fields, f)


def _synthetic_mp4() -> bytes:
    return subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=24:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=32000:duration=1",
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "32000",
            "-ac",
            "2",
            "-movflags",
            "frag_keyframe+empty_moov",
            "-f",
            "mp4",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    ).stdout


@app.post("/v1/videos")
async def create_video(request: Request):
    form = await request.form()
    fields = {}
    for key, value in form.items():
        if isinstance(value, UploadFile):
            data = await value.read()
            fields[key] = {
                "file": True,
                "size": len(data),
                "filename": value.filename,
                "content_type": value.content_type,
            }
            if value.content_type == "application/json":
                fields[key]["json"] = json.loads(data)
        else:
            fields[key] = str(value)
    _record(fields)

    job_id = str(uuid.uuid4())
    _JOBS[job_id] = {"status": "queued"}
    return {"id": job_id, "status": "queued"}


@app.get("/v1/videos/{job_id}")
async def get_video(job_id: str):
    if job_id not in _JOBS:
        return {"status": "failed", "error": "unknown job"}
    return {"id": job_id, "status": "completed"}


@app.get("/v1/videos/{job_id}/content")
async def get_video_content(job_id: str):
    if job_id not in _JOBS:
        return Response(status_code=404)
    mp4 = await asyncio.to_thread(_synthetic_mp4)
    return Response(content=mp4, media_type="video/mp4")


@app.delete("/v1/videos/{job_id}")
async def delete_video(job_id: str):
    _JOBS.pop(job_id, None)
    return {"deleted": True}
