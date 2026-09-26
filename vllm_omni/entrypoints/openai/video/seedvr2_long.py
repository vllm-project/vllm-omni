# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One asynchronous SeedVR2 request, bounded model windows, one MP4 writer."""

import asyncio
import hashlib
import io
import itertools
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from uuid import uuid4

import av
import imageio_ffmpeg
import numpy as np
import regex as re
import requests
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from vllm_omni.diffusion import envs
from vllm_omni.diffusion.models.seedvr2.video import max_frames, sharded_budget
from vllm_omni.inputs.data import COLOR_CORRECTION_METHODS, DEFAULT_COLOR_CORRECTION_METHOD

logger = logging.getLogger(__name__)
router = APIRouter()
MAX_FRAMES = 7200
FPS = 24
OVERLAP = 4
# Windows cross the HTTP boundary losslessly (x264 qp 0, which is also its
# fastest mode); only the final MP4 is compressed.
LOSSLESS = {"preset": "ultrafast", "qp": "0"}
UPLOAD_CHUNK = 1 << 20
JOB_ID = re.compile(r"[0-9a-f]{32}")
active_task: asyncio.Task[None] | None = None
job_lock = asyncio.Lock()


class JobError(Exception):
    """An error whose message is safe to return to the client."""


class JobCancelledError(JobError):
    """The client asked the job to stop."""


def _jobs_root() -> Path:
    return Path(envs.VLLM_OMNI_SEEDVR2_LONG_OUTPUT_DIR) / "seedvr2-long"


def _job_dir(job_id: str) -> Path:
    """Resolve a client-supplied job id, which must never shape the path."""
    if not JOB_ID.fullmatch(job_id):
        raise HTTPException(404, "SeedVR2 long-video job not found")
    return _jobs_root() / job_id


def _sweep_expired_jobs() -> None:
    """Drop settled job directories so repeated submissions cannot fill the disk."""
    deadline = time.time() - envs.VLLM_OMNI_SEEDVR2_LONG_JOB_TTL_SECONDS
    for job in _jobs_root().glob("*"):
        status = job / "status.json"
        if not job.is_dir() or not JOB_ID.fullmatch(job.name) or not status.exists():
            continue
        record = json.loads(status.read_text())
        # A job owned by a dead process is settled too, or a restart would leak it.
        settled = record["status"] not in {"queued", "running"} or record.get("pid") != os.getpid()
        if settled and status.stat().st_mtime <= deadline:
            shutil.rmtree(job, ignore_errors=True)


def _store_upload(upload: UploadFile, destination: Path) -> None:
    """Copy the upload under a size cap so one client cannot fill the disk."""
    limit = envs.VLLM_OMNI_SEEDVR2_LONG_MAX_UPLOAD_BYTES
    written = 0
    with destination.open("wb") as target:
        while chunk := upload.file.read(UPLOAD_CHUNK):
            written += len(chunk)
            if written > limit:
                raise HTTPException(413, f"SeedVR2 long-video upload exceeds {limit} bytes")
            target.write(chunk)
    if not written:
        raise HTTPException(400, "SeedVR2 long-video upload is empty")


def _window(width: int, height: int) -> int:
    """Longest window the whole-clip budget admits at this output size.

    Every window pays a fixed cost for its round trip through the serving
    stack and throws away its overlap, so the route uses the largest window the
    model accepts. The model pads clips to 4n+1 frames, so the window has that
    length too; 0 means not even one frame fits.
    """
    frame_pixels, clip_pixels = sharded_budget()
    if width * height > frame_pixels:
        return 0
    fit = min(clip_pixels // (width * height), max_frames(), envs.VLLM_OMNI_SEEDVR2_LONG_MAX_WINDOW)
    return fit - (fit - 1) % 4


def _status(job: Path, stage: str, frames: int, error: str = "") -> None:
    # The owning PID lets a poller detect a job that a server restart abandoned.
    record = {"status": stage, "frames": frames, "error": error, "pid": os.getpid()}
    pending = job / "status.json.tmp"
    pending.write_text(json.dumps(record) + "\n")
    pending.replace(job / "status.json")


def _frames(source: Path, width: int, height: int, loop: bool) -> Iterator[av.VideoFrame]:
    while True:
        count = 0
        with av.open(str(source)) as container:
            video = container.streams.video[0]
            if video.average_rate != Fraction(FPS):
                raise JobError("SeedVR2 long video requires 24 FPS input")
            for frame in container.decode(video=0):
                if frame.pts is None or frame.time_base is None or frame.pts * frame.time_base != Fraction(count, FPS):
                    raise JobError("SeedVR2 long video requires constant 24 FPS timestamps starting at zero")
                count += 1
                # The model resizes to the output on the device, so windows carry
                # source pixels; only a larger source shrinks here to stay in budget.
                if frame.width * frame.height > width * height:
                    yield frame.reformat(width=width, height=height, format="yuv420p", interpolation="BICUBIC")
                else:
                    yield frame.reformat(width=frame.width & ~1, height=frame.height & ~1, format="yuv420p")
        if count == 0:
            raise JobError("Input video has no frames")
        if not loop:
            return


# Consecutive windows share their overlap frames and encoding stamps each
# frame's pts, so two in-flight windows must not encode at the same time.
_segment_lock = threading.Lock()


def _segment(frames: list[av.VideoFrame]) -> bytes:
    buffer = io.BytesIO()
    with _segment_lock, av.open(buffer, "w", format="matroska") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = frames[0].width, frames[0].height, "yuv420p"
        stream.options = LOSSLESS
        for index, frame in enumerate(frames):
            frame.pts, frame.time_base = index, Fraction(1, FPS)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


def _restore(
    frames: list[av.VideoFrame],
    width: int,
    height: int,
    seed: int,
    color_correction_method: str,
    port: int,
    authorization: str,
) -> list[np.ndarray]:
    response = requests.post(
        f"http://127.0.0.1:{port}/v1/videos/sync",
        data={
            "prompt": " ",
            "size": f"{width}x{height}",
            "num_frames": str(len(frames)),
            "num_inference_steps": "1",
            "guidance_scale": "1",
            "seed": str(seed),
            "color_correction_method": color_correction_method,
            "extra_params": json.dumps({"video_codec_options": LOSSLESS}),
        },
        files={"input_references": ("window.mkv", _segment(frames), "video/x-matroska")},
        headers={"Authorization": authorization} if authorization else {},
        timeout=900,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        raise JobError("SeedVR2 window restoration failed") from error
    with av.open(io.BytesIO(response.content)) as container:
        video = container.streams.video[0]
        decoded = list(container.decode(video=0))
        if (video.width, video.height, video.average_rate, len(decoded)) != (width, height, Fraction(FPS), len(frames)):
            raise JobError("SeedVR2 window restoration returned an unexpected geometry or frame count")
        if [frame.pts * frame.time_base for frame in decoded] != [Fraction(index, FPS) for index in range(len(frames))]:
            raise JobError("SeedVR2 window restoration returned invalid timestamps")
        return [frame.to_ndarray(format="rgb24") for frame in decoded]


def _run(
    job: Path,
    width: int,
    height: int,
    target: int,
    loop: bool,
    seed: int,
    color_correction_method: str,
    port: int,
    authorization: str,
) -> None:
    window = _window(width, height)
    if window <= OVERLAP:
        raise JobError("SeedVR2 long video output exceeds the configured SeedVR2 clip budget")
    with tempfile.TemporaryDirectory(prefix="seedvr2-long-") as scratch:
        _restore_to(
            job, Path(scratch), window, width, height, target, loop, seed, color_correction_method, port, authorization
        )


def _restore_to(
    job: Path,
    scratch: Path,
    window: int,
    width: int,
    height: int,
    target: int,
    loop: bool,
    seed: int,
    color_correction_method: str,
    port: int,
    authorization: str,
) -> None:
    """Restore into local scratch files, then publish the result into ``job``."""
    source = job / "input.mp4"
    video_path = scratch / "video.mp4"
    output = scratch / "output.mp4"
    source_frames = _frames(source, width, height, loop)

    def windows() -> Iterator[list[av.VideoFrame]]:
        batch = list(itertools.islice(source_frames, min(window, target)))
        start = 0
        while True:
            if len(batch) != min(window, target - start):
                raise JobError("Input video has fewer frames than requested; set loop_input=true to repeat it")
            yield batch
            start += len(batch) - OVERLAP
            if start + OVERLAP >= target:
                return
            batch = batch[-OVERLAP:] + list(itertools.islice(source_frames, min(window, target - start) - OVERLAP))

    def restore(frames: list[av.VideoFrame]) -> Future[list[np.ndarray]]:
        if (job / "cancel").exists():
            raise JobCancelledError("SeedVR2 long-video job was cancelled")
        return pool.submit(_restore, frames, width, height, seed, color_correction_method, port, authorization)

    written = 0
    pending: list[np.ndarray] = []
    # Two windows stay in flight, so the engine always has the next one queued
    # while this thread blends and encodes, and the other prepares its upload.
    with ThreadPoolExecutor(max_workers=2) as pool, av.open(str(video_path), "w", format="mp4") as container:
        video = container.add_stream("libx264", rate=FPS)
        video.width, video.height, video.pix_fmt = width, height, "yuv420p"
        video.options = {"preset": "veryfast", "crf": "18", "tune": "zerolatency", "bf": "0"}

        def write(array: np.ndarray) -> None:
            nonlocal written
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame.pts, frame.time_base = written, Fraction(1, FPS)
            for packet in video.encode(frame):
                container.mux(packet)
            written += 1

        batches = windows()
        in_flight = deque(restore(batch) for batch in itertools.islice(batches, 2))
        while in_flight:
            restored = in_flight.popleft().result()
            following = next(batches, None)
            if following is not None:
                in_flight.append(restore(following))
            last = not in_flight
            if not pending:
                for frame in restored if last else restored[:-OVERLAP]:
                    write(frame)
            else:
                for offset in range(OVERLAP):
                    newer = (offset + 1) / (OVERLAP + 1)
                    write(np.rint(pending[offset] * (1 - newer) + restored[offset] * newer).astype(np.uint8))
                for frame in restored[OVERLAP:] if last else restored[OVERLAP:-OVERLAP]:
                    write(frame)
            _status(job, "running", written)
            pending = restored[-OVERLAP:]
        for packet in video.encode():
            container.mux(packet)
    if written != target:
        raise JobError(f"SeedVR2 restored {written} of {target} requested frames")

    with av.open(str(source)) as container:
        has_audio = bool(container.streams.audio)
    if has_audio:
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-nostdin",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-stream_loop",
                "-1" if loop else "0",
                "-i",
                str(source),
                "-i",
                str(video_path),
                "-map",
                "1:v:0",
                "-map",
                "0:a:0",
                "-frames:v",
                str(target),
                "-t",
                str(target / FPS),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
        )
    else:
        video_path.replace(output)
    with av.open(str(output)) as container:
        video = container.streams.video[0]
        decoded = 0
        for decoded, frame in enumerate(container.decode(video=0), start=1):
            if (
                frame.pts is None
                or frame.time_base is None
                or frame.pts * frame.time_base != Fraction(decoded - 1, FPS)
            ):
                raise JobError("SeedVR2 output encoding produced invalid timestamps")
        if (video.width, video.height, video.average_rate, decoded) != (width, height, Fraction(FPS), target):
            raise JobError("SeedVR2 output encoding failed frame or geometry validation")
        if has_audio and not container.streams.audio:
            raise JobError("SeedVR2 output encoding lost the audio track")
    if has_audio:
        with av.open(str(output)) as container:
            audio = container.streams.audio[0]
            samples = sum(frame.samples for frame in container.decode(audio=0))
            if samples < (target / FPS - 1) * audio.rate:
                raise JobError("SeedVR2 output audio ends before the video")
    with output.open("rb") as content:
        digest = hashlib.file_digest(content, "sha256").hexdigest()
    # One sequential copy is the only write to the job directory, which may sit
    # on network storage where the encoder's and muxer's small writes are slow.
    staged = job / "output.mp4.tmp"
    shutil.move(output, staged)
    staged.replace(job / "output.mp4")
    (job / "result.json").write_text(
        json.dumps({"frames": target, "width": width, "height": height, "fps": FPS, "sha256": digest}) + "\n"
    )
    _status(job, "completed", target)


def _background(
    job: Path,
    width: int,
    height: int,
    target: int,
    loop: bool,
    seed: int,
    color_correction_method: str,
    port: int,
    authorization: str,
) -> None:
    try:
        _run(job, width, height, target, loop, seed, color_correction_method, port, authorization)
    except Exception as error:
        logger.exception("SeedVR2 long-video job %s failed", job.name)
        current = json.loads((job / "status.json").read_text())
        detail = str(error) if isinstance(error, JobError) else "SeedVR2 long-video restoration failed"
        _status(job, "cancelled" if isinstance(error, JobCancelledError) else "failed", current["frames"], detail)


@router.post("/v1/seedvr2/restore-long", status_code=202)
async def create_long_video(
    raw_request: Request,
    input_references: UploadFile = File(...),
    num_frames: int = Form(...),
    size: str = Form(...),
    loop_input: bool = Form(False),
    prompt: str = Form(" "),
    seed: int = Form(7723),
    color_correction_method: str = Form(DEFAULT_COLOR_CORRECTION_METHOD),
) -> dict[str, str | int]:
    global active_task
    if raw_request.app.state.api_server_count != 1:
        raise HTTPException(409, "SeedVR2 long-video jobs require one API server")
    try:
        width, height = (int(value) for value in size.lower().split("x"))
    except ValueError as error:
        raise HTTPException(400, "size must be WIDTHxHEIGHT") from error
    if not 1 <= num_frames <= MAX_FRAMES or min(width, height) < 16 or width % 16 or height % 16:
        raise HTTPException(400, "SeedVR2 long-video frame count or dimensions are invalid")
    if prompt.strip():
        raise HTTPException(400, "SeedVR2 long video requires a blank prompt")
    if _window(width, height) <= OVERLAP:
        raise HTTPException(400, "SeedVR2 long video output exceeds the configured SeedVR2 frame or clip budget")
    if not 0 <= seed <= 2**32 - 1:
        raise HTTPException(400, "Seed must be a 32-bit unsigned integer")
    if color_correction_method not in COLOR_CORRECTION_METHODS:
        raise HTTPException(400, f"color_correction_method must be one of {list(COLOR_CORRECTION_METHODS)}")
    async with job_lock:
        if active_task is not None and not active_task.done():
            raise HTTPException(409, "A SeedVR2 long-video job is already running")
        await asyncio.to_thread(_sweep_expired_jobs)
        job_id = uuid4().hex
        job = _jobs_root() / job_id
        job.mkdir(parents=True)
        try:
            await asyncio.to_thread(_store_upload, input_references, job / "input.mp4")
        except HTTPException:
            shutil.rmtree(job, ignore_errors=True)
            raise
        _status(job, "queued", 0)
        active_task = asyncio.create_task(
            asyncio.to_thread(
                _background,
                job,
                width,
                height,
                num_frames,
                loop_input,
                seed,
                color_correction_method,
                raw_request.app.state.seedvr2_long_port,
                raw_request.headers.get("authorization", ""),
            )
        )
    return {"id": job_id, "status": "queued", "max_frames": MAX_FRAMES}


@router.get("/v1/seedvr2/restore-long/{job_id}")
async def get_long_video(job_id: str) -> dict[str, str | int]:
    status = _job_dir(job_id) / "status.json"
    if not status.exists():
        raise HTTPException(404, "SeedVR2 long-video job not found")
    record = json.loads(status.read_text())
    if record["status"] in {"queued", "running"} and record.get("pid") != os.getpid():
        # Only this process runs jobs, so another owner means a restart lost it.
        record["status"], record["error"] = "failed", "SeedVR2 long-video job was interrupted by a server restart"
    return record


@router.get("/v1/seedvr2/restore-long/{job_id}/content")
async def download_long_video(job_id: str) -> FileResponse:
    job = _job_dir(job_id)
    if not (job / "result.json").exists():
        raise HTTPException(404, "SeedVR2 long-video output is not ready")
    return FileResponse(job / "output.mp4", media_type="video/mp4", filename=f"{job_id}.mp4")


@router.delete("/v1/seedvr2/restore-long/{job_id}", status_code=202)
async def cancel_long_video(job_id: str) -> dict[str, str]:
    """Ask the running job to stop; it settles at the next window boundary."""
    job = _job_dir(job_id)
    status = job / "status.json"
    if not status.exists():
        raise HTTPException(404, "SeedVR2 long-video job not found")
    record = json.loads(status.read_text())
    if record["status"] not in {"queued", "running"}:
        raise HTTPException(409, f"SeedVR2 long-video job already {record['status']}")
    (job / "cancel").touch()
    return {"id": job_id, "status": "cancelling"}
