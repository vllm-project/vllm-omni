# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Measure reference-upload persistence in the current checkout, without a model.

Run from either the base or candidate checkout with the same Python environment.
Memory and timing are separate runs. Synthetic inputs are not encoded media;
this measures neither network uploads, media decoding, generation, nor RSS.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import UploadFile

MIB = 1024 * 1024
HELPERS = Path("vllm_omni/entrypoints/openai/video/generation/helpers.py")


@dataclass
class Sample:
    trial: int
    peak_extra_python_bytes: int | None = None
    helper_seconds: float | None = None
    completion_through_file_fsync_seconds: float | None = None
    helper_io_delta: dict[str, int] = field(default_factory=dict)
    completion_io_delta: dict[str, int] = field(default_factory=dict)
    contents_verified: bool = False
    temporary_files_cleaned: bool = False


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(MIB), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _io_counters() -> dict[str, int]:
    path = Path("/proc/self/io")
    if not path.exists():
        return {}
    return {key: int(value) for key, value in (line.split(":", 1) for line in path.read_text().splitlines())}


def _io_delta(before: dict[str, int]) -> dict[str, int]:
    return {key: value - before[key] for key, value in _io_counters().items()}


def _filesystem(path: Path) -> dict[str, object]:
    result: dict[str, object] = {"path": str(path), "device": path.stat().st_dev}
    if shutil.which("findmnt"):
        mount = subprocess.run(
            ["findmnt", "-J", "-T", str(path), "-o", "SOURCE,FSTYPE,TARGET"],
            capture_output=True,
            text=True,
            check=False,
        )
        if mount.returncode == 0:
            result["mount"] = json.loads(mount.stdout)
    return result


def _prepare_uploads(args: argparse.Namespace, directory: Path) -> tuple[list[UploadFile], list[dict[str, object]]]:
    from fastapi import UploadFile
    from starlette.datastructures import Headers

    uploads: list[UploadFile] = []
    expected: list[dict[str, object]] = []
    suffix, mime = (".mp4", "video/mp4") if args.media == "video" else (".wav", "audio/wav")
    try:
        for index in range(args.count):
            source = tempfile.SpooledTemporaryFile(max_size=MIB, mode="w+b", dir=str(directory))
            try:
                pattern = bytes((position + 17 * index + 31) % 251 for position in range(4096))
                block = pattern * (MIB // len(pattern))
                digest = hashlib.sha256()
                for _ in range(args.size_mib):
                    source.write(block)
                    digest.update(block)
                source.rollover()
                uploads.append(
                    UploadFile(
                        file=source,
                        filename=f"reference-{index}{suffix}",
                        size=args.size_mib * MIB,
                        headers=Headers({"content-type": mime}),
                    )
                )
                expected.append({"size_bytes": args.size_mib * MIB, "sha256": digest.hexdigest()})
            except BaseException:
                source.close()
                raise
    except BaseException:
        for upload in uploads:
            upload.file.close()
        raise
    return uploads, expected


async def _run(args: argparse.Namespace, checkout: Path) -> dict[str, object]:
    from vllm_omni.entrypoints.openai.video.generation import helpers

    source = checkout / HELPERS
    if Path(helpers.__file__).resolve() != source:
        raise RuntimeError(f"Imported a different checkout: {helpers.__file__}")
    source_hash = _digest(source)
    packages = ("torch", "vllm", "vllm-omni", "fastapi", "starlette", "anyio")
    versions = {name: importlib.metadata.version(name) for name in packages}
    result: dict[str, object] = {
        "command": [sys.executable, *sys.argv],
        "checkout": str(checkout),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "production_file": str(HELPERS),
        "production_sha256": source_hash,
        "benchmark_sha256": _digest(Path(__file__).resolve()),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "packages": versions,
        },
        "input_storage": "UploadFile backed by SpooledTemporaryFile(max_size=1 MiB), explicitly rolled to disk",
        "limitations": [
            "Synthetic bytes; persistence after multipart parsing, not network, media decoding or model generation.",
            "Memory is the extra traced Python allocation peak during the helper call, not RSS or total server memory.",
            "Imports, inputs, cache preparation, warmups, output hashing and cleanup are outside measurement.",
            "Time mode disables tracemalloc; optional output fsync is additional benchmark work, not production behavior.",
            "Cold advises eviction of owned source files only; read_bytes records actual process-charged storage reads.",
            "Shared-host and device caches are uncontrolled; file fsync does not establish directory crash durability.",
        ],
    }
    samples: list[Sample] = []
    with tempfile.TemporaryDirectory(prefix="reference-upload-benchmark-", dir=args.io_dir) as owned:
        directory = Path(owned)
        result["filesystem"] = _filesystem(directory)
        input_directory, output_directory = directory / "inputs", directory / "outputs"
        input_directory.mkdir()
        output_directory.mkdir()
        uploads, expected = _prepare_uploads(args, input_directory)
        result["expected_files"] = expected
        previous_tempdir = tempfile.tempdir
        tempfile.tempdir = str(output_directory)
        try:
            for trial in range(1 - args.warmups, args.repetitions + 1):
                if _digest(source) != source_hash:
                    raise RuntimeError("Production source changed during measurement")
                for upload in uploads:
                    upload.file.flush()
                    os.fsync(upload.file.fileno())
                    if args.input_cache == "cold":
                        os.posix_fadvise(upload.file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                    else:
                        upload.file.seek(0)
                        while upload.file.read(MIB):
                            pass
                    await upload.seek(0)
                gc.collect()
                sample = Sample(trial=trial)
                before_io = _io_counters()
                tracing = args.mode == "memory" and trial > 0
                if tracing:
                    tracemalloc.start()
                    traced_start, _ = tracemalloc.get_traced_memory()
                    tracemalloc.reset_peak()
                started = time.perf_counter_ns() if args.mode == "time" else 0
                try:
                    images, videos, audios = await helpers._persist_uploaded_media_references(uploads)
                finally:
                    if tracing:
                        _, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        sample.peak_extra_python_bytes = max(0, peak - traced_start)
                if args.mode == "time":
                    sample.helper_seconds = (time.perf_counter_ns() - started) / 1e9
                sample.helper_io_delta = _io_delta(before_io)
                paths = videos if args.media == "video" else audios
                if images or (audios if args.media == "video" else videos) or len(paths) != len(expected):
                    raise RuntimeError("Incorrect output classification or file count")
                if any(not Path(path).resolve().is_relative_to(output_directory) for path in paths):
                    raise RuntimeError("Helper returned a path outside the owned output directory")
                if args.output_fsync:
                    for raw_path in paths:
                        with Path(raw_path).open("r+b") as output:
                            os.fsync(output.fileno())
                    sample.completion_through_file_fsync_seconds = (time.perf_counter_ns() - started) / 1e9
                sample.completion_io_delta = _io_delta(before_io)
                for raw_path, want in zip(paths, expected, strict=True):
                    path = Path(raw_path)
                    actual = {"size_bytes": path.stat().st_size, "sha256": _digest(path)}
                    if actual != want:
                        raise RuntimeError(f"Output contents differ for {path.name}")
                    path.unlink()
                sample.contents_verified = True
                sample.temporary_files_cleaned = not any(output_directory.iterdir())
                if not sample.temporary_files_cleaned:
                    raise RuntimeError("Unexpected files remain after helper output cleanup")
                if trial > 0:
                    samples.append(sample)
        finally:
            for upload in uploads:
                await upload.close()
            tempfile.tempdir = previous_tempdir
    if _digest(source) != source_hash:
        raise RuntimeError("Production source changed during measurement")
    metrics = ("peak_extra_python_bytes", "helper_seconds", "completion_through_file_fsync_seconds")
    summary = {}
    for metric in metrics:
        values = [value for sample in samples if (value := getattr(sample, metric)) is not None]
        if values:
            summary[metric] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
    result.update(
        status="complete",
        samples=[asdict(sample) for sample in samples],
        summary=summary,
        source_unchanged=True,
        contents_verified=True,
        temporary_directory_removed=not directory.exists(),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("memory", "time"), required=True)
    parser.add_argument("--media", choices=("video", "audio"), required=True)
    parser.add_argument("--size-mib", type=int, required=True)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--io-dir",
        type=Path,
        default=Path(tempfile.gettempdir()),
        help="Existing input/output parent; choose the filesystem to measure",
    )
    parser.add_argument("--input-cache", choices=("warm", "cold"), default="warm")
    parser.add_argument(
        "--output-fsync", action="store_true", help="Time mode only: also report completion through output-file fsync"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New JSON file; existing results are never overwritten"
    )
    args = parser.parse_args()
    if not 1 <= args.size_mib <= (50 if args.media == "video" else 15):
        parser.error("--size-mib must be 1..50 for video or 1..15 for audio")
    if not 1 <= args.count <= 12 or args.repetitions < 1 or args.warmups < 0:
        parser.error("--count must be 1..12, --repetitions positive, and --warmups non-negative")
    if not args.io_dir.is_dir() or not args.output.parent.is_dir() or args.output.exists():
        parser.error("--io-dir and output parent must exist; --output must be a new file")
    if args.mode == "memory" and args.output_fsync:
        parser.error("--output-fsync requires --mode time")
    if args.input_cache == "cold" and not hasattr(os, "posix_fadvise"):
        parser.error("--input-cache cold requires POSIX_FADV_DONTNEED support")
    args.io_dir = args.io_dir.resolve()
    args.output = args.output.resolve()
    checkout = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()).resolve()
    sys.path.insert(0, str(checkout))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    result = asyncio.run(_run(args, checkout))
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
