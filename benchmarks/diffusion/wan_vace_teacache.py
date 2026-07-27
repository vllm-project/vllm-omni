#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reproducible Wan VACE TeaCache serving and quality benchmark.

The cache backend is selected when the server starts, so run ``request`` once
per server configuration (no cache, each TeaCache threshold, and Cache-DiT).
The ``quality`` subcommand compares measured clips with the same-seed no-cache
output, no-cache repeat runs, and the source video.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from urllib.parse import urljoin, urlparse

import numpy as np
import requests

_LABEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SERVER_SYNC_TIMEOUT_ENV = "VLLM_OMNI_VIDEO_SYNC_TIMEOUT"
_SERVER_SYNC_TIMEOUT_GRACE_S = 60.0


@dataclass(frozen=True)
class GenerationConfig:
    prompt: str
    negative_prompt: str
    width: int
    height: int
    num_frames: int
    fps: int
    num_inference_steps: int
    guidance_scale: float
    boundary_ratio: float
    flow_shift: float
    seed: int
    model: str | None = None

    def form_fields(self) -> dict[str, str]:
        fields = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "width": str(self.width),
            "height": str(self.height),
            "num_frames": str(self.num_frames),
            "fps": str(self.fps),
            "num_inference_steps": str(self.num_inference_steps),
            "guidance_scale": str(self.guidance_scale),
            "boundary_ratio": str(self.boundary_ratio),
            "flow_shift": str(self.flow_shift),
            "seed": str(self.seed),
        }
        if self.model:
            fields["model"] = self.model
        return fields


@dataclass(frozen=True)
class RequestMeasurement:
    phase: str
    index: int
    wall_time_ms: float
    server_inference_time_ms: float | None
    server_stage_durations_ms: dict[str, float] | None
    server_peak_memory_mb: float | None
    server_model: str | None
    request_id: str | None
    response_bytes: int
    sha256: str
    output_path: str


@dataclass(frozen=True)
class ServerConfiguration:
    label: str
    cache_arguments: tuple[str, ...]


def _server_configurations() -> tuple[ServerConfiguration, ...]:
    """Return the fixed comparison matrix for issue #5079."""
    return (
        ServerConfiguration("no_cache", ("--cache-backend", "none")),
        ServerConfiguration(
            "tea_0_1",
            ("--cache-backend", "tea_cache", "--cache-config", '{"rel_l1_thresh":0.1}'),
        ),
        ServerConfiguration(
            "tea_0_2",
            ("--cache-backend", "tea_cache", "--cache-config", '{"rel_l1_thresh":0.2}'),
        ),
        ServerConfiguration(
            "cache_dit",
            (
                "--cache-backend",
                "cache_dit",
                "--cache-config",
                '{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4}',
            ),
        ),
    )


def _package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _run_git(*arguments: str) -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ("git", *arguments),
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _environment_metadata() -> dict[str, object]:
    git_status = _run_git("status", "--porcelain")
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name) for name in ("torch", "torch-npu", "vllm", "vllm-omni", "transformers")
        },
        "git": {
            "commit": _run_git("rev-parse", "HEAD"),
            "dirty": bool(git_status) if git_status is not None else None,
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_optional_float(raw_value: str | None) -> float | None:
    if raw_value is None:
        return None
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"Expected a numeric response header, got {raw_value!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Expected a finite response header, got {raw_value!r}")
    return value


def _parse_stage_durations_ms(raw_value: str | None) -> dict[str, float] | None:
    if raw_value is None:
        return None
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid X-Stage-Durations JSON: {raw_value!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("X-Stage-Durations must contain a JSON object")

    durations: dict[str, float] = {}
    for stage_name, duration_s in parsed.items():
        if not isinstance(stage_name, str) or isinstance(duration_s, bool) or not isinstance(duration_s, int | float):
            raise ValueError("X-Stage-Durations must map string stage names to numeric durations")
        duration_ms = float(duration_s) if stage_name.endswith("_ms") else float(duration_s) * 1000.0
        if not math.isfinite(duration_ms) or duration_ms < 0:
            raise ValueError(f"Invalid stage duration for {stage_name!r}: {duration_s!r}")
        durations[stage_name] = duration_ms
    return durations


def _validate_label(label: str) -> str:
    if not _LABEL_PATTERN.fullmatch(label):
        raise ValueError(f"Label must match {_LABEL_PATTERN.pattern!r}, got {label!r}")
    return label


def _build_endpoint(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"base URL must be an absolute HTTP(S) URL, got {base_url!r}")
    return urljoin(base_url.rstrip("/") + "/", "v1/videos/sync")


def _request_once(
    *,
    endpoint: str,
    input_video: Path,
    config: GenerationConfig,
    timeout_s: float,
    phase: str,
    index: int,
    output_path: Path,
) -> RequestMeasurement:
    start_time = time.perf_counter()
    with input_video.open("rb") as video_file:
        response = requests.post(
            endpoint,
            data=config.form_fields(),
            files={"input_reference": (input_video.name, video_file, "video/mp4")},
            headers={"Accept": "video/mp4"},
            timeout=timeout_s,
        )
    wall_time_ms = (time.perf_counter() - start_time) * 1000.0
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if not content_type.startswith("video/"):
        raise RuntimeError(f"Expected a video response, got content-type={content_type!r}")
    if not response.content:
        raise RuntimeError("Video endpoint returned an empty response body")

    output_path.write_bytes(response.content)
    server_inference_time_s = _parse_optional_float(response.headers.get("X-Inference-Time-S"))
    return RequestMeasurement(
        phase=phase,
        index=index,
        wall_time_ms=wall_time_ms,
        server_inference_time_ms=(server_inference_time_s * 1000.0 if server_inference_time_s is not None else None),
        server_stage_durations_ms=_parse_stage_durations_ms(response.headers.get("X-Stage-Durations")),
        server_peak_memory_mb=_parse_optional_float(response.headers.get("X-Peak-Memory-MB")),
        server_model=response.headers.get("X-Model"),
        request_id=response.headers.get("X-Request-Id"),
        response_bytes=len(response.content),
        sha256=hashlib.sha256(response.content).hexdigest(),
        output_path=str(output_path),
    )


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("At least one measured value is required")
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _summarize_measurements(measurements: list[RequestMeasurement]) -> dict[str, object]:
    measured = [measurement for measurement in measurements if measurement.phase == "measured"]
    result: dict[str, object] = {
        "wall_time_ms": _summary([measurement.wall_time_ms for measurement in measured]),
    }

    server_times = [
        measurement.server_inference_time_ms
        for measurement in measured
        if measurement.server_inference_time_ms is not None
    ]
    if server_times:
        result["server_inference_time_ms"] = _summary(server_times)

    peak_memory = [
        measurement.server_peak_memory_mb for measurement in measured if measurement.server_peak_memory_mb is not None
    ]
    if peak_memory:
        result["server_peak_memory_mb"] = _summary(peak_memory)

    stage_names = sorted(
        {stage_name for measurement in measured for stage_name in (measurement.server_stage_durations_ms or {})}
    )
    if stage_names:
        result["server_stage_durations_ms"] = {
            stage_name: _summary(
                [
                    measurement.server_stage_durations_ms[stage_name]
                    for measurement in measured
                    if measurement.server_stage_durations_ms is not None
                    and stage_name in measurement.server_stage_durations_ms
                ]
            )
            for stage_name in stage_names
        }
    return result


def _validate_request_args(args: argparse.Namespace) -> None:
    if args.warmup < 1 or args.runs < 3:
        raise ValueError("A reportable benchmark requires at least 1 warmup and 3 measured runs")
    for field_name in ("width", "height", "num_frames", "fps", "num_inference_steps"):
        if getattr(args, field_name) <= 0:
            raise ValueError(f"{field_name} must be positive")
    if args.guidance_scale < 0 or args.flow_shift <= 0 or args.timeout <= 0:
        raise ValueError("guidance_scale must be non-negative; flow_shift and timeout must be positive")
    if not 0 < args.boundary_ratio <= 1:
        raise ValueError("boundary_ratio must be in (0, 1]")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")


def run_request_benchmark(args: argparse.Namespace) -> None:
    _validate_request_args(args)
    label = _validate_label(args.label)
    input_video = Path(args.input_video).expanduser().resolve()
    if not input_video.is_file():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    output_dir = Path(args.output_dir).expanduser().resolve() / label
    output_dir.mkdir(parents=True, exist_ok=True)
    endpoint = _build_endpoint(args.base_url)
    config = GenerationConfig(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        fps=args.fps,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        boundary_ratio=args.boundary_ratio,
        flow_shift=args.flow_shift,
        seed=args.seed,
        model=args.model,
    )

    measurements: list[RequestMeasurement] = []
    for phase, count in (("warmup", args.warmup), ("measured", args.runs)):
        for index in range(1, count + 1):
            output_path = output_dir / f"{phase}_{index:02d}.mp4"
            measurement = _request_once(
                endpoint=endpoint,
                input_video=input_video,
                config=config,
                timeout_s=args.timeout,
                phase=phase,
                index=index,
                output_path=output_path,
            )
            measurements.append(measurement)
            print(
                f"{phase} {index}/{count}: wall={measurement.wall_time_ms:.1f} ms "
                f"server={measurement.server_inference_time_ms} ms output={output_path}"
            )

    manifest = {
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "label": label,
        "endpoint": endpoint,
        "server": {
            "command": args.server_command,
            "hardware": args.server_hardware,
            "software": args.server_software,
        },
        "client_command": getattr(args, "client_command", None) or shlex.join(sys.argv),
        "input_video": str(input_video),
        "input_video_sha256": _sha256_file(input_video),
        "generation_config": asdict(config),
        "client_environment": _environment_metadata(),
        "measurements": [asdict(measurement) for measurement in measurements],
        "summary": _summarize_measurements(measurements),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=2))
    print(f"Manifest: {manifest_path}")


def _build_server_command(args: argparse.Namespace, configuration: ServerConfiguration) -> list[str]:
    command_prefix = shlex.split(args.serve_command)
    if not command_prefix:
        raise ValueError("serve_command must not be empty")
    extra_arguments = shlex.split(args.server_args)
    return [
        *command_prefix,
        args.model,
        "--omni",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--enable-diffusion-pipeline-profiler",
        "--log-stats",
        *configuration.cache_arguments,
        *extra_arguments,
    ]


def _server_sync_timeout_s(args: argparse.Namespace) -> float:
    """Keep the server alive slightly longer than the HTTP client timeout."""
    return args.timeout + _SERVER_SYNC_TIMEOUT_GRACE_S


def _server_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    environment[_SERVER_SYNC_TIMEOUT_ENV] = str(_server_sync_timeout_s(args))
    return environment


def _server_command_text(args: argparse.Namespace, server_command: list[str]) -> str:
    """Render the command together with the required sync-endpoint timeout."""
    return shlex.join(
        [
            "env",
            f"{_SERVER_SYNC_TIMEOUT_ENV}={_server_sync_timeout_s(args)}",
            *server_command,
        ]
    )


def _build_equivalent_request_command(
    args: argparse.Namespace,
    configuration: ServerConfiguration,
    server_command: list[str],
) -> str:
    script_path = Path(__file__).resolve()
    command = [
        sys.executable,
        str(script_path),
        "request",
        "--base-url",
        f"http://{args.client_host}:{args.port}",
        "--input-video",
        str(Path(args.input_video).expanduser().resolve()),
        "--prompt",
        args.prompt,
        "--negative-prompt",
        args.negative_prompt,
        "--label",
        configuration.label,
        "--output-dir",
        str(Path(args.output_dir).expanduser().resolve()),
        "--server-command",
        _server_command_text(args, server_command),
        "--server-hardware",
        args.server_hardware,
        "--model",
        args.model,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--num-frames",
        str(args.num_frames),
        "--fps",
        str(args.fps),
        "--num-inference-steps",
        str(args.num_inference_steps),
        "--guidance-scale",
        str(args.guidance_scale),
        "--boundary-ratio",
        str(args.boundary_ratio),
        "--flow-shift",
        str(args.flow_shift),
        "--seed",
        str(args.seed),
        "--warmup",
        str(args.warmup),
        "--runs",
        str(args.runs),
        "--timeout",
        str(args.timeout),
    ]
    for software_item in args.server_software:
        command.extend(("--server-software", software_item))
    return shlex.join(command)


def _health_url(args: argparse.Namespace) -> str:
    return f"http://{args.client_host}:{args.port}/health"


def _server_is_healthy(health_url: str) -> bool:
    try:
        response = requests.get(health_url, timeout=5.0)
    except requests.RequestException:
        return False
    return response.status_code == 200


def _tail_text(path: Path, byte_count: int = 16_384) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as log_file:
        log_file.seek(0, os.SEEK_END)
        size = log_file.tell()
        log_file.seek(max(0, size - byte_count))
        return log_file.read().decode("utf-8", errors="replace")


def _wait_for_server(
    process: subprocess.Popen[bytes],
    *,
    health_url: str,
    timeout_s: float,
    log_path: Path,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"Server exited with status {return_code} before becoming healthy. Log tail:\n{_tail_text(log_path)}"
            )
        if _server_is_healthy(health_url):
            return
        time.sleep(2.0)
    raise TimeoutError(f"Server did not become healthy within {timeout_s:.1f}s. Log tail:\n{_tail_text(log_path)}")


def _stop_server(process: subprocess.Popen[bytes], timeout_s: float) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        # The server may exit between ``poll`` and ``killpg``. There is no
        # process group left to stop in that case.
        return
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=timeout_s)


def _matrix_request_namespace(
    args: argparse.Namespace,
    configuration: ServerConfiguration,
    server_command: list[str],
) -> argparse.Namespace:
    return argparse.Namespace(
        base_url=f"http://{args.client_host}:{args.port}",
        input_video=args.input_video,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        label=configuration.label,
        output_dir=args.output_dir,
        server_command=_server_command_text(args, server_command),
        server_hardware=args.server_hardware,
        server_software=args.server_software,
        model=args.model,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        fps=args.fps,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        boundary_ratio=args.boundary_ratio,
        flow_shift=args.flow_shift,
        seed=args.seed,
        warmup=args.warmup,
        runs=args.runs,
        timeout=args.timeout,
        client_command=_build_equivalent_request_command(args, configuration, server_command),
    )


def _quality_namespace(args: argparse.Namespace) -> argparse.Namespace:
    output_dir = Path(args.output_dir).expanduser().resolve()
    baseline_dir = output_dir / "no_cache"
    return argparse.Namespace(
        source_video=args.input_video,
        baseline_video=str(baseline_dir / "measured_01.mp4"),
        baseline_repeat=[
            f"no_cache_{index:02d}={baseline_dir / f'measured_{index:02d}.mp4'}" for index in range(2, args.runs + 1)
        ],
        candidate=[
            f"{configuration.label}={output_dir / configuration.label / 'measured_01.mp4'}"
            for configuration in _server_configurations()
            if configuration.label != "no_cache"
        ],
        output_json=str(output_dir / "quality.json"),
        sample_frames=args.sample_frames,
        pixel_resize=args.pixel_resize,
        dinov2_model=args.dinov2_model,
        device=args.quality_device,
        batch_size=args.batch_size,
    )


def _validate_matrix_args(args: argparse.Namespace) -> None:
    request_args = _matrix_request_namespace(
        args, _server_configurations()[0], _build_server_command(args, _server_configurations()[0])
    )
    _validate_request_args(request_args)
    if args.port <= 0 or args.port > 65535:
        raise ValueError("port must be in [1, 65535]")
    if args.server_start_timeout <= 0 or args.server_stop_timeout <= 0:
        raise ValueError("server start/stop timeouts must be positive")
    _validate_quality_args(_quality_namespace(args))


def run_server_matrix(args: argparse.Namespace) -> None:
    """Launch every fixed server configuration and collect comparable evidence."""
    _validate_matrix_args(args)
    configurations = _server_configurations()
    commands = [
        {
            "label": configuration.label,
            "server_command": _server_command_text(args, _build_server_command(args, configuration)),
            "request_command": _build_equivalent_request_command(
                args,
                configuration,
                _build_server_command(args, configuration),
            ),
        }
        for configuration in configurations
    ]
    if args.dry_run:
        print(json.dumps(commands, indent=2, ensure_ascii=False))
        return

    input_video = Path(args.input_video).expanduser().resolve()
    if not input_video.is_file():
        raise FileNotFoundError(f"Input video not found: {input_video}")
    health_url = _health_url(args)
    if _server_is_healthy(health_url):
        raise RuntimeError(f"A server is already healthy at {health_url}; stop it before running the matrix")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_manifest: dict[str, object] = {
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "commands": commands,
        "completed": [],
        "active": None,
    }
    matrix_manifest_path = output_dir / "matrix_manifest.json"
    matrix_manifest_path.write_text(
        json.dumps(matrix_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    for configuration in configurations:
        server_command = _build_server_command(args, configuration)
        configuration_dir = output_dir / configuration.label
        configuration_dir.mkdir(parents=True, exist_ok=True)
        server_log_path = configuration_dir / "server.log"
        matrix_manifest["active"] = configuration.label
        matrix_manifest_path.write_text(
            json.dumps(matrix_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Starting {configuration.label}: {shlex.join(server_command)}")
        with server_log_path.open("wb") as server_log:
            process = subprocess.Popen(
                server_command,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=_server_environment(args),
            )
            try:
                _wait_for_server(
                    process,
                    health_url=health_url,
                    timeout_s=args.server_start_timeout,
                    log_path=server_log_path,
                )
                run_request_benchmark(_matrix_request_namespace(args, configuration, server_command))
            finally:
                _stop_server(process, args.server_stop_timeout)

        completed = matrix_manifest["completed"]
        assert isinstance(completed, list)
        completed.append(configuration.label)
        matrix_manifest["active"] = None
        matrix_manifest_path.write_text(
            json.dumps(matrix_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    if not args.skip_quality:
        run_quality_comparison(_quality_namespace(args))


def _load_sampled_frames(video_path: Path, sample_count: int) -> list[np.ndarray]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Quality comparison requires opencv-python") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    actual_count = min(sample_count, len(frames))
    indices = np.linspace(0, len(frames) - 1, num=actual_count).round().astype(np.int64)
    return [frames[int(index)] for index in indices]


def _resize_frame(frame: np.ndarray, size: int) -> np.ndarray:
    import cv2

    resized = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def _frame_pearson(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_flat = reference.reshape(-1).astype(np.float64)
    candidate_flat = candidate.reshape(-1).astype(np.float64)
    reference_centered = reference_flat - reference_flat.mean()
    candidate_centered = candidate_flat - candidate_flat.mean()
    denominator = np.linalg.norm(reference_centered) * np.linalg.norm(candidate_centered)
    if denominator == 0:
        return 1.0 if np.array_equal(reference_flat, candidate_flat) else 0.0
    return float(np.dot(reference_centered, candidate_centered) / denominator)


def _pixel_metrics(
    reference_frames: list[np.ndarray],
    candidate_frames: list[np.ndarray],
    resize: int,
) -> dict[str, object]:
    pair_count = min(len(reference_frames), len(candidate_frames))
    if pair_count == 0:
        raise ValueError("At least one aligned frame pair is required")

    absolute_differences: list[float] = []
    correlations: list[float] = []
    for reference_frame, candidate_frame in zip(
        reference_frames[:pair_count],
        candidate_frames[:pair_count],
        strict=True,
    ):
        reference = _resize_frame(reference_frame, resize)
        candidate = _resize_frame(candidate_frame, resize)
        absolute_differences.append(float(np.abs(reference - candidate).mean()))
        correlations.append(_frame_pearson(reference, candidate))

    return {
        "frame_count": pair_count,
        "mean_absolute_pixel_diff": _summary(absolute_differences),
        "frame_pearson": _summary(correlations),
    }


def _build_dinov2_embedder(
    *,
    model_name_or_path: str,
    device: str,
    batch_size: int,
) -> Callable[[list[np.ndarray]], np.ndarray]:
    import torch
    import torch.nn.functional as torch_functional
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path).to(device).eval()

    def embed(frames: list[np.ndarray]) -> np.ndarray:
        embeddings: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(frames), batch_size):
                batch = frames[start : start + batch_size]
                inputs = processor(images=batch, return_tensors="pt")
                inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
                outputs = model(**inputs)
                features = outputs.last_hidden_state[:, 0]
                features = torch_functional.normalize(features.float(), dim=1)
                embeddings.append(features.cpu().numpy())
        return np.concatenate(embeddings, axis=0)

    return embed


def _embedding_similarity(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    pair_count = min(reference.shape[0], candidate.shape[0])
    if pair_count == 0:
        raise ValueError("At least one aligned embedding pair is required")
    similarities = np.sum(reference[:pair_count] * candidate[:pair_count], axis=1).tolist()
    return _summary([float(similarity) for similarity in similarities])


def _parse_labeled_videos(candidate_specs: list[str], *, field_name: str) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    seen_labels: set[str] = set()
    for candidate_spec in candidate_specs:
        label, separator, raw_path = candidate_spec.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError(f"{field_name} must use LABEL=/path/to/video.mp4 syntax, got {candidate_spec!r}")
        label = _validate_label(label)
        if label in seen_labels:
            raise ValueError(f"Duplicate {field_name} label: {label!r}")
        candidate_path = Path(raw_path).expanduser().resolve()
        if not candidate_path.is_file():
            raise FileNotFoundError(f"{field_name} video not found: {candidate_path}")
        candidates.append((label, candidate_path))
        seen_labels.add(label)
    return candidates


def _validate_quality_args(args: argparse.Namespace) -> None:
    for field_name in ("sample_frames", "pixel_resize", "batch_size"):
        if getattr(args, field_name) <= 0:
            raise ValueError(f"{field_name} must be positive")


def run_quality_comparison(args: argparse.Namespace) -> None:
    _validate_quality_args(args)
    source_video = Path(args.source_video).expanduser().resolve()
    baseline_video = Path(args.baseline_video).expanduser().resolve()
    for video_path in (source_video, baseline_video):
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")

    candidates = _parse_labeled_videos(args.candidate, field_name="candidate")
    baseline_repeats = _parse_labeled_videos(args.baseline_repeat, field_name="baseline repeat")
    all_paths = dict.fromkeys(
        [source_video, baseline_video, *(path for _, path in baseline_repeats), *(path for _, path in candidates)]
    )
    frames_by_path = {video_path: _load_sampled_frames(video_path, args.sample_frames) for video_path in all_paths}

    source_frames = frames_by_path[source_video]
    baseline_frames = frames_by_path[baseline_video]
    baseline_vs_source: dict[str, object] = {
        "pixel": _pixel_metrics(source_frames, baseline_frames, args.pixel_resize),
    }
    baseline_repeat_report: dict[str, object] = {}
    candidate_report: dict[str, object] = {}

    embed: Callable[[list[np.ndarray]], np.ndarray] | None = None
    embeddings_by_path: dict[Path, np.ndarray] = {}
    if args.dinov2_model:
        embed = _build_dinov2_embedder(
            model_name_or_path=args.dinov2_model,
            device=args.device,
            batch_size=args.batch_size,
        )
        embeddings_by_path = {video_path: embed(frames) for video_path, frames in frames_by_path.items()}
        baseline_vs_source["dinov2_cosine"] = _embedding_similarity(
            embeddings_by_path[source_video],
            embeddings_by_path[baseline_video],
        )

    for label, repeat_path in baseline_repeats:
        repeat_frames = frames_by_path[repeat_path]
        repeat_vs_primary: dict[str, object] = {
            "pixel": _pixel_metrics(baseline_frames, repeat_frames, args.pixel_resize),
        }
        repeat_vs_source: dict[str, object] = {
            "pixel": _pixel_metrics(source_frames, repeat_frames, args.pixel_resize),
        }
        metrics: dict[str, object] = {
            "video": str(repeat_path),
            "sha256": _sha256_file(repeat_path),
            "vs_primary_no_cache": repeat_vs_primary,
            "vs_source": repeat_vs_source,
        }
        if embed is not None:
            repeat_vs_primary["dinov2_cosine"] = _embedding_similarity(
                embeddings_by_path[baseline_video],
                embeddings_by_path[repeat_path],
            )
            repeat_vs_source["dinov2_cosine"] = _embedding_similarity(
                embeddings_by_path[source_video],
                embeddings_by_path[repeat_path],
            )
        baseline_repeat_report[label] = metrics

    for label, candidate_path in candidates:
        candidate_frames = frames_by_path[candidate_path]
        candidate_vs_source: dict[str, object] = {
            "pixel": _pixel_metrics(source_frames, candidate_frames, args.pixel_resize),
        }
        candidate_vs_no_cache: dict[str, object] = {
            "pixel": _pixel_metrics(baseline_frames, candidate_frames, args.pixel_resize),
        }
        metrics = {
            "video": str(candidate_path),
            "sha256": _sha256_file(candidate_path),
            "vs_source": candidate_vs_source,
            "vs_no_cache": candidate_vs_no_cache,
        }
        if embed is not None:
            candidate_vs_source["dinov2_cosine"] = _embedding_similarity(
                embeddings_by_path[source_video],
                embeddings_by_path[candidate_path],
            )
            candidate_vs_no_cache["dinov2_cosine"] = _embedding_similarity(
                embeddings_by_path[baseline_video],
                embeddings_by_path[candidate_path],
            )
        candidate_report[label] = metrics

    report = {
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "source_video": str(source_video),
        "source_video_sha256": _sha256_file(source_video),
        "baseline_video": str(baseline_video),
        "baseline_video_sha256": _sha256_file(baseline_video),
        "sample_frames": args.sample_frames,
        "pixel_resize": args.pixel_resize,
        "pixel_metric_note": "frame_pearson is not assumed to equal issue #5079's unspecified corr metric",
        "dinov2_model": args.dinov2_model,
        "baseline_vs_source": baseline_vs_source,
        "baseline_repeatability": baseline_repeat_report,
        "candidates": candidate_report,
    }

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Quality report: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    request_parser = subparsers.add_parser("request", help="Send fixed V2V requests and record latency")
    request_parser.add_argument("--base-url", default="http://localhost:8090")
    request_parser.add_argument("--input-video", required=True)
    request_parser.add_argument("--prompt", default="A cat walking on a street, high quality video")
    request_parser.add_argument("--negative-prompt", default="low quality, blurry")
    request_parser.add_argument("--label", required=True, help="Configuration label, e.g. no_cache or tea_0_2")
    request_parser.add_argument("--output-dir", default="./wan_vace_teacache_results")
    request_parser.add_argument("--server-command", required=True, help="Exact command used to start this server")
    request_parser.add_argument("--server-hardware", required=True, help="Server accelerator model/count/VRAM")
    request_parser.add_argument(
        "--server-software",
        action="append",
        default=[],
        help="Repeat for server software details, e.g. torch=2.8.0",
    )
    request_parser.add_argument("--model")
    request_parser.add_argument("--width", type=int, default=1280)
    request_parser.add_argument("--height", type=int, default=736)
    request_parser.add_argument("--num-frames", type=int, default=61)
    request_parser.add_argument("--fps", type=int, default=16)
    request_parser.add_argument("--num-inference-steps", type=int, default=20)
    request_parser.add_argument("--guidance-scale", type=float, default=5.0)
    request_parser.add_argument("--boundary-ratio", type=float, default=0.875)
    request_parser.add_argument("--flow-shift", type=float, default=3.0)
    request_parser.add_argument("--seed", type=int, default=1)
    request_parser.add_argument("--warmup", type=int, default=1)
    request_parser.add_argument("--runs", type=int, default=3)
    request_parser.add_argument("--timeout", type=float, default=7200.0)
    request_parser.set_defaults(handler=run_request_benchmark)

    matrix_parser = subparsers.add_parser(
        "matrix",
        help="Launch no-cache, TeaCache, and Cache-DiT servers sequentially",
    )
    matrix_parser.add_argument("--model", required=True)
    matrix_parser.add_argument("--input-video", required=True)
    matrix_parser.add_argument("--output-dir", default="./wan_vace_teacache_results")
    matrix_parser.add_argument(
        "--serve-command",
        default="vllm serve",
        help="Command prefix used to launch serving, e.g. 'vllm serve' or 'vllm-omni serve'",
    )
    matrix_parser.add_argument(
        "--server-args",
        default="",
        help="Additional identical server arguments applied to every matrix entry",
    )
    matrix_parser.add_argument("--host", default="127.0.0.1", help="Server bind host")
    matrix_parser.add_argument("--client-host", default="127.0.0.1", help="Host used by health/request clients")
    matrix_parser.add_argument("--port", type=int, default=8090)
    matrix_parser.add_argument("--server-hardware", required=True, help="Accelerator model/count/VRAM")
    matrix_parser.add_argument(
        "--server-software",
        action="append",
        default=[],
        help="Repeat for server software details, e.g. torch=2.8.0",
    )
    matrix_parser.add_argument("--prompt", default="A cat walking on a street, high quality video")
    matrix_parser.add_argument("--negative-prompt", default="low quality, blurry")
    matrix_parser.add_argument("--width", type=int, default=1280)
    matrix_parser.add_argument("--height", type=int, default=736)
    matrix_parser.add_argument("--num-frames", type=int, default=61)
    matrix_parser.add_argument("--fps", type=int, default=16)
    matrix_parser.add_argument("--num-inference-steps", type=int, default=20)
    matrix_parser.add_argument("--guidance-scale", type=float, default=5.0)
    matrix_parser.add_argument("--boundary-ratio", type=float, default=0.875)
    matrix_parser.add_argument("--flow-shift", type=float, default=3.0)
    matrix_parser.add_argument("--seed", type=int, default=1)
    matrix_parser.add_argument("--warmup", type=int, default=1)
    matrix_parser.add_argument("--runs", type=int, default=3)
    matrix_parser.add_argument("--timeout", type=float, default=7200.0, help="Per-request timeout in seconds")
    matrix_parser.add_argument("--server-start-timeout", type=float, default=1800.0)
    matrix_parser.add_argument("--server-stop-timeout", type=float, default=60.0)
    matrix_parser.add_argument("--sample-frames", type=int, default=16)
    matrix_parser.add_argument("--pixel-resize", type=int, default=224)
    matrix_parser.add_argument("--dinov2-model", help="Optional Hugging Face model ID or local path")
    matrix_parser.add_argument("--quality-device", default="cpu")
    matrix_parser.add_argument("--batch-size", type=int, default=4)
    matrix_parser.add_argument("--skip-quality", action="store_true")
    matrix_parser.add_argument("--dry-run", action="store_true", help="Print exact commands without launching servers")
    matrix_parser.set_defaults(handler=run_server_matrix)

    quality_parser = subparsers.add_parser("quality", help="Compare cached clips with source and no-cache")
    quality_parser.add_argument("--source-video", required=True)
    quality_parser.add_argument("--baseline-video", required=True, help="Primary same-seed no-cache measured output")
    quality_parser.add_argument(
        "--baseline-repeat",
        action="append",
        default=[],
        help="Repeat LABEL=/path/to/no-cache.mp4 to measure no-cache self-run variance",
    )
    quality_parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Repeat LABEL=/path/to/measured.mp4 for each cache configuration",
    )
    quality_parser.add_argument("--output-json", default="./wan_vace_teacache_results/quality.json")
    quality_parser.add_argument("--sample-frames", type=int, default=16)
    quality_parser.add_argument("--pixel-resize", type=int, default=224)
    quality_parser.add_argument(
        "--dinov2-model",
        help="Optional Hugging Face model ID or local path, e.g. facebook/dinov2-base",
    )
    quality_parser.add_argument("--device", default="cpu")
    quality_parser.add_argument("--batch-size", type=int, default=4)
    quality_parser.set_defaults(handler=run_quality_comparison)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
