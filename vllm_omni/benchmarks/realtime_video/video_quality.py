# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Complete-video coverage and isolated native semantic quality measurements.

Scores use each suite's native scale and are custom-input measurements, never
a leaderboard total or a recorded-baseline accuracy acceptance verdict.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import regex as re

VBENCH_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
)
STANDARD_VBENCH_DIMENSIONS = (*VBENCH_DIMENSIONS, "temporal_flickering")
WBENCH_DIMENSIONS = (
    "aesthetic_quality",
    "imaging_quality",
    "temporal_flickering",
    "dynamic_degree",
    "motion_smoothness",
    "hpsv3_quality",
)
_SUITES = {
    "vbench": (STANDARD_VBENCH_DIMENSIONS, "custom_input"),
    "wbench": (WBENCH_DIMENSIONS, "video_quality"),
}


class VideoQualityError(RuntimeError):
    """Video coverage or evaluator output cannot support a valid comparison."""


def _capture_metadata(capture: Any, path: Path, cv2: Any) -> dict[str, Any]:
    if not capture.isOpened():
        raise VideoQualityError(f"Cannot open video: {path}")
    values = {
        "fps": capture.get(cv2.CAP_PROP_FPS),
        "frame_count": capture.get(cv2.CAP_PROP_FRAME_COUNT),
        "width": capture.get(cv2.CAP_PROP_FRAME_WIDTH),
        "height": capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
    }
    if any(not math.isfinite(value) or value <= 0 for value in values.values()):
        raise VideoQualityError(f"Empty video or invalid metadata: {path}: {values}")
    for key in ("frame_count", "width", "height"):
        if not values[key].is_integer():
            raise VideoQualityError(f"Non-integral video {key}: {path}")
        values[key] = int(values[key])
    return {"path": str(path), **values, "duration_s": values["frame_count"] / values["fps"]}


def verify_video_coverage(path: Path, *, frames: int, width: int, height: int, fps: int) -> dict:
    """Decode the complete input; semantic scorers cannot accept a partial rollout."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        metadata = _capture_metadata(capture, path, cv2)
        if tuple(metadata[key] for key in ("frame_count", "width", "height", "fps")) != (frames, width, height, fps):
            raise VideoQualityError("Video does not match the requested rollout")
        count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (height, width):
                raise VideoQualityError("Decoded frame geometry changed")
            count += 1
        if count != frames:
            raise VideoQualityError("Video does not decode to the complete frame population")
        return {**metadata, "decoded_frames": count}
    finally:
        capture.release()


# The worker runs only in the caller-selected evaluator environment. Inputs are
# JSON and argv, never interpolated into executable source or a shell command.
_EVALUATOR_WORKER = r"""
import json
import math
import pathlib
import sys
import traceback

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
repo = pathlib.Path(manifest["repo"])
root = pathlib.Path(manifest["run_dir"])
sys.path.insert(0, str(repo))
result = {"scores": {}, "errors": [], "native_provenance": {}}

def native_scalar(value):
    if type(value).__module__.split(".")[0] == "numpy" and getattr(value, "ndim", 0) == 0:
        return value.item()
    raise TypeError("Unsupported evaluator output type: " + type(value).__name__)

def strict_json(value):
    return json.dumps(value, indent=2, allow_nan=False, default=native_scalar)

try:
    background_stats = None
    if manifest["suite"] == "vbench" and "background_consistency" in manifest["dimensions"]:
        import hashlib
        import importlib.util
        adapter_path = pathlib.Path(manifest["encoder_adapter"])
        adapter_sha256 = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
        if adapter_sha256 != manifest["protocol"]["background_clip_inference"]["adapter_sha256"]:
            raise ValueError("Background CLIP inference adapter differs from the recorded source")
        # The isolated evaluator intentionally has its own Torch, without vLLM.
        # Loading this dependency-free file avoids executing Omni package init.
        spec = importlib.util.spec_from_file_location("video_encoder_adapter", adapter_path)
        if spec is None or spec.loader is None:
            raise ValueError("Cannot load background CLIP inference adapter")
        encoder_adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(encoder_adapter)
        background_stats = encoder_adapter.install_vbench_background_inference()
    if manifest["suite"] == "wbench":
        if "hpsv3_quality" in manifest["dimensions"]:
            # RAFT adds core/ (which contains datasets.py) to sys.path.
            # Resolve the Hugging Face package before that native side effect.
            import datasets
        from src.metrics.video_quality.evaluator import VideoQualityEvaluator
        evaluator = VideoQualityEvaluator(device="cuda", metrics=manifest["dimensions"])
    else:
        import torch
        from vbench import VBench as Evaluator
    for index, (label, video) in enumerate(manifest["videos"].items()):
        try:
            if background_stats is not None:
                for key in ("encoded_images", "forward_batches", "max_forward_batch"):
                    background_stats[key] = 0
            if manifest["suite"] == "wbench":
                raw = evaluator.evaluate(video, metrics=manifest["dimensions"])
                provenance = {}
                if "dynamic_degree" in manifest["dimensions"]:
                    metric = getattr(evaluator, "_metric_instances", {}).get("dynamic_degree")
                    provenance["dynamic_degree_backend"] = (
                        "raft" if metric.raft_model is not None else "farneback"
                    ) if hasattr(metric, "raft_model") else "unknown"
                result["native_provenance"][label] = provenance
                # HPSv3 can clamp a nonfinite raw mean to a finite final score.
                encoded = strict_json(raw)
                raw = json.loads(encoded)
                raw_path = root / (str(index) + "_wbench_raw.json")
                raw_path.write_text(encoded)
                if pathlib.Path(raw["video_path"]).resolve() != pathlib.Path(video).resolve():
                    raise ValueError("WBench returned results for a different video")
                if set(raw["metrics"]) != set(manifest["dimensions"]):
                    raise ValueError("Unexpected dimension coverage in WBench output")
                failures = {dim: row["error"] for dim, row in raw["metrics"].items() if "error" in row}
                if failures:
                    raise ValueError("WBench metric failures: " + strict_json(failures))
                scores = {dim: raw["metrics"][dim]["score"] for dim in manifest["dimensions"]}
            else:
                stage = root / ("video_" + str(index))
                stage.mkdir()
                staged = stage / ("rollout-0" + pathlib.Path(video).suffix.lower())
                staged.symlink_to(video)
                output = root / ("raw_" + str(index))
                output.mkdir()
                evaluator = Evaluator(torch.device("cuda"), str(repo / "vbench/VBench_full_info.json"), str(output))
                kwargs = {"imaging_quality_preprocessing_mode": "longer"}
                evaluator.evaluate(
                    str(stage), "result", dimension_list=manifest["dimensions"], local=True,
                    mode=manifest["mode"], **kwargs,
                )
                raw = json.loads((output / "result_eval_results.json").read_text())
                strict_json(raw)
                if set(raw) != set(manifest["dimensions"]):
                    raise ValueError("Unexpected dimension coverage in VBench output")
                if any(not isinstance(value, list) or len(value) < 2 or not value[1] for value in raw.values()):
                    raise ValueError("Missing per-video detail in VBench output")
                expected = {staged.resolve()}
                for dimension, value in raw.items():
                    covered = []
                    if not isinstance(value[1], list):
                        raise ValueError("Unexpected VBench per-video result format")
                    for row in value[1]:
                        path = pathlib.Path(row["video_path"]).resolve()
                        score = row["video_results"]
                        if not isinstance(score, (float, int)) or not math.isfinite(score):
                            raise ValueError("Invalid VBench per-video score")
                        matches = [clip for clip in expected if path == clip or clip in path.parents]
                        if len(matches) != 1:
                            raise ValueError("VBench returned detail for an unexpected video or clip")
                        covered.append(matches[0])
                    if len(covered) != len(expected) or set(covered) != expected:
                        raise ValueError("VBench skipped or duplicated a video or clip")
                scores = {dim: raw[dim][0] for dim in manifest["dimensions"]}
            result["scores"][label] = scores
        except Exception as exc:
            traceback.print_exc()
            result["errors"].append(label + ": " + type(exc).__name__ + ": " + str(exc))
        if background_stats is not None:
            result["native_provenance"][label] = {"background_clip_inference": dict(background_stats)}
        (root / "worker_result.json").write_text(strict_json(result))
except Exception as exc:
    traceback.print_exc()
    result["errors"].append(type(exc).__name__ + ": " + str(exc))
(root / "worker_result.json").write_text(strict_json(result))
"""


def _run_worker(command: list[str], repo: Path, log: Path, timeout_s: float) -> int:
    with log.open("w") as handle:
        process = subprocess.Popen(command, cwd=repo, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise


def evaluate_video_quality(
    videos: dict[str, Path],
    output_dir: Path,
    *,
    suite: str,
    repo: Path,
    python: Path,
    dimensions: list[str],
    timeout_s: float = 5400.0,
) -> dict[str, Any]:
    """Evaluate every video in a separate environment and validate exact coverage.

    ``status=passed`` means valid evaluation output, not quality acceptance.
    WBench is its six-metric video-quality subset; no VLM/API metrics are run.
    """
    report: dict[str, Any] = {
        "status": "failed",
        "suite": suite,
        "dimensions": list(dimensions),
        "scores": {},
        "errors": [],
        "native_provenance": {},
        "limitations": (
            [
                "WBench dynamic_degree may fall back from RAFT to Farneback. The effective backend is recorded when "
                "exposed by the evaluator, otherwise unknown; thresholds require calibration for that backend."
            ]
            if suite == "wbench"
            else []
        ),
        "repo_revision": None,
        "scope": (
            "Reference-free custom-input scores on native scales; not an official benchmark total or quality approval."
        ),
    }
    try:
        if suite not in _SUITES:
            raise VideoQualityError(f"Unknown evaluator suite: {suite}")
        allowed, mode = _SUITES[suite]
        report["mode"] = mode
        report["protocol"] = {"sampling": "native evaluator defaults"}
        if suite == "vbench" and "background_consistency" in dimensions:
            from vllm_omni.benchmarks.realtime_video.encoder_inference import BACKGROUND_CLIP_BATCH_SIZE

            report["protocol"]["background_clip_inference"] = {
                "batch_size": BACKGROUND_CLIP_BATCH_SIZE,
                "gradients": False,
                "frame_selection": "unchanged native input population",
                "adapter_sha256": hashlib.sha256(
                    Path(__file__).with_name("encoder_inference.py").read_bytes()
                ).hexdigest(),
            }
        if not dimensions or len(set(dimensions)) != len(dimensions) or set(dimensions) - set(allowed):
            raise VideoQualityError(f"Invalid dimensions for {suite}: {dimensions}")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise VideoQualityError("timeout_s must be finite and positive")
        if not videos or any(not isinstance(label, str) or not label for label in videos):
            raise VideoQualityError("At least one nonempty video label is required")
        paths = {label: str(Path(path).resolve()) for label, path in videos.items()}
        if any(not Path(path).is_file() or Path(path).stat().st_size == 0 for path in paths.values()):
            raise VideoQualityError("Every requested video must be a nonempty file")
        if suite == "vbench" and any(Path(path).suffix.lower() != ".mp4" for path in paths.values()):
            raise VideoQualityError("VBench video loaders require MP4 inputs; re-encode explicitly before evaluation")
        repo = Path(repo).expanduser().resolve()
        # Resolving venv/bin/python's symlink selects the base environment.
        python = Path(python).expanduser().absolute()
        if not repo.is_dir() or not python.is_file():
            raise VideoQualityError("Evaluator repository and Python executable must exist")
        revision = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
        if not re.fullmatch(r"[a-fA-F0-9]{40,64}", revision):
            raise VideoQualityError("Could not determine evaluator repository revision")
        report["repo_revision"] = revision
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix=f"{suite}-", dir=Path(output_dir).resolve()))
        manifest_path = run_dir / "inputs.json"
        manifest = {
            "encoder_adapter": str(Path(__file__).with_name("encoder_inference.py").resolve()),
            "suite": suite,
            "mode": mode,
            "repo": str(repo),
            "run_dir": str(run_dir),
            "videos": paths,
            "dimensions": list(dimensions),
            "protocol": report["protocol"],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        log, raw_result = run_dir / "evaluator.log", run_dir / "worker_result.json"
        report.update(log=str(log), raw_result=str(raw_result), inputs=str(manifest_path))
        returncode = _run_worker([str(python), "-c", _EVALUATOR_WORKER, str(manifest_path)], repo, log, timeout_s)
        report["returncode"] = returncode
        if returncode != 0:
            raise VideoQualityError(f"Evaluator exited with code {returncode}; see {log}")
        raw = json.loads(raw_result.read_text())
        if not isinstance(raw, dict):
            raise VideoQualityError("Evaluator result must be a JSON object")
        provenance = raw.get("native_provenance", {})
        if not isinstance(provenance, dict):
            raise VideoQualityError("Evaluator native provenance must be a JSON object")
        report["native_provenance"] = provenance
        if suite == "wbench" and "dynamic_degree" in dimensions:
            backends = {
                provenance.get(label, {}).get("dynamic_degree_backend", "unknown")
                for label in videos
                if isinstance(provenance.get(label, {}), dict)
            }
            report["protocol"]["dynamic_degree_backend"] = next(iter(backends)) if len(backends) == 1 else "unknown"
        if raw.get("errors"):
            raise VideoQualityError(f"Evaluator reported errors: {raw['errors']}")
        scores = raw.get("scores")
        if not isinstance(scores, dict) or set(scores) != set(videos):
            raise VideoQualityError("Evaluator did not score exactly the requested videos")
        for label, row in scores.items():
            if not isinstance(row, dict) or set(row) != set(dimensions):
                raise VideoQualityError(f"Incomplete dimension coverage for {label}")
            for dimension, value in row.items():
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                    raise VideoQualityError(f"Invalid scalar score for {label}/{dimension}: {value}")
        report.update(status="passed", scores=scores)
    except (OSError, ValueError, subprocess.SubprocessError, VideoQualityError) as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    return report
