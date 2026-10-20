# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Decoded-video diagnostics and isolated, reference-free quality evaluators.

Pixel distances describe rollout divergence, not tensor numerical parity. Quality
scores use each suite's native scale and are custom-input diagnostics, never a
leaderboard total. Same-code repeat rollouts are needed to interpret differences.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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
    "vbench_long": (VBENCH_DIMENSIONS, "long_custom_input"),
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


def _distances(absolute_sum: float, squared_sum: float, samples: int) -> dict[str, Any]:
    mse = squared_sum / samples
    return {
        "mae": absolute_sum / samples,
        "rmse": math.sqrt(mse),
        "psnr_db": 10 * math.log10(255.0**2 / mse) if mse else None,
        "psnr_is_infinite": mse == 0,
    }


def compare_video_pair(
    reference: Path, candidate: Path, *, window_seconds: float = 30.0, ssim_stride: int = 0
) -> dict[str, Any]:
    """Stream every decoded frame; reject unequal or incomplete video coverage.

    Distances use decoded 8-bit BGR samples. Identical frames have infinite PSNR,
    represented by ``psnr_db=None, psnr_is_infinite=True`` for strict JSON output.
    """
    if not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be finite and positive")
    if not isinstance(ssim_stride, int) or ssim_stride < 0:
        raise ValueError("ssim_stride must be a nonnegative integer")
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise VideoQualityError("Video comparison requires opencv-python and numpy") from exc

    reference, candidate = Path(reference).resolve(), Path(candidate).resolve()
    captures = [cv2.VideoCapture(str(path)) for path in (reference, candidate)]
    try:
        ref, cand = [_capture_metadata(cap, path, cv2) for cap, path in zip(captures, (reference, candidate))]
        for key in ("fps", "frame_count", "width", "height"):
            if ref[key] != cand[key]:
                raise VideoQualityError(f"Video {key} mismatch: {ref[key]} != {cand[key]}")
        frame_metrics: list[dict[str, Any]] = []
        ssim_values: list[float] = []
        windows: list[dict[str, Any]] = []
        uniform_frames: dict[str, list[int]] = {"reference": [], "candidate": []}
        totals = {"absolute_sum": 0.0, "squared_sum": 0.0, "samples": 0, "different_frames": 0}
        window: dict[str, Any] | None = None
        index = 0
        while True:
            ref_ok, ref_frame = captures[0].read()
            cand_ok, cand_frame = captures[1].read()
            if ref_ok != cand_ok:
                raise VideoQualityError(f"Video decode coverage mismatch at frame {index}")
            if not ref_ok:
                break
            expected_shape = (ref["height"], ref["width"], 3)
            if ref_frame.shape != expected_shape or cand_frame.shape != expected_shape:
                raise VideoQualityError(f"Decoded geometry changed at frame {index}")
            if ref_frame.dtype != np.uint8 or cand_frame.dtype != np.uint8:
                raise VideoQualityError(f"Expected 8-bit decoded frames at frame {index}")
            for role, frame in (("reference", ref_frame), ("candidate", cand_frame)):
                if not np.any(frame != frame[0, 0]):
                    uniform_frames[role].append(index)
            delta = ref_frame.astype(np.float64) - cand_frame
            if ssim_stride and index % ssim_stride == 0:
                from skimage.metrics import structural_similarity

                ssim_values.append(
                    float(
                        structural_similarity(
                            ref_frame,
                            cand_frame,
                            channel_axis=2,
                            data_range=255,
                            gaussian_weights=True,
                            sigma=1.5,
                            use_sample_covariance=False,
                        )
                    )
                )
            absolute_sum = float(np.abs(delta).sum())
            squared_sum = float(np.square(delta).sum())
            samples = delta.size
            timestamp_s = index / ref["fps"]
            window_index = math.floor(timestamp_s / window_seconds)
            if window is None or window["index"] != window_index:
                window = {
                    "index": window_index,
                    "start_s": window_index * window_seconds,
                    "end_s": min((window_index + 1) * window_seconds, ref["duration_s"]),
                    "start_frame": index,
                    "frame_count": 0,
                    "absolute_sum": 0.0,
                    "squared_sum": 0.0,
                    "samples": 0,
                    "different_frames": 0,
                }
                windows.append(window)
            for accumulator in (totals, window):
                accumulator["absolute_sum"] += absolute_sum
                accumulator["squared_sum"] += squared_sum
                accumulator["samples"] += samples
                accumulator["different_frames"] += int(squared_sum != 0)
            window["frame_count"] += 1
            frame_metrics.append(
                {"index": index, "time_s": timestamp_s, **_distances(absolute_sum, squared_sum, samples)}
            )
            index += 1
        if index != ref["frame_count"]:
            raise VideoQualityError(f"Truncated or inconsistent video: decoded {index} of {ref['frame_count']} frames")
        for accumulator in windows:
            accumulator.update(
                _distances(accumulator.pop("absolute_sum"), accumulator.pop("squared_sum"), accumulator.pop("samples"))
            )
        return {
            "status": "complete",
            "reference": ref,
            "candidate": cand,
            "decoded_frames": index,
            "uniform_frame_indices": uniform_frames,
            "window_seconds": window_seconds,
            "total": {
                "frame_count": index,
                "different_frames": totals["different_frames"],
                **_distances(totals["absolute_sum"], totals["squared_sum"], totals["samples"]),
            },
            "windows": windows,
            "frame_metrics": frame_metrics,
            "ssim": {
                "stride": ssim_stride,
                "samples": len(ssim_values),
                "mean": sum(ssim_values) / len(ssim_values) if ssim_values else None,
            },
            "scope": "Decoded-frame divergence diagnostics, not tensor parity or a quality acceptance test.",
            "baseline_repeat_required": True,
        }
    finally:
        for capture in captures:
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
        sys.path.insert(0, manifest["adapter_root"])
        import hashlib
        import benchmarks.lingbot_world.encoder_inference as encoder_adapter
        from benchmarks.lingbot_world.encoder_inference import install_vbench_background_inference
        adapter_sha256 = hashlib.sha256(pathlib.Path(encoder_adapter.__file__).read_bytes()).hexdigest()
        if adapter_sha256 != manifest["protocol"]["background_clip_inference"]["adapter_sha256"]:
            raise ValueError("Background CLIP inference adapter differs from the recorded source")
        background_stats = install_vbench_background_inference()
    if manifest["suite"] == "wbench":
        if "hpsv3_quality" in manifest["dimensions"]:
            # RAFT adds core/ (which contains datasets.py) to sys.path.
            # Resolve the Hugging Face package before that native side effect.
            import datasets
        from src.metrics.video_quality.evaluator import VideoQualityEvaluator
        evaluator = VideoQualityEvaluator(device="cuda", metrics=manifest["dimensions"])
    else:
        import torch
        if manifest["suite"] == "vbench_long":
            from vbench2_beta_long import VBenchLong as Evaluator
        else:
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
                if manifest["suite"] == "vbench_long":
                    configs = repo / "vbench2_beta_long/configs"
                    kwargs.update(
                        use_semantic_splitting=False, clip_length_config="clip_length_mix.yaml",
                        sb_clip2clip_feat_extractor="dinov2", bg_clip2clip_feat_extractor="dreamsim",
                        w_inclip=1.0, w_clip2clip=0.0, dev_flag=True,
                        slow_fast_eval_config=str(configs / "slow_fast_params.yaml"),
                        sb_mapping_file_path=str(configs / "subject_mapping_table.yaml"),
                        bg_mapping_file_path=str(configs / "background_mapping_table.yaml"),
                        num_of_samples_per_prompt=1, static_filter_flag=False,
                    )
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
                expected = (
                    {path.resolve() for path in (stage / "split_clip").rglob("*.mp4")}
                    if manifest["suite"] == "vbench_long" else {staged.resolve()}
                )
                if not expected:
                    raise ValueError("VBenchLong produced no clips")
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


_PIXEL_WORKER = r"""
import hashlib, importlib.metadata, json, sys
from pathlib import Path
inputs = json.loads(Path(sys.argv[1]).read_text())
sys.path.insert(0, inputs["adapter_root"])
import cv2
from benchmarks.lingbot_world.video_quality import compare_video_pair
environment = {
    "python": sys.executable,
    "packages": {name: importlib.metadata.version(name) for name in ("numpy", "opencv-python", "scikit-image")},
    "opencv_version": cv2.__version__,
    "opencv_build_sha256": hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
}
result = {"status": "failed", "pixel_environment": environment}
try:
    if environment["packages"] != inputs["expected_packages"]:
        raise ValueError("Pixel dependencies changed from the verified evaluator environment")
    result = {
        **compare_video_pair(
            Path(inputs["reference"]), Path(inputs["candidate"]),
            window_seconds=inputs["window_seconds"], ssim_stride=inputs["ssim_stride"],
        ),
        "pixel_environment": environment,
    }
finally:
    Path(inputs["result"]).write_text(json.dumps(result, allow_nan=False) + "\n")
"""


def compare_video_pair_in_environment(
    reference: Path,
    candidate: Path,
    output_dir: Path,
    *,
    python: Path,
    expected_packages: Mapping[str, str],
    window_seconds: float,
    ssim_stride: int,
    timeout_s: float,
) -> dict[str, Any]:
    """Compare pixels in the verified evaluator interpreter and retain evidence."""
    # Preserve the virtualenv path rather than resolving its interpreter symlink.
    python = Path(python).expanduser().absolute()
    names = ("numpy", "opencv-python", "scikit-image")
    if any(not isinstance(expected_packages.get(name), str) or not expected_packages[name] for name in names):
        raise VideoQualityError("Pixel environment requires verified decoder dependency versions")
    if not python.is_file() or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise VideoQualityError("Pixel comparison requires an interpreter and a positive timeout")
    expected = {name: expected_packages[name] for name in names}
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="pixel-", dir=output_dir.resolve()))
    result_path = run_dir / "worker_result.json"
    result_path.write_text('{"status": "incomplete"}\n')
    inputs = {
        "adapter_root": str(Path(__file__).resolve().parents[2]),
        "reference": str(reference.resolve()),
        "candidate": str(candidate.resolve()),
        "expected_packages": expected,
        "window_seconds": window_seconds,
        "ssim_stride": ssim_stride,
        "result": str(result_path),
    }
    manifest = run_dir / "inputs.json"
    manifest.write_text(json.dumps(inputs, allow_nan=False) + "\n")
    log = run_dir / "worker.log"
    # Isolated mode ignores the parent's PYTHONPATH and user site packages.
    status = _run_worker(
        [str(python), "-I", "-c", _PIXEL_WORKER, str(manifest)],
        Path(inputs["adapter_root"]),
        log,
        timeout_s,
    )
    if status != 0:
        raise VideoQualityError(f"Pixel worker exited with code {status}; see {log}")
    result = json.loads(result_path.read_text())
    if not isinstance(result, dict) or result.get("status") != "complete":
        raise VideoQualityError(f"Pixel worker did not complete; see {result_path}")
    environment = result.get("pixel_environment")
    if (
        not isinstance(environment, dict)
        or environment.get("packages") != expected
        or environment.get("python") != str(python)
    ):
        raise VideoQualityError(f"Pixel worker environment differs from the selected interpreter; see {result_path}")
    return result


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
        report["protocol"] = (
            {
                "clip_seconds": 2,
                "use_semantic_splitting": False,
                "dev_flag": True,
                "subject_feature": "dinov2",
                "background_feature": "dreamsim",
                "consistency_weights": "vbench2_beta_long/configs/slow_fast_params.yaml",
            }
            if suite == "vbench_long"
            else {"sampling": "native evaluator defaults"}
        )
        if suite == "vbench" and "background_consistency" in dimensions:
            from benchmarks.lingbot_world.encoder_inference import BACKGROUND_CLIP_BATCH_SIZE

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
        if suite in {"vbench", "vbench_long"} and any(Path(path).suffix.lower() != ".mp4" for path in paths.values()):
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
            "adapter_root": str(Path(__file__).resolve().parents[2]),
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
