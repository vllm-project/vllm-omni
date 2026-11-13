# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Measure candidate numerical and native semantic accuracy from saved outputs.

This path validates the capture and reports scorer output. Score floors are
applied by the caller (``tests/e2e/accuracy/lingbot_world``). Numerical
precision has no stored reference, so tensor error, PSNR and SSIM are not
computed here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .metrics import RegressionError

EVALUATOR_LOCK = Path(__file__).with_name("evaluators.lock.json")
SUITES = ("precision", "wbench", "vbench")


def read_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise RegressionError(f"Nonfinite JSON constant in {path}: {value}")

    value = json.loads(path.read_text(), parse_constant=reject)
    if not isinstance(value, dict):
        raise RegressionError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_environment(python: Path, requirements: Path) -> dict[str, str]:
    """Check the selected evaluator interpreter against the complete package lock."""
    expected = {}
    for line in requirements.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, version = line.partition("==")
        if not separator or not name or not version:
            raise RegressionError(f"Evaluator dependency is not exactly pinned: {line}")
        expected[name] = version
    if not expected:
        raise RegressionError("Empty evaluator environment lock")
    command = (
        "import importlib.metadata as m,json,sys; print(json.dumps({p:m.version(p) for p in json.loads(sys.argv[1])}))"
    )
    result = subprocess.run(
        [str(python), "-c", command, json.dumps(list(expected))], capture_output=True, text=True, timeout=30, check=True
    )
    actual = json.loads(result.stdout)
    if actual != expected:
        raise RegressionError("Evaluator environment differs from requirements/lingbot-quality.txt")
    return actual


def checked_artifact(root: Path, manifest: Mapping[str, Any], name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or name not in manifest.get("artifacts", {}):
        raise RegressionError(f"Missing or unsafe measurement artifact: {name}")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise RegressionError("Artifact escapes the measurement directory")
    if sha256(path) != manifest["artifacts"][name]:
        raise RegressionError(f"Measurement artifact checksum mismatch: {name}")
    return path


def _full_revision(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(c in "0123456789abcdef" for c in value)


def load_candidate(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    contract = manifest.get("contract", {})
    profile = contract.get("profile", {})
    cases = profile.get("cases", [])
    if (
        manifest.get("schema_version") != 2
        or manifest.get("kind") != "realtime-video-accuracy-candidate"
        or not _full_revision(manifest.get("candidate_revision"))
        or contract.get("capture_recipe") != {"candidate_starts": 1, "sessions_per_case": 1}
        or not isinstance(manifest.get("artifacts"), dict)
        or not isinstance(cases, list)
        or not cases
    ):
        raise RegressionError("Invalid candidate source/capture contract")
    ids = [case.get("id") for case in cases]
    if any(
        not isinstance(value, str) or not value or Path(value).name != value or value in (".", "..") for value in ids
    ):
        raise RegressionError("Case IDs must be nonempty single path components")
    if len(set(ids)) != len(ids):
        raise RegressionError("Candidate case IDs must be unique")
    errors = manifest.get("errors")
    if not isinstance(errors, list) or any(
        not isinstance(error, dict) or error.get("component") not in ("precision", "stream", "server", "cleanup")
        for error in errors
    ):
        raise RegressionError("Invalid capture error diagnostics")
    if any(error["component"] in ("server", "cleanup") for error in errors):
        raise RegressionError(f"Candidate server/cleanup failed: {errors}")
    cleanup = read_json(checked_artifact(path.parent, manifest, "candidate/cleanup.json"))
    if cleanup.get("passed") is not True:
        raise RegressionError("Candidate device cleanup was not successful")
    return manifest


def verify_evaluator_source(repository: Path, revision: str) -> None:
    actual = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        raise RegressionError("Wrong evaluator revision")
    # CI may add agent metadata here; evaluator scoring imports no .claude code.
    # Importing a locked checkout writes bytecode caches (torch.hub's local
    # DINO checkout has no .gitignore); Python revalidates them against the
    # tracked sources, so they are not source changes.
    status = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            ".",
            ":(exclude).claude",
            ":(exclude,glob)**/__pycache__/**",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if status.stdout.strip():
        raise RegressionError("Evaluator checkout has tracked or untracked changes")


@contextmanager
def _evaluator_environment(root: Path):
    values = {
        "VBENCH_CACHE_DIR": str(root / "vbench-cache"),
        "WBENCH_WEIGHTS_DIR": str(root / "wbench/weights"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _record_numerical(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    from .precision import describe_tensor_sample

    if any(error["component"] == "precision" for error in manifest["errors"]):
        raise RegressionError("Candidate precision capture failed")
    path = checked_artifact(root, manifest, "candidate/precision.npz")
    return {
        "replay": describe_tensor_sample(path),
        "artifact_sha256": sha256(path),
        "scope": "Finite fixed-input replay measurements; tensor error, PSNR and SSIM need a stored reference.",
        "packages": {name: importlib.metadata.version(name) for name in ("numpy",)},
    }


def _candidate_videos(root: Path, manifest: dict[str, Any]) -> tuple[dict[str, Path], dict[str, Any]]:
    from .video_quality import verify_video_coverage
    from .workload import num_frames_for_chunks

    profile = manifest["contract"]["profile"]
    if any(error["component"] == "stream" for error in manifest["errors"]):
        raise RegressionError("Candidate video capture failed")
    for case in profile["cases"]:
        benchmark = read_json(checked_artifact(root, manifest, f"candidate/{case['id']}/benchmark.json"))
        workload = benchmark.get("workload", {})
        expected = {
            "num_chunks": profile["num_chunks"],
            "width": profile["width"],
            "height": profile["height"],
            "fps": profile["mux_fps"],
            "seed": case["seed"],
        }
        if (
            benchmark.get("passed") is not True
            or benchmark.get("git_revision") != manifest["candidate_revision"]
            or benchmark.get("server_revision") != manifest["candidate_revision"]
            or benchmark.get("config", {}).get("sessions") != 1
            or benchmark.get("config", {}).get("camera_pattern") != case["camera_pattern"]
            or any(workload.get(key) != value for key, value in expected.items())
        ):
            raise RegressionError(f"Candidate benchmark did not complete the recorded workload: {case['id']}")
    videos = {
        case["id"]: checked_artifact(root, manifest, f"candidate/{case['id']}/rollout.mp4") for case in profile["cases"]
    }
    coverage = {
        label: verify_video_coverage(
            path,
            frames=num_frames_for_chunks(profile["num_chunks"]),
            width=profile["width"],
            height=profile["height"],
            fps=profile["mux_fps"],
        )
        for label, path in videos.items()
    }
    return videos, coverage


def _record_semantic(
    root: Path,
    manifest: dict[str, Any],
    *,
    suite: str,
    output_dir: Path,
    evaluator_root: Path,
    evaluator_python: Path,
    evaluator_requirements: Path,
) -> dict[str, Any]:
    from .encoder_inference import verify_background_coverage
    from .video_quality import STANDARD_VBENCH_DIMENSIONS, WBENCH_DIMENSIONS, evaluate_video_quality
    from .workload import num_frames_for_chunks

    evaluator_root = evaluator_root.expanduser().resolve()
    lock = read_json(EVALUATOR_LOCK)
    packages = verify_environment(evaluator_python, evaluator_requirements)
    for name, repository in lock["repositories"].items():
        verify_evaluator_source(evaluator_root / name, repository["revision"])
    for name, digest in lock["files"].items():
        if sha256(evaluator_root / name) != digest:
            raise RegressionError(f"Evaluator weight checksum mismatch: {name}")
    videos, coverage = _candidate_videos(root, manifest)
    dimensions = list(WBENCH_DIMENSIONS if suite == "wbench" else STANDARD_VBENCH_DIMENSIONS)
    with _evaluator_environment(evaluator_root):
        native = evaluate_video_quality(
            videos,
            output_dir / suite,
            suite=suite,
            repo=evaluator_root / suite,
            python=evaluator_python,
            dimensions=dimensions,
            timeout_s=1800,
        )
    write_json(output_dir / f"{suite}-raw.json", native)
    if native["status"] != "passed":
        raise RegressionError(f"{suite} evaluator failed: {native['errors']}")
    if suite == "wbench" and native["protocol"].get("dynamic_degree_backend") != "raft":
        raise RegressionError("WBench requires the pinned RAFT backend")
    for label, row in native["scores"].items():
        if set(row) != set(dimensions):
            raise RegressionError(f"Incomplete {suite} dimension coverage for {label}")
        if any(type(value) not in (float, int) or not math.isfinite(value) for value in row.values()):
            raise RegressionError(f"Nonfinite or invalid {suite} scores for {label}")
        if row["dynamic_degree"] not in (0.0, 1.0):
            raise RegressionError("Native dynamic_degree must be a binary classification")
        if suite == "vbench":
            verify_background_coverage(
                native["native_provenance"].get(label, {}).get("background_clip_inference", {}),
                num_frames_for_chunks(manifest["contract"]["profile"]["num_chunks"]),
            )
    if set(native["scores"]) != set(videos):
        raise RegressionError(f"Incomplete {suite} case coverage")
    return {
        "scores": native["scores"],
        "video_inputs": coverage,
        "protocol": native["protocol"],
        "native_provenance": native["native_provenance"],
        "evaluator_revision": native["repo_revision"],
        "evaluator_lock_sha256": sha256(EVALUATOR_LOCK),
        "packages": packages,
        "raw_result": str(output_dir / f"{suite}-raw.json"),
    }


def evaluate_saved(
    candidate_path: Path,
    *,
    suite: str,
    output_dir: Path,
    evaluator_root: Path | None = None,
    evaluator_python: Path | None = None,
    evaluator_requirements: Path | None = None,
) -> dict[str, Any]:
    """Measure one independent suite, retaining failures; comparison is the caller's job."""
    if suite not in SUITES:
        raise RegressionError(f"Unknown accuracy suite: {suite}")
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "failed",
        "suite": suite,
        "passed": False,
        "measurements": {},
        "errors": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scope": "Measurement validity only; score floors are applied by the accuracy test.",
    }
    try:
        manifest = load_candidate(candidate_path)
        root = candidate_path.parent
        report["provenance"] = {
            "candidate_revision": manifest["candidate_revision"],
            "candidate_manifest_sha256": sha256(candidate_path),
            "contract": manifest["contract"],
            "generation_hardware": manifest.get("hardware"),
            "artifacts": manifest["artifacts"],
        }
        if suite == "precision":
            report["measurements"] = _record_numerical(root, manifest)
        else:
            if evaluator_root is None or evaluator_python is None or evaluator_requirements is None:
                raise RegressionError("Semantic evaluation requires evaluator root, Python and requirements lock")
            report["measurements"] = _record_semantic(
                root,
                manifest,
                suite=suite,
                output_dir=output_dir,
                evaluator_root=evaluator_root,
                evaluator_python=evaluator_python,
                evaluator_requirements=evaluator_requirements,
            )
        report.update(status="measured", passed=True)
    except Exception as error:
        # A scorer/capture failure fails this test while preserving independent
        # suite results and a JSON diagnostic for CI artifacts and console logs.
        report["errors"].append(f"{type(error).__name__}: {error}")
    write_json(output_dir / f"{suite}.json", report)
    print(f"Realtime video accuracy measurements ({suite}):")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return report


def evaluate_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Use the real serving frontend's result path for offline evaluation."""
    from vllm.benchmarks.serve import compute_result_filename

    report = evaluate_saved(
        args.realtime_video_candidate,
        suite=args.realtime_video_evaluate,
        output_dir=args.realtime_video_eval_output_dir,
        evaluator_root=args.realtime_video_evaluator_root,
        evaluator_python=args.realtime_video_evaluator_python,
        evaluator_requirements=args.realtime_video_evaluator_requirements,
    )
    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    result: dict[str, Any] = {
        "date": current_dt,
        "endpoint_type": args.backend,
        "backend": args.backend,
        "model_id": args.model,
        "dataset_name": "realtime-video",
        "num_prompts": 0,
        "realtime_video_evaluation": report,
    }
    if args.metadata:
        for item in args.metadata:
            key, separator, value = item.partition("=")
            if not separator:
                raise RegressionError("Invalid metadata format; use KEY=VALUE")
            if key.strip() in result:
                raise RegressionError(f"Metadata cannot replace recorded result field: {key.strip()}")
            result[key.strip()] = value.strip()
    filename = compute_result_filename(args, args.model, args.label, current_dt)
    if args.save_result or args.append_result:
        assert filename is not None
        with Path(filename).open("a+" if args.append_result else "w") as handle:
            if args.append_result and handle.tell():
                handle.write("\n")
            json.dump(result, handle, allow_nan=False)
    return result
