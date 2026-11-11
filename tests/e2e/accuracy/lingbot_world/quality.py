# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run one native semantic evaluator, independently of all other verdicts."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from benchmarks.lingbot_world.encoder_inference import verify_background_coverage
from benchmarks.lingbot_world.regression import (
    RegressionError,
    check_binary_activity,
    check_quality,
    read_json,
    sha256,
    verify_environment,
    write_json,
)
from benchmarks.lingbot_world.video_quality import evaluate_video_quality, verify_video_coverage
from tests.e2e.accuracy.lingbot_world.runner import ROOT, checked_artifact, git_revision
from vllm_omni.benchmarks.realtime_video.workload import num_frames_for_chunks


def verify_evaluator_source(repository: Path, revision: str) -> None:
    if git_revision(repository) != revision:
        raise RegressionError("Wrong evaluator revision")
    # CI may add agent instructions in .claude; scoring code is never loaded
    # from that metadata directory. All other tracked/untracked dirt fails.
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
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if status.stdout.strip():
        raise RegressionError("Evaluator checkout has tracked or untracked changes")


def evaluate_quality(settings, data, suite, monkeypatch):
    if suite not in ("wbench", "vbench"):
        raise RegressionError("Select one semantic suite")
    root, manifest = data
    output = settings["output"]
    report = {"passed": False, "suite": suite, "cases": {}, "errors": []}
    try:
        policy = read_json(ROOT / "tests/e2e/accuracy/lingbot_world/quality.json")
        lock = read_json(ROOT / "tests/e2e/accuracy/lingbot_world/evaluators.lock.json")
        from tests.e2e.accuracy.lingbot_world.prepare import evaluator_cache_key, prepare_evaluators

        evaluator_root = Path(
            os.environ.get(
                "LINGBOT_EVALUATOR_ROOT",
                str(Path(os.environ.get("HF_HOME", "/tmp")) / f"lingbot-evaluators-{evaluator_cache_key()}"),
            )
        ).resolve()
        python = (
            Path(os.environ["LINGBOT_EVALUATOR_PYTHON"])
            if "LINGBOT_EVALUATOR_PYTHON" in os.environ
            else prepare_evaluators(evaluator_root)
        )
        report["packages"] = verify_environment(python, ROOT / "requirements/lingbot-quality.txt")
        # Both suites share a read-only environment, never a semantic result.
        repository = lock["repositories"][suite]
        verify_evaluator_source(evaluator_root / suite, repository["revision"])
        for name, digest in lock["files"].items():
            if sha256(evaluator_root / name) != digest:
                raise RegressionError(f"Evaluator weight checksum mismatch: {name}")
        monkeypatch.setenv("VBENCH_CACHE_DIR", str(evaluator_root / "vbench-cache"))
        monkeypatch.setenv("WBENCH_WEIGHTS_DIR", str(evaluator_root / "wbench/weights"))
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
        videos = {
            f"case{index}_{role}{repeat}": checked_artifact(root, manifest, f"{role}/{repeat}/{case['id']}/rollout.mp4")
            for index, case in enumerate(settings["cases"])
            for role in ("reference", "candidate")
            for repeat in range(3)
        }
        report["video_inputs"] = {
            label: verify_video_coverage(
                path,
                frames=num_frames_for_chunks(settings["num_chunks"]),
                width=settings["width"],
                height=settings["height"],
                fps=settings["mux_fps"],
            )
            for label, path in videos.items()
        }
        tolerances = policy["candidate_tolerances"][settings["platform"]][suite]
        native = evaluate_video_quality(
            videos,
            output / suite,
            suite=suite,
            repo=evaluator_root / suite,
            python=python,
            dimensions=list(tolerances),
            timeout_s=policy["timeout_s"],
        )
        write_json(output / f"{suite}-raw.json", native)
        if native["status"] != "passed":
            raise RegressionError(f"{suite} evaluator failed: {native['errors']}")
        if suite == "wbench" and native["protocol"].get("dynamic_degree_backend") != "raft":
            raise RegressionError("WBench requires the pinned RAFT backend")
        for index, case in enumerate(settings["cases"]):
            refs = [native["scores"][f"case{index}_reference{r}"] for r in range(3)]
            cands = [native["scores"][f"case{index}_candidate{r}"] for r in range(3)]
            if suite == "vbench":
                for role in ("reference", "candidate"):
                    for repeat in range(3):
                        verify_background_coverage(
                            native["native_provenance"]
                            .get(f"case{index}_{role}{repeat}", {})
                            .get("background_clip_inference", {}),
                            num_frames_for_chunks(settings["num_chunks"]),
                        )
            continuous = {k: v for k, v in tolerances.items() if k not in policy["binary_activity_dimensions"]}
            checks = [
                check_quality(
                    [{k: row[k] for k in continuous} for row in refs],
                    {k: row[k] for k in continuous},
                    tolerances=continuous,
                    max_reference_spreads=policy["max_reference_spreads"][settings["platform"]][suite],
                )
                for row in cands
            ]
            checks += [
                check_binary_activity([row[k] for row in refs], [row[k] for row in cands], tolerance=tolerances[k])
                for k in policy["binary_activity_dimensions"]
            ]
            report["cases"][case["id"]] = checks
        report["passed"] = bool(report["cases"]) and all(c["passed"] for cs in report["cases"].values() for c in cs)
    except Exception as error:
        report["errors"].append(str(error))
    finally:
        write_json(output / f"{suite}.json", report)
    return report
