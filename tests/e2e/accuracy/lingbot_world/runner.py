# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Capture one candidate lifetime for temporary accuracy recording."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from vllm_omni.benchmarks.realtime_video.client import resolve_image_reference
from vllm_omni.benchmarks.realtime_video.eval import read_json, sha256, write_json
from vllm_omni.benchmarks.realtime_video.metrics import RegressionError

ROOT = Path(__file__).resolve().parents[4]
ENGINE_ENVIRONMENT = {"VLLM_OMNI_AR_DIFFUSION_KV_GATHER": "0", "AR_DIFFUSION_KV_SCRATCH_BLOCKS_PER_BRANCH": "0"}
CAPTURE_RECIPE = {"candidate_starts": 1, "sessions_per_case": 1}


def load_settings(profile: Path, output: Path, *, generation=None):
    settings = read_json(profile)
    settings.update(
        profile_path=profile.resolve(),
        output=output.resolve(),
        generation=generation,
    )
    return settings


def server_environment(source: Path, *, cpu_threads: int | None = None) -> dict[str, str]:
    # Keep CANN's compiler modules (tbe/te) and any other backend import paths.
    # The selected checkout must still win over an inherited candidate path.
    paths = [str(source)]
    if inherited := os.environ.get("PYTHONPATH"):
        paths.append(inherited)
    environment = {
        "PYTHONPATH": os.pathsep.join(paths),
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        **ENGINE_ENVIRONMENT,
    }
    if cpu_threads is not None:
        if type(cpu_threads) is not int or cpu_threads < 1:
            raise RegressionError("CPU threads per worker must be a positive integer")
        environment.update(
            {name: str(cpu_threads) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
        )
    return environment


def server_arguments(settings: dict[str, Any]) -> list[str]:
    # Keep inner initialization deadlines aligned with the readiness wait.
    timeout = str(settings["startup_timeout_s"])
    return [
        "--deploy-config",
        str(ROOT / settings["deploy_config"]),
        "--init-timeout",
        timeout,
        "--stage-init-timeout",
        timeout,
    ]


def assert_device_cleanup(dies: int, output: Path) -> None:
    """Gate the shared cleanup condition without granting another wait."""
    from tests.helpers.clean import wait_for_gpu_memory_to_clear

    devices = list(range(dies)) if type(dies) is int and dies > 0 else []
    record = {
        "passed": False,
        "logical_devices": devices,
        "threshold_ratio": 0.05,
        "timeout_s": 0,
        "scope": "shared helper engine-memory view after server cleanup; excluded from performance",
    }
    write_json(output, record)
    if not devices:
        raise RegressionError("Device cleanup requires a positive integer die count")
    wait_for_gpu_memory_to_clear(devices=devices, threshold_ratio=0.05, timeout_s=0)
    write_json(output, {**record, "passed": True})


def git_revision(root: Path) -> str:
    if not (root / ".git").exists() and root == ROOT:
        # CI Docker contexts exclude .git; the image tag is the Buildkite SHA.
        revision = os.environ.get("BUILDKITE_COMMIT", "")
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise RegressionError("Source image has no Git metadata or full BUILDKITE_COMMIT")
        return revision
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def record_benchmark_revisions(document: dict[str, Any]) -> None:
    """Bind both client and candidate server measurements to this source."""
    revision = git_revision(ROOT)
    if "git_revision" not in document or document["git_revision"] not in (None, revision):
        raise RegressionError("Benchmark client revision differs from the attested harness")
    # Source images omit .git, so the standalone client deliberately reports
    # None. This harness can attest its image's full Buildkite commit without
    # letting an installed standalone wheel claim an unrelated checkout.
    document["git_revision"] = revision
    document["server_revision"] = revision


def checked_assets(settings: dict[str, Any]) -> tuple[Path, dict[str, Path]]:
    """Resolve the pinned snapshot and verify every requested input checksum."""
    model = Path(os.environ["LINGBOT_MODEL_PATH"]).resolve()
    for subdir in ("transformer", "vae", "text_encoder", "tokenizer", "scheduler"):
        if not (model / subdir).is_dir():
            raise RegressionError(f"Incomplete checkpoint: {subdir}")
    lock = read_json(ROOT / "tests/e2e/accuracy/lingbot_world/assets.lock.json")
    if settings["model_revision"] != lock["model_revision"]:
        raise RegressionError("Profile does not name the locked model revision")
    for relative, digest in lock["model_files"].items():
        if sha256(model / relative) != digest:
            raise RegressionError(f"Checkpoint checksum mismatch: {relative}")
    from vllm.assets.image import ImageAsset

    images = {}
    for case in settings["cases"]:
        asset = case["image"]
        path = Path(ImageAsset(asset).get_path("jpg"))
        if sha256(path) != lock["images"][asset]:
            raise RegressionError(f"Input image checksum mismatch: {asset}")
        images[asset] = path
    return model, images


def _capture(model: Path, destination: Path, settings: dict[str, Any]) -> None:
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("capture_precision.py")),
            "--model",
            str(model),
            "--output",
            str(destination),
            "--height",
            str(settings["height"]),
            "--width",
            str(settings["width"]),
        ],
        check=True,
        timeout=settings["precision_timeout_s"],
        cwd=ROOT,
        env={**os.environ, **server_environment(ROOT, cpu_threads=settings.get("cpu_threads_per_worker"))},
    )


def hardware_contract(settings):
    from vllm_omni.platforms import current_omni_platform as platform

    if platform.device_type != settings["platform"] or platform.get_device_count() != settings["dies"]:
        raise RegressionError("Wrong platform or visible device count")
    from tests.e2e.accuracy.lingbot_world.prepare import verify_inference_runtime

    return {
        "platform": platform.device_type,
        "cards": settings["cards"],
        "dies": settings["dies"],
        "devices": [
            {"name": platform.get_device_name(i), "total_bytes": platform.get_device_total_memory(i)}
            for i in range(settings["dies"])
        ],
        **verify_inference_runtime(settings["platform"]),
    }


def benchmark_args(server, settings, case, destination):
    from vllm_omni.benchmarks.realtime_video.serving import build_serve_args
    from vllm_omni.benchmarks.realtime_video.workload import default_workload

    workload = default_workload(
        image_reference=resolve_image_reference(str(case["path"])),
        num_chunks=settings["num_chunks"],
        camera_pattern=case["camera_pattern"],
        prompt=case["prompt"],
        seed=case["seed"],
        height=settings["height"],
        width=settings["width"],
        fps=settings["mux_fps"],
    )
    return build_serve_args(
        [
            "--port",
            str(server.port),
            "--model",
            str(server.model),
            "--num-prompts",
            "1",
            "--num-warmups",
            "1",
            "--realtime-video-camera-pattern",
            case["camera_pattern"],
            "--realtime-video-warmup-num-chunks",
            str(settings["warmup_num_chunks"]),
            "--realtime-video-warmup-chunks",
            str(settings["warmup_chunks"]),
            "--realtime-video-target-fps",
            str(settings["target_fps"]),
            "--realtime-video-playback-buffer-chunks",
            "1",
            "--realtime-video-first-chunk-timeout",
            str(settings["first_chunk_timeout_s"]),
            "--realtime-video-chunk-timeout",
            str(settings["chunk_timeout_s"]),
            "--realtime-video-save-video",
            str(destination / "rollout.mp4"),
            "--save-result",
            "--result-dir",
            str(destination),
            "--result-filename",
            "serve-result.json",
        ],
        workload=workload,
    )


def generation_paths(settings):
    return {
        "candidate/precision.npz",
        "candidate/cleanup.json",
        *{
            f"candidate/{case['id']}/{file}"
            for case in settings["cases"]
            for file in ("serve-result.json", "benchmark.json", "rollout.mp4")
        },
    }


def generation_contract(settings):
    """Retain model, workload and topology identity with raw candidate data."""
    return {
        "profile": read_json(settings["profile_path"]),
        "capture_recipe": dict(CAPTURE_RECIPE),
        "model_assets_sha256": sha256(ROOT / "tests/e2e/accuracy/lingbot_world/assets.lock.json"),
        "deploy_sha256": sha256(ROOT / settings["deploy_config"]),
        "engine_environment": dict(ENGINE_ENVIRONMENT),
        "hardware": {name: settings[name] for name in ("platform", "hardware", "cards", "dies")},
    }


def collect_measurements(settings):
    from tests.helpers.runtime import OmniServer

    output = settings["output"]
    if (output / "candidate").exists():
        raise RegressionError("Use a fresh measurement output")
    model, images = checked_assets(settings)
    report = {
        "schema_version": 2,
        "kind": "realtime-video-accuracy-candidate",
        "complete": False,
        "candidate_revision": git_revision(ROOT),
        "contract": generation_contract(settings),
        "hardware": hardware_contract(settings),
        "errors": [],
        "artifacts": {},
    }
    write_json(output / "generation.json", report)
    arm = output / "candidate"
    arm.mkdir(parents=True)
    try:
        _capture(model, arm / "precision.npz", settings)
    except (OSError, subprocess.SubprocessError) as error:
        report["errors"].append({"component": "precision", "error": str(error)})
    try:
        with OmniServer(
            str(model),
            server_arguments(settings),
            env_dict=server_environment(ROOT, cpu_threads=settings.get("cpu_threads_per_worker")),
            startup_timeout=settings["startup_timeout_s"],
        ) as server:
            for case in settings["cases"]:
                destination = arm / case["id"]
                destination.mkdir()
                try:
                    from vllm_omni.benchmarks.serve import main as bench_serve

                    result = bench_serve(
                        benchmark_args(server, settings, {**case, "path": images[case["image"]]}, destination)
                    )
                    document = result["realtime_video"]
                    record_benchmark_revisions(document)
                    write_json(destination / "benchmark.json", document)
                    if not document["passed"]:
                        raise RegressionError(f"Incomplete realtime-video measurement: {document['errors']}")
                except Exception as error:
                    # Preserve partial inputs so each independent scorer can
                    # diagnose its own missing/incomplete case after shutdown.
                    report["errors"].append({"case": case["id"], "component": "stream", "error": str(error)})
    except Exception as error:
        report["errors"].append({"component": "server", "error": str(error)})
    try:
        assert_device_cleanup(settings["dies"], arm / "cleanup.json")
    except Exception as error:
        report["errors"].append({"component": "cleanup", "error": str(error)})
    report["artifacts"] = {str(p.relative_to(output)): sha256(p) for p in arm.rglob("*") if p.is_file()}
    report["complete"] = not report["errors"] and generation_paths(settings) <= report["artifacts"].keys()
    write_json(output / "generation.json", report)
    return report


def validate_manifest(settings, manifest):
    if (
        manifest.get("schema_version") != 2
        or manifest.get("kind") != "realtime-video-accuracy-candidate"
        or manifest.get("candidate_revision") != git_revision(ROOT)
        or manifest.get("contract") != generation_contract(settings)
    ):
        raise RegressionError("Measurement source/profile/runtime contract mismatch")


def measurement_data(settings):
    root = settings["generation"]
    if root is None:
        raise RegressionError("--lingbot-generation must name real collected measurements")
    manifest = read_json(root / "generation.json")
    validate_manifest(settings, manifest)
    # Completion and semantic verdicts are deliberately not shared prerequisites.
    # Each consumer validates only its own required raw inputs.
    return root, manifest
