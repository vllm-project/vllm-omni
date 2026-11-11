# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Generate real accuracy references; each pytest case judges its own inputs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmarks.lingbot_world.regression import RegressionError, read_json, sha256, write_json
from vllm_omni.benchmarks.realtime_video.client import resolve_image_reference

ROOT = Path(__file__).resolve().parents[4]
ENGINE_ENVIRONMENT = {"VLLM_OMNI_AR_DIFFUSION_KV_GATHER": "0", "AR_DIFFUSION_KV_SCRATCH_BLOCKS_PER_BRANCH": "0"}
CAPTURE_RECIPE = {"independent_starts_per_source": 3, "sessions_per_case": 1}


def load_settings(profile: Path, output: Path, *, reference_root=None, generation=None):
    settings = read_json(profile)
    settings.update(
        profile_path=profile.resolve(),
        output=output.resolve(),
        reference_root=reference_root,
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


def record_benchmark_revisions(document: dict[str, Any], source: Path) -> None:
    """Bind client measurements to this harness and the separate server source."""
    revision = git_revision(ROOT)
    if "git_revision" not in document or document["git_revision"] not in (None, revision):
        raise RegressionError("Benchmark client revision differs from the attested harness")
    # Source images omit .git, so the standalone client deliberately reports
    # None. This harness can attest its image's full Buildkite commit without
    # letting an installed standalone wheel claim an unrelated checkout.
    document["git_revision"] = revision
    document["server_revision"] = git_revision(source)


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


def validate_reference(settings: dict[str, Any]) -> Path:
    root = settings["reference_root"]
    if root is None:
        raise RegressionError("Accuracy generation requires the pinned reference checkout")
    root = root.resolve()
    if root == ROOT or git_revision(root) != settings["reference_revision"]:
        raise RegressionError("Reference must be a separate checkout at the frozen revision")
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)
    if dirty.strip():
        raise RegressionError("Reference checkout must be clean")
    return root


def _capture(source: Path, model: Path, destination: Path, settings: dict[str, Any]) -> None:
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("capture_precision.py")),
            "--source-root",
            str(source),
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
        cwd=source,
        env={**os.environ, **server_environment(source, cpu_threads=settings.get("cpu_threads_per_worker"))},
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


def generation_paths(settings, repeats=range(3)):
    return {
        f"{role}/{repeat}/{name}"
        for role in ("reference", "candidate")
        for repeat in repeats
        for name in [
            "precision.npz",
            *[
                f"{case['id']}/{file}"
                for case in settings["cases"]
                for file in ("serve-result.json", "benchmark.json", "rollout.mp4")
            ],
        ]
    }


def collect_measurements(settings):
    from tests.helpers.runtime import OmniServer

    output = settings["output"]
    if any((output / role).exists() for role in ("reference", "candidate")):
        raise RegressionError("Use a fresh measurement output")
    model, images = checked_assets(settings)
    reference = validate_reference(settings)
    repeats = range(3)
    report = {
        "schema_version": 1,
        "complete": False,
        "candidate_revision": git_revision(ROOT),
        "reference_revision": settings["reference_revision"],
        "contract": {
            "hardware": hardware_contract(settings),
            "profile": read_json(settings["profile_path"]),
            "capture_recipe": dict(CAPTURE_RECIPE),
            "model_assets_sha256": sha256(ROOT / "tests/e2e/accuracy/lingbot_world/assets.lock.json"),
            "deploy_sha256": sha256(ROOT / settings["deploy_config"]),
            "engine_environment": dict(ENGINE_ENVIRONMENT),
        },
        "errors": [],
        "artifacts": {},
    }
    write_json(output / "generation.json", report)
    for repeat in repeats:
        for role in ("reference", "candidate") if repeat % 2 == 0 else ("candidate", "reference"):
            source = reference if role == "reference" else ROOT
            arm = output / role / str(repeat)
            arm.mkdir(parents=True)
            try:
                _capture(source, model, arm / "precision.npz", settings)
            except (OSError, subprocess.SubprocessError) as error:
                report["errors"].append({"role": role, "repeat": repeat, "component": "precision", "error": str(error)})
            try:
                with OmniServer(
                    str(model),
                    server_arguments(settings),
                    source_root=str(source),
                    env_dict=server_environment(source, cpu_threads=settings.get("cpu_threads_per_worker")),
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
                            record_benchmark_revisions(document, source)
                            write_json(destination / "benchmark.json", document)
                            if not document["passed"]:
                                raise RegressionError(f"Incomplete realtime-video measurement: {document['errors']}")
                        except Exception as error:
                            report["errors"].append(
                                {
                                    "role": role,
                                    "repeat": repeat,
                                    "case": case["id"],
                                    "component": "stream",
                                    "error": str(error),
                                }
                            )
            except Exception as error:
                report["errors"].append({"role": role, "repeat": repeat, "component": "server", "error": str(error)})
            try:
                assert_device_cleanup(settings["dies"], arm / "cleanup.json")
            except Exception as error:
                report["errors"].append({"role": role, "repeat": repeat, "component": "cleanup", "error": str(error)})
            report["artifacts"] = {
                str(p.relative_to(output)): sha256(p)
                for role in ("reference", "candidate")
                for p in (output / role).rglob("*")
                if p.is_file()
            }
            write_json(output / "generation.json", report)
    report["complete"] = not report["errors"] and generation_paths(settings, repeats) <= report["artifacts"].keys()
    write_json(output / "generation.json", report)
    return report


def validate_manifest(settings, manifest):
    contract = manifest.get("contract", {})
    hardware = contract.get("hardware", {})
    if (
        manifest.get("schema_version") != 1
        or manifest.get("candidate_revision") != git_revision(ROOT)
        or manifest.get("reference_revision") != settings["reference_revision"]
        or contract.get("profile") != read_json(settings["profile_path"])
        or contract.get("capture_recipe") != CAPTURE_RECIPE
        or contract.get("model_assets_sha256") != sha256(ROOT / "tests/e2e/accuracy/lingbot_world/assets.lock.json")
        or hardware.get("platform") != settings["platform"]
        or hardware.get("cards") != settings["cards"]
        or hardware.get("dies") != settings["dies"]
        or len(hardware.get("devices", [])) != settings["dies"]
        or contract.get("deploy_sha256") != sha256(ROOT / settings["deploy_config"])
        or contract.get("engine_environment") != ENGINE_ENVIRONMENT
    ):
        raise RegressionError("Measurement source/profile/runtime contract mismatch")


def checked_artifact(root, manifest, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or name not in manifest.get("artifacts", {}):
        raise RegressionError(f"Missing or unsafe measurement artifact: {name}")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise RegressionError("Artifact escapes the measurement directory")
    if sha256(path) != manifest["artifacts"][name]:
        raise RegressionError(f"Measurement artifact checksum mismatch: {name}")
    return path


def measurement_data(settings):
    root = settings["generation"]
    if root is None:
        raise RegressionError("--lingbot-generation must name real collected measurements")
    manifest = read_json(root / "generation.json")
    validate_manifest(settings, manifest)
    # Completion and semantic verdicts are deliberately not shared prerequisites.
    # Each consumer validates only its own required raw inputs.
    return root, manifest
