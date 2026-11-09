# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Collect real paired measurements; semantic pytest cases judge them independently."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmarks.lingbot_world.regression import RegressionError, read_json, sha256, write_json
from vllm_omni.benchmarks.realtime_video.client import resolve_image_reference, run_session

ROOT = Path(__file__).resolve().parents[3]
ENGINE_ENVIRONMENT = {"VLLM_OMNI_AR_DIFFUSION_KV_GATHER": "0", "AR_DIFFUSION_KV_SCRATCH_BLOCKS_PER_BRANCH": "0"}


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
    # Keep the subprocess's inner initialization deadlines aligned with the
    # explicit profile budget used by the outer readiness wait.
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
    lock = read_json(ROOT / "tests/dfx/lingbot_world/assets.lock.json")
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
        raise RegressionError("--lingbot-reference-root is required for real paired measurements")
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
    from tests.dfx.lingbot_world.prepare import verify_inference_runtime

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
            str(settings.get("measured_sessions", 1)),
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


def collect_measurements(settings, measurement_repeat=None):
    from tests.helpers.runtime import OmniServer

    if measurement_repeat is not None and (
        type(measurement_repeat) is not int or measurement_repeat not in range(3) or settings["platform"] != "npu"
    ):
        raise RegressionError("Split measurements require NPU repeat 0, 1 or 2")
    output = settings["output"]
    if any((output / role).exists() for role in ("reference", "candidate")):
        raise RegressionError("Use a fresh measurement output")
    model, images = checked_assets(settings)
    reference = validate_reference(settings)
    repeats = [measurement_repeat] if measurement_repeat is not None else list(range(3))
    report = {
        "schema_version": 1,
        "complete": False,
        "candidate_revision": git_revision(ROOT),
        "reference_revision": settings["reference_revision"],
        "repeat": measurement_repeat,
        "contract": {
            "hardware": hardware_contract(settings),
            "profile": read_json(settings["profile_path"]),
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
    if measurement_repeat is not None:
        write_json(output / "measurement.json", report)
    return report


def aggregate_measurements(settings):
    source, output = settings["generation"], settings["output"]
    if source is None or source.resolve().is_relative_to(output) or output.is_relative_to(source.resolve()):
        raise RegressionError("Use disjoint input/output measurement directories")
    if any((output / role).exists() for role in ("reference", "candidate")):
        raise RegressionError("Use a fresh aggregate output directory")
    report = {
        "schema_version": 1,
        "complete": False,
        "candidate_revision": git_revision(ROOT),
        "reference_revision": settings["reference_revision"],
        "repeat": None,
        "contract": None,
        "artifacts": {},
        "errors": [],
    }
    complete_parts = 0
    try:
        for repeat in range(3):
            root = source / f"pair-{repeat}"
            try:
                part = read_json(root / "measurement.json")
                validate_manifest(settings, part)
                if type(part.get("repeat")) is not int or part["repeat"] != repeat:
                    raise RegressionError("Wrong measurement repeat")
                if report["contract"] is None:
                    report["contract"] = part["contract"]
                elif part["contract"] != report["contract"]:
                    raise RegressionError("Measurement hardware/runtime contracts differ")
            except (RegressionError, OSError, KeyError) as error:
                report["errors"].append({"repeat": repeat, "error": str(error)})
                continue
            complete_parts += part.get("complete") is True
            report["errors"].extend({"repeat": repeat, "producer_error": error} for error in part.get("errors", []))
            for name in sorted(generation_paths(settings, [repeat])):
                try:
                    path = checked_artifact(root, part, name)
                    destination = output / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, destination)
                    report["artifacts"][name] = part["artifacts"][name]
                except (RegressionError, OSError) as error:
                    report["errors"].append({"repeat": repeat, "artifact": name, "error": str(error)})
        report["complete"] = (
            complete_parts == 3 and not report["errors"] and generation_paths(settings) <= report["artifacts"].keys()
        )
    finally:
        write_json(output / "generation.json", report)
    return report


def validate_manifest(settings, manifest):
    contract = manifest.get("contract", {})
    if (
        manifest.get("schema_version") != 1
        or manifest.get("candidate_revision") != git_revision(ROOT)
        or manifest.get("reference_revision") != settings["reference_revision"]
        or contract.get("profile") != read_json(settings["profile_path"])
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


def stability_session(server, settings: dict[str, Any], image: Path, destination: Path) -> dict[str, Any]:
    """Separate instrumented rollout for control updates and bounded memory."""
    from dataclasses import replace

    from vllm_omni.benchmarks.realtime_video.workload import default_workload
    from vllm_omni.platforms import current_omni_platform as platform

    memory = []

    def observe(record):
        if record.index >= settings["warmup_chunks"] and (
            record.index % 16 == 0 or record.index == settings["num_chunks"] - 1
        ):
            used = []
            for device in range(settings["dies"]):
                with platform.device(device):
                    free, total = platform.get_device_memory()
                used.append(total - free)
            memory.append({"chunk": record.index, "used_bytes": used})

    workload = default_workload(
        prompt=settings["cases"][0]["prompt"],
        seed=settings["cases"][0]["seed"],
        image_reference=resolve_image_reference(str(image)),
        num_chunks=settings["num_chunks"],
        width=settings["width"],
        height=settings["height"],
        fps=settings["mux_fps"],
    )
    # LingBot supports camera interactions, not prompt replacement. Scripted
    # trajectories and live camera input are mutually exclusive in the pipeline.
    workload = replace(workload, extra_params={"camera_action_script": None})
    camera_update = {
        "type": "session.interaction",
        "interaction": {
            "event_id": "camera-chunk-2",
            "event": {
                "multi_modal_data": {
                    "camera": {
                        "mode": "target",
                        "data": {"translation": [0.0, 0.0, 1.0], "rotation": [0.0, 0.0, 0.0, 1.0]},
                    }
                }
            },
            "transition_chunks": 2,
        },
    }
    result = asyncio.run(
        run_session(
            f"ws://{server.host}:{server.port}/v1/realtime/video",
            server.model,
            workload,
            first_chunk_timeout=settings["first_chunk_timeout_s"],
            chunk_timeout=settings["chunk_timeout_s"],
            ping_interval=20,
            collect_media=False,
            print_chunks=False,
            on_chunk=observe,
            interactions_after_chunk={2: camera_update},
        )
    )
    if (
        len(result.records) != settings["num_chunks"]
        or [r.index for r in result.records] != list(range(settings["num_chunks"]))
        or any(r.num_frames != (9 if i == 0 else 12) or r.byte_length <= 0 for i, r in enumerate(result.records))
    ):
        raise RegressionError("Incomplete long-session chunk/frame population")
    write_json(
        destination,
        {
            "events": result.events,
            "memory": memory,
            "timing_scope": "instrumented; excluded from performance",
            "request_id": result.request_id,
            "chunks": [row.to_dict() for row in result.records],
        },
    )
    applied = [
        event
        for event in result.events
        if any(
            "camera-chunk-2" in event.get(key, [])
            for key in ("started_event_ids", "active_event_ids", "completed_event_ids")
        )
    ]
    if not applied:
        raise RegressionError("Camera update was not acknowledged as applied in the streamed metadata")
    if len(memory) < 2:
        raise RegressionError("Insufficient post-warmup memory observations")
    for device in range(settings["dies"]):
        samples = [entry["used_bytes"][device] for entry in memory]
        if max(samples) - min(samples) > settings["memory_growth_budget_bytes"]:
            raise RegressionError(f"Unbounded memory growth on device {device}: {samples}")
    return {"memory": memory, "events": len(result.events), "request_id": result.request_id}


def run_stability(settings):
    import pytest

    from tests.e2e.online_serving import test_lingbot_world_v2_stepwise as lifecycle
    from tests.helpers.runtime import OmniServer

    if settings["num_chunks"] != 600 or settings.get("measured_sessions") != 2 or len(settings["cases"]) != 1:
        raise RegressionError("Long stability requires two candidate 600-chunk sessions")
    output = settings["output"]
    report = {
        "passed": False,
        "candidate_revision": git_revision(ROOT),
        "sessions": [],
        "errors": [],
        "profile": read_json(settings["profile_path"]),
        "timing_scope": "instrumented stability, not performance",
    }
    try:
        model, images = checked_assets(settings)
        report["hardware"] = hardware_contract(settings)
        image = images[settings["cases"][0]["image"]]
        with OmniServer(
            str(model),
            server_arguments(settings),
            env_dict=server_environment(ROOT, cpu_threads=settings.get("cpu_threads_per_worker")),
            startup_timeout=settings["startup_timeout_s"],
        ) as server:
            for index in range(2):
                row = stability_session(server, settings, image, output / f"session-{index}.json")
                if row["request_id"] in {r["request_id"] for r in report["sessions"]}:
                    raise RegressionError("Stability sessions reused a request")
                report["sessions"].append(row)
                write_json(output / "stability.json", report)
            for device in range(settings["dies"]):
                samples = [m["used_bytes"][device] for r in report["sessions"] for m in r["memory"]]
                if max(samples) - min(samples) > settings["memory_growth_budget_bytes"]:
                    raise RegressionError("Memory growth across stability sessions exceeds the profile budget")
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(lifecycle, "_IMAGE_PATH_OVERRIDE", str(image))
                lifecycle.test_lingbot_world_v2_stepwise_streams_one_chunk_per_block(server)
                lifecycle.test_lingbot_world_v2_stepwise_paced_playback_bounds_lead(server)
                lifecycle.test_lingbot_world_v2_stepwise_stop_during_paced_wait_releases_session(server)
        assert_device_cleanup(settings["dies"], output / "cleanup.json")
        report["passed"] = True
    except Exception as error:
        report["errors"].append(str(error))
    finally:
        write_json(output / "stability.json", report)
    return report
