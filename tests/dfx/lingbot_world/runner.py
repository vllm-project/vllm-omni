# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real reference/candidate generation using the existing server lifecycle."""

from __future__ import annotations

import asyncio
import importlib.metadata
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmarks.lingbot_world.benchmark_lingbot_world_realtime import (
    parse_args,
    resolve_image_reference,
    run_benchmark,
    run_session,
)
from benchmarks.lingbot_world.precision import compare_tensor_samples
from benchmarks.lingbot_world.regression import (
    RegressionError,
    calibrate,
    check_performance,
    performance_metrics,
    read_json,
    sha256,
    verify_reference_calibration_contract,
    write_json,
)

ROOT = Path(__file__).resolve().parents[3]
ENGINE_ENVIRONMENT = {
    "VLLM_OMNI_AR_DIFFUSION_KV_GATHER": "0",
    "AR_DIFFUSION_KV_SCRATCH_BLOCKS_PER_BRANCH": "0",
}


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


def git_revision(root: Path) -> str:
    if not (root / ".git").exists() and root == ROOT:
        # CI Docker contexts exclude .git; the image tag is the Buildkite SHA.
        revision = os.environ.get("BUILDKITE_COMMIT", "")
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise RegressionError("Source image has no Git metadata or full BUILDKITE_COMMIT")
        return revision
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


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


def hardware_contract(settings: dict[str, Any]) -> dict[str, Any]:
    from vllm_omni.platforms import current_omni_platform as platform

    if platform.device_type != settings["platform"] or platform.get_device_count() != settings["dies"]:
        raise RegressionError("Wrong platform or visible die count for the requested profile")
    allocation = None
    if settings.get("reserved_cards", settings["cards"]) != settings["cards"]:
        path = os.environ.get("LINGBOT_ALLOCATION_FILE")
        if path is None:
            raise RegressionError("Reserved-device profiles require the launcher allocation preflight")
        allocation = read_json(Path(path))
        npu = settings["platform"] == "npu"
        identity_key = "runtime_id" if npu else "uuid"
        visibility_key = "ASCEND_RT_VISIBLE_DEVICES" if npu else "CUDA_VISIBLE_DEVICES"
        identities = {
            device.get(identity_key)
            for device in allocation.get("devices", [])
            if isinstance(device, dict) and isinstance(device.get(identity_key), str) and device.get(identity_key)
        }
        if (
            allocation.get("reserved_cards") != settings["reserved_cards"]
            or allocation.get("visible_dies") != settings["dies"]
            or len(allocation.get("devices", [])) != settings["reserved_cards"]
            or len(identities) != settings["reserved_cards"]
            or allocation.get("visible_devices") != os.environ.get(visibility_key)
        ):
            raise RegressionError("Device allocation evidence does not match the requested profile/visibility")
        if npu:
            from tests.dfx.lingbot_world.prepare import select_npu_devices

            original = ",".join(device["runtime_id"] for device in allocation["devices"])
            selected = select_npu_devices(
                len(allocation["devices"]), original, dies=settings["dies"], reserved_cards=settings["reserved_cards"]
            )
            if allocation.get("platform") != "npu" or selected != os.environ.get(visibility_key):
                raise RegressionError("NPU allocation evidence does not preserve the original device mapping")
            for index, device in enumerate(allocation["devices"][: settings["dies"]]):
                if device.get("name") != platform.get_device_name(index) or device.get(
                    "total_bytes"
                ) != platform.get_device_total_memory(index):
                    raise RegressionError("NPU allocation evidence does not match the visible hardware")
    devices = []
    for index in range(settings["dies"]):
        devices.append(
            {"name": platform.get_device_name(index), "total_bytes": platform.get_device_total_memory(index)}
        )
    packages = {}
    for package in ("torch", "vllm", "diffusers", "transformers", "torch-npu", "vllm-ascend"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            if package not in {"torch-npu", "vllm-ascend"} or settings["platform"] == "npu":
                raise
    return {
        "platform": platform.device_type,
        "cards": settings["cards"],
        "dies": settings["dies"],
        "devices": devices,
        "packages": packages,
        **({"allocation": allocation} if allocation is not None else {}),
    }


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


def diagnostic_session(server, settings: dict[str, Any], image: Path, destination: Path) -> None:
    """Separate instrumented rollout for control updates and bounded memory."""
    from dataclasses import replace

    from benchmarks.lingbot_world.workload import default_workload
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
    write_json(
        destination,
        {"events": result.events, "memory": memory, "timing_scope": "instrumented; excluded from performance"},
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


def _measured_metrics(document: dict, settings: dict) -> dict[str, float]:
    sessions = document.get("sessions", [])
    count = settings.get("measured_sessions", 1)
    ids = [session.get("request_id") for session in sessions]
    if (
        len(sessions) != count
        or [session.get("session_index") for session in sessions] != list(range(count))
        or any(not isinstance(identity, str) or not identity for identity in ids)
        or len(set(ids)) != count
    ):
        raise RegressionError("Missing or duplicated measured sessions in an independent start")
    if any(document["config"].get(key) != settings[key] for key in ("warmup_chunks", "target_fps")):
        raise RegressionError("Benchmark measurement policy differs from the profile")
    if document["config"].get("playback_buffer_chunks") != 1:
        raise RegressionError("LingBot regression requires the pinned one-chunk playback buffer")
    return performance_metrics(document, num_chunks=settings["num_chunks"], min_steady=settings["min_steady_samples"])


def generate(settings: dict[str, Any], *, measurement_repeat: int | None = None) -> dict[str, Any]:
    """Collect all starts, or one NPU pair for a separately gated long campaign."""
    if measurement_repeat is not None and (
        type(measurement_repeat) is not int or measurement_repeat not in range(3) or settings["platform"] != "npu"
    ):
        raise RegressionError("Measurement collection requires an NPU repeat index 0, 1 or 2")
    from tests.helpers.runtime import OmniServer

    output = settings["output"]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "reference").exists() or (output / "candidate").exists():
        raise RegressionError("Use a fresh output directory; never mix independent campaigns")
    write_json(output / "generation.json", {"passed": False, "status": "running"})
    model, images = checked_assets(settings)
    reference = validate_reference(settings)
    hardware = hardware_contract(settings)
    deployment = ROOT / settings["deploy_config"]
    config = read_json(settings["profile_path"])
    contract = {
        "hardware": hardware,
        "profile": config,
        "deploy_sha256": sha256(deployment),
        "engine_environment": dict(ENGINE_ENVIRONMENT),
    }
    write_json(output / "contract.json", contract)
    records: dict[str, Any] = {"reference": [], "candidate": []}
    for repeat in [measurement_repeat] if measurement_repeat is not None else range(3):
        # Alternate AB/BA to reduce bias from thermal/host-load drift.
        order = ("reference", "candidate") if repeat % 2 == 0 else ("candidate", "reference")
        for role in order:
            source = reference if role == "reference" else ROOT
            arm = output / role / str(repeat)
            arm.mkdir(parents=True, exist_ok=True)
            _capture(source, model, arm / "precision.npz", settings)
            with OmniServer(
                str(model),
                server_arguments(settings),
                source_root=str(source),
                env_dict=server_environment(source, cpu_threads=settings.get("cpu_threads_per_worker")),
                startup_timeout=settings["startup_timeout_s"],
            ) as server:
                process = server.proc
                assert process is not None
                import psutil

                children = [(p.pid, p.create_time()) for p in psutil.Process(process.pid).children(recursive=True)]
                case_results = {}
                for case in settings["cases"]:
                    destination = arm / case["id"]
                    destination.mkdir(parents=True, exist_ok=True)
                    args = parse_args(
                        [
                            "--port",
                            str(server.port),
                            "--model",
                            str(model),
                            "--image",
                            str(images[case["image"]]),
                            "--prompt",
                            case["prompt"],
                            "--camera-pattern",
                            case["camera_pattern"],
                            "--seed",
                            str(case["seed"]),
                            "--height",
                            str(settings["height"]),
                            "--width",
                            str(settings["width"]),
                            "--fps",
                            str(settings["mux_fps"]),
                            "--target-fps",
                            str(settings["target_fps"]),
                            "--playback-buffer-chunks",
                            "1",
                            "--num-chunks",
                            str(settings["num_chunks"]),
                            "--sessions",
                            str(settings.get("measured_sessions", 1)),
                            "--warmup-sessions",
                            "1",
                            "--warmup-num-chunks",
                            str(settings["warmup_num_chunks"]),
                            "--warmup-chunks",
                            str(settings["warmup_chunks"]),
                            "--first-chunk-timeout",
                            str(settings["first_chunk_timeout_s"]),
                            "--chunk-timeout",
                            str(settings["chunk_timeout_s"]),
                            "--output-json",
                            str(destination / "benchmark.json"),
                            "--save-video",
                            str(destination / "rollout.mp4"),
                        ]
                    )
                    document = asyncio.run(run_benchmark(args))
                    # Server revision, not the client harness revision.
                    document["server_revision"] = git_revision(source)
                    write_json(destination / "benchmark.json", document)
                    case_results[case["id"]] = _measured_metrics(document, settings)
                # Reuse lifecycle tests against this exact server, with the
                # same pinned first frame; no second untracked service launch.
                if settings["lifecycle_checks"]:
                    import pytest

                    from tests.e2e.online_serving import test_lingbot_world_v2_stepwise as lifecycle

                    with pytest.MonkeyPatch.context() as patch:
                        patch.setattr(lifecycle, "_IMAGE_PATH_OVERRIDE", str(images[settings["cases"][0]["image"]]))
                        lifecycle.test_lingbot_world_v2_stepwise_streams_one_chunk_per_block(server)
                        lifecycle.test_lingbot_world_v2_stepwise_paced_playback_bounds_lead(server)
                        lifecycle.test_lingbot_world_v2_stepwise_stop_during_paced_wait_releases_session(server)
                if repeat == 0:
                    diagnostic_session(server, settings, images[settings["cases"][0]["image"]], arm / "diagnostic.json")
            lingering = []
            for pid, created in children:
                if psutil.pid_exists(pid):
                    child = psutil.Process(pid)
                    if child.create_time() == created and child.status() != psutil.STATUS_ZOMBIE:
                        lingering.append(pid)
            if lingering:
                raise RegressionError(f"Server children survived teardown: {lingering}")
            records[role].append(case_results)
            write_json(output / "progress.json", {"records": records, "contract": contract})
    if measurement_repeat is not None:
        measurement = {
            "measurement_only": True,
            "complete": True,
            "repeat": measurement_repeat,
            "contract": contract,
            "records": records,
            "reference_revision": settings["reference_revision"],
            "candidate_revision": git_revision(ROOT),
            "buildkite_job_id": os.environ.get("BUILDKITE_JOB_ID"),
            "artifacts": {
                str(path.relative_to(output)): sha256(path)
                for role in ("reference", "candidate")
                for path in (output / role).rglob("*")
                if path.is_file()
            },
        }
        write_json(output / "measurement.json", measurement)
        write_json(output / "generation.json", {"passed": False, "status": "awaiting_aggregate"})
        return measurement
    return _finalize_measurements(settings, contract, records)


def _finalize_measurements(settings: dict[str, Any], contract: dict, records: dict) -> dict[str, Any]:
    output = settings["output"]
    cases = {case["id"] for case in settings["cases"]}
    if any(len(records.get(role, [])) != 3 for role in ("reference", "candidate")):
        raise RegressionError("Final gates require three independent starts per role")
    if any(set(row) != cases for role in ("reference", "candidate") for row in records[role]):
        raise RegressionError("Measurement cases do not cover the complete profile")
    startup_stall_floor = 0.0
    ttfc_floor = 0.0
    p95_max_cv: float | None = None
    calibration_error: str | None = None
    if settings.get("platform") == "cuda":
        calibration_path = ROOT / "tests/dfx/lingbot_world/calibration/cuda_nightly_reference.json"
        calibration = read_json(calibration_path)
        write_json(output / "reference-calibration.json", calibration)
        try:
            verify_reference_calibration_contract(contract, calibration)
        except RegressionError as error:
            calibration_error = str(error)
            write_json(
                output / "calibration-error.json",
                {"error": calibration_error, "reference_record_sha256": sha256(calibration_path)},
            )
        else:
            startup_stall_floor = calibration["startup_stall_calibration"]["selected_floor_ms"]
            ttfc_floor = calibration["ttfc_calibration"]["selected_floor_ms"]
            p95_max_cv = calibration["p95_reference_cv_calibration"]["selected_ceiling"]
            write_json(
                output / "p95-cv-calibration.json",
                {"reference_record_sha256": sha256(calibration_path), **calibration["p95_reference_cv_calibration"]},
            )
            write_json(
                output / "ttfc-calibration.json",
                {"reference_record_sha256": sha256(calibration_path), **calibration["ttfc_calibration"]},
            )
            write_json(
                output / "startup-stall-calibration.json",
                {
                    "reference_record_sha256": sha256(calibration_path),
                    **calibration["startup_stall_calibration"],
                },
            )
    report: dict[str, Any] = {
        "contract": contract,
        "cases": {},
        "passed": True,
        "reference_revision": settings["reference_revision"],
        "candidate_revision": git_revision(ROOT),
    }
    for case in settings["cases"]:
        case_id = case["id"]
        try:
            if calibration_error is not None:
                raise RegressionError(calibration_error)
            baseline = calibrate(
                [row[case_id] for row in records["reference"]],
                contract=contract,
                reference_revision=settings["reference_revision"],
                max_cv=settings["max_reference_cv"],
                regression_fraction=settings["performance_regression_fraction"],
                transient_noise_sigma=settings["transient_noise_sigma"],
                startup_stall_jitter_floor_ms=startup_stall_floor,
                ttfc_jitter_floor_ms=ttfc_floor,
                p95_max_cv=p95_max_cv,
            )
        except RegressionError as error:
            # Invalid references still fail acceptance. Preserve the remaining
            # cases, precision and hashes so independent scoring can diagnose it.
            report["cases"][case_id] = {"baseline": None, "checks": [], "calibration_error": str(error)}
            report["passed"] = False
            continue
        checks = [check_performance(row[case_id], baseline, contract=contract) for row in records["candidate"]]
        report["cases"][case_id] = {"baseline": baseline, "checks": checks}
        report["passed"] &= all(check["passed"] for check in checks)
    # A separate repeatability gate precedes reference-vs-candidate precision.
    precision_checks = []
    for role in ("reference", "candidate"):
        for repeat in (1, 2):
            precision_checks.append(
                compare_tensor_samples(
                    output / role / "0/precision.npz", output / role / str(repeat) / "precision.npz", atol=0, rtol=0
                )
            )
    for repeat in range(3):
        precision_checks.append(
            compare_tensor_samples(
                output / "reference/0/precision.npz",
                output / "candidate" / str(repeat) / "precision.npz",
                atol=settings["precision_atol"],
                rtol=settings["precision_rtol"],
            )
        )
    report["precision"] = precision_checks
    report["passed"] &= all(row["status"] == "passed" for row in precision_checks)
    report["artifacts"] = {
        str(path.relative_to(output)): sha256(path)
        for path in output.rglob("*")
        if path.is_file() and path.suffix in {".mp4", ".npz"}
    }
    write_json(output / "generation.json", report)
    return report


def aggregate_measurements(settings: dict[str, Any]) -> dict[str, Any]:
    """Verify three complete NPU pairs before applying the full regression gate."""
    source = settings["generation"]
    output = settings["output"].resolve()
    if source is None or settings["platform"] != "npu":
        raise RegressionError("NPU aggregation requires a measurement artifact directory")
    source = source.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise RegressionError("Aggregate input and output directories must be disjoint")
    output.mkdir(parents=True, exist_ok=True)
    if any((output / role).exists() for role in ("reference", "candidate")):
        raise RegressionError("Use a fresh aggregate output directory")
    write_json(output / "generation.json", {"passed": False, "status": "aggregating"})
    revision = git_revision(ROOT)
    profile = read_json(settings["profile_path"])
    cases = {case["id"] for case in settings["cases"]}
    contract = None
    records: dict[str, list] = {"reference": [], "candidate": []}
    provenance = []
    request_ids: set[str] = set()
    for repeat in range(3):
        root = source / f"pair-{repeat}"
        if not (root / "measurement.json").is_file():
            raise RegressionError(f"Missing measurement pair {repeat}")
        manifest = read_json(root / "measurement.json")
        if (
            manifest.get("complete") is not True
            or manifest.get("measurement_only") is not True
            or type(manifest.get("repeat")) is not int
            or manifest["repeat"] != repeat
            or manifest.get("candidate_revision") != revision
            or manifest.get("reference_revision") != settings["reference_revision"]
        ):
            raise RegressionError(f"Incomplete, duplicate or wrong-revision measurement pair {repeat}")
        current = manifest["contract"]
        if current.get("engine_environment") != ENGINE_ENVIRONMENT:
            raise RegressionError(f"Measurement engine environment is missing or changed in pair {repeat}")
        hardware = current["hardware"]
        if (
            current["profile"] != profile
            or current["deploy_sha256"] != sha256(ROOT / settings["deploy_config"])
            or any(hardware.get(key) != settings[key] for key in ("platform", "cards", "dies"))
            or len(hardware.get("devices", [])) != settings["dies"]
        ):
            raise RegressionError(f"Measurement profile or hardware mismatch in pair {repeat}")
        if contract is not None and current != contract:
            raise RegressionError("Measurement hardware/dependency contracts differ between pairs")
        contract = current
        required = {f"{role}/{repeat}/precision.npz" for role in records}
        required |= {
            f"{role}/{repeat}/{case}/{name}"
            for role in records
            for case in cases
            for name in ("benchmark.json", "rollout.mp4")
        }
        if repeat == 0:
            required |= {f"{role}/0/diagnostic.json" for role in records}
        artifacts = manifest["artifacts"]
        if not required.issubset(artifacts):
            raise RegressionError(f"Missing measurement artifacts in pair {repeat}")
        for relative, digest in artifacts.items():
            path = Path(relative)
            if (
                path.is_absolute()
                or ".." in path.parts
                or len(path.parts) < 3
                or path.parts[0] not in records
                or path.parts[1] != str(repeat)
            ):
                raise RegressionError(f"Unsafe or wrong-repeat measurement path: {relative}")
            if sha256(root / path) != digest:
                raise RegressionError(f"Measurement artifact checksum mismatch: {relative}")
            destination = output / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / path, destination)
        for role in records:
            completed = manifest["records"].get(role, [])
            if len(completed) != 1 or set(completed[0]) != cases:
                raise RegressionError(f"Incomplete case population in {role} pair {repeat}")
            metrics = {}
            for case in cases:
                raw = read_json(output / role / str(repeat) / case / "benchmark.json")
                server_revision = settings["reference_revision"] if role == "reference" else revision
                if raw.get("git_revision") != revision or raw.get("server_revision") != server_revision:
                    raise RegressionError(f"Wrong benchmark producer/server revision in {role} pair {repeat}")
                # Recompute from chunks; a claimed aggregate cannot set its own limits.
                metrics[case] = _measured_metrics(raw, settings)
                identities = {session["request_id"] for session in raw["sessions"]}
                if identities & request_ids:
                    raise RegressionError("Measured requests were reused across independent starts")
                request_ids.update(identities)
            records[role].append(metrics)
        provenance.append({"repeat": repeat, "buildkite_job_id": manifest.get("buildkite_job_id")})
    assert contract is not None
    write_json(output / "progress.json", {"records": records, "contract": contract})
    write_json(output / "measurement-jobs.json", provenance)
    report = _finalize_measurements(settings, contract, records)
    report["measurement_jobs"] = provenance
    write_json(output / "generation.json", report)
    return report
