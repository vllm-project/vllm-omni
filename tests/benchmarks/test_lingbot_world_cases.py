# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Consumer isolation and malformed actual-measurement contracts, without model substitutes."""

import copy
import shutil
from pathlib import Path

import numpy as np
import pytest

from benchmarks.lingbot_world.precision import save_tensor_sample
from benchmarks.lingbot_world.regression import RegressionError, read_json, sha256, write_json
from tests.benchmarks.test_lingbot_world_regression import document
from tests.dfx.lingbot_world import checks, prepare, quality, runner
from vllm_omni.benchmarks.realtime_video.metrics import measured_metrics

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def measurements(tmp_path, monkeypatch):
    profile = tmp_path / "profile.json"
    deploy = tmp_path / "deploy.yaml"
    deploy.write_text("pipeline: existing\n")
    settings = {
        "platform": "cuda",
        "cases": [{"id": "scene"}],
        "num_chunks": 40,
        "height": 480,
        "width": 832,
        "mux_fps": 16,
        "target_fps": 12,
        "warmup_chunks": 6,
        "warmup_num_chunks": 8,
        "min_steady_samples": 33,
        "max_reference_cv": 0.03,
        "performance_regression_fraction": 0.05,
        "transient_noise_sigma": 3,
        "precision_atol": 0,
        "precision_rtol": 0,
        "reference_revision": "a" * 40,
        "deploy_config": str(deploy),
        "cards": 4,
        "dies": 4,
    }
    write_json(profile, settings)
    root = tmp_path / "raw"
    settings.update(profile_path=profile, generation=root, output=tmp_path / "judgments")
    monkeypatch.setattr(runner, "git_revision", lambda path: "c" * 40)
    manifest = {
        "schema_version": 1,
        "complete": True,
        "passed": False,
        "candidate_revision": "c" * 40,
        "reference_revision": "a" * 40,
        "repeat": None,
        "errors": [],
        "artifacts": {},
        "contract": {
            "hardware": {
                "platform": "cuda",
                "cards": 4,
                "dies": 4,
                "devices": ["H100"] * 4,
                "packages": {"vllm": "0.33.0", "torch": "2.14.0"},
            },
            "profile": read_json(profile),
            "deploy_sha256": sha256(deploy),
            "engine_environment": runner.ENGINE_ENVIRONMENT,
        },
    }
    for role in ("reference", "candidate"):
        for repeat in range(3):
            arm = root / role / str(repeat)
            scene = arm / "scene"
            scene.mkdir(parents=True)
            raw = document()
            raw["config"].update(warmup_sessions=1, warmup_num_chunks=8)
            raw["sessions"][0].update(session_index=0, request_id=f"{role}-{repeat}")
            raw["warmups"] = [
                {
                    "measured": False,
                    "request_id": f"warm-{role}-{repeat}",
                    "chunks": copy.deepcopy(raw["sessions"][0]["chunks"][:8]),
                }
            ]
            raw.update(git_revision="c" * 40, server_revision="a" * 40 if role == "reference" else "c" * 40)
            write_json(scene / "benchmark.json", raw)
            write_json(scene / "serve-result.json", {"realtime_video": raw})
            (scene / "rollout.mp4").write_bytes(b"video-fixture")
            save_tensor_sample(arm / "precision.npz", inputs={"latent": np.ones(2)}, outputs={"decoded": np.ones(2)})
    manifest["artifacts"] = {str(p.relative_to(root)): sha256(p) for p in root.rglob("*") if p.is_file()}
    write_json(root / "generation.json", manifest)
    return settings, (root, manifest)


def fake_native_workers(monkeypatch, tmp_path):
    """Stub only native scorer subprocesses; exercise real threshold and consumer code."""
    lock = {"repositories": {suite: {"revision": "e" * 40} for suite in ("vbench", "wbench")}, "files": {}}
    original = quality.read_json
    monkeypatch.setattr(
        quality, "read_json", lambda path: lock if path.name == "evaluators.lock.json" else original(path)
    )
    monkeypatch.setattr(quality, "verify_environment", lambda *args: {})
    monkeypatch.setattr(quality, "git_revision", lambda *args: "e" * 40)
    monkeypatch.setattr(quality.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setenv("LINGBOT_EVALUATOR_ROOT", str(tmp_path))
    monkeypatch.setenv("LINGBOT_EVALUATOR_PYTHON", "/unused/native-worker")
    monkeypatch.setattr(quality, "verify_video_coverage", lambda *args, **kwargs: {"decoded_frames": 477})

    def worker(videos, output, *, suite, dimensions, **kwargs):
        scores = {
            label: {name: (0 if suite == "wbench" and "candidate" in label else 1) for name in dimensions}
            for label in videos
        }
        stats = {
            "batch_size": 32,
            "encoded_images": 477,
            "forward_batches": 15,
            "max_forward_batch": 32,
            "gradients": False,
        }
        return {
            "status": "passed",
            "scores": scores,
            "protocol": {"dynamic_degree_backend": "raft"},
            "native_provenance": {label: {"background_clip_inference": stats} for label in videos},
        }

    monkeypatch.setattr(quality, "evaluate_video_quality", worker)


def fake_pixels(monkeypatch):
    def compare(*args, **kwargs):
        metadata = {"frame_count": 477, "width": 832, "height": 480, "fps": 16}
        metric = {"psnr_db": None, "psnr_is_infinite": True}
        return {
            "reference": metadata,
            "candidate": metadata,
            "decoded_frames": 477,
            "uniform_frame_indices": {"reference": [], "candidate": []},
            "total": metric,
            "windows": [{"index": 0, "start_frame": 0, "frame_count": 477, **metric}],
            "ssim": {"mean": 1},
        }

    monkeypatch.setattr(checks, "compare_video_pair", compare)


def test_wbench_failure_does_not_block_other_real_consumer_checks(measurements, monkeypatch, tmp_path):
    settings, data = measurements
    fake_native_workers(monkeypatch, tmp_path)
    fake_pixels(monkeypatch)
    assert not quality.evaluate_quality(settings, data, "wbench", monkeypatch)["passed"]
    assert quality.evaluate_quality(settings, data, "vbench", monkeypatch)["passed"]
    assert checks.numeric_report(settings, data)["passed"]
    assert checks.performance_report(settings, data)["passed"]
    assert all(
        (settings["output"] / f"{name}.json").is_file() for name in ("wbench", "vbench", "precision", "performance")
    )


def test_partial_aggregate_keeps_authenticated_inputs_for_independent_consumers(measurements, monkeypatch, tmp_path):
    settings, (raw_root, manifest) = measurements
    pairs = tmp_path / "pairs"
    for repeat in range(3):
        part = copy.deepcopy(manifest)
        part["repeat"] = repeat
        part["artifacts"] = {k: v for k, v in manifest["artifacts"].items() if k.split("/")[1] == str(repeat)}
        if repeat == 0:
            part["complete"] = False
            part["errors"] = ["precision capture failed"]
            del part["artifacts"]["candidate/0/precision.npz"]
        destination = pairs / f"pair-{repeat}"
        for name in part["artifacts"]:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(raw_root / name, target)
        write_json(destination / "measurement.json", part)
    aggregate = tmp_path / "aggregate"
    result = runner.aggregate_measurements({**settings, "generation": pairs, "output": aggregate})
    assert not result["complete"] and result["errors"]
    assert any(row.get("producer_error") == "precision capture failed" for row in result["errors"])
    assert (aggregate / "generation.json").is_file()
    settings = {**settings, "generation": aggregate}
    data = runner.measurement_data(settings)
    fake_native_workers(monkeypatch, tmp_path)
    fake_pixels(monkeypatch)
    assert checks.performance_report(settings, data)["passed"]
    assert quality.evaluate_quality(settings, data, "vbench", monkeypatch)["passed"]
    assert not checks.numeric_report(settings, data)["passed"]


@pytest.mark.parametrize("change", ["missing", "duplicate", "warmup", "nonfinite", "measured_failed", "warmup_failed"])
def test_performance_consumer_rejects_its_incomplete_population(measurements, change):
    settings, (root, manifest) = measurements
    name = "candidate/1/scene/benchmark.json"
    if change == "missing":
        del manifest["artifacts"][name]
    else:
        raw = read_json(root / name)
        if change == "duplicate":
            raw["sessions"][0]["request_id"] = "candidate-0"
        elif change == "warmup":
            raw["warmups"] = []
        elif change == "measured_failed":
            raw["sessions"][0]["success"] = False
        elif change == "warmup_failed":
            raw["warmups"][0]["success"] = False
        else:
            raw["sessions"][0]["chunks"][7]["arrival_s"] = "NaN"
        write_json(root / name, raw)
        manifest["artifacts"][name] = sha256(root / name)
    assert not checks.performance_report(settings, (root, manifest))["passed"]


def test_performance_uses_complete_timing_despite_unrelated_media_finalization_failure(measurements):
    settings, (root, manifest) = measurements
    name = "candidate/1/scene/benchmark.json"
    raw = read_json(root / name)
    raw["passed"] = False
    raw["errors"] = ["media finalization failed"]
    raw["sessions"][0]["success"] = True
    raw["warmups"][0]["success"] = True
    write_json(root / name, raw)
    manifest["artifacts"][name] = sha256(root / name)
    assert checks.performance_report(settings, (root, manifest))["passed"]


def test_valid_timing_and_npu_device_parser_execute_actual_helpers(measurements):
    settings, (root, _) = measurements
    raw = read_json(root / "candidate/0/scene/benchmark.json")
    assert measured_metrics(raw, settings)["steady_samples"] == 33
    assert prepare.select_npu_devices(8, "0,1,2,3,4,5,6,7", dies=4, reserved_cards=8) == "0,1,2,3"
    with pytest.raises(RegressionError):
        prepare.select_npu_devices(8, "0,0,2,3,4,5,6,7", dies=4, reserved_cards=8)


@pytest.mark.parametrize("change", ["valid", "ack", "growth", "samples"])
def test_stability_uses_observed_camera_and_memory_counters(tmp_path, monkeypatch, change):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from vllm_omni import platforms
    from vllm_omni.benchmarks.realtime_video.client import SessionResult
    from vllm_omni.benchmarks.realtime_video.workload import ChunkRecord

    current = {"chunk": 0}

    @contextmanager
    def device(index):
        yield

    def memory():
        used = 2**30 + (current["chunk"] * 2**20 if change == "growth" else 0)
        return 64 * 2**30 - used, 64 * 2**30

    monkeypatch.setattr(platforms, "current_omni_platform", SimpleNamespace(device=device, get_device_memory=memory))

    async def session(*args, on_chunk, **kwargs):
        records = [ChunkRecord(i, i + 1.0, 1.0, 1024, 9 if i == 0 else 12) for i in range(600)]
        for row in records:
            current["chunk"] = row.index
            if change != "samples" or row.index == 0:
                on_chunk(row)
        events = [] if change == "ack" else [{"active_event_ids": ["camera-chunk-2"]}]
        return SessionResult(
            request_id="stability-session",
            records=records,
            media=b"",
            ttfc_s=None,
            video_start_s=None,
            wall_s=600.0,
            done_event={},
            events=events,
        )

    monkeypatch.setattr(runner, "run_session", session)
    settings = {
        "num_chunks": 600,
        "warmup_chunks": 6,
        "dies": 2,
        "height": 480,
        "width": 832,
        "mux_fps": 16,
        "first_chunk_timeout_s": 900,
        "chunk_timeout_s": 120,
        "memory_growth_budget_bytes": 256 * 2**20,
        "cases": [{"prompt": "scene", "seed": 42}],
    }
    server = SimpleNamespace(host="localhost", port=8000, model="fixture")
    destination = tmp_path / "session.json"
    (tmp_path / "image.jpg").write_bytes(b"first-frame fixture")
    if change == "valid":
        report = runner.stability_session(server, settings, tmp_path / "image.jpg", destination)
        assert len(report["memory"]) >= 2 and report["events"] == 1
        assert len(read_json(destination)["chunks"]) == 600
    else:
        with pytest.raises(RegressionError):
            runner.stability_session(server, settings, tmp_path / "image.jpg", destination)
        assert destination.is_file()


def test_npu_three_pairs_use_shared_serving_bridge_and_preserve_source_environment(tmp_path, monkeypatch):
    """Only server/model boundaries are stubbed; parse the real serving options."""
    from contextlib import contextmanager
    from types import SimpleNamespace

    from tests.helpers import runtime
    from vllm_omni.benchmarks import serve
    from vllm_omni.benchmarks.realtime_video.serving import get_realtime_video_samples

    settings = read_json(runner.ROOT / "tests/dfx/lingbot_world/configs/npu_nightly.json")
    settings["profile_path"] = runner.ROOT / "tests/dfx/lingbot_world/configs/npu_nightly.json"
    reference = tmp_path / "reference"
    model = tmp_path / "model"
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"local image fixture")
    starts, calls = [], []
    monkeypatch.setenv("PYTHONPATH", "/fixture/cann-backend")
    monkeypatch.setattr(
        runner, "checked_assets", lambda _: (model, {case["image"]: image for case in settings["cases"]})
    )
    monkeypatch.setattr(runner, "validate_reference", lambda _: reference)
    monkeypatch.setattr(runner, "hardware_contract", lambda _: {"platform": "npu", "dies": 4})
    monkeypatch.setattr(
        runner, "git_revision", lambda path: settings["reference_revision"] if path == reference else "c" * 40
    )
    monkeypatch.setattr(runner, "_capture", lambda source, model, path, settings: path.write_bytes(b"NPZ fixture"))
    monkeypatch.setattr(runner, "assert_device_cleanup", lambda dies, path: write_json(path, {"passed": True}))

    @contextmanager
    def server(model_path, arguments, *, source_root, env_dict, startup_timeout):
        starts.append(source_root)
        assert env_dict["PYTHONPATH"] == f"{source_root}:/fixture/cann-backend"
        assert env_dict["OMP_NUM_THREADS"] == env_dict["MKL_NUM_THREADS"] == env_dict["OPENBLAS_NUM_THREADS"] == "1"
        assert env_dict["VLLM_OMNI_AR_DIFFUSION_KV_GATHER"] == "0"
        assert startup_timeout == 1200
        yield SimpleNamespace(model=model_path, port=8000)

    monkeypatch.setattr(runtime, "OmniServer", server)

    def serving(args):
        calls.append(args)
        assert args.realtime_video_camera_pattern == settings["cases"][(len(calls) - 1) % 3]["camera_pattern"]
        assert args.backend == args.dataset_name == "realtime-video"
        assert args.num_prompts == 7 and args.num_warmups == 1
        assert args.realtime_video_warmup_num_chunks == 8 and args.realtime_video_warmup_chunks == 6
        assert args.realtime_video_target_fps == 12 and args.skip_tokenizer_init
        assert args.ready_check_timeout_sec == 0 and args.max_concurrency == 1
        samples = get_realtime_video_samples(args)
        assert len(samples) == 7 and all(sample.workload.num_chunks == 40 for sample in samples)
        raw = {**document(), "git_revision": "c" * 40, "passed": True}
        template = raw["sessions"][0]
        raw["config"].update(sessions=7, warmup_sessions=1, warmup_num_chunks=8)
        raw["sessions"] = [
            {**copy.deepcopy(template), "session_index": index, "request_id": f"call{len(calls)}-session{index}"}
            for index in range(7)
        ]
        raw["warmups"] = [
            {
                "measured": False,
                "request_id": f"call{len(calls)}-warmup",
                "chunks": copy.deepcopy(template["chunks"][:8]),
            }
        ]
        assert measured_metrics(raw, settings)["steady_samples"] == 231
        # The endpoint is stubbed here; real shared-lifecycle WS coverage lives
        # in test_realtime_video_serve.py and test_lingbot_world_realtime.py.
        result = {"realtime_video": raw}
        write_json(Path(args.result_dir) / args.result_filename, result)
        Path(args.realtime_video_save_video).write_bytes(b"video fixture")
        return result

    monkeypatch.setattr(serve, "main", serving)
    for repeat in range(3):
        output = tmp_path / f"pair-{repeat}"
        result = runner.collect_measurements({**settings, "output": output}, measurement_repeat=repeat)
        assert result["complete"] and not result["errors"]
        assert runner.generation_paths(settings, [repeat]) <= result["artifacts"].keys()
        for role in ("reference", "candidate"):
            for case in settings["cases"]:
                case_dir = output / role / str(repeat) / case["id"]
                canonical = read_json(case_dir / "serve-result.json")["realtime_video"]
                projection = read_json(case_dir / "benchmark.json")
                assert projection["git_revision"] == canonical["git_revision"] == "c" * 40
                expected = settings["reference_revision"] if role == "reference" else "c" * 40
                assert projection["server_revision"] == expected
    assert starts == [
        str(reference),
        str(runner.ROOT),
        str(runner.ROOT),
        str(reference),
        str(reference),
        str(runner.ROOT),
    ]
    assert len(calls) == 18
