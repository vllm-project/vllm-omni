# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Consumer isolation and malformed actual-measurement contracts, without model substitutes."""

import copy
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.lingbot_world.precision import save_tensor_sample
from benchmarks.lingbot_world.regression import RegressionError, read_json, sha256, write_json
from tests.benchmarks.test_lingbot_world_regression import document
from tests.e2e.accuracy.lingbot_world import checks, quality, runner

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
            "capture_recipe": dict(runner.CAPTURE_RECIPE),
            "model_assets_sha256": sha256(runner.ROOT / "tests/e2e/accuracy/lingbot_world/assets.lock.json"),
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
    monkeypatch.setattr(quality.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=""))
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


def test_wbench_failure_does_not_block_other_accuracy_tests(measurements, monkeypatch, tmp_path):
    settings, data = measurements
    fake_native_workers(monkeypatch, tmp_path)
    fake_pixels(monkeypatch)
    assert not quality.evaluate_quality(settings, data, "wbench", monkeypatch)["passed"]
    assert quality.evaluate_quality(settings, data, "vbench", monkeypatch)["passed"]
    assert checks.numeric_report(settings, data)["passed"]
    assert all((settings["output"] / f"{name}.json").is_file() for name in ("wbench", "vbench", "precision"))


def test_partial_producer_does_not_hide_complete_semantic_inputs(measurements, monkeypatch, tmp_path):
    settings, (root, manifest) = measurements
    manifest["complete"] = False
    manifest["errors"] = ["precision capture failed"]
    del manifest["artifacts"]["candidate/0/precision.npz"]
    write_json(root / "generation.json", manifest)
    data = runner.measurement_data(settings)
    fake_native_workers(monkeypatch, tmp_path)
    assert not checks.numeric_report(settings, data)["passed"]
    assert quality.evaluate_quality(settings, data, "vbench", monkeypatch)["passed"]
    assert not quality.evaluate_quality(settings, data, "wbench", monkeypatch)["passed"]


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "changed_bytes",
        "wrong_source",
        "wrong_profile",
        "wrong_deploy",
        "old_recipe",
        "wrong_hardware",
        "wrong_assets",
    ],
)
def test_accuracy_rejects_unauthenticated_or_wrong_inputs(measurements, change):
    settings, (root, manifest) = measurements
    name = "candidate/0/precision.npz"
    if change == "missing":
        del manifest["artifacts"][name]
    elif change == "changed_bytes":
        (root / name).write_bytes(b"changed input")
    elif change == "wrong_source":
        manifest["candidate_revision"] = "d" * 40
    elif change == "wrong_profile":
        manifest["contract"]["profile"]["num_chunks"] = 160
    elif change == "wrong_deploy":
        manifest["contract"]["deploy_sha256"] = "0" * 64
    elif change == "old_recipe":
        manifest["contract"]["capture_recipe"]["sessions_per_case"] = 7
    elif change == "wrong_hardware":
        manifest["contract"]["hardware"]["platform"] = "npu"
    else:
        manifest["contract"]["model_assets_sha256"] = "0" * 64
    write_json(root / "generation.json", manifest)
    if change in ("missing", "changed_bytes"):
        assert not checks.numeric_report(settings, runner.measurement_data(settings))["passed"]
    else:
        with pytest.raises(RegressionError, match="contract mismatch"):
            runner.measurement_data(settings)


def test_npu_results_keep_generation_identity_separate_from_h100_scorer(measurements, monkeypatch):
    from tests.e2e.accuracy import helpers

    settings, (root, manifest) = measurements
    settings.update(platform="npu", hardware="A2", model_revision="59cccf49f2d2dd27418ae7a04b82b10868d455c2")
    manifest["contract"]["hardware"].update(platform="npu", devices=["Ascend 910B"] * 4)
    manifest["contract"]["profile"]["platform"] = "npu"
    monkeypatch.setattr(helpers, "_accuracy_hardware_label", lambda: "H100")
    path = checks.record_accuracy_result(settings, (root, manifest), "vbench", {"passed": False, "cases": {}})
    result = read_json(path)
    assert result["Hardware"] == "A2"
    assert result["generation_platform"] == "npu"
    assert result["generation_hardware"]["devices"] == ["Ascend 910B"] * 4
    assert result["evaluator_hardware"] == "H100"
    assert result["candidate_revision"] == "c" * 40 and result["reference_revision"] == "a" * 40
    assert result["workload"]["platform"] == "npu" and not result["passed"]
    assert set(result["baseline"]) == {"A2"}


@pytest.mark.parametrize("change", ["tracked_code", "untracked_code", "agent_metadata"])
def test_evaluator_rejects_tracked_and_untracked_code_except_agent_metadata(tmp_path, change):
    repository = tmp_path / "evaluator"
    repository.mkdir()
    (repository / "scorer.py").write_text("score = 1\n")
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "scorer.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=CI fixture",
            "-c",
            "user.email=ci@example.invalid",
            "commit",
            "-qm",
            "Pinned scorer",
        ],
        check=True,
    )
    revision = runner.git_revision(repository)
    quality.verify_evaluator_source(repository, revision)
    if change == "tracked_code":
        (repository / "scorer.py").write_text("score = 0\n")
    elif change == "untracked_code":
        (repository / "new_scorer.py").write_text("score = 0\n")
    else:
        (repository / ".claude").mkdir()
        (repository / ".claude" / "instructions.md").write_text("Agent metadata, not evaluator code.\n")
    if change == "agent_metadata":
        quality.verify_evaluator_source(repository, revision)
    else:
        with pytest.raises(RegressionError, match="tracked or untracked"):
            quality.verify_evaluator_source(repository, revision)


@pytest.mark.parametrize("provided", [False, True])
@pytest.mark.parametrize("phase", ["success", "checkout", "capture"])
def test_accuracy_cleans_only_owned_temporary_reference_on_success_and_failure(tmp_path, monkeypatch, provided, phase):
    from tests.e2e.accuracy.lingbot_world import prepare

    reference = tmp_path / "caller-reference" if provided else None
    if reference is not None:
        reference.mkdir()
        (reference / "keep.txt").write_text("caller-owned")
    settings = runner.load_settings(
        runner.ROOT / "tests/e2e/accuracy/lingbot_world/configs/cuda.json",
        tmp_path / "accuracy",
        reference_root=reference,
    )
    seen = []
    monkeypatch.setattr(prepare, "prepare_cuda_allocation", lambda *args: "0,1")
    monkeypatch.setattr(prepare, "verify_inference_runtime", lambda platform: {"packages": {"vllm": "0.33.0"}})
    monkeypatch.setattr(prepare, "verify_source_imports", lambda source: None)
    monkeypatch.setenv("LINGBOT_MODEL_PATH", str(tmp_path / "model"))

    def checkout(path, *args, **kwargs):
        seen.append(path)
        assert path.is_dir()
        (path / "source.txt").write_text("reference fixture")
        if phase == "checkout":
            raise RuntimeError("checkout failed")

    def collect(actual):
        assert actual["reference_root"] == seen[0] and seen[0].is_dir()
        if phase == "capture":
            raise RuntimeError("capture failed")
        return {"complete": False, "errors": ["independent precision capture failed"]}

    monkeypatch.setattr(prepare, "checkout", checkout)
    monkeypatch.setattr(runner, "collect_measurements", collect)
    if phase == "success":
        assert prepare.prepare_capture(settings)["complete"] is False
    else:
        with pytest.raises(RuntimeError, match=f"{phase} failed"):
            prepare.prepare_capture(settings)
    assert seen[0].exists() is provided
    if provided:
        assert (reference / "keep.txt").read_text() == "caller-owned"


def test_npu_accuracy_uses_three_fresh_starts_and_shared_serving_bridge(tmp_path, monkeypatch):
    """Only server/model boundaries are stubbed; parse the real serving options."""
    from contextlib import contextmanager
    from types import SimpleNamespace

    from tests.helpers import runtime
    from vllm_omni.benchmarks import serve
    from vllm_omni.benchmarks.realtime_video.serving import get_realtime_video_samples

    settings = read_json(runner.ROOT / "tests/e2e/accuracy/lingbot_world/configs/npu.json")
    settings["profile_path"] = runner.ROOT / "tests/e2e/accuracy/lingbot_world/configs/npu.json"
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
        assert args.num_prompts == 1 and args.num_warmups == 1
        assert args.realtime_video_warmup_num_chunks == 8 and args.realtime_video_warmup_chunks == 6
        assert args.realtime_video_target_fps == 12 and args.skip_tokenizer_init
        assert args.ready_check_timeout_sec == 0 and args.max_concurrency == 1
        samples = get_realtime_video_samples(args)
        assert len(samples) == 1 and all(sample.workload.num_chunks == 40 for sample in samples)
        raw = {**document(), "git_revision": "c" * 40, "passed": True}
        template = raw["sessions"][0]
        raw["config"].update(sessions=1, warmup_sessions=1, warmup_num_chunks=8)
        raw["sessions"] = [
            {**copy.deepcopy(template), "session_index": index, "request_id": f"call{len(calls)}-session{index}"}
            for index in range(1)
        ]
        raw["warmups"] = [
            {
                "measured": False,
                "request_id": f"call{len(calls)}-warmup",
                "chunks": copy.deepcopy(template["chunks"][:8]),
            }
        ]
        # The endpoint is stubbed here; real shared-lifecycle WS coverage lives
        # in test_realtime_video_serve.py and test_lingbot_world_realtime.py.
        result = {"realtime_video": raw}
        write_json(Path(args.result_dir) / args.result_filename, result)
        Path(args.realtime_video_save_video).write_bytes(b"video fixture")
        return result

    monkeypatch.setattr(serve, "main", serving)
    output = tmp_path / "accuracy"
    result = runner.collect_measurements({**settings, "output": output})
    assert result["complete"] and not result["errors"]
    assert runner.generation_paths(settings) <= result["artifacts"].keys()
    for repeat in range(3):
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
