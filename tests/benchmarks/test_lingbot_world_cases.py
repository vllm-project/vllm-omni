# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Candidate capture lifecycle and independent shared benchmark recording."""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from tests.e2e.accuracy.lingbot_world import checks, runner
from vllm_omni.benchmarks.realtime_video import eval as evaluation
from vllm_omni.benchmarks.realtime_video.eval import read_json, sha256, write_json
from vllm_omni.benchmarks.realtime_video.metrics import RegressionError
from vllm_omni.benchmarks.realtime_video.precision import save_tensor_sample

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def measurements(tmp_path, monkeypatch):
    profile = runner.ROOT / "tests/e2e/accuracy/lingbot_world/configs/cuda.json"
    root = tmp_path / "raw"
    settings = runner.load_settings(profile, tmp_path / "judgments", generation=root)
    monkeypatch.setattr(runner, "git_revision", lambda _: "c" * 40)
    arm = root / "candidate"
    save_tensor_sample(arm / "precision.npz", inputs={"latent": np.ones(2)}, outputs={"decoded": np.ones(2)})
    write_json(arm / "cleanup.json", {"passed": True})
    for case in settings["cases"]:
        path = arm / case["id"]
        path.mkdir()
        (path / "rollout.mp4").write_bytes(b"video-fixture")
        write_json(path / "benchmark.json", {"passed": True, "errors": [], "git_revision": "c" * 40})
        write_json(path / "serve-result.json", {"realtime_video": {"passed": True}})
    manifest = {
        "schema_version": 2,
        "kind": "realtime-video-accuracy-candidate",
        "complete": True,
        "candidate_revision": "c" * 40,
        "errors": [],
        "contract": runner.generation_contract(settings),
        "hardware": {"platform": "cuda", "devices": ["H100"] * 2, "packages": {"vllm": "0.33.0"}},
        "artifacts": {str(path.relative_to(root)): sha256(path) for path in root.rglob("*") if path.is_file()},
    }
    write_json(root / "generation.json", manifest)
    return settings, (root, manifest)


def stub_semantic_boundary(monkeypatch, tmp_path):
    from tests.e2e.accuracy import helpers

    monkeypatch.setenv("LINGBOT_EVALUATOR_ROOT", str(tmp_path / "evaluator"))
    monkeypatch.setenv("LINGBOT_EVALUATOR_PYTHON", "/unused/native-worker")
    monkeypatch.setattr(helpers, "_accuracy_hardware_label", lambda: "H100")

    def scorer(root, manifest, *, suite, **kwargs):
        if suite == "wbench":
            raise RegressionError("native WBench failed")
        return {"scores": {case["id"]: {"dynamic_degree": 1.0} for case in manifest["contract"]["profile"]["cases"]}}

    monkeypatch.setattr(evaluation, "_record_semantic", scorer)


def test_three_scorers_use_shared_frontend_and_keep_independent_verdicts(measurements, monkeypatch, tmp_path, capsys):
    settings, data = measurements
    stub_semantic_boundary(monkeypatch, tmp_path)
    reports = {suite: checks.run_accuracy(settings, data, suite) for suite in ("wbench", "vbench", "precision")}
    assert not reports["wbench"]["passed"]
    assert reports["vbench"]["passed"] and reports["precision"]["passed"]
    assert all(row["comparison_status"] == "pending_baseline" for row in reports.values())
    assert reports["precision"]["measurements"]["replay"]["tensors"]["output/decoded"]["mean"] == 1.0
    assert "baseline comparison pending" in capsys.readouterr().out
    files = list((settings["output"] / "results").glob("result_acc_*.json"))
    assert len(files) == 3
    records = [read_json(path) for path in files]
    assert all(
        row["accuracy_mode"] == "record_only" and row["comparison_status"] == "pending_baseline" for row in records
    )
    assert all(row["baseline"] == {} for row in records)
    assert any(row["recording_status"] == "failed" for row in records)
    numerical = next(row for row in records if row["dataset"] == "lingbot_world_precision")
    semantic = next(row for row in records if row["dataset"] == "lingbot_world_vbench")
    assert numerical["output_decoded_mean"] == 1.0 and numerical["output_decoded_std"] == 0.0
    assert semantic["boardwalk-42_dynamic_degree"] == 1.0
    assert numerical["recording_report"]["measurements"]["replay"]["tensors"]["output/decoded"]["shape"] == [2]
    assert all("passed" not in row and "status" not in row and "measurements" not in row for row in records)


def test_partial_numerical_capture_does_not_hide_valid_semantic_inputs(measurements, monkeypatch, tmp_path):
    settings, (root, manifest) = measurements
    manifest["complete"] = False
    manifest["errors"] = [{"component": "precision", "error": "capture failed"}]
    del manifest["artifacts"]["candidate/precision.npz"]
    write_json(root / "generation.json", manifest)
    stub_semantic_boundary(monkeypatch, tmp_path)
    data = runner.measurement_data(settings)
    assert not checks.run_accuracy(settings, data, "precision")["passed"]
    assert checks.run_accuracy(settings, data, "vbench")["passed"]


@pytest.mark.parametrize("change", ["source", "profile", "deploy", "recipe", "hardware", "assets"])
def test_measurement_contract_rejects_wrong_or_old_run(measurements, change):
    settings, (root, manifest) = measurements
    if change == "source":
        manifest["candidate_revision"] = "d" * 40
    elif change == "profile":
        manifest["contract"]["profile"]["num_chunks"] = 160
    elif change == "deploy":
        manifest["contract"]["deploy_sha256"] = "0" * 64
    elif change == "recipe":
        manifest["contract"]["capture_recipe"] = {"independent_starts_per_source": 3}
    elif change == "hardware":
        manifest["contract"]["hardware"]["platform"] = "npu"
    else:
        manifest["contract"]["model_assets_sha256"] = "0" * 64
    write_json(root / "generation.json", manifest)
    with pytest.raises(RegressionError, match="contract mismatch"):
        runner.measurement_data(settings)


def test_npu_results_preserve_generation_identity_separate_from_h100_evaluator(measurements, monkeypatch, tmp_path):
    settings, (root, manifest) = measurements
    settings.update(platform="npu", hardware="A2")
    manifest["hardware"] = {"platform": "npu", "devices": ["Ascend 910B"] * 4}
    write_json(root / "generation.json", manifest)
    stub_semantic_boundary(monkeypatch, tmp_path)
    checks.run_accuracy(settings, (root, manifest), "vbench")
    result = read_json(next((settings["output"] / "results").glob("result_acc_*.json")))
    assert result["Hardware"] == "A2" and result["generation_platform"] == "npu"
    assert result["generation_hardware"]["devices"] == ["Ascend 910B"] * 4
    assert result["evaluator_hardware"] == "H100"
    assert result["candidate_revision"] == "c" * 40 and "reference_revision" not in result


@dataclass
class CapturedServer:
    model: str
    port: int = 8000


@pytest.mark.parametrize("platform", ["cuda", "npu"])
def test_accuracy_has_one_candidate_server_and_three_shared_serving_calls(tmp_path, monkeypatch, platform):
    from tests.helpers import runtime
    from vllm_omni.benchmarks import serve
    from vllm_omni.benchmarks.realtime_video.serving import get_realtime_video_samples

    settings = runner.load_settings(
        runner.ROOT / f"tests/e2e/accuracy/lingbot_world/configs/{platform}.json", tmp_path / "capture"
    )
    model, image = tmp_path / "model", tmp_path / "frame.jpg"
    image.write_bytes(b"local fixture")
    monkeypatch.setenv("PYTHONPATH", "/fixture/backend")
    monkeypatch.setattr(
        runner, "checked_assets", lambda _: (model, {case["image"]: image for case in settings["cases"]})
    )
    monkeypatch.setattr(runner, "hardware_contract", lambda _: {"platform": platform, "dies": settings["dies"]})
    monkeypatch.setattr(runner, "git_revision", lambda _: "c" * 40)
    captures, starts, calls = [], [], []

    def capture(actual_model, path, config):
        captures.append(path)
        save_tensor_sample(path, inputs={"latent": np.ones(2)}, outputs={"decoded": np.ones(2)})

    monkeypatch.setattr(runner, "_capture", capture)
    monkeypatch.setattr(runner, "assert_device_cleanup", lambda dies, path: write_json(path, {"passed": True}))

    @contextmanager
    def server(model_path, arguments, *, env_dict, startup_timeout):
        starts.append(model_path)
        assert env_dict["PYTHONPATH"] == f"{runner.ROOT}:/fixture/backend"
        assert startup_timeout == 1200
        if platform == "npu":
            assert env_dict["OMP_NUM_THREADS"] == env_dict["MKL_NUM_THREADS"] == "1"
        yield CapturedServer(model_path)

    monkeypatch.setattr(runtime, "OmniServer", server)

    def serving(args):
        calls.append(args)
        assert args.backend == args.dataset_name == "realtime-video"
        assert args.num_prompts == args.num_warmups == 1
        assert args.realtime_video_warmup_num_chunks == 8 and args.realtime_video_warmup_chunks == 6
        assert args.skip_tokenizer_init and args.ready_check_timeout_sec == 0 and args.max_concurrency == 1
        samples = get_realtime_video_samples(args)
        assert len(samples) == 1 and samples[0].workload.num_chunks == 40
        result = {"realtime_video": {"git_revision": "c" * 40, "passed": True, "errors": []}}
        write_json(Path(args.result_dir) / args.result_filename, result)
        Path(args.realtime_video_save_video).write_bytes(b"video fixture")
        return result

    monkeypatch.setattr(serve, "main", serving)
    report = runner.collect_measurements(settings)
    assert report["complete"] and not report["errors"]
    assert len(starts) == len(captures) == 1 and len(calls) == 3
    assert runner.generation_paths(settings) <= report["artifacts"].keys()
    assert not (settings["output"] / "reference").exists()
    for case in settings["cases"]:
        document = read_json(settings["output"] / "candidate" / case["id"] / "benchmark.json")
        assert document["git_revision"] == document["server_revision"] == "c" * 40
