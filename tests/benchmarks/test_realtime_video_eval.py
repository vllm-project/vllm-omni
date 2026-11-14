# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU fixtures exercise offline scoring, failure integrity and the real CLI."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import pytest

from vllm_omni.benchmarks.realtime_video import eval as evaluation
from vllm_omni.benchmarks.realtime_video.encoder_inference import BACKGROUND_CLIP_BATCH_SIZE
from vllm_omni.benchmarks.realtime_video.precision import save_tensor_sample
from vllm_omni.benchmarks.realtime_video.video_quality import STANDARD_VBENCH_DIMENSIONS, WBENCH_DIMENSIONS
from vllm_omni.benchmarks.realtime_video.workload import num_frames_for_chunks

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def recording(tmp_path):
    root = tmp_path / "capture"
    root.mkdir()
    profile = {
        "num_chunks": 2,
        "width": 32,
        "height": 24,
        "mux_fps": 4,
        "cases": [{"id": "cpu-fixture", "seed": 42, "camera_pattern": "forward"}],
    }
    manifest = {
        "schema_version": 2,
        "kind": "realtime-video-accuracy-candidate",
        "candidate_revision": "a" * 40,
        "contract": {"profile": profile, "capture_recipe": {"candidate_starts": 1, "sessions_per_case": 1}},
        "hardware": {"fixture": "CPU; no native model was executed"},
        "errors": [],
        "artifacts": {},
    }
    save_tensor_sample(
        root / "candidate/precision.npz",
        inputs={"latent": np.array([1, 2], dtype=np.float32)},
        outputs={"decoded": np.array([3, 5], dtype=np.float32)},
        metadata={"context": "CPU fixture"},
    )
    evaluation.write_json(root / "candidate/cleanup.json", {"passed": True})
    evaluation.write_json(
        root / "candidate/cpu-fixture/benchmark.json",
        {
            "passed": True,
            "git_revision": "a" * 40,
            "server_revision": "a" * 40,
            "config": {"sessions": 1, "camera_pattern": "forward"},
            "workload": {"num_chunks": 2, "width": 32, "height": 24, "fps": 4, "seed": 42},
        },
    )
    (root / "candidate/cpu-fixture/rollout.mp4").write_bytes(b"scripted scorer boundary only")

    def persist():
        manifest["artifacts"] = {
            str(path.relative_to(root)): evaluation.sha256(path)
            for path in (root / "candidate").rglob("*")
            if path.is_file()
        }
        evaluation.write_json(root / "generation.json", manifest)
        return root / "generation.json"

    return root, manifest, persist


def test_actual_shared_cli_records_and_saves_numeric_values_without_a_server(recording, tmp_path, monkeypatch, capsys):
    from vllm_omni.benchmarks import serve
    from vllm_omni.benchmarks.realtime_video.serving import build_serve_args
    from vllm_omni.entrypoints.cli.benchmark.serve import OmniBenchmarkServingSubcommand

    _, _, persist = recording

    async def unexpected_server_discovery(args):
        pytest.fail("offline evaluation CLI must not run model discovery or benchmark traffic")

    monkeypatch.setattr(serve, "vllm_main_async", unexpected_server_discovery)
    arguments = [
        "--realtime-video-evaluate",
        "precision",
        "--realtime-video-candidate",
        str(persist()),
        "--realtime-video-eval-output-dir",
        str(tmp_path / "raw"),
        "--save-result",
        "--result-dir",
        str(tmp_path / "results"),
        "--result-filename",
        "recorded.json",
    ]
    OmniBenchmarkServingSubcommand.cmd(build_serve_args(arguments))
    saved = evaluation.read_json(tmp_path / "results/recorded.json")["realtime_video_evaluation"]
    assert saved["status"] == "measured" and saved["passed"]
    assert "record_only" not in saved and "comparison_status" not in saved
    values = saved["measurements"]["replay"]["tensors"]["output/decoded"]
    assert values["mean"] == 4 and values["std"] == 1 and values["shape"] == [2]
    assert values["minimum"] == 3 and values["maximum"] == 5
    assert "reference" not in saved["measurements"] and "psnr_db" not in saved["measurements"]
    output = capsys.readouterr().out
    assert "accuracy measurements (precision)" in output and '"mean": 4.0' in output


@pytest.mark.parametrize("component", ["server", "cleanup", "precision"])
def test_precision_rejects_capture_errors_even_when_a_valid_npz_exists(recording, tmp_path, component):
    _, manifest, persist = recording
    manifest["errors"].append({"component": component, "error": "CPU injected capture failure"})
    report = evaluation.evaluate_saved(persist(), suite="precision", output_dir=tmp_path / "output")
    assert report["status"] == "failed" and not report["passed"] and report["errors"]


@pytest.mark.parametrize("corruption", ["changed_bytes", "nonfinite", "wrong_source", "old_recipe", "cleanup_false"])
def test_invalid_numeric_inputs_fail_closed(recording, tmp_path, corruption):
    root, manifest, persist = recording
    path = persist()
    if corruption == "changed_bytes":
        (root / "candidate/precision.npz").write_bytes(b"not the recorded payload")
    else:
        if corruption == "nonfinite":
            save_tensor_sample(root / "candidate/precision.npz", inputs={"x": [1]}, outputs={"y": [np.nan]})
        elif corruption == "wrong_source":
            manifest["candidate_revision"] = "HEAD"
        elif corruption == "old_recipe":
            manifest["contract"]["capture_recipe"] = {"candidate_starts": 3, "sessions_per_case": 1}
        else:
            evaluation.write_json(root / "candidate/cleanup.json", {"passed": False})
        persist()
    report = evaluation.evaluate_saved(path, suite="precision", output_dir=tmp_path / "output")
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("name", ["../escape", "/absolute"])
def test_artifact_paths_cannot_escape_capture(recording, tmp_path, name):
    root, manifest, _ = recording
    manifest["artifacts"][name] = "a" * 64
    with pytest.raises(evaluation.RegressionError, match="unsafe"):
        evaluation.checked_artifact(root, manifest, name)


@pytest.fixture
def scripted_native(monkeypatch, tmp_path):
    from vllm_omni.benchmarks.realtime_video import video_quality

    lock = tmp_path / "lock.json"
    evaluation.write_json(lock, {"repositories": {"vbench": {"revision": "b" * 40}}, "files": {}})
    monkeypatch.setattr(evaluation, "EVALUATOR_LOCK", lock)
    monkeypatch.setattr(evaluation, "verify_environment", lambda *args: {"fixture": "CPU boundary"})
    checked_sources = []
    monkeypatch.setattr(evaluation, "verify_evaluator_source", lambda *args: checked_sources.append(args))
    monkeypatch.setattr(
        video_quality, "verify_video_coverage", lambda path, **kwargs: {"decoded_frames": kwargs["frames"]}
    )
    frames = num_frames_for_chunks(2)
    stats = {
        "gradients": False,
        "batch_size": BACKGROUND_CLIP_BATCH_SIZE,
        "encoded_images": frames,
        "forward_batches": 1,
        "max_forward_batch": frames,
    }
    reports = {}
    for suite, dimensions in (("wbench", WBENCH_DIMENSIONS), ("vbench", STANDARD_VBENCH_DIMENSIONS)):
        reports[suite] = {
            "status": "passed",
            "errors": [],
            "scores": {
                "cpu-fixture": {dimension: 1.0 if dimension == "dynamic_degree" else 0.75 for dimension in dimensions}
            },
            "protocol": {"dynamic_degree_backend": "raft"} if suite == "wbench" else {},
            "native_provenance": {"cpu-fixture": {"background_clip_inference": stats}},
            "repo_revision": "b" * 40,
        }
    calls = []

    def native(videos, output_dir, *, suite, **kwargs):
        calls.append((suite, set(videos)))
        return copy.deepcopy(reports[suite])

    monkeypatch.setattr(video_quality, "evaluate_video_quality", native)
    return reports, calls, checked_sources


def _semantic(path, tmp_path, suite):
    return evaluation.evaluate_saved(
        path,
        suite=suite,
        output_dir=tmp_path / suite,
        evaluator_root=tmp_path / "scorer",
        evaluator_python=Path("cpu-fixture-python"),
        evaluator_requirements=Path("cpu-fixture-requirements"),
    )


@pytest.mark.parametrize("suite,dimensions", [("wbench", WBENCH_DIMENSIONS), ("vbench", STANDARD_VBENCH_DIMENSIONS)])
def test_semantic_records_every_real_native_dimension_independently_of_missing_precision(
    recording,
    tmp_path,
    scripted_native,
    suite,
    dimensions,
):
    root, manifest, persist = recording
    (root / "candidate/precision.npz").unlink()
    manifest["errors"].append({"component": "precision", "error": "CPU fixture has only video"})
    report = _semantic(persist(), tmp_path, suite)
    assert report["passed"] and report["status"] == "measured"
    assert set(report["measurements"]["scores"]["cpu-fixture"]) == set(dimensions)
    assert report["measurements"]["scores"]["cpu-fixture"]["dynamic_degree"] == 1.0
    assert scripted_native[1] == [(suite, {"cpu-fixture"})]
    assert scripted_native[2]


@pytest.mark.parametrize("corruption", ["failed_stream", "failed_benchmark", "wrong_source", "wrong_geometry"])
def test_semantic_rejects_failed_or_mismatched_capture_with_complete_video(
    recording,
    tmp_path,
    scripted_native,
    corruption,
):
    root, manifest, persist = recording
    benchmark_path = root / "candidate/cpu-fixture/benchmark.json"
    benchmark = evaluation.read_json(benchmark_path)
    if corruption == "failed_stream":
        manifest["errors"].append({"component": "stream", "case": "cpu-fixture", "error": "failed finalizer"})
    elif corruption == "failed_benchmark":
        benchmark["passed"] = False
    elif corruption == "wrong_source":
        benchmark["server_revision"] = "c" * 40
    else:
        benchmark["workload"]["width"] = 99
    evaluation.write_json(benchmark_path, benchmark)
    report = _semantic(persist(), tmp_path, "wbench")
    assert not report["passed"] and report["errors"]
    assert scripted_native[1] == []


@pytest.mark.parametrize("suite", ["wbench", "vbench"])
@pytest.mark.parametrize("invalid", ["missing_dimension", "missing_case", "nan", "boolean", "binary_fraction"])
def test_invalid_native_scores_fail_recording(recording, tmp_path, scripted_native, suite, invalid):
    reports, _, _ = scripted_native
    scores = reports[suite]["scores"]
    if invalid == "missing_dimension":
        scores["cpu-fixture"].pop("aesthetic_quality")
    elif invalid == "missing_case":
        scores.clear()
    elif invalid == "nan":
        scores["cpu-fixture"]["aesthetic_quality"] = float("nan")
    elif invalid == "boolean":
        scores["cpu-fixture"]["aesthetic_quality"] = True
    else:
        scores["cpu-fixture"]["dynamic_degree"] = 0.4
    report = _semantic(recording[2](), tmp_path, suite)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("suite", ["wbench", "vbench"])
def test_required_native_protocol_is_checked(recording, tmp_path, scripted_native, suite):
    reports, _, _ = scripted_native
    if suite == "wbench":
        reports[suite]["protocol"]["dynamic_degree_backend"] = "farneback"
    else:
        reports[suite]["native_provenance"]["cpu-fixture"]["background_clip_inference"]["encoded_images"] -= 1
    report = _semantic(recording[2](), tmp_path, suite)
    assert not report["passed"] and report["errors"]


def test_relative_evaluator_root_stays_valid_when_the_worker_changes_directory(
    recording,
    tmp_path,
    scripted_native,
    monkeypatch,
):
    from vllm_omni.benchmarks.realtime_video import video_quality

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VBENCH_CACHE_DIR", "CPU-fixture-original")
    native = video_quality.evaluate_video_quality

    def check_environment(*args, **kwargs):
        assert os.environ["VBENCH_CACHE_DIR"] == str(tmp_path / "scorer/vbench-cache")
        assert os.environ["WBENCH_WEIGHTS_DIR"] == str(tmp_path / "scorer/wbench/weights")
        assert kwargs["repo"] == tmp_path / "scorer/wbench"
        return native(*args, **kwargs)

    monkeypatch.setattr(video_quality, "evaluate_video_quality", check_environment)
    report = evaluation.evaluate_saved(
        recording[2](),
        suite="wbench",
        output_dir=tmp_path / "output",
        evaluator_root=Path("scorer"),
        evaluator_python=Path("cpu-fixture-python"),
        evaluator_requirements=Path("cpu-fixture-requirements"),
    )
    assert report["passed"]
    assert os.environ["VBENCH_CACHE_DIR"] == "CPU-fixture-original"


@pytest.mark.parametrize(
    "arguments", [["--realtime-video-evaluate", "precision"], ["--realtime-video-candidate", "generation.json"]]
)
def test_accuracy_cli_requires_suite_and_candidate(arguments):
    from vllm_omni.benchmarks.realtime_video.serving import build_serve_args

    with pytest.raises(ValueError, match="requires both"):
        build_serve_args(arguments)


def test_evaluator_checkout_guard_ignores_bytecode_caches_only(tmp_path):
    import subprocess

    repository = tmp_path / "dino"
    repository.mkdir()
    git = ["git", "-c", "user.name=CPU fixture", "-c", "user.email=cpu@fixture", "-C", str(repository)]
    subprocess.run([*git, "init", "-q"], check=True)
    (repository / "hubconf.py").write_text("dependencies = []\n")
    subprocess.run([*git, "add", "hubconf.py"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "fixture"], check=True)
    revision = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    # torch.hub imports a local checkout and leaves bytecode next to its sources.
    for cache in (repository / "__pycache__", repository / "utils/__pycache__"):
        cache.mkdir(parents=True)
        (cache / "hubconf.cpython-312.pyc").write_bytes(b"bytecode")
    evaluation.verify_evaluator_source(repository, revision)

    (repository / "patched.py").write_text("scores = 1\n")
    with pytest.raises(evaluation.RegressionError, match="tracked or untracked"):
        evaluation.verify_evaluator_source(repository, revision)
    (repository / "patched.py").unlink()
    (repository / "hubconf.py").write_text("dependencies = ['changed']\n")
    with pytest.raises(evaluation.RegressionError, match="tracked or untracked"):
        evaluation.verify_evaluator_source(repository, revision)
