# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Coverage and failure semantics for long-rollout quality diagnostics."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from benchmarks.lingbot_world import video_quality as quality
from benchmarks.lingbot_world.regression import RegressionError, check_video_precision

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _write_video(path: Path, levels: list[int], *, fps: int = 4, size: tuple[int, int] = (32, 24)) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    if not writer.isOpened():
        pytest.skip("OpenCV MJPEG encoder unavailable")
    try:
        for level in levels:
            writer.write(np.full((size[1], size[0], 3), level, dtype=np.uint8))
    finally:
        writer.release()


def test_video_windows_reveal_late_drift(tmp_path: Path) -> None:
    reference, candidate = tmp_path / "reference.avi", tmp_path / "candidate.avi"
    _write_video(reference, [40] * 12)
    _write_video(candidate, [40] * 4 + [60] * 4 + [100] * 4)
    result = quality.compare_video_pair(reference, candidate, window_seconds=1)
    assert result["reference"]["frame_count"] == result["decoded_frames"] == 12
    assert result["candidate"]["duration_s"] == 3
    assert [window["frame_count"] for window in result["windows"]] == [4, 4, 4]
    assert [window["mae"] for window in result["windows"]] == pytest.approx([0, 20, 60], abs=2)
    assert result["windows"][0]["psnr_is_infinite"] is True
    assert result["windows"][1]["psnr_db"] > result["windows"][2]["psnr_db"]
    assert result["total"]["different_frames"] == 8
    assert result["baseline_repeat_required"] is True
    json.dumps(result, allow_nan=False)


def _write_textured_video(path: Path, base: int, *, corrupt_tail: bool = False) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 16, (32, 24))
    if not writer.isOpened():
        pytest.skip("OpenCV MJPEG encoder unavailable")
    y, x = np.indices((24, 32))
    texture = ((x * 3 + y * 5) % 40).astype(np.uint8)
    image = np.stack([texture + base, texture + base + 5, texture + base + 10], axis=-1)
    try:
        for index in range(160):
            frame = np.roll(image, index % 32, axis=1)
            writer.write(np.zeros_like(frame) if corrupt_tail and index >= 128 else frame)
    finally:
        writer.release()


def _pixel_packages() -> dict[str, str]:
    from importlib.metadata import version

    return {name: version(name) for name in ("numpy", "opencv-python", "scikit-image")}


def test_pixel_comparison_uses_isolated_evaluator_when_parent_decoder_is_unavailable(tmp_path, monkeypatch):
    video = tmp_path / "reference.avi"
    _write_textured_video(video, 20)
    expected = _pixel_packages()
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "cv2.py").write_text('raise RuntimeError("uncontrolled parent decoder")\n')
    monkeypatch.setenv("PYTHONPATH", str(shadow))
    monkeypatch.setitem(sys.modules, "cv2", None)
    result = quality.compare_video_pair_in_environment(
        video,
        video,
        tmp_path / "pixels",
        python=Path(sys.executable),
        expected_packages=expected,
        window_seconds=2,
        ssim_stride=24,
        timeout_s=30,
    )
    assert result["decoded_frames"] == 160 and result["total"]["psnr_is_infinite"]
    assert result["pixel_environment"]["packages"] == expected
    assert result["pixel_environment"]["python"] == sys.executable
    assert len(result["pixel_environment"]["opencv_build_sha256"]) == 64
    assert result["uniform_frame_indices"] == {"reference": [], "candidate": []}


def test_pixel_dependency_drift_fails_and_retains_actual_environment(tmp_path):
    video = tmp_path / "reference.avi"
    _write_textured_video(video, 20)
    expected = {**_pixel_packages(), "numpy": "0.0.0"}
    output = tmp_path / "pixels"
    with pytest.raises(quality.VideoQualityError, match="Pixel worker"):
        quality.compare_video_pair_in_environment(
            video,
            video,
            output,
            python=Path(sys.executable),
            expected_packages=expected,
            window_seconds=2,
            ssim_stride=24,
            timeout_s=30,
        )
    result = json.loads(next(output.rglob("worker_result.json")).read_text())
    assert result["status"] != "complete"
    assert result["pixel_environment"]["packages"]["numpy"] == _pixel_packages()["numpy"]
    assert "Pixel dependencies changed" in next(output.rglob("worker.log")).read_text()


def test_pixel_worker_timeout_reaps_process_and_retains_incomplete_evidence(tmp_path, monkeypatch):
    processes = []
    original_popen = subprocess.Popen

    def capture_process(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(quality.subprocess, "Popen", capture_process)
    monkeypatch.setattr(quality, "_PIXEL_WORKER", "import time;time.sleep(60)")
    video = tmp_path / "reference.avi"
    _write_textured_video(video, 20)
    output = tmp_path / "pixels"
    with pytest.raises(subprocess.TimeoutExpired):
        quality.compare_video_pair_in_environment(
            video,
            video,
            output,
            python=Path(sys.executable),
            expected_packages=_pixel_packages(),
            window_seconds=2,
            ssim_stride=24,
            timeout_s=0.5,
        )
    assert len(processes) == 1 and processes[0].returncode is not None
    assert next(output.rglob("worker.log")).is_file()
    assert json.loads(next(output.rglob("worker_result.json")).read_text())["status"] != "complete"


def test_pixel_worker_preserves_virtualenv_interpreter_path(tmp_path, monkeypatch):
    executable = tmp_path / "venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    video = tmp_path / "reference.avi"
    _write_textured_video(video, 20)
    packages = _pixel_packages()

    def run(command, repo, log, timeout_s):
        assert command[0] == str(executable) and command[1] == "-I"
        inputs = json.loads(Path(command[-1]).read_text())
        Path(inputs["result"]).write_text(
            json.dumps(
                {
                    "status": "complete",
                    "pixel_environment": {"python": str(executable), "packages": packages},
                }
            )
        )
        return 0

    monkeypatch.setattr(quality, "_run_worker", run)
    result = quality.compare_video_pair_in_environment(
        video,
        video,
        tmp_path / "pixels",
        python=executable,
        expected_packages=packages,
        window_seconds=2,
        ssim_stride=24,
        timeout_s=30,
    )
    assert result["pixel_environment"]["python"] == str(executable)


@pytest.mark.parametrize("platform", ["cuda", "npu"])
def test_noisy_reference_budget_cannot_hide_a_spatially_empty_window(tmp_path: Path, platform: str) -> None:
    paths = [tmp_path / f"reference-{index}.avi" for index in range(3)]
    for path, base in zip(paths, (20, 160, 90)):
        _write_textured_video(path, base)
    references = [
        quality.compare_video_pair(paths[a], paths[b], window_seconds=2, ssim_stride=24)
        for a, b in ((0, 1), (0, 2), (1, 2))
    ]
    candidate = tmp_path / "late-black-window.avi"
    _write_textured_video(candidate, 20, corrupt_tail=True)
    measured = quality.compare_video_pair(paths[0], candidate, window_seconds=2, ssim_stride=24)
    assert measured["uniform_frame_indices"]["reference"] == []
    assert measured["uniform_frame_indices"]["candidate"] == list(range(128, 160))
    policy = json.loads((Path(__file__).parents[2] / "tests/dfx/lingbot_world/quality.json").read_text())
    options = {"min_psnr_db": policy["min_psnr_db"], **policy["pixel_budgets"][platform]}
    assert check_video_precision(references, measured, **options)["passed"]
    strict = {**options, "reject_uniform_frames": policy["reject_uniform_frames"]}
    check = check_video_precision(references, measured, **strict)
    assert not check["passed"] and any("uniform" in violation for violation in check["violations"])
    assert check_video_precision(references, references[0], **strict)["passed"]


def test_identically_blank_reference_and_candidate_cannot_pass_content_validation(tmp_path: Path) -> None:
    path = tmp_path / "blank.avi"
    _write_video(path, [0] * 16)
    pair = quality.compare_video_pair(path, path, window_seconds=2, ssim_stride=8)
    options = {"min_psnr_db": 40, "psnr_drop_db": 3, "ssim_drop": 0.04}
    assert check_video_precision([pair] * 3, pair, **options)["passed"]
    check = check_video_precision([pair] * 3, pair, reject_uniform_frames=True, **options)
    assert not check["passed"] and all("uniform" in violation for violation in check["violations"])


def test_selected_content_validation_requires_decoded_frame_evidence(tmp_path: Path) -> None:
    path = tmp_path / "valid.avi"
    _write_textured_video(path, 20)
    pair = quality.compare_video_pair(path, path, window_seconds=2, ssim_stride=24)
    missing = {key: value for key, value in pair.items() if key != "uniform_frame_indices"}
    with pytest.raises(RegressionError, match="uniform-frame coverage"):
        check_video_precision(
            [pair] * 3, missing, reject_uniform_frames=True, min_psnr_db=40, psnr_drop_db=3, ssim_drop=0.04
        )


@pytest.mark.parametrize("platform", ["cuda", "npu"])
def test_late_corruption_fails_even_when_global_pixel_scores_pass(tmp_path: Path, platform: str) -> None:
    reference, candidate = tmp_path / "reference.avi", tmp_path / "candidate.avi"
    _write_video(reference, [40] * 512, fps=16)
    _write_video(candidate, [40] * 504 + [60] * 8, fps=16)
    repeats = [quality.compare_video_pair(reference, reference, window_seconds=2, ssim_stride=16)] * 3
    measured = quality.compare_video_pair(reference, candidate, window_seconds=2, ssim_stride=16)
    assert measured["total"]["psnr_db"] > 40
    assert measured["ssim"]["mean"] >= 0.99
    limits = json.loads((Path(__file__).parents[2] / "tests/dfx/lingbot_world/quality.json").read_text())
    result = check_video_precision(
        repeats,
        measured,
        min_psnr_db=limits["min_psnr_db"],
        **limits["pixel_budgets"][platform],
    )
    assert not result["passed"]
    assert result["psnr"][0]["passed"]
    assert "window 15" in result["violations"][0]


def test_pixel_gate_rejects_incomplete_window_coverage(tmp_path: Path) -> None:
    reference = tmp_path / "reference.avi"
    _write_video(reference, [40] * 16)
    measured = quality.compare_video_pair(reference, reference, ssim_stride=4)
    assert check_video_precision([measured] * 2, measured, min_psnr_db=40, psnr_drop_db=1, ssim_drop=0.01)["passed"]
    with pytest.raises(RegressionError, match="same frames"):
        check_video_precision(
            [measured, {**measured, "windows": []}],
            measured,
            min_psnr_db=40,
            psnr_drop_db=1,
            ssim_drop=0.01,
        )


@pytest.mark.parametrize("change", ["length", "fps", "geometry"])
def test_video_pair_rejects_metadata_mismatches(tmp_path: Path, change: str) -> None:
    reference, candidate = tmp_path / "reference.avi", tmp_path / "candidate.avi"
    _write_video(reference, [40] * 8)
    _write_video(
        candidate,
        [40] * (4 if change == "length" else 8),
        fps=8 if change == "fps" else 4,
        size=(48, 24) if change == "geometry" else (32, 24),
    )
    with pytest.raises(quality.VideoQualityError, match="mismatch"):
        quality.compare_video_pair(reference, candidate)


def test_truncated_real_video_cannot_pass_on_matching_headers(tmp_path: Path) -> None:
    reference, truncated = tmp_path / "reference.avi", tmp_path / "truncated.avi"
    _write_video(reference, [40] * 20)
    content = reference.read_bytes()
    truncated.write_bytes(content[: len(content) // 2])
    with pytest.raises(quality.VideoQualityError):
        quality.compare_video_pair(reference, truncated)


def test_video_pair_rejects_empty_video(tmp_path: Path) -> None:
    video = tmp_path / "empty.avi"
    video.write_bytes(b"")
    with pytest.raises(quality.VideoQualityError, match="Cannot open|Empty"):
        quality.compare_video_pair(video, video)


@pytest.fixture
def evaluator_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"mock evaluator input")
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(
        quality.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout="a" * 40 + "\n"),
    )
    return {
        "videos": {"reference": video, "candidate": video},
        "output_dir": tmp_path / "output",
        "suite": "vbench_long",
        "repo": repo,
        "python": Path(sys.executable),
        "dimensions": ["subject_consistency"],
    }


def _worker_output(monkeypatch: pytest.MonkeyPatch, payload: dict, *, returncode: int = 0) -> list[dict]:
    calls = []

    def run(command: list[str], repo: Path, log: Path, timeout_s: float) -> int:
        manifest = json.loads(Path(command[-1]).read_text())
        calls.append({"command": command, "manifest": manifest})
        (Path(manifest["run_dir"]) / "worker_result.json").write_text(json.dumps(payload))
        return returncode

    monkeypatch.setattr(quality, "_run_worker", run)
    return calls


def test_quality_adapter_checks_every_video_and_records_revision(
    evaluator_inputs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _worker_output(
        monkeypatch,
        {
            "scores": {"reference": {"subject_consistency": 0.8}, "candidate": {"subject_consistency": 0.7}},
            "errors": [],
        },
    )
    result = quality.evaluate_video_quality(**evaluator_inputs)
    assert result["status"] == "passed"
    assert result["scores"]["candidate"]["subject_consistency"] == 0.7
    assert result["repo_revision"] == "a" * 40
    assert result["mode"] == "long_custom_input"
    assert calls[0]["command"][1] == "-c"
    assert calls[0]["manifest"]["videos"].keys() == {"reference", "candidate"}


@pytest.mark.parametrize(
    "payload",
    [
        {"scores": {"reference": {"subject_consistency": 0.8}}},
        {"scores": {"reference": {}, "candidate": {"subject_consistency": 0.7}}},
        {"scores": {"reference": {"subject_consistency": 0.8}, "candidate": {"subject_consistency": None}}},
        {"scores": {"reference": {"subject_consistency": 0.8}, "candidate": {"subject_consistency": float("nan")}}},
        {"scores": {"reference": {"subject_consistency": 0.8}, "candidate": {"subject_consistency": True}}},
        {"scores": {}, "errors": ["HPSv3 checkpoint missing"]},
    ],
)
def test_quality_adapter_fails_closed_on_invalid_scores(
    evaluator_inputs: dict, monkeypatch: pytest.MonkeyPatch, payload: dict
) -> None:
    _worker_output(monkeypatch, payload)
    result = quality.evaluate_video_quality(**evaluator_inputs)
    assert result["status"] == "failed"
    assert result["errors"]
    assert result["scores"] == {}


def test_quality_adapter_rejects_nonzero_exit(evaluator_inputs: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    _worker_output(monkeypatch, {"scores": {}}, returncode=2)
    result = quality.evaluate_video_quality(**evaluator_inputs)
    assert result["status"] == "failed"
    assert "code 2" in result["errors"][0]


def test_quality_adapter_rejects_unsupported_long_dimension(
    evaluator_inputs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _worker_output(monkeypatch, {})
    evaluator_inputs["dimensions"] = ["temporal_flickering"]
    result = quality.evaluate_video_quality(**evaluator_inputs)
    assert result["status"] == "failed"
    assert calls == []


def test_quality_adapter_does_not_reuse_stale_outputs(evaluator_inputs: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    _worker_output(
        monkeypatch,
        {
            "scores": {"reference": {"subject_consistency": 0.8}, "candidate": {"subject_consistency": 0.7}},
            "errors": [],
        },
    )
    first = quality.evaluate_video_quality(**evaluator_inputs)
    second = quality.evaluate_video_quality(**evaluator_inputs)
    assert first["status"] == second["status"] == "passed"
    assert first["raw_result"] != second["raw_result"]


@pytest.mark.parametrize("suite", ["vbench", "vbench_long"])
def test_quality_adapter_rejects_non_mp4_vbench_input(
    evaluator_inputs: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, suite: str
) -> None:
    video = tmp_path / "video.avi"
    video.write_bytes(b"mock video")
    evaluator_inputs.update(suite=suite, videos={"reference": video})
    calls = _worker_output(monkeypatch, {})
    result = quality.evaluate_video_quality(**evaluator_inputs)
    assert result["status"] == "failed"
    assert "require MP4 inputs" in result["errors"][0]
    assert calls == []


def test_quality_adapter_preserves_virtualenv_python_symlink(
    evaluator_inputs: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    executable = tmp_path / "venv" / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    evaluator_inputs["python"] = executable
    calls = _worker_output(
        monkeypatch,
        {"scores": {"reference": {"subject_consistency": 0.8}, "candidate": {"subject_consistency": 0.7}}},
    )
    result = quality.evaluate_video_quality(**evaluator_inputs)
    assert result["status"] == "passed"
    assert calls[0]["command"][0] == str(executable)


def test_quality_adapter_reports_timeout(evaluator_inputs: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(*args: Any, **kwargs: Any) -> int:
        raise subprocess.TimeoutExpired("evaluator", 1)

    monkeypatch.setattr(quality, "_run_worker", timeout)
    result = quality.evaluate_video_quality(**evaluator_inputs)
    assert result["status"] == "failed"
    assert "TimeoutExpired" in result["errors"][0]


@pytest.mark.parametrize("suite", ["vbench", "vbench_long"])
@pytest.mark.parametrize("wrong_identity", [False, True])
def test_worker_checks_native_vbench_video_identity(tmp_path: Path, suite: str, wrong_identity: bool) -> None:
    repo, run_dir = tmp_path / "repo", tmp_path / "run"
    repo.mkdir()
    run_dir.mkdir()
    (repo / "torch.py").write_text("def device(value): return value\n")
    for package in ("vbench", "vbench2_beta_long"):
        (repo / package).mkdir()
    (repo / "vbench" / "__init__.py").write_text(
        "import json\nfrom pathlib import Path\n"
        "class VBench:\n"
        "    def __init__(self, device, info, output): self.output = Path(output)\n"
        "    def evaluate(self, stage, name, dimension_list, mode, **kwargs):\n"
        "        stage = Path(stage)\n"
        "        if mode == 'long_custom_input':\n"
        "            assert kwargs['dev_flag'] is True\n"
        "            video = stage / 'split_clip/rollout-0/rollout-0_000.mp4'\n"
        "            video.parent.mkdir(parents=True)\n"
        "            video.write_bytes(b'fake clip')\n"
        "        else:\n"
        "            video = next(stage.glob('*.mp4'))\n"
        + ("        video = stage / 'unrequested.mp4'\n" if wrong_identity else "")
        + "        row = [0.8, [{'video_path': str(video), 'video_results': 0.8}]]\n"
        "        raw = {dim: row for dim in dimension_list}\n"
        "        (self.output / (name + '_eval_results.json')).write_text(json.dumps(raw))\n"
    )
    (repo / "vbench2_beta_long" / "__init__.py").write_text("from vbench import VBench as VBenchLong\n")
    videos = {}
    for label in ("reference", "candidate"):
        path = tmp_path / (label + ".mp4")
        path.write_bytes(b"fake video")
        videos[label] = str(path)
    manifest_path = run_dir / "inputs.json"
    manifest_path.write_text(
        json.dumps(
            {
                "repo": str(repo),
                "run_dir": str(run_dir),
                "suite": suite,
                "mode": "long_custom_input" if suite == "vbench_long" else "custom_input",
                "dimensions": ["subject_consistency"],
                "videos": videos,
            }
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", quality._EVALUATOR_WORKER, str(manifest_path)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    raw = json.loads((run_dir / "worker_result.json").read_text())
    if wrong_identity:
        assert raw["scores"] == {}
        assert len(raw["errors"]) == 2
    else:
        assert raw["scores"] == {label: {"subject_consistency": 0.8} for label in videos}
        assert raw["errors"] == []


@pytest.mark.parametrize(
    ("raw_detail", "backend", "fails"),
    [
        ("float('nan')", "farneback", True),
        ("float('inf')", "raft", True),
        ("np.float32('nan')", "unknown", True),
        ("np.float32(0.25)", "raft", False),
        ("np.float32(0.25)", "farneback", False),
        ("np.float32(0.25)", "unknown", False),
    ],
)
def test_worker_rejects_hps_nonfinite_details_and_records_flow_backend(
    evaluator_inputs: dict, raw_detail: str, backend: str, fails: bool
) -> None:
    pytest.importorskip("numpy")
    repo = evaluator_inputs["repo"]
    for directory in ("src", "src/metrics", "src/metrics/video_quality"):
        (repo / directory).mkdir(exist_ok=True)
        (repo / directory / "__init__.py").write_text("")
    metric = (
        "object()"
        if backend == "unknown"
        else f"SimpleNamespace(raft_model={'object()' if backend == 'raft' else 'None'})"
    )
    (repo / "src/metrics/video_quality/evaluator.py").write_text(
        "import numpy as np\nfrom types import SimpleNamespace\n"
        "class VideoQualityEvaluator:\n"
        "    def __init__(self, **kwargs):\n"
        f"        self._metric_instances = {{'dynamic_degree': {metric}}}\n"
        "    def evaluate(self, video, **kwargs):\n"
        "        return {'video_path': video, 'metrics': {\n"
        "            'dynamic_degree': {'score': np.float64(0.0)},\n"
        "            'hpsv3_quality': {'score': np.float32(1.0),\n"
        f"                'details': {{'hpsv3_quality_raw_scores': [{raw_detail}]}}}}}}}}\n"
    )
    evaluator_inputs.update(suite="wbench", dimensions=["hpsv3_quality", "dynamic_degree"])
    result = quality.evaluate_video_quality(**evaluator_inputs)
    assert result["status"] == ("failed" if fails else "passed")
    assert result["protocol"]["dynamic_degree_backend"] == backend
    assert "thresholds require calibration" in result["limitations"][0]
    assert result["native_provenance"]["reference"]["dynamic_degree_backend"] == backend
    if fails:
        assert result["scores"] == {}
        assert "Out of range float values" in result["errors"][0]
    else:
        assert result["scores"]["reference"] == {"hpsv3_quality": 1.0, "dynamic_degree": 0.0}
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "reference_psnr,reference_ssim,candidate_psnr,candidate_ssim",
    [
        ([26.0183304, 23.5055139, 24.4058944], [0.4613212, 0.4506127, 0.4628122], 22.3198337, 0.4450724),
        ([30.0] * 3, [0.3195815, 0.2982344, 0.3221995], 30.0, 0.2779404),
    ],
)
@pytest.mark.parametrize("platform", ["cuda", "npu"])
def test_full_rollout_pixel_calibration_remains_bounded(
    reference_psnr, reference_ssim, candidate_psnr, candidate_ssim, platform
):
    # Observed H100-generated boardwalk window 3 and street SSIM comparisons.
    # The input populations are retained; target-runner calibration is separate.
    def row(psnr, ssim):
        distance = {"psnr_db": psnr, "psnr_is_infinite": False}
        return {
            "total": distance,
            "windows": [{"index": 3, "start_frame": 96, "frame_count": 32, **distance}],
            "ssim": {"mean": ssim},
        }

    reference = [row(p, s) for p, s in zip(reference_psnr, reference_ssim)]
    candidate = row(candidate_psnr, candidate_ssim)
    assert not check_video_precision(reference, candidate, min_psnr_db=40, psnr_drop_db=1, ssim_drop=0.01)["passed"]
    limits = json.loads((Path(__file__).parents[2] / "tests/dfx/lingbot_world/quality.json").read_text())
    budgets = limits["pixel_budgets"][platform]
    options = {"min_psnr_db": limits["min_psnr_db"], **budgets}
    assert check_video_precision(reference, candidate, **options)["passed"]
    worse_psnr = row(min(reference_psnr) - budgets["psnr_drop_db"] - 0.1, candidate_ssim)
    worse_ssim = row(candidate_psnr, min(reference_ssim) - budgets["ssim_drop"] - 0.001)
    assert not check_video_precision(reference, worse_psnr, **options)["passed"]
    assert not check_video_precision(reference, worse_ssim, **options)["passed"]
