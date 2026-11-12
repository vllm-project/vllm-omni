# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Coverage and failure semantics for long-rollout quality diagnostics."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from vllm_omni.benchmarks.realtime_video import video_quality as quality

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
        "suite": "vbench",
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
    assert result["mode"] == "custom_input"
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


@pytest.mark.parametrize("suite", ["vbench"])
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


@pytest.mark.parametrize("suite", ["vbench"])
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
                "mode": "custom_input",
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


def test_semantic_video_coverage_checks_its_own_full_input(tmp_path):
    path = tmp_path / "rollout.avi"
    _write_video(path, [30, 60, 90], fps=4)
    assert quality.verify_video_coverage(path, frames=3, width=32, height=24, fps=4)["decoded_frames"] == 3
    with pytest.raises(quality.VideoQualityError):
        quality.verify_video_coverage(path, frames=4, width=32, height=24, fps=4)


@pytest.mark.parametrize("wrong_adapter_hash", [False, True])
def test_isolated_background_worker_loads_only_the_pinned_adapter_file(tmp_path, wrong_adapter_hash):
    """The evaluator venv has neither vLLM nor the Omni runtime dependencies."""
    repo, output = tmp_path / "native-fixture", tmp_path / "run"
    (repo / "vbench").mkdir(parents=True)
    output.mkdir()
    (repo / "torch.py").write_text("def device(value): return value\n")
    (repo / "vbench/background_consistency.py").write_text(
        "from types import SimpleNamespace\n"
        "def load(*args, **kwargs): raise AssertionError('native loader is not used in CPU fixture')\n"
        "clip = SimpleNamespace(load=load)\n"
    )
    (repo / "vbench/__init__.py").write_text(
        "import json\nfrom pathlib import Path\n"
        "class VBench:\n"
        "    def __init__(self, device, info, output): self.output = Path(output)\n"
        "    def evaluate(self, stage, name, dimension_list, **kwargs):\n"
        "        from . import background_consistency as background\n"
        "        assert background._lingbot_clip_inference_stats['gradients'] is False\n"
        "        assert background.clip.load.__module__ == 'video_encoder_adapter'\n"
        "        video = next(Path(stage).glob('*.mp4'))\n"
        "        raw = {'background_consistency': [0.8, [{'video_path': str(video), 'video_results': 0.8}]]}\n"
        "        (self.output / (name + '_eval_results.json')).write_text(json.dumps(raw))\n"
    )
    video = tmp_path / "cpu-fixture.mp4"
    video.write_bytes(b"scripted evaluator fixture, not an actual native video")
    adapter = Path(quality.__file__).with_name("encoder_inference.py").resolve()
    digest = hashlib.sha256(adapter.read_bytes()).hexdigest()
    manifest = {
        "repo": str(repo),
        "run_dir": str(output),
        "suite": "vbench",
        "mode": "custom_input",
        "dimensions": ["background_consistency"],
        "videos": {"cpu-fixture": str(video)},
        "encoder_adapter": str(adapter),
        "protocol": {"background_clip_inference": {"adapter_sha256": "0" * 64 if wrong_adapter_hash else digest}},
    }
    path = output / "inputs.json"
    path.write_text(json.dumps(manifest))
    # -I ignores inherited PYTHONPATH and -S removes site-packages: only the
    # stdlib, explicit fake native repository and hash-checked adapter can load.
    worker = (
        "import importlib.util\nassert importlib.util.find_spec('vllm_omni') is None\n"
        + quality._EVALUATOR_WORKER
        + "\nassert 'vllm_omni' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", worker, str(path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    raw = json.loads((output / "worker_result.json").read_text())
    if wrong_adapter_hash:
        assert not raw["scores"] and "differs from the recorded source" in raw["errors"][0]
    else:
        assert raw["errors"] == []
        assert raw["scores"] == {"cpu-fixture": {"background_consistency": 0.8}}
