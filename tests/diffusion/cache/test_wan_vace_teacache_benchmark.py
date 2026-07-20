# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""L1 tests for the Wan VACE TeaCache benchmark helpers."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pytest

from benchmarks.diffusion import wan_vace_teacache as benchmark

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cpu,
    pytest.mark.diffusion,
    pytest.mark.cache,
    pytest.mark.benchmark,
]


@dataclass
class _FakeResponse:
    content: bytes
    headers: dict[str, str]
    status_checked: bool = False

    def raise_for_status(self) -> None:
        self.status_checked = True


def _generation_config() -> benchmark.GenerationConfig:
    return benchmark.GenerationConfig(
        prompt="A cat walking on a street, high quality video",
        negative_prompt="low quality, blurry",
        width=1280,
        height=736,
        num_frames=61,
        fps=16,
        num_inference_steps=20,
        guidance_scale=5.0,
        boundary_ratio=0.875,
        flow_shift=3.0,
        seed=1,
        model="example/model",
    )


def test_generation_form_fields_match_sync_video_api() -> None:
    fields = _generation_config().form_fields()

    assert fields == {
        "prompt": "A cat walking on a street, high quality video",
        "negative_prompt": "low quality, blurry",
        "width": "1280",
        "height": "736",
        "num_frames": "61",
        "fps": "16",
        "num_inference_steps": "20",
        "guidance_scale": "5.0",
        "boundary_ratio": "0.875",
        "flow_shift": "3.0",
        "seed": "1",
        "model": "example/model",
    }
    assert benchmark._build_endpoint("http://localhost:8090") == "http://localhost:8090/v1/videos/sync"


def test_request_once_records_response_metrics_and_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_video = tmp_path / "source.mp4"
    input_video.write_bytes(b"source-video")
    output_path = tmp_path / "output.mp4"
    response = _FakeResponse(
        content=b"generated-video",
        headers={
            "content-type": "video/mp4",
            "X-Inference-Time-S": "2.5",
            "X-Stage-Durations": '{"diffuse":1.75,"vae.decode":0.25,"queue_wait_ms":12.5}',
            "X-Peak-Memory-MB": "4096.5",
            "X-Model": "example/model",
            "X-Request-Id": "video_sync-123",
        },
    )

    def fake_post(
        endpoint: str,
        *,
        data: dict[str, str],
        files: dict[str, tuple[str, BinaryIO, str]],
        headers: dict[str, str],
        timeout: float,
    ) -> _FakeResponse:
        assert endpoint == "http://localhost:8090/v1/videos/sync"
        assert data == _generation_config().form_fields()
        assert files["input_reference"][0] == "source.mp4"
        assert files["input_reference"][1].read() == b"source-video"
        assert files["input_reference"][2] == "video/mp4"
        assert headers == {"Accept": "video/mp4"}
        assert timeout == 30.0
        return response

    monkeypatch.setattr(benchmark.requests, "post", fake_post)
    measurement = benchmark._request_once(
        endpoint="http://localhost:8090/v1/videos/sync",
        input_video=input_video,
        config=_generation_config(),
        timeout_s=30.0,
        phase="measured",
        index=1,
        output_path=output_path,
    )

    assert response.status_checked
    assert output_path.read_bytes() == b"generated-video"
    assert measurement.wall_time_ms >= 0
    assert measurement.server_inference_time_ms == 2500.0
    assert measurement.server_stage_durations_ms == {
        "diffuse": 1750.0,
        "vae.decode": 250.0,
        "queue_wait_ms": 12.5,
    }
    assert measurement.server_peak_memory_mb == 4096.5
    assert measurement.server_model == "example/model"
    assert measurement.request_id == "video_sync-123"
    assert measurement.response_bytes == len(b"generated-video")


def test_request_benchmark_writes_reproducible_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_video = tmp_path / "source.mp4"
    input_video.write_bytes(b"source")
    output_dir = tmp_path / "results"
    calls: list[tuple[str, int]] = []

    def fake_request_once(
        *,
        endpoint: str,
        input_video: Path,
        config: benchmark.GenerationConfig,
        timeout_s: float,
        phase: str,
        index: int,
        output_path: Path,
    ) -> benchmark.RequestMeasurement:
        assert endpoint == "http://localhost:8090/v1/videos/sync"
        assert input_video.name == "source.mp4"
        assert config == _generation_config()
        assert timeout_s == 60.0
        calls.append((phase, index))
        output_path.write_bytes(f"{phase}-{index}".encode())
        return benchmark.RequestMeasurement(
            phase=phase,
            index=index,
            wall_time_ms=float(index * 100),
            server_inference_time_ms=float(index * 90),
            server_stage_durations_ms={"diffuse": float(index * 80)},
            server_peak_memory_mb=4096.0,
            server_model="example/model",
            request_id=f"request-{phase}-{index}",
            response_bytes=1,
            sha256=f"hash-{phase}-{index}",
            output_path=str(output_path),
        )

    monkeypatch.setattr(benchmark, "_request_once", fake_request_once)
    monkeypatch.setattr(benchmark, "_environment_metadata", lambda: {"python": "test"})
    monkeypatch.setattr(benchmark, "_sha256_file", lambda path: f"sha256:{path.name}")
    args = argparse.Namespace(
        base_url="http://localhost:8090",
        input_video=str(input_video),
        prompt="A cat walking on a street, high quality video",
        negative_prompt="low quality, blurry",
        label="tea_0_2",
        output_dir=str(output_dir),
        server_command="vllm serve example/model --cache-backend tea_cache",
        server_hardware="8x Ascend 910B3 64GB",
        server_software=["torch=2.8.0", "vllm-omni=test"],
        model="example/model",
        width=1280,
        height=736,
        num_frames=61,
        fps=16,
        num_inference_steps=20,
        guidance_scale=5.0,
        boundary_ratio=0.875,
        flow_shift=3.0,
        seed=1,
        warmup=1,
        runs=3,
        timeout=60.0,
    )

    benchmark.run_request_benchmark(args)

    manifest = json.loads((output_dir / "tea_0_2" / "manifest.json").read_text(encoding="utf-8"))
    assert calls == [("warmup", 1), ("measured", 1), ("measured", 2), ("measured", 3)]
    assert manifest["server"] == {
        "command": "vllm serve example/model --cache-backend tea_cache",
        "hardware": "8x Ascend 910B3 64GB",
        "software": ["torch=2.8.0", "vllm-omni=test"],
    }
    assert manifest["input_video_sha256"] == "sha256:source.mp4"
    assert manifest["summary"]["wall_time_ms"]["count"] == 3


def test_measurement_summary_excludes_warmup_and_groups_stage_metrics() -> None:
    measurements = [
        benchmark.RequestMeasurement(
            phase="warmup",
            index=1,
            wall_time_ms=9999.0,
            server_inference_time_ms=9999.0,
            server_stage_durations_ms={"diffuse": 9000.0},
            server_peak_memory_mb=9999.0,
            server_model=None,
            request_id=None,
            response_bytes=1,
            sha256="warmup",
            output_path="warmup.mp4",
        ),
        benchmark.RequestMeasurement(
            phase="measured",
            index=1,
            wall_time_ms=1000.0,
            server_inference_time_ms=900.0,
            server_stage_durations_ms={"diffuse": 700.0, "vae.decode": 100.0},
            server_peak_memory_mb=4096.0,
            server_model=None,
            request_id=None,
            response_bytes=1,
            sha256="one",
            output_path="one.mp4",
        ),
        benchmark.RequestMeasurement(
            phase="measured",
            index=2,
            wall_time_ms=1200.0,
            server_inference_time_ms=1100.0,
            server_stage_durations_ms={"diffuse": 900.0, "vae.decode": 100.0},
            server_peak_memory_mb=4200.0,
            server_model=None,
            request_id=None,
            response_bytes=1,
            sha256="two",
            output_path="two.mp4",
        ),
    ]

    summary = benchmark._summarize_measurements(measurements)

    assert summary["wall_time_ms"]["count"] == 2
    assert summary["wall_time_ms"]["mean"] == 1100.0
    assert summary["server_inference_time_ms"]["mean"] == 1000.0
    assert summary["server_peak_memory_mb"]["max"] == 4200.0
    assert summary["server_stage_durations_ms"]["diffuse"]["mean"] == 800.0


def test_labeled_video_parser_rejects_duplicate_labels(tmp_path: Path) -> None:
    video = tmp_path / "candidate.mp4"
    video.write_bytes(b"video")

    with pytest.raises(ValueError, match="Duplicate candidate label"):
        benchmark._parse_labeled_videos(
            [f"tea={video}", f"tea={video}"],
            field_name="candidate",
        )


def test_pixel_metrics_detect_spatial_rearrangement(monkeypatch: pytest.MonkeyPatch) -> None:
    reference = np.array([[[0], [1]], [[2], [3]]], dtype=np.float32)
    rearranged = np.array([[[3], [2]], [[1], [0]]], dtype=np.float32)

    def identity_resize(frame: np.ndarray, size: int) -> np.ndarray:
        del size
        return frame

    monkeypatch.setattr(benchmark, "_resize_frame", identity_resize)
    metrics = benchmark._pixel_metrics([reference], [rearranged], resize=2)

    assert metrics["mean_absolute_pixel_diff"]["mean"] == 2.0
    assert metrics["frame_pearson"]["mean"] == pytest.approx(-1.0)


def test_quality_report_loads_dinov2_once_and_records_no_cache_variance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = {name: tmp_path / f"{name}.mp4" for name in ("source", "baseline", "baseline_repeat", "tea")}
    for index, path in enumerate(paths.values()):
        path.write_bytes(f"video-{index}".encode())

    frame_values = {
        paths["source"]: 10,
        paths["baseline"]: 12,
        paths["baseline_repeat"]: 13,
        paths["tea"]: 14,
    }

    def fake_load_frames(video_path: Path, sample_count: int) -> list[np.ndarray]:
        assert sample_count == 1
        return [np.full((2, 2, 3), frame_values[video_path], dtype=np.uint8)]

    def identity_resize(frame: np.ndarray, size: int) -> np.ndarray:
        del size
        return frame.astype(np.float32) / 255.0

    build_calls = 0

    def fake_build_embedder(
        *,
        model_name_or_path: str,
        device: str,
        batch_size: int,
    ):
        nonlocal build_calls
        build_calls += 1
        assert (model_name_or_path, device, batch_size) == ("local-dinov2", "cpu", 2)

        def embed(frames: list[np.ndarray]) -> np.ndarray:
            value = float(frames[0].mean()) / 255.0
            vector = np.array([[1.0, value]], dtype=np.float32)
            return vector / np.linalg.norm(vector, axis=1, keepdims=True)

        return embed

    monkeypatch.setattr(benchmark, "_load_sampled_frames", fake_load_frames)
    monkeypatch.setattr(benchmark, "_resize_frame", identity_resize)
    monkeypatch.setattr(benchmark, "_build_dinov2_embedder", fake_build_embedder)
    output_json = tmp_path / "quality.json"
    args = argparse.Namespace(
        source_video=str(paths["source"]),
        baseline_video=str(paths["baseline"]),
        baseline_repeat=[f"no_cache_02={paths['baseline_repeat']}"],
        candidate=[f"tea_0_2={paths['tea']}"],
        output_json=str(output_json),
        sample_frames=1,
        pixel_resize=2,
        dinov2_model="local-dinov2",
        device="cpu",
        batch_size=2,
    )

    benchmark.run_quality_comparison(args)

    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert build_calls == 1
    assert "dinov2_cosine" in report["baseline_vs_source"]
    assert "no_cache_02" in report["baseline_repeatability"]
    assert "tea_0_2" in report["candidates"]
    assert report["pixel_metric_note"].startswith("frame_pearson is not assumed")


def test_server_matrix_dry_run_prints_fixed_comparison_commands(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = benchmark.build_parser().parse_args(
        [
            "matrix",
            "--model",
            "example/model",
            "--input-video",
            str(tmp_path / "not-needed-for-dry-run.mp4"),
            "--output-dir",
            str(tmp_path / "results"),
            "--server-hardware",
            "2x H100 80GB",
            "--server-software",
            "torch=2.8.0",
            "--server-args=--tensor-parallel-size 2",
            "--dry-run",
        ]
    )

    benchmark.run_server_matrix(args)

    commands = json.loads(capsys.readouterr().out)
    assert [entry["label"] for entry in commands] == [
        "no_cache",
        "tea_0_1",
        "tea_0_2",
        "cache_dit",
    ]
    for entry in commands:
        assert "VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7260.0 vllm serve example/model --omni" in entry["server_command"]
        assert "--enable-diffusion-pipeline-profiler" in entry["server_command"]
        assert "--tensor-parallel-size 2" in entry["server_command"]
        assert "wan_vace_teacache.py" in entry["request_command"]
        assert " request " in entry["request_command"]

    assert "--cache-backend tea_cache" in commands[2]["server_command"]
    assert "rel_l1_thresh" in commands[2]["server_command"]
    assert "--cache-backend cache_dit" in commands[3]["server_command"]
    assert "max_warmup_steps" in commands[3]["server_command"]


def test_stop_server_tolerates_process_group_exit_race(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProcess:
        pid = 123

        def poll(self) -> None:
            return None

    def missing_process_group(process_group_id: int, requested_signal: int) -> None:
        assert process_group_id == 123
        assert requested_signal == benchmark.signal.SIGTERM
        raise ProcessLookupError

    monkeypatch.setattr(benchmark.os, "killpg", missing_process_group)

    benchmark._stop_server(FakeProcess(), timeout_s=1.0)


def test_server_matrix_persists_active_configuration_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_video = tmp_path / "source.mp4"
    input_video.write_bytes(b"source")
    output_dir = tmp_path / "results"
    args = benchmark.build_parser().parse_args(
        [
            "matrix",
            "--model",
            "example/model",
            "--input-video",
            str(input_video),
            "--output-dir",
            str(output_dir),
            "--server-hardware",
            "2x H100 80GB",
            "--skip-quality",
        ]
    )

    def server_is_not_running(health_url: str) -> bool:
        del health_url
        return False

    def fail_after_manifest(
        command: list[str],
        *,
        stdout: BinaryIO,
        stderr: int,
        start_new_session: bool,
        env: dict[str, str],
    ) -> None:
        del command, stdout, stderr, start_new_session
        assert env["VLLM_OMNI_VIDEO_SYNC_TIMEOUT"] == "7260.0"
        manifest = json.loads((output_dir / "matrix_manifest.json").read_text(encoding="utf-8"))
        assert manifest["active"] == "no_cache"
        assert manifest["completed"] == []
        assert [entry["label"] for entry in manifest["commands"]] == [
            "no_cache",
            "tea_0_1",
            "tea_0_2",
            "cache_dit",
        ]
        raise OSError("server launch failed")

    monkeypatch.setattr(benchmark, "_server_is_healthy", server_is_not_running)
    monkeypatch.setattr(benchmark.subprocess, "Popen", fail_after_manifest)

    with pytest.raises(OSError, match="server launch failed"):
        benchmark.run_server_matrix(args)


def test_matrix_quality_inputs_use_no_cache_repeats_and_first_candidate_clips(tmp_path: Path) -> None:
    args = benchmark.build_parser().parse_args(
        [
            "matrix",
            "--model",
            "example/model",
            "--input-video",
            str(tmp_path / "source.mp4"),
            "--output-dir",
            str(tmp_path / "results"),
            "--server-hardware",
            "1x accelerator",
            "--runs",
            "4",
            "--dry-run",
        ]
    )

    quality_args = benchmark._quality_namespace(args)

    assert quality_args.baseline_video.endswith("no_cache/measured_01.mp4")
    assert len(quality_args.baseline_repeat) == 3
    assert quality_args.baseline_repeat[-1].startswith("no_cache_04=")
    assert [candidate.partition("=")[0] for candidate in quality_args.candidate] == [
        "tea_0_1",
        "tea_0_2",
        "cache_dit",
    ]
