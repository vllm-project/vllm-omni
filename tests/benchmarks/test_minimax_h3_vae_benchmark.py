# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Guard the evidence and timing contracts of the H3 VAE benchmark."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _benchmark():
    name = "h3_vae_benchmark_test_target"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[2] / "benchmarks/diffusion/bench_minimax_h3_vae.py"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


def test_metrics_measure_same_lite_reference_before_quantization():
    metrics = _benchmark().compare_outputs(torch.zeros(1, 3, 2, 16, 16), torch.full((1, 3, 2, 16, 16), 0.01))
    assert metrics["psnr"] == pytest.approx(40.0, abs=1e-4)
    assert metrics["mae"] == pytest.approx(0.01)
    assert metrics["ssim"] == pytest.approx(0.5, abs=1e-4)
    assert _benchmark().compare_outputs(torch.zeros(1, 3, 2, 16, 16), torch.zeros(1, 3, 2, 16, 16))["psnr"] is None


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_metrics_reject_range_and_nonfinite_inputs(value):
    with pytest.raises(ValueError):
        _benchmark().compare_outputs(torch.zeros(1, 3, 2, 16, 16), torch.full((1, 3, 2, 16, 16), value))


def test_metrics_reject_unaligned_shapes():
    with pytest.raises(ValueError, match="shape"):
        _benchmark().compare_outputs(torch.zeros(1, 3, 2, 16, 16), torch.zeros(1, 3, 3, 16, 16))


def test_timing_excludes_warmup_and_synchronizes_boundaries(monkeypatch):
    benchmark = _benchmark()
    events = []
    ticks = iter(index * 0.001 for index in range(120))
    api = SimpleNamespace(
        synchronize=lambda: events.append("sync"),
        reset_peak_memory_stats=lambda: events.append("reset"),
        max_memory_allocated=lambda: 10,
        max_memory_reserved=lambda: 20,
    )
    monkeypatch.setattr(benchmark, "_device_api", lambda _: api)
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(ticks))

    def decode():
        events.append("decode")
        return torch.zeros(1)

    result = benchmark.measure_decode(decode, torch.device("cpu"))
    assert events.count("decode") == 3 * (5 + 20)
    assert events.count("reset") == 3
    assert events.count("sync") == 3 * (1 + 2 * 20)
    assert [len(group) for group in result["samples_ms"]] == [20, 20, 20]
    assert result["median_ms_by_round"] == pytest.approx([1.0, 1.0, 1.0])
    assert result["peak_allocated_bytes"] == 10
    assert result["peak_reserved_bytes"] == 20


def _report(tmp_path, variant, value, **changes):
    directory = tmp_path / variant
    directory.mkdir()
    save_file(
        {"video": torch.full((1, 3, 2, 16, 16), value)},
        directory / "decoded.safetensors",
    )
    report = {
        "variant": variant,
        "status": "measured",
        "output_file": "decoded.safetensors",
        "weights_sha256": "original" if variant == "original" else "lite",
        "environment": {
            "device": "npu:0",
            "device_name": "Ascend910B",
            "versions": {"torch": "2.8.0", "torch-npu": "2.8.0", "mindiesd": None},
            "code_revision": "same-revision",
            "source_sha256": {"vae.py": "same-source"},
        },
        "signature": {
            "input_sha256": "input",
            "shape": [1, 2, 2, 1, 1],
            "dtype": "fp32",
            "tile_size": 256,
            "tile_overlap": 64,
            "parallel_size": 1,
            "offload": "resident",
            "output_range": [0, 1],
        },
        "median_ms_by_round": [10.0, 11.0, 10.0] if variant == "original" else [4.0, 5.0, 4.0],
        "samples_ms": [[10.0] * 20] * 3,
    }
    report.update(changes)
    if "samples_ms" not in changes:
        report["samples_ms"] = [[value] * 20 for value in report["median_ms_by_round"]]
    path = directory / "report.json"
    path.write_text(json.dumps(report))
    return path


def _compare(tmp_path, paths):
    output = tmp_path / "comparison"
    code = _benchmark().main(
        [
            "--variant",
            "compare",
            "--reports",
            *map(str, paths),
            "--output-dir",
            str(output),
        ]
    )
    return code, json.loads((output / "report.json").read_text())


def test_reference_missing_is_not_validated(tmp_path):
    paths = [_report(tmp_path, "original", 0.0), _report(tmp_path, "lite", 0.5)]
    code, report = _compare(tmp_path, paths)
    assert code == 2
    assert report["status"] == "not_validated"


@pytest.mark.parametrize("failure", ["oom", "configuration", "wrong_weights"])
def test_oom_or_config_mismatch_cannot_pass_comparison(tmp_path, failure):
    original = _report(tmp_path, "original", 0.0)
    changes: dict[str, object] = {"status": "failed", "error": "out of memory"} if failure == "oom" else {}
    if failure == "configuration":
        changes["signature"] = {"input_sha256": "different-input"}
    if failure == "wrong_weights":
        changes["weights_sha256"] = "different-lite"
    lite = _report(tmp_path, "lite", 0.5, **changes)
    official = _report(tmp_path, "official", 0.5)
    code, report = _compare(tmp_path, [original, lite, official])
    assert code != 0
    assert report["status"] != "passed"


def test_quality_threshold_applies_only_to_same_lite_reference(tmp_path):
    paths = [
        _report(tmp_path, "original", 0.0),
        _report(tmp_path, "lite", 0.5),
        _report(tmp_path, "official", 0.5),
    ]
    code, report = _compare(tmp_path, paths)
    assert code == 0
    assert report["status"] == "passed"
    assert report["lite_reference"]["exact"]
    assert report["original_vs_lite"]["psnr"] < 40
    assert report["speedup_verified"] is True


@pytest.mark.parametrize(
    "change",
    [
        {"samples_ms": [[4.0] * 19] * 3},
        {"median_ms_by_round": [0.001] * 3, "samples_ms": [[4.0] * 20] * 3},
    ],
)
def test_insufficient_or_inconsistent_samples_cannot_verify_speedup(tmp_path, change):
    paths = [
        _report(tmp_path, "original", 0.0),
        _report(tmp_path, "lite", 0.5, **change),
        _report(tmp_path, "official", 0.5),
    ]
    code, report = _compare(tmp_path, paths)
    assert code != 0
    assert not report.get("speedup_verified", False)


def test_different_runtime_devices_cannot_verify_speedup(tmp_path):
    paths = [
        _report(
            tmp_path,
            "original",
            0.0,
            environment={"device": "npu:0", "device_name": "Ascend910B"},
        ),
        _report(tmp_path, "lite", 0.5, environment={"device": "cpu", "device_name": "CPU"}),
        _report(tmp_path, "official", 0.5),
    ]
    code, report = _compare(tmp_path, paths)
    assert code != 0
    assert report["status"] == "failed"


def test_reference_just_below_40_db_fails(tmp_path):
    paths = [
        _report(tmp_path, "original", 0.0),
        _report(tmp_path, "lite", 0.0101),
        _report(tmp_path, "official", 0.0),
    ]
    code, report = _compare(tmp_path, paths)
    assert code == 1
    assert report["lite_reference"]["psnr"] < 40
    assert report["speedup_verified"] is False


@pytest.mark.parametrize("field", ["versions", "code_revision", "source_sha256"])
@pytest.mark.parametrize("failure", ["different", "missing", "empty"])
def test_software_or_source_mismatch_cannot_verify_speedup(tmp_path, field, failure):
    original = _report(tmp_path, "original", 0.0)
    environment = json.loads(original.read_text())["environment"]
    if failure == "missing":
        del environment[field]
    elif failure == "empty":
        environment[field] = {} if field != "code_revision" else ""
    else:
        environment[field] = {
            "versions": {"torch": "2.9.0", "torch-npu": "2.8.0", "mindiesd": None},
            "code_revision": "other-revision",
            "source_sha256": {"vae.py": "modified-source"},
        }[field]
    lite = _report(tmp_path, "lite", 0.5, environment=environment)
    official = _report(tmp_path, "official", 0.5)
    code, report = _compare(tmp_path, [original, lite, official])
    assert code == 1
    assert report["status"] == "failed"
    assert not report.get("speedup_verified", False)


def test_official_reference_can_use_a_separate_environment(tmp_path):
    paths = [
        _report(tmp_path, "original", 0.0),
        _report(tmp_path, "lite", 0.5),
        _report(tmp_path, "official", 0.5, environment={"device": "cuda:0"}),
    ]
    code, report = _compare(tmp_path, paths)
    assert code == 0
    assert report["speedup_verified"] is True


def test_environment_fingerprints_decoder_helpers_and_remote_code(tmp_path, monkeypatch):
    benchmark = _benchmark()
    script = tmp_path / "benchmarks/diffusion/bench.py"
    script.parent.mkdir(parents=True)
    script.write_text("benchmark")
    helpers = tmp_path / "vllm_omni/diffusion/models/minimax_h3"
    helpers.mkdir(parents=True)
    temporal = helpers / "vae_temporal.py"
    temporal.write_text("old temporal implementation")
    component = tmp_path / "video_vae"
    component.mkdir()
    remote = component / "modeling_vae.py"
    remote.write_text("old remote implementation")
    monkeypatch.setattr(benchmark, "__file__", str(script))
    monkeypatch.setattr(benchmark, "_revision", lambda _: "same-revision")
    monkeypatch.setattr(benchmark, "_device_api", lambda _: SimpleNamespace())
    first = benchmark._environment(torch.device("cpu"), component_path=component)
    temporal.write_text("new temporal implementation")
    second = benchmark._environment(torch.device("cpu"), component_path=component)
    assert first["source_sha256"] != second["source_sha256"]
    remote.write_text("new remote implementation")
    third = benchmark._environment(torch.device("cpu"), component_path=component)
    assert second["source_sha256"] != third["source_sha256"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("versions", {"torch": None, "torch-npu": None}),
        ("source_sha256", {"vae.py": None}),
    ],
)
def test_matching_but_unknown_runtime_evidence_cannot_verify_speedup(tmp_path, field, value):
    original = _report(tmp_path, "original", 0.0)
    data = json.loads(original.read_text())
    data["environment"][field] = value
    original.write_text(json.dumps(data))
    lite = _report(tmp_path, "lite", 0.5, environment=data["environment"])
    official = _report(tmp_path, "official", 0.5)
    code, report = _compare(tmp_path, [original, lite, official])
    assert code == 1
    assert not report.get("speedup_verified", False)
