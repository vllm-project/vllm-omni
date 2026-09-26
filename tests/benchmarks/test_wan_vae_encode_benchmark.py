# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Input, validation and reporting contracts for the standalone encoder benchmark."""

import json
import math
from types import SimpleNamespace

import pytest
import torch

from benchmarks.diffusion import bench_wan_vae_encode as benchmark
from benchmarks.diffusion.bench_wan_vae_encode import differences, make_pixels, parse_args

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_benchmark_reference_always_runs_first(monkeypatch):
    monkeypatch.setattr("sys.argv", ["bench", "--fast-path", "channels_last,lossless,channels_last"])
    args = parse_args()
    assert args.levels == ["off", "channels_last", "lossless"]
    assert args.warmup == 3 and args.iters == 10


def test_seeded_pixels_and_real_input(tmp_path):
    args = SimpleNamespace(input=None, size="32x16", frames=5, seed=123)
    a, b = make_pixels(args), make_pixels(args)
    assert torch.equal(a, b) and a.shape == (1, 3, 5, 16, 32)
    args.input = tmp_path / "pixels.pt"
    torch.save(a.to(torch.bfloat16), args.input)
    assert torch.equal(make_pixels(args), a.to(torch.bfloat16))
    torch.save(torch.full_like(a, float("nan")), args.input)
    with pytest.raises(ValueError, match="finite"):
        make_pixels(args)
    torch.save(torch.full_like(a, 2), args.input)
    with pytest.raises(ValueError, match="normalized"):
        make_pixels(args)


@pytest.mark.parametrize("size,frames", [("32x16", 6), ("31x16", 5)])
def test_invalid_input_schedule(size, frames):
    with pytest.raises(ValueError, match="1\\+4k"):
        make_pixels(SimpleNamespace(input=None, size=size, frames=frames, seed=0))


def test_metrics_distinguish_signed_zero_and_measure_relative_error():
    assert differences(torch.tensor([-0.0]), torch.tensor([0.0]))["bitwise_equal"] is False
    result = differences(torch.tensor([1.01]), torch.tensor([1.0]))
    assert result["normalized_rmse"] == pytest.approx(0.01, abs=1e-6)
    assert result["max_abs_diff"] == pytest.approx(0.01, abs=1e-6)
    assert result["bitwise_equal"] is False


def _result(level, value=1.0, reconstruction=None, dtype=torch.float32):
    tensor = torch.tensor([value], dtype=dtype)
    stats = dict(
        level=level,
        status="ok",
        median_s=1.25,
        min_s=1.0,
        times_s=[1.0, 1.5],
        frames_per_s=4.0,
        peak_gib=2.0,
        install_s=0.05,
        first_call_s=3.0,
    )
    return stats, tensor, tensor, tensor, reconstruction


def test_lossless_failure_identifies_every_tensor_and_keeps_timings():
    result = _result("lossless", -0.0)
    benchmark.validate_result(result, _result("off", 0.0))
    stats = result[0]
    assert stats["status"] == "validation_failed"
    assert stats["speedup"] == 1.0 and stats["median_s"] == 1.25
    assert len(stats["errors"]) == 3
    for name, error in zip(("moments", "mean", "logvar"), stats["errors"]):
        assert f"[lossless] {name}: bitwise_equal=False, required True" in error
        assert "max_abs_diff=0" in error
    assert "failed for 1 level(s)" in benchmark.failure_summary([stats])


@pytest.mark.parametrize("dtype,limit", [(torch.bfloat16, 0.02), (torch.float16, 0.01), (torch.float32, 0.01)])
@pytest.mark.parametrize("case", ["below", "boundary", "above"])
def test_channels_last_threshold_and_error_precision(monkeypatch, dtype, limit, case):
    nrmse = {"below": limit / 2, "boundary": limit, "above": math.nextafter(limit, 1.0)}[case]
    failed = case == "above"
    monkeypatch.setattr(
        benchmark,
        "differences",
        lambda *_: dict(bitwise_equal=False, max_abs_diff=0.02, normalized_rmse=nrmse),
    )
    result = _result("channels_last", dtype=dtype)
    benchmark.validate_result(result, _result("off", dtype=dtype))
    assert result[0]["status"] == ("validation_failed" if failed else "ok")
    assert bool(result[0]["errors"]) is failed
    if failed:
        assert f"normalized_rmse={nrmse!r} exceeds {limit:g} ({limit:.0%})" in result[0]["errors"][0]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("level", ["off", "lossless", "channels_last"])
def test_nonfinite_metrics_report_level_and_tensor(value, level):
    result = _result(level, value)
    benchmark.validate_result(result, result if level == "off" else _result("off"))
    assert result[0]["status"] == "validation_failed"
    assert any(f"[{level}] mean: nonfinite comparison metrics" in error for error in result[0]["errors"])


@pytest.mark.parametrize(
    "error,failed", [(0.0, False), (0.001, False), (0.01, True), (float("nan"), True), (float("inf"), True)]
)
def test_reconstruction_quality_reports_psnr_threshold(error, failed):
    reference = _result("off", reconstruction=torch.zeros(2))
    result = _result("channels_last", reconstruction=torch.full((2,), error))
    benchmark.validate_result(result, reference)
    stats = result[0]
    assert stats["status"] == ("validation_failed" if failed else "ok")
    if failed:
        assert len(stats["errors"]) == 1
        assert "[channels_last] reconstruction: PSNR=" in stats["errors"][0]
        assert "required >= 50 dB (MSE=" in stats["errors"][0]
    elif error == 0:
        assert stats["reconstruction_psnr_db"] == "inf"


def test_missing_reference_is_unvalidated_not_ok():
    result = _result("lossless")
    benchmark.validate_result(result, None)
    assert result[0]["status"] == "unvalidated"
    assert "speedup" not in result[0] and "moments" not in result[0]
    assert "reference level 'off' is unavailable" in result[0]["errors"][0]


def test_tables_show_timings_quality_statuses_and_optional_reconstruction(capsys):
    reference = _result("off", reconstruction=torch.zeros(2))
    failed = _result("lossless", 1.01, reconstruction=torch.full((2,), 0.01))
    unvalidated = _result("channels_last")
    benchmark.validate_result(reference, reference)
    benchmark.validate_result(failed, reference)
    benchmark.validate_result(unvalidated, None)
    oom = dict(level="channels_last", status="oom", errors=["out of memory"])
    environment = dict(
        gpu="test GPU", dtype="bf16", input_shape=[1, 3, 5, 16, 32], model="test model", world_size=1, tiling=False
    )
    benchmark.print_results([reference[0], failed[0], oom, unvalidated[0]], environment)
    text = capsys.readouterr().out
    assert "test GPU | bf16 | RGB 1x3x5x16x32" in text
    assert "Median (s)" in text and "Peak (GiB)" in text and "Speedup" in text
    assert "PASS" in text and "FAIL" in text and "OOM" in text and "NO REF" in text
    assert "1.2500" in text and "1.00x" in text
    assert "Max abs diff" in text and "NRMSE" in text and "logvar" in text
    assert "PSNR (dB)" in text and "inf" in text and "46.02" in text
    assert '"median_s"' not in text  # No raw JSON dump on stdout.


@pytest.fixture
def cpu_main(monkeypatch, tmp_path):
    """Exercise the real CLI/reporting flow with only GPU execution mocked."""
    output = tmp_path / "results.json"
    monkeypatch.setattr("sys.argv", ["bench", "--tiny", "--size", "32x16", "--frames", "1", "--json", str(output)])
    monkeypatch.setattr(benchmark, "distributed_setup", lambda _: 0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "test GPU")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    monkeypatch.setattr(benchmark.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(benchmark, "max_across_ranks", lambda value: value)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    # main sets this backend option; restore it even when the benchmark fails.
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", torch.backends.cudnn.allow_tf32)
    return output


@pytest.mark.parametrize("failed_level", [None, "lossless", "channels_last"])
def test_main_tables_json_and_exit_match_validation(monkeypatch, cpu_main, capsys, failed_level):
    monkeypatch.setattr(
        benchmark,
        "run_level",
        lambda args, level, pixels, rank: _result(level, 1.02 if level == failed_level else 1.0),
    )
    if failed_level:
        with pytest.raises(SystemExit) as exc:
            benchmark.main()
        assert f"[{failed_level}] moments:" in str(exc.value)
        assert (
            "bitwise_equal=False" in str(exc.value) if failed_level == "lossless" else "exceeds 0.01" in str(exc.value)
        )
    else:
        benchmark.main()
    text = capsys.readouterr().out
    report = json.loads(cpu_main.read_text())
    assert "Detailed results saved to" in text
    assert "All requested validation checks passed." in text if failed_level is None else "FAIL" in text
    for stats in report["results"]:
        failed = stats["level"] == failed_level
        assert stats["status"] == ("validation_failed" if failed else "ok")
        assert bool(stats["errors"]) is failed
        assert stats["times_s"] == [1.0, 1.5]


@pytest.mark.parametrize("oom_levels", [("lossless",), ("off",), ("off", "lossless", "channels_last")])
def test_main_oom_results_and_missing_reference(monkeypatch, cpu_main, capsys, oom_levels):
    def run(args, level, pixels, rank):
        if level in oom_levels:
            raise torch.OutOfMemoryError(f"[{level}] out of memory during timed encoding: allocation detail")
        return _result(level)

    monkeypatch.setattr(benchmark, "run_level", run)
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert "allocation detail" in str(exc.value)
    text = capsys.readouterr().out
    assert "OOM" in text and "All requested validation checks passed" not in text
    for stats in json.loads(cpu_main.read_text())["results"]:
        expected = "oom" if stats["level"] in oom_levels else "unvalidated" if "off" in oom_levels else "ok"
        assert stats["status"] == expected
        assert bool(stats["errors"]) == (expected != "ok")
    if oom_levels == ("off",):
        assert "NO REF" in text
        assert "reference level 'off' is unavailable" in str(exc.value)


def test_model_loading_oom_preserves_context_and_cause(monkeypatch):
    def fail(_):
        raise torch.OutOfMemoryError("Tried to allocate 2 GiB")

    monkeypatch.setattr(benchmark, "load_vae", fail)
    with pytest.raises(torch.OutOfMemoryError) as exc:
        benchmark.run_level(SimpleNamespace(dtype="bf16"), "lossless", torch.zeros(1, 3, 5, 16, 32), 0)
    assert "[lossless] out of memory during model loading" in str(exc.value)
    assert "input=[1, 3, 5, 16, 32], dtype=bf16" in str(exc.value)
    assert "Tried to allocate 2 GiB" in str(exc.value)
    assert isinstance(exc.value.__cause__, torch.OutOfMemoryError)


@pytest.mark.parametrize(
    "phase,encode_call",
    [("first encode", 1), ("encode warmup", 2), ("timed encoding", 3), ("reference-decoder reconstruction", None)],
)
def test_run_level_oom_identifies_execution_phase(monkeypatch, phase, encode_call):
    calls = 0
    tensor = torch.ones(1)

    def encode(_):
        nonlocal calls
        calls += 1
        if calls == encode_call:
            raise torch.OutOfMemoryError("encode allocation failed")
        return SimpleNamespace(latent_dist=SimpleNamespace(parameters=tensor, logvar=tensor, mode=lambda: tensor))

    def decode(_):
        raise torch.OutOfMemoryError("decode allocation failed")

    vae = SimpleNamespace(encode=encode, decode=decode, config=SimpleNamespace())
    report = SimpleNamespace(installed=True, patched={}, fused_silu_dtypes=())
    monkeypatch.setattr(benchmark, "load_vae", lambda _: vae)
    monkeypatch.setattr(benchmark, "install_wan_vae_encoder_fastpath", lambda *args, **kwargs: report)
    monkeypatch.setattr(benchmark, "sync", lambda: None)
    monkeypatch.setattr(benchmark, "max_across_ranks", lambda value: value)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(torch.accelerator, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.accelerator, "max_memory_allocated", lambda: 0)
    original_to = torch.Tensor.to

    def cpu_to(tensor, *args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    args = SimpleNamespace(dtype="bf16", warmup=1, iters=1, check_reconstruction=True, profile=False)
    with pytest.raises(torch.OutOfMemoryError) as exc:
        benchmark.run_level(args, "channels_last", torch.zeros(1, 3, 5, 16, 32), 0)
    assert f"[channels_last] out of memory during {phase}" in str(exc.value)
    if phase == "reference-decoder reconstruction":
        assert "omit --check-reconstruction to benchmark encoding alone" in str(exc.value)
        assert "decode allocation failed" in str(exc.value)
