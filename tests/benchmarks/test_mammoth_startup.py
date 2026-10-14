# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression checks for benchmark completion and timestamp boundaries (no GPU)."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]
BENCH = Path(__file__).resolve().parents[2] / "benchmarks" / "mammoth_moda2"
spec = importlib.util.spec_from_file_location("parse_startup_log", BENCH / "parse_startup_log.py")
assert spec and spec.loader
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)


@pytest.mark.parametrize(
    ("start", "end"),
    [("09-10 23:59:58", "09-11 00:00:03"), ("12-31 23:59:58", "01-01 00:00:03")],
)
def test_timeline_date_rollover(start, end):
    lines = [
        f"INFO {start} [Omni] Initializing with model test",
        f"INFO {end} AsyncOmniEngine initialized in 5.00 seconds",
    ]
    assert parser.timeline(lines)["phases"]["engine_ready_s"] == 5


def test_incomplete_log_is_not_a_success(tmp_path):
    log = tmp_path / "failed.log"
    log.write_text("RuntimeError: No available memory for the cache blocks\n")
    record = parser.parse(str(log))
    assert record["status"] == "incomplete"
    assert "Incomplete log" in parser.to_markdown([record])
    result = subprocess.run(
        [sys.executable, str(BENCH / "parse_startup_log.py"), str(log), "--require-complete"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "missing BENCH_JSON" in result.stderr


@pytest.mark.parametrize("exit_code", [0, 23])
def test_storage_script_completion_and_log_selection(tmp_path, exit_code):
    # Copy the wrapper beside a failing engine and the real parser. No GPU or
    # cache eviction: local-warm reads one tiny dummy shard before the failure.
    for name in ("bench_storage_scenarios.sh", "parse_startup_log.py"):
        shutil.copy(BENCH / name, tmp_path / name)
    completion = 'BENCH_JSON {"label": "current"}'
    (tmp_path / "bench_startup.py").write_text(
        f"raise SystemExit({exit_code})\n" if exit_code else f"print({completion!r})\n"
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"test")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "python").symlink_to(sys.executable)
    env = dict(
        os.environ,
        PATH=str(binaries) + os.pathsep + os.environ["PATH"],
        MODEL_LOCAL=str(model),
        SCENARIOS="local-warm",
        OUT_DIR=str(tmp_path / "out"),
    )
    output = tmp_path / "out"
    (output / "logs").mkdir(parents=True)
    (output / "summary.json").write_text("stale summary")
    (output / "logs" / "old.log").write_text('BENCH_JSON {"label": "old"}\n')
    result = subprocess.run(
        ["bash", str(tmp_path / "bench_storage_scenarios.sh")], env=env, capture_output=True, text=True
    )
    assert result.returncode == exit_code
    if exit_code:
        assert "BENCH_ALL_DONE" not in result.stdout
        assert not (output / "summary.json").exists()
    else:
        summary = json.loads((output / "summary.json").read_text())
        assert [record["bench"]["label"] for record in summary] == ["current"]


def test_raw_loader_checks_shards_and_cpu_materialization(tmp_path, monkeypatch, capsys):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    spec = importlib.util.spec_from_file_location("raw_load_bench", BENCH / "raw_load_bench.py")
    assert spec and spec.loader
    raw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(raw)
    safetensors.save_file({"weight": torch.ones(16)}, tmp_path / "model-00006-of-00008.safetensors")
    for selection in ("999", "6,999"):
        with pytest.raises(SystemExit, match="not found"):
            raw.shard_files(str(tmp_path), selection)
    assert len(raw.shard_files(str(tmp_path), "6")) == 1
    monkeypatch.setattr(sys, "argv", ["raw", "--model", str(tmp_path), "--shards", "6", "--no-cuda"])
    raw.main()
    result = json.loads(capsys.readouterr().out.removeprefix("RAWLOAD_JSON "))
    assert result["n_tensors"] == 1
    assert result["h2d_s"] == 0
    assert "materialize_s" in result and "read_gbps" not in result
