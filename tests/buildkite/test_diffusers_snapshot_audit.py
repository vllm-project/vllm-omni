# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reject drift in effective model configuration before comparison inference."""

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "tests/dfx/perf/tests/test_hunyuanvideo15_sglang.json"
spec = importlib.util.spec_from_file_location(
    "snapshot_audit", ROOT / "tests/dfx/perf/scripts/audit_diffusers_snapshot.py"
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def cached_snapshot(tmp_path):
    (config,) = json.loads(CONFIG.read_text())
    snapshot = tmp_path / "snapshots" / config["server_params"]["serve_args"]["revision"]
    for name, values in config["model_audit"]["config_values"].items():
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(values))
    return snapshot


def test_cached_configuration_records_revision_hashes_and_precision(cached_snapshot, monkeypatch):
    monkeypatch.setenv("BUILDKITE_JOB_ID", "job-id")
    result = module.audit_snapshot(cached_snapshot, CONFIG)
    assert result["status"] == "passed"
    assert result["purpose"] == "comparison_effective_config"
    assert result["job_id"] == "job-id"
    assert result["revision"] == cached_snapshot.name
    assert len(result["configs"]) == 3
    assert all(len(c["sha256"]) == 64 for c in result["configs"])
    assert result["precision"]["vae"] == "bf16"
    assert result["precision"]["vllm_reference_vae"] == "fp32"


@pytest.mark.parametrize(
    ("name", "key", "value"),
    [("guider/guider_config.json", "guidance_scale", 5.0), ("scheduler/scheduler_config.json", "shift", 3.0)],
)
def test_request_values_cannot_hide_different_effective_configuration(cached_snapshot, name, key, value):
    path = cached_snapshot / name
    data = json.loads(path.read_text())
    data[key] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Unexpected effective model configuration"):
        module.audit_snapshot(cached_snapshot, CONFIG)


def test_wrong_snapshot_is_rejected(cached_snapshot):
    moved = cached_snapshot.with_name("different-revision")
    cached_snapshot.rename(moved)
    with pytest.raises(ValueError, match="Unexpected resolved snapshot"):
        module.audit_snapshot(moved, CONFIG)
