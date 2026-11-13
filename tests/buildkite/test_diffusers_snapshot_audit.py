# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reject drift in effective model configuration before comparison inference."""

import hashlib
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


@pytest.fixture
def runtime_environment(tmp_path, monkeypatch):
    source_files = []
    for package, relative in (
        ("sglang", "multimodal_gen/runtime/pipelines/diffusers_pipeline.py"),
        ("diffusers", "pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5.py"),
    ):
        source = tmp_path / package / relative
        source.parent.mkdir(parents=True)
        source.write_text(f"# {package} runtime\n")
        source_files.append(
            {"package": package, "path": relative, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        )
    versions = {"sglang": "0.5.23", "diffusers": "0.37.0"}
    environment = {
        "environment_lock": {"installed_packages": {"sglang": versions.copy()}, "runtime_source_files": source_files},
        "environment_lock_sha256": "f" * 64,
    }
    monkeypatch.setattr(module, "version", versions.__getitem__)
    monkeypatch.setattr(module, "comparison_environment", lambda _: environment)
    monkeypatch.setattr(
        module.importlib.util,
        "find_spec",
        lambda name: importlib.util.spec_from_file_location(name, tmp_path / name / "__init__.py"),
    )
    return environment, versions, tmp_path


def test_runtime_uses_this_weeks_environment(runtime_environment):
    environment, versions, _ = runtime_environment
    result = module.audit_runtime("a" * 40)
    assert result["versions"] == versions
    assert result["source_files"] == environment["environment_lock"]["runtime_source_files"]
    assert result["environment_lock_sha256"] == environment["environment_lock_sha256"]


def test_runtime_rejects_package_drift(runtime_environment):
    _, versions, _ = runtime_environment
    versions["sglang"] = "0.5.24"
    with pytest.raises(ValueError, match="Unexpected runtime versions"):
        module.audit_runtime("a" * 40)


def test_runtime_rejects_source_modified_after_image_build(runtime_environment):
    _, _, directory = runtime_environment
    (directory / "sglang/multimodal_gen/runtime/pipelines/diffusers_pipeline.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="Runtime source differs"):
        module.audit_runtime("a" * 40)
