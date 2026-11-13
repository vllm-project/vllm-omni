# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audit cached Diffusers configuration without downloading model weights."""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
from importlib.metadata import version
from pathlib import Path

from tests.dfx.perf.scripts.experiment_metadata import comparison_environment


def audit_snapshot(snapshot: Path, config_path: Path) -> dict:
    (config,) = json.loads(config_path.read_text())
    server = config["server_params"]
    args = server["serve_args"]
    expected = config["model_audit"]
    snapshot = snapshot.resolve()
    if snapshot.name != args["revision"] or snapshot.parent.name != "snapshots":
        raise ValueError(f"Unexpected resolved snapshot: {snapshot}")
    if args["backend"] != "diffusers" or args["dit-precision"] != expected["precision"]["load_dtype"]:
        raise ValueError("Diffusers loading precision does not match the audited configuration")

    configs = []
    for name, values in expected["config_values"].items():
        raw = (snapshot / name).read_bytes()
        actual = json.loads(raw)
        if any(actual.get(key) != value for key, value in values.items()):
            raise ValueError(f"Unexpected effective model configuration in {name}: {actual}")
        configs.append({"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "values": actual})
    return {
        "schema_version": 1,
        "purpose": "comparison_effective_config",
        "runner_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "build_id": os.environ.get("BUILDKITE_BUILD_ID"),
        "job_id": os.environ.get("BUILDKITE_JOB_ID"),
        "test_name": config["test_name"],
        "model": server["model"],
        "revision": args["revision"],
        "snapshot_path": str(snapshot),
        "backend": "diffusers",
        "precision": expected["precision"],
        "test_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "configs": configs,
        "status": "passed",
    }


def audit_runtime(runner_commit: str) -> dict:
    environment = comparison_environment(runner_commit)
    lock = environment.get("environment_lock")
    versions = {name: version(name) for name in ("sglang", "diffusers")}
    expected_versions = (
        {name: lock["installed_packages"]["sglang"][name] for name in versions}
        if lock
        else {"sglang": "0.5.22", "diffusers": "0.37.0"}
    )
    if versions != expected_versions:
        raise ValueError(f"Unexpected runtime versions: {versions}")
    sources = {
        "sglang": "multimodal_gen/runtime/pipelines/diffusers_pipeline.py",
        "diffusers": "pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5.py",
    }
    source_files = []
    for package, relative in sources.items():
        spec = importlib.util.find_spec(package)
        assert spec is not None and spec.origin is not None
        source = Path(spec.origin).parent / relative
        source_files.append(
            {"package": package, "path": relative, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        )
    result = {"versions": versions, "source_files": source_files}
    if lock:
        expected_sources = {(item["package"], item["path"]): item["sha256"] for item in lock["runtime_source_files"]}
        if any(expected_sources.get((item["package"], item["path"])) != item["sha256"] for item in source_files):
            raise ValueError("Runtime source differs from the frozen comparison image")
        result["environment_lock_sha256"] = environment["environment_lock_sha256"]
    return result


def main() -> None:
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (config,) = json.loads(args.config.read_text())
    server = config["server_params"]
    snapshot = snapshot_download(
        repo_id=server["model"], revision=server["serve_args"]["revision"], local_files_only=True
    )
    result = audit_snapshot(Path(snapshot), args.config)
    result.update(audit_runtime(result["runner_commit"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Audited cached {server['model']} snapshot {result['revision']} with {len(result['configs'])} config files")


if __name__ == "__main__":
    main()
