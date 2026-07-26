#!/usr/bin/env python3
"""Capture a reproducible local environment snapshot for competition runs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]


def _run(command: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    names = [
        "vllm",
        "vllm-omni",
        "vllm-ascend",
        "torch",
        "torch-npu",
        "transformers",
        "stepaudio2-minicpmo",
    ]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return versions


def _git_snapshot() -> dict[str, Any]:
    head = _run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    status = _run(["git", "status", "--short"], cwd=REPO_ROOT)
    diff = _run(["git", "diff", "--binary", "--no-ext-diff"], cwd=REPO_ROOT)
    diff_text = diff.get("stdout", "")
    return {
        "head": head,
        "status": status,
        "diff_sha256": hashlib.sha256(diff_text.encode("utf-8")).hexdigest(),
        "diff": diff,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("environment_manifest.yaml"))
    parser.add_argument("--starter-kit", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--model-manifest",
        type=Path,
        help="Optional precomputed SHA256 manifest. Model trees are not hashed implicitly.",
    )
    args = parser.parse_args()

    artifacts = {}
    for name, path in (
        ("environment_manifest", args.manifest),
        ("starter_kit", args.starter_kit),
        ("model_manifest", args.model_manifest),
    ):
        if path is not None:
            resolved = path.expanduser().resolve()
            artifacts[name] = {
                "path": str(resolved),
                "exists": resolved.is_file(),
                "sha256": _sha256(resolved) if resolved.is_file() else None,
            }

    snapshot = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "repo_root": str(REPO_ROOT),
        "platform": {
            "python": sys.version,
            "executable": sys.executable,
            "os": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "environment": {
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "ASCEND_RT_VISIBLE_DEVICES": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        },
        "packages": _package_versions(),
        "git": _git_snapshot(),
        "npu": {
            "npu_smi_path": shutil.which("npu-smi"),
            "info": _run(["npu-smi", "info"]) if shutil.which("npu-smi") else {"error": "npu-smi not found"},
        },
        "cann": {
            "ASCEND_HOME_PATH": os.environ.get("ASCEND_HOME_PATH"),
            "ASCEND_TOOLKIT_HOME": os.environ.get("ASCEND_TOOLKIT_HOME"),
            "version_files": {},
        },
        "model": {
            "path": str(args.model_path.expanduser().resolve()) if args.model_path else None,
            "revision": os.environ.get("MINICPMO_MODEL_REVISION", "UNRESOLVED"),
        },
        "artifacts": artifacts,
    }
    for path in (
        Path("/usr/local/Ascend/ascend-toolkit/latest/version.cfg"),
        Path("/usr/local/Ascend/driver/version.info"),
    ):
        if path.is_file():
            snapshot["cann"]["version_files"][str(path)] = path.read_text(errors="replace").strip()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
