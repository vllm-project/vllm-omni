#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validate .buildkite YAML against the official Buildkite schema.

Test pipelines (test-ready.yml, etc.) use Omni-only keys such as
``mirror_hardwares`` and ``source_file_dependencies``. Those are expanded or
stripped the same way ``upload_pipeline.py`` does before upload, then the
rendered YAML is checked with check-jsonschema.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".buildkite" / "common" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from upload_pipeline import _render_test_pipeline  # noqa: E402

SKIP_FILES = {
    ".buildkite/common/ci_mirror_hardwares.yml",
    ".buildkite/npu/pipeline-npu-a3.yml",
}

# AMD pipelines are uploaded by a different path and use list-valued
# mirror_hardwares plus agent_pool/grade, which CUDA's expander rejects.
AMD_STRIP_KEYS = frozenset({"mirror_hardwares", "agent_pool", "grade", "source_file_dependencies"})


def _normalize(path: str) -> str:
    return path.replace("\\", "/")


def _strip_keys(obj: Any, keys: frozenset[str]) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_keys(v, keys) for k, v in obj.items() if k not in keys}
    if isinstance(obj, list):
        return [_strip_keys(item, keys) for item in obj]
    return obj


def _prepare_pipeline(path: Path) -> dict[str, Any]:
    rel = _normalize(str(path.relative_to(ROOT) if path.is_absolute() else path))
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"{rel}: expected a mapping (pipeline YAML), got {type(doc).__name__}")

    if "/amd/" in f"/{rel}":
        return _strip_keys(doc, AMD_STRIP_KEYS)

    return _render_test_pipeline(doc, changed_files=None)


def _validate(path: Path, doc: dict[str, Any]) -> int:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yml",
        encoding="utf-8",
        delete=False,
    ) as handle:
        yaml.safe_dump(doc, handle, sort_keys=False)
        tmp = Path(handle.name)

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "check_jsonschema",
                "--builtin-schema",
                "vendor.buildkite",
                str(tmp),
            ],
            capture_output=True,
            text=True,
        )
    finally:
        tmp.unlink(missing_ok=True)

    if result.returncode == 0:
        return 0

    rel = _normalize(str(path))
    sys.stderr.write(f"{rel}: Buildkite schema validation failed\n")
    sys.stderr.write(result.stdout or "")
    sys.stderr.write(result.stderr or "")
    return 1


def main(argv: list[str]) -> int:
    rc = 0
    for raw in argv:
        path = Path(raw)
        rel = _normalize(str(path))
        if rel in SKIP_FILES:
            continue
        if not path.is_file():
            path = ROOT / path
        try:
            doc = _prepare_pipeline(path)
        except Exception as exc:  # noqa: BLE001 - surface expander errors as hook failures
            sys.stderr.write(f"{rel}: {exc}\n")
            rc = 1
            continue
        rc |= _validate(path, doc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
