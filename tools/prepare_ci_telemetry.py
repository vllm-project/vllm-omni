# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Prepare a coherent telemetry solve without replacing native KV connectors."""

import argparse
import csv
import importlib.metadata as metadata
import io
import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

_COMMON = "opentelemetry-exporter-otlp-common"
_OLD_EXPORTERS = (
    "opentelemetry-api",
    "opentelemetry-sdk",
    "opentelemetry-exporter-otlp",
    "opentelemetry-exporter-otlp-proto-grpc",
    "opentelemetry-exporter-otlp-proto-http",
)


def _metadata_files(distribution: metadata.Distribution) -> list[Path]:
    root = Path(distribution.locate_file(""))
    text = distribution.read_text("METADATA")
    if text is not None:
        candidates = root.glob("*.dist-info/METADATA")
    else:
        text = distribution.read_text("PKG-INFO")
        candidates = (
            candidate / "PKG-INFO" if candidate.is_dir() else candidate for candidate in root.glob("*.egg-info")
        )
    return [candidate for candidate in candidates if text is not None and candidate.read_text() == text]


def _record_files(distribution: metadata.Distribution) -> set[Path] | None:
    # Distribution.files can omit absent paths. Read every declared ownership
    # row, including absent files, before authorizing a namespace uninstall.
    record = distribution.read_text("RECORD")
    if not record or not record.strip():
        return None
    paths: set[Path] = set()
    for row in csv.reader(io.StringIO(record)):
        if len(row) != 3 or not row[0]:
            raise RuntimeError("Invalid installed RECORD; cannot prove file ownership")
        paths.add(Path(distribution.locate_file(row[0])).resolve())
    return paths


def _os_owner(path: Path, receipts: list[tuple[str, int, str, str]]) -> bool | None:
    try:
        result = subprocess.run(
            ["dpkg-query", "--search", "--", str(path)],
            capture_output=True,
            text=True,
            env=dict(os.environ, LC_ALL="C.UTF-8"),
            check=False,
            timeout=10,
        )
    except FileNotFoundError:
        receipts.append((str(path), -1, "", "dpkg-query unavailable"))
        return None
    except subprocess.TimeoutExpired:
        receipts.append((str(path), -2, "", "dpkg-query timed out"))
        return None
    receipts.append((str(path), result.returncode, result.stdout, result.stderr))
    # A partial no-match also returns 1. Never interpret a nonempty ownership
    # response as unowned, and query each exact path rather than a wildcard.
    if result.returncode in (0, 1) and result.stdout.strip():
        return True
    if result.returncode == 1 and not result.stdout.strip():
        return False
    return None


def prepare(roots: Path, orphans: Path) -> None:
    installed: dict[str, metadata.Distribution] = {}
    # uv --system installs into this interpreter's purelib/platlib. Ubuntu's
    # separately managed apt metadata may duplicate these names; it still
    # participates in reverse-consumer and OS-ownership checks below.
    managed_paths = sorted({str(Path(sysconfig.get_path(name)).resolve()) for name in ("purelib", "platlib")})
    search_paths = list(dict.fromkeys(str(Path(path).resolve()) for path in sys.path))
    observed = list(metadata.distributions(path=search_paths))
    for distribution in metadata.distributions(path=managed_paths):
        if not distribution.metadata.get("Name"):
            raise RuntimeError("Installed distribution has no Name; cannot prove orphan status")
        name = canonicalize_name(distribution.metadata["Name"])
        if name in installed:
            raise RuntimeError(f"Duplicate installed distribution {name}; cannot prove orphan status")
        installed[name] = distribution
    # Include optional/marked requirements conservatively: a retained consumer
    # always prevents pruning, even if its extra is currently inactive.
    if any(not distribution.metadata.get("Name") for distribution in observed):
        raise RuntimeError("Observed distribution has no Name; cannot prove orphan status")
    consumers = sorted(
        distribution.metadata["Name"]
        for distribution in observed
        if any(canonicalize_name(Requirement(r).name) == _COMMON for r in distribution.requires or ())
    )
    # The pinned v0.32 bootstrap's 1.40 exporters use proto-common, not the
    # distinct new common distribution. Prune only that verified old-family
    # orphan; never remove common from a newer exporter family that uses it.
    old_family = all(name in installed and installed[name].version == "1.40.0" for name in _OLD_EXPORTERS)
    shared_files: list[str] = []
    records = [(distribution, _record_files(distribution)) for distribution in observed]
    unknown_records = sorted(name for name, distribution in installed.items() if _record_files(distribution) is None)
    common_files: set[Path] = set()
    if _COMMON in installed:
        common = installed[_COMMON]
        common_files = _record_files(common) or set()
        shared_files = sorted(
            distribution.metadata["Name"]
            for distribution, files in records
            if canonicalize_name(distribution.metadata["Name"]) != _COMMON
            and files is not None
            and not files.isdisjoint(common_files)
        )
    outside_unknown = [
        distribution
        for distribution, files in records
        if str(Path(distribution.locate_file("")).resolve()) not in managed_paths and files is None
    ]
    outside_common = any(
        canonicalize_name(distribution.metadata["Name"]) == _COMMON
        and str(Path(distribution.locate_file("")).resolve()) not in managed_paths
        for distribution in observed
    )
    os_queries: list[tuple[str, int, str, str]] = []
    eligible = (
        _COMMON in installed
        and old_family
        and not consumers
        and bool(common_files)
        and not shared_files
        and not unknown_records
        and not outside_common
    )
    os_metadata_proven: bool | None = None
    os_payload_unowned: bool | None = None
    if eligible:
        os_metadata_proven = True
        for distribution in outside_unknown:
            paths = _metadata_files(distribution)
            if not paths or any(_os_owner(path, os_queries) is not True for path in paths):
                os_metadata_proven = False
        os_payload_unowned = bool(common_files) and all(_os_owner(path, os_queries) is False for path in common_files)
    prune = bool(eligible and os_metadata_proven and os_payload_unowned)
    selected = {name for name in installed if name.startswith("opentelemetry-") and not (prune and name == _COMMON)}
    requirements = sorted(installed[name].metadata["Name"] for name in selected)
    if "lmcache" in installed:
        # Preserve the actual connector binary/version and enforce its reverse
        # telemetry constraints in the same transaction as the new vLLM wheel.
        requirements.append("lmcache==" + installed["lmcache"].version)
    roots.write_text("\n".join(requirements) + "\n")
    orphans.write_text(_COMMON + "\n" if prune else "")
    print(
        json.dumps(
            {
                "installed": {name: d.version for name, d in sorted(installed.items())},
                "telemetry_roots": requirements,
                "common_consumers": consumers,
                "common_payload_overlaps": shared_files,
                "common_record_present": bool(common_files),
                "unknown_file_records": unknown_records,
                "managed_paths": managed_paths,
                "outside_unknown_metadata": [distribution.metadata["Name"] for distribution in outside_unknown],
                "outside_unknown_metadata_os_owned": os_metadata_proven,
                "common_payload_os_unowned": os_payload_unowned,
                "outside_common_distribution": outside_common,
                "os_ownership_queries": os_queries,
                "old_exporter_family": old_family,
                "prune_orphan_common": prune,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, required=True)
    parser.add_argument("--orphans", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.roots, args.orphans)
