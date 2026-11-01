# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Prepare a coherent telemetry solve without replacing native KV connectors."""

import argparse
import importlib.metadata as metadata
import json
import sys
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


def prepare(roots: Path, orphans: Path) -> None:
    installed: dict[str, metadata.Distribution] = {}
    search_paths = sorted({str(Path(path).resolve()) for path in sys.path})
    for distribution in metadata.distributions(path=search_paths):
        if not distribution.metadata.get("Name"):
            raise RuntimeError("Installed distribution has no Name; cannot prove orphan status")
        name = canonicalize_name(distribution.metadata["Name"])
        if name in installed:
            raise RuntimeError(f"Duplicate installed distribution {name}; cannot prove orphan status")
        installed[name] = distribution
    # Include optional/marked requirements conservatively: a retained consumer
    # always prevents pruning, even if its extra is currently inactive.
    consumers = sorted(
        name
        for name, distribution in installed.items()
        if any(canonicalize_name(Requirement(r).name) == _COMMON for r in distribution.requires or ())
    )
    # The pinned v0.32 bootstrap's 1.40 exporters use proto-common, not the
    # distinct new common distribution. Prune only that verified old-family
    # orphan; never remove common from a newer exporter family that uses it.
    old_family = all(name in installed and installed[name].version == "1.40.0" for name in _OLD_EXPORTERS)
    shared_files: list[str] = []
    unknown_records = sorted(name for name, distribution in installed.items() if not distribution.files)
    common_files: set[Path] = set()
    if _COMMON in installed:
        common = installed[_COMMON]
        common_files = {Path(common.locate_file(path)).resolve() for path in common.files or ()}
        shared_files = sorted(
            name
            for name, distribution in installed.items()
            if name != _COMMON
            and any(Path(distribution.locate_file(path)).resolve() in common_files for path in distribution.files or ())
        )
    prune = (
        _COMMON in installed
        and old_family
        and not consumers
        and bool(common_files)
        and not shared_files
        and not unknown_records
    )
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
