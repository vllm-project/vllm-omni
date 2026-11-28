# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Publish the benchmark source snapshot without a torn read."""

import hashlib
import json
import os
import shutil
from pathlib import Path


def publish_source_snapshot(src: Path, dst: Path) -> None:
    """Install ``src`` at ``dst`` without a reader-visible partial file.

    ``shutil.copy2`` opens an existing destination with truncation before the
    bytes are copied. Every rank preflights ``--out`` at the same time, so a
    slower rank can read that short file and report a source mismatch for a
    snapshot that is actually current. Creating the file in place has the same
    window: ``exists()`` becomes true as soon as the empty destination is
    opened. Copying to a sibling temp file and replacing it into place leaves
    readers on either the previous complete file or the new complete file.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    temporary = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(src, temporary)
        os.replace(temporary, dst)
    finally:
        temporary.unlink(missing_ok=True)


def record_source_snapshot(parser, root: Path, out: Path, files: list[Path], rank: int) -> None:
    """Reject a stale snapshot on every rank, then let rank 0 publish it."""
    for file in files:
        relative = file.relative_to(root)
        destination = out / "source_snapshot" / relative
        if destination.exists() and destination.read_bytes() != file.read_bytes():
            parser.error(f"result directory contains different source: {relative}")
    if rank != 0:
        return
    manifest = {}
    for file in files:
        relative = file.relative_to(root)
        publish_source_snapshot(file, out / "source_snapshot" / relative)
        manifest[str(relative)] = hashlib.sha256(file.read_bytes()).hexdigest()
    (out / "source_snapshot" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
