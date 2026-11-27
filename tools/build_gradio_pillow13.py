#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Build the pinned Gradio 6.31 Pillow 13 source backport with Python 3.11-3.13."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

INPUTS = Path(__file__).resolve().parent / "gradio_pillow13"


def checked(path: Path, expected: str) -> Path:
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            raise RuntimeError(f"SHA256 mismatch: {path.name}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    output = parser.parse_args().output_dir.resolve()
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        raise RuntimeError("Python 3.11-3.13 required")
    manifest = json.loads((INPUTS / "manifest.json").read_text())
    for name, expected in manifest["local_inputs"].items():
        checked(INPUTS / name, expected)
    output.mkdir(parents=True, exist_ok=True)
    cache = output / ".gradio-pillow13-inputs"
    cache.mkdir(exist_ok=True)
    for name, pin in manifest["inputs"].items():
        path = cache / name
        if not path.exists():
            with urllib.request.urlopen(pin["url"], timeout=60) as source:
                path.write_bytes(source.read())
        checked(path, pin["sha256"])
    with tempfile.TemporaryDirectory(prefix="gradio-pillow13-", dir=output) as scratch_name:
        scratch = Path(scratch_name)
        with tarfile.open(cache / "gradio-6.31.0.tar.gz") as archive:
            # The archive hash is pinned above; use the standard safe filter where available.
            if hasattr(tarfile, "data_filter"):
                archive.extractall(scratch, filter="data")
            else:
                archive.extractall(scratch)
        source = scratch / "gradio-6.31.0"
        subprocess.run(
            ["git", "apply", "--unidiff-zero", str(INPUTS / "backport.patch")],
            cwd=source,
            env=dict(os.environ, GIT_CEILING_DIRECTORIES=str(scratch)),
            check=True,
        )
        # Regenerate dynamic metadata after the source patch; the official sdist omits MCP requirements.
        (source / "PKG-INFO").unlink()
        shutil.copyfile(cache / "requirements-mcp.txt", source / "requirements-mcp.txt")
        shutil.copyfile(INPUTS / "NOTICE.md", source / "gradio/PILLOW13_BACKPORT_NOTICE.md")
        with (output / "gradio-pillow13-build.log").open("w") as log:
            subprocess.run(
                [
                    "uv",
                    "build",
                    "--no-config",
                    "--wheel",
                    "--python",
                    sys.executable,
                    "--default-index",
                    "https://pypi.org/simple",
                    "--build-constraints",
                    str(INPUTS / "build-constraints.txt"),
                    "--out-dir",
                    str(scratch / "dist"),
                    str(source),
                ],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=600,
            )
        built = checked(scratch / "dist" / manifest["wheel"], manifest["expected_wheel_sha256"])
        destination = output / manifest["wheel"]
        shutil.copyfile(built, destination)
    receipt = {"wheel": destination.name, "sha256": manifest["expected_wheel_sha256"], "inputs": manifest["inputs"]}
    (output / "gradio-pillow13-provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
