#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Build the hash-pinned temporary Gradio Pillow13 source backport.

Python 3.11-3.13, uv >=0.11.8. No package installation or GPU import.
The wheel reproduces on Python 3.11-3.13 with uv 0.11.8. Runtime and
component qualification is narrower; see the adjacent manifest.
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from email.parser import BytesParser
from pathlib import Path

INPUTS = Path(__file__).resolve().parent / "gradio_pillow13"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked(path, expected):
    data = path.read_bytes()
    require(digest(data) == expected, f"SHA256 mismatch: {path.name}")
    return data


def fetch(name, pin, cache):
    path = cache / name
    if not path.exists():
        request = urllib.request.Request(pin["url"], headers={"User-Agent": "vllm-omni-pillow13-builder/1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        require(digest(data) == pin["sha256"], f"Downloaded SHA256 mismatch: {name}")
        path.write_bytes(data)
    checked(path, pin["sha256"])
    return path


def apply_patch(source, patch, manifest):
    """Apply exact unified hunks, rejecting fuzzy matches and all path drift."""
    lines = patch.decode().splitlines(keepends=True)
    i = 0
    changed = set()
    while i < len(lines):
        require(lines[i].startswith("--- official6.31/"), "Unexpected patch header")
        relative = lines[i].removeprefix("--- official6.31/").rstrip("\n")
        require(relative in manifest["source_files"], "Unapproved patch path")
        require(lines[i + 1] == "+++ " + manifest["version"] + "/" + relative + "\n", "Patch target mismatch")
        path = source / relative
        raw = checked(path, manifest["source_files"][relative]["before_sha256"])
        before = raw.decode().splitlines(keepends=True)
        after, cursor = [], 0
        i += 2
        while i < len(lines) and lines[i].startswith("@@ "):
            match = re.fullmatch(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*\n", lines[i])
            if match is None:
                raise RuntimeError("Invalid patch hunk")
            old_start, old_count, new_start, new_count = (int(v or "1") for v in match.groups())
            require(old_start - 1 >= cursor, "Overlapping patch hunks")
            after.extend(before[cursor : old_start - 1])
            cursor = old_start - 1
            require(new_start - 1 == len(after), "Patch target position mismatch")
            consumed_old = consumed_new = 0
            i += 1
            while consumed_old < old_count or consumed_new < new_count:
                require(i < len(lines) and lines[i][:1] in (" ", "+", "-"), "Incomplete patch hunk")
                kind, content = lines[i][0], lines[i][1:]
                if kind != "+":
                    require(cursor < len(before) and before[cursor] == content, "Patch context mismatch")
                    cursor += 1
                    consumed_old += 1
                if kind != "-":
                    after.append(content)
                    consumed_new += 1
                i += 1
            require((consumed_old, consumed_new) == (old_count, new_count), "Patch hunk count mismatch")
        after.extend(before[cursor:])
        result = "".join(after).encode()
        require(digest(result) == manifest["source_files"][relative]["after_sha256"], "Patched source SHA256 mismatch")
        path.write_bytes(result)
        changed.add(relative)
    require(changed == set(manifest["source_files"]), "Incomplete patch file set")


def function(tree):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "encode_pil_to_bytes")


def verify_wheel(original, built, manifest):
    with zipfile.ZipFile(original) as old, zipfile.ZipFile(built) as new:
        old_names, new_names = old.namelist(), new.namelist()
        require(len(new_names) == len(set(new_names)), "Duplicate wheel ZIP entries")
        a = {name: old.read(name) for name in old_names if not name.endswith("/")}
        b = {name: new.read(name) for name in new_names if not name.endswith("/")}

        def payload(names):
            return {name for name in names if ".dist-info/" not in name}

        require(payload(a) - payload(b) == set(), "Removed wheel payload")
        require(payload(b) - payload(a) == {"gradio/PILLOW13_BACKPORT_NOTICE.md"}, "Unexpected added wheel payload")
        changed = {name for name in payload(a) & payload(b) if a[name] != b[name]}
        require(changed == {"gradio/package.json", "gradio/processing_utils.py"}, "Unexpected modified wheel payload")
        old_meta = next(name for name in a if name.endswith(".dist-info/METADATA"))
        new_meta = next(name for name in b if name.endswith(".dist-info/METADATA"))
        am, bm = BytesParser().parsebytes(a[old_meta]), BytesParser().parsebytes(b[new_meta])
        require(bm["Name"] == "gradio" and bm["Version"] == manifest["version"], "Wheel identity mismatch")

        def normalize(value):
            return re.sub(r"\s+", "", value).lower()

        ar = {normalize(r) for r in am.get_all("Requires-Dist", [])}
        br = {normalize(r) for r in bm.get_all("Requires-Dist", [])}
        require(
            ar - br == {"pillow<13.0,>=8.0"} and br - ar == {"pillow<14.0,>=8.0"},
            "Unexpected dependency metadata change",
        )
        licenses = [name for name in a if "LICENSE" in name or "NOTICE" in name or "/licenses/" in name]
        for name in licenses:
            mapped = name.replace("gradio-6.31.0.dist-info/", "gradio-" + manifest["version"] + ".dist-info/")
            require(a[name] == b.get(mapped), f"License changed: {name}")
        record = next(name for name in b if name.endswith(".dist-info/RECORD"))
        rows = list(csv.reader(io.StringIO(b[record].decode())))
        names = [row[0] for row in rows]
        require(len(names) == len(set(names)) == len(b) and set(names) == set(b), "Incomplete/duplicate wheel RECORD")
        for name, encoded, size in rows:
            if name == record:
                require(encoded == size == "", "Invalid self RECORD")
            else:
                expected = base64.urlsafe_b64encode(hashlib.sha256(b[name]).digest()).rstrip(b"=").decode()
                require(encoded == "sha256=" + expected and int(size) == len(b[name]), f"RECORD mismatch: {name}")
        return {"record_entries": len(rows), "licenses_preserved": len(licenses), "modified_payload": sorted(changed)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require((3, 11) <= sys.version_info[:2] <= (3, 13), "Python3.11-3.13 required")
    manifest_bytes = (INPUTS / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    for name, pin in manifest["local_inputs"].items():
        checked(INPUTS / name, pin)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = output / ".gradio-pillow13-inputs"
    cache.mkdir(exist_ok=True)
    fetched = {name: fetch(name, pin, cache) for name, pin in manifest["inputs"].items()}
    uv_version = subprocess.check_output(["uv", "--version"], text=True).strip()
    with tempfile.TemporaryDirectory(prefix="gradio-pillow13-", dir=output) as scratch:
        scratch = Path(scratch)
        with tarfile.open(fetched["gradio-6.31.0.tar.gz"]) as archive:
            for item in archive.getmembers():
                require(item.isfile() or item.isdir(), "Unsupported source archive member")
                require((scratch / item.name).resolve().is_relative_to(scratch), "Unsafe source archive path")
            # Python3.11.0 predates tarfile's data filter. All members above
            # were already restricted to regular files/directories in-root.
            if hasattr(tarfile, "data_filter"):
                archive.extractall(scratch, filter="data")
            else:
                archive.extractall(scratch)
        source = scratch / "gradio-6.31.0"
        original_ast = ast.parse((source / "gradio/processing_utils.py").read_bytes())
        apply_patch(source, (INPUTS / "backport.patch").read_bytes(), manifest)
        patched_ast = ast.parse((source / "gradio/processing_utils.py").read_bytes())
        upstream = ast.parse(fetched["upstream-processing_utils.py"].read_bytes())
        guard = function(patched_ast).body.pop(0)
        require(
            ast.dump(guard, include_attributes=False) == ast.dump(function(upstream).body[0], include_attributes=False),
            "Guard differs from exact upstream",
        )
        require(
            ast.dump(patched_ast, include_attributes=False) == ast.dump(original_ast, include_attributes=False),
            "Additional Python AST modification",
        )
        # Regenerate dynamic metadata: the official sdist's PKG-INFO is stale
        # after patching, and requirements-mcp.txt is absent from that sdist.
        (source / "PKG-INFO").unlink()
        shutil.copyfile(fetched["requirements-mcp.txt"], source / "requirements-mcp.txt")
        shutil.copyfile(INPUTS / "NOTICE.md", source / "gradio/PILLOW13_BACKPORT_NOTICE.md")
        command = [
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
        ]
        log_path = output / "gradio-pillow13-build.log"
        with log_path.open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        built = scratch / "dist" / manifest["wheel"]
        checked(built, manifest["expected_wheel_sha256"])
        verification = verify_wheel(fetched["gradio-6.31.0-py3-none-any.whl"], built, manifest)
        destination = output / manifest["wheel"]
        shutil.copyfile(built, destination)
    receipt = {
        "wheel": destination.name,
        "sha256": digest(destination.read_bytes()),
        "bytes": destination.stat().st_size,
        "manifest_sha256": digest(manifest_bytes),
        "python": sys.version,
        "uv": uv_version,
        "inputs": manifest["inputs"],
        "local_inputs": manifest["local_inputs"],
        "verification": verification,
        "qualification": manifest["qualification"],
        "package_install_performed": False,
    }
    (output / "gradio-pillow13-provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
