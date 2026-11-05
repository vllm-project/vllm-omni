# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Read-only attestation of an explicitly reviewed editable-source snapshot.

The supplier's Dockerfile remains modified. Matching that exact snapshot never
means the checkout is clean, or that a native library/kernel was executed.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import stat
import subprocess
import urllib.parse
from pathlib import Path

from benchmarks.lingbot_world.regression import RegressionError

LOCK_PATH = Path(__file__).with_name("runtime_sources.lock.json")


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise RegressionError("Invalid runtime source snapshot object")
    return value


def _canonical(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise RegressionError("Invalid runtime source attestation") from error


def _lock(expected_sha256: str) -> dict[str, object]:
    try:
        data = LOCK_PATH.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_sha256:
            raise RegressionError("Runtime source snapshot lock digest changed")
        result = _object(json.loads(data))
    except (OSError, ValueError) as error:
        raise RegressionError("Cannot verify runtime source snapshot lock") from error
    if type(result.get("schema_version")) is not int or result["schema_version"] != 1:
        raise RegressionError("Unsupported runtime source snapshot schema")
    return result


def _expected_attestations(lock: dict[str, object], digest: str) -> dict[str, object]:
    return {"schema_version": 1, "source_snapshot_sha256": digest, "packages": _object(lock["packages"])}


def verify_source_attestations(observed: object, *, expected_sha256: str) -> None:
    """Validate reported provenance using only the canonical lock, never Git/runtime imports."""
    expected = _expected_attestations(_lock(expected_sha256), expected_sha256)
    # JSON comparison keeps booleans distinct from integers and rejects extra fields.
    if _canonical(observed) != _canonical(expected):
        raise RegressionError("Reported runtime source attestation differs from the reviewed snapshot")


def _git(source: Path, *arguments: str, preserve_index_flags: bool = False) -> bytes:
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_OPTIONAL_LOCKS="0", GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null")
    options = [] if preserve_index_flags else ["-c", "core.fsmonitor="]
    try:
        return subprocess.check_output(
            [
                "git",
                "--no-optional-locks",
                "-c",
                f"safe.directory={source}",
                "-c",
                "core.filemode=true",
                *options,
                "-c",
                "core.untrackedCache=false",
                "-C",
                str(source),
                *arguments,
            ],
            env=env,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RegressionError("Cannot inspect editable runtime source") from error


def _flags(source: Path) -> None:
    # These are different bits: -v exposes assume-unchanged and S skip-worktree;
    # -f separately exposes fsmonitor-valid. No per-tracked-file Git processes.
    for flag in ("-v", "-f"):
        # Inspect persisted bits before disabling fsmonitor for status: newer
        # Git can otherwise clear its valid bits while loading the index.
        rows = _git(source, "ls-files", flag, "-z", preserve_index_flags=True).split(b"\0")
        if not rows[-1:] == [b""] or any(not row.startswith(b"H ") for row in rows[:-1]):
            raise RegressionError("Runtime source index hides tracked files")


def _status(source: Path) -> str:
    return _git(source, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignore-submodules=none").decode(
        "utf-8"
    )


def _editable_source(package: str) -> Path:
    try:
        provenance = importlib.metadata.distribution(package).read_text("direct_url.json")
        record = _object(json.loads(provenance or "{}"))
        url = urllib.parse.urlparse(str(record.get("url", "")))
        if (
            url.scheme != "file"
            or url.netloc not in ("", "localhost")
            or _object(record.get("dir_info", {})).get("editable") is not True
        ):
            raise ValueError("No editable source provenance")
        return Path(urllib.parse.unquote(url.path)).resolve()
    except (OSError, ValueError, importlib.metadata.PackageNotFoundError) as error:
        raise RegressionError(f"Cannot attest {package} editable runtime source") from error


def _file(source: Path, relative: str) -> dict[str, object]:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.parts != (relative,):
        raise RegressionError("Unsafe runtime source snapshot file")
    target = source / path
    try:
        before = target.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise RegressionError("Runtime source snapshot file is not regular")
        fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            data = stream.read()
        after = target.lstat()
    except OSError as error:
        raise RegressionError("Cannot read runtime source snapshot file") from error
    if (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (after.st_dev, after.st_ino, after.st_mode, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise RegressionError("Runtime source snapshot file changed during inspection")
    head_row = _git(source, "ls-tree", "-z", "HEAD", "--", relative).split(b"\t", 1)[0].split()
    index_row = _git(source, "ls-files", "--stage", "-z", "--", relative).split(b"\t", 1)[0].split()
    if len(head_row) != 3 or len(index_row) != 3:
        raise RegressionError("Missing runtime source snapshot index/HEAD file")
    head = _git(source, "cat-file", "blob", f"HEAD:{relative}")
    return {
        "head_mode": head_row[0].decode(),
        "head_blob": head_row[2].decode(),
        "head_bytes": len(head),
        "head_sha256": hashlib.sha256(head).hexdigest(),
        "index_mode": index_row[0].decode(),
        "index_blob": index_row[1].decode(),
        "index_stage": index_row[2].decode(),
        "worktree": {
            "type": "regular",
            "bytes": len(data),
            "mode": oct(stat.S_IMODE(before.st_mode)),
            "sha256": hashlib.sha256(data).hexdigest(),
            "git_blob": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
        },
    }


def _inspect(source: Path, expected: dict[str, object]) -> dict[str, object]:
    _flags(source)
    head = _git(source, "rev-parse", "HEAD").decode().strip()
    tree = _git(source, "rev-parse", "HEAD^{tree}").decode().strip()
    index = _git(source, "ls-files", "--stage", "-z")
    status_before = _status(source)
    files = {name: _file(source, name) for name in _object(expected["files"])}
    gitlinks = {}
    for name in _object(expected["gitlinks"]):
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise RegressionError("Unsafe runtime source snapshot gitlink")
        nested = source / path
        _flags(nested)
        nested_status = _status(nested)
        gitlinks[name] = {
            "head": _git(nested, "rev-parse", "HEAD").decode().strip(),
            "status_porcelain_z": nested_status,
            "clean": nested_status == "",
            "index_flags_clean": True,
        }
        _flags(nested)
        if nested_status != _status(nested):
            raise RegressionError("Runtime source gitlink changed during inspection")
    _flags(source)
    if (
        head != _git(source, "rev-parse", "HEAD").decode().strip()
        or index != _git(source, "ls-files", "--stage", "-z")
        or status_before != _status(source)
    ):
        raise RegressionError("Runtime source snapshot changed during inspection")
    return {
        "head": head,
        "head_tree": tree,
        "index_sha256": hashlib.sha256(index).hexdigest(),
        "tracked_file_count": len(index.split(b"\0")) - 1,
        "status_porcelain_z": status_before,
        "files": files,
        "gitlinks": gitlinks,
        "clean": status_before == "",
        "exact_snapshot_verified": True,
        "index_flags_clean": True,
    }


def attest_runtime_sources(expected: dict[str, str], *, expected_sha256: str) -> dict[str, object]:
    """Read actual editable checkouts and reject every state outside the reviewed lock."""
    lock = _lock(expected_sha256)
    packages = _object(lock["packages"])
    if {name: _object(value)["head"] for name, value in packages.items()} != expected:
        raise RegressionError("Runtime source snapshot disagrees with pinned revisions")
    observed: dict[str, object] = {
        "schema_version": 1,
        "source_snapshot_sha256": expected_sha256,
        "packages": {name: _inspect(_editable_source(name), _object(value)) for name, value in packages.items()},
    }
    verify_source_attestations(observed, expected_sha256=expected_sha256)
    return observed
