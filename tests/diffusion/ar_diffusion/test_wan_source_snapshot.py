# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Rank-0 source snapshots must not be visible until the copy is complete."""

import argparse
import builtins
import hashlib
import json
import os
import threading
from pathlib import Path

import pytest

from benchmarks.wan import source_snapshot

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(prog="benchmark_noisy_pp")


def _layout(tmp_path: Path, payload: bytes, *, existing: bytes | None) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "repo"
    src = root / "benchmarks" / "wan" / "model.py"
    out = tmp_path / "out"
    dst = out / "source_snapshot" / src.relative_to(root)
    src.parent.mkdir(parents=True)
    src.write_bytes(payload)
    if existing is not None:
        dst.parent.mkdir(parents=True)
        dst.write_bytes(existing)
    return root, out, src, dst


def _publish_while_copy_is_open(monkeypatch, root: Path, out: Path, src: Path, check) -> None:
    """Run rank 0 until its copy opens a write, then let ``check`` observe it."""
    started = threading.Event()
    release = threading.Event()
    opened: dict[str, Path] = {}
    real_open = builtins.open

    def wrapped(file, mode="r", *args, **kwargs):
        handle = real_open(file, mode, *args, **kwargs)
        text_mode = mode if isinstance(mode, str) else ""
        writing = isinstance(file, (str, Path)) and "w" in text_mode and "b" in text_mode
        if writing and "path" not in opened:
            opened["path"] = Path(file)
            started.set()
            if not release.wait(timeout=10):
                raise TimeoutError("blocked snapshot publish was not released")
        return handle

    monkeypatch.setattr(builtins, "open", wrapped)
    errors: list[BaseException] = []

    def rank0() -> None:
        try:
            source_snapshot.record_source_snapshot(_parser(), root, out, [src], 0)
        except (SystemExit, Exception) as exc:
            errors.append(exc)

    thread = threading.Thread(target=rank0)
    thread.start()
    try:
        assert started.wait(5), "rank 0 did not open the snapshot copy"
        check(opened["path"])
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert errors == []


def test_reused_output_stays_complete_while_rank0_rewrites(tmp_path, monkeypatch):
    payload = b"matching-source-snapshot"
    root, out, src, dst = _layout(tmp_path, payload, existing=payload)

    def check(opened: Path) -> None:
        assert opened != dst
        assert dst.read_bytes() == payload
        source_snapshot.record_source_snapshot(_parser(), root, out, [src], 1)

    _publish_while_copy_is_open(monkeypatch, root, out, src, check)
    assert dst.read_bytes() == payload
    assert not list(dst.parent.glob(".*"))


def test_first_snapshot_stays_invisible_until_replace(tmp_path, monkeypatch):
    payload = b"first-source-snapshot"
    root, out, src, dst = _layout(tmp_path, payload, existing=None)

    def check(opened: Path) -> None:
        assert opened != dst
        assert not dst.exists()
        source_snapshot.record_source_snapshot(_parser(), root, out, [src], 1)

    _publish_while_copy_is_open(monkeypatch, root, out, src, check)
    assert dst.read_bytes() == payload
    assert not list(dst.parent.glob(".*"))
    manifest = (out / "source_snapshot" / "manifest.json").read_text()
    assert manifest == json.dumps({"benchmarks/wan/model.py": hashlib.sha256(payload).hexdigest()}, indent=2) + "\n"


def test_stale_snapshot_fails_every_rank_without_rewrite(tmp_path, capsys):
    payload = b"current-source"
    root, out, src, dst = _layout(tmp_path, payload, existing=b"previous-run")
    other = root / "vllm_omni" / "kept.py"
    other.parent.mkdir()
    other.write_bytes(b"unchanged")
    kept = out / "source_snapshot" / other.relative_to(root)
    kept.parent.mkdir(parents=True)
    kept.write_bytes(b"unchanged")
    # A matching file is checked before the stale one. Publishing must wait
    # until every file matches, so neither destination is rewritten.
    os.utime(dst, ns=(1_000_000_000, 1_000_000_000))
    os.utime(kept, ns=(1_000_000_000, 1_000_000_000))

    for rank in (0, 1):
        with pytest.raises(SystemExit) as caught:
            source_snapshot.record_source_snapshot(_parser(), root, out, [other, src], rank)
        assert caught.value.code == 2
        assert "result directory contains different source: benchmarks/wan/model.py" in capsys.readouterr().err

    assert dst.read_bytes() == b"previous-run"
    assert dst.stat().st_mtime_ns == 1_000_000_000
    assert kept.stat().st_mtime_ns == 1_000_000_000
    assert not (out / "source_snapshot" / "manifest.json").exists()
    recorded = [name for _dir, _dirs, names in os.walk(out) for name in names]
    assert not any(name.endswith(".tmp") for name in recorded)


def test_non_zero_rank_does_not_create_or_refresh_snapshot(tmp_path):
    payload = b"existing-source"
    root, out, src, dst = _layout(tmp_path, payload, existing=payload)
    inode = dst.stat().st_ino
    source_snapshot.record_source_snapshot(_parser(), root, out, [src], 1)
    assert dst.stat().st_ino == inode
    assert not (out / "source_snapshot" / "manifest.json").exists()

    missing_root, missing_out, missing_src, missing_dst = _layout(tmp_path / "fresh", payload, existing=None)
    source_snapshot.record_source_snapshot(_parser(), missing_root, missing_out, [missing_src], 1)
    assert not missing_dst.exists()


def test_failed_replace_removes_the_temporary_file(tmp_path, monkeypatch):
    payload = b"source-to-publish"
    root, out, src, dst = _layout(tmp_path, payload, existing=None)

    def fail_replace(src_name, dst_name):
        raise OSError("replace failed")

    monkeypatch.setattr(source_snapshot.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        source_snapshot.publish_source_snapshot(src, dst)
    assert not dst.exists()
    assert list(dst.parent.glob(".*")) == []
