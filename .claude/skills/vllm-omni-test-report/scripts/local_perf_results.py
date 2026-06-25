"""Scan local DFX perf JSON under ``tests/dfx/perf/results``."""

from __future__ import annotations

import json
import re
from pathlib import Path

PERF_JSON_GLOBS = (
    "result_test_*.json",
    "diffusion_result_*.json",
    "benchmark_results_*.json",
)

_LOCAL_PERF_STEM_PREFIXES = (
    "diffusion_result_",
    "benchmark_results_",
    "result_test_",
    "result_",
)
_TIMESTAMP_SUFFIX_RE = re.compile(r"_\d{8}[-_]\d{6}$")
_JSON_TEST_FIELDS = (
    "test_name",
    "test",
    "benchmark_name",
    "name",
    "job_name",
    "source_file",
)


def local_perf_result_files(result_dir: Path) -> list[Path]:
    if not result_dir.is_dir():
        return []
    paths: dict[Path, None] = {}
    for pattern in PERF_JSON_GLOBS:
        for path in result_dir.rglob(pattern):
            if path.is_file():
                paths[path] = None
    return sorted(paths)


def _pick_latest_local_perf_result_dir(result_root: Path) -> Path | None:
    if not result_root.is_dir():
        return None
    dirs = [path for path in result_root.iterdir() if path.is_dir()]
    if not dirs:
        return None

    def _created_at(path: Path) -> float:
        stat = path.stat()
        return float(getattr(stat, "st_birthtime", stat.st_mtime))

    return max(dirs, key=_created_at)


def resolve_local_perf_result_dir(result_root: Path) -> Path | None:
    root = result_root.resolve()
    if not root.is_dir():
        return None
    if local_perf_result_files(root):
        return root
    sub = _pick_latest_local_perf_result_dir(root)
    if sub is not None and local_perf_result_files(sub):
        return sub
    return sub


def normalize_test_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (value or "").lower()).strip("_")


def test_key_from_perf_filename(filename: str) -> str:
    stem = Path(filename).stem
    for prefix in _LOCAL_PERF_STEM_PREFIXES:
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    return _TIMESTAMP_SUFFIX_RE.sub("", stem)


def test_keys_from_perf_file(path: Path) -> set[str]:
    keys: set[str] = set()
    base = test_key_from_perf_filename(path.name)
    if base:
        keys.add(base)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return keys
    if not isinstance(payload, dict):
        return keys
    for field in _JSON_TEST_FIELDS:
        val = payload.get(field)
        if val:
            text = str(val).strip()
            if text:
                keys.add(text)
                keys.add(Path(text).stem)
    return {k for k in keys if k}


def collect_local_perf_test_keys(result_dir: Path | None) -> frozenset[str]:
    if result_dir is None or not result_dir.is_dir():
        return frozenset()
    keys: set[str] = set()
    for path in local_perf_result_files(result_dir):
        keys.update(test_keys_from_perf_file(path))
    return frozenset(keys)


def perf_row_matches_local_test(row: dict, local_keys: frozenset[str]) -> bool:
    if not local_keys:
        return False
    norm_keys = {normalize_test_key(k) for k in local_keys if k}
    norm_keys = {k for k in norm_keys if k}
    candidates: set[str] = set()
    for field in ("test_name", "config_key", "model", "model_type"):
        raw = str(row.get(field) or "").strip()
        if not raw:
            continue
        candidates.add(normalize_test_key(raw))
        candidates.add(normalize_test_key(Path(raw).stem))
    candidates = {c for c in candidates if c}
    for cand in candidates:
        for key in norm_keys:
            if cand == key or key in cand or cand in key:
                return True
    return False
