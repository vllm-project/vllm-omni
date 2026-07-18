"""Scan local DFX perf JSON under ``logs/nightly_jobs`` (recursive)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def default_nightly_log_dir(repo_root: Path) -> Path:
    return repo_root / "logs" / "nightly_jobs"


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
_TIMESTAMP_SUFFIX_RE = re.compile(r"_(\d{8})[-_]\d{6}$")
# Verified on hk01dgx012 / omni_wy_24g (2026-07-13): the cluster uses
# ``nightly_stability_jobs_*`` (NOT ``nightly_jobs_stability_*``) for long-stability
# runs, and the merged ``logs/nightly_jobs/`` contains the union of all three
# families. The source-marker file (.nightly_jobs_source) records the original
# per-run dir basenames so we can match the right glob back.
_NIGHTLY_JOBS_RUN_DIR_RE = re.compile(
    r"^nightly_(?:jobs_(?:local)?|stability_jobs)_(\d{8})(?:[-_]\d{6})?$",
    re.I,
)
NIGHTLY_JOBS_SOURCE_MARKER = ".nightly_jobs_source"
KANBAN_PERF_SOURCE_DIRNAME = ".kanban_perf_source"
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


def _yyyymmdd_from_nightly_jobs_basename(name: str) -> str | None:
    match = _NIGHTLY_JOBS_RUN_DIR_RE.match((name or "").strip())
    return match.group(1) if match else None


def _read_nightly_jobs_source_marker(marker_path: Path) -> str | None:
    if not marker_path.is_file():
        return None
    try:
        text = marker_path.read_text(encoding="utf-8")
    except OSError:
        return None
    dates: list[str] = []
    for line in text.splitlines():
        day = _yyyymmdd_from_nightly_jobs_basename(line.strip())
        if day:
            dates.append(day)
    return max(dates) if dates else None


def _yyyymmdd_from_perf_filenames(result_dir: Path) -> str | None:
    dates: list[str] = []
    for path in local_perf_result_files(result_dir):
        match = _TIMESTAMP_SUFFIX_RE.search(path.stem)
        if match:
            dates.append(match.group(1))
    return max(dates) if dates else None


def kanban_perf_source_dir(repo_root: Path) -> Path:
    return repo_root / "logs" / KANBAN_PERF_SOURCE_DIRNAME


def is_nightly_jobs_local_source_name(name: str) -> bool:
    return (name or "").strip().lower().startswith("nightly_jobs_local_")


def nightly_jobs_source_names(log_dir: Path) -> list[str]:
    log_dir = log_dir.resolve()
    for marker in (log_dir / NIGHTLY_JOBS_SOURCE_MARKER, log_dir.parent / NIGHTLY_JOBS_SOURCE_MARKER):
        if not marker.is_file():
            continue
        try:
            names = [line.strip() for line in marker.read_text(encoding="utf-8").splitlines() if line.strip()]
        except OSError:
            continue
        if names:
            return names
    if is_nightly_jobs_local_source_name(log_dir.name):
        return [log_dir.name]
    return []


def infer_kanban_manual_date_yyyymmdd(*, repo_root: Path, log_dir: Path) -> str | None:
    """``YYYYMMDD`` for ``manual_*`` — from ``nightly_jobs_local_*`` sources only."""
    dates: list[str] = []
    for name in nightly_jobs_source_names(log_dir):
        if not is_nightly_jobs_local_source_name(name):
            continue
        day = _yyyymmdd_from_nightly_jobs_basename(name)
        if day:
            dates.append(day)
    if dates:
        return max(dates)

    staged = kanban_perf_source_dir(repo_root)
    if staged.is_dir():
        day = _yyyymmdd_from_perf_filenames(staged)
        if day:
            return day
    return None


def resolve_kanban_manual_log_dir(*, repo_root: Path, log_dir: Path) -> Path | None:
    """Perf/log tree for kanban ``manual_*`` — ``nightly_jobs_local_*`` only, not general ``nightly_jobs_*``."""
    staged = kanban_perf_source_dir(repo_root).resolve()
    if staged.is_dir():
        resolved = resolve_local_perf_result_dir(staged)
        if resolved is not None and local_perf_result_files(resolved):
            return resolved

    names = nightly_jobs_source_names(log_dir)
    if names and all(is_nightly_jobs_local_source_name(n) for n in names):
        resolved = resolve_local_perf_result_dir(log_dir)
        if resolved is not None and local_perf_result_files(resolved):
            return resolved
    return None


def infer_nightly_run_date_yyyymmdd(log_dir: Path) -> str | None:
    """Infer ``YYYYMMDD`` from synced run marker (any source).

    Prefer ``infer_kanban_manual_date_yyyymmdd`` for kanban manual dirs.
    """
    log_dir = log_dir.resolve()
    for marker in (
        log_dir / NIGHTLY_JOBS_SOURCE_MARKER,
        log_dir.parent / NIGHTLY_JOBS_SOURCE_MARKER,
    ):
        day = _read_nightly_jobs_source_marker(marker)
        if day:
            return day

    day = _yyyymmdd_from_nightly_jobs_basename(log_dir.name)
    if day:
        return day

    resolved = resolve_local_perf_result_dir(log_dir)
    if resolved is not None:
        day = _yyyymmdd_from_perf_filenames(resolved)
        if day:
            return day
    return _yyyymmdd_from_perf_filenames(log_dir)


def _add_test_key_variants(keys: set[str], raw: str) -> None:
    text = str(raw).strip()
    if not text:
        return
    keys.add(text)
    keys.add(Path(text).stem)


def _test_keys_from_json_obj(obj: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for field in _JSON_TEST_FIELDS:
        val = obj.get(field)
        if val:
            _add_test_key_variants(keys, str(val))
    return keys


def _test_keys_from_json_payload(payload: Any) -> set[str]:
    """Collect test keys from dict payloads and diffusion aggregated JSON arrays."""
    keys: set[str] = set()
    if isinstance(payload, dict):
        keys.update(_test_keys_from_json_obj(payload))
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                keys.update(_test_keys_from_json_obj(item))
    return keys


def test_keys_from_perf_file(path: Path) -> set[str]:
    keys: set[str] = set()
    base = test_key_from_perf_filename(path.name)
    if base:
        keys.add(base)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {k for k in keys if k}
    keys.update(_test_keys_from_json_payload(payload))
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
