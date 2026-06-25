"""Discover nightly job logs under ``logs/nightly_jobs`` (shared by report + kanban prep)."""

from __future__ import annotations

from pathlib import Path

LOG_SUFFIXES = (".log", ".out", ".txt")
_LOG_GLOBS = ("*.log", "*.out", "*.txt")


def discover_job_logs(log_dir: Path) -> list[tuple[str, list[Path]]]:
    """Return ``(job_name, log_paths)`` using the same rules as ``nightly_local_log_report``."""
    if not log_dir.is_dir():
        return []
    subs = sorted(
        [p for p in log_dir.iterdir() if not p.name.startswith(".")],
        key=lambda p: p.name,
    )
    if not subs:
        return []

    merged: dict[str, list[Path]] = {}

    for d in sorted((p for p in subs if p.is_dir()), key=lambda p: p.name):
        resolved: list[Path] = []
        for pat in _LOG_GLOBS:
            resolved.extend(sorted(d.glob(pat)))
        if resolved:
            merged.setdefault(d.name, []).extend(resolved)

    for p in subs:
        if not p.is_file():
            continue
        if p.suffix.lower() not in LOG_SUFFIXES:
            continue
        merged.setdefault(p.stem, []).append(p)

    out: list[tuple[str, list[Path]]] = []
    for name in sorted(merged.keys()):
        paths = merged[name]
        if not paths:
            continue
        seen: set[str] = set()
        uniq: list[Path] = []
        for path in sorted(paths, key=lambda q: q.as_posix().lower()):
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                uniq.append(path)
        out.append((name, uniq))
    return out


def read_combined_job_logs(paths: list[Path], *, include_headers: bool = False) -> str:
    parts: list[str] = []
    for p in paths:
        if include_headers:
            parts.append(f"===== {p.name} =====\n")
        try:
            parts.append(p.read_text(encoding="utf-8-sig", errors="replace"))
        except OSError as e:
            parts.append(f"<<< read error {p}: {e} >>>\n")
        if include_headers:
            parts.append("\n")
    return "".join(parts)
