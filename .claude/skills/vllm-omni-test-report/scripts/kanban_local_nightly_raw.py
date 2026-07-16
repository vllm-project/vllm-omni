"""Helpers for ``data/local_nightly_raw/manual_*`` sync and archive push."""

from __future__ import annotations

from pathlib import Path

LOCAL_NIGHTLY_RAW = Path("data") / "local_nightly_raw"
LAST_MANUAL_MARKER = LOCAL_NIGHTLY_RAW / ".last_manual_dir"
MANUAL_DIR_PREFIX = "manual_"
HUNYUAN_NIGHTLY_SOURCE_LOG = "local_pytest_hunyuan_image.log"
HUNYUAN_MANUAL_DEST_LOG = "test_hunyuan_image3.log"
# Cluster/local sync may use either pytest-style or kanban dest basename.
HUNYUAN_NIGHTLY_SOURCE_LOG_CANDIDATES = (
    HUNYUAN_NIGHTLY_SOURCE_LOG,
    HUNYUAN_MANUAL_DEST_LOG,
)


def resolve_hunyuan_nightly_source_log(log_dir: Path) -> Path | None:
    """Return the first existing Hunyuan Image nightly job log under ``log_dir``."""
    root = log_dir.resolve()
    for name in HUNYUAN_NIGHTLY_SOURCE_LOG_CANDIDATES:
        path = root / name
        if path.is_file():
            return path
    return None


def local_nightly_raw_root(kanban_repo: Path) -> Path:
    return kanban_repo.resolve() / LOCAL_NIGHTLY_RAW


def _is_manual_dir(path: Path, *, kanban_repo: Path) -> bool:
    try:
        rel = path.resolve().relative_to(local_nightly_raw_root(kanban_repo))
    except ValueError:
        return False
    return rel.parts == (path.name,) and path.name.startswith(MANUAL_DIR_PREFIX)


def clear_last_manual_marker(kanban_repo: Path) -> None:
    marker = kanban_repo.resolve() / LAST_MANUAL_MARKER
    if marker.is_file():
        marker.unlink()


def write_last_manual_marker(kanban_repo: Path, manual_dir: Path) -> None:
    """Record the latest ``manual_*`` dir for archive push (marker is not committed)."""
    kanban_repo = kanban_repo.resolve()
    manual_dir = manual_dir.resolve()
    if not _is_manual_dir(manual_dir, kanban_repo=kanban_repo):
        raise ValueError(f"Not a manual_* dir under {LOCAL_NIGHTLY_RAW}: {manual_dir}")
    marker = kanban_repo / LAST_MANUAL_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    rel_name = manual_dir.relative_to(local_nightly_raw_root(kanban_repo)).as_posix()
    marker.write_text(f"{rel_name}\n", encoding="utf-8")


def read_last_manual_marker(kanban_repo: Path) -> Path | None:
    marker = kanban_repo.resolve() / LAST_MANUAL_MARKER
    if not marker.is_file():
        return None
    name = marker.read_text(encoding="utf-8").strip()
    if not name:
        return None
    candidate = local_nightly_raw_root(kanban_repo) / name
    return candidate if candidate.is_dir() else None


def resolve_manual_dir_for_archive(
    kanban_repo: Path,
    explicit: Path | None = None,
    *,
    auto: bool = True,
) -> Path | None:
    """Resolve ``manual_*`` to include in kanban archive push (marker or explicit only)."""
    kanban_repo = kanban_repo.resolve()
    if explicit is not None:
        path = explicit.expanduser()
        if not path.is_absolute():
            path = (kanban_repo / path).resolve()
        else:
            path = path.resolve()
        if not _is_manual_dir(path, kanban_repo=kanban_repo):
            raise ValueError(f"--local-nightly-manual-dir must be under {LOCAL_NIGHTLY_RAW}/manual_*: {path}")
        return path

    if not auto:
        return None

    return read_last_manual_marker(kanban_repo)


def manual_dir_rel(manual_dir: Path, kanban_repo: Path) -> str:
    return manual_dir.resolve().relative_to(kanban_repo.resolve()).as_posix()
