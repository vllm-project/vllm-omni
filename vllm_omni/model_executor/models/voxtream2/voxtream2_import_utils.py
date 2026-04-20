# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamic import utilities for the Voxtream package.

Supports three discovery modes, first match wins:
1. ``VLLM_OMNI_VOXTREAM_CODE_PATH`` env var pointing at a source tree.
2. Local/sibling ``voxtream`` source trees used during development.
3. pip-installed ``voxtream`` package.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Iterable
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)


def _iter_voxtream_src_candidates() -> list[Path]:
    candidates: list[Path] = []
    for env_key in (
        "VLLM_OMNI_VOXTREAM_CODE_PATH",
        "VOXTREAM2_ROOT",
        "VOXTREAM_ROOT",
    ):
        env_path = os.environ.get(env_key)
        if env_path:
            candidates.append(Path(env_path).expanduser())

    repo_root = Path(__file__).resolve().parents[4]
    candidates.extend(
        [
            repo_root / "voxtream",
            repo_root.parent / "voxtream",
            repo_root.parent / "Voxtream",
        ]
    )

    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path.expanduser())
        if key not in seen:
            seen.add(key)
            unique.append(path.expanduser())
    return unique


def _path_to_prepend(candidate: Path) -> Path | None:
    """Return the sys.path entry that exposes ``voxtream``."""
    if (candidate / "voxtream" / "__init__.py").is_file():
        return candidate
    if candidate.name == "voxtream" and (candidate / "__init__.py").is_file():
        return candidate.parent
    return None


def _repo_root_from_candidate(candidate: Path) -> Path | None:
    """Return the Voxtream repo root that contains configs/ for a candidate."""
    if (candidate / "configs").is_dir():
        return candidate
    if candidate.name == "voxtream" and (candidate.parent / "configs").is_dir():
        return candidate.parent
    if (candidate / "voxtream" / "__init__.py").is_file():
        return candidate
    if candidate.name == "voxtream" and (candidate / "__init__.py").is_file():
        return candidate.parent
    return None


def _prepend_src(candidate: Path) -> bool:
    src = _path_to_prepend(candidate)
    if src is None:
        return False
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    return True


def _find_source_candidate() -> tuple[Path, Path] | None:
    for candidate in _iter_voxtream_src_candidates():
        if not candidate.exists():
            continue
        src = _path_to_prepend(candidate)
        if src is not None:
            return candidate, src
    return None


def _repo_root_from_imported_module() -> Path | None:
    module = sys.modules.get("voxtream")
    if module is None:
        spec = importlib.util.find_spec("voxtream")
        module_file = spec.origin if spec is not None else None
    else:
        module_file = getattr(module, "__file__", None)

    if module_file is None:
        return None
    package_dir = Path(module_file).resolve().parent
    candidate = _repo_root_from_candidate(package_dir)
    if candidate is not None:
        return candidate
    return _repo_root_from_candidate(package_dir.parent)


def iter_voxtream_repo_roots() -> list[Path]:
    """Return candidate Voxtream repo roots that may contain configs/*.json."""
    roots: list[Path] = []
    imported_root = _repo_root_from_imported_module()
    if imported_root is not None:
        roots.append(imported_root)

    for candidate in _iter_voxtream_src_candidates():
        root = _repo_root_from_candidate(candidate)
        if root is not None:
            roots.append(root)

    return _dedupe_paths(roots)


def resolve_voxtream_file(relative_path: str) -> str:
    """Resolve a file from the Voxtream repo, e.g. configs/generator.json."""
    path = Path(relative_path).expanduser()
    if path.is_absolute():
        if path.is_file():
            return str(path)
        raise FileNotFoundError(f"Missing required Voxtream file: {path}")

    for root in iter_voxtream_repo_roots():
        candidate = root / relative_path
        if candidate.is_file():
            logger.info("Resolved Voxtream file %s via %s: %s", relative_path, root, candidate)
            return str(candidate)

    searched = ", ".join(str(p) for p in iter_voxtream_repo_roots()) or "<none>"
    raise FileNotFoundError(
        f"Missing required Voxtream file: {relative_path}. "
        f"Expected it under a Voxtream repo root such as ./voxtream/{relative_path}. "
        f"Searched roots: {searched}"
    )


def ensure_voxtream_available() -> None:
    """Ensure ``voxtream`` can be imported, trying source trees first."""
    if "voxtream" in sys.modules:
        return

    source_candidate = _find_source_candidate()
    if source_candidate is not None:
        candidate, _ = source_candidate
        _prepend_src(candidate)
        if importlib.util.find_spec("voxtream") is not None:
            logger.info("Using Voxtream package from %s", candidate)
            return

    if importlib.util.find_spec("voxtream") is not None:
        return

    raise ImportError(
        "Could not import Voxtream package. Install it with "
        "`pip install --no-deps -e /path/to/voxtream`, or set "
        "VLLM_OMNI_VOXTREAM_CODE_PATH to the Voxtream source tree."
    )
