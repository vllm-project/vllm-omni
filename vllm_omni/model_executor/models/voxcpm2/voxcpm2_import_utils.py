# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Dynamic import utilities for the native VoxCPM2 package.

Supports three discovery modes (first match wins):
1. ``VLLM_OMNI_VOXCPM_CODE_PATH`` env var (explicit source tree; deprecated)
2. Sibling ``../VoxCPM/src`` relative to the vllm-omni repo root
3. pip-installed ``voxcpm`` package (>= 2.0)
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


def _iter_voxcpm2_src_candidates() -> list[Path]:
    """Yield candidate source directories for VoxCPM2."""
    candidates: list[Path] = []
    env_path = os.environ.get("VLLM_OMNI_VOXCPM_CODE_PATH")
    if env_path:
        logger.warning_once(
            "VLLM_OMNI_VOXCPM_CODE_PATH is deprecated and will be removed in a future release. "
            "Install voxcpm>=2.0 instead (pip install voxcpm, or pip install -e <path> for a source checkout)."
        )
        candidates.append(Path(env_path).expanduser())

    repo_root = Path(__file__).resolve().parents[4]
    candidates.append(repo_root.parent / "VoxCPM" / "src")

    seen: set[str] = set()
    unique: list[Path] = []
    for c in candidates:
        key = str(c)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def _prepend_src(candidate: Path) -> None:
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


def _import_voxcpm2_attrs(module_name: str, *attr_names: str) -> tuple[Any, ...]:
    """Import attributes from the voxcpm package, trying source tree first."""
    last_exc: ImportError | None = None

    for candidate in _iter_voxcpm2_src_candidates():
        if not candidate.exists():
            continue
        _prepend_src(candidate)
        try:
            mod = importlib.import_module(module_name)
            return tuple(getattr(mod, name) for name in attr_names)
        except (ImportError, AttributeError) as exc:
            last_exc = ImportError(str(exc))
            continue

    try:
        mod = importlib.import_module(module_name)
        return tuple(getattr(mod, name) for name in attr_names)
    except (ImportError, AttributeError) as exc:
        last_exc = ImportError(str(exc))

    raise ImportError(
        f"Could not import {attr_names} from {module_name}. "
        f"Install voxcpm>=2.0: pip install voxcpm. "
        f"Last error: {last_exc}"
    )


def import_voxcpm2_core():
    """Import the VoxCPM core class used to load the native TTS model."""
    (VoxCPM,) = _import_voxcpm2_attrs("voxcpm.core", "VoxCPM")
    return VoxCPM
