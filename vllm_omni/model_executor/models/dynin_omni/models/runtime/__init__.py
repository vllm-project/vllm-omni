"""Runtime package bootstrap and import compatibility helpers."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]

_LAZY_SUBMODULES = {
    "config_resolver",
    "prompting_utils",
}


def ensure_project_root_on_path(*, prepend: bool = True) -> str:
    """Ensure repository root is importable for absolute imports like `models.*`."""
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        if prepend:
            sys.path.insert(0, project_root)
        else:
            sys.path.append(project_root)
    return project_root


# Keep compatibility with scripts that rely on `models.*` absolute imports.
ensure_project_root_on_path(prepend=True)


def __getattr__(name: str) -> ModuleType:
    """Lazily expose selected runtime submodules."""
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "PACKAGE_ROOT",
    "PROJECT_ROOT",
    "ensure_project_root_on_path",
]
