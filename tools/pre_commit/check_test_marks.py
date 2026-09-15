#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run a pre-commit hook that fails if test files are modified or added
that (probably) never run in the CI. For now, this means that every tests file
needs to have a CI level marker (e.g., core_model, advanced_model, full_model,
local_model, slow, etc) and hardware mark / helper so that we ensure mutated
tests will actually be selected as long as there are pytest commands pointing
at the right paths.

SKU markers (``H100``, ``L4``, … tagged ``[hardware-resource]`` in
``pyproject.toml``) must be applied via ``hardware_test(`` / ``hardware_marks(``
so ``cards_{n}`` is attached. Direct ``pytest.mark.H100`` is rejected.

Platform names allowed as ``pytest.mark.cpu`` / ``cuda`` come from
``get_supported_platforms()`` (``[hardware-platform]`` in ``pyproject.toml``).
CI level names come from ``get_level_markers()`` (``[ci-level]``).
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

# Helpers from tests/helpers/mark.py that auto-apply hardware + cards_* marks.
HARDWARE_HELPERS = ("hardware_test", "hardware_marks")

# The helper implementation is the only file allowed to write pytest.mark.<SKU>.
_ALLOWED_DIRECT_SKU_FILES = frozenset({"tests/helpers/mark.py"})

# Match mark.X since we could also do `from pytest import mark`.
# \b prevents matching prefixes (e.g., mark.slow vs mark.slow_test).
HELPER_RE = re.compile(r"(?:" + "|".join(HARDWARE_HELPERS) + r")\s*\(")

MISSING_LEVEL_MARKER = "Level"
MISSING_HARDWARE_MARKER = "Hardware"
DIRECT_SKU_MARKER = "Direct SKU"

# Check if a file is located under tests/ and matches test_<something>.py
# or <something>_test.py, since pytest technically collects on both.
# Note that we use the former everywhere in this repo by convention.
TEST_FILE_RE = re.compile(r"^tests/(?:.*/)?(?:test_[^/]*\.py$|[^/]*_test\.py$)")


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _mark_module() -> ModuleType:
    """Load ``mark.py`` by file path (no pytest/vllm; skip helpers ``__init__``)."""
    path = _repo_root() / "tests" / "helpers" / "mark.py"
    spec = importlib.util.spec_from_file_location("_vllm_omni_mark", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load mark helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mark_name_re(names: tuple[str, ...]) -> re.Pattern[str]:
    if not names:
        return re.compile(r"(?!)")
    return re.compile(r"mark\.(?:" + "|".join(re.escape(n) for n in names) + r")\b")


@lru_cache(maxsize=1)
def level_markers() -> tuple[str, ...]:
    """CI level names tagged ``[ci-level]`` (``core_model``, ``slow``, …)."""
    return tuple(sorted(_mark_module().get_level_markers()))


@lru_cache(maxsize=1)
def platform_markers() -> tuple[str, ...]:
    """Names tests may apply as ``pytest.mark.cpu`` / ``cuda`` (not SKUs)."""
    return tuple(sorted(_mark_module().get_supported_platforms()))


@lru_cache(maxsize=1)
def sku_markers() -> tuple[str, ...]:
    """SKU marker names tagged ``[hardware-resource]`` in ``pyproject.toml``."""
    return tuple(sorted(_mark_module().get_hardware_mark_list()))


@lru_cache(maxsize=1)
def _level_re() -> re.Pattern[str]:
    return _mark_name_re(level_markers())


@lru_cache(maxsize=1)
def _platform_re() -> re.Pattern[str]:
    return _mark_name_re(platform_markers())


@lru_cache(maxsize=1)
def _sku_mark_re() -> re.Pattern[str]:
    return _mark_name_re(sku_markers())


def is_test_file(path: str) -> bool:
    """Determine whether or not a path is pointing at a test file or not."""
    return bool(TEST_FILE_RE.match(_normalize_path(path)))


def read_test_file(path: str) -> str | None:
    """Read a test file's contents, or return None if it doesn't exist."""
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


def has_level_marker(contents: str) -> bool:
    """Check if file contents contain at least one CI level marker."""
    return bool(_level_re().search(contents))


def has_hardware_marker(contents: str) -> bool:
    """Check if file contents contain a platform marker or hardware helper."""
    return bool(_platform_re().search(contents) or HELPER_RE.search(contents))


def has_direct_sku_marker(path: str, contents: str) -> bool:
    """True when a test applies ``pytest.mark.<SKU>`` instead of the helpers."""
    if _normalize_path(path) in _ALLOWED_DIRECT_SKU_FILES:
        return False
    return bool(_sku_mark_re().search(contents))


def get_files_missing_markers(
    staged_files: list[str],
) -> dict[str, list[str]]:
    """Return a dict mapping file path to list of missing / invalid marker types."""
    results: dict[str, list[str]] = {}
    for path in staged_files:
        if is_test_file(path) and (contents := read_test_file(path)) is not None:
            missing = []
            if has_direct_sku_marker(path, contents):
                missing.append(DIRECT_SKU_MARKER)
            if not has_level_marker(contents):
                missing.append(MISSING_LEVEL_MARKER)
            if not has_hardware_marker(contents):
                missing.append(MISSING_HARDWARE_MARKER)
            if missing:
                results[path] = missing
    return results


if __name__ == "__main__":
    missing = get_files_missing_markers(sys.argv[1:])

    if missing:
        file_lines = "\n".join(f"  - {path} [{' and '.join(problems)}]" for path, problems in missing.items())
        sku = ", ".join(sku_markers())
        print(
            "\033[91merror:\033[0m test files are missing pytest marks "
            "required for Buildkite CI collection, or apply SKU marks directly.\n\n"
            f"Level marks, e.g.: {', '.join(level_markers()[:4])}\n"
            f"Hardware marks, e.g.: {', '.join(platform_markers()[:4])}, ...\n"
            f"  or helpers: {', '.join(HARDWARE_HELPERS)}\n"
            f"Do not write pytest.mark.<SKU> ({sku}). "
            "Use hardware_test(...) / hardware_marks(...) so cards_* is attached.\n\n"
            "The following files are missing marks:\n"
            f"{file_lines}\n\n"
            "To skip: SKIP=check-mark git commit ..."
        )
        sys.exit(1)
