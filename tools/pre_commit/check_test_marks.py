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
"""

from __future__ import annotations

import os
import re
import sys
from functools import lru_cache
from pathlib import Path

# CI level markers
LEVEL_MARKERS = ("core_model", "advanced_model", "full_model", "local_model", "slow")

# Platform markers that tests may apply directly (``pytest.mark.cpu``, …).
# SKU names come from pyproject ``[hardware-resource]`` and must not be applied
# as ``pytest.mark.H100`` — use ``hardware_test`` / ``hardware_marks`` instead.
PLATFORM_MARKERS = (
    "cpu",
    "gpu",
    "cuda",
    "rocm",
    "xpu",
    "npu",
    "musa",
)

# Helpers from tests/helpers/mark.py that auto-apply hardware + cards_* marks.
HARDWARE_HELPERS = ("hardware_test", "hardware_marks")

# The helper implementation is the only file allowed to write pytest.mark.<SKU>.
_ALLOWED_DIRECT_SKU_FILES = frozenset({"tests/helpers/mark.py"})

_HARDWARE_RESOURCE_MARKER_TAG = "[hardware-resource]"
_FALLBACK_SKU_MARKERS = (
    "H100",
    "H800",
    "H200",
    "L4",
    "B200",
    "MI325",
    "B60",
    "S5000",
    "A2",
    "A3",
)

# Match mark.X since we could also do `from pytest import mark`.
# \b prevents matching prefixes (e.g., mark.slow vs mark.slow_test).
LEVEL_RE = re.compile(r"mark\.(?:" + "|".join(LEVEL_MARKERS) + r")\b")
PLATFORM_RE = re.compile(r"mark\.(?:" + "|".join(PLATFORM_MARKERS) + r")\b")
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
def sku_markers() -> tuple[str, ...]:
    """SKU marker names tagged ``[hardware-resource]`` in ``pyproject.toml``."""
    path = _repo_root() / "pyproject.toml"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return _FALLBACK_SKU_MARKERS
    names = tuple(
        re.findall(
            rf'^\s*"([A-Za-z0-9_]+)\s*:\s*{re.escape(_HARDWARE_RESOURCE_MARKER_TAG)}',
            text,
            flags=re.M,
        )
    )
    return names or _FALLBACK_SKU_MARKERS


@lru_cache(maxsize=1)
def _sku_mark_re() -> re.Pattern[str]:
    names = sku_markers()
    return re.compile(r"mark\.(?:" + "|".join(re.escape(n) for n in names) + r")\b")


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
    return bool(LEVEL_RE.search(contents))


def has_hardware_marker(contents: str) -> bool:
    """Check if file contents contain a platform marker or hardware helper."""
    return bool(PLATFORM_RE.search(contents) or HELPER_RE.search(contents))


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
            f"Level marks, e.g.: {', '.join(LEVEL_MARKERS[:4])}\n"
            f"Hardware marks, e.g.: {', '.join(PLATFORM_MARKERS[:4])}, ...\n"
            f"  or helpers: {', '.join(HARDWARE_HELPERS)}\n"
            f"Do not write pytest.mark.<SKU> ({sku}). "
            "Use hardware_test(...) / hardware_marks(...) so cards_* is attached.\n\n"
            "The following files are missing marks:\n"
            f"{file_lines}\n\n"
            "To skip: SKIP=check-mark git commit ..."
        )
        sys.exit(1)
