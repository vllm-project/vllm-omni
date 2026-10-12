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

import ast
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

# Common diffusion tests receive per-case marks from this parametrization helper.
_DELEGATED_MARK_SOURCES = {
    "tests/model_tests/diffusion/test_common_offline.py": "tests/model_tests/diffusion/case_filtering.py",
    "tests/model_tests/diffusion/test_common_online.py": "tests/model_tests/diffusion/case_filtering.py",
}

# Match mark.X since we could also do `from pytest import mark`.
# \b prevents matching prefixes (e.g., mark.slow vs mark.slow_test).
HELPER_RE = re.compile(r"(?:" + "|".join(HARDWARE_HELPERS) + r")\s*\(")

MISSING_LEVEL_MARKER = "Level"
MISSING_HARDWARE_MARKER = "Hardware"
DIRECT_SKU_MARKER = "Direct SKU"
INVALID_HARDWARE_HELPER = "Invalid hardware helper"
CONFLICTING_DEVICE_MARKERS = "Conflicting CPU/GPU marks"

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
    if _platform_re().search(contents) or HELPER_RE.search(contents):
        return True
    try:
        tree = ast.parse(contents)
    except SyntaxError:
        return False
    helpers = _hardware_helper_names(tree)
    return any(isinstance(node, ast.Call) and _name(node.func) in helpers for node in ast.walk(tree))


def has_direct_sku_marker(path: str, contents: str) -> bool:
    """True when a test applies ``pytest.mark.<SKU>`` instead of the helpers."""
    if _normalize_path(path) in _ALLOWED_DIRECT_SKU_FILES:
        return False
    return bool(_sku_mark_re().search(contents))


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


_DYNAMIC = object()


def _hardware_helper_names(tree: ast.AST) -> dict[str, str]:
    helpers = {name: name for name in HARDWARE_HELPERS}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "tests.helpers.mark":
            for alias in node.names:
                if alias.name in HARDWARE_HELPERS:
                    helpers[alias.asname or alias.name] = alias.name
    return helpers


def _literal(node: ast.AST):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return _DYNAMIC


def hardware_contract_errors(contents: str) -> list[str]:
    """Validate literal helper arguments and inherited device-class marks.

    Do not import test modules: collection may need Torch, models or GPUs.
    Dynamic helper arguments remain the responsibility of runtime collection.
    """
    try:
        tree = ast.parse(contents)
    except SyntaxError:
        return []  # Python syntax is checked by a separate hook.
    helpers = _hardware_helper_names(tree)
    # Validate calls evaluated during collection. Calls in a test body can
    # intentionally exercise invalid arguments under pytest.raises.
    collection_nodes = []
    decorators = set()

    class CollectionVisitor(ast.NodeVisitor):
        def visit(self, node):
            collection_nodes.append(node)
            return super().visit(node)

        def visit_FunctionDef(self, node):
            for decorator in node.decorator_list:
                decorators.add(id(decorator))
                self.visit(decorator)
            for default in [*node.args.defaults, *node.args.kw_defaults]:
                if default is not None:
                    self.visit(default)

        def visit_AsyncFunctionDef(self, node):
            self.visit_FunctionDef(node)

        def visit_Lambda(self, node):
            for default in [*node.args.defaults, *node.args.kw_defaults]:
                if default is not None:
                    self.visit(default)

        def visit_ClassDef(self, node):
            decorators.update(id(decorator) for decorator in node.decorator_list)
            self.generic_visit(node)

    CollectionVisitor().visit(tree)
    invalid = False
    mark_module = _mark_module()
    platforms = mark_module._res_platforms()
    counts = mark_module.get_supported_card_counts()
    for node in collection_nodes:
        if not isinstance(node, ast.Call):
            continue
        helper = helpers.get(_name(node.func))
        if helper is None:
            continue
        # hardware_marks returns a list for pytest.param/pytestmark. Only
        # hardware_test returns a callable decorator.
        if helper == "hardware_marks" and id(node) in decorators:
            invalid = True
        kwargs = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
        resource_node = kwargs.get("res")
        resources = _literal(resource_node) if resource_node is not None else _DYNAMIC
        if resource_node is None and not any(keyword.arg is None for keyword in node.keywords):
            invalid = True
        if resources is not _DYNAMIC and (not isinstance(resources, dict) or not resources):
            invalid = True
        if isinstance(resources, dict):
            for platform, value in resources.items():
                if platform not in platforms:
                    invalid = True
                    continue
                values = value if isinstance(value, (list, tuple)) else [value]
                if not values or (platform != "cuda" and not isinstance(value, str)):
                    invalid = True
                if any(
                    not isinstance(sku, str) or sku not in mark_module.get_skus_for_platform(platform) for sku in values
                ):
                    invalid = True
        cards_node = kwargs.get("num_cards")
        cards = _literal(cards_node) if cards_node is not None else 1
        if cards is not _DYNAMIC:
            values = list(cards.values()) if isinstance(cards, dict) else [cards]
            if any(type(value) is not int or value not in counts for value in values):
                invalid = True
            if isinstance(cards, dict) and isinstance(resources, dict):
                if set(cards) - set(resources):
                    invalid = True
                if helper == "hardware_marks" and len({cards.get(platform, 1) for platform in resources}) > 1:
                    invalid = True

    def device_marks(nodes):
        result = set()
        for root in nodes:
            for node in ast.walk(root):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
                    if node.value.attr == "mark":
                        result.add(node.attr)
                if isinstance(node, ast.Call) and helpers.get(_name(node.func)):
                    resources = next((keyword.value for keyword in node.keywords if keyword.arg == "res"), None)
                    value = _literal(resources) if resources is not None else None
                    if isinstance(value, dict):
                        result.update(set(value) & mark_module._gpu_res_platforms())
        return result

    gpu_marks = mark_module._gpu_res_platforms() | {"gpu"}

    def conflicting_marks(body, inherited):
        scope_marks = []
        for node in body:
            if isinstance(node, ast.Assign) and any(_name(target) == "pytestmark" for target in node.targets):
                scope_marks.append(node.value)
            elif isinstance(node, ast.AnnAssign) and _name(node.target) == "pytestmark" and node.value:
                scope_marks.append(node.value)
        inherited = inherited | device_marks(scope_marks)
        for node in body:
            if isinstance(node, ast.ClassDef):
                if conflicting_marks(node.body, inherited | device_marks(node.decorator_list)):
                    return True
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                marks = inherited | device_marks(node.decorator_list)
                if "cpu" in marks and marks & gpu_marks:
                    return True
        return False

    conflict = conflicting_marks(tree.body, set())
    return ([INVALID_HARDWARE_HELPER] if invalid else []) + ([CONFLICTING_DEVICE_MARKERS] if conflict else [])


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
            mark_source = _DELEGATED_MARK_SOURCES.get(_normalize_path(path))
            if mark_source is not None and "get_parametrized_options(" in contents:
                contents += "\n" + (read_test_file(str(_repo_root() / mark_source)) or "")
            if not has_level_marker(contents):
                missing.append(MISSING_LEVEL_MARKER)
            if not has_hardware_marker(contents):
                missing.append(MISSING_HARDWARE_MARKER)
            missing.extend(hardware_contract_errors(contents))
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
            "Use hardware_test as a decorator; hardware_marks returns a list. "
            "Use registered resource/card values and separate CPU and GPU tests.\n\n"
            "The following files are missing marks:\n"
            f"{file_lines}\n\n"
            "To skip: SKIP=check-mark git commit ..."
        )
        sys.exit(1)
