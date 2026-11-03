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

``[hardware-platform]`` marks other than ``cpu`` (``gpu``, ``cuda``, ``rocm``,
``xpu``, ``npu``, ``musa``) are also rejected. Those tests must use the same
helpers so a platform SKU and ``cards_{n}`` are attached and CI can select with
``-m "cuda and L4"`` instead of a bare platform mark. The helper itself is the
only file allowed to emit those marks.

``pytest.mark.cpu`` stays direct: CPU has no SKU, and ``hardware_test(res=...)``
rejects ``cpu`` as a res key. ``gpu`` is auto-added for ``[gpu]`` platforms and
must not be written by hand.

Platform names come from ``get_supported_platforms()`` (``[hardware-platform]``
in ``pyproject.toml``). CI level names come from ``get_level_markers()``
(``[ci-level]``).
"""

from __future__ import annotations

import ast
import importlib.util
import io
import os
import re
import sys
import tokenize
from functools import lru_cache
from pathlib import Path
from types import ModuleType

# Helpers from tests/helpers/mark.py that auto-apply hardware + cards_* marks.
HARDWARE_HELPERS = ("hardware_test", "hardware_marks")

# The helper implementation is the only file allowed to write pytest.mark.<SKU>
# or a non-CPU hardware-platform mark.
_ALLOWED_DIRECT_SKU_FILES = frozenset({"tests/helpers/mark.py"})

# cpu has no SKU. hardware_test(res=...) rejects it; pytest.mark.cpu stays valid.
_DIRECT_PLATFORM_ALLOWLIST = frozenset({"cpu"})

# Common diffusion tests receive per-case marks from this parametrization helper.
_DELEGATED_MARK_SOURCES = {
    "tests/model_tests/diffusion/test_common_offline.py": "tests/model_tests/diffusion/case_filtering.py",
    "tests/model_tests/diffusion/test_common_online.py": "tests/model_tests/diffusion/case_filtering.py",
}
_DELEGATED_MARK_CALLS = ("get_parametrized_options(",)

# Match mark.X since we could also do `from pytest import mark`.
# \b prevents matching prefixes (e.g., mark.slow vs mark.slow_test).
HELPER_RE = re.compile(r"(?:" + "|".join(HARDWARE_HELPERS) + r")\s*\(")

MISSING_LEVEL_MARKER = "Level"
MISSING_HARDWARE_MARKER = "Hardware"
DIRECT_SKU_MARKER = "Direct SKU"
BARE_PLATFORM_MARKER = "Bare platform"
INVALID_HARDWARE_HELPER = "Invalid hardware helper"
CONFLICTING_DEVICE_MARKERS = "Conflicting CPU/accelerator marks"
CONFLICTING_PLATFORM_MARKERS = "Conflicting rocm/nvidia_only marks"
_DISALLOWED_MARKERS = frozenset(
    {
        DIRECT_SKU_MARKER,
        BARE_PLATFORM_MARKER,
        INVALID_HARDWARE_HELPER,
        CONFLICTING_DEVICE_MARKERS,
        CONFLICTING_PLATFORM_MARKERS,
    }
)
_PROBLEM_TEXT = {
    DIRECT_SKU_MARKER: "disallowed SKU mark",
    BARE_PLATFORM_MARKER: "disallowed platform mark",
    INVALID_HARDWARE_HELPER: "invalid hardware helper",
    CONFLICTING_DEVICE_MARKERS: "conflicting CPU/accelerator marks",
    CONFLICTING_PLATFORM_MARKERS: "conflicting rocm and nvidia_only marks",
    MISSING_LEVEL_MARKER: "missing level mark",
    MISSING_HARDWARE_MARKER: "missing hardware mark",
}

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
    """``[hardware-platform]`` names (``cpu``, ``cuda``, …), not SKUs."""
    return tuple(sorted(_mark_module().get_supported_platforms()))


@lru_cache(maxsize=1)
def bare_platform_markers() -> tuple[str, ...]:
    """Platform marks that must come from ``hardware_test`` / ``hardware_marks``."""
    return tuple(sorted(set(platform_markers()) - _DIRECT_PLATFORM_ALLOWLIST))


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


@lru_cache(maxsize=1)
def _bare_platform_re() -> re.Pattern[str]:
    return _mark_name_re(bare_platform_markers())


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


def _blanked_token_types() -> frozenset[int]:
    """Token types that are not marker applications."""
    types = {tokenize.COMMENT, tokenize.STRING}
    for name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"):
        token_type = getattr(tokenize, name, None)
        if token_type is not None:
            types.add(token_type)
    return frozenset(types)


def _blank_keeping_newlines(text: str) -> str:
    return "".join("\n" if char == "\n" else " " for char in text)


def _source_for_disallowed_mark_scan(contents: str) -> str:
    """Blank comments and string literals before searching for ``mark.<name>``.

    A mention in a comment or string is not a marker application. Real
    ``pytest.mark.cuda`` / ``mark.cuda`` expressions stay in the scanned text.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(contents).readline))
    except (tokenize.TokenError, SyntaxError):
        return contents
    blanked_token_types = _blanked_token_types()
    spans = [tok for tok in tokens if tok.type in blanked_token_types and tok.start < tok.end]
    if not spans:
        return contents
    lines = contents.splitlines(keepends=True)
    for tok in spans:
        (start_line, start_col), (end_line, end_col) = tok.start, tok.end
        if start_line == end_line:
            index = start_line - 1
            line = lines[index]
            lines[index] = line[:start_col] + _blank_keeping_newlines(line[start_col:end_col]) + line[end_col:]
            continue
        first = lines[start_line - 1]
        lines[start_line - 1] = first[:start_col] + _blank_keeping_newlines(first[start_col:])
        for index in range(start_line, end_line - 1):
            lines[index] = _blank_keeping_newlines(lines[index])
        lines[end_line - 1] = _blank_keeping_newlines(lines[end_line - 1][:end_col]) + lines[end_line - 1][end_col:]
    return "".join(lines)


def has_direct_sku_marker(path: str, contents: str) -> bool:
    """True when a test applies ``pytest.mark.<SKU>`` instead of the helpers."""
    if _normalize_path(path) in _ALLOWED_DIRECT_SKU_FILES:
        return False
    return bool(_sku_mark_re().search(_source_for_disallowed_mark_scan(contents)))


def has_bare_platform_marker(path: str, contents: str) -> bool:
    """True when a test writes ``pytest.mark.<platform>`` instead of a SKU helper.

    ``pytest.mark.cpu`` is allowed. ``gpu`` / ``cuda`` / ``rocm`` / ``xpu`` /
    ``npu`` / ``musa`` must go through ``hardware_test`` / ``hardware_marks``.
    Comments and string literals that mention those marks are ignored.
    """
    if _normalize_path(path) in _ALLOWED_DIRECT_SKU_FILES:
        return False
    return bool(_bare_platform_re().search(_source_for_disallowed_mark_scan(contents)))


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _hardware_helper_names(tree: ast.AST) -> dict[str, str]:
    helpers = {name: name for name in HARDWARE_HELPERS}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "tests.helpers.mark":
            for alias in node.names:
                if alias.name in HARDWARE_HELPERS:
                    helpers[alias.asname or alias.name] = alias.name
    return helpers


_DYNAMIC = object()


def _literal(node: ast.AST):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return _DYNAMIC


def _walk_without_pytest_params(root: ast.AST):
    """Walk a decorator while keeping per-item pytest.param marks separate."""
    stack = [root]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Call) and _name(node.func) == "param":
            continue
        yield node
        stack.extend(reversed(list(ast.iter_child_nodes(node))))


def hardware_contract_errors(contents: str) -> list[str]:
    """Validate collection-time helper calls and incompatible mark combinations.

    Test modules are never imported. Dynamic helper arguments remain subject to
    runtime collection, and calls inside test bodies stay available to negative
    tests that intentionally exercise rejected inputs.
    """
    try:
        tree = ast.parse(contents)
    except SyntaxError:
        return []  # Python syntax is checked by a separate hook.

    helpers = _hardware_helper_names(tree)
    collection_calls: list[ast.Call] = []
    decorators: set[int] = set()

    class CollectionVisitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call):
            collection_calls.append(node)
            self.generic_visit(node)

        def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
            for decorator in node.decorator_list:
                decorators.add(id(decorator))
                self.visit(decorator)
            for default in [*node.args.defaults, *node.args.kw_defaults]:
                if default is not None:
                    self.visit(default)
            # A test body may intentionally call a helper under pytest.raises.
            # Other function bodies can provide marks evaluated at collection.
            if not node.name.startswith("test_"):
                for statement in node.body:
                    self.visit(statement)

        def visit_FunctionDef(self, node: ast.FunctionDef):
            self._visit_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            self._visit_function(node)

        def visit_Lambda(self, node: ast.Lambda):
            for default in [*node.args.defaults, *node.args.kw_defaults]:
                if default is not None:
                    self.visit(default)

        def visit_ClassDef(self, node: ast.ClassDef):
            for decorator in node.decorator_list:
                decorators.add(id(decorator))
                self.visit(decorator)
            for base in node.bases:
                self.visit(base)
            for keyword in node.keywords:
                self.visit(keyword.value)
            for statement in node.body:
                self.visit(statement)

    CollectionVisitor().visit(tree)

    invalid = False
    mark_module = _mark_module()
    platforms = mark_module._res_platforms()
    counts = mark_module.get_supported_card_counts()
    for node in collection_calls:
        helper = helpers.get(_name(node.func))
        if helper is None:
            continue
        if node.args:
            invalid = True  # Both helpers are keyword-only.
        if helper == "hardware_marks" and id(node) in decorators:
            invalid = True  # hardware_marks returns a list, not a decorator.

        kwargs = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
        if set(kwargs) - {"res", "num_cards"}:
            invalid = True
        has_dynamic_kwargs = any(keyword.arg is None for keyword in node.keywords)

        resource_node = kwargs.get("res")
        resources = _literal(resource_node) if resource_node is not None else _DYNAMIC
        if resource_node is None and not has_dynamic_kwargs:
            invalid = True
        if resources is not _DYNAMIC and (not isinstance(resources, dict) or not resources):
            invalid = True
        if isinstance(resources, dict):
            for platform, resource in resources.items():
                if platform not in platforms:
                    invalid = True
                    continue
                values = resource if isinstance(resource, (list, tuple)) else [resource]
                if not values or (platform != "cuda" and not isinstance(resource, str)):
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
                effective_counts = {cards.get(platform, 1) for platform in resources}
                if helper == "hardware_marks" and len(effective_counts) > 1:
                    invalid = True

    def device_marks(nodes, walker=ast.walk) -> set[str]:
        result: set[str] = set()
        for root in nodes:
            for node in walker(root):
                if isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Attribute) and node.value.attr == "mark":
                        result.add(node.attr)
                    elif isinstance(node.value, ast.Name) and node.value.id == "mark":
                        result.add(node.attr)
                if isinstance(node, ast.Call) and helpers.get(_name(node.func)):
                    resource_node = next((kw.value for kw in node.keywords if kw.arg == "res"), None)
                    resources = _literal(resource_node) if resource_node is not None else None
                    if isinstance(resources, dict):
                        result.update(set(resources) & platforms)
        return result

    accelerator_marks = platforms | {"gpu"}

    def conflict_kinds(marks: set[str]) -> set[str]:
        kinds: set[str] = set()
        if "cpu" in marks and bool(marks & accelerator_marks):
            kinds.add(CONFLICTING_DEVICE_MARKERS)
        # nvidia_only is an explicit NVIDIA dependency. A ROCm qualification
        # mark on the same item would both select and exclude that test.
        if "rocm" in marks and "nvidia_only" in marks:
            kinds.add(CONFLICTING_PLATFORM_MARKERS)
        return kinds

    def parameter_mark_axes(decorators: list[ast.expr]) -> list[list[set[str]]]:
        """Return one mark axis for each statically specified parametrization."""
        axes: list[list[set[str]]] = []
        for decorator in decorators:
            if not isinstance(decorator, ast.Call) or _name(decorator.func) != "parametrize":
                continue
            values = (
                decorator.args[1]
                if len(decorator.args) > 1
                else next(
                    (keyword.value for keyword in decorator.keywords if keyword.arg == "argvalues"),
                    None,
                )
            )
            if not isinstance(values, (ast.List, ast.Tuple, ast.Set)):
                continue
            axes.append(
                [
                    device_marks([value]) if isinstance(value, ast.Call) and _name(value.func) == "param" else set()
                    for value in values.elts
                ]
                or [set()]
            )
        return axes

    def expand_parameter_marks(base: list[set[str]], decorators: list[ast.expr]) -> list[set[str]]:
        effective = base
        for axis in parameter_mark_axes(decorators):
            effective = [marks | parameter_marks for marks in effective for parameter_marks in axis]
        return effective

    def conflicting_marks(body, inherited: list[set[str]]) -> set[str]:
        scope_marks = []
        for node in body:
            if isinstance(node, ast.Assign) and any(_name(target) == "pytestmark" for target in node.targets):
                scope_marks.append(node.value)
            elif isinstance(node, ast.AnnAssign) and _name(node.target) == "pytestmark" and node.value:
                scope_marks.append(node.value)
        scope_device_marks = device_marks(scope_marks)
        inherited = [marks | scope_device_marks for marks in inherited]
        found = set().union(*(conflict_kinds(marks) for marks in inherited))

        for node in body:
            if isinstance(node, ast.ClassDef):
                class_marks = device_marks(node.decorator_list, _walk_without_pytest_params)
                class_variants = expand_parameter_marks(
                    [marks | class_marks for marks in inherited],
                    node.decorator_list,
                )
                found |= conflicting_marks(node.body, class_variants)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                function_marks = device_marks(node.decorator_list, _walk_without_pytest_params)
                variants = expand_parameter_marks(
                    [marks | function_marks for marks in inherited],
                    node.decorator_list,
                )
                found |= set().union(*(conflict_kinds(marks) for marks in variants))
        return found

    conflicts = conflicting_marks(tree.body, [set()])
    return ([INVALID_HARDWARE_HELPER] if invalid else []) + [
        kind for kind in (CONFLICTING_DEVICE_MARKERS, CONFLICTING_PLATFORM_MARKERS) if kind in conflicts
    ]


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
            if has_bare_platform_marker(path, contents):
                missing.append(BARE_PLATFORM_MARKER)
            mark_source = _DELEGATED_MARK_SOURCES.get(_normalize_path(path))
            if mark_source is not None and any(call in contents for call in _DELEGATED_MARK_CALLS):
                contents += "\n" + (read_test_file(str(_repo_root() / mark_source)) or "")
            if not has_level_marker(contents):
                missing.append(MISSING_LEVEL_MARKER)
            if not has_hardware_marker(contents):
                missing.append(MISSING_HARDWARE_MARKER)
            missing.extend(hardware_contract_errors(contents))
            if missing:
                results[path] = missing
    return results


def _format_problem_list(problems: list[str]) -> str:
    return " and ".join(_PROBLEM_TEXT.get(problem, problem) for problem in problems)


def _failure_summary(findings: dict[str, list[str]]) -> tuple[str, str]:
    """Return the opening sentence and the file-list heading."""
    kinds = {problem for problems in findings.values() for problem in problems}
    disallowed = bool(kinds & _DISALLOWED_MARKERS)
    absent = bool(kinds - _DISALLOWED_MARKERS)
    if disallowed and absent:
        return (
            "test files failed the mark check: some apply marks that are not allowed, "
            "and some are missing the marks CI uses to collect them.",
            "The following files failed the mark check:",
        )
    if disallowed:
        return (
            "test files apply pytest marks that are not allowed. "
            "Write them with hardware_test(...) / hardware_marks(...).",
            "The following files apply disallowed marks:",
        )
    return (
        "test files are missing pytest marks required for Buildkite CI collection.",
        "The following files are missing marks:",
    )


if __name__ == "__main__":
    missing = get_files_missing_markers(sys.argv[1:])

    if missing:
        file_lines = "\n".join(f"  - {path} [{_format_problem_list(problems)}]" for path, problems in missing.items())
        summary, heading = _failure_summary(missing)
        sku = ", ".join(sku_markers())
        platforms = ", ".join(bare_platform_markers())
        print(
            f"\033[91merror:\033[0m {summary}\n\n"
            f"Level marks, e.g.: {', '.join(level_markers()[:4])}\n"
            f"Hardware: pytest.mark.cpu, or helpers: {', '.join(HARDWARE_HELPERS)}\n"
            f"Do not write pytest.mark.<platform> ({platforms}) or pytest.mark.<SKU> ({sku}). "
            "Use hardware_test(...) / hardware_marks(...) so the platform, a SKU, and cards_* are attached. "
            "pytest.mark.cpu stays direct because CPU has no SKU.\n"
            "Use hardware_test as a decorator; hardware_marks returns a mark list. "
            "Use registered resource/card values and do not combine CPU and accelerator marks on one item.\n\n"
            f"{heading}\n"
            f"{file_lines}\n\n"
            "To skip: SKIP=check-mark git commit ..."
        )
        sys.exit(1)
