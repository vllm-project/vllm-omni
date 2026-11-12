# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Choose a Buildkite shard count from measured durations.

``automatic_shard: true`` does not set ``parallelism`` in the pipeline YAML.
The uploader sets it. With timings, it keeps one job unless this step's own
timings show its measured files would miss 80% of ``timeout_in_minutes``.
The rest of the job timeout is left for checkout, collection, and process
startup, which are not in the file durations. It then packs those files with
a greedy longest-processing-time assignment, never more than
``MAX_AUTOMATIC_SHARDS``, and each shard runs only the files in its bin.
Pytest still receives the original directories, so a directory ``conftest.py``
loads with those tests; the timings plugin then drops every file that is not
in the bin. The packing is not the smallest possible.

With no usable timings the uploader splits the files whose static marks
satisfy the command's ``-m`` expression, using the smaller of
``MAX_AUTOMATIC_SHARDS`` and that file count. A file that cannot be checked
stays beside a file that does match, and it does not open a bin by itself.
No confirmed match keeps the step at one job, so a shard cannot collect zero
tests. One confirmed file stays one job. A finished session's unmeasured
files were filtered by ``-m`` and stay at weight 0; they are still placed in
a bin so a new file keeps running.

Durations are keyed by step label, with a `` · Shard N/M`` suffix removed, so
one step does not inherit another step's time for the same path.

Every enrolled step always records per-file timings through
``pytest_file_timings`` and uploads ``timings/jobs/*.json``, even when it
stays one job.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any, NamedTuple

MAX_AUTOMATIC_SHARDS = 4
# File seconds stop at each test. Checkout, collection, and startup still run
# inside ``timeout_in_minutes``, so packing uses only this fraction of it.
FILE_TIME_BUDGET_RATIO = 0.8
PLUGIN_MODULE = "pytest_file_timings"
PLUGIN_FLAG = f"-p {PLUGIN_MODULE}"
SCRIPTS_DIR = ".buildkite/common/scripts"
# `$$` is required. Buildkite interpolates `$` before the shell runs, and
# `${PYTHONPATH:+...}` is parsed as an offset, which rejects `+`.
PYTHONPATH_EXPORT = f'export PYTHONPATH="{SCRIPTS_DIR}$${{PYTHONPATH:+:$$PYTHONPATH}}"'
ARTIFACT_PATHS = ["timings/jobs/*.json"]
_SHARD_SUFFIX = re.compile(r" · Shard(?: %N/%t| \d+/\d+)$")
_SHELL_EXPAND_ARG = re.compile(r"^(?:--(?:num-shards|shard-id)=)?\${1,2}[A-Za-z_][A-Za-z0-9_]*$")
FILE_SHARD_ENV = "VLLM_OMNI_FILE_SHARD_LIST"
FILE_SHARD_ROOT = Path("timings/file-shards")
_VALUE_OPTIONS = {
    "-m",
    "--run-level",
    "-k",
    "--deselect",
    "-n",
    "--num-shards",
    "--shard-id",
    "-p",
    "--override-ini",
    "-o",
    "--tb",
    "-c",
}


def base_label(label: str) -> str:
    """Step identity shared by an unsplit job and its ``Shard N/M`` copies."""
    return _SHARD_SUFFIX.sub("", label)


def load_shard_timings() -> dict[str, dict[str, float]] | None:
    """Read ``{step label: {relative path: seconds}}`` from ``VLLM_OMNI_SHARD_TIMINGS``.

    The file uses a ``steps`` object. A flat ``file_seconds`` map is ignored,
    because the same path is timed by more than one step. A missing variable
    or file means there is nothing to plan from.
    """
    raw = os.environ.get("VLLM_OMNI_SHARD_TIMINGS", "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    steps = payload.get("steps") if isinstance(payload, dict) else None
    if not isinstance(steps, dict):
        return None
    parsed: dict[str, dict[str, float]] = {}
    for label, files in steps.items():
        if not isinstance(files, dict):
            continue
        weights = {str(key).replace("\\", "/"): float(value) for key, value in files.items() if float(value) > 0}
        if weights:
            parsed[base_label(str(label))] = weights
    return parsed or None


def _profile_payload() -> dict | None:
    raw = os.environ.get("VLLM_OMNI_SHARD_TIMINGS", "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def load_step_queues() -> dict[str, str]:
    """Agent queue recorded for each step label, if the profile has one."""
    payload = _profile_payload()
    queues = payload.get("step_queues") if isinstance(payload, dict) else None
    if not isinstance(queues, dict):
        return {}
    return {base_label(str(label)): str(queue) for label, queue in queues.items() if queue}


def fallback_shard_count(file_count: int, *, max_shards: int = MAX_AUTOMATIC_SHARDS) -> int:
    """Without usable timings, take the smaller of the job cap and the files that can fill a shard."""
    if file_count < 1 or max_shards < 1:
        return 1
    return min(max_shards, file_count)


def _spread_files(files: list[str], count: int) -> list[list[str]]:
    """Deal sorted paths round-robin so a missing profile still divides the suite."""
    count = max(1, min(count, len(files)))
    bins: list[list[str]] = [[] for _ in range(count)]
    for index, name in enumerate(sorted(files)):
        bins[index % count].append(name)
    return bins


def plan_shard_count(
    weights: list[float],
    budget_seconds: float,
    *,
    max_shards: int = MAX_AUTOMATIC_SHARDS,
) -> int:
    """Greedy longest-processing-time count, capped by ``max_shards``.

    This is not the smallest packing. Weights ``[8, 6, 5, 5, 4, 4]`` fit a
    budget of 16 in two bins, and this search returns three. A file that
    already exceeds the budget does not add shards, because extra bins cannot
    shorten it. ``pack_files`` places paths into the bins this count selects.
    """
    if budget_seconds <= 0 or not weights or max_shards < 1:
        return 1
    best_n = 1
    best_load = _longest_bin(weights, 1)
    if best_load <= budget_seconds:
        return 1
    for count in range(2, max_shards + 1):
        load = _longest_bin(weights, count)
        if load < best_load - 1e-9:
            best_n = count
            best_load = load
        if load <= budget_seconds:
            return count
    return best_n


def pack_files(
    weights: dict[str, float],
    budget_seconds: float,
    *,
    max_shards: int = MAX_AUTOMATIC_SHARDS,
) -> list[list[str]]:
    """Place each path into the bins selected by :func:`plan_shard_count`."""
    if not weights:
        return []
    count = plan_shard_count(list(weights.values()), budget_seconds, max_shards=max_shards)
    loads = [0.0] * count
    bins: list[list[str]] = [[] for _ in range(count)]
    for name, weight in sorted(weights.items(), key=lambda item: (-item[1], item[0])):
        index = min(range(count), key=loads.__getitem__)
        bins[index].append(name)
        loads[index] += weight
    return [names for names in bins if names]


def apply_automatic_shard(
    step: dict[str, Any],
    *,
    timings: dict[str, dict[str, float]] | None,
    root: Path,
    log,
) -> dict[str, Any]:
    """Strip ``automatic_shard``, always collect timings, maybe set parallelism."""
    enabled = step.pop("automatic_shard", False)
    if not enabled:
        return step
    label = str(step.get("label") or "<step>")
    commands = step.get("commands")
    if not isinstance(commands, list):
        log(f"automatic-shard: {label!r} stays 1 job (no command list)")
        return step

    shardable: list[int] = []
    instrumentable: list[int] = []
    files: list[str] = []
    per_command: list[tuple[str, list[str]]] = []
    for index, command in enumerate(commands):
        if not isinstance(command, str):
            continue
        if _pytest_argv(command) is None and "pytest" not in command:
            continue
        instrumentable.append(index)
        discovered = discover_test_files(command, root)
        if discovered:
            shardable.append(index)
            per_command.append((command, discovered))
            files.extend(discovered)
    files = sorted(set(files))

    paths = list(step.get("artifact_paths") or [])
    for pattern in ARTIFACT_PATHS:
        if pattern not in paths:
            paths.append(pattern)
    step["artifact_paths"] = paths

    if step.get("parallelism"):
        step["commands"] = _instrument_commands(commands, instrumentable)
        log(f"automatic-shard: {label!r} keeps explicit parallelism")
        return step
    if not shardable:
        step["commands"] = _instrument_commands(commands, instrumentable)
        log(f"automatic-shard: {label!r} stays 1 job (no directory pytest)")
        return step

    timeout_min = step.get("timeout_in_minutes")
    try:
        budget_seconds = float(timeout_min) * 60.0 * FILE_TIME_BUDGET_RATIO
    except (TypeError, ValueError):
        budget_seconds = 0.0
    step_timings = None if not timings else timings.get(base_label(label))
    recorded_queue = load_step_queues().get(base_label(label), "")
    step_queue = str((step.get("agents") or {}).get("queue") or "")
    if step_timings is not None and recorded_queue and step_queue and recorded_queue != step_queue:
        log(f"automatic-shard: {label!r} ignores timings recorded on {recorded_queue}")
        step_timings = None
    planned = _plan_weights(files, step_timings)
    if planned is None:
        fallback = _plan_fallback(per_command, root)
        if fallback.bins is None:
            reason = "no timings for this step" if timings else "no timing profile"
            log(f"automatic-shard: {label!r} stays 1 job ({reason}; {fallback.hold})")
            step["commands"] = _instrument_commands(commands, instrumentable)
            return step
        bins = fallback.bins
        detail = (
            f"no timings; {len(bins)} jobs is min({MAX_AUTOMATIC_SHARDS} jobs, "
            f"{fallback.eligible} marker-matching files)"
        )
    else:
        weights, measured = planned
        bins = pack_files(weights, budget_seconds)
        if len(bins) <= 1:
            reason = f"measured {measured}/{len(files)} files fit"
            log(
                f"automatic-shard: {label!r} stays 1 job "
                f"({reason}; file budget {FILE_TIME_BUDGET_RATIO:.0%} of {timeout_min} min)"
            )
            step["commands"] = _instrument_commands(commands, instrumentable)
            return step
        detail = (
            f"measured {measured}/{len(files)} files, file budget {FILE_TIME_BUDGET_RATIO:.0%} of {timeout_min} min"
        )

    slug = _shard_slug(label)
    _write_file_shards(root, slug, bins)
    step["parallelism"] = len(bins)
    if "Shard %N/%t" not in label:
        step["label"] = f"{label} · Shard %N/%t"
    rewritten: list[Any] = []
    for index, command in enumerate(commands):
        if index in shardable and isinstance(command, str):
            rewritten.append(_file_shard_command(command, slug))
        else:
            rewritten.append(command)
    step["commands"] = _instrument_commands(rewritten, instrumentable)
    log(f"automatic-shard: {label!r} -> {len(bins)} jobs ({detail})")
    return step


def discover_test_files(command: str, root: Path) -> list[str]:
    """Test modules a plain pytest command would collect, excluding ``node::id`` selections."""
    argv = _pytest_argv(command)
    if argv is None:
        return []
    files: list[str] = []
    for target in _targets(argv):
        if "::" in target:
            continue
        files.extend(_files_for_target(target, root))
    return sorted(set(files))


class _Fallback(NamedTuple):
    bins: list[list[str]] | None
    eligible: int
    hold: str


_HARDWARE_CALLS = frozenset({"hardware_marks", "hardware_test"})
# Platforms whose ``hardware_test`` / ``hardware_marks`` also add ``pytest.mark.gpu``.
# ``npu`` is a hardware platform and does not get that mark.
_GPU_PLATFORMS = frozenset({"cuda", "rocm", "xpu", "musa"})
_ITEM_CACHE: dict[tuple[str, int, int], list[set[str]] | None] = {}
_EXPR_CACHE: dict[str, Any] = {}


def fallback_file_bins(command: str, root: Path) -> list[list[str]] | None:
    """Bins the no-profile fallback would publish for one pytest command."""
    return _plan_fallback([(command, discover_test_files(command, root))], root).bins


def _plan_fallback(per_command: list[tuple[str, list[str]]], root: Path) -> _Fallback:
    """Spread files that still have a test under each command's ``-m``.

    Shard count follows confirmed matches only. A file that cannot be checked
    is attached to the first bin, which already holds a confirmed match, so it
    still runs and cannot be the only file in a bin. No confirmed match keeps
    the step at one job.
    """
    matched_groups: list[list[str]] = []
    unknown: list[str] = []
    for command, discovered in per_command:
        expression = _marker_expression(command)
        if expression is None:
            matched_groups.append(list(discovered))
            continue
        matched: list[str] = []
        for name in discovered:
            verdict = _file_matches_marker(root / name, expression)
            if verdict is None:
                unknown.append(name)
            elif verdict:
                matched.append(name)
        if not matched:
            return _Fallback(None, 0, "no confirmed marker match")
        matched_groups.append(matched)
    eligible = min(len(set(group)) for group in matched_groups)
    count = fallback_shard_count(eligible, max_shards=MAX_AUTOMATIC_SHARDS)
    if count <= 1:
        noun = "file" if eligible == 1 else "files"
        return _Fallback(None, eligible, f"{eligible} marker-matching {noun} cannot split further")
    bins = _spread_groups(matched_groups, count)
    if not bins:
        return _Fallback(None, eligible, "marker matches do not cover every command")
    seen = {name for group in bins for name in group}
    for name in unknown:
        if name not in seen:
            bins[0].append(name)
            seen.add(name)
    return _Fallback(bins, eligible, "")


def _spread_groups(groups: list[list[str]], count: int) -> list[list[str]] | None:
    """Deal each command's matches so every bin keeps a file for every command."""
    if count <= 1:
        return None
    if len(groups) == 1:
        unique = sorted(set(groups[0]))
        if len(unique) < count:
            return None
        return _spread_files(unique, count)
    bins: list[list[str]] = [[] for _ in range(count)]
    assigned: dict[str, int] = {}
    for group in groups:
        ordered = sorted(set(group))
        covered = {assigned[name] for name in ordered if name in assigned}
        missing = [index for index in range(count) if index not in covered]
        fresh = [name for name in ordered if name not in assigned]
        if len(fresh) < len(missing):
            return None
        for bin_index, name in zip(missing, fresh, strict=False):
            assigned[name] = bin_index
            bins[bin_index].append(name)
        cursor = 0
        for name in fresh[len(missing) :]:
            bins[cursor % count].append(name)
            assigned[name] = cursor % count
            cursor += 1
    if any(not bucket for bucket in bins):
        return None
    return bins


def _marker_expression(command: str) -> str | None:
    argv = _pytest_argv(command)
    if argv is None:
        return None
    for index, arg in enumerate(argv):
        if arg == "-m" and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith("-m=") and len(arg) > 3:
            return arg[3:]
    return None


def _file_matches_marker(path: Path, expression: str) -> bool | None:
    """Whether any static item in ``path`` satisfies ``expression``.

    ``True`` for a path that is not a file, so a test can supply names the
    planner did not read from disk. ``None`` means the file or the expression
    could not be checked.
    """
    if not path.is_file():
        return True
    predicate = _marker_predicate(expression)
    if predicate is None:
        return None
    items = _item_marks(path)
    if not items:
        return None
    matched = False
    for marks in items:
        verdict = predicate(marks)
        if verdict is None:
            return None
        matched = matched or verdict
    return matched


def _marker_predicate(expression: str):
    if expression in _EXPR_CACHE:
        return _EXPR_CACHE[expression]
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        _EXPR_CACHE[expression] = None
        return None

    def evaluate(node: ast.AST, marks: set[str]) -> bool | None:
        if isinstance(node, ast.Expression):
            return evaluate(node.body, marks)
        if isinstance(node, ast.Name):
            return node.id in marks
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            inner = evaluate(node.operand, marks)
            return None if inner is None else not inner
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            values = [evaluate(child, marks) for child in node.values]
            if any(value is None for value in values):
                return None
            return all(values) if isinstance(node.op, ast.And) else any(values)
        return None

    if evaluate(tree, set()) is None:
        _EXPR_CACHE[expression] = None
        return None

    def predicate(marks: set[str]) -> bool | None:
        return evaluate(tree, marks)

    _EXPR_CACHE[expression] = predicate
    return predicate


def _item_marks(path: Path) -> list[set[str]] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key in _ITEM_CACHE:
        return _ITEM_CACHE[key]
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError, RecursionError):
        _ITEM_CACHE[key] = None
        return None
    items = _items_in_body(tree.body, set())
    _ITEM_CACHE[key] = items
    return items


def _items_in_body(body: list[ast.stmt], inherited: set[str]) -> list[set[str]] | None:
    inherited = set(inherited) | _pytestmark_assigns(body)
    items: list[set[str]] = []
    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name.startswith("test_"):
            found = _items_for_function(stmt, inherited)
            if found is None:
                return None
            items.extend(found)
        elif isinstance(stmt, ast.ClassDef) and stmt.name.startswith("Test"):
            bases = _class_bases(stmt, inherited)
            if bases is None:
                return None
            for base in bases:
                found = _items_in_body(stmt.body, base)
                if found is None:
                    return None
                items.extend(found)
    return items


def _class_bases(stmt: ast.ClassDef, inherited: set[str]) -> list[set[str]] | None:
    deco_marks, calls = _split_decorators(stmt.decorator_list)
    base = set(inherited) | deco_marks | _pytestmark_assigns(stmt.body)
    if not calls:
        return [base]
    variants: list[set[str]] = []
    for call in calls:
        found = _marks_from_hardware_call(call)
        if found is None:
            return None
        variants.extend(set(base) | marks for marks in found)
    return variants


def _items_for_function(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    inherited: set[str],
) -> list[set[str]] | None:
    deco_marks, _ = _split_decorators(fn.decorator_list)
    base = set(inherited) | deco_marks
    found = _function_mark_extras(fn)
    if found is None:
        return None
    calls, mark_sets = found
    items: list[set[str]] = []
    for call in calls:
        variants = _marks_from_hardware_call(call)
        if variants is None:
            return None
        items.extend(set(base) | marks for marks in variants)
    items.extend(set(base) | names for names in mark_sets)
    return items or [base]


def _split_decorators(decorators: list[ast.expr]) -> tuple[set[str], list[ast.Call]]:
    marks: set[str] = set()
    calls: list[ast.Call] = []
    for decorator in decorators:
        func = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(decorator, ast.Call) and _call_name(func) in _HARDWARE_CALLS:
            calls.append(decorator)
            continue
        marks |= _pytest_mark_names(decorator)
    return marks, calls


def _function_mark_extras(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[list[ast.Call], list[set[str]]] | None:
    """Hardware calls and ``marks=`` sets on one test, including parametrize decorators.

    A ``marks`` value that is not a literal ``pytest.mark`` or ``hardware_*`` call
    cannot be checked. The whole test is then unknown, so the file still runs.
    """
    calls: list[ast.Call] = []
    mark_sets: list[set[str]] = []
    stack: list[ast.AST] = [*fn.decorator_list, *fn.body]
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Call) and _call_name(node.func) in _HARDWARE_CALLS:
            calls.append(node)
            continue
        if isinstance(node, ast.keyword) and node.arg == "marks":
            if _marks_value_is_opaque(node.value):
                return None
            names = _pytest_mark_names(node.value)
            if names:
                mark_sets.append(names)
        stack.extend(ast.iter_child_nodes(node))
    return calls, mark_sets


def _marks_value_is_opaque(node: ast.AST) -> bool:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return any(_marks_value_is_opaque(elt) for elt in node.elts)
    if isinstance(node, ast.Starred):
        return True
    if isinstance(node, ast.Call) and _call_name(node.func) in _HARDWARE_CALLS:
        return _marks_from_hardware_call(node) is None
    if _pytest_mark_names(node) or isinstance(node, ast.Constant):
        return False
    return True


def _pytestmark_assigns(body: list[ast.stmt]) -> set[str]:
    marks: set[str] = set()
    for stmt in body:
        value: ast.AST | None = None
        if isinstance(stmt, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "pytestmark" for target in stmt.targets
        ):
            value = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == "pytestmark":
            value = stmt.value
        if value is not None:
            marks |= _pytest_mark_names(value)
    return marks


def _pytest_mark_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Attribute)
            and child.attr != "mark"
            and isinstance(child.value, ast.Attribute)
            and child.value.attr == "mark"
        ):
            names.add(child.attr)
    return names


def _marks_from_hardware_call(node: ast.Call) -> list[set[str]] | None:
    res_node: ast.AST | None = None
    cards_node: ast.AST | None = None
    for keyword in node.keywords:
        if keyword.arg == "res":
            res_node = keyword.value
        elif keyword.arg == "num_cards":
            cards_node = keyword.value
    if not isinstance(res_node, ast.Dict):
        return None
    platforms: list[tuple[str, list[str]]] = []
    for key, value in zip(res_node.keys, res_node.values, strict=False):
        if key is None:
            return None
        platform = _const_str(key)
        skus = _const_skus(value)
        if platform is None or skus is None:
            return None
        platforms.append((platform, skus))
    if not platforms:
        return None
    counts = _card_counts(cards_node, [platform for platform, _skus in platforms])
    if counts is None:
        return None
    mixed = len(platforms) > 1 and len(set(counts.values())) > 1
    if _call_name(node.func) == "hardware_marks" and mixed:
        return None

    def variant(platform: str, skus: list[str]) -> set[str]:
        marks = {platform, f"cards_{counts[platform]}", *skus}
        if platform in _GPU_PLATFORMS:
            marks.add("gpu")
        return marks

    if _call_name(node.func) == "hardware_test" and mixed:
        return [variant(platform, skus) for platform, skus in platforms]
    union: set[str] = set()
    for platform, skus in platforms:
        union |= variant(platform, skus)
    return [union]


def _card_counts(node: ast.AST | None, platforms: list[str]) -> dict[str, int] | None:
    if node is None:
        return {platform: 1 for platform in platforms}
    number = _const_int(node)
    if number is not None:
        return {platform: number for platform in platforms} if number >= 1 else None
    if not isinstance(node, ast.Dict):
        return None
    specified: dict[str, int] = {}
    for key, value in zip(node.keys, node.values, strict=False):
        if key is None:
            return None
        platform = _const_str(key)
        count = _const_int(value)
        if platform is None or count is None or count < 1:
            return None
        specified[platform] = count
    return {platform: specified.get(platform, 1) for platform in platforms}


def _const_skus(node: ast.AST) -> list[str] | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.List, ast.Tuple)):
        skus: list[str] = []
        for elt in node.elts:
            sku = _const_str(elt)
            if sku is None:
                return None
            skus.append(sku)
        return skus or None
    return None


def _const_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _const_int(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    return None


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _instrument_commands(commands: list[Any], indexes: list[int]) -> list[Any]:
    rewritten = list(commands)
    for index in indexes:
        command = rewritten[index]
        if not isinstance(command, str):
            continue
        rewritten[index] = _inject_plugin(command)
    if PYTHONPATH_EXPORT not in rewritten:
        rewritten.insert(0, PYTHONPATH_EXPORT)
    return rewritten


def _inject_plugin(command: str) -> str:
    if PLUGIN_MODULE in command:
        return command
    plain = _pytest_argv(command)
    if plain is not None and command.lstrip().startswith(("pytest", "timeout")):
        # Keep a single-line pytest / timeout pytest command as one argv list.
        argv = shlex.split(command)
        try:
            pytest_at = next(i for i, arg in enumerate(argv) if arg == "pytest" or arg.endswith("/pytest"))
        except StopIteration:
            return command
        argv.insert(pytest_at + 1, "-p")
        argv.insert(pytest_at + 2, PLUGIN_MODULE)
        return _shell_join(argv)

    # Multi-line scripts: rewrite every bare pytest invocation.
    lines = command.splitlines(keepends=True)
    out: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if stripped.startswith("pytest ") and PLUGIN_MODULE not in stripped:
            out.append(f"{indent}pytest {PLUGIN_FLAG} {stripped[len('pytest ') :]}")
        elif stripped.startswith("timeout ") and " pytest " in stripped and PLUGIN_MODULE not in stripped:
            out.append(line.replace(" pytest ", f" pytest {PLUGIN_FLAG} ", 1))
        else:
            out.append(line)
    return "".join(out)


def _plan_weights(
    files: list[str],
    timings: dict[str, float] | None,
) -> tuple[dict[str, float], int] | None:
    """Weight for every discovered file, plus the measured count.

    A file with no duration stays at 0 and is still placed in a bin. No positive
    measurement means there is nothing to pack.
    """
    if not timings:
        return None
    measured: dict[str, float] = {}
    for name in files:
        raw = timings.get(name)
        if raw is None:
            continue
        value = float(raw)
        if value > 0:
            measured[name] = value
    if not measured:
        return None
    weights = {name: measured.get(name, 0.0) for name in files}
    return weights, len(measured)


def _shard_slug(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", base_label(label).lower()).strip("-")
    return slug[:80] or "step"


def _write_file_shards(root: Path, slug: str, bins: list[list[str]]) -> None:
    # Schema checks render the same pipeline and only need the YAML. The
    # uploader writes these lists when it is about to publish them.
    if os.environ.get("VLLM_OMNI_SKIP_SHARD_FILE_WRITE", "").strip().lower() in {"1", "true", "yes"}:
        return
    dest = root / FILE_SHARD_ROOT / slug
    if dest.exists():
        for old in dest.glob("*.txt"):
            old.unlink()
    dest.mkdir(parents=True, exist_ok=True)
    for index, names in enumerate(bins):
        (dest / f"{index}.txt").write_text("".join(f"{name}\n" for name in names), encoding="utf-8")


def _file_shard_command(command: str, slug: str) -> str:
    """Run one packed bin. ``$$`` survives Buildkite interpolation.

    The pytest command keeps its directory targets. Passing the bin as file
    arguments collects those modules without the directory ``conftest.py``.
    ``pytest_file_timings`` reads ``VLLM_OMNI_FILE_SHARD_LIST`` and drops every
    other file after collection.
    """
    if _pytest_argv(command) is None:
        return command
    return "\n".join(
        (
            "set -eu",
            f'shard_list="{FILE_SHARD_ROOT.as_posix()}/{slug}/$${{BUILDKITE_PARALLEL_JOB:-0}}.txt"',
            'buildkite-agent artifact download "$$shard_list" .',
            'test -s "$$shard_list"',
            f'export {FILE_SHARD_ENV}="$$shard_list"',
            command,
        )
    )


def _shell_join(argv: list[str]) -> str:
    """Quote like a shell, but leave ``$BUILDKITE_*`` shard flags for expansion.

    Buildkite turns ``$$`` into ``$`` before the shell runs. ``shlex.join``
    would wrap those tokens in quotes, so the shell would pass the variable
    name through as a literal shard index.
    """
    parts: list[str] = []
    for arg in argv:
        if _SHELL_EXPAND_ARG.fullmatch(arg):
            parts.append(arg)
        else:
            parts.append(shlex.quote(arg))
    return " ".join(parts)


def _pytest_argv(command: str) -> list[str] | None:
    try:
        argv = shlex.split(command)
    except ValueError:
        return None
    if not argv:
        return None
    if argv[0] == "timeout":
        try:
            return argv[argv.index("pytest") :]
        except ValueError:
            return None
    if argv[0] != "pytest":
        return None
    return argv


def _targets(argv: list[str]) -> list[str]:
    targets: list[str] = []
    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg in _VALUE_OPTIONS:
            index += 2
            continue
        if arg.startswith("-"):
            index += 1
            continue
        targets.append(arg)
        index += 1
    return targets


def _files_for_target(target: str, root: Path) -> list[str]:
    if any(char in target for char in "*?["):
        matches = [path for path in root.glob(target) if path.is_file() and _is_test_module(path)]
        return [_relative(path, root) for path in matches]
    path = root / target
    if path.is_file() and _is_test_module(path):
        return [_relative(path, root)]
    if not path.is_dir():
        return []
    found: list[str] = []
    for file in path.rglob("*.py"):
        if _is_test_module(file):
            found.append(_relative(file, root))
    return found


def _is_test_module(path: Path) -> bool:
    name = path.name
    return name.startswith("test_") or name.endswith("_test.py")


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _longest_bin(weights: list[float], bins: int) -> float:
    loads = [0.0] * bins
    for weight in sorted(weights, reverse=True):
        smallest = min(range(bins), key=loads.__getitem__)
        loads[smallest] += weight
    return max(loads)
