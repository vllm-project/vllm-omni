# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Choose a Buildkite shard count from measured durations.

``automatic_shard: true`` does not reserve a pool. The uploader keeps one job
unless this step's own timings show its measured files would miss
``timeout_in_minutes``. It then sets ``parallelism`` from a greedy
longest-processing-time estimate, never more than ``MAX_AUTOMATIC_SHARDS``.
That count is only a budget guess: pytest-shard hashes collected node ids and
does not place the files the estimate packed. The packing itself is not the
smallest possible. Unmeasured files add no weight, so a missing profile stays
one job.

Durations are keyed by step label, with a `` · Shard N/M`` suffix removed, so
one step does not inherit another step's time for the same path.

Every enrolled step always records per-file timings through
``pytest_file_timings`` and uploads ``timings/jobs/*.json``, even when it
stays one job.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path
from typing import Any

MAX_AUTOMATIC_SHARDS = 4
SHARD_FLAGS = "--num-shards=$$BUILDKITE_PARALLEL_JOB_COUNT --shard-id=$$BUILDKITE_PARALLEL_JOB"
PLUGIN_MODULE = "pytest_file_timings"
PLUGIN_FLAG = f"-p {PLUGIN_MODULE}"
SCRIPTS_DIR = ".buildkite/common/scripts"
# `$$` is required. Buildkite interpolates `$` before the shell runs, and
# `${PYTHONPATH:+...}` is parsed as an offset, which rejects `+`.
PYTHONPATH_EXPORT = f'export PYTHONPATH="{SCRIPTS_DIR}$${{PYTHONPATH:+:$$PYTHONPATH}}"'
ARTIFACT_PATHS = ["timings/jobs/*.json"]
_SHARD_SUFFIX = re.compile(r" · Shard(?: %N/%t| \d+/\d+)$")
_SHELL_EXPAND_ARG = re.compile(r"^(?:--(?:num-shards|shard-id)=)?\${1,2}[A-Za-z_][A-Za-z0-9_]*$")
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
    shorten it. pytest-shard then hashes node ids instead of placing these files.
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
    for index, command in enumerate(commands):
        if not isinstance(command, str):
            continue
        if _pytest_argv(command) is None and "pytest" not in command:
            continue
        instrumentable.append(index)
        discovered = discover_test_files(command, root)
        if discovered:
            shardable.append(index)
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
        budget_seconds = float(timeout_min) * 60.0
    except (TypeError, ValueError):
        budget_seconds = 0.0
    step_timings = None if not timings else timings.get(base_label(label))
    recorded_queue = load_step_queues().get(base_label(label), "")
    step_queue = str((step.get("agents") or {}).get("queue") or "")
    if step_timings is not None and recorded_queue and step_queue and recorded_queue != step_queue:
        log(f"automatic-shard: {label!r} ignores timings recorded on {recorded_queue}")
        step_timings = None
    weights = _measured_weights(files, step_timings)
    count = 1 if weights is None else plan_shard_count(weights, budget_seconds)
    measured = 0 if weights is None else len(weights)
    if count <= 1:
        if weights is None:
            reason = "no timings for this step" if timings else "no timing profile"
        else:
            reason = f"measured {measured}/{len(files)} files fit"
        log(f"automatic-shard: {label!r} stays 1 job ({reason}; budget {timeout_min} min)")
        step["commands"] = _instrument_commands(commands, instrumentable)
        return step

    step["parallelism"] = count
    if "Shard %N/%t" not in label:
        step["label"] = f"{label} · Shard %N/%t"
    rewritten: list[Any] = []
    for index, command in enumerate(commands):
        if index in shardable and isinstance(command, str) and SHARD_FLAGS not in command:
            rewritten.append(f"{command.rstrip()} {SHARD_FLAGS}")
        else:
            rewritten.append(command)
    # Instrument after shard flags so command indexes stay stable.
    step["commands"] = _instrument_commands(rewritten, instrumentable)
    log(
        f"automatic-shard: {label!r} -> {count} jobs (measured {measured}/{len(files)} files, budget {timeout_min} min)"
    )
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


def _measured_weights(files: list[str], timings: dict[str, float] | None) -> list[float] | None:
    if not timings:
        return None
    weights = [float(timings[name]) for name in files if name in timings and float(timings[name]) > 0]
    return weights or None


def _longest_bin(weights: list[float], bins: int) -> float:
    loads = [0.0] * bins
    for weight in sorted(weights, reverse=True):
        smallest = min(range(bins), key=loads.__getitem__)
        loads[smallest] += weight
    return max(loads)
