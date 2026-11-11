# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Choose a Buildkite shard count from measured durations.

``automatic_shard: true`` does not reserve a pool. The uploader keeps one job
unless this step's own timings show its measured files would miss
``timeout_in_minutes``. It then packs those files with a greedy
longest-processing-time assignment, never more than ``MAX_AUTOMATIC_SHARDS``,
and each shard runs only the files in its bin. The packing is not the smallest
possible. Unmeasured files add no weight, so a missing profile stays one job.
A finished session's unmeasured files were filtered by ``-m`` and stay at
weight 0; they are still placed in a bin so a new file keeps running.

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
PLUGIN_MODULE = "pytest_file_timings"
PLUGIN_FLAG = f"-p {PLUGIN_MODULE}"
SCRIPTS_DIR = ".buildkite/common/scripts"
# `$$` is required. Buildkite interpolates `$` before the shell runs, and
# `${PYTHONPATH:+...}` is parsed as an offset, which rejects `+`.
PYTHONPATH_EXPORT = f'export PYTHONPATH="{SCRIPTS_DIR}$${{PYTHONPATH:+:$$PYTHONPATH}}"'
ARTIFACT_PATHS = ["timings/jobs/*.json"]
_SHARD_SUFFIX = re.compile(r" · Shard(?: %N/%t| \d+/\d+)$")
_SHELL_EXPAND_ARG = re.compile(r"^(?:--(?:num-shards|shard-id)=)?\${1,2}[A-Za-z_][A-Za-z0-9_]*$")
_SHARD_FILE_ARG = '"$${_shard_files[@]}"'
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


def load_incomplete_steps() -> dict[str, dict[str, Any]]:
    """TEMPORARY: delete after the CI check.

    Keyed by step label. ``selected_files`` is the marker selection, or None
    when the killed job's artifact did not record it. Absent labels finished.
    """
    payload = _profile_payload()
    raw = payload.get("incomplete_steps") if isinstance(payload, dict) else None
    if not isinstance(raw, dict):
        return {}
    parsed: dict[str, dict[str, Any]] = {}
    for label, info in raw.items():
        if not isinstance(info, dict) or "selected_files" not in info:
            continue
        selected = info.get("selected_files")
        if selected is None:
            parsed[base_label(str(label))] = {"selected_files": None}
            continue
        if isinstance(selected, list):
            parsed[base_label(str(label))] = {"selected_files": [str(item).replace("\\", "/") for item in selected]}
    return parsed


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
    planned = _plan_weights(files, step_timings, load_incomplete_steps().get(base_label(label)))
    if planned is None:
        reason = "no timings for this step" if timings else "no timing profile"
        log(f"automatic-shard: {label!r} stays 1 job ({reason}; budget {timeout_min} min)")
        step["commands"] = _instrument_commands(commands, instrumentable)
        return step
    weights, measured, estimated = planned
    bins = pack_files(weights, budget_seconds)
    if len(bins) <= 1:
        reason = f"measured {measured}/{len(files)} files fit"
        log(f"automatic-shard: {label!r} stays 1 job ({reason}; budget {timeout_min} min)")
        step["commands"] = _instrument_commands(commands, instrumentable)
        return step

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
    estimate = f", temporary estimate {estimated} unfinished" if estimated else ""
    log(
        f"automatic-shard: {label!r} -> {len(bins)} jobs "
        f"(measured {measured}/{len(files)} files{estimate}, budget {timeout_min} min)"
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


def _plan_weights(
    files: list[str],
    timings: dict[str, float] | None,
    incomplete: dict[str, Any] | None,
) -> tuple[dict[str, float], int, int] | None:
    """Weight for every discovered file, plus measured and estimated counts.

    A file with no duration stays at 0. TEMPORARY: when ``incomplete`` is set,
    selected files that never finished take the mean of the files that did.
    ``selected_files is None`` means the killed artifact did not record the
    marker selection, so every unmeasured discovered file is estimated.
    Delete this estimate with ``incomplete_steps``.
    """
    measured_list = _measured_weights(files, timings)
    if measured_list is None or timings is None:
        return None
    measured = {name: float(timings[name]) for name in files if name in timings and float(timings[name]) > 0}
    weights = {name: measured.get(name, 0.0) for name in files}
    estimated = 0
    if incomplete is not None and measured:
        selected = incomplete.get("selected_files")
        if selected is None:
            pending = [name for name in files if name not in measured]
        else:
            chosen = {str(name).replace("\\", "/") for name in selected}
            pending = [name for name in files if name in chosen and name not in measured]
        if pending:
            mean = sum(measured.values()) / len(measured)
            for name in pending:
                weights[name] = mean
            estimated = len(pending)
    return weights, len(measured), estimated


def _shard_slug(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", base_label(label).lower()).strip("-")
    return slug[:80] or "step"


def _write_file_shards(root: Path, slug: str, bins: list[list[str]]) -> None:
    dest = root / FILE_SHARD_ROOT / slug
    if dest.exists():
        for old in dest.glob("*.txt"):
            old.unlink()
    dest.mkdir(parents=True, exist_ok=True)
    for index, names in enumerate(bins):
        (dest / f"{index}.txt").write_text("".join(f"{name}\n" for name in names), encoding="utf-8")


def _file_shard_command(command: str, slug: str) -> str:
    """Run one packed bin. ``$$`` survives Buildkite interpolation."""
    argv = _pytest_argv(command)
    if argv is None:
        return command
    kept = [argv[0]]
    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg in _VALUE_OPTIONS:
            if index + 1 < len(argv):
                kept.extend((arg, argv[index + 1]))
            else:
                kept.append(arg)
            index += 2
            continue
        if arg.startswith("-"):
            kept.append(arg)
            index += 1
            continue
        index += 1
    kept.append(_SHARD_FILE_ARG)
    pytest_line = _shell_join(kept)
    return "\n".join(
        (
            "set -eu",
            f'shard_list="{FILE_SHARD_ROOT.as_posix()}/{slug}/$${{BUILDKITE_PARALLEL_JOB:-0}}.txt"',
            'buildkite-agent artifact download "$$shard_list" .',
            'mapfile -t _shard_files < "$$shard_list"',
            'test "$${#_shard_files[@]}" -gt 0',
            pytest_line,
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
        if arg == _SHARD_FILE_ARG or _SHELL_EXPAND_ARG.fullmatch(arg):
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
