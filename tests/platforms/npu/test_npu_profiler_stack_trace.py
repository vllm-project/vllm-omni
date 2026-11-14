# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend hardware validation for the NPU profiler ``with_stack`` forwarding.

``tests/platforms/npu/test_npu_torch_profiler.py`` installs a fake
``torch_npu`` and asserts the keyword arguments that
``NPUTorchProfilerWrapper._create_profiler`` forwards. That covers the source
contract and nothing else: the module it fakes is the component the issue
reports on, so it cannot answer whether the real SDK plus the real device put
Python stacks into an exported trace.

This module is the instrument for that remaining question, and it drives the
real path (``NPUTorchProfilerWrapper`` -> ``torch_npu.profiler.profile``) in two
matched arms:

* control arm -- ``torch_profiler_with_stack=False``
* head arm    -- ``torch_profiler_with_stack=True``

Each arm exports whatever the NPU route produces and counts the Python frames
found in that artifact. The run is only accepted when the control arm stays
empty **and** the head arm carries frames. The control arm is what makes the
count mean something: without it, a frame the SDK writes unconditionally would
be indistinguishable from the feature working.

Two export schemas are counted on the union, so a schema difference is not read
as a missing feature: Kineto-style exporters emit Python frames as
``cat == "python_function"`` events named ``path/to/file.py(line): func``, while
other exporters write a ``stack`` array on the event or in ``args``.

As a test (skips on hosts without an Ascend device)::

    pytest tests/platforms/npu/test_npu_profiler_stack_trace.py -v

As a standalone instrument that leaves the matched traces plus an environment
manifest for the reviewer (run from the repo root, so ``tests`` is importable)::

    python -m tests.platforms.npu.test_npu_profiler_stack_trace --out-dir /tmp/npu-stack-8877

Neither count is admissible on its own. Before the two arms are compared, each
export is audited: it has to sit in the directory this invocation created
(``--out-dir`` may be reused, so a trace left by an earlier run can never stand
in for a fresh export), and it has to be non-empty and carry a ``traceEvents``
list. An arm that failed to export is reported as an aborted instrument, not as
a zero that the head arm can beat. The audit, the counter and the verdict are
covered by CPU-only tests at the bottom of this file, so the guard itself is
exercised on hosts without an Ascend device.

Exit status: ``0`` the discriminator held, ``1`` it did not, ``2`` the
instrument could not run or could not produce both artifacts.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import torch

try:  # a direct-path ``python <file>`` run has no package root on sys.path
    from tests.helpers.mark import hardware_marks
except ModuleNotFoundError:  # pragma: no cover - standalone instrument only
    hardware_marks = None

_MARKER = "HUELLA-INSTRUMENTO-STACK-A5-20261023"
_CONTROL_ARM = "control_with_stack_false"
_HEAD_ARM = "head_with_stack_true"
_ARM_ORDER: tuple[tuple[str, bool], ...] = ((_CONTROL_ARM, False), (_HEAD_ARM, True))
_WORKLOAD = "3 x relu(fp16 1024x1024 @ npu)"

# ``path/to/file.py(812): __enter__`` (kineto) or ``path/to/file.py:812``.
_PY_FRAME_NAME = re.compile(r"\.py\((?:line )?\d+\)|\.py:\d+")
_PY_ID_KEYS = ("Python id", "Python parent id")


def _npu_available() -> bool:
    """True only for a real device: a fake ``torch_npu`` cannot answer this."""
    npu = getattr(torch, "npu", None)
    if npu is None:
        return False
    try:
        import torch_npu  # noqa: F401

        return bool(npu.is_available())
    except Exception:
        return False


# One SKU per item, per repo convention. A3 is the Ascend SKU the CI pools are
# built on -- most ``npu`` entries under ``tests/dfx/**`` declare A3 -- so the
# probe is selected where it can actually run. The issue was reported on A5;
# ad-hoc runs there do not need the mark (the standalone CLI is SKU-agnostic),
# and the skip below still refuses to pretend on a host without the device.
#
# The marks and the device skip sit on the hardware test rather than on the
# module: the audit and verdict tests at the bottom decide on synthetic traces
# and have to keep running on a host without an NPU.
pytestmark = [pytest.mark.local_model]


def _needs_ascend_npu(func: Any) -> Any:
    """Attach the A3 hardware marks and the device skip to a single test."""
    decorated = func
    marks = hardware_marks(res={"npu": "A3"}, num_cards=1) if hardware_marks is not None else []
    for mark in marks:
        decorated = mark(decorated)
    return pytest.mark.skipif(
        not _npu_available(),
        reason="requires an Ascend NPU device: this test measures the exported trace, not a mock",
    )(decorated)


def _workload() -> None:
    """Small real NPU workload; these are the ops the profiler has to annotate."""
    device = torch.device("npu")
    left = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    right = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    for _ in range(3):
        left = torch.relu(left @ right)
    torch.npu.synchronize()


def _iter_stacks(node: Any) -> Iterator[list[Any]]:
    """Yield every non-empty ``stack`` list found anywhere in the trace structure."""
    if isinstance(node, dict):
        stack = node.get("stack")
        if isinstance(stack, list) and stack:
            yield stack
        for value in node.values():
            yield from _iter_stacks(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_stacks(item)


def _carries_python_frame(event: Any) -> bool:
    """True when one trace entry is a Python frame, in either export schema."""
    if not isinstance(event, dict):
        return False
    if event.get("cat") == "python_function":
        return True
    name = event.get("name")
    if isinstance(name, str) and _PY_FRAME_NAME.search(name):
        return True
    args = event.get("args")
    if isinstance(args, dict):
        if any(key in args for key in _PY_ID_KEYS):
            return True
        stack = args.get("stack")
        if isinstance(stack, list) and stack:
            return True
    stack = event.get("stack")
    return isinstance(stack, list) and bool(stack)


def _read_events(trace_file: Path) -> list[Any]:
    """Parse one export into its event list; raise ``ValueError`` on any defect.

    The trace is read once, so the artifact that was admitted and the artifact
    that was counted are the same bytes.
    """
    if not trace_file.is_file():
        raise ValueError(f"{trace_file.name} is not a file")
    if trace_file.stat().st_size == 0:
        raise ValueError(f"{trace_file.name} is empty")
    try:
        payload = json.loads(trace_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{trace_file.name} is unreadable: {exc}") from exc
    events = payload.get("traceEvents") if isinstance(payload, dict) else payload
    if not isinstance(events, list) or not events:
        raise ValueError(f"{trace_file.name} carries no traceEvents list")
    return events


def _measure(events: list[Any]) -> dict[str, Any]:
    """Count Python frames and ``stack`` arrays in one exported trace."""
    stack_lists = 0
    stack_frames = 0
    python_frames = 0
    samples: list[list[Any]] = []
    for event in events:
        for stack in _iter_stacks(event):
            stack_lists += 1
            stack_frames += len(stack)
            if len(samples) < 3:
                samples.append(stack[:3])
        if _carries_python_frame(event):
            python_frames += 1
            name = event.get("name") if isinstance(event, dict) else None
            if isinstance(name, str) and _PY_FRAME_NAME.search(name) and len(samples) < 3:
                samples.append([name])

    return {
        "events": len(events),
        "stack_lists": stack_lists,
        "stack_frames": stack_frames,
        "python_frames": python_frames,
        "sample_frames": samples,
    }


def _run_token() -> str:
    """Per-invocation directory name; two runs never share an export tree."""
    return f"run-{uuid.uuid4().hex}"


def _export_dir(root: Path, arm: str, run_token: str) -> Path:
    """Create the arm's export directory, refusing to inherit an older one.

    ``exist_ok=False`` is the provenance guarantee: this directory did not exist
    before the call, so every file inside it afterwards was written by this
    invocation's profiler rather than left behind in a reused ``--out-dir``.
    """
    target = root / run_token / arm
    target.mkdir(parents=True, exist_ok=False)
    return target


@dataclass(frozen=True)
class ArmArtifact:
    """Admissibility of one arm's export: the artifact, and why it was refused."""

    path: Path | None
    valid: bool
    error: str | None = None
    events: list[Any] | None = field(default=None, repr=False, compare=False)


def _audit_artifact(export_dir: Path, run_token: str) -> ArmArtifact:
    """Admit an arm's export only if this invocation produced it and it parses."""
    if run_token not in export_dir.parts:
        return ArmArtifact(
            None,
            False,
            f"{export_dir} is not scoped to run {run_token}: a trace found there was not exported by this "
            "invocation, so it cannot serve as this arm's artifact",
        )
    candidates = [path for path in export_dir.rglob("*.json") if path.is_file()]
    if not candidates:
        return ArmArtifact(None, False, f"no trace was exported to {export_dir}")
    trace_file = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    try:
        events = _read_events(trace_file)
    except (OSError, ValueError) as exc:
        return ArmArtifact(trace_file, False, str(exc))
    return ArmArtifact(trace_file, True, events=events)


def _audit_and_measure(export_dir: Path, *, with_stack: bool, run_token: str) -> dict[str, Any]:
    """Admit and count one arm's export; a refused artifact reports zero frames."""
    artifact = _audit_artifact(export_dir, run_token)
    measurement: dict[str, Any] = {
        "with_stack": with_stack,
        "trace_dir": str(export_dir),
        "artifact_valid": artifact.valid,
        "artifact_error": artifact.error,
        "trace_file": str(artifact.path) if artifact.path is not None else None,
        "events": 0,
        "stack_lists": 0,
        "stack_frames": 0,
        "python_frames": 0,
        "sample_frames": [],
    }
    if artifact.valid and artifact.events is not None:
        measurement.update(_measure(artifact.events))
    return measurement


def _profile_arm(export_dir: Path, *, with_stack: bool, run_token: str) -> dict[str, Any]:
    """Run the real NPU profiler route once, then admit and measure its export."""
    from vllm.config import ProfilerConfig

    from vllm_omni.platforms.npu.profiler import NPUTorchProfilerWrapper

    config = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(export_dir),
        torch_profiler_with_stack=with_stack,
        torch_profiler_use_gzip=False,
    )
    profiler = NPUTorchProfilerWrapper(
        config,
        worker_name="stack_trace_probe",
        local_rank=0,
    ).profiler

    profiler.start()
    try:
        _workload()
    finally:
        profiler.stop()
        torch.npu.synchronize()

    return _audit_and_measure(export_dir, with_stack=with_stack, run_token=run_token)


def _manifest() -> dict[str, str]:
    import torch_npu

    try:
        device_name = torch.npu.get_device_name(0)
    except Exception:
        device_name = "unknown"
    return {
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", "unknown"),
        "cann": getattr(getattr(torch, "version", None), "cann", "unknown"),
        "device": device_name,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "vllm_omni_commit": _git_commit(),
        "workload": _WORKLOAD,
        "probe": _MARKER,
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class Verdict:
    """Outcome of the matched probe: ``pass``, ``fail`` or ``abort``."""

    passed: bool
    status: str
    reason: str


def _verdict(arms: dict[str, dict[str, Any]]) -> Verdict:
    """Decide from both arms; a missing artifact is an abort, never a zero."""
    for name in (_CONTROL_ARM, _HEAD_ARM):
        arm = arms[name]
        if not arm["artifact_valid"]:
            return Verdict(
                False,
                "abort",
                f"{name} has no admissible trace artifact ({arm['artifact_error']}): an absent export is not an "
                "arm that stayed empty, so this run cannot claim the discriminator held.",
            )

    control = arms[_CONTROL_ARM]
    head = arms[_HEAD_ARM]

    if control["python_frames"] != 0:
        return Verdict(
            False,
            "fail",
            "control arm is contaminated: with torch_profiler_with_stack=False the exported trace already carries "
            f"{control['python_frames']} python frame(s), so this count cannot discriminate the fix. Treat the "
            "instrument as invalid and compare both artifacts in the Ascend trace viewer instead.",
        )
    if head["python_frames"] == 0:
        return Verdict(
            False,
            "fail",
            "head arm carries no python frame "
            f"({head['stack_lists']} stack array(s), {head['stack_frames']} frame(s) total): the flag reaches "
            "torch_npu but the exported trace still has no Python stack. The artifact-level claim of #8877 does "
            "not hold yet.",
        )
    return Verdict(
        True,
        "pass",
        f"head arm carries {head['python_frames']} python frame(s) "
        f"({head['stack_lists']} stack array(s), {head['stack_frames']} frame(s) total); control arm carries none.",
    )


@_needs_ascend_npu
def test_npu_trace_carries_python_stacks_only_when_requested(tmp_path: Path) -> None:
    run_token = _run_token()
    arms = {
        name: _profile_arm(_export_dir(tmp_path, name, run_token), with_stack=with_stack, run_token=run_token)
        for name, with_stack in _ARM_ORDER
    }
    verdict = _verdict(arms)
    detail = f"{verdict.reason}\ncontrol={arms[_CONTROL_ARM]}\nhead={arms[_HEAD_ARM]}"
    assert verdict.status != "abort", detail
    assert verdict.passed, detail


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Matched-arm validation of torch_profiler_with_stack on an Ascend NPU.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where the matched traces and manifest are written (default: a temporary directory).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not _npu_available():
        print(
            "CANNOT RUN: no Ascend NPU device. This instrument measures the real SDK and device; "
            "a mock cannot stand in for them.",
            file=sys.stderr,
        )
        return 2

    out_dir: Path = args.out_dir or Path(tempfile.mkdtemp(prefix="npu-stack-8877-"))
    run_token = _run_token()
    run_dir = out_dir / run_token
    # A reused ``--out-dir`` must not leak an older export into this run: the
    # token directory is created here and cannot already exist.
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest = _manifest()
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    arms: dict[str, dict[str, Any]] = {}
    for name, with_stack in _ARM_ORDER:
        export_dir = _export_dir(out_dir, name, run_token)
        arms[name] = _profile_arm(export_dir, with_stack=with_stack, run_token=run_token)
        (export_dir / "measurement.json").write_text(json.dumps(arms[name], indent=2), encoding="utf-8")

    verdict = _verdict(arms)
    report = {
        "manifest": manifest,
        "run_token": run_token,
        "arms": arms,
        "passed": verdict.passed,
        "status": verdict.status,
        "reason": verdict.reason,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    if verdict.passed:
        print("PASS: " + verdict.reason)
    elif verdict.status == "abort":
        print("ABORT: " + verdict.reason, file=sys.stderr)
    else:
        print("FAIL: " + verdict.reason)
    print(f"artifacts: {run_dir}")

    if verdict.status == "abort":
        return 2
    return 0 if verdict.passed else 1


# --- CPU-level falsifiers for the instrument's own guard --------------------
#
# The probe above cannot run here; its guard can, and has to. An instrument whose
# rejection paths are never exercised is the exact failure it was written to
# catch. These build traces with the schema Kineto exports, so the audit, the
# counter and the verdict are all decided on a host without an NPU.


def _write_trace(export_dir: Path, *, python_frames: int, name: str = "trace_view.json") -> Path:
    """Write a synthetic export carrying ``python_frames`` Kineto Python frames."""
    export_dir.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, Any]] = [{"ph": "X", "cat": "cpu_op", "name": "aten::relu", "args": {}}]
    events.extend(
        {
            "ph": "X",
            "cat": "python_function",
            "name": f"workload.py({index}): forward",
            "args": {"Python id": index, "Python parent id": index - 1},
        }
        for index in range(python_frames)
    )
    trace_file = export_dir / name
    trace_file.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
    return trace_file


def _arm(export_dir: Path, *, with_stack: bool, run_token: str) -> dict[str, Any]:
    """The admission-plus-measurement path the hardware test drives, minus the device."""
    return _audit_and_measure(export_dir, with_stack=with_stack, run_token=run_token)


def test_export_dir_refuses_to_reuse_a_previous_invocation(tmp_path: Path) -> None:
    run_token = _run_token()
    _export_dir(tmp_path, _CONTROL_ARM, run_token)
    with pytest.raises(FileExistsError):
        _export_dir(tmp_path, _CONTROL_ARM, run_token)


def test_audit_rejects_a_control_trace_left_by_an_earlier_run(tmp_path: Path) -> None:
    stale_dir = tmp_path / "run-previous" / _CONTROL_ARM
    _write_trace(stale_dir, python_frames=0)
    artifact = _audit_artifact(stale_dir, _run_token())
    assert artifact.valid is False
    assert artifact.path is None
    assert "is not scoped to run" in (artifact.error or "")


def test_audit_rejects_an_empty_or_unparsable_artifact(tmp_path: Path) -> None:
    run_token = _run_token()
    empty_dir = _export_dir(tmp_path, "empty", run_token)
    (empty_dir / "trace_view.json").write_text("", encoding="utf-8")
    empty = _audit_artifact(empty_dir, run_token)
    assert empty.valid is False
    assert "is empty" in (empty.error or "")

    broken_dir = _export_dir(tmp_path, "broken", run_token)
    (broken_dir / "trace_view.json").write_text("{not json", encoding="utf-8")
    broken = _audit_artifact(broken_dir, run_token)
    assert broken.valid is False
    assert "unreadable" in (broken.error or "")

    silent_dir = _export_dir(tmp_path, "silent", run_token)
    assert _audit_artifact(silent_dir, run_token).valid is False


def test_verdict_aborts_when_the_control_exported_nothing(tmp_path: Path) -> None:
    run_token = _run_token()
    control_dir = _export_dir(tmp_path, _CONTROL_ARM, run_token)
    head_dir = _export_dir(tmp_path, _HEAD_ARM, run_token)
    _write_trace(head_dir, python_frames=7)

    arms = {
        _CONTROL_ARM: _arm(control_dir, with_stack=False, run_token=run_token),
        _HEAD_ARM: _arm(head_dir, with_stack=True, run_token=run_token),
    }
    verdict = _verdict(arms)
    assert verdict.passed is False
    assert verdict.status == "abort", verdict.reason
    assert _CONTROL_ARM in verdict.reason


def test_verdict_aborts_when_the_head_exported_nothing(tmp_path: Path) -> None:
    run_token = _run_token()
    control_dir = _export_dir(tmp_path, _CONTROL_ARM, run_token)
    head_dir = _export_dir(tmp_path, _HEAD_ARM, run_token)
    _write_trace(control_dir, python_frames=0)

    arms = {
        _CONTROL_ARM: _arm(control_dir, with_stack=False, run_token=run_token),
        _HEAD_ARM: _arm(head_dir, with_stack=True, run_token=run_token),
    }
    verdict = _verdict(arms)
    assert verdict.passed is False
    assert verdict.status == "abort", verdict.reason
    assert _HEAD_ARM in verdict.reason


def test_verdict_aborts_when_only_a_stale_control_trace_exists(tmp_path: Path) -> None:
    run_token = _run_token()
    stale_dir = tmp_path / "run-previous" / _CONTROL_ARM
    _write_trace(stale_dir, python_frames=0)
    head_dir = _export_dir(tmp_path, _HEAD_ARM, run_token)
    _write_trace(head_dir, python_frames=9)

    arms = {
        _CONTROL_ARM: _arm(stale_dir, with_stack=False, run_token=run_token),
        _HEAD_ARM: _arm(head_dir, with_stack=True, run_token=run_token),
    }
    verdict = _verdict(arms)
    assert verdict.passed is False, "a trace from an earlier run cannot stand in for the control arm"
    assert verdict.status == "abort", verdict.reason


def test_verdict_fails_when_the_control_carries_frames(tmp_path: Path) -> None:
    run_token = _run_token()
    control_dir = _export_dir(tmp_path, _CONTROL_ARM, run_token)
    head_dir = _export_dir(tmp_path, _HEAD_ARM, run_token)
    _write_trace(control_dir, python_frames=4)
    _write_trace(head_dir, python_frames=15)

    arms = {
        _CONTROL_ARM: _arm(control_dir, with_stack=False, run_token=run_token),
        _HEAD_ARM: _arm(head_dir, with_stack=True, run_token=run_token),
    }
    verdict = _verdict(arms)
    assert verdict.passed is False
    assert verdict.status == "fail", verdict.reason
    assert "contaminated" in verdict.reason


def test_verdict_passes_on_a_fresh_control_zero_head_positive_pair(tmp_path: Path) -> None:
    run_token = _run_token()
    control_dir = _export_dir(tmp_path, _CONTROL_ARM, run_token)
    head_dir = _export_dir(tmp_path, _HEAD_ARM, run_token)
    _write_trace(control_dir, python_frames=0)
    _write_trace(head_dir, python_frames=15)

    arms = {
        _CONTROL_ARM: _arm(control_dir, with_stack=False, run_token=run_token),
        _HEAD_ARM: _arm(head_dir, with_stack=True, run_token=run_token),
    }
    verdict = _verdict(arms)
    assert verdict.passed is True, verdict.reason
    assert verdict.status == "pass"
    assert arms[_CONTROL_ARM]["python_frames"] == 0
    assert arms[_HEAD_ARM]["python_frames"] == 15


if __name__ == "__main__":
    raise SystemExit(main())
