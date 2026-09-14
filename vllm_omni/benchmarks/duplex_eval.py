# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Omni-DuplexEval generation and artifacts for serving benchmarks.

The serving backend only *generates* samples inside the timed benchmark window.
Judging is deliberately absent from this module (v2 design §5.2): the DFX runner
performs Phase 2 (evaluate) and Phase 3 (summarize + merge) in separate
sub-processes after the omni server has exited, then merges the score summary
back into the perf result through
:func:`merge_duplex_eval_into_result` (the only writer, guarded by the
``phase == "generate"`` assertion).

Generation reuses
:func:`vllm_omni.benchmarks.duplex.omni_duplex_eval_runner.generate_sample`;
no per-sample inline judging ever happens inside the timed window.
"""

from __future__ import annotations

import contextlib
import copy
import fcntl
import hashlib
import json
import logging
import os
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm_omni.benchmarks.data_modules.duplex_eval_dataset import (
    DuplexEvalSampleRequest,
    DuplexEvalSessionOptions,
    normalize_exclude_ids,
)
from vllm_omni.benchmarks.duplex.omni_duplex_eval_dataset import DuplexSample
from vllm_omni.benchmarks.duplex.omni_duplex_eval_runner import generate_sample

BATCH_ARTIFACTS = ("batch_summary.json", "eval_manifest.jsonl")
ARTIFACT_LOCK_FILE = ".duplex_eval.lock"
_MERGE_PROVENANCE_SOURCE = "tests/dfx/perf/scripts/run_benchmark.py"
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config / result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DuplexEvalRequestConfig:
    """Thin per-request wrapper; generation layout lives in ``options``."""

    model: str
    url: str
    options: DuplexEvalSessionOptions


@dataclass
class DuplexEvalCaseResult:
    """Lifecycle signals of one generated sample (no judge fields in v2)."""

    id: str
    split: str
    family: str
    task_type: str | None = None
    response_path: str = ""
    success: bool = False
    error: str = ""
    latency_s: float = 0.0
    response_done: bool | None = None
    drain_timeout: str | None = None
    close_timeout: str | None = None
    clock: str = ""

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(vars(self))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


async def run_duplex_eval_case(
    sample: DuplexSample,
    config: DuplexEvalRequestConfig,
) -> DuplexEvalCaseResult:
    """Generate one sample and report its lifecycle signals.

    Generation only; judging is deferred to the DFX runner so the timed
    benchmark window never includes judge calls (v2 design §5.2).
    """
    options = config.options
    if not options.ref_audio:
        raise ValueError("ref_audio is required for MiniCPM-o native-duplex audio output")
    if not config.url:
        raise ValueError("a Realtime WebSocket url is required for Omni-DuplexEval generation")
    result = DuplexEvalCaseResult(id=sample.id, split=sample.split, family=sample.family, task_type=sample.task_type)
    started_at = time.monotonic()
    try:
        output = await generate_sample(
            sample,
            url=config.url,
            model=config.model,
            ref_audio=options.ref_audio,
            output_root=options.response_root,
            fps=options.fps,
            mix=options.mix,
            pace=options.pace,
            clock=options.clock,
            overwrite=options.overwrite,
            unit_ms=options.unit_ms,
        )
        result.response_path = str(output)
        meta = _read_meta(output)
        result.response_done = bool(meta.get("response_done"))
        result.drain_timeout = meta.get("drain_timeout")
        result.close_timeout = meta.get("close_timeout")
        result.clock = str(meta.get("clock", ""))
        result.success = result.response_done and result.drain_timeout is None and result.close_timeout is None
        if not result.success:
            result.error = _lifecycle_error(result)
    except Exception as exc:  # noqa: BLE001 - a failed sample must not abort the batch
        result.success = False
        result.error = str(exc)
    result.latency_s = max(0.0, time.monotonic() - started_at)
    return result


def _read_meta(output: Path) -> dict[str, Any]:
    meta_path = output.with_name(output.stem + ".meta.json")
    if not meta_path.exists():
        return {}
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _lifecycle_error(result: DuplexEvalCaseResult) -> str:
    if not result.response_done:
        return f"response.done was not observed (drain_timeout={result.drain_timeout!r})"
    if result.drain_timeout is not None:
        return f"drain timed out: {result.drain_timeout}"
    if result.close_timeout is not None:
        return f"close timed out: {result.close_timeout}"
    return ""


# ---------------------------------------------------------------------------
# Generation ledger
# ---------------------------------------------------------------------------


def generation_summary(results: list[DuplexEvalCaseResult]) -> dict[str, Any]:
    """Compact generation counts (v2: no ``scored`` / ``score_summary``)."""
    total = len(results)
    generated = sum(result.success for result in results)
    return {
        "total": total,
        "generated": generated,
        "generation_failed": total - generated,
        "failure_ids": [f"{result.split}/{result.id}" for result in results if not result.success],
    }


def _options_from_requests(
    input_requests: list[Any],
) -> DuplexEvalSessionOptions | None:
    for request in input_requests:
        if isinstance(request, DuplexEvalSampleRequest) and request.duplex_eval_options is not None:
            return request.duplex_eval_options
    return None


def _clock_summary(
    results: list[DuplexEvalCaseResult],
) -> tuple[str, list[str]]:
    observed: Counter[str] = Counter()
    for result in results:
        if result.clock:
            observed[result.clock] += 1
    clock = observed.most_common(1)[0][0] if observed else ""
    mismatch_ids = [f"{result.split}/{result.id}" for result in results if result.clock and result.clock != "media"]
    return clock, mismatch_ids


def _ref_audio_sha256(path: str) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Batch artifacts
# ---------------------------------------------------------------------------


def _manifest_row(sample: DuplexSample, result: DuplexEvalCaseResult) -> dict[str, Any]:
    return {
        "id": sample.id,
        "split": sample.split,
        "family": sample.family,
        "task_type": sample.task_type,
        "response": result.response_path,
        "generated": result.success,
        "response_done": result.response_done,
        "drain_timeout": result.drain_timeout,
        "close_timeout": result.close_timeout,
        "clock": result.clock,
        "error": result.error,
    }


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_batch_artifacts(
    score_dir: str | Path,
    samples: list[DuplexSample],
    results: list[DuplexEvalCaseResult],
    summary: dict[str, Any],
) -> None:
    manifest = [_manifest_row(sample, result) for sample, result in zip(samples, results, strict=True)]
    _atomic_write_json(Path(score_dir) / "batch_summary.json", summary)
    _atomic_write_text(
        Path(score_dir) / "eval_manifest.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest),
    )


def clear_batch_artifacts(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in BATCH_ARTIFACTS:
        (root / name).unlink(missing_ok=True)


@contextlib.contextmanager
def duplex_eval_output_lock(root: Path) -> Iterator[None]:
    """Serialize runs that share one batch-artifact root."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ARTIFACT_LOCK_FILE).open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def rows_from_outputs(
    input_requests: list[Any],
    outputs: list[Any],
) -> tuple[list[DuplexSample], list[DuplexEvalCaseResult], list[Any]]:
    """Pair dataset samples with their executed outputs, preserving order."""
    samples: list[DuplexSample] = []
    results: list[DuplexEvalCaseResult] = []
    paired_outputs: list[Any] = []
    for sample, output in zip(input_requests, outputs, strict=True):
        if not isinstance(sample, DuplexEvalSampleRequest) or sample.duplex_eval_sample is None:
            continue
        result = getattr(output, "duplex_eval_case_result", None)
        if not isinstance(result, DuplexEvalCaseResult):
            continue
        samples.append(sample.duplex_eval_sample)
        results.append(result)
        paired_outputs.append(output)
    return samples, results, paired_outputs


def finalize_duplex_eval_batch(
    input_requests: list[Any],
    outputs: list[Any],
) -> dict[str, Any] | None:
    """Window-external wrap-up (v2: generation-ledger only, never calls a judge).

    1) collect ``duplex_eval_case_result`` from the executed outputs;
    2) compute ``total`` / ``generated`` / ``generation_failed`` / ``failure_ids``;
    3) aggregate ``clock`` and list ``clock_mismatch_ids``;
    4) echo the effective ``exclude_ids`` and ``selected_ids``;
    5) write ``<score_dir>/batch_summary.json`` + ``eval_manifest.jsonl``.

    Returns a compact summary (without ``results[]``) for
    ``result["duplex_eval"]``; ``None`` when no duplex-eval sample ran.
    """
    samples, results, _ = rows_from_outputs(input_requests, outputs)
    if not results:
        return None
    options = _options_from_requests(input_requests)
    if options is None:
        raise ValueError("duplex_eval results present but session options are missing")

    summary = generation_summary(results)
    clock, clock_mismatch_ids = _clock_summary(results)
    summary.update(
        {
            "phase": "generate",
            "clock": clock,
            "clock_mismatch_ids": clock_mismatch_ids,
            "pace": options.pace,
            "fps": options.fps,
            "mix": options.mix,
            "unit_ms": options.unit_ms,
            "overwrite": options.overwrite,
            "response_root": str(options.response_root),
            "score_dir": str(options.score_dir),
            "exclude_ids": sorted(normalize_exclude_ids(options.exclude_ids)),
            "selected_ids": [f"{sample.split}/{sample.id}" for sample in samples],
            "judge_enabled": False,
            "scored": 0,
            "score_failed": 0,
            "ref_audio_sha256": _ref_audio_sha256(options.ref_audio),
        }
    )
    if options.write_artifacts:
        try:
            write_batch_artifacts(options.score_dir, samples, results, summary)
        except OSError as exc:  # noqa: BLE001 - preserve benchmark metrics
            summary["artifacts_complete"] = False
            summary["artifact_errors"] = [str(exc)]
            logger.exception("Omni-DuplexEval batch artifact publication failed: %s", exc)
        else:
            summary["artifacts_complete"] = True
    else:
        summary["artifacts_complete"] = False
        summary["artifact_errors"] = ["batch artifacts disabled by --duplex-eval-no-artifacts"]
    return summary


# ---------------------------------------------------------------------------
# CLI translation
# ---------------------------------------------------------------------------


def options_from_args(args: Any) -> DuplexEvalSessionOptions:
    """Translate parsed CLI arguments into the shared session options (v2)."""
    return DuplexEvalSessionOptions(
        response_root=Path(getattr(args, "duplex_eval_output_dir", "duplex-eval-responses")),
        score_dir=Path(getattr(args, "duplex_eval_score_dir", "duplex-eval-scores")),
        ref_audio=str(getattr(args, "duplex_eval_ref_audio", "") or ""),
        fps=float(getattr(args, "duplex_eval_fps", 1.0)),
        mix=str(getattr(args, "duplex_eval_mix", "question")),
        pace=str(getattr(args, "duplex_eval_pace", "realtime")),
        clock=str(getattr(args, "duplex_eval_clock", "media")),
        unit_ms=int(getattr(args, "duplex_eval_unit_ms", 1000)),
        overwrite=bool(getattr(args, "duplex_eval_overwrite", False)),
        exclude_ids=tuple(getattr(args, "duplex_eval_exclude_ids", None) or ()),
        write_artifacts=not bool(getattr(args, "duplex_eval_no_artifacts", False)),
    )


# ---------------------------------------------------------------------------
# Result-file location and Phase-3 merge (used by the DFX runner)
# ---------------------------------------------------------------------------


def _locate_result_file(
    *,
    bench_dir: str | Path,
    test_name: str,
    dataset_name: str,
    flow: str | int,
    num_prompt: int,
    since: float,
) -> Path:
    """Locate this run's perf result by filename prefix + mtime uniqueness.

    The file name embeds a ``%Y%m%d-%H%M%S`` timestamp (conftest), so it
    cannot be reconstructed by path joining; use prefix glob +
    ``mtime >= since`` and demand exactly one hit.
    """
    pattern = f"result_{test_name}_*{dataset_name}_{flow}_{num_prompt}_in*_out*.json"
    hits = [p for p in Path(bench_dir).glob(pattern) if p.stat().st_mtime >= since - 1.0]
    if len(hits) != 1:
        raise AssertionError(
            f"expected exactly one perf result for {test_name}/{flow}/"
            f"{num_prompt}, got {[str(p) for p in hits]} under {bench_dir}"
        )
    return hits[0]


def _pending_ids_from_manifest(score_dir: str | Path) -> list[str]:
    """Generated-but-not-scored ``split/id`` keys (troubleshooting)."""
    manifest = Path(score_dir) / "eval_manifest.jsonl"
    if not manifest.exists():
        return []
    pending: list[str] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not row.get("generated"):
            continue
        split, sample_id = row.get("split"), row.get("id")
        if split and sample_id:
            score_path = Path(score_dir) / str(split) / f"{sample_id}.json"
            if not score_path.exists():
                pending.append(f"{split}/{sample_id}")
    return pending


def merge_duplex_eval_into_result(
    result_path: Path,
    *,
    score_summary: dict[str, Any],
    judge_meta: dict[str, Any],
    phases: list[str],
) -> dict[str, Any]:
    """Merge the Phase-3 score summary into the perf result (unique writer).

    - asserts ``duplex_eval.phase == "generate"`` (W2) so a second merge is
      refused instead of silently drifting;
    - only adds/overwrites the ``duplex_eval`` subtree, leaving ``completed`` /
      ``e2el`` / ``Hardware`` / ``baseline`` untouched;
    - replaces the file atomically via ``tmp`` + :func:`os.replace` (W3).
    """
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    duplex = payload.get("duplex_eval")
    if not isinstance(duplex, dict):
        raise AssertionError(f"duplex_eval side-channel missing in {result_path}")
    if duplex.get("phase") != "generate":
        raise AssertionError(f"refusing to merge: duplex_eval.phase={duplex.get('phase')!r}")

    duplex["phase"] = "generate+evaluate+summarize"
    duplex["phases"] = phases
    duplex["judge"] = judge_meta
    duplex["judge_enabled"] = True
    duplex["score_summary"] = score_summary
    duplex["scored"] = int(score_summary.get("samples", 0) or 0)
    duplex["score_failed"] = max(0, int(duplex.get("generated", 0) or 0) - duplex["scored"])
    duplex["pending_ids"] = _pending_ids_from_manifest(str(duplex.get("score_dir", "duplex-eval-scores")))
    duplex["merge_provenance"] = {
        "merged_by": _MERGE_PROVENANCE_SOURCE,
        "merged_at_unix": time.time(),
        "score_root": str(duplex.get("score_dir", "")),
        "summary_source": "score_summary.json",
    }

    tmp = result_path.with_suffix(result_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, result_path)
    return payload


__all__ = [
    "BATCH_ARTIFACTS",
    "ARTIFACT_LOCK_FILE",
    "DuplexEvalCaseResult",
    "DuplexEvalRequestConfig",
    "clear_batch_artifacts",
    "duplex_eval_output_lock",
    "finalize_duplex_eval_batch",
    "generation_summary",
    "merge_duplex_eval_into_result",
    "options_from_args",
    "rows_from_outputs",
    "run_duplex_eval_case",
    "write_batch_artifacts",
]
