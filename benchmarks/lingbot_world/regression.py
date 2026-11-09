# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fail-closed per-run regression checks and reference baselines for real LingBot rollouts.

The baseline uses independent starts of a frozen reference revision with a
matching hardware and dependency contract.
Neither a different accelerator topology nor another precision policy may reuse
it. Reference limits are computed from the current paired measurements, never from candidate scores.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vllm_omni.benchmarks.realtime_video.metrics import (
    RegressionError as RegressionError,
)
from vllm_omni.benchmarks.realtime_video.metrics import (
    number as number,
)
from vllm_omni.benchmarks.realtime_video.metrics import (
    performance_metrics as performance_metrics,
)


def read_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise RegressionError(f"Nonfinite JSON constant in {path}: {value}")

    value = json.loads(path.read_text(), parse_constant=reject)
    if not isinstance(value, dict):
        raise RegressionError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contract_digest(contract: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(contract, sort_keys=True, allow_nan=False).encode()).hexdigest()


def verify_environment(python: Path, requirements: Path) -> dict[str, str]:
    """Check the evaluator interpreter, including every resolved dependency."""
    expected = {}
    for line in requirements.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, version = line.partition("==")
        if not separator or not name or not version:
            raise RegressionError(f"Evaluator dependency is not exactly pinned: {line}")
        expected[name] = version
    if not expected:
        raise RegressionError("Empty evaluator environment lock")
    command = (
        "import importlib.metadata as m,json,sys; print(json.dumps({p:m.version(p) for p in json.loads(sys.argv[1])}))"
    )
    result = subprocess.run(
        [str(python), "-c", command, json.dumps(list(expected))], capture_output=True, text=True, timeout=30, check=True
    )
    actual = json.loads(result.stdout)
    if actual != expected:
        raise RegressionError("Evaluator environment differs from requirements/lingbot-quality.txt")
    return actual


def calibrate(
    reference: Sequence[Mapping[str, float]],
    *,
    contract: Mapping[str, Any],
    reference_revision: str,
    max_cv: float = 0.03,
    regression_fraction: float = 0.05,
    transient_noise_sigma: float = 3.0,
) -> dict[str, Any]:
    """Freeze limits from >=3 independent reference server starts.

    Unstable cadence rejects calibration. The worst reference value receives
    a regression allowance; transient jitter is derived from these
    same independent references only.
    """
    if len(reference) < 3:
        raise RegressionError("Calibration needs three independent server starts")
    if len(reference_revision) != 40 or any(c not in "0123456789abcdef" for c in reference_revision):
        raise RegressionError("Freeze a full reference commit SHA")
    for label, value in (("max_cv", max_cv), ("regression_fraction", regression_fraction)):
        number(value, label, positive=True)
        if value >= 1:
            raise RegressionError(f"{label} must be below one")
    number(transient_noise_sigma, "transient_noise_sigma")
    if transient_noise_sigma > 3:
        raise RegressionError("Transient noise allowance cannot exceed three reference standard deviations")
    limits = {}
    metrics = ["mean_ms", "p50_ms", "p95_ms", "ttfc_ms", "generated_fps", "stall_ms"]
    for metric in metrics:
        values = [number(row[metric], metric) for row in reference]
        mean = statistics.mean(values)
        cv = statistics.pstdev(values) / mean if mean else 0.0
        # Startup and discrete playback stalls have measured reference jitter,
        # unlike stationary cadence. Calibrate their allowance from references
        # only; never use a failed candidate to raise a limit.
        cv_ceiling = max_cv
        if metric in {"mean_ms", "p50_ms", "p95_ms"} and cv > cv_ceiling:
            raise RegressionError(f"Unstable reference {metric}: CV={cv:.3%} > {cv_ceiling:.3%}")
        lower_is_better = metric != "generated_fps"
        worst = max(values) if lower_is_better else min(values)
        relative_allowance = worst * regression_fraction
        jitter_allowance = (
            statistics.pstdev(values) * transient_noise_sigma
            if metric in {"ttfc_ms", "stall_ms", "steady_stall_ms"}
            else 0.0
        )
        allowance = max(relative_allowance, jitter_allowance)
        limits[metric] = {
            "direction": "max" if lower_is_better else "min",
            "value": worst + allowance if lower_is_better else worst - allowance,
            "reference": values,
            "cv": cv,
            "relative_allowance": relative_allowance,
            "reference_jitter_allowance": jitter_allowance,
        }
    if all(row["steady_samples"] >= 1000 for row in reference):
        limits["p99_ms"] = {
            "direction": "max",
            "value": max(number(row["p99_ms"], "p99") for row in reference) * (1 + regression_fraction),
        }
    return {
        "schema_version": 1,
        "contract": dict(contract),
        "contract_sha256": contract_digest(contract),
        "reference_revision": reference_revision,
        "independent_starts": len(reference),
        "limits": limits,
    }


def check_performance(
    measured: Mapping[str, float], baseline: Mapping[str, Any], *, contract: Mapping[str, Any]
) -> dict[str, Any]:
    if baseline.get("schema_version") != 1 or baseline.get("independent_starts", 0) < 3:
        raise RegressionError("Missing calibrated baseline")
    if baseline.get("contract") != contract or baseline.get("contract_sha256") != contract_digest(contract):
        raise RegressionError("Baseline hardware/model/workload/environment contract mismatch")
    limits = baseline.get("limits", {})
    required = {"mean_ms", "p50_ms", "p95_ms", "ttfc_ms", "generated_fps", "stall_ms"}
    if not required <= limits.keys():
        raise RegressionError("Incomplete performance limits")
    violations = []
    for metric, limit in limits.items():
        actual = number(measured[metric], metric)
        threshold = number(limit["value"], f"{metric} limit")
        direction = "min" if metric == "generated_fps" else "max"
        if limit["direction"] != direction:
            raise RegressionError(f"Invalid gate direction: {metric}")
        if metric == "p99_ms" and measured["steady_samples"] < 1000:
            raise RegressionError("P99 gate requires at least 1000 steady samples")
        if (direction == "max" and actual > threshold) or (direction == "min" and actual < threshold):
            violations.append(f"{metric}: {actual:.6g} violates {direction} {threshold:.6g}")
    return {"passed": not violations, "violations": violations, "measured": dict(measured)}


def check_quality(
    reference: Sequence[Mapping[str, float]],
    candidate: Mapping[str, float],
    *,
    tolerances: Mapping[str, float],
    activity_dimensions: Sequence[str] = (),
    max_reference_spreads: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Protect each requested dimension; never average across cases/dimensions.

    Candidate tolerances and reference spread limits use native evaluator
    units. Separate reference limits do not widen candidate degradation
    budgets. Without explicit limits, the candidate tolerances also bound
    reference spread for compatibility with strict callers.
    """
    if len(reference) < 3 or not tolerances or set(candidate) != set(tolerances):
        raise RegressionError("Quality requires three repeats and exact dimension coverage")
    if any(set(row) != set(tolerances) for row in reference):
        raise RegressionError("Reference quality dimension coverage mismatch")
    if max_reference_spreads is None:
        max_reference_spreads = tolerances
    elif set(max_reference_spreads) != set(tolerances):
        raise RegressionError("Reference spread limits require exact dimension coverage")
    violations = []
    dimensions = {}
    for dimension, tolerance in tolerances.items():
        tolerance = number(tolerance, f"{dimension} tolerance")
        values = [number(row[dimension], dimension) for row in reference]
        actual = number(candidate[dimension], dimension)
        noise = max(values) - min(values)
        reference_limit = number(max_reference_spreads[dimension], f"{dimension} reference spread limit")
        if noise > reference_limit:
            raise RegressionError(f"Unstable reference quality {dimension}: spread {noise} > {reference_limit}")
        lower_bound = min(values) - tolerance
        upper_bound = max(values) + tolerance if dimension in activity_dimensions else None
        passed = actual >= lower_bound and (upper_bound is None or actual <= upper_bound)
        dimensions[dimension] = {
            "reference": values,
            "reference_spread": noise,
            "max_reference_spread": reference_limit,
            "candidate_tolerance": tolerance,
            "candidate": actual,
            "minimum": lower_bound,
            "maximum": upper_bound,
            "passed": passed,
        }
        if not passed:
            violations.append(f"{dimension}: {actual} outside [{lower_bound}, {upper_bound}]")
    return {"passed": not violations, "violations": violations, "dimensions": dimensions}


def check_binary_activity(
    reference: Sequence[float], candidate: Sequence[float], *, tolerance: float
) -> dict[str, Any]:
    """Compare native binary activity incidence across independent starts.

    A per-video 0/1 classification has no intermediate score: applying a
    fractional tolerance to each video rejects any classification change.
    Preserve every classification and apply that budget to the three-trial
    activity rate instead. This is a fixed count budget, not a confidence test.
    """
    if len(reference) != 3 or len(candidate) != 3:
        raise RegressionError("Binary activity requires three reference and candidate starts")
    rows = [[number(value, "binary activity") for value in values] for values in (reference, candidate)]
    if any(value not in (0.0, 1.0) for values in rows for value in values):
        raise RegressionError("Native binary activity classifications must be exactly zero or one")
    tolerance = number(tolerance, "binary activity tolerance")
    if tolerance >= 1:
        raise RegressionError("Binary activity tolerance must reject a complete activity change")
    reference_count, candidate_count = [int(sum(values)) for values in rows]
    allowed_count_change = math.floor(tolerance * 3)
    passed = abs(candidate_count - reference_count) <= allowed_count_change
    return {
        "passed": passed,
        "protocol": "binary activity incidence over three independent starts",
        "reference": rows[0],
        "candidate": rows[1],
        "reference_active_count": reference_count,
        "candidate_active_count": candidate_count,
        "allowed_count_change": allowed_count_change,
        "candidate_tolerance": tolerance,
    }


def check_video_precision(
    reference: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    *,
    psnr_drop_db: float,
    min_psnr_db: float,
    ssim_drop: float,
    reject_uniform_frames: bool = False,
) -> dict[str, Any]:
    """Gate decoded divergence globally and in every aligned time window."""
    if len(reference) < 2:
        raise RegressionError("Pixel calibration needs comparisons from three reference starts")
    for label, value in (("psnr_drop_db", psnr_drop_db), ("min_psnr_db", min_psnr_db), ("ssim_drop", ssim_drop)):
        number(value, label)

    def ssim(row):
        value = row["ssim"]["mean"]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise RegressionError("Missing finite sampled SSIM")
        if not -1 <= value <= 1:
            raise RegressionError("SSIM must be in [-1, 1]")
        return value

    def psnr(row):
        if type(row["psnr_is_infinite"]) is not bool:
            raise RegressionError("Invalid PSNR infinity flag")
        return None if row["psnr_is_infinite"] else number(row["psnr_db"], "PSNR")

    minimum_ssim = min(ssim(row) for row in reference) - ssim_drop
    actual_ssim = ssim(candidate)
    violations = [] if actual_ssim >= minimum_ssim else ["Sampled SSIM regression"]
    if type(reject_uniform_frames) is not bool:
        raise RegressionError("Uniform-frame policy must be a boolean")
    if reject_uniform_frames:
        populations = [(f"reference comparison {index}", row) for index, row in enumerate(reference)]
        for label, row in [*populations, ("candidate", candidate)]:
            frames = row.get("decoded_frames")
            indices = row.get("uniform_frame_indices")
            if type(frames) is not int or frames < 1 or not isinstance(indices, dict):
                raise RegressionError("Missing decoded uniform-frame coverage")
            for role in ("reference", "candidate"):
                values = indices.get(role)
                if not isinstance(values, list) or any(
                    type(value) is not int or not 0 <= value < frames for value in values
                ):
                    raise RegressionError("Invalid decoded uniform-frame coverage")
                if values:
                    violations.append(f"{label} {role} contains {len(values)} spatially uniform frames")
    psnr_checks = []

    def check(label, references, measured):
        finite = [value for row in references if (value := psnr(row)) is not None]
        minimum = min(finite) - psnr_drop_db if finite else min_psnr_db
        actual = psnr(measured)
        passed = actual is None or actual >= minimum
        psnr_checks.append({"scope": label, "minimum_db": minimum, "actual_db": actual, "passed": passed})
        if not passed:
            violations.append(f"{label} PSNR regression: {actual} < {minimum}")

    check("total", [row["total"] for row in reference], candidate["total"])
    windows = candidate["windows"]

    def signature(rows):
        return [(row["index"], row["start_frame"], row["frame_count"]) for row in rows]

    if not windows or any(signature(row["windows"]) != signature(windows) for row in reference):
        raise RegressionError("Pixel calibration windows do not cover the same frames")
    for index, window in enumerate(windows):
        check(f"window {window['index']}", [row["windows"][index] for row in reference], window)
    return {
        "passed": not violations,
        "violations": violations,
        "ssim": {"minimum": minimum_ssim, "actual": actual_ssim},
        "psnr": psnr_checks,
    }
