# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare completed baseline/FASTER runs; exit 2 when acceptance fails."""

import argparse
import hashlib
import json
from pathlib import Path

from .e2e_latency import SCHEMA, build_e2e_report, percentile
from .success_quality import (
    DEFAULT_OVERALL_DROP_PP,
    DEFAULT_SUITE_DROP_PP,
    paired_success_confidence,
    paired_task_strata,
    success_rate_check,
    validate_quality_options,
)

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_run(directory):
    summary = json.loads((directory / "summary.json").read_text())
    records = jsonl(directory / "request_timings.jsonl")
    report = build_e2e_report(jsonl(directory / "client_timeline.jsonl"), records)
    config = summary["run_config"]
    episodes = [episode for task in summary["task_results"] for episode in task["episode_results"]]
    expected = len(config["task_ids"]) * config["num_trials_per_task"]
    if summary["total_episodes"] != expected or len(episodes) != expected:
        raise ValueError(f"Incomplete episodes in {directory}")
    if len({task["task_id"] for task in summary["task_results"]}) != len(config["task_ids"]):
        raise ValueError(f"Incomplete tasks in {directory}")
    if any(ep.get("error") or ep.get("skipped") for ep in episodes):
        raise ValueError(f"Failed/skipped episode in {directory}")
    if sum(bool(ep["success"]) for ep in episodes) != summary["total_successes"]:
        raise ValueError(f"Success count mismatch in {directory}")
    by_episode = {}
    for window in report["windows"]:
        key = (window["task_id"], window["episode"])
        by_episode[key] = by_episode.get(key, 0) + window["executed_actions"]
    for task in summary["task_results"]:
        for index, episode in enumerate(task["episode_results"]):
            if by_episode.get((task["task_id"], index), 0) != episode["executed_actions"]:
                raise ValueError(f"E2E timeline/action count mismatch in {directory}")
    proof = []
    for name in ["summary.json", "request_timings.jsonl", "client_timeline.jsonl"]:
        path = directory / name
        proof.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return dict(directory=str(directory.resolve()), summary=summary, latency=report, evidence=proof)


def compare(baseline_dirs, faster_dirs, *, minimum_windows=100, minimum_improvement_percent=10,
            expected_suites=SUITES, overall_success_drop_pp=DEFAULT_OVERALL_DROP_PP,
            suite_success_drop_pp=DEFAULT_SUITE_DROP_PP, success_policy="observed",
            success_confidence=0.95, success_bootstrap_samples=5000, success_bootstrap_seed=0):
    if minimum_windows < 1 or not 0 <= minimum_improvement_percent < 100:
        raise ValueError("Invalid acceptance thresholds")
    validate_quality_options(overall_success_drop_pp, suite_success_drop_pp, success_policy,
                             success_confidence, success_bootstrap_samples)
    groups = []
    for paths in [baseline_dirs, faster_dirs]:
        loaded = [load_run(Path(p)) for p in paths]
        indexed = {r["summary"]["task_suite"]: r for r in loaded}
        if len(indexed) != len(loaded) or set(indexed) != set(expected_suites):
            raise ValueError("Expected exactly one completed run per requested suite and profile")
        groups.append(indexed)
    rows, pooled, checks, evidence = [], [[], []], [], []
    quality_checks, quality_suites, all_strata = [], [], []
    total_base_successes = total_base_episodes = total_ours_successes = total_ours_episodes = 0
    for suite in expected_suites:
        base, faster = (group[suite] for group in groups)
        configs = [run["summary"]["run_config"] for run in [base, faster]]
        for key in [
            "task_suite",
            "task_ids",
            "num_trials_per_task",
            "num_envs",
            "seed",
            "max_steps",
            "action_horizon",
            "warmup_steps",
            "exclude_dummy_time",
            "timing_warmup_requests",
            "camera",
            "action_space",
            "rotation_space",
            "action_dim",
            "initial_states_path",
            "debug_output_enabled",
            "render_mode",
            "image_size",
            "env_image_size",
            "numba_disable_jit",
        ]:
            if key not in configs[0] or key not in configs[1] or configs[0][key] != configs[1][key]:
                raise ValueError(f"Unpaired or missing evaluation setting: {suite}/{key}")
        if (
            configs[0]["debug_output_enabled"]
            or configs[0]["action_horizon"] != 16
            or configs[0]["task_ids"] != list(range(10))
        ):
            raise ValueError("Acceptance requires H16, all 10 tasks and disabled debug output")
        for key in [
            "num_steps",
            "guidance",
            "flow_shift",
            "seed",
            "action_chunk_size",
            "sampler_dtype",
            "action_normalization",
        ]:
            a, b = [c["server_info"] for c in configs]
            if key not in a or key not in b or a[key] != b[key]:
                raise ValueError(f"Unpaired or missing model setting: {suite}/{key}")
        if configs[0].get("streaming") is not False or configs[1].get("streaming") is not True:
            raise ValueError("Acceptance requires baseline synchronous and FASTER streaming execution")
        for config, schedule in zip(configs, ["constant_euler", "has_euler"]):
            if config["server_info"].get("action_time_schedule") != schedule:
                raise ValueError(f"Unexpected policy schedule in {suite}")
        p95 = []
        for index, run in enumerate([base, faster]):
            windows = [w["e2e_no_render_ms"] for w in run["latency"]["windows"] if w["complete"] and not w["warmup"]]
            pooled[index].extend(windows)
            checks.append(
                dict(
                    check=f"{suite}/{'baseline' if index == 0 else 'faster'}/sample_count",
                    passed=len(windows) >= minimum_windows,
                    count=len(windows),
                )
            )
            p95.append(percentile(windows, 95))
            evidence.extend(run["evidence"])
        if any(value is None for value in p95):
            raise ValueError(f"No complete measured windows for {suite}")
        bsum, fsum = [r["summary"] for r in [base, faster]]
        observed = success_rate_check(f"{suite}/success_rate_tolerance", bsum["total_successes"],
                                      bsum["total_episodes"], fsum["total_successes"], fsum["total_episodes"],
                                      suite_success_drop_pp)
        strata = paired_task_strata(bsum, fsum)
        confidence = paired_success_confidence(strata, suite_success_drop_pp, confidence=success_confidence,
                                              samples=success_bootstrap_samples, seed=success_bootstrap_seed)
        quality_checks.append(observed)
        if success_policy == "confidence":
            quality_checks.append(dict(check=f"{suite}/success_rate_confidence", passed=confidence["supported"]))
        quality_suites.append(dict(task_suite=suite, observed=observed, confidence=confidence))
        all_strata.extend(strata)
        total_base_successes += bsum["total_successes"]
        total_base_episodes += bsum["total_episodes"]
        total_ours_successes += fsum["total_successes"]
        total_ours_episodes += fsum["total_episodes"]
        rows.append(
            dict(
                task_suite=suite,
                baseline_p95_ms=p95[0],
                faster_p95_ms=p95[1],
                reduction_percent=(1 - p95[1] / p95[0]) * 100,
                baseline_successes=bsum["total_successes"],
                faster_successes=fsum["total_successes"],
                episodes_per_profile=bsum["total_episodes"],
                baseline_checkpoint=configs[0]["server_info"].get("checkpoint"),
                faster_checkpoint=configs[1]["server_info"].get("checkpoint"),
            )
        )
    overall_observed = success_rate_check("overall/success_rate_tolerance", total_base_successes,
                                          total_base_episodes, total_ours_successes, total_ours_episodes,
                                          overall_success_drop_pp)
    overall_confidence = paired_success_confidence(all_strata, overall_success_drop_pp,
                                                  confidence=success_confidence, samples=success_bootstrap_samples,
                                                  seed=success_bootstrap_seed)
    quality_checks.append(overall_observed)
    if success_policy == "confidence":
        quality_checks.append(dict(check="overall/success_rate_confidence", passed=overall_confidence["supported"]))
    checks.extend(quality_checks)
    p95 = [percentile(values, 95) for values in pooled]
    reduction = (1 - p95[1] / p95[0]) * 100
    # Compare durations directly: equality at the 10% boundary passes.
    checks.append(
        dict(check="pooled_e2e_p95_target", passed=p95[1] <= p95[0] * (1 - minimum_improvement_percent / 100))
    )
    return dict(
        schema=SCHEMA,
        passed=all(c["passed"] for c in checks),
        checks=checks,
        failed_checks=[c["check"] for c in checks if not c["passed"]],
        quality=dict(
            policy=success_policy,
            overall_maximum_drop_pp=overall_success_drop_pp,
            suite_maximum_drop_pp=suite_success_drop_pp,
            passed=all(c["passed"] for c in quality_checks),
            observed_tolerance_passed=overall_observed["passed"] and all(s["observed"]["passed"] for s in quality_suites),
            statistical_noninferiority_supported=overall_confidence["supported"] and all(
                s["confidence"]["supported"] for s in quality_suites),
            overall=dict(observed=overall_observed, confidence=overall_confidence),
            suites=quality_suites,
            motion_continuity="not assessed by success-rate checks",
        ),
        suites=rows,
        aggregate=dict(
            method="pool all complete nonwarmup windows; do not average suite percentiles",
            baseline_windows=len(pooled[0]),
            faster_windows=len(pooled[1]),
            baseline_p95_ms=p95[0],
            faster_p95_ms=p95[1],
            reduction_percent=reduction,
            required_reduction_percent=minimum_improvement_percent,
        ),
        evidence=evidence,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, action="append", required=True)
    parser.add_argument("--faster", type=Path, action="append", required=True)
    parser.add_argument("--minimum-windows", type=int, default=100)
    parser.add_argument("--minimum-improvement-percent", type=float, default=10)
    parser.add_argument("--overall-success-drop-pp", type=float, default=DEFAULT_OVERALL_DROP_PP)
    parser.add_argument("--suite-success-drop-pp", type=float, default=DEFAULT_SUITE_DROP_PP)
    parser.add_argument("--success-policy", choices=["observed", "confidence"], default="observed")
    parser.add_argument("--success-confidence", type=float, default=0.95)
    parser.add_argument("--success-bootstrap-samples", type=int, default=5000)
    parser.add_argument("--success-bootstrap-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare(
            args.baseline,
            args.faster,
            minimum_windows=args.minimum_windows,
            minimum_improvement_percent=args.minimum_improvement_percent,
            overall_success_drop_pp=args.overall_success_drop_pp,
            suite_success_drop_pp=args.suite_success_drop_pp,
            success_policy=args.success_policy,
            success_confidence=args.success_confidence,
            success_bootstrap_samples=args.success_bootstrap_samples,
            success_bootstrap_seed=args.success_bootstrap_seed,
        )
    except (ValueError, KeyError, OSError, TypeError) as exc:
        result = dict(schema=SCHEMA, passed=False, error=f"{type(exc).__name__}: {exc}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    terminal = dict(passed=result["passed"], failed_checks=result.get("failed_checks", []),
                    aggregate=result.get("aggregate"), quality=result.get("quality"), error=result.get("error"))
    print(json.dumps(terminal, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 2)


if __name__ == "__main__":
    main()
