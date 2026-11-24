# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Observed success-rate tolerances and conditional paired-bootstrap evidence.

These checks measure task success, not motion continuity. Confidence bounds are
approximate, conditional on the fixed benchmark tasks and the recorded pairing.
"""

import math
import random
from fractions import Fraction

from .e2e_latency import percentile

DEFAULT_OVERALL_DROP_PP = 1.0
DEFAULT_SUITE_DROP_PP = 2.0


def validate_quality_options(overall_drop_pp, suite_drop_pp, policy, confidence, samples):
    if any(not math.isfinite(v) or not 0 <= v <= 100 for v in (overall_drop_pp, suite_drop_pp)):
        raise ValueError("Success-rate tolerances must be finite percentage points in [0, 100]")
    if policy not in {"observed", "confidence"}:
        raise ValueError("Success policy must be observed or confidence")
    if not math.isfinite(confidence) or not 0.5 < confidence < 1:
        raise ValueError("Success confidence must be strictly between 0.5 and 1")
    if type(samples) is not int or samples < 1000:
        raise ValueError("At least 1000 success bootstrap samples are required")


def success_rate_check(name, baseline_successes, baseline_episodes, ours_successes, ours_episodes, max_drop_pp):
    """Inclusive boundary, using exact fractions rather than rounded percentages."""
    for successes, episodes in [(baseline_successes, baseline_episodes), (ours_successes, ours_episodes)]:
        if type(episodes) is not int or episodes <= 0 or type(successes) is not int or not 0 <= successes <= episodes:
            raise ValueError("Invalid success counts or episode denominator")
    if not math.isfinite(max_drop_pp) or not 0 <= max_drop_pp <= 100:
        raise ValueError("Invalid success-rate tolerance")
    base = Fraction(baseline_successes, baseline_episodes)
    ours = Fraction(ours_successes, ours_episodes)
    change = 100 * (ours - base)
    return dict(
        check=name,
        passed=change >= -Fraction(str(max_drop_pp)),
        baseline_successes=baseline_successes,
        baseline_episodes=baseline_episodes,
        ours_successes=ours_successes,
        ours_episodes=ours_episodes,
        baseline_success_rate_percent=float(100 * base),
        ours_success_rate_percent=float(100 * ours),
        success_rate_change_pp=float(change),
        observed_drop_pp=max(0.0, float(-change)),
        maximum_drop_pp=max_drop_pp,
    )


def paired_task_strata(baseline_summary, ours_summary):
    """Match by task and episode ID; never pair different rows by success rank."""
    def outcomes(summary):
        result = {}
        for task in summary["task_results"]:
            for index, episode in enumerate(task["episode_results"]):
                key = (task["task_id"], episode.get("episode", index))
                if key in result or type(episode.get("success")) is not bool:
                    raise ValueError("Duplicate episode identity or non-boolean success outcome")
                result[key] = episode["success"]
        return result

    base, ours = outcomes(baseline_summary), outcomes(ours_summary)
    if not base or base.keys() != ours.keys():
        raise ValueError("Success confidence requires matched task/episode identities")
    tasks = {}
    for key in sorted(base):
        counts = tasks.setdefault(key[0], [0, 0, 0])
        counts[int(ours[key]) - int(base[key]) + 1] += 1
    return list(tasks.values())  # losses, ties, wins within each fixed task


def paired_success_confidence(strata, max_drop_pp, *, confidence=0.95, samples=5000, seed=0):
    """One-sided percentile bound; resample paired episodes within each task.

    Sampling the {-1,0,+1} paired differences from their task-specific counts is
    equivalent to sampling matched episode indices with replacement. Degenerate
    bootstrap distributions are explicitly inconclusive, not proof of equality.
    """
    validate_quality_options(max_drop_pp, max_drop_pp, "confidence", confidence, samples)
    n = sum(sum(c) for c in strata)
    if n <= 0:
        raise ValueError("Missing paired episodes")
    prepared = []
    losses = wins = 0
    for negative, tied, positive in strata:
        count = negative + tied + positive
        if count <= 0 or min(negative, tied, positive) < 0:
            raise ValueError("Invalid paired task counts")
        prepared.append((count, (negative, negative + tied, count)))
        losses += negative
        wins += positive
    result = dict(
        method="paired_percentile_bootstrap_within_fixed_tasks",
        confidence_level=confidence,
        interval="one-sided lower bound for Ours minus Baseline, in percentage points",
        bootstrap_samples=samples,
        bootstrap_seed=seed,
        paired_episodes=n,
        task_strata=len(strata),
        baseline_only_successes=losses,
        ours_only_successes=wins,
        success_rate_change_pp=100 * (wins - losses) / n,
        lower_bound_pp=None,
        noninferiority_boundary_pp=-max_drop_pp,
        supported=False,
        status="inconclusive",
        scope="conditional on fixed tasks and matched initial states; not unseen-task generalization",
    )
    if all(sum(c > 0 for c in counts) == 1 for counts in strata):
        return dict(result, reason="degenerate paired outcomes within every task")
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        difference = sum(sum(rng.choices((-1, 0, 1), cum_weights=cumulative, k=count))
                         for count, cumulative in prepared)
        draws.append(100 * difference / n)
    lower = round(percentile(draws, 100 * (1 - confidence)), 12)
    supported = lower > -max_drop_pp
    return dict(result, lower_bound_pp=lower, supported=supported,
                status="supported" if supported else "not_demonstrated")
