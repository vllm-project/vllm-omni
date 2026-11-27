# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import math

import pytest

from vllm_omni_cosmos3_client.success_quality import (
    paired_success_confidence,
    paired_task_strata,
    success_rate_check,
    validate_quality_options,
)


def test_one_lost_episode_and_exact_percentage_point_boundaries():
    one = success_rate_check("suite", 492, 500, 491, 500, 2)
    assert one["passed"] and one["success_rate_change_pp"] == -0.2
    assert success_rate_check("suite", 492, 500, 482, 500, 2)["passed"]
    assert not success_rate_check("suite", 492, 500, 481, 500, 2)["passed"]
    assert success_rate_check("overall", 1952, 2000, 1932, 2000, 1)["passed"]
    assert not success_rate_check("overall", 1952, 2000, 1931, 2000, 1)["passed"]
    assert success_rate_check("improvement", 450, 500, 495, 500, 1)["passed"]
    assert not success_rate_check("strict", 492, 500, 491, 500, 0)["passed"]


def test_comparison_uses_rates_and_does_not_round_a_drop_into_acceptance():
    assert success_rate_check("different_n", 90, 100, 180, 200, 0)["passed"]
    assert success_rate_check("decimal_boundary", 999, 1000, 998, 1000, 0.1)["passed"]
    assert not success_rate_check("decimal_outside", 999, 1000, 997, 1000, 0.1)["passed"]


@pytest.mark.parametrize("tolerance", [-1, 101, math.nan, math.inf])
def test_invalid_tolerances_cannot_disable_quality_checks(tolerance):
    with pytest.raises(ValueError):
        validate_quality_options(tolerance, 2, "observed", 0.95, 1000)
    with pytest.raises(ValueError):
        success_rate_check("x", 90, 100, 90, 100, tolerance)


def summary(outcomes):
    return {
        "task_results": [
            {"task_id": 0, "episode_results": [{"episode": i, "success": s} for i, s in enumerate(outcomes)]}
        ]
    }


def test_pairing_is_by_identity_not_row_order_and_marginals_are_insufficient():
    base = summary([True] * 490 + [False] * 10)
    shared = summary([True] * 489 + [False] * 11)
    disjoint = summary([False] * 11 + [True] * 489)
    assert paired_task_strata(base, shared) == [[1, 499, 0]]
    assert paired_task_strata(base, disjoint) == [[11, 479, 10]]
    shared["task_results"][0]["episode_results"].reverse()
    assert paired_task_strata(base, shared) == [[1, 499, 0]]
    del shared["task_results"][0]["episode_results"][0]
    with pytest.raises(ValueError, match="matched"):
        paired_task_strata(base, shared)


def test_confidence_bounds_are_paired_reproducible_and_not_a_point_estimate_gate():
    narrow = paired_success_confidence([[1, 499, 0]], 1, samples=1000, seed=7)
    wide = paired_success_confidence([[11, 479, 10]], 1, samples=1000, seed=7)
    assert narrow == paired_success_confidence([[1, 499, 0]], 1, samples=1000, seed=7)
    assert narrow["success_rate_change_pp"] == wide["success_rate_change_pp"] == -0.2
    assert narrow["lower_bound_pp"] > wide["lower_bound_pp"]
    assert narrow["supported"] and not wide["supported"]
    assert wide["status"] == "not_demonstrated"
    # A non-significant difference alone would not express this distinction.
    high = paired_success_confidence([[11, 479, 10]], 1, confidence=0.99, samples=1000, seed=7)
    assert high["lower_bound_pp"] <= wide["lower_bound_pp"]


def test_degenerate_bootstrap_is_inconclusive_not_proof_of_equality():
    report = paired_success_confidence([[0, 500, 0]], 2, samples=1000)
    assert report["lower_bound_pp"] is None
    assert not report["supported"] and report["status"] == "inconclusive"
