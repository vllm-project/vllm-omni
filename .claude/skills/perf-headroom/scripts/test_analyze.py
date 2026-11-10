# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU feature/negative tests for the assessment contract; no model dependencies."""

import copy
from graphlib import CycleError

import pytest
from analyze import CARD_FIELDS, assess, digest, read_json, render
from trace_math import ledger, replay, trace_ledger, union

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def evidence(tmp_path):
    artifact = tmp_path / "raw.txt"
    artifact.write_text("synthetic measurements", encoding="utf-8")
    return {
        "schema_version": 1,
        "scope": {
            k: "synthetic"
            for k in (
                "model",
                "weight_revision",
                "code_sha",
                "hardware",
                "workload",
                "runtime",
                "boundary",
                "quality_contract",
            )
        },
        "artifacts": [{"id": "raw", "path": str(artifact), "sha256": digest(artifact)}],
        "provenance": {"scope_verified": True, "origin": "synthetic", "evidence": ["raw"]},
        "benchmark": {"profiled": False, "cohorts_ms": [[100] * 5, [100] * 5], "evidence": ["raw"]},
        "diagnostic": {
            "window_ms": 100,
            "exclusive_ledger_ms": {"work": 100},
            "evidence": ["raw"],
            "timeline_coverage": {
                "status": "complete",
                "reason": "Synthetic complete node events",
                "evidence": ["raw"],
            },
        },
        "coverage": {"ledger_modules": {"work": ["m1"]}, "evidence": ["raw"]},
        "modules": [
            {
                "id": f"m{i}",
                "layer": f"L{i}",
                "applicable": True,
                "measured": True,
                "evidence": ["raw"],
                "exclusive_ms": 100 if i == 1 else 20,
                "structural_review": {"complete": True, "evidence": ["raw"]},
            }
            for i in range(1, 6)
        ],
    }


def candidate():
    return {
        "id": "hoist",
        "evidence": ["raw"],
        "next_action": "measure replacement",
        "validation": "paired unprofiled and quality gate",
    }


def stop_argument():
    return {
        "remaining_gain_upper_pct": 1,
        "search_scope": "synthetic finite candidates",
        "assumptions": ["joint transformations included"],
        "reviewed": True,
        "evidence": ["raw"],
    }


def detailed(evidence):
    evidence["analysis_depth"] = "detailed"
    evidence["layer_reviews"] = {
        f"L{i}": {
            "verdict": "synthetic reviewed scope",
            "confidence": "medium",
            "evidence": ["raw"],
            "mature": "matched scope only",
            "opportunities": "none within synthetic scope",
            "unknowns": "outside scope",
            "next_experiment": "independent residual review",
        }
        for i in range(1, 6)
    }
    evidence["modules"][3]["kernel_audit"] = {
        "basis": "in_model_ncu",
        "population": "synthetic shape: 1 of 1 calls",
        "selection": "only kernel",
        "limitations": "synthetic test, not hardware evidence",
        "counter_findings": "synthetic counters",
        "evidence": ["raw"],
    }
    return evidence


def test_detailed_requires_layer_and_counter_coverage(evidence):
    evidence["analysis_depth"] = "detailed"
    evidence["scoped_stop"] = stop_argument()
    evidence["policy_predeclared"] = True
    result = assess(evidence)
    assert not result["detail_checks"]["complete"]
    assert "detailed_contract" in result["L6"]["stop_blockers"]


def test_complete_deep_contract_is_not_stop_authorization(evidence):
    result = assess(detailed(evidence))
    assert result["detail_checks"]["complete"]
    assert result["L6"]["performance_decision"] == "NEED_EVIDENCE"
    assert not result["L6"]["stop_authorized"]


@pytest.mark.parametrize("basis", ["isolated_ncu", "timeline_only"])
def test_proxy_counters_do_not_close_in_model_analysis(evidence, basis):
    detailed(evidence)["modules"][3]["kernel_audit"]["basis"] = basis
    assert not assess(evidence)["detail_checks"]["complete"]


def test_counter_omission_requires_reason(evidence):
    audit = detailed(evidence)["modules"][3]["kernel_audit"]
    audit["basis"] = "not_required"
    assert not assess(evidence)["detail_checks"]["complete"]
    audit["reason"] = "synthetic zero-leverage scope with declared evidence"
    assert assess(evidence)["detail_checks"]["complete"]


def test_generic_advice_is_incomplete_but_retained(evidence):
    detailed(evidence)["modules"][3]["candidates"] = [candidate()]
    result = assess(evidence)
    assert not result["detail_checks"]["complete"]
    assert result["L6"]["next_candidates"][0]["id"] == "hoist"


def test_complete_recommendation_survives_machine_and_text_reports(evidence):
    card = candidate() | {key: f"synthetic {key}" for key in CARD_FIELDS}
    card.update(priority="P1", confidence="medium")
    detailed(evidence)["modules"][3]["candidates"] = [card]
    result = assess(evidence)
    assert result["detail_checks"]["complete"]
    assert result["L6"]["next_candidates"][0]["counter_evidence"] == card["counter_evidence"]
    text = render(result)
    assert card["counter_evidence"] in text and card["quality_gate"] in text
    positions = [text.index(f"L{i}:") for i in range(7)]
    assert positions == sorted(positions)


def fresh_campaign(evidence):
    evidence["campaign"] = {
        "id": "new",
        "evaluator_revision": "frozen-test",
        "reuse_policy": "fresh_only",
        "started_at_utc": "2026-10-20T00:00:00Z",
    }
    evidence["artifacts"][0].update(kind="measurement", run_id="new", collected_at_utc="2026-10-20T01:00:00Z")
    return evidence


@pytest.mark.parametrize(
    "change",
    [{"run_id": "old"}, {"collected_at_utc": "2026-10-19T23:59:00Z"}, {"kind": "source"}],
)
def test_relabeling_old_runtime_evidence_cannot_pass_fresh_gate(evidence, change):
    fresh_campaign(evidence)["artifacts"][0].update(change)
    result = assess(evidence)
    assert result["freshness"]["status"] == "invalid"
    assert not result["L0"]["provenance_valid"]
    assert result["L6"]["performance_decision"] == "NEED_EVIDENCE"


def test_fresh_manifest_does_not_certify_science(evidence):
    result = assess(fresh_campaign(evidence))
    assert result["freshness"]["status"] == "declared_fresh"
    assert result["L6"]["performance_decision"] == "NEED_EVIDENCE"


def test_unknown_artifact_in_deep_review_is_rejected(evidence):
    detailed(evidence)["layer_reviews"]["L2"]["evidence"] = ["absent"]
    with pytest.raises(ValueError, match="evidence ids"):
        assess(evidence)


def test_idle_union_not_kernel_sum():
    result = ledger([(1, 5, "a"), (3, 7, "b"), (9, 10, "a")], 0, 12)
    assert result["busy_union_ms"] == 7
    assert result["internal_gap_ms"] == 2
    assert result["exclusive_ledger_ms"]["cross_category_overlap"] == 2
    assert sum(result["exclusive_ledger_ms"].values()) == 12


@pytest.mark.parametrize(
    ("events", "end", "work_sum", "busy", "excess"),
    [
        ([(0, 10, "a"), (0, 8, "b")], 10, 18, 10, 8),
        ([(0, 10, "a"), (10, 18, "b")], 18, 18, 18, 0),
        ([(0, 10, "a"), (0, 10, "b"), (0, 10, "c")], 10, 30, 10, 20),
    ],
)
def test_work_sum_is_not_wall_time(events, end, work_sum, busy, excess) -> None:
    result = ledger(events, 0, end)
    assert result["summed_event_ms"] == work_sum
    assert result["busy_union_ms"] == busy
    assert result["overlap_excess_ms"] == excess
    assert sum(result["exclusive_ledger_ms"].values()) == end


def test_work_sum_clips_events_and_excludes_outside_work() -> None:
    result = ledger([(-4, 3, "a"), (1, 15, "b"), (30, 40, "c")], 0, 10)
    assert result["summed_event_ms"] == 12
    assert result["busy_union_ms"] == 10
    assert result["overlap_excess_ms"] == 2


def test_trace_work_sum_includes_copies_but_not_cpu_or_other_devices() -> None:
    events = [
        {"ph": "X", "cat": cat, "ts": 0, "dur": duration, "name": cat, "args": {"device": device}}
        for cat, duration, device in (
            ("kernel", 10000, 0),
            ("gpu_memcpy", 8000, 0),
            ("gpu_memset", 2000, 0),
            ("cpu_op", 30000, 0),
            ("kernel", 40000, 1),
        )
    ]
    result = trace_ledger(events, 0)
    assert result["summed_event_ms"] == 20
    assert result["busy_union_ms"] == 10
    assert result["window_ms"] == 10


def test_request_head_and_tail_are_not_kernel_work() -> None:
    result = ledger([(2, 12, "a"), (2, 10, "b")], 0, 15)
    assert result["window_ms"] == 15
    assert result["summed_event_ms"] == 18
    assert result["busy_union_ms"] == 10
    assert result["exclusive_ledger_ms"]["request_head"] == 2
    assert result["exclusive_ledger_ms"]["request_tail"] == 3


def test_same_category_overlap_counted_once():
    assert ledger([(0, 3, "a"), (1, 4, "a")], 0, 4)["exclusive_ledger_ms"]["a"] == 4


def test_events_clipped_and_outside_ignored():
    result = ledger([(-4, 3, "a"), (8, 15, "a"), (30, 40, "b")], 0, 10)
    assert result["busy_union_ms"] == 5
    assert result["internal_gap_ms"] == 5


def test_empty_device_window_not_saturation():
    result = ledger([], 0, 10)
    assert not result["has_device_events"]
    assert result["internal_gap_pct"] is None


def test_invalid_interval_rejected():
    with pytest.raises(ValueError):
        union([(2, 1)])


def test_trace_selects_single_device():
    events = [{"ph": "X", "cat": "kernel", "ts": 0, "dur": 1000, "name": "GEMM", "args": {"device": i}} for i in (0, 1)]
    assert trace_ledger(events, 0)["busy_union_ms"] == 1


def test_ambiguous_role_stays_unattributed():
    event = {"ph": "X", "cat": "kernel", "ts": 0, "dur": 1000, "name": "GEMM", "args": {"device": 0}}
    rules = [{"label": label, "contains": ["GEMM"]} for label in ("a", "b")]
    assert trace_ledger([event], 0, rules=rules)["exclusive_ledger_ms"]["unattributed"] == 1


def test_no_candidates_does_not_mean_stop(evidence):
    assert assess(evidence)["L6"]["performance_decision"] == "NEED_EVIDENCE"


def test_high_efficiency_with_redundancy_stays_actionable(evidence):
    m = evidence["modules"][0]
    m["reference"] = {
        "kind": "measured_reference",
        "current_ms": 20,
        "time_ms_range": [19, 20],
        "comparability": "matched",
        "evidence": ["raw"],
    }
    m["candidates"] = [candidate()]
    result = assess(evidence)
    assert result["modules"][0]["execution_state"] == "NEAR_REFERENCE"
    assert result["modules"][0]["state"] == "ACTIONABLE_CANDIDATE"
    assert result["L6"]["performance_decision"] == "CONTINUE_TARGETED_OPTIMIZATION"


def test_reference_above_current_requests_recheck(evidence):
    evidence["modules"][0]["reference"] = {
        "kind": "measured_reference",
        "current_ms": 10,
        "time_ms_range": [12, 13],
        "comparability": "matched",
        "evidence": ["raw"],
    }
    assert assess(evidence)["modules"][0]["execution_state"] == "REFERENCE_RECHECK"


def test_historical_trace_is_labeled(evidence):
    evidence["provenance"]["origin"] = "historical verified trace"
    assert assess(evidence)["L0"]["evidence_origin"] == "historical verified trace"


def test_hash_mismatch_blocks_decision(evidence):
    evidence["artifacts"][0]["sha256"] = "0" * 64
    evidence["modules"][0]["candidates"] = [candidate()]
    result = assess(evidence)
    assert not result["L0"]["provenance_valid"]
    assert result["L6"]["performance_decision"] == "NEED_EVIDENCE"


def test_missing_artifact_blocks_decision(evidence):
    evidence["artifacts"][0]["path"] += ".missing"
    assert not assess(evidence)["L0"]["provenance_valid"]


def test_scoped_stop_needs_stable_covered_predeclared_evidence(evidence):
    evidence.update(scoped_stop=stop_argument(), policy_predeclared=True)
    result = assess(evidence)
    assert result["L6"]["performance_decision"] == "STOP_REVIEW_REQUIRED"
    assert not result["L6"]["stop_authorized"]
    assert result["L6"]["stop_blockers"] == []


@pytest.mark.parametrize("missing", ["ledger", "coverage"])
def test_missing_coverage_cannot_propose_stop(evidence, missing):
    evidence.update(scoped_stop=stop_argument(), policy_predeclared=True)
    if missing == "ledger":
        del evidence["diagnostic"]["exclusive_ledger_ms"]
    else:
        del evidence["coverage"]
    assert assess(evidence)["L6"]["performance_decision"] == "NEED_EVIDENCE"


def test_unattributed_work_blocks_stop_even_when_mapped(evidence):
    evidence.update(scoped_stop=stop_argument(), policy_predeclared=True)
    evidence["diagnostic"]["exclusive_ledger_ms"] = {"work": 1, "unattributed": 99}
    evidence["coverage"]["ledger_modules"]["unattributed"] = ["m1"]
    result = assess(evidence)
    assert result["coverage"]["uncovered_ms"] == 99
    assert result["L6"]["performance_decision"] == "NEED_EVIDENCE"


def test_declared_unknown_and_unmapped_work_stays_unknown(evidence):
    evidence["diagnostic"]["exclusive_ledger_ms"] = {"work": 1, "opaque": 98, "missing": 1}
    evidence["coverage"]["ledger_modules"]["opaque"] = ["m1"]
    evidence["coverage"]["unclassified_labels"] = ["opaque"]
    assert assess(evidence)["coverage"]["uncovered_ms"] == 99


@pytest.mark.parametrize("bindings", [{"work": ["absent"]}, {"absent": ["m1"]}, {"work": ["m1", "m1"]}])
def test_invalid_coverage_binding_rejected(evidence, bindings):
    evidence["coverage"]["ledger_modules"] = bindings
    with pytest.raises(ValueError):
        assess(evidence)


def test_token_module_cannot_cover_whole_window(evidence):
    evidence["modules"][0]["exclusive_ms"] = 1
    with pytest.raises(ValueError):
        assess(evidence)


def test_noise_must_resolve_useful_gain(evidence):
    evidence.update(scoped_stop=stop_argument(), policy_predeclared=True)
    evidence["policy"] = {"minimum_useful_gain_pct": 0.1}
    evidence["scoped_stop"]["remaining_gain_upper_pct"] = 0.05
    evidence["benchmark"]["cohorts_ms"][1] = [101.5] * 5
    result = assess(evidence)
    assert result["L0"]["benchmark"]["stable_for_policy"]
    assert not result["L0"]["benchmark"]["precise_for_policy"]
    assert result["L6"]["performance_decision"] == "NEED_EVIDENCE"


def test_noise_margin_blocks_near_threshold_argument(evidence):
    evidence.update(scoped_stop=stop_argument(), policy_predeclared=True)
    evidence["scoped_stop"]["remaining_gain_upper_pct"] = 1.9
    evidence["benchmark"]["cohorts_ms"][1] = [100.2] * 5
    result = assess(evidence)
    assert result["L0"]["benchmark"]["precise_for_policy"]
    assert result["L6"]["performance_decision"] == "NEED_EVIDENCE"


@pytest.mark.parametrize("comparability", ["matched", "unknown"])
def test_unresolved_reference_blocks_stop(evidence, comparability):
    evidence.update(scoped_stop=stop_argument(), policy_predeclared=True)
    evidence["modules"][0]["reference"] = {
        "kind": "measured_reference",
        "current_ms": 10,
        "time_ms_range": [12, 13],
        "comparability": comparability,
        "evidence": ["raw"],
    }
    assert assess(evidence)["L6"]["performance_decision"] == "NEED_EVIDENCE"


def test_drift_blocks_scoped_stop(evidence):
    evidence.update(scoped_stop=stop_argument(), policy_predeclared=True)
    evidence["benchmark"]["cohorts_ms"][1] = [110] * 5
    assert assess(evidence)["L6"]["performance_decision"] == "NEED_EVIDENCE"


def test_drift_preserves_structural_candidate(evidence):
    evidence["benchmark"]["cohorts_ms"][1] = [110] * 5
    evidence["modules"][0]["candidates"] = [candidate()]
    result = assess(evidence)
    assert not result["L0"]["benchmark"]["stable_for_policy"]
    assert result["L6"]["performance_decision"] == "CONTINUE_TARGETED_OPTIMIZATION"


def test_partial_assessment_cannot_stop(evidence):
    evidence.update(scoped_stop=stop_argument(), policy_predeclared=True)
    result = assess(evidence, layers={"L4"})
    assert result["selected_layers"] == ["L0", "L4", "L6"]
    assert result["L6"]["performance_decision"] == "NEED_EVIDENCE"


def test_missing_layer_cannot_stop(evidence):
    evidence.update(scoped_stop=stop_argument(), policy_predeclared=True)
    evidence["modules"].pop()
    assert assess(evidence)["layer_states"]["L5"] == "UNKNOWN"
    assert assess(evidence)["L6"]["performance_decision"] == "NEED_EVIDENCE"


def test_unmeasured_module_cannot_stop(evidence):
    evidence.update(scoped_stop=stop_argument(), policy_predeclared=True)
    evidence["modules"][0]["measured"] = False
    assert assess(evidence)["L6"]["performance_decision"] == "NEED_EVIDENCE"


def test_structural_review_missing_cannot_stop(evidence):
    evidence.update(scoped_stop=stop_argument(), policy_predeclared=True)
    evidence["modules"][0]["structural_review"]["complete"] = False
    assert assess(evidence)["L6"]["performance_decision"] == "NEED_EVIDENCE"


def test_na_is_not_unknown(evidence):
    evidence["modules"][4].update(applicable=False, reason="single GPU")
    assert assess(evidence)["layer_states"]["L5"] == "NOT_APPLICABLE"


def test_na_requires_reason(evidence):
    evidence["modules"][4]["applicable"] = False
    with pytest.raises(ValueError):
        assess(evidence)


def test_budget_stop_not_saturation(evidence):
    evidence["budget_exhausted"] = True
    assert assess(evidence)["L6"]["performance_decision"] == "STOP_BUDGET"


def test_ledger_must_close(evidence):
    evidence["diagnostic"]["exclusive_ledger_ms"]["work"] = 120
    with pytest.raises(ValueError):
        assess(evidence)


def test_unknown_evidence_id_rejected(evidence):
    evidence["modules"][0]["evidence"] = ["missing"]
    with pytest.raises(ValueError):
        assess(evidence)


def test_global_bound_never_inferred_from_reference(evidence):
    assert assess(evidence)["L6"]["certified_remaining_gain_upper_pct"] is None


def test_validated_gain_requires_quality(evidence):
    c = candidate()
    c["impact"] = {"type": "validated_gain", "range_ms": [1, 2], "assumptions": ["same scope"], "evidence": ["raw"]}
    evidence["modules"][0]["candidates"] = [c]
    with pytest.raises(ValueError):
        assess(evidence)


def test_inputs_are_not_mutated(evidence):
    old = copy.deepcopy(evidence)
    assess(evidence)
    assert evidence == old


def test_nan_rejected(tmp_path):
    path = tmp_path / "nan.json"
    path.write_text('{"x": NaN}', encoding="utf-8")
    with pytest.raises(ValueError):
        read_json(path)


def test_replay_recomputes_critical_path():
    graph = {
        "resource_order_confirmed": True,
        "nodes": [
            {"id": "a", "ms": 10, "deps": []},
            {"id": "b", "ms": 8, "deps": []},
            {"id": "c", "ms": 1, "deps": ["a", "b"]},
        ],
    }
    assert replay(graph)["predicted_ms"] == 11
    graph["nodes"][0]["ms"] = 5
    assert replay(graph)["predicted_ms"] == 9  # not 11 - 5; the critical path changes.


def test_faster_hidden_work_does_not_shorten_fixed_duration_join() -> None:
    graph = {
        "resource_order_confirmed": True,
        "nodes": [
            {"id": "a", "ms": 10, "deps": []},
            {"id": "b", "ms": 8, "deps": []},
            {"id": "join", "ms": 0, "deps": ["a", "b"]},
        ],
    }
    baseline = replay(graph)
    graph["nodes"][1]["ms"] = 4
    alternative = replay(graph)
    assert baseline["predicted_ms"] == alternative["predicted_ms"] == 10
    assert alternative["critical_path"] == ["a", "join"]
    assert alternative["hard_lower_bound"] is False


def test_serial_dependency_makes_previously_hidden_work_exposed() -> None:
    graph = {
        "resource_order_confirmed": True,
        "nodes": [{"id": "a", "ms": 10, "deps": []}, {"id": "b", "ms": 8, "deps": ["a"]}],
    }
    assert replay(graph)["predicted_ms"] == 18
    graph["nodes"][1]["ms"] = 4
    assert replay(graph)["predicted_ms"] == 14


def test_joint_branch_changes_are_not_sum_of_individual_gains() -> None:
    graph = {
        "resource_order_confirmed": True,
        "nodes": [
            {"id": "a", "ms": 10, "deps": []},
            {"id": "b", "ms": 8, "deps": []},
            {"id": "join", "ms": 0, "deps": ["a", "b"]},
        ],
    }
    baseline = replay(graph)["predicted_ms"]
    first, second, combined = (copy.deepcopy(graph) for _ in range(3))
    first["nodes"][0]["ms"] = combined["nodes"][0]["ms"] = 5
    second["nodes"][1]["ms"] = combined["nodes"][1]["ms"] = 3
    individual_sum = (baseline - replay(first)["predicted_ms"]) + (baseline - replay(second)["predicted_ms"])
    joint_gain = baseline - replay(combined)["predicted_ms"]
    assert individual_sum == 2
    assert joint_gain == 5


def test_replay_requires_resource_order():
    with pytest.raises(ValueError):
        replay({"nodes": []})


def test_replay_rejects_cycle():
    with pytest.raises(CycleError):
        replay({"resource_order_confirmed": True, "nodes": [{"id": "a", "ms": 1, "deps": ["a"]}]})


def test_replay_rejects_missing_dep():
    with pytest.raises(ValueError):
        replay({"resource_order_confirmed": True, "nodes": [{"id": "a", "ms": 1, "deps": ["b"]}]})


@pytest.fixture
def source_evidence(evidence):
    evidence["assessment_mode"] = "source_only"
    for field in ("benchmark", "diagnostic", "coverage"):
        del evidence[field]
    for module in evidence["modules"]:
        module["measured"] = False
        del module["exclusive_ms"]
    evidence["scope"]["weight_revision"] = {"status": "unavailable", "reason": "Target snapshot absent"}
    return evidence


def test_source_only_reports_unknown_without_synthetic_timings(source_evidence):
    source_evidence["modules"][2]["candidates"] = [candidate()]
    result = assess(source_evidence)
    assert result["assessment_mode"] == "source_only"
    assert result["L0"]["provenance_valid"]
    assert result["L0"]["benchmark"]["noise_proxy_pct"] is None
    assert result["diagnostic"] is None
    assert result["coverage"]["uncovered_ms"] is None
    assert result["L6"]["performance_decision"] == "NEED_EVIDENCE"
    assert not result["L6"]["next_candidates"]
    assert result["L6"]["source_hypotheses"][0]["id"] == "hoist"
    assert all(m["execution_state"] == "UNKNOWN" for m in result["modules"])
    assert result["modules"][2]["work_state"] == "HYPOTHESES_PRESENT"
    assert result["modules"][0]["work_state"] == "REVIEWED"
    assert not result["L6"]["stop_authorized"]


@pytest.mark.parametrize("field", ["benchmark", "diagnostic", "coverage", "scoped_stop"])
def test_source_only_rejects_runtime_and_stop_claims(source_evidence, field):
    source_evidence[field] = {}
    with pytest.raises(ValueError, match="source_only cannot contain"):
        assess(source_evidence)


@pytest.mark.parametrize(
    "field,value",
    [("measured", True), ("exclusive_ms", 1), ("reference", {}), ("metrics", {"mfu": 0.9})],
)
def test_source_only_rejects_module_measurements(source_evidence, field, value):
    source_evidence["modules"][0][field] = value
    with pytest.raises(ValueError, match="cannot contain runtime measurements"):
        assess(source_evidence)


def test_unmeasured_hypothesis_requires_valid_evidence(source_evidence):
    c = candidate()
    c["evidence"] = ["absent"]
    source_evidence["modules"][2]["candidates"] = [c]
    with pytest.raises(ValueError, match="unknown evidence"):
        assess(source_evidence)


def test_unmeasured_hypothesis_cannot_claim_gain(source_evidence):
    c = candidate()
    c["impact"] = {"type": "scenario", "range_ms": [1, 2], "assumptions": ["unverified"]}
    source_evidence["modules"][2]["candidates"] = [c]
    with pytest.raises(ValueError, match="cannot carry quantified impact"):
        assess(source_evidence)


def test_omni_context_survives_report(source_evidence):
    context = {"component": "DiT", "stage_id": 0, "phase": "denoise", "source_paths": ["reviewed-source.py"]}
    source_evidence["modules"][2]["context"] = context
    result = assess(source_evidence)
    assert result["modules"][2]["context"] == context
    assert str(context) in render(result)


def test_source_only_preserves_na_and_rejects_false_saturation(source_evidence):
    source_evidence["modules"][4].update(applicable=False, reason="Single-rank target")
    source_evidence.update(policy_predeclared=True, budget_exhausted=True)
    result = assess(source_evidence)
    assert result["layer_states"]["L5"] == "NOT_APPLICABLE"
    assert result["L6"]["performance_decision"] == "NEED_EVIDENCE"
    assert "runtime_measurements" in result["L6"]["stop_blockers"]


@pytest.mark.parametrize("field", ["benchmark", "diagnostic"])
def test_measured_mode_does_not_silently_fall_back(evidence, field):
    evidence[field] = None
    with pytest.raises(ValueError, match="measured mode requires"):
        assess(evidence)


def test_unknown_mode_is_rejected(evidence):
    evidence["assessment_mode"] = "auto"
    with pytest.raises(ValueError, match="Unknown assessment_mode"):
        assess(evidence)


@pytest.mark.parametrize("status", [None, "partial", "unknown"])
def test_closed_ledger_does_not_prove_collection_complete(evidence, status):
    evidence.update(scoped_stop=stop_argument(), policy_predeclared=True)
    if status is None:
        del evidence["diagnostic"]["timeline_coverage"]
    else:
        evidence["diagnostic"]["timeline_coverage"]["status"] = status
    result = assess(evidence)
    assert result["coverage"]["complete"]
    assert result["L6"]["performance_decision"] == "NEED_EVIDENCE"
    assert "timeline_coverage" in result["L6"]["stop_blockers"]


def test_limited_reference_cannot_support_stop(evidence):
    evidence.update(scoped_stop=stop_argument(), policy_predeclared=True)
    evidence["modules"][0]["reference"] = {
        "kind": "measured_reference",
        "current_ms": 100,
        "time_ms_range": [95, 100],
        "comparability": "limited",
        "evidence": ["raw"],
    }
    result = assess(evidence)
    assert result["modules"][0]["execution_state"] == "NEAR_REFERENCE_LIMITED"
    assert result["component_summary"]["near_reference"] == []
    assert result["component_summary"]["limited_or_unresolved_reference"] == ["m1"]
    assert "references_resolved" in result["L6"]["stop_blockers"]


def test_summary_preserves_efficiency_and_work_opportunity(evidence):
    m = evidence["modules"][0]
    m["reference"] = {
        "kind": "measured_reference",
        "current_ms": 100,
        "time_ms_range": [95, 100],
        "comparability": "matched",
        "evidence": ["raw"],
    }
    m["candidates"] = [candidate()]
    result = assess(evidence)["component_summary"]
    assert result["near_reference"] == ["m1"]
    assert result["optimization_candidates"] == ["m1"]
    assert "m2" in result["unknown_execution"]


def test_timeline_audit_requires_evidence(evidence):
    evidence["diagnostic"]["timeline_coverage"]["evidence"] = ["absent"]
    with pytest.raises(ValueError, match="timeline coverage"):
        assess(evidence)


def test_null_reference_in_source_only_stays_unknown(source_evidence):
    source_evidence["modules"][0]["reference"] = None
    result = assess(source_evidence)
    assert "m1" in result["component_summary"]["unknown_execution"]
    assert result["component_summary"]["near_reference"] == []
