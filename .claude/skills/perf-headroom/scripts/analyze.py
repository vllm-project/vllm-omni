# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Single CPU-only entry for layered headroom assessment; see references/evidence.md."""

import argparse
import gzip
import hashlib
import json
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

from trace_math import number, replay, trace_ledger

LAYERS = {f"L{i}" for i in range(7)}
POLICY = {
    "minimum_useful_gain_pct": 2.0,
    "max_cohort_drift_pct": 2.0,
    "max_sample_spread_pct": 3.0,
    "max_noise_to_useful_gain_ratio": 0.25,
    "min_samples_per_cohort": 5,
    "min_cohorts": 2,
    "near_reference_ratio": 0.9,
    "low_leverage_pct": 2.0,
}

CARD_FIELDS = (
    "mechanism",
    "source_paths",
    "applicability",
    "counter_evidence",
    "alternatives",
    "expected_effect",
    "risks",
    "quality_gate",
    "overlap_group",
)


def detail_checks(bundle: dict, rows: list[dict], known: set[str]) -> dict:
    """Validate the submitted deep-analysis contract, not its scientific truth."""
    depth = bundle.get("analysis_depth", "triage")
    if depth not in {"triage", "detailed"}:
        raise ValueError("Invalid analysis_depth")
    missing = []
    reviews = bundle.get("layer_reviews", {})
    for layer in sorted(LAYERS - {"L0", "L6"}):
        review = reviews.get(layer)
        if not review:
            missing.append(f"{layer}: missing layer review")
            continue
        evidence_refs(review, known, f"{layer} review")
        for field in ("verdict", "confidence", "mature", "opportunities", "unknowns", "next_experiment"):
            if not review.get(field):
                missing.append(f"{layer}: missing {field}")
        if review.get("confidence") not in {"high", "medium", "low"}:
            raise ValueError("Layer confidence must be high, medium or low")
    for row in rows:
        for card in row.get("candidates", []):
            absent = [key for key in CARD_FIELDS if not card.get(key)]
            if card.get("priority") not in {"P0", "P1", "P2"}:
                absent.append("priority")
            if card.get("confidence") not in {"high", "medium", "low"}:
                absent.append("confidence")
            if absent:
                missing.append(f"{row['id']}/{card['id']}: incomplete recommendation: {', '.join(absent)}")
        if row["layer"] != "L4" or not row["applicable"]:
            continue
        audit = row.get("kernel_audit")
        if not audit:
            missing.append(f"{row['id']}: missing kernel population/counter audit")
            continue
        evidence_refs(audit, known, "kernel audit")
        if audit.get("basis") not in {"in_model_ncu", "isolated_ncu", "timeline_only", "not_required"}:
            raise ValueError("Invalid kernel audit basis")
        for field in ("population", "selection", "limitations", "counter_findings"):
            if not audit.get(field):
                missing.append(f"{row['id']}: missing kernel audit {field}")
        if audit["basis"] in {"isolated_ncu", "timeline_only"}:
            missing.append(f"{row['id']}: in-model counter coverage unresolved ({audit['basis']})")
        if audit["basis"] == "not_required" and not audit.get("reason"):
            missing.append(f"{row['id']}: counter omission needs a leverage-based reason")
    return {"depth": depth, "complete": depth == "detailed" and not missing, "missing": missing}


def freshness_check(bundle: dict) -> dict:
    """Reject explicit stale/mixed runtime provenance in a fresh-only campaign."""
    campaign = bundle.get("campaign")
    if not campaign:
        return {"status": "not_requested", "reason": "No campaign freshness contract"}
    if campaign.get("reuse_policy") not in {"fresh_only", "declared_reuse"}:
        raise ValueError("Invalid campaign reuse_policy")
    if not campaign.get("id") or not campaign.get("evaluator_revision"):
        raise ValueError("Campaign requires id and evaluator_revision")
    start = datetime.fromisoformat(campaign["started_at_utc"].replace("Z", "+00:00"))
    if start.utcoffset() is None:
        raise ValueError("Campaign timestamp requires timezone")
    problems = []
    runtime_ids = set()
    for field in ("benchmark", "diagnostic"):
        runtime_ids.update((bundle.get(field) or {}).get("evidence", []))
    for artifact in bundle["artifacts"]:
        if artifact.get("kind") not in {"measurement", "source", "method"}:
            problems.append(f"{artifact['id']}: undeclared artifact kind")
        if artifact["id"] in runtime_ids and artifact.get("kind") != "measurement":
            problems.append(f"{artifact['id']}: runtime evidence must be a measurement")
        if artifact.get("kind") != "measurement":
            continue
        if not artifact.get("collected_at_utc"):
            problems.append(f"{artifact['id']}: missing collection timestamp")
            continue
        collected = datetime.fromisoformat(artifact["collected_at_utc"].replace("Z", "+00:00"))
        if collected.utcoffset() is None:
            raise ValueError("Collection timestamp requires timezone")
        if campaign["reuse_policy"] == "fresh_only" and (artifact.get("run_id") != campaign["id"] or collected < start):
            problems.append(f"{artifact['id']}: historical or mixed-campaign measurement")
    return {
        "status": "invalid"
        if problems
        else "declared_fresh"
        if campaign["reuse_policy"] == "fresh_only"
        else "declared_reuse",
        "problems": problems,
        "campaign": campaign,
        "note": "Manifest check only; an analyst must verify raw collection lineage, not file modification time.",
    }


def read_json(path):
    def invalid(token):
        raise ValueError(f"Non-finite JSON value: {token}")

    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream, parse_constant=invalid)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def evidence_refs(value, known, label):
    refs = value.get("evidence", [])
    if not refs or any(ref not in known for ref in refs):
        raise ValueError(f"{label}: missing or unknown evidence ids")


def artifact_checks(bundle, root):
    records, known = [], set()
    for artifact in bundle["artifacts"]:
        key = artifact["id"]
        if key in known:
            raise ValueError(f"Duplicate artifact: {key}")
        known.add(key)
        path = (root / artifact["path"]).resolve()
        try:
            actual = digest(path)
            valid = actual == artifact["sha256"]
            error = None if valid else "SHA256 mismatch"
        except OSError as exc:
            actual, valid, error = None, False, str(exc)
        records.append(
            {**artifact, "resolved_path": str(path), "actual_sha256": actual, "valid": valid, "error": error}
        )
    if not known:
        raise ValueError("At least one evidence artifact is required")
    return records, known


def benchmark_stats(benchmark, policy, known):
    if benchmark is None:
        return {
            "stable_for_policy": False,
            "precise_for_policy": False,
            "noise_proxy_pct": None,
            "cohort_medians_ms": [],
            "cohort_counts": [],
            "cohort_drift_pct": None,
            "sample_range_pct": [],
            "note": "No runtime benchmark collected; source-only assessment.",
        }
    evidence_refs(benchmark, known, "benchmark")
    cohorts = benchmark["cohorts_ms"]
    medians, spreads, counts = [], [], []
    for cohort in cohorts:
        if not cohort:
            raise ValueError("Empty benchmark cohort")
        samples = [number(x, "benchmark sample", 1e-12) for x in cohort]
        median = statistics.median(samples)
        medians.append(median)
        counts.append(len(samples))
        spreads.append(100 * (max(samples) - min(samples)) / median)
    drift = 100 * (max(medians) - min(medians)) / min(medians) if medians else None
    stable = (
        benchmark.get("profiled") is False
        and len(cohorts) >= policy["min_cohorts"]
        and min(counts, default=0) >= policy["min_samples_per_cohort"]
        and drift <= policy["max_cohort_drift_pct"]
        and max(spreads, default=100) <= policy["max_sample_spread_pct"]
    )
    noise = max([drift or 0, *spreads]) if medians else None
    return {
        "stable_for_policy": stable,
        "noise_proxy_pct": noise,
        "precise_for_policy": stable
        and noise <= policy["minimum_useful_gain_pct"] * policy["max_noise_to_useful_gain_ratio"],
        "cohort_medians_ms": medians,
        "cohort_counts": counts,
        "cohort_drift_pct": drift,
        "sample_range_pct": spreads,
        "note": "Descriptive spreads, NOT confidence intervals or effect-size tests.",
    }


def validate_module_hypotheses(module, known):
    """Source review can survive missing runtime, but quantified impact cannot."""
    review = module.get("structural_review", {})
    if review.get("complete") is True:
        evidence_refs(review, known, "structural review")
    candidates = module.get("candidates", [])
    for candidate in candidates:
        evidence_refs(candidate, known, "candidate")
        if not candidate.get("next_action") or not candidate.get("validation"):
            raise ValueError("Candidate requires next_action and validation")
        impact = candidate.get("impact")
        if impact:
            if not module["measured"]:
                raise ValueError("Unmeasured hypotheses cannot carry quantified impact")
            if impact["type"] not in {"reference_gap", "scenario", "model_prediction", "validated_gain"}:
                raise ValueError("Invalid impact type")
            low, high = impact["range_ms"]
            number(low, "impact low", -1e100)
            number(high, "impact high", low)
            if not impact.get("assumptions"):
                raise ValueError("Impact assumptions required")
            if impact["type"] == "validated_gain":
                evidence_refs(impact, known, "validated gain")
                if impact.get("quality_passed") is not True or impact.get("paired_unprofiled") is not True:
                    raise ValueError("Validated gain needs paired unprofiled and quality evidence")
    return review, candidates


def module_state(module, policy, known, window_ms):
    row = dict(module)
    if module["layer"] not in {"L1", "L2", "L3", "L4", "L5"}:
        raise ValueError("Module layer must be L1-L5")
    if type(module["applicable"]) is not bool or type(module["measured"]) is not bool:
        raise ValueError("applicable and measured must be booleans")
    if not module["applicable"]:
        if not module.get("reason") or module.get("candidates"):
            raise ValueError("N/A requires a reason and cannot contain active candidates")
        row.update(execution_state="NOT_APPLICABLE", work_state="NOT_APPLICABLE", state="NOT_APPLICABLE")
        return row
    review, candidates = validate_module_hypotheses(module, known)
    if not module["measured"]:
        if module.get("evidence"):
            evidence_refs(module, known, module["id"])
        work = "HYPOTHESES_PRESENT" if candidates else "REVIEWED" if review.get("complete") is True else "UNKNOWN"
        row.update(execution_state="UNKNOWN", work_state=work, state="UNKNOWN")
        return row
    evidence_refs(module, known, module["id"])
    share = None
    if module.get("exclusive_ms") is not None:
        ms = number(module["exclusive_ms"], "exclusive_ms")
        if ms > window_ms + 1e-6:
            raise ValueError("Exclusive module cost exceeds its diagnostic window")
        share = 100 * ms / window_ms
    row["request_share_pct"] = share
    execution = "MEASURED_NO_REFERENCE"
    if module.get("reference"):
        ref = module["reference"]
        evidence_refs(ref, known, "reference")
        if ref["kind"] not in {"measured_reference", "conditional_resource_model"}:
            raise ValueError("Kernel references are not accepted as certified hard lower bounds")
        current = number(ref["current_ms"], "reference current_ms", 1e-12)
        lo, hi = ref["time_ms_range"]
        number(lo, "reference lower", 1e-12)
        number(hi, "reference upper", lo)
        if ref["comparability"] not in {"matched", "limited", "unknown"}:
            raise ValueError("Invalid reference comparability")
        row["reference_ratio_range"] = [lo / current, hi / current]
        row["reference_gap_ms_range"] = [current - hi, current - lo]
        if ref["comparability"] == "unknown":
            execution = "REFERENCE_NOT_COMPARABLE"
        elif hi > current * (1 + policy["max_cohort_drift_pct"] / 100):
            execution = "REFERENCE_RECHECK"
        elif lo / current >= policy["near_reference_ratio"]:
            execution = "NEAR_REFERENCE" if ref["comparability"] == "matched" else "NEAR_REFERENCE_LIMITED"
        else:
            execution = "REFERENCE_GAP"
    work = "CANDIDATES_PRESENT" if candidates else "REVIEWED" if review.get("complete") is True else "UNKNOWN"
    if candidates:
        state = "ACTIONABLE_CANDIDATE"
    elif execution.startswith("NEAR_REFERENCE"):
        state = execution
    elif share is not None and share <= policy["low_leverage_pct"]:
        state = "LOW_OBSERVED_LEVERAGE"
    else:
        state = execution
    row.update(execution_state=execution, work_state=work, state=state)
    return row


def coverage_checks(bundle, rows, known, window):
    """Check submitted coverage accounting, not the truth of semantic mappings."""
    ledger = (bundle.get("diagnostic") or {}).get("exclusive_ledger_ms")
    coverage = bundle.get("coverage")
    if ledger is None or not coverage:
        return {"complete": False, "uncovered_ms": window, "reason": "Missing ledger or coverage bindings"}
    evidence_refs(coverage, known, "coverage")
    bindings = coverage["ledger_modules"]
    unknown = set(coverage.get("unclassified_labels", [])) | {"unattributed"}
    if set(bindings) - set(ledger) or (unknown - {"unattributed"}) - set(ledger):
        raise ValueError("Coverage names an unknown ledger bucket")
    modules = {row["id"]: row for row in rows}
    assigned = {key: 0.0 for key in modules}
    uncovered = {}
    for label, cost in ledger.items():
        ids = bindings.get(label, [])
        if not isinstance(ids, list) or len(ids) != len(set(ids)) or any(key not in modules for key in ids):
            raise ValueError("Coverage requires unique, known module ids")
        reviewed = bool(ids) and all(
            modules[key]["applicable"] and modules[key]["measured"] and modules[key]["work_state"] == "REVIEWED"
            for key in ids
        )
        if cost > 0 and (label in unknown or not reviewed):
            uncovered[label] = cost
        for key in ids:
            assigned[key] += cost
    for key, cost in assigned.items():
        declared = modules[key].get("exclusive_ms")
        if declared is not None and cost > declared + 1e-6:
            raise ValueError("Coverage cost exceeds module exclusive_ms")
    return {
        "complete": not uncovered,
        "uncovered_ms": sum(uncovered.values()),
        "uncovered_buckets_ms": uncovered,
        "note": "Bindings require source/trace review; cross-layer module costs are not summed.",
    }


def timeline_coverage_check(diagnostic, known):
    """Check the submitted collection audit, not profiler completeness itself."""
    audit = (diagnostic or {}).get("timeline_coverage")
    if audit is None:
        return {"status": "unknown", "reason": "No trace collection audit supplied"}
    if audit.get("status") not in {"complete", "partial", "unknown"}:
        raise ValueError("Invalid timeline coverage status")
    if not audit.get("reason"):
        raise ValueError("Timeline coverage requires an explanation of event/window coverage")
    evidence_refs(audit, known, "timeline coverage")
    return dict(audit)


def component_summary(rows):
    """Non-exclusive views: efficient execution can coexist with redundant work."""
    return {
        "near_reference": [m["id"] for m in rows if m["execution_state"] == "NEAR_REFERENCE"],
        "limited_or_unresolved_reference": [
            m["id"]
            for m in rows
            if (m.get("reference") or {}).get("comparability") in {"limited", "unknown"}
            or m["execution_state"] == "REFERENCE_RECHECK"
        ],
        "low_observed_leverage": [m["id"] for m in rows if m["state"] == "LOW_OBSERVED_LEVERAGE"],
        "optimization_candidates": [m["id"] for m in rows if m["state"] == "ACTIONABLE_CANDIDATE"],
        "unknown_execution": [m["id"] for m in rows if m["execution_state"] in {"UNKNOWN", "MEASURED_NO_REFERENCE"}],
        "unreviewed_work": [
            m["id"] for m in rows if m["applicable"] and m.get("structural_review", {}).get("complete") is not True
        ],
    }


def check_repo(repo, scope):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    paths = ["vllm_omni", "requirements", "pyproject.toml"]
    sha = git("rev-parse", "HEAD")
    diff = git("diff", "HEAD", "--", *paths)
    untracked = git("ls-files", "--others", "--exclude-standard", "--", *paths)
    return {
        "sha": sha,
        "matches": sha == scope["code_sha"] and not diff and not untracked,
        "runtime_tracked_diff": bool(diff),
        "runtime_untracked": bool(untracked),
        "note": "Runtime overrides/dependencies outside repo require separate evidence.",
    }


def assess(bundle, root=Path("."), layers=None, repo=None):
    if bundle["schema_version"] != 1:
        raise ValueError("Unsupported evidence schema")
    mode = bundle.get("assessment_mode", "measured")
    if mode not in {"measured", "source_only"}:
        raise ValueError("Unknown assessment_mode")
    if mode == "source_only":
        for field in ("benchmark", "diagnostic", "coverage", "scoped_stop"):
            if bundle.get(field) is not None:
                raise ValueError(f"source_only cannot contain {field}; omit it or use null")
        for module in bundle["modules"]:
            if (
                module["measured"] is not False
                or any(module.get(key) is not None for key in ("exclusive_ms", "reference"))
                or module.get("metrics")
            ):
                raise ValueError("source_only modules cannot contain runtime measurements")
    scope = bundle["scope"]
    for key in (
        "model",
        "weight_revision",
        "code_sha",
        "hardware",
        "workload",
        "runtime",
        "boundary",
        "quality_contract",
    ):
        if not scope.get(key):
            raise ValueError(f"Missing scope.{key}")
    selected = set(layers) if layers is not None else LAYERS
    if not selected or not selected <= LAYERS:
        raise ValueError("Unknown or empty layer selection")
    selected = selected | {"L0", "L6"}
    policy = POLICY | bundle.get("policy", {})
    for key, value in policy.items():
        if key not in POLICY:
            raise ValueError(f"Unknown policy: {key}")
        number(value, key, 1e-12)
    for key in ("min_samples_per_cohort", "min_cohorts"):
        if type(policy[key]) is not int or policy[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if policy["near_reference_ratio"] > 1 or policy["minimum_useful_gain_pct"] >= 100:
        raise ValueError("Invalid ratio/gain policy")
    checks, known = artifact_checks(bundle, root)
    freshness = freshness_check(bundle)
    provenance = bundle["provenance"]
    evidence_refs(provenance, known, "provenance")
    valid = (
        all(a["valid"] for a in checks)
        and provenance.get("scope_verified") is True
        and freshness["status"] != "invalid"
    )
    repo_result = check_repo(repo, scope) if repo else None
    if repo_result:
        valid = valid and repo_result["matches"]
    diagnostic, window = None, None
    if mode == "measured":
        if bundle.get("benchmark") is None or bundle.get("diagnostic") is None:
            raise ValueError("measured mode requires benchmark and diagnostic evidence")
        diagnostic = bundle["diagnostic"]
        evidence_refs(diagnostic, known, "diagnostic")
        window = number(diagnostic["window_ms"], "diagnostic window", 1e-12)
        ledger = diagnostic.get("exclusive_ledger_ms")
        if ledger is not None:
            total = sum(number(v, "ledger value") for v in ledger.values())
            if abs(total - window) > max(1e-5, window * 1e-8):
                raise ValueError("Exclusive ledger does not close to diagnostic window")
    stats = benchmark_stats(bundle.get("benchmark"), policy, known)
    timeline_coverage = timeline_coverage_check(diagnostic, known)
    all_rows = [module_state(m, policy, known, window) for m in bundle["modules"]]
    if len({r["id"] for r in all_rows}) != len(all_rows):
        raise ValueError("Duplicate module id")
    coverage = coverage_checks(bundle, all_rows, known, window)
    rows = [m for m in all_rows if m["layer"] in selected]
    details = detail_checks(bundle, rows, known)
    layer_states = {}
    for layer in sorted(selected - {"L0", "L6"}):
        members = [m for m in rows if m["layer"] == layer]
        if not members:
            state = "UNKNOWN"
        elif all(m["state"] == "NOT_APPLICABLE" for m in members):
            state = "NOT_APPLICABLE"
        elif any(m["state"] == "ACTIONABLE_CANDIDATE" for m in members):
            state = "OPPORTUNITIES_PRESENT"
        elif any(m["state"] == "UNKNOWN" for m in members):
            state = "INCOMPLETE"
        else:
            state = "ASSESSED_WITH_LIMITS"
        layer_states[layer] = state
    candidates = [
        {"module": m["id"], **c} for m in rows for c in m.get("candidates", []) if m["state"] == "ACTIONABLE_CANDIDATE"
    ]
    stop = bundle.get("scoped_stop")
    stop_supported = False
    if stop:
        evidence_refs(stop, known, "scoped stop")
        upper = number(stop["remaining_gain_upper_pct"], "scoped remaining gain")
        if not stop.get("search_scope") or not stop.get("assumptions"):
            raise ValueError("Scoped stop requires search scope and assumptions")
        stop_supported = (
            upper + (stats["noise_proxy_pct"] or 0) < policy["minimum_useful_gain_pct"]
            and bundle.get("policy_predeclared") is True
        )
    covered = (
        selected == LAYERS
        and len(layer_states) == 5
        and not any(s in {"UNKNOWN", "INCOMPLETE"} for s in layer_states.values())
        and all(m["work_state"] in {"REVIEWED", "NOT_APPLICABLE"} for m in rows)
    )
    stop_checks = {
        "runtime_measurements": mode == "measured",
        "provenance": valid,
        "measurement_precision": stats["precise_for_policy"],
        "layer_and_structural_coverage": covered,
        "ledger_coverage": coverage["complete"],
        "timeline_coverage": timeline_coverage["status"] == "complete",
        "references_resolved": not any(
            m["execution_state"] in {"REFERENCE_RECHECK", "REFERENCE_NOT_COMPARABLE"}
            or (m.get("reference") or {}).get("comparability") == "limited"
            for m in rows
        ),
        "residual_argument_and_margin": stop_supported,
        "no_active_candidates": not candidates,
        "detailed_contract": details["complete"] if details["depth"] == "detailed" else True,
    }
    if not valid or mode == "source_only":
        decision = "NEED_EVIDENCE"
    elif bundle.get("budget_exhausted") is True:
        decision = "STOP_BUDGET"
    elif candidates:
        decision = "CONTINUE_TARGETED_OPTIMIZATION"
    elif all(stop_checks.values()):
        decision = "STOP_REVIEW_REQUIRED"
    else:
        decision = "NEED_EVIDENCE"
    return {
        "schema_version": 1,
        "assessment_mode": mode,
        "detail_checks": details,
        "freshness": freshness,
        "layer_reviews": bundle.get("layer_reviews", {}),
        "scope": scope,
        "policy": policy,
        "policy_predeclared": bundle.get("policy_predeclared", False),
        "selected_layers": sorted(selected),
        "artifacts": checks,
        "repository_check": repo_result,
        "L0": {
            "provenance_valid": valid,
            "evidence_origin": provenance.get("origin", "unspecified"),
            "benchmark": stats,
            "limitations": bundle.get("limitations", []),
        },
        "diagnostic": diagnostic,
        "coverage": coverage,
        "timeline_coverage": timeline_coverage,
        "component_summary": component_summary(rows),
        "layer_states": layer_states,
        "modules": rows,
        "L6": {
            "performance_decision": decision,
            "quality_status": bundle.get("quality_status", "UNKNOWN"),
            "next_candidates": candidates,
            "source_hypotheses": [
                {"module": m["id"], **c} for m in rows if not m["measured"] for c in m.get("candidates", [])
            ],
            "scoped_stop": stop,
            "stop_checks": stop_checks,
            "stop_blockers": [key for key, passed in stop_checks.items() if not passed],
            "stop_authorized": False,
            "certified_remaining_gain_upper_pct": None,
            "note": "No stop authorization or global optimum certification. Independent review required.",
        },
    }


def render(report):
    uncovered = report["coverage"]["uncovered_ms"] if report["diagnostic"] else "UNKNOWN（无运行窗口）"
    lines = [
        "vLLM-Omni 分层性能状态报告",
        "",
        f"模型: {report['scope']['model']}",
        f"源码: {report['scope']['code_sha']}",
        f"边界: {report['scope']['boundary']}",
        f"分析模式: {report['assessment_mode']}",
        "",
        "L0: 实验身份、边界和测量有效性",
        f"完整实验范围: {report['scope']}",
        f"L0 来源有效: {report['L0']['provenance_valid']}",
        f"L0 测量满足当前稳定性策略: {report['L0']['benchmark']['stable_for_policy']}",
        f"基准统计及噪声口径: {report['L0']['benchmark']}",
        f"数据来源: {report['L0']['evidence_origin']}",
        f"采集新鲜度: {report['freshness']}",
        f"深度分析契约: {report['detail_checks']}",
        f"未完成结构审查的诊断账本时间(ms，非缺失trace): {uncovered}",
        f"时间线采集覆盖: {report['timeline_coverage']['status']} — {report['timeline_coverage']['reason']}",
        f"停止建议缺失条件: {', '.join(report['L6']['stop_blockers'])}",
        "停止授权: 无；通过自动检查仍须独立审查",
        f"质量状态: {report['L6']['quality_status']}",
        "",
    ]
    for layer, state in report["layer_states"].items():
        lines.extend([f"{layer}: {state}", ""])
        review = report["layer_reviews"].get(layer, {})
        for key, value in review.items():
            lines.append(f"  {key}: {value}")
        for row in report["modules"]:
            if row["layer"] != layer:
                continue
            lines += [
                f"  {row['id']}: {row['state']}",
                f"    执行效率: {row['execution_state']} / 工作量: {row['work_state']}",
            ]
            if row.get("context"):
                lines.append(f"    Omni 组件定位: {row['context']}")
            if row.get("metrics"):
                lines.append(f"    指标及口径: {row['metrics']}")
            if row.get("kernel_audit"):
                lines.append(f"    算子采样/计数器审查: {row['kernel_audit']}")
            if row.get("request_share_pct") is not None:
                lines.append(f"    诊断请求互斥占比: {row['request_share_pct']:.3f}%")
            if "reference_ratio_range" in row:
                lines.append(f"    参考成本/当前成本: {row['reference_ratio_range']}")
                lines.append(f"    参考差距(ms，非可兑现收益): {row['reference_gap_ms_range']}")
            lines += [
                f"    观察: {row.get('observation', row.get('reason', ''))}",
                f"    证据: {', '.join(row.get('evidence', []))}",
            ]
            for c in row.get("candidates", []):
                lines += [
                    f"    {'候选' if row['measured'] else '待验证源码假设'}: {c['id']}",
                    f"      下一步: {c['next_action']}",
                    f"      验证: {c['validation']}",
                    f"      影响: {c.get('impact', '尚未量化')}",
                ]
                for key in (*CARD_FIELDS, "priority", "confidence"):
                    if c.get(key):
                        lines.append(f"      {key}: {c[key]}")
            lines.append("")
    lines += [
        "L6: 剩余空间、建议优先级和停止条件",
        f"性能决策: {report['L6']['performance_decision']}",
        f"停止建议缺失条件: {report['L6']['stop_blockers']}",
        "优化建议队列（不等于已验证收益）:",
    ]
    for card in sorted(report["L6"]["next_candidates"], key=lambda c: c.get("priority", "P2")):
        lines.append(f"  {card.get('priority', '未排序')} {card['module']}/{card['id']}: {card['next_action']}")
    lines.append("组件概览（视角可重叠，不相加）：")
    for category, ids in report["component_summary"].items():
        lines.append(f"  {category}: {', '.join(ids) or '无'}")
    lines += [
        "限制/待补证据:",
        *[f"- {x}" for x in report["L0"]["limitations"]],
        "",
        "各项参考差距、情景和预测不相加；未证明全局最优或总剩余空间上界。",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    assess_parser = modes.add_parser("assess", help="Validate evidence and report L0-L6 state")
    assess_parser.add_argument("--evidence", type=Path, required=True)
    assess_parser.add_argument("--output", type=Path, required=True)
    assess_parser.add_argument("--layers", default="all")
    assess_parser.add_argument("--repo", type=Path, help="Check current runtime tree against measured commit")
    trace = modes.add_parser("trace", help="Account for one device; timestamps in microseconds")
    trace.add_argument("--trace", type=Path, required=True)
    trace.add_argument("--device", required=True)
    trace.add_argument("--start-us", type=float)
    trace.add_argument("--end-us", type=float)
    trace.add_argument("--rules", type=Path)
    trace.add_argument("--output", type=Path, required=True)
    graph = modes.add_parser("replay", help="Fixed-resource-order DAG sensitivity, not a hard bound")
    graph.add_argument("--graph", type=Path, required=True)
    graph.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "assess":
        layers = None if args.layers == "all" else args.layers.split(",")
        result = assess(read_json(args.evidence), args.evidence.resolve().parent, layers, args.repo)
        result["input_sha256"] = digest(args.evidence)
        result["evaluator_sha256"] = {
            p.name: digest(p) for p in (Path(__file__), Path(__file__).with_name("trace_math.py"))
        }
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "report.txt").write_text(render(result), encoding="utf-8")
        output = args.output / "report.json"
    elif args.mode == "trace":
        if (args.start_us is None) != (args.end_us is None):
            parser.error("Pass both --start-us and --end-us, or neither")
        opener = gzip.open if args.trace.suffix == ".gz" else open
        with opener(args.trace, "rt", encoding="utf-8") as stream:
            result = trace_ledger(
                json.load(stream), args.device, args.start_us, args.end_us, read_json(args.rules) if args.rules else ()
            )
        result["trace_sha256"] = digest(args.trace)
        output = args.output
    else:
        result = replay(read_json(args.graph))
        result["graph_sha256"] = digest(args.graph)
        output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(
        json.dumps(
            {"output": str(output), "decision": result.get("L6", {}).get("performance_decision")}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
