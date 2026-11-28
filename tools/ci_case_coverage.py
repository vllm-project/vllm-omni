#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare CUDA/AMD pytest evidence within an explicit same-commit job scope."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def job_key(job: dict) -> tuple[str, str, str]:
    platform, tier, job_id = (job.get(key) for key in ("platform", "tier", "job_id"))
    if platform not in {"cuda", "amd"} or tier not in {"ready", "merge", "nightly", "weekly"}:
        raise ValueError(f"Invalid platform/tier: {platform!r}/{tier!r}")
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("Each scoped job must have a nonempty Buildkite job_id")
    return platform, tier, job_id


def policies(scope: dict, name: str) -> dict[tuple[str, str, str], str]:
    result = {}
    for entry in scope.get(name, []):
        key = entry.get("platform"), entry.get("tier"), entry.get("nodeid")
        reason = entry.get("reason")
        if key[0] not in {"cuda", "amd"} or key[1] not in {"ready", "merge", "nightly", "weekly"}:
            raise ValueError(f"Invalid {name} platform/tier")
        if not isinstance(key[2], str) or not key[2] or not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{name} entries require exact nodeids and nonempty reasons")
        if key in result:
            raise ValueError(f"Duplicate {name} entry: {key}")
        result[key] = reason
    return result


def compare(scope: dict, artifact_dir: Path) -> dict:
    commit = scope.get("commit", "")
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("Scope requires a full lowercase commit SHA")
    if scope.get("schema_version") != 1:
        raise ValueError("Unsupported scope schema_version")
    keys = [job_key(job) for job in scope.get("jobs", [])]
    expected = set(keys)
    if not keys or len(keys) != len(expected):
        raise ValueError("Scope requires a nonempty list of unique jobs")
    if {key[0] for key in keys} != {"cuda", "amd"}:
        raise ValueError("Scope must include both CUDA and AMD jobs")
    if {key[1] for key in keys if key[0] == "cuda"} != {key[1] for key in keys if key[0] == "amd"}:
        raise ValueError("Scope must include both platforms for every compared tier")
    unsupported = policies(scope, "unsupported")
    allowed_skips = policies(scope, "allowed_skips")
    seen: set[tuple[str, str, str]] = set()
    selected: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    phases: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    attempts: dict[tuple[str, str, str, str], list[list[dict]]] = defaultdict(list)
    blocking: dict[tuple[str, str, str], bool] = {}
    problems: list[str] = []
    for path in sorted(artifact_dir.rglob("*.jsonl")):
        lines = path.read_text(encoding="utf-8").splitlines()
        try:
            events = [json.loads(line) for line in lines]
        except json.JSONDecodeError as error:
            raise ValueError(f"Truncated or malformed report: {path}") from error
        if not events or events[0].get("event") != "session":
            raise ValueError(f"Missing session header: {path}")
        header = events[0]
        key = job_key(header)
        if key not in expected:
            continue
        if header.get("schema_version") != 1 or header.get("commit") != commit:
            raise ValueError(f"Report revision/schema mismatch: {path}")
        grade = header.get("blocking")
        if grade not in {"true", "false"}:
            raise ValueError(f"Missing blocking status: {path}")
        if key in blocking and blocking[key] != (grade == "true"):
            raise ValueError(f"Conflicting blocking status for {key}")
        blocking[key] = grade == "true"
        # Collection-only guards are useful evidence, but never count as test execution.
        if header.get("collect_only"):
            continue
        seen.add(key)
        if events[-1].get("event") != "finish":
            problems.append(f"Incomplete session: {key} ({path.name})")
        elif events[-1].get("exitstatus") != 0:
            problems.append(f"Failed session: {key} (exit {events[-1].get('exitstatus')})")
        local_phases: dict[str, list[dict]] = defaultdict(list)
        for event in events[1:]:
            kind = event.get("event")
            if kind == "selection":
                nodeids = event.get("nodeids")
                if not isinstance(nodeids, list) or any(not isinstance(node, str) for node in nodeids):
                    raise ValueError(f"Invalid selection: {path}")
                if not nodeids:
                    problems.append(f"Empty selection: {key} ({path.name})")
                if len(nodeids) != len(set(nodeids)):
                    problems.append(f"Duplicate selected node IDs: {key} ({path.name})")
                selected[key].update(nodeids)
            elif kind == "result":
                nodeid = event.get("nodeid")
                if not isinstance(nodeid, str) or event.get("phase") not in {"setup", "call", "teardown"}:
                    raise ValueError(f"Invalid result: {path}")
                if event.get("outcome") not in {"passed", "failed", "skipped"}:
                    raise ValueError(f"Invalid result outcome: {path}")
                phases[(*key, nodeid)].append(event)
                local_phases[nodeid].append(event)
            elif kind == "collection_issue":
                problems.append(f"Collection {event.get('outcome')}: {key} {event.get('nodeid')}")
            elif kind != "finish":
                raise ValueError(f"Unknown report event {kind!r}: {path}")
        for nodeid, reports in local_phases.items():
            attempts[(*key, nodeid)].append(reports)
            present = {report["phase"] for report in reports}
            setup_passed = any(report["phase"] == "setup" and report["outcome"] == "passed" for report in reports)
            if not {"setup", "teardown"} <= present or (setup_passed and "call" not in present):
                problems.append(f"Incomplete execution: {key} {nodeid} ({path.name})")
    for key in sorted(expected - seen):
        problems.append(f"Missing execution artifact: {key}")
    for key in sorted(seen):
        if not selected[key]:
            problems.append(f"No selected nodes recorded: {key}")
    for (platform, tier, job_id, nodeid), reports in phases.items():
        key = (platform, tier, job_id)
        if nodeid not in selected[key]:
            problems.append(f"Result outside recorded selection: {key} {nodeid}")
    cases = sorted({(tier, node) for (_, tier, _), nodes in selected.items() for node in nodes})
    rows = []
    for tier, nodeid in cases:
        row: dict[str, object] = {"tier": tier, "nodeid": nodeid}
        for platform in ("cuda", "amd"):
            jobs = sorted(key for key in selected if key[:2] == (platform, tier) and nodeid in selected[key])
            reports = [report for key in jobs for report in phases[(*key, nodeid)]]
            reason = unsupported.get((platform, tier, nodeid))
            if reason and jobs:
                problems.append(f"Unsupported policy contradicts selection: {platform}/{tier} {nodeid}")
            skips = [report for report in reports if report["outcome"] == "skipped"]
            allowed = (platform, tier, nodeid) in allowed_skips
            if skips and not allowed:
                problems.append(f"Unexpected skip: {platform}/{tier} {nodeid}")
            passed = []
            for key in jobs:
                runs = attempts[(*key, nodeid)]
                if runs and all(
                    {event["phase"] for event in run} == {"setup", "call", "teardown"}
                    and all(event["outcome"] == "passed" and "xfail" not in event for event in run)
                    for run in runs
                ):
                    passed.append(key)
                if not runs:
                    problems.append(f"Incomplete execution: {key} {nodeid}")
            if not jobs:
                status = "unsupported" if reason else "absent"
            elif not any(blocking[key] for key in jobs):
                status = "nonblocking"
            elif any(blocking[key] for key in passed):
                status = "already-covered"
            else:
                status = "unverified"
            row[platform] = {
                "status": status,
                "selected_jobs": [key[2] for key in jobs],
                "blocking_jobs": [key[2] for key in jobs if blocking[key]],
                "passed_jobs": [key[2] for key in passed],
                "phase_results": dict(Counter(report["outcome"] for report in reports)),
                "skips": [{"phase": event["phase"], "reason": event.get("reason", "")} for event in skips],
                "unsupported_reason": reason,
                "allowed_skip_reason": allowed_skips.get((platform, tier, nodeid)),
            }
        rows.append(row)
    return {
        "schema_version": 1,
        "commit": commit,
        "scope_jobs": scope["jobs"],
        "valid": not problems,
        "problems": sorted(set(problems)),
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope", type=Path, required=True, help="Same-commit job inventory and explicit skip/support policy"
    )
    parser.add_argument(
        "--artifacts", type=Path, required=True, help="Directory containing downloaded case-report JSONL files"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = compare(read_object(args.scope), args.artifacts)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(report['cases'])} cases; {len(report['problems'])} evidence problems; report: {args.output}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
