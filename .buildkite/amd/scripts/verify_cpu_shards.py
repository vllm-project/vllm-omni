# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Fail unless successful baseline/candidate CPU runs execute the same node set."""

import argparse
import json
from pathlib import Path

from cpu_shard_plan import SUITE_MARKERS, profile_from_artifacts, read_object


def skipped_nodes(directories: list[Path]) -> set[str]:
    skipped = set()
    for directory in directories:
        for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("event") == "test_report" and event.get("outcome") == "skipped":
                skipped.add(event["nodeid"])
    return skipped


def verify(suite: str, baseline: list[Path], candidate: list[Path]) -> int:
    # Reuse the profile builder's complete-run, duplicate and phase validation.
    before = profile_from_artifacts(baseline, suite)
    after = profile_from_artifacts(candidate, suite)
    if len(before["sources"]) != 1 or len(after["sources"]) != 1:
        raise ValueError("Compare exactly one complete baseline and one complete candidate build")
    expected = set(before["node_seconds"])
    actual = set(after["node_seconds"])
    if expected != actual:
        raise ValueError(f"Selection changed: missing={sorted(expected - actual)}, added={sorted(actual - expected)}")
    if skipped_nodes(baseline) != skipped_nodes(candidate):
        raise ValueError("Skip outcomes changed between baseline and candidate")
    shared_plan = read_object(candidate[0] / "plan.json")
    if set(shared_plan["node_shards"]) != actual:
        raise ValueError("Plan does not cover exactly the executed node union")
    for directory in candidate:
        plan = read_object(directory / "plan.json")
        invocation = read_object(directory / "invocation.json")
        if plan != shared_plan:
            raise ValueError("Workers disagree on profile, eligible nodes or assignments")
        nodes = set(read_object(directory / "selection.json")["selected"])
        planned = {node for node, owner in plan["node_shards"].items() if owner == invocation["shard_id"]}
        if nodes != planned:
            raise ValueError(f"Execution does not match the planned shard: {directory}")
    return len(actual)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITE_MARKERS, required=True)
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    try:
        count = verify(args.suite, args.baseline, args.candidate)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(f"Verified {count} nodes: unchanged union, no duplicates, identical skip outcomes and worker plans")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
