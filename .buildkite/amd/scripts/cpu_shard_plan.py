# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Build a versioned timing profile from complete successful CPU suite artifacts."""

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.helpers.cpu_sharding import SUITE_MARKERS, duration_seconds, in_suite  # noqa: E402


def read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def profile_from_artifacts(directories: list[Path], suite: str = "model-executor") -> dict:
    samples: dict[str, list[float]] = defaultdict(list)
    runs: dict[tuple[str, str], list[tuple[int, int, set[str]]]] = defaultdict(list)
    for directory in directories:
        summary = read_object(directory / "summary.json")
        invocation = read_object(directory / "invocation.json")
        selection = read_object(directory / "selection.json")
        if (
            summary.get("schema_version") != 1
            or summary.get("pytest_exit_code") != 0
            or summary.get("collection_finished") is not True
            or summary.get("artifact_errors") != []
        ):
            raise ValueError(f"Need a completed, successful TIME-01 run: {directory}")
        if invocation.get("schema_version") != 1 or invocation.get("markers") != SUITE_MARKERS[suite]:
            raise ValueError(f"Not a {suite} CPU timing invocation: {directory}")
        metadata = invocation.get("buildkite", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid Buildkite metadata: {directory}")
        build_url, commit = metadata.get("BUILD_URL"), metadata.get("COMMIT")
        if not isinstance(build_url, str) or not build_url or not isinstance(commit, str) or not commit:
            raise ValueError(f"Missing source build URL/commit: {directory}")
        count, index = invocation.get("num_shards"), invocation.get("shard_id")
        if type(count) is not int or type(index) is not int or count < 1 or not 0 <= index < count:
            raise ValueError(f"Invalid source shard metadata: {directory}")
        nodes = selection.get("selected")
        if not isinstance(nodes, list) or any(not isinstance(node, str) or not in_suite(node, suite) for node in nodes):
            raise ValueError(f"Invalid selected {suite} nodes: {directory}")
        if len(set(nodes)) != len(nodes):
            raise ValueError(f"Duplicate selected nodes: {directory}")
        reports: dict[str, dict[str, float]] = defaultdict(dict)
        starts: Counter[str] = Counter()
        zero_calls: set[str] = set()
        calls_required: set[str] = set()
        with (directory / "events.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError(f"Invalid execution event: {directory}")
                if event.get("event") == "test_start":
                    starts[event["nodeid"]] += 1
                if event.get("event") != "test_report":
                    continue
                node, phase = event.get("nodeid"), event.get("phase")
                if node not in nodes or phase not in ("setup", "call", "teardown"):
                    raise ValueError(f"Report outside the selected nodes/phases: {directory}")
                if event.get("outcome") not in ("passed", "skipped"):
                    raise ValueError(f"Duplicate or unsuccessful phase report: {directory}")
                seconds = duration_seconds(event.get("seconds"))
                if phase in reports[node] and not (phase == "call" and node in zero_calls):
                    raise ValueError(f"Duplicate phase report: {directory}")
                # pytest 9 unittest subtests emitted zero-duration passed call
                # reports before the parent call in legacy TIME-01 artifacts.
                # Only this prefix can be replaced; a second start or a second
                # nonzero call still fails. The runner now tags subtests separately.
                reports[node][phase] = seconds
                if phase == "call":
                    if seconds == 0 and event["outcome"] == "passed":
                        zero_calls.add(node)
                    else:
                        zero_calls.discard(node)
                if phase == "setup" and event["outcome"] == "passed":
                    calls_required.add(node)
        if starts != Counter({node: 1 for node in nodes}):
            raise ValueError(f"Need exactly one test start per selected node: {directory}")
        if set(reports) != set(nodes) or any(not {"setup", "teardown"} <= phases.keys() for phases in reports.values()):
            raise ValueError(f"Incomplete execution reports: {directory}")
        if any("call" not in reports[node] for node in calls_required):
            raise ValueError(f"Missing call reports after successful setup: {directory}")
        for node, phases in reports.items():
            samples[node].append(math.fsum(phases.values()))
        runs[(build_url, commit)].append((count, index, set(nodes)))
    if not runs:
        raise ValueError("No source runs were provided")
    for key, shards in runs.items():
        count = shards[0][0]
        if any(total != count for total, _, _ in shards) or sorted(index for _, index, _ in shards) != list(
            range(count)
        ):
            raise ValueError(f"Need exactly one successful attempt of every shard: {key}")
        all_nodes: set[str] = set()
        for _, _, nodes in shards:
            if all_nodes & nodes:
                raise ValueError(f"Source shards overlap: {key}")
            all_nodes.update(nodes)
    return {
        "schema_version": 1,
        "suite": suite,
        "sources": [{"build_url": url, "commit": commit} for url, commit in sorted(runs)],
        "node_seconds": {node: statistics.median(values) for node, values in sorted(samples.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+", help="One artifact directory per successful source shard")
    parser.add_argument("--suite", choices=SUITE_MARKERS, default="model-executor")
    parser.add_argument("--output", type=Path, required=True, help="New profile JSON path")
    args = parser.parse_args()
    try:
        profile = profile_from_artifacts(args.directories, args.suite)
        content = json.dumps(profile, indent=2, sort_keys=True, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(content)
    except (OSError, ValueError, OverflowError) as error:
        parser.error(str(error))
    print(f"Wrote {len(profile['node_seconds'])} node timings from {len(profile['sources'])} builds to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
