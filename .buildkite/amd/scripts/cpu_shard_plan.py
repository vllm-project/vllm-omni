# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Build a versioned timing profile from complete successful CPU suite artifacts."""

import argparse
import json
import math
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.helpers.cpu_sharding import SUITE_MARKERS, duration_seconds, in_suite  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]


def junit_nodeid(classname: str, name: str) -> str:
    parts = classname.split(".")
    for length in range(len(parts), 0, -1):
        module = "/".join(parts[:length]) + ".py"
        if (ROOT / module).is_file():
            return "::".join([module, *parts[length:], name])
    raise ValueError(f"JUnit class does not resolve to a source module: {classname}")


def profile_from_junit(paths: list[Path], suite: str, build_url: str, commit: str) -> dict:
    """Import legacy pytest JUnit with its default total (setup/call/teardown) time."""
    weights: dict[str, float] = {}
    shards = []
    for path in paths:
        # Keep the original CI artifact names so completeness is verifiable.
        prefix, index, separator, count = path.stem.split("-")
        if prefix != "pytest" or separator != "of":
            raise ValueError("Expected pytest-INDEX-of-COUNT.xml artifact names")
        shards.append((int(index), int(count)))
        root = ET.parse(path).getroot()
        cases = root.findall(".//testcase")
        if not cases or root.findall(".//failure") or root.findall(".//error"):
            raise ValueError(f"Need complete successful JUnit artifacts: {path}")
        for case in cases:
            classname, name = case.get("classname", ""), case.get("name", "")
            seconds = duration_seconds(float(case.attrib["time"]))
            if not classname:
                # pytest emits collection-level importorskip entries in every
                # shard. These are not selected test nodes and have no phase cost.
                if (
                    case.find("skipped") is None
                    or seconds != 0
                    or not (ROOT / (name.replace(".", "/") + ".py")).is_file()
                ):
                    raise ValueError(f"Unrecognized collection result: {name}")
                continue
            node = junit_nodeid(classname, name)
            if not in_suite(node, suite) or node in weights:
                raise ValueError(f"Duplicate or out-of-suite JUnit node: {node}")
            weights[node] = seconds
    if not shards or not weights:
        raise ValueError("No JUnit timing evidence")
    count = shards[0][1]
    if (
        count < 1
        or any(total != count for _, total in shards)
        or sorted(index for index, _ in shards) != list(range(count))
    ):
        raise ValueError("Need exactly one successful JUnit artifact for every shard")
    if not build_url or not commit:
        raise ValueError("JUnit evidence requires a source build URL and commit")
    return {
        "schema_version": 1,
        "suite": suite,
        "sources": [{"build_url": build_url, "commit": commit}],
        "timing_format": "pytest-junit-total",
        "node_seconds": dict(sorted(weights.items())),
    }


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
        selected_nodes = set(nodes)
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
                    node = event.get("nodeid")
                    if node not in selected_nodes:
                        raise ValueError(f"Test start outside the selected nodes: {directory}")
                    starts[node] += 1
                if event.get("event") != "test_report":
                    continue
                node, phase = event.get("nodeid"), event.get("phase")
                if node not in selected_nodes or phase not in ("setup", "call", "teardown"):
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
        if starts != Counter({node: 1 for node in selected_nodes}):
            raise ValueError(f"Need exactly one test start per selected node: {directory}")
        if set(reports) != selected_nodes or any(
            not {"setup", "teardown"} <= phases.keys() for phases in reports.values()
        ):
            raise ValueError(f"Incomplete execution reports: {directory}")
        if any("call" not in reports[node] for node in calls_required):
            raise ValueError(f"Missing call reports after successful setup: {directory}")
        for node, phases in reports.items():
            samples[node].append(math.fsum(phases.values()))
        runs[(build_url, commit)].append((count, index, selected_nodes))
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
    parser.add_argument("directories", type=Path, nargs="*", help="One artifact directory per successful source shard")
    parser.add_argument("--junit", type=Path, nargs="+", help="Legacy JUnit artifacts; time must include all phases")
    parser.add_argument("--source-build-url")
    parser.add_argument("--source-commit")
    parser.add_argument("--suite", choices=SUITE_MARKERS, default="model-executor")
    parser.add_argument("--output", type=Path, required=True, help="New profile JSON path")
    args = parser.parse_args()
    try:
        if args.junit:
            if args.directories:
                parser.error("Choose instrumented directories or legacy JUnit, not both")
            profile = profile_from_junit(args.junit, args.suite, args.source_build_url, args.source_commit)
        else:
            profile = profile_from_artifacts(args.directories, args.suite)
        content = json.dumps(profile, indent=2, sort_keys=True, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(content)
    except (OSError, ValueError, OverflowError, ET.ParseError) as error:
        parser.error(str(error))
    print(f"Wrote {len(profile['node_seconds'])} node timings from {len(profile['sources'])} builds to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
