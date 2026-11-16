# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Deterministic, coverage-preserving module partitions for AMD CPU suites."""

import hashlib
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

SUITE_MARKERS = {
    "model-executor": "core_model and cpu and not omni",
    "other": "core_model and cpu",
    "diffusion": "core_model and cpu and not (cards_2 or cards_3 or cards_4 or cards_5 or cards_6 or cards_7 or cards_8)",
}


def in_suite(nodeid: str, suite: str) -> bool:
    module, separator, _ = nodeid.partition("::")
    if not separator or not module.endswith(".py"):
        return False
    if suite == "model-executor":
        return module.startswith("tests/model_executor/")
    if suite == "diffusion":
        return module.startswith("tests/diffusion/")
    if suite == "other":
        return module.startswith("tests/") and not module.startswith(
            ("tests/model_executor/", "tests/diffusion/", "tests/engine/", "tests/entrypoints/")
        )
    raise ValueError(f"Unknown CPU suite: {suite}")


@dataclass(frozen=True)
class TimingProfile:
    node_seconds: dict[str, float]
    sha256: str
    sources: list[dict[str, str]]
    suite: str = "model-executor"


@dataclass(frozen=True)
class ShardPlan:
    schema_version: int
    algorithm: str
    profile_sha256: str
    source_builds: list[dict[str, str]]
    eligible_nodeids_sha256: str
    node_shards: dict[str, int]
    estimated_shard_seconds: list[float]
    fallback_node_seconds: float
    matched_timing_nodes: int
    unknown_timing_nodes: int
    stale_timing_nodes: int


def duration_seconds(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("Duration must be a nonnegative finite number")
    try:
        seconds = float(value)
    except OverflowError as error:
        raise ValueError("Duration is too large") from error
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Duration must be a nonnegative finite number")
    return seconds


def load_profile(path: Path) -> TimingProfile:
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("CPU shard profile must have schema_version 1")
    weights = data.get("node_seconds")
    sources = data.get("sources")
    suite = data.get("suite", "model-executor")
    if not isinstance(suite, str) or suite not in SUITE_MARKERS:
        raise ValueError("Unknown CPU timing profile suite")
    if not isinstance(weights, dict) or not isinstance(sources, list):
        raise ValueError("CPU shard profile requires node_seconds and sources")
    for source in sources:
        if not isinstance(source, dict) or any(
            not isinstance(source.get(key), str) or not source[key] for key in ("build_url", "commit")
        ):
            raise ValueError("Each timing source requires a build_url and commit")
    if weights and not sources:
        raise ValueError("Nonempty timing weights require source provenance")
    durations = {}
    for nodeid, duration in weights.items():
        if not in_suite(nodeid, suite):
            raise ValueError(f"Timing node is outside {suite}: {nodeid}")
        durations[nodeid] = duration_seconds(duration)
    provenance = [{key: source[key] for key in ("build_url", "commit")} for source in sources]
    return TimingProfile(durations, hashlib.sha256(raw).hexdigest(), provenance, suite)


def build_plan(nodeids: list[str], profile: TimingProfile, shard_count: int) -> ShardPlan:
    """Weight live collection, never use timing history as a test allowlist."""
    if shard_count < 1:
        raise ValueError("CPU shard count must be positive")
    if len(set(nodeids)) != len(nodeids):
        raise ValueError("Duplicate collected node IDs cannot be partitioned unambiguously")
    groups: dict[str, list[str]] = defaultdict(list)
    for nodeid in sorted(nodeids):
        module, separator, _ = nodeid.partition("::")
        if not separator or not in_suite(nodeid, profile.suite):
            raise ValueError(f"CPU partition node is outside {profile.suite}: {nodeid}")
        groups[module].append(nodeid)
    live_weights = [profile.node_seconds[node] for node in nodeids if node in profile.node_seconds]
    positive_weights = [weight for weight in live_weights if weight > 0]
    fallback = statistics.median(positive_weights) if positive_weights else 1.0
    costs = {
        module: math.fsum(profile.node_seconds.get(node, fallback) for node in nodes)
        for module, nodes in groups.items()
    }
    loads = [0.0] * shard_count
    counts = [0] * shard_count
    assignments = {}
    # Longest estimated module first; stable path/index ties across workers.
    for module in sorted(groups, key=lambda name: (-costs[name], name)):
        shard = min(range(shard_count), key=lambda index: (loads[index], counts[index], index))
        for nodeid in groups[module]:
            assignments[nodeid] = shard
        loads[shard] += costs[module]
        counts[shard] += len(groups[module])
    return ShardPlan(
        schema_version=1,
        algorithm="module-lpt-v1",
        profile_sha256=profile.sha256,
        source_builds=profile.sources,
        eligible_nodeids_sha256=hashlib.sha256(json.dumps(sorted(nodeids)).encode()).hexdigest(),
        node_shards=assignments,
        estimated_shard_seconds=loads,
        fallback_node_seconds=fallback,
        matched_timing_nodes=len(live_weights),
        unknown_timing_nodes=len(nodeids) - len(live_weights),
        stale_timing_nodes=len(profile.node_seconds.keys() - set(nodeids)),
    )
