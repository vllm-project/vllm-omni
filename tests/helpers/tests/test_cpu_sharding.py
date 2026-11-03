# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from tests.helpers.cpu_sharding import TimingProfile, build_plan, load_profile

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
ROOT = Path(__file__).resolve().parents[3]
BUILDER = ROOT / ".buildkite/amd/scripts/cpu_shard_plan.py"
PLUGIN = "tests.helpers.fixtures.pytest_collection"
SELECTOR = "core_model and cpu and not omni"


def node(module: str, case: str = "test_case") -> str:
    return f"tests/model_executor/test_{module}.py::{case}"


def write_profile(path: Path, weights: dict[str, object]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "node_seconds": weights,
                "sources": [
                    {"build_url": "https://buildkite.com/example/synthetic/builds/1", "commit": "test-fixture"}
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_module_lpt_is_order_independent_and_conserves_nodes() -> None:
    weights = {node(name): cost for name, cost in zip("abcde", [10.0, 9.0, 8.0, 7.0, 6.0])}
    profile = TimingProfile(weights, "synthetic", [])
    plan = build_plan(list(weights), profile, 3)
    assert plan.estimated_shard_seconds == [10, 15, 15]
    assert plan == build_plan(list(reversed(weights)), profile, 3)
    partitions = [{n for n, owner in plan.node_shards.items() if owner == index} for index in range(3)]
    assert set.union(*partitions) == set(weights)
    assert sum(map(len, partitions)) == len(weights)


def test_parameters_and_classes_stay_with_their_module() -> None:
    nodes = [node("a", "TestShared::test_case[left::right]"), node("a", "test_case[other]"), node("b")]
    profile = TimingProfile({nodes[0]: 100}, "synthetic", [])
    plan = build_plan(nodes, profile, 3)
    assert plan.node_shards[nodes[0]] == plan.node_shards[nodes[1]]
    assert plan.node_shards[nodes[0]] != plan.node_shards[nodes[2]]


@pytest.mark.parametrize("history", [{}, {node("stale"): 100000}, {node("a"): 0.0}])
def test_unknown_and_stale_timings_never_select_or_drop_nodes(history: dict[str, float]) -> None:
    nodes = [node("a"), node("b"), node("new")]
    plan = build_plan(nodes, TimingProfile(history, "synthetic", []), 3)
    assert set(plan.node_shards) == set(nodes)
    assert plan.fallback_node_seconds == 1.0
    assert plan.unknown_timing_nodes == len(set(nodes) - history.keys())
    assert plan.stale_timing_nodes == len(history.keys() - set(nodes))


def test_empty_and_zero_cost_partitions() -> None:
    profile = TimingProfile({node(name): 0.0 for name in "abc"}, "synthetic", [])
    assert set(build_plan(list(profile.node_seconds), profile, 3).node_shards.values()) == {0, 1, 2}
    assert build_plan([], profile, 3).node_shards == {}


@pytest.mark.parametrize(
    "nodes,count", [([node("a"), node("a")], 3), (["tests/other.py::test_x"], 3), ([node("a")], 0)]
)
def test_ambiguous_or_invalid_plans_fail(nodes: list[str], count: int) -> None:
    with pytest.raises(ValueError):
        build_plan(nodes, TimingProfile({}, "synthetic", []), count)


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf"), True, "1", None])
def test_bad_profile_costs_are_rejected(tmp_path: Path, weight: object) -> None:
    profile = write_profile(tmp_path / "profile.json", {node("a"): weight})
    with pytest.raises(ValueError):
        load_profile(profile)


def test_profile_digest_and_provenance(tmp_path: Path) -> None:
    path = write_profile(tmp_path / "profile.json", {node("a"): 1.0})
    first = load_profile(path)
    write_profile(path, {node("a"): 2.0})
    assert load_profile(path).sha256 != first.sha256
    plan = asdict(build_plan([node("a")], first, 3))
    assert plan["source_builds"] == first.sources
    assert plan["profile_sha256"] == first.sha256


@pytest.fixture
def suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("PYTEST_ADDOPTS", "")
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers =\n    core_model\n    cpu\n    omni\n", encoding="utf-8")
    directory = tmp_path / "tests/model_executor"
    directory.mkdir(parents=True)
    for name in "abcd":
        skipped_case = (
            "@pytest.mark.skip(reason='synthetic platform skip')\ndef test_skip(): pass\n" if name == "a" else ""
        )
        (directory / f"test_{name}.py").write_text(
            "from pathlib import Path\nimport pytest\npytestmark = [pytest.mark.core_model, pytest.mark.cpu]\n"
            "@pytest.fixture(scope='module')\ndef shared():\n"
            "    with Path('fixtures.txt').open('a') as stream: stream.write(__name__ + '\\n')\n"
            "    return object()\n"
            "@pytest.mark.parametrize('value', [1, 2])\ndef test_case(shared, value): pass\n"
            "@pytest.mark.omni\ndef test_omni(): assert False\n" + skipped_case,
            encoding="utf-8",
        )
    write_profile(tmp_path / "profile.json", {node("a", "test_case[1]"): 10.0, node("removed"): 99999.0})
    return tmp_path


def run_pytest(suite: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            PLUGIN,
            "-p",
            "pytest_shard.pytest_shard",
            "-q",
            "tests/model_executor",
            "-m",
            SELECTOR,
            *args,
        ],
        cwd=suite,
        text=True,
        capture_output=True,
        timeout=30,
    )


def partition_args(index: int, output: str) -> tuple[str, ...]:
    return (
        "--cpu-shard-profile=profile.json",
        "--cpu-shard-count=3",
        f"--cpu-shard-index={index}",
        f"--cpu-shard-plan={output}",
    )


def test_three_executed_partitions_match_filtered_baseline(suite: Path) -> None:
    filters = (f"--deselect={node('a', 'test_case[2]')}", "-k", "not test_b")
    baseline = run_pytest(suite, "--collect-only", *filters)
    assert baseline.returncode == 0, baseline.stdout + baseline.stderr
    nodes = {line for line in baseline.stdout.splitlines() if line.startswith("tests/model_executor/")}
    owners: dict[str, int] = {}
    plans = []
    for index in range(3):
        result = run_pytest(suite, *filters, *partition_args(index, f"plan-{index}.json"), "-vv")
        assert result.returncode == 0, result.stdout + result.stderr
        plan = json.loads((suite / f"plan-{index}.json").read_text())
        plans.append(plan)
        selected = {n for n, shard in plan["node_shards"].items() if shard == index}
        completed = {
            line.split(f" {outcome}")[0]
            for line in result.stdout.splitlines()
            for outcome in ("PASSED", "SKIPPED")
            if f" {outcome}" in line
        }
        assert selected == completed, result.stdout
        assert not owners.keys() & selected
        owners.update({n: index for n in selected})
    assert set(owners) == nodes
    assert plans[0] == plans[1] == plans[2]
    assert sorted((suite / "fixtures.txt").read_text().splitlines()) == ["test_a", "test_c", "test_d"]


def test_skip_outcome_matches_unsharded_execution(suite: Path) -> None:
    filters = ("-k", "test_skip", "-vv")
    baseline = run_pytest(suite, *filters)
    assert baseline.returncode == 0, baseline.stdout + baseline.stderr
    baseline_skips = {line.split(" SKIPPED")[0] for line in baseline.stdout.splitlines() if " SKIPPED" in line}
    assert baseline_skips == {node("a", "test_skip")}

    shard_skips: set[str] = set()
    plans = []
    for index in range(3):
        result = run_pytest(suite, *filters, *partition_args(index, f"skip-plan-{index}.json"))
        plan = json.loads((suite / f"skip-plan-{index}.json").read_text())
        plans.append(plan)
        selected = {n for n, shard in plan["node_shards"].items() if shard == index}
        completed = {
            line.split(f" {outcome}")[0]
            for line in result.stdout.splitlines()
            for outcome in ("PASSED", "SKIPPED")
            if f" {outcome}" in line
        }
        assert selected == completed, result.stdout + result.stderr
        assert result.returncode == (0 if selected else 5), result.stdout + result.stderr
        shard_skips.update(line.split(" SKIPPED")[0] for line in result.stdout.splitlines() if " SKIPPED" in line)

    assert plans[0] == plans[1] == plans[2]
    assert shard_skips == baseline_skips


def test_default_hash_sharding_is_unchanged(suite: Path) -> None:
    for index in range(3):
        with_plugin = run_pytest(suite, "--collect-only", "--num-shards=3", f"--shard-id={index}")
        without_plugin = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "pytest_shard.pytest_shard",
                "-q",
                "tests/model_executor",
                "-m",
                SELECTOR,
                "--collect-only",
                "--num-shards=3",
                f"--shard-id={index}",
            ],
            cwd=suite,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert with_plugin.returncode == without_plugin.returncode == 0

        def get_nodes(output: str) -> list[str]:
            return [line for line in output.splitlines() if line.startswith("tests/model_executor/")]

        assert get_nodes(with_plugin.stdout) == get_nodes(without_plugin.stdout)


@pytest.mark.parametrize(
    "extra",
    [
        ("--num-shards=3",),
        ("--shard-id=1",),
        ("--cpu-shard-index=3",),
        ("--cpu-shard-count=0",),
        ("--cpu-shard-profile=missing.json",),
    ],
)
def test_unsafe_configuration_fails_before_execution(suite: Path, extra: tuple[str, ...]) -> None:
    result = run_pytest(suite, *partition_args(0, "plan.json"), *extra)
    assert result.returncode == 4, result.stdout + result.stderr
    assert not (suite / "plan.json").exists()


def test_xdist_guard_and_default_fixture_grouping(suite: Path) -> None:
    guarded = run_pytest(suite, "-p", "xdist", "-n", "2", *partition_args(0, "plan.json"))
    assert guarded.returncode == 4, guarded.stdout + guarded.stderr
    assert not (suite / "fixtures.txt").exists()
    normal = run_pytest(suite, "-p", "xdist", "-n", "2", "--dist=loadgroup")
    assert normal.returncode == 0, normal.stdout + normal.stderr
    assert sorted((suite / "fixtures.txt").read_text().splitlines()) == ["test_a", "test_b", "test_c", "test_d"]


def test_collection_error_is_not_hidden(suite: Path) -> None:
    (suite / "tests/model_executor/test_broken.py").write_text("raise RuntimeError('collection failed')\n")
    result = run_pytest(suite, *partition_args(0, "plan.json"))
    assert result.returncode == 2, result.stdout + result.stderr
    assert not (suite / "fixtures.txt").exists()


def test_bad_scope_and_artifact_overwrite_fail(suite: Path) -> None:
    result = run_pytest(suite, *partition_args(0, "plan.json"), "-m", "core_model")
    assert result.returncode == 4
    (suite / "plan.json").write_text("existing", encoding="utf-8")
    result = run_pytest(suite, *partition_args(0, "plan.json"))
    assert result.returncode == 4
    assert (suite / "plan.json").read_text() == "existing"


def test_failure_and_empty_selection_remain_failures(suite: Path) -> None:
    path = suite / "tests/model_executor/test_a.py"
    path.write_text(
        "import pytest\npytestmark=[pytest.mark.core_model,pytest.mark.cpu]\ndef test_case(): assert False\n"
    )
    # One eligible module belongs to shard 0; the other shards remain empty (exit 5).
    failed = run_pytest(suite, "-k", "test_a", *partition_args(0, "fail.json"))
    empty = run_pytest(suite, "-k", "test_a", *partition_args(1, "empty.json"))
    assert failed.returncode == 1, failed.stdout + failed.stderr
    assert empty.returncode == 5, empty.stdout + empty.stderr


def write_artifact(directory: Path, index: int, cost: float, *, build: int = 1) -> None:
    directory.mkdir()
    selected = node("abc"[index])
    documents = {
        "summary.json": {
            "schema_version": 1,
            "pytest_exit_code": 0,
            "collection_finished": True,
            "artifact_errors": [],
        },
        "invocation.json": {
            "schema_version": 1,
            "markers": SELECTOR,
            "num_shards": 3,
            "shard_id": index,
            "buildkite": {
                "BUILD_URL": f"https://buildkite.com/example/synthetic/builds/{build}",
                "COMMIT": "test-fixture",
            },
        },
        "selection.json": {"selected": [selected]},
    }
    for filename, document in documents.items():
        (directory / filename).write_text(json.dumps(document), encoding="utf-8")
    reports = [
        {"event": "test_report", "nodeid": selected, "phase": phase, "outcome": "passed", "seconds": cost / 3}
        for phase in ("setup", "call", "teardown")
    ]
    (directory / "events.jsonl").write_text("".join(json.dumps(report) + "\n" for report in reports), encoding="utf-8")


def run_builder(tmp_path: Path, directories: list[Path]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(BUILDER), *map(str, directories), "--output", str(tmp_path / "profile.json")],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_profile_builder_uses_complete_runs_and_median_phase_totals(tmp_path: Path) -> None:
    directories = []
    for build, cost in ((1, 6), (2, 12), (3, 30)):
        for index in range(3):
            directory = tmp_path / f"run-{build}-{index}"
            write_artifact(directory, index, cost, build=build)
            directories.append(directory)
    result = run_builder(tmp_path, directories)
    assert result.returncode == 0, result.stdout + result.stderr
    profile = load_profile(tmp_path / "profile.json")
    assert profile.node_seconds == {node(name): 12 for name in "abc"}
    assert len(profile.sources) == 3


@pytest.mark.parametrize("problem", ["missing-shard", "duplicate-shard", "failed", "partial", "missing-call"])
def test_profile_builder_rejects_unusable_evidence(tmp_path: Path, problem: str) -> None:
    directories = [tmp_path / str(index) for index in range(3)]
    for index, directory in enumerate(directories):
        write_artifact(directory, index, 6)
    if problem == "missing-shard":
        directories.pop()
    elif problem == "duplicate-shard":
        directories.append(directories[0])
    elif problem == "failed":
        path = directories[0] / "summary.json"
        summary = json.loads(path.read_text())
        summary["pytest_exit_code"] = 1
        path.write_text(json.dumps(summary))
    else:
        path = directories[0] / "events.jsonl"
        lines = path.read_text().splitlines()
        lines.pop(-1 if problem == "partial" else 1)
        path.write_text("\n".join(lines) + "\n")
    result = run_builder(tmp_path, directories)
    assert result.returncode == 2, result.stdout + result.stderr
    assert not (tmp_path / "profile.json").exists()
