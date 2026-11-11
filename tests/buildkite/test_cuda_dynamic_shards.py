# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA ready/merge directory sweeps shard from measured durations.

``automatic_shard: true`` does not reserve agents. With no timing profile the
uploader leaves one job (and still records ``timings/jobs/*.json``). A profile
that exceeds ``timeout_in_minutes`` packs files into bins, writes each bin as a
file list, and sets ``parallelism``. A shard runs only its list.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".buildkite" / "common" / "scripts"))

import pytest_file_timings  # noqa: E402
import upload_pipeline as upload_pipeline_mod  # noqa: E402
from merge_shard_timings import (  # noqa: E402
    HISTORY_LOOKUP_BUDGET_SECONDS,
    MAX_CONSECUTIVE_MISSES,
    build_step_details,
    details_from_download,
    list_recent_build_ids,
    merge_build_details,
    merge_build_observations,
    percentile,
    prepare_profile_from_history,
    write_merged_profile,
)
from runtime_shard import PLUGIN_MODULE, _file_shard_command, _inject_plugin, pack_files, plan_shard_count  # noqa: E402
from upload_pipeline import _render_test_pipeline, filter_pipeline_by_source_dependencies  # noqa: E402

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

READY = Path(".buildkite/cuda/test-ready.yml")
NUM_SHARDS = "--num-shards=$$BUILDKITE_PARALLEL_JOB_COUNT"
SHARD_ID = "--shard-id=$$BUILDKITE_PARALLEL_JOB"


@pytest.fixture(autouse=True)
def _skip_timing_download(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_OMNI_SKIP_SHARD_TIMINGS_DOWNLOAD", "1")
    monkeypatch.delenv("VLLM_OMNI_SHARD_TIMINGS", raising=False)
    # A shard job exports this for its own pytest. Tests that call the plugin
    # hook, or chdir, must not inherit that relative list.
    monkeypatch.delenv("VLLM_OMNI_FILE_SHARD_LIST", raising=False)
    monkeypatch.delenv("BUILDKITE_API_TOKEN", raising=False)
    monkeypatch.delenv("BUILDKITE_TOKEN", raising=False)
    upload_pipeline_mod._shard_timings_prepared = False
    upload_pipeline_mod._shard_timings_pipeline = None
    yield
    shards = Path("timings/file-shards")
    if shards.exists():
        shutil.rmtree(shards)
    timings = Path("timings")
    if timings.is_dir() and not any(timings.iterdir()):
        timings.rmdir()


def _leaves(steps: list) -> list[dict]:
    leaves: list[dict] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        if "wait" in step:
            continue
        nested = step.get("steps")
        if nested is not None:
            leaves.extend(_leaves(nested))
        else:
            leaves.append(step)
    return leaves


def test_source_filter_does_not_plan_cuda_shards() -> None:
    """AMD nightly filtering imports this uploader and must not plan shards."""
    document = {
        "steps": [
            {
                "label": "Nightly",
                "commands": ["pytest tests/amd_test.py"],
                "source_file_dependencies": ["tests/amd_test.py"],
                "automatic_shard": True,
            }
        ]
    }
    filtered = filter_pipeline_by_source_dependencies(
        document,
        ["tests/amd_test.py"],
        pipeline_path=Path(".buildkite/amd/test-amd-nightly.yml"),
    )
    step = filtered["steps"][0]
    assert step["label"] == "Nightly"
    assert step["automatic_shard"] is True
    assert "parallelism" not in step
    assert upload_pipeline_mod._shard_timings_prepared is False


def test_uploader_keeps_one_simple_test_job_without_timings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MIRROR_HW", raising=False)
    document = yaml.safe_load(READY.read_text(encoding="utf-8"))
    rendered = _render_test_pipeline(document, changed_files=None, pipeline_path=READY)
    assert all("automatic_shard" not in leaf for leaf in _leaves(rendered["steps"]))
    step = next(leaf for leaf in _leaves(rendered["steps"]) if leaf.get("label") == "Simple Test")
    assert "parallelism" not in step
    assert "automatic_shard" not in step
    assert "mirror_hardwares" not in step
    assert step["agents"]["queue"] == "l4-k8s"
    assert "timings/jobs/*.json" in step["artifact_paths"]
    text = "\n".join(step["commands"])
    assert PLUGIN_MODULE in text
    assert ".buildkite/common/scripts" in text
    assert "$${PYTHONPATH:+:$$PYTHONPATH}" in text
    assert NUM_SHARDS not in text
    assert SHARD_ID not in text


def test_uploader_splits_simple_test_when_durations_exceed_the_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    files = ["tests/a_test.py", "tests/b_test.py"]
    timings = tmp_path / "timings.json"
    timings.write_text(
        json.dumps({"steps": {"Simple Test": {name: 1000 for name in files}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("VLLM_OMNI_SHARD_TIMINGS", str(timings))
    monkeypatch.delenv("MIRROR_HW", raising=False)
    monkeypatch.setattr(
        "runtime_shard.discover_test_files",
        lambda command, root: list(files) if "tests/" in command else [],
    )

    document = yaml.safe_load(READY.read_text(encoding="utf-8"))
    rendered = _render_test_pipeline(document, changed_files=None, pipeline_path=READY)
    step = next(leaf for leaf in _leaves(rendered["steps"]) if str(leaf.get("label", "")).startswith("Simple Test"))
    assert step["parallelism"] == 2
    assert step["label"] == "Simple Test · Shard %N/%t"
    assert "automatic_shard" not in step
    assert "timings/jobs/*.json" in step["artifact_paths"]
    command = step["commands"][-1]
    assert NUM_SHARDS not in command
    assert SHARD_ID not in command
    assert "timings/file-shards/simple-test/$${BUILDKITE_PARALLEL_JOB:-0}.txt" in command
    assert PLUGIN_MODULE in command
    lists = _read_shard_lists("simple-test")
    assert lists == [["tests/a_test.py"], ["tests/b_test.py"]]


def _read_shard_lists(slug: str) -> list[list[str]]:
    directory = Path("timings/file-shards") / slug
    lists = []
    for path in sorted(directory.glob("*.txt"), key=lambda item: int(item.stem)):
        lists.append([line for line in path.read_text(encoding="utf-8").splitlines() if line])
    return lists


def test_packed_bins_separate_slow_files() -> None:
    bins = pack_files({"a.py": 8, "b.py": 7, "c.py": 3, "d.py": 2}, budget_seconds=15)
    assert bins == [["a.py", "d.py"], ["b.py", "c.py"]]


def test_unmeasured_file_stays_weight_zero_and_still_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    discovered = ["tests/a_test.py", "tests/b_test.py", "tests/new_test.py"]
    timings = tmp_path / "timings.json"
    timings.write_text(
        json.dumps({"steps": {"Simple Test": {"tests/a_test.py": 1000, "tests/b_test.py": 1000}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("VLLM_OMNI_SHARD_TIMINGS", str(timings))
    monkeypatch.delenv("MIRROR_HW", raising=False)
    monkeypatch.setattr(
        "runtime_shard.discover_test_files", lambda command, root: list(discovered) if "tests/" in command else []
    )

    document = yaml.safe_load(READY.read_text(encoding="utf-8"))
    rendered = _render_test_pipeline(document, changed_files=None, pipeline_path=READY)
    step = next(leaf for leaf in _leaves(rendered["steps"]) if str(leaf.get("label", "")).startswith("Simple Test"))
    assert step["parallelism"] == 2
    assert "temporary estimate" not in "\n".join(step["commands"])
    placed = [name for shard in _read_shard_lists("simple-test") for name in shard]
    assert sorted(placed) == sorted(discovered)


def test_one_over_budget_file_does_not_open_extra_jobs() -> None:
    assert plan_shard_count([4000], budget_seconds=25 * 60) == 1


def test_plan_uses_a_greedy_shard_count() -> None:
    assert plan_shard_count([1000, 1000], budget_seconds=25 * 60) == 2
    assert plan_shard_count([400, 400, 400], budget_seconds=25 * 60) == 1
    assert plan_shard_count([1000, 1000, 1000, 1000], budget_seconds=25 * 60) == 4
    # Two bins can hold 16 exactly (8+4+4 and 6+5+5). Greedy placement returns 3.
    assert plan_shard_count([8, 6, 5, 5, 4, 4], budget_seconds=16) == 3


def test_merge_build_observations_uses_p80() -> None:
    builds = [
        {"tests/a_test.py": 10},
        {"tests/a_test.py": 20},
        {"tests/a_test.py": 30},
        {"tests/a_test.py": 40},
        {"tests/a_test.py": 100},
    ]
    merged = merge_build_observations(builds)
    assert merged["tests/a_test.py"] == round(percentile([10, 20, 30, 40, 100], 0.8), 3)


def _job_timing(
    path: Path,
    *,
    job_id: str,
    label: str,
    file_seconds: Mapping[str, float],
    parallel_job: int | None = None,
    queue: str = "",
) -> None:
    path.write_text(
        json.dumps(
            {
                "job_id": job_id,
                "label": label,
                "parallel_job": parallel_job,
                "queue": queue,
                "file_seconds": file_seconds,
            }
        ),
        encoding="utf-8",
    )


def test_different_jobs_take_the_max_per_file(tmp_path: Path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(
        json.dumps({"job_id": "a", "label": "Simple Test", "file_seconds": {"tests/a_test.py": 3}}),
        encoding="utf-8",
    )
    b.write_text(
        json.dumps({"job_id": "b", "label": "Simple Test", "file_seconds": {"tests/a_test.py": 9}}),
        encoding="utf-8",
    )
    assert build_step_details([a, b]) == {"Simple Test": ({"tests/a_test.py": 9}, "")}


def test_hash_shards_sum_slices_and_retries_keep_the_max(tmp_path: Path) -> None:
    files = {
        "tests/buildkite/test_cuda_dynamic_shards.py": 500,
        "tests/buildkite/test_upload_pipeline.py": 500,
    }
    retry = {name: 400 for name in files}
    _job_timing(
        tmp_path / "s0.json", job_id="s0", label="Simple Test", file_seconds=files, parallel_job=0, queue="l4-k8s"
    )
    _job_timing(
        tmp_path / "s0-retry.json",
        job_id="s0-retry",
        label="Simple Test",
        file_seconds=retry,
        parallel_job=0,
        queue="l4-k8s",
    )
    _job_timing(
        tmp_path / "s1.json", job_id="s1", label="Simple Test", file_seconds=files, parallel_job=1, queue="l4-k8s"
    )
    observed, _queue = build_step_details(sorted(tmp_path.glob("*.json")))["Simple Test"]
    assert observed == {name: 1000 for name in files}
    assert plan_shard_count(list(observed.values()), budget_seconds=1500) == 2


def test_two_invocations_in_one_job_keep_both_measurements(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BUILDKITE_JOB_ID", "job-1")
    monkeypatch.setenv("BUILDKITE_LABEL", "Entrypoints Test · Shard 1/2")
    pytest_file_timings._timings.clear()
    pytest_file_timings._timings["tests/entrypoints/test_async_omni.py"] = 3
    monkeypatch.setattr(pytest_file_timings.os, "getpid", lambda: 111)
    pytest_file_timings._write()
    pytest_file_timings._timings["tests/entrypoints/test_async_omni.py"] = 4
    monkeypatch.setattr(pytest_file_timings.os, "getpid", lambda: 222)
    pytest_file_timings._write()
    pytest_file_timings._timings.clear()

    written = sorted((tmp_path / "timings" / "jobs").glob("*.json"))
    assert [path.name for path in written] == ["job-1-111.json", "job-1-222.json"]
    assert build_step_details(written) == {"Entrypoints Test": ({"tests/entrypoints/test_async_omni.py": 7}, "")}


def test_profile_uses_raw_job_files_not_a_previous_merged_profile(tmp_path: Path) -> None:
    def build(root: Path, raw_seconds: float, rolled_seconds: float) -> None:
        jobs = root / "timings" / "jobs"
        jobs.mkdir(parents=True)
        (jobs / "job.json").write_text(
            json.dumps(
                {
                    "job_id": "job",
                    "label": "Simple Test · Shard 1/2",
                    "file_seconds": {"tests/a_test.py": raw_seconds},
                }
            ),
            encoding="utf-8",
        )
        (root / "timings").mkdir(exist_ok=True)
        (root / "timings" / "merged.json").write_text(
            json.dumps({"steps": {"Simple Test": {"tests/a_test.py": rolled_seconds}}}),
            encoding="utf-8",
        )

    build(tmp_path / "b1", 10, 100)
    build(tmp_path / "b2", 30, 100)
    per_build = [details_from_download(tmp_path / "b1"), details_from_download(tmp_path / "b2")]
    assert None not in per_build
    steps, _queues = merge_build_details([item for item in per_build if item is not None])
    assert steps["Simple Test"]["tests/a_test.py"] == round(percentile([10, 30], 0.8), 3)
    assert steps["Simple Test"]["tests/a_test.py"] != 100


def test_public_history_keeps_only_successful_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BUILDKITE_ORGANIZATION_SLUG", "vllm")
    monkeypatch.setenv("BUILDKITE_PIPELINE_SLUG", "vllm-omni")
    monkeypatch.delenv("BUILDKITE_API_TOKEN", raising=False)
    monkeypatch.delenv("BUILDKITE_TOKEN", raising=False)
    listing = " ".join(f'href="/vllm/vllm-omni/builds/{number}"' for number in ("4", "3", "2", "1"))
    details = {
        "4": {"id": "unknown-id", "branch_name": "main", "state": ""},
        "3": {"id": "failed-id", "branch_name": "main", "state": "failed", "artifacts": ["timings/jobs/a.json"]},
        "2": {"id": "running-id", "branch_name": "main", "state": "running", "artifacts": ["timings/jobs/a.json"]},
        "1": {"id": "passed-id", "branch_name": "main", "state": "passed", "artifacts": ["timings/jobs/a.json"]},
    }

    def fetch(url: str, token: str | None = None) -> tuple[str, str | None]:
        del token
        if url.endswith("/builds?branch=main"):
            return listing, None
        number = url.rsplit("/", 1)[-1].removesuffix(".json")
        return json.dumps(details[number]), None

    assert list_recent_build_ids(branch="main", limit=20, fetch=fetch) == [("passed-id", "1")]


def test_api_history_rejects_a_non_passed_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BUILDKITE_ORGANIZATION_SLUG", "vllm")
    monkeypatch.setenv("BUILDKITE_PIPELINE_SLUG", "vllm-omni")
    monkeypatch.setenv("BUILDKITE_API_TOKEN", "token")

    def fetch(url: str, token: str | None = None) -> tuple[str, str | None]:
        assert token == "token"
        assert "state=passed" in url
        return (
            json.dumps(
                [
                    {"id": "bad", "number": 9, "state": "failed"},
                    {"id": "blank", "number": 8, "state": ""},
                    {"id": "good", "number": 7, "state": "passed"},
                ]
            ),
            None,
        )

    assert list_recent_build_ids(branch="main", limit=20, fetch=fetch) == [("good", "7")]


def test_ready_history_excludes_main_passed_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BUILDKITE_ORGANIZATION_SLUG", "vllm")
    monkeypatch.setenv("BUILDKITE_PIPELINE_SLUG", "vllm-omni")
    monkeypatch.setenv("BUILDKITE_API_TOKEN", "token")

    def fetch(url: str, token: str | None = None) -> tuple[str, str | None]:
        assert token == "token"
        assert "branch=" not in url
        assert "state=passed" in url
        return (
            json.dumps(
                [
                    {"id": "main-build", "number": 3, "state": "passed", "branch": "main"},
                    {"id": "failed-pr", "number": 2, "state": "failed", "branch": "ready-pr"},
                    {"id": "ready-pr", "number": 1, "state": "passed", "branch": "ready-pr"},
                ]
            ),
            None,
        )

    assert list_recent_build_ids(branch="", limit=20, exclude_branches={"main"}, fetch=fetch) == [("ready-pr", "1")]


def test_api_history_follows_the_next_page_after_excluded_main_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BUILDKITE_ORGANIZATION_SLUG", "vllm")
    monkeypatch.setenv("BUILDKITE_PIPELINE_SLUG", "vllm-omni")
    monkeypatch.setenv("BUILDKITE_API_TOKEN", "token")
    first = "https://api.buildkite.com/v2/organizations/vllm/pipelines/vllm-omni/builds?state=passed&per_page=100"
    second = "https://api.example/builds?page=2"
    pages = {
        first: (
            json.dumps(
                [
                    {"id": f"main-{index}", "number": 50 - index, "state": "passed", "branch": "main"}
                    for index in range(3)
                ]
            ),
            second,
        ),
        second: (
            json.dumps([{"id": "ready-pr", "number": 1, "state": "passed", "branch": "ready-pr"}]),
            None,
        ),
    }
    seen: list[str] = []

    def fetch(url: str, token: str | None = None) -> tuple[str, str | None]:
        assert token == "token"
        seen.append(url)
        return pages[url]

    assert list_recent_build_ids(branch="", limit=20, exclude_branches={"main"}, fetch=fetch) == [("ready-pr", "1")]
    assert seen == [first, second]
    assert "per_page=200" not in first


def test_history_scan_stops_after_consecutive_misses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def list_ids(**_kwargs: object) -> list[tuple[str, str]]:
        return [(f"id-{index}", str(index)) for index in range(MAX_CONSECUTIVE_MISSES + 15)]

    def download(build_id: str, dest: Path, timeout: int = 180) -> None:
        del dest, timeout
        calls.append(build_id)

    monkeypatch.setattr("merge_shard_timings.list_recent_build_ids", list_ids)
    monkeypatch.setattr("merge_shard_timings.download_build_timings", download)
    assert prepare_profile_from_history(tmp_path / "out.json", branch="") is None
    assert len(calls) == MAX_CONSECUTIVE_MISSES


def test_history_scan_keeps_going_when_a_hit_breaks_the_miss_streak(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    streak = MAX_CONSECUTIVE_MISSES - 1
    ids = [f"miss-{index}" for index in range(streak)] + ["hit-a"]
    ids += [f"miss-{index}" for index in range(streak, streak * 2)] + ["hit-b"]

    def list_ids(**_kwargs: object) -> list[tuple[str, str]]:
        return [(build_id, str(index)) for index, build_id in enumerate(ids)]

    def download(build_id: str, dest: Path, timeout: int = 180) -> None:
        del timeout
        calls.append(build_id)
        if not build_id.startswith("hit-"):
            return
        jobs = dest / "timings" / "jobs"
        jobs.mkdir(parents=True)
        (jobs / "job.json").write_text(
            json.dumps({"job_id": build_id, "label": "Simple Test", "file_seconds": {"tests/a_test.py": 1}}),
            encoding="utf-8",
        )

    monkeypatch.setattr("merge_shard_timings.list_recent_build_ids", list_ids)
    monkeypatch.setattr("merge_shard_timings.download_build_timings", download)
    path = prepare_profile_from_history(tmp_path / "out.json", branch="")
    assert path is not None
    assert json.loads(path.read_text(encoding="utf-8"))["builds"] == 2
    assert calls == ids


def test_history_scan_stops_at_the_lookup_budget(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    ticks = iter([0.0, 0.0, float(HISTORY_LOOKUP_BUDGET_SECONDS)])

    def list_ids(**_kwargs: object) -> list[tuple[str, str]]:
        return [(f"id-{index}", str(index)) for index in range(30)]

    def download(build_id: str, dest: Path, timeout: int = 180) -> None:
        del dest, timeout
        calls.append(build_id)

    monkeypatch.setattr("merge_shard_timings.list_recent_build_ids", list_ids)
    monkeypatch.setattr("merge_shard_timings.download_build_timings", download)
    monkeypatch.setattr("merge_shard_timings.time.monotonic", lambda: next(ticks))
    assert prepare_profile_from_history(tmp_path / "out.json", branch="") is None
    assert calls == ["id-0"]


def test_ready_plans_simple_test_from_non_main_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = yaml.safe_load(READY.read_text(encoding="utf-8"))
    files = {
        "tests/buildkite/test_cuda_dynamic_shards.py": 500.0,
        "tests/buildkite/test_upload_pipeline.py": 500.0,
    }
    upload_pipeline_mod._shard_timings_prepared = False
    monkeypatch.delenv("VLLM_OMNI_SHARD_TIMINGS", raising=False)
    monkeypatch.delenv("VLLM_OMNI_SKIP_SHARD_TIMINGS_DOWNLOAD", raising=False)
    monkeypatch.delenv("MIRROR_HW", raising=False)
    seen_branch = ""
    seen_excluded: frozenset[str] = frozenset()

    def list_ids(
        *,
        branch: str = "",
        limit: int = 0,
        current_build_id: str = "",
        exclude_branches: frozenset[str] | set[str] = frozenset(),
        **_kwargs: object,
    ) -> list[tuple[str, str]]:
        nonlocal seen_branch, seen_excluded
        del limit, current_build_id, _kwargs
        seen_branch = branch
        seen_excluded = frozenset(exclude_branches)
        return [("ready-pr", "42")]

    def download(build_id: str, dest: Path, timeout: int = 180) -> None:
        del timeout
        assert build_id == "ready-pr"
        jobs = dest / "timings" / "jobs"
        jobs.mkdir(parents=True)
        for index in (0, 1):
            _job_timing(
                jobs / f"shard-{index}.json",
                job_id=f"shard-{index}",
                label="Simple Test",
                file_seconds=files,
                parallel_job=index,
                queue="l4-k8s",
            )
        _job_timing(
            jobs / "shard-0-retry.json",
            job_id="shard-0-retry",
            label="Simple Test",
            file_seconds={name: 400 for name in files},
            parallel_job=0,
            queue="l4-k8s",
        )

    monkeypatch.setattr("merge_shard_timings.list_recent_build_ids", list_ids)
    monkeypatch.setattr("merge_shard_timings.download_build_timings", download)
    profile = upload_pipeline_mod.ROOT / "timings" / "upload-profile.json"
    try:
        rendered = _render_test_pipeline(document, changed_files=None, pipeline_path=READY)
        assert seen_branch == ""
        assert seen_excluded == frozenset({"main"})
        step = next(leaf for leaf in _leaves(rendered["steps"]) if str(leaf.get("label", "")).startswith("Simple Test"))
        assert step["parallelism"] == 2
    finally:
        profile.unlink(missing_ok=True)
        try:
            profile.parent.rmdir()
        except OSError:
            pass


def test_ready_ignores_timings_recorded_on_another_queue(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    files = ["tests/buildkite/test_cuda_dynamic_shards.py", "tests/buildkite/test_upload_pipeline.py"]
    profile = tmp_path / "timings.json"
    profile.write_text(
        json.dumps(
            {
                "steps": {"Simple Test": {name: 1000 for name in files}},
                "step_queues": {"Simple Test": "mithril-h100-pool"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("VLLM_OMNI_SHARD_TIMINGS", str(profile))
    monkeypatch.delenv("MIRROR_HW", raising=False)
    document = yaml.safe_load(READY.read_text(encoding="utf-8"))
    rendered = _render_test_pipeline(document, changed_files=None, pipeline_path=READY)
    step = next(leaf for leaf in _leaves(rendered["steps"]) if leaf.get("label") == "Simple Test")
    assert "parallelism" not in step


def test_file_shard_command_reads_the_parallel_job_list(tmp_path: Path) -> None:
    bash = _shell_bash()
    slug = "simple-test"
    shard_dir = tmp_path / "timings" / "file-shards" / slug
    shard_dir.mkdir(parents=True)
    (shard_dir / "1.txt").write_text("tests/b_test.py\n", encoding="utf-8")
    dumper = tmp_path / "dump_argv.py"
    dumper.write_text(
        "import os, sys\n"
        "print('\\n'.join(sys.argv[1:]))\n"
        "print('SHARD=' + os.environ.get('VLLM_OMNI_FILE_SHARD_LIST', ''))\n",
        encoding="utf-8",
    )
    interpreter = _shell_python(bash)
    dumper_arg = _bash_command(dumper, bash=bash)
    rendered = _inject_plugin(_file_shard_command("pytest -sv tests/ -m 'core_model and cpu'", slug))
    assert NUM_SHARDS not in rendered
    assert SHARD_ID not in rendered
    assert "'$" not in rendered
    assert PLUGIN_MODULE in rendered
    assert "VLLM_OMNI_FILE_SHARD_LIST" in rendered
    # Directory targets stay so each directory conftest.py is collected.
    assert "pytest -p pytest_file_timings -sv tests/ -m 'core_model and cpu'" in rendered
    expanded = rendered.replace("$$", "$").replace("pytest ", f"{interpreter} {dumper_arg} ", 1)
    if sys.platform == "win32" and "system32" in bash.lower():
        expanded = expanded.replace("$", r"\$")
    shell = "buildkite-agent() { return 0; }; export BUILDKITE_PARALLEL_JOB=1; " + expanded
    completed = subprocess.run(
        [bash, "-c", shell],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    argv = completed.stdout.splitlines()
    assert "tests/" in argv, completed.stderr
    assert "tests/b_test.py" not in argv, completed.stderr
    assert any(line.startswith("SHARD=") and line.endswith("1.txt") for line in argv), completed.stderr
    assert "pytest_file_timings" in argv


def test_shard_list_keeps_directory_conftest_and_drops_other_files(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    pkg = proj / "tests" / "pkg"
    pkg.mkdir(parents=True)
    (proj / "pyproject.toml").write_text("[tool.pytest.ini_options]\naddopts = []\n", encoding="utf-8")
    (pkg / "conftest.py").write_text(
        "import pytest\n\n@pytest.fixture\ndef contracts():\n    return 1\n",
        encoding="utf-8",
    )
    (pkg / "test_one.py").write_text("def test_one(contracts):\n    assert contracts == 1\n", encoding="utf-8")
    (pkg / "test_two.py").write_text("def test_two(contracts):\n    assert contracts == 1\n", encoding="utf-8")
    shard = proj / "0.txt"
    shard.write_text("tests/pkg/test_one.py\n", encoding="utf-8")
    env = os.environ.copy()
    env["VLLM_OMNI_FILE_SHARD_LIST"] = str(shard)
    scripts = Path(".buildkite/common/scripts").resolve()
    env["PYTHONPATH"] = str(scripts) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-o", "addopts=", "-p", "pytest_file_timings", "-q", "tests/pkg"],
        cwd=proj,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "1 passed" in output
    assert "1 deselected" in output


def test_simple_test_ignores_durations_recorded_by_another_step(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    files = ["tests/a_test.py", "tests/b_test.py"]
    timings = tmp_path / "timings.json"
    timings.write_text(
        json.dumps(
            {
                "steps": {
                    "Diffusion · Model Test": {name: 1000 for name in files},
                    "Simple Test · Shard 1/2": {name: 1 for name in files},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("VLLM_OMNI_SHARD_TIMINGS", str(timings))
    monkeypatch.delenv("MIRROR_HW", raising=False)
    monkeypatch.setattr(
        "runtime_shard.discover_test_files",
        lambda command, root: list(files) if "tests/" in command else [],
    )
    document = yaml.safe_load(READY.read_text(encoding="utf-8"))
    rendered = _render_test_pipeline(document, changed_files=None, pipeline_path=READY)
    step = next(leaf for leaf in _leaves(rendered["steps"]) if leaf.get("label") == "Simple Test")
    assert "parallelism" not in step
    assert NUM_SHARDS not in "\n".join(step["commands"])


def _shell_bash() -> str:
    if sys.platform == "win32":
        for candidate in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"):
            if Path(candidate).is_file():
                return candidate
    found = shutil.which("bash")
    if not found:
        raise AssertionError("bash is required to check shard-flag expansion")
    return found


def _shell_python(bash: str) -> str:
    if sys.platform == "win32" and "system32" in bash.lower():
        return "python3"
    return _bash_command(sys.executable, bash=bash)


def _bash_command(path: str | Path, *, bash: str) -> str:
    text = Path(path).resolve().as_posix()
    if sys.platform == "win32" and len(text) >= 2 and text[1] == ":":
        prefix = "/mnt/" if "system32" in bash.lower() else "/"
        text = f"{prefix}{text[0].lower()}{text[2:]}"
    return shlex.quote(text)


def test_write_merged_profile_round_trip(tmp_path: Path) -> None:
    path = write_merged_profile(
        {"Simple Test": {"tests/a_test.py": 1.5}},
        tmp_path / "merged.json",
        builds=2,
        build_ids=["x"],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["builds"] == 2
    assert payload["steps"]["Simple Test"]["tests/a_test.py"] == 1.5
