# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Pytest collection-phase hooks.

All ``pytest_collection_modifyitems`` policies live here so their order is
explicit. Add a named helper and call it from the hook; do not register a
second ``pytest_collection_modifyitems`` in another plugin. Policies placed
after ``_partition_cpu_suite`` see only the selected timing shard, not the
full filtered collection.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from dataclasses import asdict
from pathlib import Path

import pytest

from tests.helpers.cpu_sharding import TimingProfile, build_plan, load_profile

_CPU_PROFILE = pytest.StashKey[TimingProfile]()


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("AMD CPU suite sharding")
    group.addoption("--cpu-shard-profile", help="Opt in to module-based CPU sharding with a timing profile")
    group.addoption("--cpu-shard-count", type=int, default=3)
    group.addoption("--cpu-shard-index", type=int)
    group.addoption("--cpu-shard-plan", help="New JSON output path, unique per job/shard/invocation")


def pytest_configure(config: pytest.Config) -> None:
    path = config.getoption("cpu_shard_profile")
    index = config.getoption("cpu_shard_index")
    plan = config.getoption("cpu_shard_plan")
    if path is None:
        if index is not None or plan is not None:
            raise pytest.UsageError("CPU shard index/plan require --cpu-shard-profile")
        return
    count = config.getoption("cpu_shard_count")
    if count < 1 or index is None or not 0 <= index < count or not plan:
        raise pytest.UsageError("CPU sharding requires a positive count, an index in [0, count), and --cpu-shard-plan")
    if config.getoption("num_shards", default=1) != 1 or config.getoption("shard_id", default=0) != 0:
        raise pytest.UsageError("CPU timing sharding cannot be combined with pytest-shard hash sharding")
    if config.getoption("numprocesses", default=0) not in (None, 0):
        raise pytest.UsageError("CPU timing sharding cannot be combined with xdist workers")
    try:
        config.stash[_CPU_PROFILE] = load_profile(Path(path))
    except (OSError, ValueError) as error:
        raise pytest.UsageError(f"Cannot load CPU shard profile: {error}") from error


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> Generator[None, None, None]:
    # xdist's worker copies ``xdist_group`` onto the nodeid in its own
    # collection hook. This hook has to run first, or ``--dist=loadgroup``
    # never sees the group and schedules each test on its own.
    _pin_xdist_serial(config, items)
    yield
    # All marker/-k/--deselect policies must run before timing-based partitioning.
    _partition_cpu_suite(config, items)


def _partition_cpu_suite(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _CPU_PROFILE not in config.stash:
        return
    profile = config.stash[_CPU_PROFILE]
    if any(
        not item.get_closest_marker("cpu")
        or not item.get_closest_marker("core_model")
        or (profile.suite == "model-executor" and item.get_closest_marker("omni"))
        or (profile.suite == "diffusion" and any(item.get_closest_marker(f"cards_{count}") for count in range(2, 9)))
        for item in items
    ):
        raise pytest.UsageError(f"CPU timing sharding requires the {profile.suite} CPU marker contract")
    try:
        plan = build_plan([item.nodeid for item in items], profile, config.getoption("cpu_shard_count"))
        destination = Path(config.getoption("cpu_shard_plan"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Refuse to overwrite another shard's diagnostic output.
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(asdict(plan), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError, OverflowError) as error:
        raise pytest.UsageError(f"Cannot create CPU shard plan: {error}") from error
    index = config.getoption("cpu_shard_index")
    deselected = [item for item in items if plan.node_shards[item.nodeid] != index]
    items[:] = [item for item in items if plan.node_shards[item.nodeid] == index]
    config.hook.pytest_deselected(items=deselected)


def _pin_xdist_serial(config, items):
    """Pin unmarked tests to one xdist worker.

    Only tests explicitly marked ``pytest.mark.xdist`` run concurrently.
    Others are pinned to a single xdist worker. ``--dist=loadgroup`` honors
    that group; ``--dist=loadfile`` does not, so the CPU coverage job must
    not select it.

    ``cpu`` tests are that job's set. They opt in here, one xdist group per
    module, so workers can split files without splitting a module or running
    an unmarked test beside them.
    """
    if not config.pluginmanager.hasplugin("xdist"):
        return
    for item in items:
        if item.get_closest_marker("xdist"):
            continue
        if item.get_closest_marker("cpu"):
            item.add_marker(pytest.mark.xdist)
            item.add_marker(pytest.mark.xdist_group(name=f"file:{item.path.as_posix()}"))
            continue
        item.add_marker(pytest.mark.xdist_group(name="serial"))
