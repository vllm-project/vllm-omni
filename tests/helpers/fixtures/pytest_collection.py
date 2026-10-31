# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Pytest collection-phase hooks.

All ``pytest_collection_modifyitems`` policies live here so their order is
explicit. Add a named helper and call it from the hook; do not register a
second ``pytest_collection_modifyitems`` in another plugin.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest

_CPU_XDIST_SCOPE = pytest.StashKey[str]()
_CPU_XDIST_WORKER_PORTS = pytest.StashKey[set[int]]()
_CPU_XDIST_SCOPE_KEY = "omni_cpu_xdist_scope"
_CPU_XDIST_PORT_KEY = "omni_cpu_xdist_master_port"


def _matches_cpu_xdist_selection(config):
    tests = Path(config.rootpath) / "tests"
    paths = (tests / "entrypoints", tests / "engine", tests / "diffusion")
    ignored = (tests / "diffusion" / "models",)
    actual_paths = tuple(Path(path).resolve() for path in config.args)
    actual_ignored = tuple(Path(path).resolve() for path in (config.option.ignore or []))
    filtered = any(
        getattr(config.option, key, None) for key in ("keyword", "deselect", "ignore_glob", "lf", "stepwise")
    )
    return (
        actual_paths == paths
        and actual_ignored == ignored
        and config.option.markexpr == "core_model and cpu"
        and not filtered
    )


def pytest_configure(config):
    """Authorize file-level workers only for the unchanged Simple CPU lane."""
    scope = os.environ.get("VLLM_OMNI_CPU_XDIST_SCOPE")
    if scope is None:
        return
    if scope != "simple-engine":
        raise pytest.UsageError("Unknown VLLM_OMNI_CPU_XDIST_SCOPE")
    if not config.pluginmanager.hasplugin("xdist"):
        raise pytest.UsageError("Scoped CPU execution requires pytest-xdist")
    if getattr(config.option, "num_shards", 1) != 1 or getattr(config.option, "shard_id", 0) != 0:
        raise pytest.UsageError("Scoped CPU execution requires the unsharded inventory")
    if not _matches_cpu_xdist_selection(config):
        raise pytest.UsageError(f"CPU xdist selection does not match the {scope} lane")
    if hasattr(config, "workerinput"):
        # xdist resets worker dist to "no". Workers must retain the exact
        # selection and scope authorized by their four-worker controller.
        if config.workerinput.get(_CPU_XDIST_SCOPE_KEY) != scope:
            raise pytest.UsageError("CPU xdist worker has no matching controller authorization")
        port = config.workerinput.get(_CPU_XDIST_PORT_KEY)
        if not isinstance(port, int) or not 0 < port < 65536:
            raise pytest.UsageError("CPU xdist worker has no valid controller port")
        os.environ["MASTER_PORT"] = str(port)
    elif config.getoption("dist") != "loadfile" or config.getoption("numprocesses") != 4:
        raise pytest.UsageError("Scoped CPU execution requires -n 4 --dist=loadfile")
    config.stash[_CPU_XDIST_SCOPE] = scope


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node):
    scope = node.config.stash.get(_CPU_XDIST_SCOPE, None)
    if scope is None:
        return
    used = node.config.stash.setdefault(_CPU_XDIST_WORKER_PORTS, set())
    while True:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        if port not in used:
            used.add(port)
            node.workerinput[_CPU_XDIST_SCOPE_KEY] = scope
            node.workerinput[_CPU_XDIST_PORT_KEY] = port
            return


def pytest_collection_modifyitems(config, items):
    _pin_xdist_serial(config, items)


def _pin_xdist_serial(config, items):
    """Pin unmarked tests to one worker except the validated Simple CPU lane."""
    if not config.pluginmanager.hasplugin("xdist"):
        return
    if config.stash.get(_CPU_XDIST_SCOPE, None) is not None:
        return
    for item in items:
        if not item.get_closest_marker("xdist"):
            item.add_marker(pytest.mark.xdist_group(name="serial"))
