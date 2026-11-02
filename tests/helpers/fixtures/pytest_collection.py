# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Pytest collection-phase hooks.

All ``pytest_collection_modifyitems`` policies live here so their order is
explicit. Add a named helper and call it from the hook; do not register a
second ``pytest_collection_modifyitems`` in another plugin.
"""

from __future__ import annotations

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    # xdist's worker copies ``xdist_group`` onto the nodeid in its own
    # collection hook. This hook has to run first, or ``--dist=loadgroup``
    # never sees the group and schedules each test on its own.
    _pin_xdist_serial(config, items)


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
