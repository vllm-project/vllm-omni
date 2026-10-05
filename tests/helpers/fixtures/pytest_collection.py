# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Pytest collection-phase hooks.

All ``pytest_collection_modifyitems`` policies live here so their order is
explicit. Add a named helper and call it from the hook; do not register a
second ``pytest_collection_modifyitems`` in another plugin.
"""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config, items):
    _pin_xdist_serial(config, items)


def _pin_xdist_serial(config, items):
    """Pin unmarked tests to one xdist worker.

    Only tests explicitly marked ``pytest.mark.xdist`` run concurrently.
    Others are pinned to a single xdist worker.
    """
    if not config.pluginmanager.hasplugin("xdist"):
        return
    for item in items:
        if not item.get_closest_marker("xdist"):
            item.add_marker(pytest.mark.xdist_group(name="serial"))
