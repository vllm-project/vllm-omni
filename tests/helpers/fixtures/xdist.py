"""
Only tests explicitly marked ``pytest.mark.xdist`` run concurrently.
Other test are pinned to a single xdist worker.
"""

import pytest


def pytest_collection_modifyitems(config, items):
    if not config.pluginmanager.hasplugin("xdist"):
        return
    for item in items:
        if not item.get_closest_marker("xdist"):
            item.add_marker(pytest.mark.xdist_group(name="serial"))
