"""Pytest hook to group MiniCPM-o online tests contiguously.

When ``minicpmo_duplex_server`` tests are scattered across the
collection order, the reference-counted server singleton (in
``runtime.py``) would start and stop repeatedly. This hook moves all
tests that depend on the fixture to a contiguous block so the server
starts once and stops when the last consumer finishes.
"""

from __future__ import annotations

_MINICPMO_CONFTEST_DIRS = (
    "tests/e2e/online_serving/minicpmo",
    "tests/dfx/reliability/invalid_param_test/minicpmo",
)


def pytest_collection_modifyitems(items: list) -> None:
    minicpmo = []
    other = []
    for item in items:
        path = item.nodeid
        if any(path.startswith(d) for d in _MINICPMO_CONFTEST_DIRS):
            minicpmo.append(item)
        else:
            other.append(item)
    if minicpmo:
        items[:] = other + minicpmo
