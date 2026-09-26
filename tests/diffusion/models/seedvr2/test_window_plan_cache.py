# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The window-SP plan cache is shared across requests.

Building a layout and its routing plans is pure host work, and on a large token
grid it costs seconds with the accelerators idle. A manager is created per
request, so the cache it reads must outlive the manager or a stream of
same-shaped requests pays that cost every time.
"""

from __future__ import annotations

import pytest

from vllm_omni.diffusion.models.seedvr2.window_sp import (
    WindowLayoutManager,
    WindowPlanCache,
    shared_plan_cache,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]
GRID = (2, 48, 64)
METHOD = "720pwin_by_size_bysize"


def _manager(grid: tuple[int, int, int], **kwargs) -> WindowLayoutManager:
    return WindowLayoutManager(grid, group=None, world_size=1, rank=0, **kwargs)


def test_a_second_request_with_the_same_geometry_reuses_the_layout() -> None:
    cache = WindowPlanCache()
    first = _manager(GRID, cache=cache)
    first.layout(METHOD)
    warm = cache.hits, cache.misses

    second = _manager(GRID, cache=cache)
    layout = second.layout(METHOD)

    assert cache.misses == warm[1], "a same-shaped request must not rebuild the layout"
    assert cache.hits > warm[0]
    assert layout is first.layout(METHOD), "both managers must see the same cached layout"


def test_a_different_grid_is_not_served_from_another_entry() -> None:
    cache = WindowPlanCache()
    _manager(GRID, cache=cache).layout(METHOD)
    before = cache.misses

    other = _manager((GRID[0], GRID[1] + 16, GRID[2]), cache=cache).layout(METHOD)

    assert cache.misses > before
    assert other.key.token_grid != GRID


def test_managers_share_one_cache_by_default() -> None:
    assert _manager(GRID).cache is shared_plan_cache()
    assert _manager(GRID).cache is _manager(GRID).cache


def test_an_explicit_cache_keeps_a_manager_off_the_shared_one() -> None:
    private = WindowPlanCache()
    manager = _manager(GRID, cache=private)
    manager.layout(METHOD)

    assert manager.cache is private
    assert private.misses > 0
