# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from tests.e2e.offline_inference.test_qwen_image_21_mindiesd import _assert_route_hits

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("missing", ["prefill", "decode", "concat"])
def test_enabled_arm_rejects_missing_route(missing):
    hits = dict(prefill=1, decode=1, concat=1)
    hits[missing] = 0
    with pytest.raises(AssertionError, match="did not execute"):
        _assert_route_hits(hits, True)


@pytest.mark.parametrize("unexpected", ["prefill", "decode", "concat"])
def test_disabled_arm_rejects_optimized_route(unexpected):
    hits = dict(prefill=0, decode=0, concat=0)
    hits[unexpected] = 1
    with pytest.raises(AssertionError, match="Disabled arm"):
        _assert_route_hits(hits, False)


def test_expected_routes_pass():
    _assert_route_hits(dict(prefill=0, decode=0, concat=0), False)
    _assert_route_hits(dict(prefill=1, decode=2, concat=4), True)
