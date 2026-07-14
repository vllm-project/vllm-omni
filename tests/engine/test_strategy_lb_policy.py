# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for strategy-derived omni_lb_policy threading into AsyncOmniEngine.

``AsyncOmniEngine._apply_strategy_lb_policy`` only touches ``self._omni_lb_policy``
and module-level logging, so we exercise it with a lightweight stub ``self`` to
avoid standing up a full engine (which needs GPUs/model weights).
"""

from __future__ import annotations

import pytest

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Stub:
    def __init__(self, current: str = "random") -> None:
        self._omni_lb_policy = current


_apply = AsyncOmniEngine._apply_strategy_lb_policy


def test_no_derived_policy_is_noop():
    stub = _Stub("random")
    _apply(stub, None, {})
    assert stub._omni_lb_policy == "random"


def test_derived_overrides_default_random():
    # "random" is the engine default == "unset", so the strategy value wins.
    stub = _Stub("random")
    _apply(stub, "round-robin", {})
    assert stub._omni_lb_policy == "round-robin"


def test_derived_overrides_when_flag_absent():
    stub = _Stub("random")
    _apply(stub, "least-queue-length", {"omni_lb_policy": None})
    assert stub._omni_lb_policy == "least-queue-length"


def test_explicit_matching_flag_is_fine():
    stub = _Stub("round-robin")
    _apply(stub, "round-robin", {"omni_lb_policy": "round-robin"})
    assert stub._omni_lb_policy == "round-robin"


def test_explicit_conflicting_flag_raises():
    stub = _Stub("round-robin")
    with pytest.raises(ValueError, match="Conflicting load-balancer policy"):
        _apply(stub, "least-queue-length", {"omni_lb_policy": "round-robin"})


def test_explicit_random_is_treated_as_unset():
    # An explicit --omni-lb-policy=random is indistinguishable from the default,
    # so a strategy value still overrides it (documented Phase-1 semantics).
    stub = _Stub("random")
    _apply(stub, "round-robin", {"omni_lb_policy": "random"})
    assert stub._omni_lb_policy == "round-robin"
