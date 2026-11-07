# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The session -> replica rule: bind once, follow the owner, never move silently."""

from __future__ import annotations

import pytest

from vllm_omni.experimental.ar_diffusion.session_affinity import SessionAffinityError, SessionRouter

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_a_follow_up_is_routed_to_the_owner_even_when_it_is_not_first():
    router = SessionRouter()
    router.bind("s1", 1)

    assert router.route("s1", [0, 1]) == 1


def test_an_unbound_session_binds_on_its_first_route():
    router = SessionRouter()

    assert router.route("s1", [2, 0]) == 2
    assert router.owner("s1") == 2


def test_a_session_cannot_be_rebound_to_another_replica():
    router = SessionRouter()
    router.bind("s1", 0)

    with pytest.raises(SessionAffinityError):
        router.bind("s1", 1)


def test_a_lost_owner_is_refused_unless_a_cold_restart_is_accepted():
    router = SessionRouter()
    router.bind("s1", 0)

    with pytest.raises(SessionAffinityError):
        router.route("s1", [1])
    assert router.owner("s1") == 0 and not router.is_stale("s1")

    assert router.route("s1", [1], allow_cold=True) == 1
    assert router.owner("s1") == 1
    assert router.is_stale("s1"), "the dropped history must stay visible"


def test_a_dead_replica_releases_its_sessions_and_records_the_loss():
    router = SessionRouter()
    router.bind("s1", 0)
    router.bind("s2", 1)

    assert router.release_replica(0) == ["s1"]
    assert router.owner("s1") is None and router.is_stale("s1")
    assert router.owner("s2") == 1

    router.release("s1")
    assert not router.is_stale("s1") and len(router) == 1


def test_no_live_replica_is_refused():
    with pytest.raises(SessionAffinityError):
        SessionRouter().route("s1", [])
