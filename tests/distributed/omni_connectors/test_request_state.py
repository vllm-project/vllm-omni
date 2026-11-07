# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import threading
from dataclasses import dataclass

import pytest

from vllm_omni.distributed.omni_connectors.transfer_adapter.request_state import (
    RequestStateAccessor,
    RequestStateUnavailableError,
    StageRequestIdentity,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class ProducerState:
    chunks: int = 0


def make_view(payloads, owners, identity, lock=None):
    owner = owners[identity.external_request_id]
    return RequestStateAccessor(
        payloads,
        identity,
        owner,
        lock or threading.Lock(),
        lambda: owners.get(identity.external_request_id) is owner,
    )


def test_namespaces_and_interleaved_requests_are_isolated():
    payloads: dict[str, object] = {}
    owners = {"a": object(), "b": object()}
    a = StageRequestIdentity("internal-a", "a")
    b = StageRequestIdentity("internal-b", "b")
    first = make_view(payloads, owners, a)
    second = make_view(payloads, owners, b)
    state = first.get_or_create(a, "producer.codec", ProducerState)
    state.chunks = 3
    assert first.get_or_create(a, "producer.codec", ProducerState) is state
    assert first.get_or_create(a, "producer.prompt", ProducerState).chunks == 0
    assert second.get_or_create(b, "producer.codec", ProducerState).chunks == 0


def test_release_and_id_reuse_fence_retained_accessors():
    payloads: dict[str, object] = {}
    owners = {"external": object()}
    old_id = StageRequestIdentity("old-internal", "external")
    old = make_view(payloads, owners, old_id)
    old.get_or_create(old_id, "codec", ProducerState).chunks = 4
    payloads.pop("external", None)
    owners.pop("external", None)
    with pytest.raises(RequestStateUnavailableError):
        old.get_or_create(old_id, "late", ProducerState)
    owners["external"] = object()
    new_id = StageRequestIdentity("new-internal", "external")
    new = make_view(payloads, owners, new_id)
    assert new.get_or_create(new_id, "codec", ProducerState).chunks == 0
    with pytest.raises(RequestStateUnavailableError):
        old.get_or_create(old_id, "codec", ProducerState)
    with pytest.raises(RequestStateUnavailableError):
        new.get_or_create(old_id, "codec", ProducerState)


def test_cleanup_during_factory_cannot_resurrect_state():
    payloads: dict[str, object] = {}
    owners = {"external": object()}
    identity = StageRequestIdentity("internal", "external")
    lock = threading.Lock()
    view = make_view(payloads, owners, identity, lock)

    def factory():
        # This would deadlock if the user factory held the sender lock.
        with lock:
            owners.pop("external")
        return ProducerState()

    with pytest.raises(RequestStateUnavailableError):
        view.get_or_create(identity, "codec", factory)
    assert payloads == {}


def test_factory_failure_leaves_no_partial_container():
    payloads: dict[str, object] = {}
    owners = {"external": object()}
    identity = StageRequestIdentity("internal", "external")
    view = make_view(payloads, owners, identity)

    def factory():
        raise ValueError("invalid conditioning")

    with pytest.raises(ValueError, match="invalid conditioning"):
        view.get_or_create(identity, "codec", factory)
    assert payloads == {}
    assert view.get_or_create(identity, "codec", ProducerState).chunks == 0


def test_legacy_state_is_not_overwritten():
    legacy = {"frames": [1, 2]}
    payloads = {"external": legacy}
    owners = {"external": object()}
    identity = StageRequestIdentity("internal", "external")
    with pytest.raises(RequestStateUnavailableError, match="legacy"):
        make_view(payloads, owners, identity).get_or_create(identity, "codec", ProducerState)
    assert payloads["external"] is legacy


@pytest.mark.parametrize("namespace", ["", "   "])
def test_empty_namespace_is_rejected(namespace):
    identity = StageRequestIdentity("internal", "external")
    with pytest.raises(ValueError, match="namespace"):
        make_view({}, {"external": object()}, identity).get_or_create(identity, namespace, ProducerState)
