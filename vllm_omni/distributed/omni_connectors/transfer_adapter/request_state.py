# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections.abc import Callable, MutableMapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import TypeVar, cast

T = TypeVar("T")


@dataclass(frozen=True)
class StageRequestIdentity:
    """Keep scheduler and connector identities explicit at the producer boundary."""

    request_id: str
    external_request_id: str


class RequestStateUnavailableError(RuntimeError):
    """The producer no longer owns this request's mutable state."""


@dataclass
class _RequestNamespaces:
    owner: object
    states: dict[str, object] = field(default_factory=dict)


class RequestStateAccessor:
    """A producer-invocation view of the adapter's canonical payload container.

    Retained views are fenced by the sender generation, including cancellation.
    Whole-request release remains the adapter's responsibility. Legacy raw
    payload entries remain supported, but cannot be mixed with this API for the
    same request. Factories should construct state only, without connector I/O.
    """

    def __init__(
        self,
        payloads: MutableMapping[str, object],
        identity: StageRequestIdentity,
        owner: object,
        lock: AbstractContextManager[object],
        is_active: Callable[[], bool],
    ) -> None:
        self._payloads = payloads
        self._identity = identity
        self._owner = owner
        self._lock = lock
        self._is_active = is_active

    def get_or_create(self, identity: StageRequestIdentity, namespace: str, factory: Callable[[], T]) -> T:
        if identity != self._identity:
            raise RequestStateUnavailableError("The identity does not belong to this producer invocation")
        if not namespace or not namespace.strip():
            raise ValueError("State namespace must be non-empty")
        with self._lock:
            if not self._is_active():
                raise RequestStateUnavailableError("The sender generation has been cancelled or released")
            container = self._payloads.get(identity.external_request_id)
            if container is not None:
                if not isinstance(container, _RequestNamespaces):
                    raise RequestStateUnavailableError("Cannot mix namespaced state with a legacy payload entry")
                if container.owner is not self._owner:
                    raise RequestStateUnavailableError("The payload belongs to a different sender generation")
                if namespace in container.states:
                    return cast(T, container.states[namespace])

        # Model constructors run outside the sender lock. Cleanup and other
        # requests must remain able to progress even if a factory is slow.
        state = factory()
        with self._lock:
            if not self._is_active():
                raise RequestStateUnavailableError("The sender generation was released during state construction")
            container = self._payloads.get(identity.external_request_id)
            if container is None:
                container = _RequestNamespaces(self._owner)
                self._payloads[identity.external_request_id] = container
            elif not isinstance(container, _RequestNamespaces) or container.owner is not self._owner:
                raise RequestStateUnavailableError("The payload entry changed during state construction")
            return cast(T, container.states.setdefault(namespace, state))
