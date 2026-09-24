# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Typed session-attribute descriptor (RFC #4480).

``SessionAttr`` lets an adapter declare each piece of session-scoped scalar or
tensor metadata as one class-level line instead of a hand-written property
pair. The descriptor reads and writes ``SessionState.attrs`` on the host
object's pinned ``_session``, using the attribute's own name as the key, so
each key string exists exactly once. Because the read falls back to the
declared default when the key is absent, an adapter clears a value simply by
assigning the default (or by clearing ``attrs``), keeping its method bodies a
line-for-line mirror of the model's original state class.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, Protocol, TypeVar, cast, overload

from vllm_omni.experimental.world_models.session_state.manager import SessionState

T = TypeVar("T")


class HasSessionState(Protocol):
    """Host contract: any object holding a pinned ``SessionState``."""

    _session: SessionState


class SessionAttr(Generic[T]):
    """Data descriptor for one session-scoped metadata value.

    Declared at class level on an adapter::

        class SomeAdapter:
            call_count = SessionAttr[int](default=0, coerce=int)
            language = SessionAttr[torch.Tensor | None](default=None)

    Reads return ``attrs.get(<name>, default)``; writes go through ``coerce``
    (when given) and straight into the session's ``attrs``, so a freshly
    constructed adapter for an existing session sees the same data.
    """

    def __init__(
        self,
        *,
        default: T,
        coerce: Callable[[T], T] | None = None,
    ) -> None:
        self.default = default
        self.coerce = coerce
        self.name = ""  # assigned by ``__set_name__``

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    @overload
    def __get__(self, obj: None, objtype: type | None = None) -> SessionAttr[T]: ...

    @overload
    def __get__(self, obj: HasSessionState, objtype: type | None = None) -> T: ...

    def __get__(self, obj: HasSessionState | None, objtype: type | None = None) -> SessionAttr[T] | T:
        if obj is None:
            return self
        return cast("T", obj._session.attrs.get(self.name, self.default))

    def __set__(self, obj: HasSessionState, value: T) -> None:
        obj._session.attrs[self.name] = self.coerce(value) if self.coerce is not None else value
