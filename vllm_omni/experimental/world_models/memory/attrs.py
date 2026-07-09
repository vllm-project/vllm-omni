# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Typed session-attribute descriptor (RFC #4480).

``SessionAttr`` lets an adapter declare each piece of session-scoped scalar or
tensor metadata as one class-level line instead of a hand-written property
pair. The descriptor reads and writes ``SessionMemory.attrs`` on the host
object's pinned ``_session``, using the attribute's own name as the key, so
each key string exists exactly once. An attribute can be marked as surviving a
window ("inference") reset; ``window_reset_survivors()`` enumerates the marked
names so reset logic never maintains a separate key list that could drift from
the declarations.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, Protocol, TypeVar, cast, overload

from vllm_omni.experimental.world_models.memory.manager import SessionMemory

T = TypeVar("T")


class HasSessionMemory(Protocol):
    """Host contract: any object holding a pinned ``SessionMemory``."""

    _session: SessionMemory


class SessionAttr(Generic[T]):
    """Data descriptor for one session-scoped metadata value.

    Declared at class level on an adapter::

        class SomeAdapter:
            call_count = SessionAttr[int](default=0, coerce=int)
            language = SessionAttr[torch.Tensor | None](
                default=None, survives_window_reset=True)

    Reads return ``attrs.get(<name>, default)``; writes go through ``coerce``
    (when given) and straight into the session's ``attrs``, so a freshly
    constructed adapter for an existing session sees the same data.
    """

    def __init__(
        self,
        *,
        default: T,
        coerce: Callable[[T], T] | None = None,
        survives_window_reset: bool = False,
    ) -> None:
        self.default = default
        self.coerce = coerce
        self.survives_window_reset = survives_window_reset
        self.name = ""  # assigned by ``__set_name__``

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    @overload
    def __get__(self, obj: None, objtype: type | None = None) -> SessionAttr[T]: ...

    @overload
    def __get__(self, obj: HasSessionMemory, objtype: type | None = None) -> T: ...

    def __get__(self, obj: HasSessionMemory | None, objtype: type | None = None) -> SessionAttr[T] | T:
        if obj is None:
            return self
        return cast("T", obj._session.attrs.get(self.name, self.default))

    def __set__(self, obj: HasSessionMemory, value: T) -> None:
        obj._session.attrs[self.name] = self.coerce(value) if self.coerce is not None else value


def window_reset_survivors(obj: object) -> list[str]:
    """Names of ``SessionAttr`` declarations on ``type(obj)`` (and its bases)
    marked ``survives_window_reset=True``, in declaration order."""
    names: list[str] = []
    for klass in reversed(type(obj).__mro__):
        for name, attr in vars(klass).items():
            if isinstance(attr, SessionAttr) and attr.survives_window_reset and name not in names:
                names.append(name)
    return names
