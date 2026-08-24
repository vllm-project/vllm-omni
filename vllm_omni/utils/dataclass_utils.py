# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any, Protocol, TypeVar, runtime_checkable

_T = TypeVar("_T")


# Protocol wrapper for static analysis
@runtime_checkable
class Trackable(Protocol):
    _init_kwargs: set[str]


def trackable(cls: type[_T]) -> type[_T]:
    """Decorator that wraps __init__ to track which args/kwargs were explicitly
    passed by the caller without special handling. This is useful for a variety
    of situations, e.g., merging a user's passed sampling params with the default
    values provided by a pipeline.

    NOTE: This decorator preserves the original __init__ signature for
    type checkers while adding runtime tracking of explicitly-passed positional
    and keyword arguments.

    It is also important to consider that @trackable currently needs to be applied
    above @dataclass, since the consumed class is expected to be a dataclass.
    You should do this explicitly on any class expected to be @trackable, including
    the case where you are inheriting from a trackable superclass, otherwise you may
    see unexpected behaviors due to the way @trackable and @dataclass interact with
    the initializer. For example, inheriting from a @trackable dataclass and including
    only @dataclass on the superclass produces a non-trackable subclass unless
    you explicitly decorate the subclass as @trackable too.
    """

    # Currently we explicitly require anything @trackable to be a dataclass
    # so that we can use .bind on the signature (i.e., don't have to handle
    # variadic args), since everything we need it on is a dataclass anyway.
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"@trackable currently requires classes to be dataclasses, but {cls.__name__} is not one")
    original_init: Callable[..., None] = cls.__init__
    sig = inspect.signature(original_init)

    @wraps(original_init)
    def new_init(self: _T, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        # Map passed args/kwargs to wrapped initializer.
        bound = sig.bind(self, *args, **kwargs)
        bound.arguments.pop("self", None)
        self._init_kwargs: set[str] = set(bound.arguments)  # type: ignore[attr-defined]

    cls.__init__ = new_init
    return cls


def trackable_to_kwargs(obj: Trackable) -> dict[str, Any]:
    """Assuming an object is wrapped as Trackable, return the filtered kwargs.
    This is analogous to what TrackingArgumentParser does to an argparse namespace,
    but in application to classes like Dataclasses, etc.
    """
    if not isinstance(obj, Trackable):
        raise TypeError(f"Provided object of type {type(obj)} is not registered as trackable")
    return {kwarg: getattr(obj, kwarg) for kwarg in obj._init_kwargs}
