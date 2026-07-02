# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
from typing import Any

from vllm.utils.argparse_utils import FlexibleArgumentParser

UNSET = object()


def build_shadow_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Build kwargs for the shadow argument with an ``UNSET`` default.

    Actions that mutate their default in place (append/extend/append_const/
    count) would crash on the bare ``UNSET`` sentinel, so they are remapped to
    an equivalent store-style action; the shadow value only needs to flip away
    from ``UNSET`` when the arg is passed explicitly.
    """
    shadow_kwargs = {**kwargs, "default": UNSET}
    action = kwargs.get("action")

    if action in ("append", "extend"):
        shadow_kwargs["action"] = "store"

    elif action in ("append_const", "count"):
        shadow_kwargs["action"] = "store_const"

        if action == "count":
            shadow_kwargs["const"] = True

    return shadow_kwargs


class TrackingNamespace(argparse.Namespace):
    """Proxy that wraps an argparse namespace with explicit keys, which
    can be filtered down to a dict containing only explicitly passed values.
    """

    def __init__(self, unfiltered_ns: argparse.Namespace, explicit_keys: frozenset[str]) -> None:
        # We never have nested tracking namespaces, but explicitly guard
        # against them to prevent bad behavior with nested __dict__ overrides.
        if isinstance(unfiltered_ns, TrackingNamespace):
            raise ValueError("Tracking namespaces cannot be nested")

        self.unfiltered_ns = unfiltered_ns
        self.explicit_keys = explicit_keys

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("unfiltered_ns", "explicit_keys"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.unfiltered_ns, name, value)

    def get_explicit_kwargs_dict(self):
        """Return a dict containing only the explicitly passed key-value pairs."""
        return {k: v for k, v in vars(self.unfiltered_ns).items() if k in self.explicit_keys}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.unfiltered_ns, name)

    @property
    def __dict__(self):
        # NOTE: We do this so that vars() etc forward directly into the encapsulated namespace,
        # which makes this class a drop-in replacement for the original namespace, while also
        # ensuring that updates to the encapsulated namespace are correctly reflected.
        return self.unfiltered_ns.__dict__


class TrackingGroup:
    """Proxy that wraps an argument group and its corresponding shadow group."""

    def __init__(
        self,
        real_group: argparse._ArgumentGroup,
        shadow_group: argparse._ArgumentGroup,
    ):
        self._real = real_group
        self._shadow = shadow_group

    def add_argument(self, *args: Any, **kwargs: Any) -> argparse.Action:
        """Add an argument to the real group and to the shadow group."""
        action = self._real.add_argument(*args, **kwargs)
        default_kwargs = build_shadow_kwargs(kwargs)
        self._shadow.add_argument(*args, **default_kwargs)
        return action

    def __getattr__(self, name: str) -> Any:
        # Any attribute access is forwarded to the real argument group.
        return getattr(self._real, name)


class TrackingSubparsers:
    """Proxy that wraps a subparser and its corresponding shadow subparser."""

    def __init__(
        self,
        real_sub: argparse._SubParsersAction,
        shadow_sub: argparse._SubParsersAction,
    ):
        self._real = real_sub
        self._shadow = shadow_sub

    def add_parser(self, name, *args, **kwargs):
        """Add a parser to the encapsulated real parser and its shadow."""
        real_parser = self._real.add_parser(name, *args, **kwargs)
        # real_parser is a TrackingArgumentParser with its own _shadow.
        # Reuse that shadow as the parent shadow's child — so when
        # real_parser.add_argument() mirrors to real_parser._shadow,
        # the parent's shadow sees it too.
        self._shadow._name_parser_map[name] = real_parser._shadow
        return real_parser

    def __getattr__(self, name: str) -> Any:
        # Any attribute access is forwarded to the real subparser.
        return getattr(self._real, name)


class TrackingArgumentParser(FlexibleArgumentParser):
    """Drop-in replacement for FlexibleArgumentParser, which tracks keys that
    were explicitly passed as args on the parser namespace.

    Unfortunately, Argparse does not provide an easy way of doing this without
    depending on a lot of internal attributes, so we implement it by instead
    using a 'shadow' parser, which is essentially a clone of the parser, where
    defaults are overridden to `None`. By comparing the parser against its
    shadow, we can tell which values were passed in a non-destructive manner.
    """

    def __init__(self, *args, **kwargs):
        # NOTE: We have to define the shadow parser before calling init,
        # with add_help=False, since otherwise init will call add_argument
        # and delegate to the override on this class and cause problems.
        shadow_kwargs = {**kwargs, "add_help": False}
        self._shadow = FlexibleArgumentParser(*args, **shadow_kwargs)
        super().__init__(*args, **kwargs)

    def add_argument(self, *args: Any, **kwargs: Any) -> argparse.Action:
        """Add an arg to the parser & the shadow, where the latter has UNSET for the default."""
        action = super().add_argument(*args, **kwargs)
        shadow_kwargs = build_shadow_kwargs(kwargs)
        self._shadow.add_argument(*args, **shadow_kwargs)
        return action

    def add_argument_group(self, *args, **kwargs) -> TrackingGroup:
        real_group = super().add_argument_group(*args, **kwargs)
        shadow_group = self._shadow.add_argument_group(*args, **kwargs)
        return TrackingGroup(real_group, shadow_group)

    def add_mutually_exclusive_group(self, *args, **kwargs) -> TrackingGroup:
        real_group = super().add_mutually_exclusive_group(*args, **kwargs)
        shadow_group: argparse._MutuallyExclusiveGroup = self._shadow.add_mutually_exclusive_group(*args, **kwargs)
        return TrackingGroup(real_group, shadow_group)

    def add_subparsers(self, *args, **kwargs) -> TrackingSubparsers:
        real_sub = super().add_subparsers(*args, **kwargs)
        shadow_sub = self._shadow.add_subparsers(*args, **kwargs)
        return TrackingSubparsers(real_sub, shadow_sub)

    def build_tracking_namespace(self, real_ns: argparse.Namespace, shadow_ns: argparse.Namespace) -> TrackingNamespace:
        """Build a tracking namespace for the real / shadow namespaces."""
        explicit_keys = frozenset(k for k, v in vars(shadow_ns).items() if v is not UNSET)
        return TrackingNamespace(real_ns, explicit_keys)

    def parse_args(
        self,
        args: list[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> TrackingNamespace:
        """Parse the args on the real/shadow parser."""
        # Only the real parser should use the namespace if one is,
        # given since shadow parser will set its own defaults to None.
        real_ns = super().parse_args(args, namespace)
        shadow_ns = self._shadow.parse_args(args)
        if real_ns is None or shadow_ns is None:
            raise ValueError("Parse args created empty namespaces")

        # If this is called through parse_known_args on self, we will already
        # get a TrackingNamespace back, which will already have set the explicit
        # keys through build_tracking_namespace, so no need to do it again.
        if isinstance(real_ns, TrackingNamespace):
            return real_ns

        return self.build_tracking_namespace(real_ns, shadow_ns)

    def parse_known_args(
        self,
        args: list[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> tuple[TrackingNamespace, list[str]]:
        """Parse the known args on the real/shadow parser."""
        real_ns, remaining = super().parse_known_args(args, namespace)
        shadow_ns, _ = self._shadow.parse_known_args(args)
        tracked_ns = self.build_tracking_namespace(real_ns, shadow_ns)

        return tracked_ns, remaining
