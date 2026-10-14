# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load per-role parallel strategies from a strategy file.

The strategy file is a small, optional companion to the deploy config. It maps a
logical role (``model_stage``, e.g. ``thinker``) to a list of axis declarations,
each of which becomes one :class:`StrategySpec`. YAML is the only format wired up
today, but the module/symbol names stay format-agnostic so a different source can
be added without renaming the public surface::

    strategies:
      thinker:
        - axis: tp
          size: 2
      talker:
        - axis: stage_replica
          size: 2
          routing: round_robin   # random | round_robin | least_queue
      code2wav:
        - axis: stage_replica
          size: 2
          routing: round_robin

Per-axis fields:

* ``axis`` (required): mesh-axis kind (``tp``, ``dp``, ``pp``, ``ep``,
  ``stage_replica`` are translatable today).
* ``size`` (required): axis degree (> 0).
* ``routing`` (optional): stateless policy for ``dp`` / ``stage_replica`` axes
  (``random`` / ``round_robin`` / ``least_queue``). Ignored for axes whose
  routing is fixed (``tp`` broadcast, ``pp`` microbatch).
* ``l1_owner`` (optional): ``engine`` or ``delegated``; overrides the default
  owner for the axis (recorded in ``shard_extension``).
* ``name`` (optional): a label for the spec; defaults to the axis kind.

Routing/aggregation patterns are filled in from sensible per-kind defaults so
the file stays compact; the translator validates the combination.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vllm_omni.config.composable_parallel.aggregation import (
    AggregationPattern,
    FanInByStage,
    StitchPipeline,
    TakeRank,
    Union,
)
from vllm_omni.config.composable_parallel.routing import (
    Broadcast,
    PipelineMicrobatch,
    RouteByStage,
    RoutingPattern,
)
from vllm_omni.config.composable_parallel.spec import MeshAxisSpec, StrategySpec
from vllm_omni.config.yaml_util import load_yaml_config, to_dict

# Per-kind defaults for routing + aggregation, so the file only needs axis+size.
_ROUTING_POLICY_KINDS = frozenset({"dp", "stage_replica"})


class StrategyLoadError(ValueError):
    """Raised when a strategy file is malformed."""


def _default_routing(kind: str, routing_policy: str | None) -> RoutingPattern:
    if kind in _ROUTING_POLICY_KINDS:
        return RouteByStage(routing_policy=routing_policy or "round_robin")  # type: ignore[arg-type]
    if kind == "pp":
        return PipelineMicrobatch()
    # tp, ep, and any other kind default to broadcast (the translator rejects
    # kinds it cannot handle before it ever inspects routing).
    return Broadcast()


def _default_aggregation(kind: str) -> AggregationPattern:
    if kind == "stage_replica":
        return FanInByStage()
    if kind == "pp":
        return StitchPipeline()
    if kind in ("dp", "ep"):
        return Union()
    return TakeRank()


def _build_spec(role: str, entry: Mapping[str, Any]) -> StrategySpec:
    if "axis" not in entry:
        raise StrategyLoadError(f"role {role!r}: each strategy entry needs an 'axis' field")
    if "size" not in entry:
        raise StrategyLoadError(f"role {role!r}: strategy entry for axis {entry['axis']!r} needs a 'size'")

    kind = str(entry["axis"])
    try:
        size = int(entry["size"])
    except (TypeError, ValueError) as exc:
        raise StrategyLoadError(f"role {role!r}: axis {kind!r} size must be an integer, got {entry['size']!r}") from exc

    routing_policy = entry.get("routing")
    if routing_policy is not None and kind not in _ROUTING_POLICY_KINDS:
        raise StrategyLoadError(
            f"role {role!r}: axis {kind!r} does not accept a 'routing' policy (only {sorted(_ROUTING_POLICY_KINDS)} do)"
        )

    shard_extension: dict[str, Any] = {}
    l1_owner = entry.get("l1_owner")
    if l1_owner is not None:
        shard_extension["l1_owner"] = str(l1_owner)

    return StrategySpec(
        name=str(entry.get("name", kind)),
        mesh_axis=MeshAxisSpec(kind=kind, size=size),  # type: ignore[arg-type]
        routing=_default_routing(kind, routing_policy),
        aggregation=_default_aggregation(kind),
        shard_extension=shard_extension,
    )


def parse_strategy_specs(data: Mapping[str, Any]) -> dict[str, list[StrategySpec]]:
    """Parse an already-loaded strategy mapping into per-role spec stacks."""
    strategies = data.get("strategies", data)
    if not isinstance(strategies, Mapping):
        raise StrategyLoadError("strategy file must contain a 'strategies' mapping of role -> list of axes")

    result: dict[str, list[StrategySpec]] = {}
    for role, entries in strategies.items():
        if not isinstance(entries, (list, tuple)):
            raise StrategyLoadError(
                f"role {role!r}: expected a list of axis declarations, got {type(entries).__name__}"
            )
        specs: list[StrategySpec] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise StrategyLoadError(
                    f"role {role!r}: each axis declaration must be a mapping, got {type(entry).__name__}"
                )
            specs.append(_build_spec(str(role), dict(entry)))
        result[str(role)] = specs
    return result


def load_strategy_specs(path: str) -> dict[str, list[StrategySpec]]:
    """Load and parse a strategy file into per-role spec stacks."""
    data = to_dict(load_yaml_config(path))
    if not isinstance(data, Mapping):
        raise StrategyLoadError(f"strategy file {path!r} did not parse to a mapping")
    return parse_strategy_specs(data)
