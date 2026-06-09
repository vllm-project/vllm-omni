# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Translate a ``StrategySpec`` stack into vLLM-Omni parallel sizing.

We read a stack of strategy specs (one per mesh axis), work out the
tensor/data/pipeline parallel sizes plus the number of stage replicas, and note
who is responsible for each axis's request routing.

vLLM-Omni has **two distinct data-distribution layers** (conflating them is the
#1 risk):

* **engine data parallelism** — upstream vLLM ``data_parallel_size`` inside a
  *single* engine: one DP coordinator, one set of weights, vLLM's own DP load
  balancer distributes requests across the DP ranks. This is the ``dp`` axis. It
  is a true world dimension (``tp * dp * pp``) and is realized **intra-engine**
  by vLLM, so its ``l1_owner`` is ``"engine"`` and it does NOT emit
  ``omni_lb_policy`` (vLLM's internal DP LB is not omni's StagePool LB).
* **omni stage replicas** — N *independent* engine replicas of one pipeline
  stage, coordinated by the omni coordinator and balanced by a load balancer
  over a StagePool (stateless policies only: random / round-robin /
  least-queue-length). This is the ``stage_replica`` axis. Each replica is its
  own world, so it is **not** a vLLM world dimension; it maps to the per-stage
  ``num_replicas`` count plus the pipeline-level ``omni_lb_policy`` string. Its
  ``l1_owner`` is ``"delegated"`` (omni's coordinator owns the routing).

What we deliberately don't do yet: key-stable "send the same key to the same
replica" routing (``PartitionByHash`` / ``RouteByStage("hash")``). Honoring that
means running a dedicated load balancer, which is a later piece of work. Until
then we reject it outright with a message that points the caller at the
stateless alternative, rather than quietly routing at random and pretending the
affinity held.

The output is a plain sizing struct keyed by the real ``OmniEngineArgs`` /
``EngineArgs`` field names, so this module runs on CPU and the deploy layer can
splat the result into a stage's engine args.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

from vllm_omni.config.composable_parallel.routing import (
    Broadcast,
    PartitionByHash,
    PipelineMicrobatch,
    RouteByStage,
    RoutingPattern,
)
from vllm_omni.config.composable_parallel.spec import MeshAxisKind, StrategySpec

# DP, TP, PP, (dense) EP and stage_replica translate today. DP/TP/PP are true
# world dimensions vLLM realizes intra-engine; dense EP is a flag over the
# existing TP*DP ranks; stage_replica is an omni-coordinator-level fan-out of
# independent engine replicas (NOT a vLLM world dimension). The remaining kinds
# (sequence/context parallel, CFG, VAE pipelines, ...) need per-layer hooks or
# custom collectives and arrive in later stages.
_SUPPORTED_KINDS: Tuple[MeshAxisKind, ...] = ("dp", "tp", "pp", "ep", "stage_replica")

# Which EngineArgs field each axis kind *sizes*. EP and stage_replica are
# deliberately absent: EP is not an independent world dimension but an
# ``enable_expert_parallel`` flag whose degree equals tp*dp; stage_replica is a
# per-stage deploy ``num_replicas`` count, not an intra-engine world dimension.
_AXIS_TO_ENGINE_FIELD: Dict[MeshAxisKind, str] = {
    "tp": "tensor_parallel_size",
    "dp": "data_parallel_size",
    "pp": "pipeline_parallel_size",
}

# Default L1 owner per axis kind. ``stage_replica`` is the only delegated axis:
# omni's coordinator/StagePool owns routing across replicas. ``dp`` is engine
# data parallelism, realized by vLLM's own intra-engine DP load balancer.
_DEFAULT_L1_OWNER: Dict[MeshAxisKind, str] = {
    "dp": "engine",
    "tp": "engine",
    "pp": "engine",
    "ep": "engine",
    "stage_replica": "delegated",
}

# How a RouteByStage policy maps onto omni's load balancer policy string. These
# are the stateless options omni can actually do; note there's no "hash" here,
# because omni has no key-stable balancer.
_STAGE_POLICY_TO_OMNI_LB: Dict[str, str] = {
    "random": "random",
    "round_robin": "round-robin",
    "least_queue": "least-queue-length",
}

_VALID_L1_OWNERS = frozenset({"delegated", "engine"})


class AxisTranslationError(ValueError):
    """Base error for spec->config translation problems."""


class UnsupportedAxisKind(AxisTranslationError):
    """Raised when a spec axis kind cannot be translated at this stage."""


class DuplicateAxisKind(AxisTranslationError):
    """Raised when the same axis kind appears more than once in the stack."""


class RoutingOwnershipError(AxisTranslationError):
    """Raised when an axis is assigned an owner that doesn't fit its kind."""


class UnsupportedRouting(AxisTranslationError):
    """Raised for routing we haven't built yet.

    Today that means key-stable DP routing (``PartitionByHash`` or
    ``RouteByStage("hash")``): honoring it needs a dedicated load balancer,
    which is still on the to-do list.
    """


@dataclass(frozen=True)
class OmniParallelConfig:
    """Result of translating a spec stack into omni parallel sizing."""

    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    # Dense expert parallelism: a flag over the TP*DP ranks, not a world axis.
    enable_expert_parallel: bool = False
    # Number of independent omni stage replicas (the ``stage_replica`` axis).
    # This is NOT a vLLM world dimension; it is the per-stage deploy
    # ``num_replicas`` count. 1 means a single (un-replicated) engine.
    stage_replica_size: int = 1
    # The omni StagePool LB policy string for the (delegated) stage_replica
    # axis, if any. Only meaningful when ``stage_replica_size > 1``.
    omni_lb_policy: Optional[str] = None
    # axis kind -> resolved L1 owner ("delegated" | "engine").
    l1_owners: Mapping[MeshAxisKind, str] = field(default_factory=dict)

    @property
    def world_size(self) -> int:
        # EP and stage_replica are intentionally excluded. EP reuses the TP*DP
        # ranks rather than adding a dimension; stage_replica spins up separate
        # engines (each its own world), not extra ranks in this engine's group.
        # This product is what vLLM calls ``world_size_across_dp`` (tp * pp * dp).
        return self.tensor_parallel_size * self.data_parallel_size * self.pipeline_parallel_size

    @property
    def delegated_axes(self) -> Tuple[MeshAxisKind, ...]:
        return tuple(kind for kind, owner in self.l1_owners.items() if owner == "delegated")

    def as_engine_kwargs(self) -> Dict[str, object]:
        """Return per-stage kwargs keyed by real OmniEngineArgs/EngineArgs field names.

        Two derived values are intentionally *not* emitted here because they are
        not per-stage engine args:

        * ``stage_replica_size`` — a per-stage deploy ``num_replicas`` knob
          (``StageDeployConfig``); the deploy layer consumes it separately.
        * ``omni_lb_policy`` — a pipeline-wide load-balancer policy the engine
          reads once at construction (see ``StrategyApplyResult.omni_lb_policy``);
          it is applied at the engine level, not folded into per-stage args.
        """
        kwargs: Dict[str, object] = {
            "tensor_parallel_size": self.tensor_parallel_size,
            "data_parallel_size": self.data_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
        }
        if self.enable_expert_parallel:
            kwargs["enable_expert_parallel"] = True
        return kwargs


def _is_affinity_dp_routing(routing: RoutingPattern) -> bool:
    """True when DP routing demands key-stable (hash) placement."""
    if isinstance(routing, PartitionByHash):
        return True
    if isinstance(routing, RouteByStage) and routing.routing_policy == "hash":
        return True
    return False


def _resolve_l1_owner(spec: StrategySpec) -> str:
    kind = spec.mesh_axis.kind
    owner = spec.shard_extension.get("l1_owner")
    if owner is None:
        owner = _DEFAULT_L1_OWNER.get(kind, "engine")
    if owner not in _VALID_L1_OWNERS:
        raise AxisTranslationError(
            f"axis {kind!r} has invalid l1_owner {owner!r}; expected one of {sorted(_VALID_L1_OWNERS)}"
        )
    return owner


def _validate_dp(spec: StrategySpec, owner: str) -> None:
    """Validate an engine data-parallel axis.

    DP is realized intra-engine by vLLM's own DP load balancer, so it is
    engine-owned and emits no ``omni_lb_policy`` (that string configures omni's
    *replica* balancer, a different layer). Key-stable (hash) request affinity is
    not something vLLM's DP LB guarantees, so it is rejected rather than silently
    dropped.
    """
    if owner != "engine":
        raise RoutingOwnershipError(
            f"dp axis {spec.name!r} is engine data parallelism realized intra-engine by "
            f"vLLM's DP load balancer; l1_owner must be 'engine', got {owner!r}. For "
            "omni-coordinator-level request fan-out across independent replicas, use a "
            "'stage_replica' axis."
        )
    if _is_affinity_dp_routing(spec.routing):
        raise UnsupportedRouting(
            f"dp axis {spec.name!r} requests key-stable (hash) routing, which vLLM's "
            "intra-engine DP load balancer does not guarantee — not supported yet. Use "
            "RouteByStage(random|round_robin|least_queue) for stateless DP balancing."
        )
    if not isinstance(spec.routing, RouteByStage):
        raise AxisTranslationError(
            f"dp axis {spec.name!r} expects RouteByStage(random|round_robin|least_queue) routing, "
            f"got {type(spec.routing).__name__}"
        )
    if spec.routing.routing_policy not in _STAGE_POLICY_TO_OMNI_LB:
        raise UnsupportedRouting(
            f"dp axis {spec.name!r} has unsupported routing_policy "
            f"{spec.routing.routing_policy!r}; expected one of "
            f"{sorted(_STAGE_POLICY_TO_OMNI_LB)}."
        )


def _stage_replica_lb_policy(spec: StrategySpec, owner: str) -> str:
    """Return the omni StagePool LB policy for a delegated stage_replica axis."""
    if owner != "delegated":
        raise RoutingOwnershipError(
            f"stage_replica axis {spec.name!r} must be 'delegated' to omni's StagePool load "
            f"balancer; got owner {owner!r}. Replica routing is owned by omni's coordinator."
        )

    routing = spec.routing
    if _is_affinity_dp_routing(routing):
        raise UnsupportedRouting(
            f"stage_replica axis {spec.name!r} requests key-stable (hash) routing, which needs a "
            "dedicated load balancer — not implemented yet. Use "
            "RouteByStage(random|round_robin|least_queue) to delegate to omni's load balancer."
        )
    if not isinstance(routing, RouteByStage):
        raise AxisTranslationError(
            f"stage_replica axis {spec.name!r} expects RouteByStage(random|round_robin|least_queue) "
            f"routing, got {type(routing).__name__}"
        )
    policy = _STAGE_POLICY_TO_OMNI_LB.get(routing.routing_policy)
    if policy is None:
        raise UnsupportedRouting(
            f"stage_replica axis {spec.name!r} routing_policy {routing.routing_policy!r} "
            "has no omni LB policy"
        )
    return policy


def _validate_tp(spec: StrategySpec, owner: str) -> None:
    if not isinstance(spec.routing, Broadcast):
        raise AxisTranslationError(
            f"tp axis {spec.name!r} expects Broadcast routing, got {type(spec.routing).__name__}"
        )
    if owner != "engine":
        raise RoutingOwnershipError(
            f"tp axis {spec.name!r} is realized intra-engine; l1_owner must be 'engine', got {owner!r}"
        )


def _validate_pp(spec: StrategySpec, owner: str) -> None:
    if not isinstance(spec.routing, PipelineMicrobatch):
        raise AxisTranslationError(
            f"pp axis {spec.name!r} expects PipelineMicrobatch routing, got {type(spec.routing).__name__}"
        )
    if owner != "engine":
        raise RoutingOwnershipError(
            f"pp axis {spec.name!r} is realized intra-engine; l1_owner must be 'engine', got {owner!r}"
        )


def _validate_ep(spec: StrategySpec, owner: str) -> None:
    # Dense EP: every rank still sees the whole batch, experts are sharded
    # across ranks inside the engine (sparse MoE all-to-all is a later stage).
    if not isinstance(spec.routing, Broadcast):
        raise AxisTranslationError(
            f"ep axis {spec.name!r} expects Broadcast routing (dense expert parallel), "
            f"got {type(spec.routing).__name__}"
        )
    if owner != "engine":
        raise RoutingOwnershipError(
            f"ep axis {spec.name!r} is realized intra-engine; l1_owner must be 'engine', got {owner!r}"
        )


def translate_strategy_stack(specs: Sequence[StrategySpec]) -> OmniParallelConfig:
    """Translate a spec stack into an ``OmniParallelConfig``.

    Supported kinds: dp (engine data parallel), tp, pp, (dense) ep, and
    stage_replica (omni replicas). Raises ``UnsupportedAxisKind`` for kinds not
    yet translatable, ``DuplicateAxisKind`` if a kind repeats,
    ``UnsupportedRouting`` for deferred (affinity) routing, and
    ``RoutingOwnershipError`` for an owner that is incompatible with the axis
    kind. The EP degree must equal tensor_parallel_size * data_parallel_size.
    """
    sizes: Dict[str, int] = {"tensor_parallel_size": 1, "data_parallel_size": 1, "pipeline_parallel_size": 1}
    owners: Dict[MeshAxisKind, str] = {}
    omni_lb_policy: Optional[str] = None
    stage_replica_size = 1
    enable_expert_parallel = False
    ep_size: Optional[int] = None

    for spec in specs:
        kind = spec.mesh_axis.kind
        if kind not in _SUPPORTED_KINDS:
            raise UnsupportedAxisKind(
                f"axis kind {kind!r} is not translatable yet (supported: {list(_SUPPORTED_KINDS)}); "
                "it is designed-for and lands in a later stage"
            )
        if kind in owners:
            raise DuplicateAxisKind(f"axis kind {kind!r} appears more than once in the spec stack")

        owner = _resolve_l1_owner(spec)
        if kind == "dp":
            _validate_dp(spec, owner)
        elif kind == "tp":
            _validate_tp(spec, owner)
        elif kind == "pp":
            _validate_pp(spec, owner)
        elif kind == "ep":
            _validate_ep(spec, owner)
            enable_expert_parallel = True
            ep_size = spec.mesh_axis.size
        elif kind == "stage_replica":
            omni_lb_policy = _stage_replica_lb_policy(spec, owner)
            stage_replica_size = spec.mesh_axis.size

        if kind in _AXIS_TO_ENGINE_FIELD:
            sizes[_AXIS_TO_ENGINE_FIELD[kind]] = spec.mesh_axis.size
        owners[kind] = owner

    if enable_expert_parallel:
        # Dense EP shards experts across exactly the TP*DP ranks, so the declared
        # EP degree must match that product (it is not its own world dimension,
        # and PP is excluded). An EP axis of size 1 is a degenerate no-op that
        # still sets the flag; it is only valid when TP*DP == 1.
        ep_ranks = sizes["tensor_parallel_size"] * sizes["data_parallel_size"]
        if ep_size != ep_ranks:
            raise AxisTranslationError(
                f"ep axis size {ep_size} must equal tensor_parallel_size*data_parallel_size "
                f"(={ep_ranks}); dense expert parallelism shards experts across exactly those ranks"
            )

    return OmniParallelConfig(
        tensor_parallel_size=sizes["tensor_parallel_size"],
        data_parallel_size=sizes["data_parallel_size"],
        pipeline_parallel_size=sizes["pipeline_parallel_size"],
        enable_expert_parallel=enable_expert_parallel,
        stage_replica_size=stage_replica_size,
        omni_lb_policy=omni_lb_policy,
        l1_owners=owners,
    )
