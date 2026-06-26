# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Apply declarative parallel strategies onto merged stage configs.

This is the *override-after-merge* seam. ``merge_pipeline_deploy`` first fuses
the pipeline topology + deploy YAML into a list of ``StageConfig`` objects; this
module then overlays the engine sizing derived from a per-role
``StrategySpec`` stack onto those stages.

Design rules (one writer per axis):

* A strategy only writes the axes it actually declares. If a stack has no ``tp``
  axis, this module never touches ``tensor_parallel_size`` — it does not force a
  default of 1.
* **Conflict-on-explicit.** If a knob was already set explicitly by the deploy
  YAML and the strategy derives a different value, we raise rather than silently
  override. Equal values are a no-op; unset values are filled.
* **Pre-spawn device check.** When a stage declares ``devices``, the device
  count must be consistent with the engine world size (``tp * dp * pp``) and the
  replica count, so misconfigurations fail at config time instead of at spawn.

``omni_lb_policy`` is *not* a per-stage config knob: omni reads it once at
orchestrator construction (``AsyncOmniEngine``), not from stage configs. So a
stage_replica axis's derived policy is surfaced on :class:`StrategyApplyResult`
for the caller to pass to the orchestrator; it is never silently written into a
stage where it would be ignored.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NoReturn

from vllm.logger import init_logger

from vllm_omni.config.composable_parallel.spec import StrategySpec
from vllm_omni.config.composable_parallel.translator import (
    OmniParallelConfig,
    translate_strategy_stack,
)

logger = init_logger(__name__)

# Role key -> the StageConfig fields each declared axis writes.
_ENGINE_FIELD_BY_KIND: dict[str, str] = {
    "tp": "tensor_parallel_size",
    "dp": "data_parallel_size",
    "pp": "pipeline_parallel_size",
}

RoleKey = str


class StrategyApplyError(ValueError):
    """Error for applying a strategy onto stage configs.

    A single error type (rather than per-cause subclasses) keeps the public
    surface small and consistent with the rest of the codebase; the specific
    cause — an unmatched role, a conflict with an explicit deploy value, or a
    device-count mismatch — is in the message and is logged before the raise so
    it stays visible even when the type is unavailable to a caller (e.g. across a
    server boundary).
    """


def _fail(msg: str) -> NoReturn:
    """Log and raise :class:`StrategyApplyError`."""
    logger.error("[composable_parallel] %s", msg)
    raise StrategyApplyError(msg)


@dataclass
class StrategyApplyResult:
    """Outcome of applying strategies to a stage list.

    ``stages`` is the same list passed in (mutated in place and returned for
    convenience). ``omni_lb_policy`` is the pipeline-wide load-balancer policy
    derived from any stage_replica axes (``None`` if none declared one); the
    caller wires it into the orchestrator, since omni does not read it from
    stage configs.

    ``per_role_config`` is keyed by the role key the caller supplied — a
    ``model_stage`` name (str). ``per_stage_config`` is the same information
    keyed by the resolved integer ``stage_id`` — handy for re-validating a
    stage after later layers (e.g. CLI overrides) have run.
    """

    stages: list[Any]
    omni_lb_policy: str | None = None
    per_role_config: dict[RoleKey, OmniParallelConfig] = field(default_factory=dict)
    per_stage_config: dict[int, OmniParallelConfig] = field(default_factory=dict)


def _resolve_stage(stages: Sequence[Any], key: RoleKey) -> Any:
    """Find the single stage matching ``key`` by ``model_stage`` name.

    Role keys are ``model_stage`` names (str) — the same human-readable names
    that strategy.yaml authors use to refer to a stage (e.g. ``"thinker"``).
    """
    matches = [s for s in stages if getattr(s, "model_stage", None) == key]
    descriptor = f"model_stage={key!r}"

    if not matches:
        available = ", ".join(f"{getattr(s, 'model_stage', '?')!r}(id={getattr(s, 'stage_id', '?')})" for s in stages)
        _fail(f"strategy role {descriptor} did not match any stage; available stages: {available}")
    if len(matches) > 1:
        _fail(f"strategy role {descriptor} is ambiguous; it matched {len(matches)} stages")
    return matches[0]


def _set_explicit(engine_args: dict[str, Any], field_name: str, value: Any, *, role: RoleKey) -> None:
    """Write ``field_name`` with conflict-on-explicit semantics.

    ``engine_args`` is the raw mapping loaded straight from the deploy YAML
    (before the engine-args dataclass is constructed), so a key is *present* only
    when the YAML set it explicitly. We therefore key off membership rather than
    ``.get``: an explicit ``None`` in the YAML is a real, present value and must
    not be conflated with a key miss (which ``.get`` would report as ``None``).
    """
    if field_name in engine_args and engine_args[field_name] != value:
        existing = engine_args[field_name]
        _fail(
            f"role {role!r}: strategy derives {field_name}={value} but the deploy config already "
            f"set {field_name}={existing}. Remove one of them (the strategy is the single writer "
            "for the axes it declares)."
        )
    engine_args[field_name] = value


def _set_num_replicas(runtime: dict[str, Any], value: int, *, role: RoleKey) -> None:
    """Write ``num_replicas`` with conflict-on-explicit (treating 1 as unset)."""
    existing = runtime.get("num_replicas", 1)
    if existing not in (1, value):
        _fail(
            f"role {role!r}: strategy derives num_replicas={value} but the deploy config already "
            f"set num_replicas={existing}. Remove one of them."
        )
    runtime["num_replicas"] = value


def _parse_device_count(devices: Any) -> int | None:
    """Return the number of device ids in a ``devices`` value, or None."""
    if devices is None:
        return None
    if isinstance(devices, (list, tuple)):
        return len([d for d in devices])
    text = str(devices).strip()
    if not text:
        return None
    return len([d for d in text.split(",") if d.strip()])


def check_device_layout(
    devices: Any,
    *,
    tensor_parallel_size: int,
    data_parallel_size: int,
    pipeline_parallel_size: int,
    num_replicas: int,
    role: RoleKey,
) -> None:
    """Validate a stage's device count against its world size (× replicas in pool mode).

    Raises :class:`StrategyApplyError` when the declared ``devices`` count is
    neither the per-replica world size nor the full ``num_replicas`` pool.
    Exposed publicly so the resolved (post-CLI) layout can be re-checked after
    later override layers, not just the strategy-derived snapshot.
    """
    count = _parse_device_count(devices)
    if count is None:
        return
    world = int(tensor_parallel_size) * int(data_parallel_size) * int(pipeline_parallel_size)
    replicas = int(num_replicas or 1)
    # Two valid shapes: a single per-replica template (== world) or the full
    # pool across replicas (== replicas * world). This mirrors omni's own
    # per-replica device splitter.
    valid = {world, replicas * world}
    if count not in valid:
        _fail(
            f"role {role!r}: declared {count} device id(s) but the strategy world size is {world} "
            f"(tp={tensor_parallel_size} * dp={data_parallel_size} * pp={pipeline_parallel_size}). "
            f"Provide either {world} (per-replica template) or {replicas * world} "
            f"(num_replicas={replicas} pool)."
        )


def _check_devices(runtime: dict[str, Any], cfg: OmniParallelConfig, *, role: RoleKey) -> None:
    """Pre-spawn check on the strategy-derived snapshot (see :func:`check_device_layout`)."""
    check_device_layout(
        runtime.get("devices"),
        tensor_parallel_size=cfg.tensor_parallel_size,
        data_parallel_size=cfg.data_parallel_size,
        pipeline_parallel_size=cfg.pipeline_parallel_size,
        num_replicas=int(runtime.get("num_replicas", 1) or 1),
        role=role,
    )


def _apply_to_stage(stage: Any, cfg: OmniParallelConfig, *, role: RoleKey) -> None:
    engine_args = stage.yaml_engine_args
    runtime = stage.yaml_runtime

    declared = set(cfg.l1_owners.keys())
    for kind, field_name in _ENGINE_FIELD_BY_KIND.items():
        if kind in declared:
            _set_explicit(engine_args, field_name, getattr(cfg, field_name), role=role)
    if "ep" in declared and cfg.enable_expert_parallel:
        _set_explicit(engine_args, "enable_expert_parallel", True, role=role)
    if "stage_replica" in declared:
        _set_num_replicas(runtime, cfg.stage_replica_size, role=role)

    _check_devices(runtime, cfg, role=role)


def apply_strategy_specs(
    stages: list[Any],
    strategy_specs: Mapping[RoleKey, Sequence[StrategySpec]],
) -> StrategyApplyResult:
    """Overlay per-role strategy specs onto a merged stage list.

    Args:
        stages: the ``list[StageConfig]`` returned by ``merge_pipeline_deploy``.
        strategy_specs: maps a role (a ``model_stage`` name, e.g. ``"thinker"``)
            to that stage's stack of ``StrategySpec`` (one per declared mesh
            axis). Role keys are ``model_stage`` names (str).

    Returns:
        A :class:`StrategyApplyResult` with the mutated stages and the derived
        pipeline-wide ``omni_lb_policy`` (if any stage_replica axis set one).

    Raises:
        StrategyApplyError: a role matched zero or multiple stages, a derived
            value conflicts with an explicit deploy value (including two roles
            deriving different ``omni_lb_policy`` values), or a stage's device
            count is inconsistent with its world size.
        AxisTranslationError: the spec stack is invalid/unsupported.
        NotImplementedError: the spec stack requests routing not built yet.
    """
    result = StrategyApplyResult(stages=stages)
    lb_policy: str | None = None
    lb_owner: RoleKey | None = None

    for key, specs in strategy_specs.items():
        stage = _resolve_stage(stages, key)
        cfg = translate_strategy_stack(specs)
        _apply_to_stage(stage, cfg, role=key)
        result.per_role_config[key] = cfg
        stage_id = getattr(stage, "stage_id", None)
        if stage_id is not None:
            result.per_stage_config[int(stage_id)] = cfg

        if cfg.stage_replica_size > 1 and cfg.omni_lb_policy is not None:
            if lb_policy is not None and lb_policy != cfg.omni_lb_policy:
                _fail(
                    f"role {key!r} derives omni_lb_policy={cfg.omni_lb_policy!r} but role "
                    f"{lb_owner!r} already derived {lb_policy!r}. The load-balancer policy is "
                    "pipeline-wide; only one value is allowed."
                )
            lb_policy = cfg.omni_lb_policy
            lb_owner = key

    result.omni_lb_policy = lb_policy
    return result
