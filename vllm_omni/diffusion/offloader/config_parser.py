# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Canonical offload policy parser and validator (RFC #7204).

Provides a single ``parse_offload_policy`` entry point that consumes the new
compact JSON config, legacy list-shaped config, or individual boolean flags
and produces a single :class:`CanonicalOffloadPolicy` dataclass that the rest
of the offloader consumes.

Design goals (from RFC #7204 section 3):

* ``mode`` selects granularity: ``none`` / ``module`` / ``layer``.  It does
  **not** uniquely identify a backend class — the backend class is an
  internal implementation detail selected from ``components`` and
  ``weight_transfer`` choices.
* ``components`` is an object keyed by component name (``dit``,
  ``text_encoder``) where each value holds per-component options.  Omitted
  components default to ``{"dit":{}}`` for new-style configs.
* ``pin_memory`` is the single canonical pinning boolean.  The legacy
  ``pin_cpu_memory`` field is kept only as an input adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from vllm.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Enums and option values
# ---------------------------------------------------------------------------


class OffloadMode(str, Enum):
    NONE = "none"
    MODULE = "module"
    LAYER = "layer"


class WeightTransfer(str, Enum):
    RANK_LOCAL = "rank-local"
    ALLGATHER = "allgather"


VALID_COMPONENTS = frozenset({"dit", "text_encoder"})
VALID_TRANSFERS = frozenset({t.value for t in WeightTransfer})
VALID_MODES = frozenset({m.value for m in OffloadMode})


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ComponentOptions:
    """Per-component offload options."""

    weight_transfer: WeightTransfer = WeightTransfer.RANK_LOCAL
    resident_layers: int = 0  # DiT only

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"weight_transfer": self.weight_transfer.value}
        if self.resident_layers > 0:
            d["resident_layers"] = self.resident_layers
        return d


@dataclass
class CanonicalOffloadPolicy:
    """The single canonical offload policy consumed by the offloader.

    This is the output of :func:`parse_offload_policy` and the sole input to
    backend selection.  The three legacy booleans (``enable_cpu_offload``,
    ``enable_layerwise_offload``, ``enable_distributed_layerwise_offload``) are
    **not** present here — they are compatibility input adapters only.
    """

    mode: OffloadMode = OffloadMode.NONE
    components: dict[str, ComponentOptions] = field(default_factory=dict)
    pin_memory: bool = True

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self.mode is not OffloadMode.NONE

    @property
    def has_dit(self) -> bool:
        return "dit" in self.components

    @property
    def has_text_encoder(self) -> bool:
        return "text_encoder" in self.components

    def dit_transfer(self) -> WeightTransfer:
        return self.components.get("dit", ComponentOptions()).weight_transfer

    def dit_resident_layers(self) -> int:
        return self.components.get("dit", ComponentOptions()).resident_layers

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "components": {k: v.to_dict() for k, v in self.components.items()},
            "pin_memory": self.pin_memory,
        }


@dataclass
class HostWeightRuntimePolicy:
    """Canonical Host Weight Runtime policy (RFC #7204 section 4)."""

    mode: str = "disabled"  # disabled / preferred / required
    root: str | None = None
    registration_limit_gib: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "root": self.root,
            "registration_limit_gib": self.registration_limit_gib,
        }


# ---------------------------------------------------------------------------
# Expansion helpers for CLI shortcuts
# ---------------------------------------------------------------------------


SHORTCUT_EXPANSIONS: dict[str, dict[str, Any]] = {
    "none": {"mode": "none"},
    "module": {
        "mode": "module",
        "components": {"dit": {}},
        "pin_memory": True,
    },
    "layer": {
        "mode": "layer",
        "components": {
            "dit": {"weight_transfer": "rank-local", "resident_layers": 0},
        },
        "pin_memory": True,
    },
}


def expand_shortcut(shortcut: str) -> dict[str, Any]:
    """Expand a ``--diffusion-offload`` shortcut into a full policy dict."""
    if shortcut not in SHORTCUT_EXPANSIONS:
        raise ValueError(
            f"Unknown diffusion-offload shortcut {shortcut!r}. "
            f"Valid values: {sorted(SHORTCUT_EXPANSIONS)}"
        )
    # Return a fresh copy so callers can mutate safely.
    import copy

    return copy.deepcopy(SHORTCUT_EXPANSIONS[shortcut])


# ---------------------------------------------------------------------------
# Schema validation (RFC #7204 section 3 — every parameter)
# ---------------------------------------------------------------------------


def _validate_component_options(name: str, raw: Any, mode: OffloadMode) -> ComponentOptions:
    """Validate and coerce one component entry."""
    if raw is None:
        raw: dict[str, Any] = {}
    if not isinstance(raw, Mapping):
        raise TypeError(
            f"components.{name} must be an options object, got {type(raw).__name__}"
        )

    known_keys = {"weight_transfer", "resident_layers"}
    unknown = set(raw.keys()) - known_keys
    if unknown:
        raise ValueError(
            f"Unknown keys in components.{name}: {sorted(unknown)}. "
            f"Known keys: {sorted(known_keys)}"
        )

    transfer_raw = raw.get("weight_transfer", "rank-local")
    if transfer_raw not in VALID_TRANSFERS:
        raise ValueError(
            f"components.{name}.weight_transfer must be one of "
            f"{sorted(VALID_TRANSFERS)}, got {transfer_raw!r}"
        )
    transfer = WeightTransfer(transfer_raw)

    resident_layers = raw.get("resident_layers", 0)
    if isinstance(resident_layers, bool) or not isinstance(resident_layers, int):
        raise TypeError(
            f"components.{name}.resident_layers must be a non-negative integer, "
            f"got {type(resident_layers).__name__}"
        )
    if resident_layers < 0:
        raise ValueError(
            f"components.{name}.resident_layers must be >= 0, got {resident_layers}"
        )

    # Cross-field constraints
    if mode is OffloadMode.MODULE and transfer is WeightTransfer.ALLGATHER:
        raise ValueError(
            f"components.{name}.weight_transfer=allgather is only valid in "
            f"mode=layer, not mode=module"
        )
    if name != "dit" and resident_layers > 0:
        raise ValueError(
            f"resident_layers is only supported for components.dit, "
            f"not components.{name}"
        )
    if mode is OffloadMode.MODULE and resident_layers > 0:
        raise ValueError("resident_layers is only valid in mode=layer, not mode=module")

    return ComponentOptions(weight_transfer=transfer, resident_layers=resident_layers)


def _validate_components(raw_components: Any, mode: OffloadMode) -> dict[str, ComponentOptions]:
    """Validate the ``components`` field."""
    if raw_components is None:
        # Default for new-style active configs.
        if mode is OffloadMode.NONE:
            return {}
        return {"dit": ComponentOptions()}

    if not isinstance(raw_components, Mapping):
        raise TypeError(
            f"components must be an object keyed by component name, "
            f"got {type(raw_components).__name__}"
        )

    unknown = set(raw_components.keys()) - VALID_COMPONENTS
    if unknown:
        raise ValueError(
            f"Unknown component names: {sorted(unknown)}. "
            f"Known components: {sorted(VALID_COMPONENTS)}"
        )

    result: dict[str, ComponentOptions] = {}
    for name, options in raw_components.items():
        # An explicit empty dict means "use all defaults for this component" —
        # both shortcut expansion (`{"dit":{}}`) and hand-written configs can
        # legitimately use it.  We never raise here; all validation happens in
        # _validate_component_options (which accepts {} → defaults).
        result[name] = _validate_component_options(name, options, mode)

    return result


def _validate_offload_dict(raw: Mapping[str, Any]) -> CanonicalOffloadPolicy:
    """Validate and coerce a dict into a :class:`CanonicalOffloadPolicy`."""
    if not isinstance(raw, Mapping):
        raise TypeError(f"Offload policy must be a JSON object, got {type(raw).__name__}")

    known_keys = {"mode", "components", "pin_memory"}
    unknown = set(raw.keys()) - known_keys
    if unknown:
        raise ValueError(
            f"Unknown top-level keys in offload config: {sorted(unknown)}. "
            f"Known keys: {sorted(known_keys)}"
        )

    # mode is required
    if "mode" not in raw:
        raise ValueError("Offload policy requires a 'mode' key")
    mode_raw = raw["mode"]
    if mode_raw not in VALID_MODES:
        raise ValueError(
            f"mode must be one of {sorted(VALID_MODES)}, got {mode_raw!r}"
        )
    mode = OffloadMode(mode_raw)

    components = _validate_components(raw.get("components"), mode)

    pin_memory = raw.get("pin_memory", True)
    if not isinstance(pin_memory, bool):
        raise TypeError(f"pin_memory must be a boolean, got {type(pin_memory).__name__}")

    # mode=none must have no component or tuning fields
    if mode is OffloadMode.NONE:
        if components:
            raise ValueError(
                "mode=none does not accept a 'components' field; "
                "remove it or switch to mode=module/layer"
            )

    return CanonicalOffloadPolicy(mode=mode, components=components, pin_memory=pin_memory)


def _validate_hwr_dict(raw: Mapping[str, Any]) -> HostWeightRuntimePolicy:
    """Validate and coerce a dict into a :class:`HostWeightRuntimePolicy`."""
    if not isinstance(raw, Mapping):
        raise TypeError(f"HWR config must be a JSON object, got {type(raw).__name__}")

    known_keys = {"mode", "root", "registration_limit_gib"}
    unknown = set(raw.keys()) - known_keys
    if unknown:
        raise ValueError(
            f"Unknown top-level keys in HWR config: {sorted(unknown)}. "
            f"Known keys: {sorted(known_keys)}"
        )

    mode = raw.get("mode", "disabled")
    if mode not in ("disabled", "preferred", "required"):
        raise ValueError(
            f"hwr.mode must be disabled/preferred/required, got {mode!r}"
        )

    root = raw.get("root", None)
    if mode in ("preferred", "required"):
        if not root or not isinstance(root, str):
            raise ValueError(
                f"HWR mode={mode} requires a non-empty 'root' path string"
            )
    elif root is not None and mode == "disabled":
        logger.warning(
            "HWR root=%r supplied but mode=disabled; root is ignored", root
        )

    limit = raw.get("registration_limit_gib", 0.0)
    if isinstance(limit, bool) or not isinstance(limit, (int, float)):
        raise TypeError(
            f"HWR registration_limit_gib must be a finite non-negative number, "
            f"got {type(limit).__name__}"
        )
    if limit < 0:
        raise ValueError(f"HWR registration_limit_gib must be >= 0, got {limit}")
    if not (limit == float("inf") * 0 == float("nan")):  # isnan check
        pass
    import math
    if math.isnan(limit) or math.isinf(limit):
        raise ValueError(f"HWR registration_limit_gib must be finite, got {limit}")

    return HostWeightRuntimePolicy(
        mode=mode, root=root, registration_limit_gib=float(limit)
    )


# ---------------------------------------------------------------------------
# Joint validation (offload × HWR)
# ---------------------------------------------------------------------------


def validate_joint(
    offload: CanonicalOffloadPolicy,
    hwr: HostWeightRuntimePolicy,
) -> None:
    """Cross-validate offload and HWR policies (RFC #7204 section 4).

    Raises :class:`ValueError` for incompatible combinations.
    """
    if hwr.mode in ("preferred", "required"):
        # HWR disabled is fine with anything — just a no-op integration.
        if offload.mode is OffloadMode.NONE:
            raise ValueError(
                "HWR mode={hwr.mode} requires active offload "
                "(mode=module or mode=layer), not mode=none"
            )
        if offload.mode is OffloadMode.MODULE:
            raise ValueError(
                "HWR integration requires mode=layer offload, not mode=module"
            )
        if not offload.has_dit:
            raise ValueError(
                "HWR integration requires components.dit to be selected"
            )
        if offload.dit_transfer() is WeightTransfer.ALLGATHER:
            raise ValueError(
                "HWR integration does not support components.dit.weight_transfer=allgather; "
                "use rank-local"
            )
        # HWR+rank-local+layer is the canonical supported combination.

    # Pinning policy cross-check
    if hwr.registration_limit_gib > 0 and not offload.pin_memory:
        raise ValueError(
            "HWR registration_limit_gib > 0 requires pin_memory=true "
            "(registration only attempted under the pinned-memory policy)"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def parse_offload_policy(
    *,
    diffusion_offload: str | None = None,
    diffusion_offload_config: Mapping[str, Any] | str | None = None,
    # Legacy boolean inputs (compatibility adapters)
    enable_cpu_offload: bool = False,
    enable_layerwise_offload: bool = False,
    enable_distributed_layerwise_offload: bool = False,
    dlo_use_allgather: bool = True,
    dlo_resident_layers: int = 0,
    pin_cpu_memory: bool = True,
) -> CanonicalOffloadPolicy:
    """Parse and validate a canonical offload policy.

    Priority: shortcut > JSON config > legacy flags.  It is an error to
    supply both shortcut and JSON config simultaneously.

    Args:
        diffusion_offload: ``"none"`` / ``"module"`` / ``"layer"`` shortcut.
        diffusion_offload_config: JSON object or JSON string for the full policy.
        enable_cpu_offload: Legacy boolean.
        enable_layerwise_offload: Legacy boolean.
        enable_distributed_layerwise_offload: Legacy boolean.
        dlo_use_allgather: Legacy DLO allgather flag.
        dlo_resident_layers: Legacy DLO resident layers.
        pin_cpu_memory: Legacy pinning boolean (used only if pin_memory is
            not supplied in JSON config).

    Returns:
        Validated :class:`CanonicalOffloadPolicy`.
    """
    # --- 1. New-style inputs ---
    if diffusion_offload is not None and diffusion_offload_config is not None:
        raise ValueError(
            "--diffusion-offload and --diffusion-offload-config are mutually exclusive; "
            "use one or the other"
        )

    if diffusion_offload is not None:
        raw = expand_shortcut(diffusion_offload)
        return _validate_offload_dict(raw)

    if diffusion_offload_config is not None:
        if isinstance(diffusion_offload_config, str):
            import json as _json

            try:
                diffusion_offload_config = _json.loads(diffusion_offload_config)
            except _json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in diffusion-offload-config: {exc}"
                ) from exc
        return _validate_offload_dict(diffusion_offload_config)

    # --- 2. Legacy boolean inputs ---
    # Mutual exclusion mirroring OffloadConfig.from_od_config in base.py.
    if enable_distributed_layerwise_offload:
        mode = OffloadMode.LAYER
        if enable_layerwise_offload or enable_cpu_offload:
            logger.info(
                "Distributed layerwise offloading takes priority, "
                "disabling other legacy offloading strategies."
            )
    elif enable_layerwise_offload:
        mode = OffloadMode.LAYER
        if enable_cpu_offload:
            logger.info(
                "Both model-level and layer-wise offloading enabled. "
                "Layer-wise takes priority."
            )
    elif enable_cpu_offload:
        mode = OffloadMode.MODULE
    else:
        return CanonicalOffloadPolicy(mode=OffloadMode.NONE, pin_memory=pin_cpu_memory)

    # Build components from legacy flags.
    components: dict[str, ComponentOptions] = {
        "dit": ComponentOptions(
            weight_transfer=WeightTransfer.ALLGATHER
            if (
                enable_distributed_layerwise_offload
                and dlo_use_allgather
                and dlo_resident_layers == 0
            )
            else WeightTransfer.RANK_LOCAL,
            resident_layers=(
                dlo_resident_layers
                if enable_distributed_layerwise_offload
                else 0
            ),
        ),
    }

    return CanonicalOffloadPolicy(
        mode=mode,
        components=components,
        pin_memory=pin_cpu_memory,
    )


def parse_hwr_policy(
    hwr_config: Mapping[str, Any] | str | None = None,
    # Legacy flat flags (compatibility adapters)
    host_weight_runtime_mode: str = "disabled",
    host_weight_runtime_root: str | None = None,
    dlo_host_registration_limit_gib: float = 0.0,
) -> HostWeightRuntimePolicy:
    """Parse and validate a canonical HWR policy.

    Priority: JSON config > legacy flat flags.
    """
    if hwr_config is not None:
        if isinstance(hwr_config, str):
            import json as _json

            try:
                hwr_config = _json.loads(hwr_config)
            except _json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in hwr config: {exc}") from exc
        return _validate_hwr_dict(hwr_config)

    return HostWeightRuntimePolicy(
        mode=host_weight_runtime_mode,
        root=host_weight_runtime_root,
        registration_limit_gib=dlo_host_registration_limit_gib,
    )


__all__ = [
    "CanonicalOffloadPolicy",
    "ComponentOptions",
    "HostWeightRuntimePolicy",
    "OffloadMode",
    "WeightTransfer",
    "expand_shortcut",
    "parse_offload_policy",
    "parse_hwr_policy",
    "validate_joint",
    "VALID_COMPONENTS",
    "VALID_MODES",
    "VALID_TRANSFERS",
]
