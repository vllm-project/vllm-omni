# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Public diffusion CPU-offload configuration helpers."""

from __future__ import annotations

from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

DIT_COMPONENT = "dit"
TEXT_ENCODER_COMPONENT = "text_encoder"
OFFLOAD_COMPONENTS = frozenset({DIT_COMPONENT, TEXT_ENCODER_COMPONENT})
DEFAULT_OFFLOAD_COMPONENTS = frozenset({DIT_COMPONENT})

_KeyT = TypeVar("_KeyT")
_ValueT = TypeVar("_ValueT")


class _FrozenMapping(Mapping[_KeyT, _ValueT]):
    """Small immutable mapping that remains safe to pickle between workers."""

    def __init__(self, value: Mapping[_KeyT, _ValueT]) -> None:
        self._items = tuple(value.items())

    def __getitem__(self, key: _KeyT) -> _ValueT:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[_KeyT]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __reduce__(self):
        return type(self), (dict(self._items),)


class OffloadMode(str, Enum):
    """User-facing offload granularity."""

    MODULE = "module"
    LAYER = "layer"


class OffloadStrategy(str, Enum):
    """Resolved internal backend strategy."""

    NONE = "none"
    MODEL_LEVEL = "model_level"
    LAYER_WISE = "layer_wise"
    DISTRIBUTED_LAYER_WISE = "distributed_layer_wise"


class DLOTransfer(str, Enum):
    """How one component's next block reaches the device."""

    ALLGATHER = "allgather"
    RANK_LOCAL = "rank-local"


@dataclass(frozen=True)
class LayerOffloadOptions:
    """Layer-mode settings for one selected diffusion component."""

    weight_transfer: DLOTransfer | None = None
    resident_layers: int = 0


@dataclass(frozen=True)
class ParsedDiffusionOffloadConfig:
    """Validated internal form of the raw public offload mapping."""

    mode: OffloadMode
    components: frozenset[str]
    layer_options: Mapping[str, LayerOffloadOptions]
    pin_memory: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_options", _FrozenMapping(self.layer_options))


@dataclass(frozen=True)
class ResolvedOffload:
    """Canonical offload policy derived once from public or legacy inputs."""

    strategy: OffloadStrategy
    public: ParsedDiffusionOffloadConfig | None
    components: frozenset[str]
    transfers: Mapping[str, DLOTransfer]
    pin_memory: bool
    resident_layers: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "transfers", _FrozenMapping(self.transfers))

    def offloads(self, component: str) -> bool:
        return component in self.components

    def uses_allgather(self, component: str = DIT_COMPONENT) -> bool:
        try:
            return self.transfers[component] is DLOTransfer.ALLGATHER
        except KeyError as exc:
            raise ValueError(f"Unknown offload component {component!r}") from exc

    @property
    def any_allgather(self) -> bool:
        return self.strategy is OffloadStrategy.DISTRIBUTED_LAYER_WISE and any(
            self.uses_allgather(component) for component in self.components
        )


_LEGACY_STRATEGY_FIELDS = {
    "enable_cpu_offload": OffloadStrategy.MODEL_LEVEL,
    "enable_layerwise_offload": OffloadStrategy.LAYER_WISE,
    "enable_distributed_layerwise_offload": OffloadStrategy.DISTRIBUTED_LAYER_WISE,
}
_LEGACY_STRATEGY_PRIORITY = (
    OffloadStrategy.DISTRIBUTED_LAYER_WISE,
    OffloadStrategy.LAYER_WISE,
    OffloadStrategy.MODEL_LEVEL,
)


_EnumT = TypeVar("_EnumT", bound=Enum)


def _parse_enum(value: Any, enum_type: type[_EnumT], label: str) -> _EnumT:
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string, got {type(value).__name__}")
    try:
        return enum_type(value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in enum_type)
        raise ValueError(f"Unknown {label} {value!r}; choose from: {choices}") from exc


def _validate_string_keys(value: Mapping[Any, Any], label: str) -> None:
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} keys must be strings")


def _parse_layer_options(component: str, value: Any) -> LayerOffloadOptions:
    if not isinstance(value, Mapping):
        raise TypeError(
            f"diffusion_offload_config.layer_options[{component!r}] must be a mapping, got {type(value).__name__}"
        )
    _validate_string_keys(value, f"diffusion_offload_config.layer_options[{component!r}]")
    unknown = sorted(set(value) - {"weight_transfer", "resident_layers"})
    if unknown:
        raise ValueError(
            f"Unknown diffusion offload setting(s) for {component}: {', '.join(unknown)}; "
            "choose from: weight_transfer, resident_layers"
        )
    weight_transfer = (
        _parse_enum(value["weight_transfer"], DLOTransfer, "offload transfer") if "weight_transfer" in value else None
    )
    resident_layers = value.get("resident_layers", 0)

    if type(resident_layers) is not int or resident_layers < 0:
        raise ValueError(f"resident_layers for {component} must be a non-negative integer")
    parsed = LayerOffloadOptions(
        weight_transfer=weight_transfer,
        resident_layers=resident_layers,
    )
    if component != DIT_COMPONENT and parsed.resident_layers:
        raise ValueError("resident_layers currently supports only the 'dit' component")
    return parsed


def parse_diffusion_offload_config(value: Any) -> ParsedDiffusionOffloadConfig | None:
    """Validate the compact ``diffusion_offload_config`` public API."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"diffusion_offload_config must be a mapping, got {type(value).__name__}")

    _validate_string_keys(value, "diffusion_offload_config")
    unknown = sorted(set(value) - {"mode", "components", "layer_options", "pin_memory"})
    if unknown:
        raise ValueError(
            f"Unknown diffusion_offload_config field(s): {', '.join(unknown)}; "
            "choose from: mode, components, layer_options, pin_memory"
        )
    if "mode" not in value:
        raise ValueError("diffusion_offload_config requires 'mode'")
    if "components" not in value:
        raise ValueError("diffusion_offload_config requires 'components'")

    mode = _parse_enum(value["mode"], OffloadMode, "diffusion offload mode")
    components = parse_offload_components(value["components"])

    raw_layer_options = value.get("layer_options", {})
    if not isinstance(raw_layer_options, Mapping):
        raise TypeError("diffusion_offload_config.layer_options must be a mapping")
    _validate_string_keys(raw_layer_options, "diffusion_offload_config.layer_options")
    unselected_options = sorted(set(raw_layer_options) - components)
    if unselected_options:
        raise ValueError(
            "diffusion_offload_config.layer_options requires selecting the same component(s): "
            + ", ".join(unselected_options)
        )
    layer_options = {
        component: _parse_layer_options(component, raw_layer_options.get(component, {})) for component in components
    }

    pin_memory = value.get("pin_memory")
    if pin_memory is not None and type(pin_memory) is not bool:
        raise TypeError("diffusion_offload_config.pin_memory must be a bool")

    if mode is OffloadMode.MODULE and raw_layer_options:
        raise ValueError(
            "diffusion_offload_config.layer_options requires mode='layer'; "
            f"configured for: {', '.join(sorted(raw_layer_options))}"
        )
    dit = layer_options.get(DIT_COMPONENT)
    if dit is not None and dit.resident_layers and dit.weight_transfer is DLOTransfer.ALLGATHER:
        raise ValueError("resident_layers requires dit.weight_transfer='rank-local'")

    return ParsedDiffusionOffloadConfig(
        mode=mode,
        components=components,
        layer_options=layer_options,
        pin_memory=pin_memory,
    )


def _legacy_strategy(config: Any) -> OffloadStrategy:
    enabled = {strategy for field, strategy in _LEGACY_STRATEGY_FIELDS.items() if bool(getattr(config, field, False))}
    return next((strategy for strategy in _LEGACY_STRATEGY_PRIORITY if strategy in enabled), OffloadStrategy.NONE)


def _public_strategy(public: ParsedDiffusionOffloadConfig) -> OffloadStrategy:
    if public.mode is OffloadMode.MODULE:
        return OffloadStrategy.MODEL_LEVEL

    needs_distributed_backend = any(
        settings.weight_transfer is DLOTransfer.ALLGATHER or settings.resident_layers
        for settings in public.layer_options.values()
    )
    return OffloadStrategy.DISTRIBUTED_LAYER_WISE if needs_distributed_backend else OffloadStrategy.LAYER_WISE


def _validate_legacy_layer_options(config: Any, public: ParsedDiffusionOffloadConfig | None) -> None:
    """Reject ambiguous or invalid compatibility DLO tuning before loading."""
    resident_layers = getattr(config, "dlo_resident_layers", 0)
    if type(resident_layers) is not int or resident_layers < 0:
        raise ValueError(f"dlo_resident_layers must be a non-negative integer, got {resident_layers!r}")

    if public is not None:
        if getattr(config, "host_weight_runtime_mode", "disabled") != "disabled":
            raise ValueError("diffusion_offload_config cannot be combined with Host Weight Runtime")
        conflicting_fields = []
        if getattr(config, "dlo_use_allgather", True) is not True:
            conflicting_fields.append("dlo_use_allgather")
        if resident_layers:
            conflicting_fields.append("dlo_resident_layers")
        if float(getattr(config, "dlo_host_registration_limit_gib", 0.0)) > 0:
            conflicting_fields.append("dlo_host_registration_limit_gib")
        if conflicting_fields:
            raise ValueError(
                "diffusion_offload_config cannot be combined with legacy DLO option(s): "
                + ", ".join(conflicting_fields)
            )
        return

    if resident_layers and bool(getattr(config, "dlo_use_allgather", True)):
        raise ValueError(
            "dlo_resident_layers requires the DiT DLO transfer to be rank-local; set dlo_use_allgather=False"
        )


def resolve_offload(config: Any) -> ResolvedOffload:
    """Resolve and cache the complete offload policy on its owning config."""
    cached = getattr(config, "_resolved_diffusion_offload", None)
    if isinstance(cached, ResolvedOffload):
        return cached

    public = parse_diffusion_offload_config(getattr(config, "diffusion_offload_config", None))
    _validate_legacy_layer_options(config, public)
    legacy = _legacy_strategy(config)
    if public is None:
        transfers = {
            DIT_COMPONENT: (
                DLOTransfer.ALLGATHER if bool(getattr(config, "dlo_use_allgather", True)) else DLOTransfer.RANK_LOCAL
            ),
            TEXT_ENCODER_COMPONENT: DLOTransfer.RANK_LOCAL,
        }
        resolved = ResolvedOffload(
            strategy=legacy,
            public=None,
            components=DEFAULT_OFFLOAD_COMPONENTS,
            transfers=transfers,
            pin_memory=bool(getattr(config, "pin_cpu_memory", True)),
            resident_layers=int(getattr(config, "dlo_resident_layers", 0)),
        )
    else:
        strategy = _public_strategy(public)
        if legacy is not OffloadStrategy.NONE and legacy is not strategy:
            raise ValueError("diffusion_offload_config cannot be combined with legacy enable_*_offload flags")
        transfers = {component: DLOTransfer.RANK_LOCAL for component in OFFLOAD_COMPONENTS}
        for component, options in public.layer_options.items():
            transfers[component] = options.weight_transfer or DLOTransfer.RANK_LOCAL
        dit_options = public.layer_options.get(DIT_COMPONENT)
        resolved = ResolvedOffload(
            strategy=strategy,
            public=public,
            components=public.components,
            transfers=transfers,
            pin_memory=(
                public.pin_memory if public.pin_memory is not None else bool(getattr(config, "pin_cpu_memory", True))
            ),
            resident_layers=0 if dit_options is None else dit_options.resident_layers,
        )

    parallel_config = getattr(config, "parallel_config", None)
    data_parallel_size = int(getattr(parallel_config, "data_parallel_size", 1))
    cache_backend = getattr(config, "cache_backend", None)
    if data_parallel_size > 1 and resolved.any_allgather and cache_backend not in (None, "none"):
        raise ValueError(
            "Diffusion cache acceleration cannot be combined with DLO AllGather "
            "across data-parallel ranks because rank-local cache decisions can skip "
            "different weight collectives. Disable cache_backend or use rank-local transfer."
        )
    if (
        data_parallel_size > 1
        and bool(getattr(config, "enable_prompt_embed_cache", False))
        and resolved.offloads(TEXT_ENCODER_COMPONENT)
        and resolved.uses_allgather(TEXT_ENCODER_COMPONENT)
    ):
        raise ValueError(
            "Prompt embedding cache cannot be combined with text_encoder AllGather "
            "across data-parallel ranks: rank-local cache hits would skip different "
            "encoder collectives. Use text_encoder weight_transfer='rank-local' or "
            "disable enable_prompt_embed_cache."
        )

    # Actual runtime configs are immutable after construction. Keeping the
    # derived object off the public dataclass fields also keeps projections and
    # serialization transport-safe while making hot-path accessors constant-time.
    try:
        setattr(config, "_resolved_diffusion_offload", resolved)
    except (AttributeError, TypeError):
        pass
    return resolved


def resolve_offload_strategy(config: Any) -> OffloadStrategy:
    """Resolve the compact config or compatibility boolean entry points."""
    return resolve_offload(config).strategy


def materialize_legacy_offload_flags(config: Any) -> OffloadStrategy:
    """Keep existing strategy readers working after resolving the compact API."""
    resolved = resolve_offload(config)
    for field, field_strategy in _LEGACY_STRATEGY_FIELDS.items():
        setattr(config, field, resolved.strategy is field_strategy)
    setattr(config, "pin_cpu_memory", resolved.pin_memory)
    return resolved.strategy


def parse_offload_components(value: Collection[str]) -> frozenset[str]:
    """Validate an internal component collection."""
    if isinstance(value, (str, Mapping)) or not isinstance(value, Collection):
        raise TypeError("diffusion_offload_config.components must be a non-empty list of component names")
    if any(not isinstance(item, str) for item in value):
        raise TypeError("offload component entries must be strings")
    components = list(value)
    if not components:
        raise ValueError("offload components must not be empty")
    unknown = sorted(set(components) - OFFLOAD_COMPONENTS)
    if unknown:
        choices = ", ".join(sorted(OFFLOAD_COMPONENTS))
        raise ValueError(f"Unknown diffusion offload component(s): {', '.join(unknown)}; choose from: {choices}")
    return frozenset(components)


def parse_dlo_transfer(value: Mapping[str, str | DLOTransfer]) -> dict[str, DLOTransfer]:
    """Resolve an internal per-component transfer mapping."""
    if not isinstance(value, Mapping):
        raise TypeError(f"offload transfers must be a mapping, got {type(value).__name__}")
    _validate_string_keys(value, "offload transfers")
    missing = sorted(OFFLOAD_COMPONENTS - set(value))
    if missing:
        raise ValueError("offload transfers require every component; missing: " + ", ".join(missing))
    resolved: dict[str, DLOTransfer] = {}
    for component, raw_transfer in value.items():
        if component not in OFFLOAD_COMPONENTS:
            choices = ", ".join(sorted(OFFLOAD_COMPONENTS))
            raise ValueError(f"Unknown offload transfer component {component!r}; choose from: {choices}")
        resolved[component] = _parse_enum(raw_transfer, DLOTransfer, "offload transfer")
    return resolved


def component_uses_allgather(config: Any, component: str = DIT_COMPONENT) -> bool:
    """Return whether one selected component uses AllGather transport."""
    resolved = resolve_offload(config)
    if not resolved.offloads(component):
        raise ValueError(f"Offload component {component!r} is not selected")
    return resolved.uses_allgather(component)


def selected_offload_components(config: Any) -> frozenset[str]:
    """Resolve selected components while preserving legacy topology defaults."""
    return resolve_offload(config).components


def should_offload_component(config: Any, component: str) -> bool:
    """Return whether an active layer policy selects ``component``."""
    if component not in OFFLOAD_COMPONENTS:
        raise ValueError(f"Unknown offload component: {component}")
    resolved = resolve_offload(config)
    if resolved.strategy not in {
        OffloadStrategy.LAYER_WISE,
        OffloadStrategy.DISTRIBUTED_LAYER_WISE,
    }:
        return False
    return resolved.offloads(component)


def any_selected_component_uses_allgather(config: Any) -> bool:
    """Return whether an enabled layer backend requires weight collectives."""
    return resolve_offload(config).any_allgather
