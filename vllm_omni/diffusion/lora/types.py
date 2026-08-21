# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

from vllm.lora.request import LoRARequest

from vllm_omni.lora.utils import stable_lora_int_id

LoRARequestInput: TypeAlias = LoRARequest | Sequence[LoRARequest] | None
LoRAScaleInput: TypeAlias = float | Sequence[float]


@dataclass(frozen=True)
class WeightedLoRA:
    """One adapter and its request/deployment-level mixing coefficient."""

    request: LoRARequest
    scale: float = 1.0

    @property
    def adapter_id(self) -> int:
        return self.request.lora_int_id


LoRAComposition: TypeAlias = tuple[WeightedLoRA, ...]
LoRARegistry: TypeAlias = tuple[LoRARequest, ...]
LoRACompositionKey: TypeAlias = tuple[tuple[int, float], ...]
LoRABatchAdapterKey: TypeAlias = int | tuple[int, ...] | None
LoRABatchScaleKey: TypeAlias = float | tuple[float, ...]

# Upstream LoRARequest requires a non-empty path.  ID-only API selectors use
# this internal placeholder until DiffusionEngine replaces them with the
# deployment-owned request before scheduler admission.
_REGISTERED_LORA_PATH_PREFIX = "vllm-omni://registered-lora/"
_STARTUP_LORA_FIELDS = frozenset(
    {
        "path",
        "lora_path",
        "local_path",
        "name",
        "lora_name",
        "int_id",
        "lora_int_id",
        "scale",
        "lora_scale",
    }
)


def _get_single_alias(
    value: Mapping[str, Any],
    fields: tuple[str, ...],
    label: str,
    default: Any = None,
) -> Any:
    present = [field for field in fields if value.get(field) is not None]
    if len(present) > 1:
        raise ValueError(f"LoRA adapter specification provides multiple {label} fields: {present}")
    return value[present[0]] if present else default


def registered_lora_request(name: str) -> LoRARequest:
    """Build a name-only request reference for deployment-time resolution."""

    name = name.strip()
    if not name:
        raise ValueError("Registered LoRA name must not be empty")
    adapter_id = stable_lora_int_id(name)

    return LoRARequest(
        lora_name=name,
        lora_int_id=adapter_id,
        lora_path=f"{_REGISTERED_LORA_PATH_PREFIX}{adapter_id}",
    )


def is_registered_lora_request(request: LoRARequest) -> bool:
    return request.lora_path == f"{_REGISTERED_LORA_PATH_PREFIX}{request.lora_int_id}"


def normalize_lora_composition(
    requests: LoRARequestInput,
    scales: LoRAScaleInput = 1.0,
) -> LoRAComposition:
    """Return a deterministic, validated composition.

    Duplicate adapter IDs are combined by adding their scales. Zero-scale
    entries are removed, and the result is sorted by adapter ID so every
    distributed rank binds the same concatenated low-rank layout.
    """

    if requests is None:
        return ()
    request_items = (requests,) if isinstance(requests, LoRARequest) else tuple(requests)
    if isinstance(scales, (int, float)):
        scale_items = (float(scales),) * len(request_items)
    else:
        scale_items = tuple(float(scale) for scale in scales)
        if len(scale_items) != len(request_items):
            raise ValueError(
                f"LoRA requests and scales must have the same length: {len(request_items)} != {len(scale_items)}"
            )

    combined: dict[int, WeightedLoRA] = {}
    for request, scale in zip(request_items, scale_items, strict=True):
        if not isinstance(request, LoRARequest):
            raise TypeError(f"Expected LoRARequest, got {type(request)!r}")
        if not math.isfinite(scale):
            raise ValueError(f"LoRA scale must be finite, got {scale!r}")

        previous = combined.get(request.lora_int_id)
        if previous is not None:
            if previous.request.lora_path != request.lora_path:
                raise ValueError(
                    f"LoRA adapter ID {request.lora_int_id} refers to both "
                    f"{previous.request.lora_path!r} and {request.lora_path!r}"
                )
            scale += previous.scale
        combined[request.lora_int_id] = WeightedLoRA(request=request, scale=scale)

    return tuple(adapter for _, adapter in sorted(combined.items()) if adapter.scale != 0.0)


def lora_composition_key(composition: LoRAComposition) -> LoRACompositionKey:
    return tuple((adapter.adapter_id, adapter.scale) for adapter in composition)


def split_lora_composition(
    composition: LoRAComposition,
) -> tuple[LoRARequest | tuple[LoRARequest, ...] | None, float | tuple[float, ...]]:
    """Project a canonical composition back to sampling-parameter fields."""

    if not composition:
        return None, 1.0
    if len(composition) == 1:
        return composition[0].request, composition[0].scale
    return tuple(adapter.request for adapter in composition), tuple(adapter.scale for adapter in composition)


def lora_batch_key_fields(
    requests: LoRARequestInput,
    scales: LoRAScaleInput = 1.0,
) -> tuple[LoRABatchAdapterKey, LoRABatchScaleKey]:
    """Return canonical adapter identity and scale fields for batching."""

    composition = normalize_lora_composition(requests, scales)
    if not composition:
        return None, 1.0
    adapter_ids: LoRABatchAdapterKey = (
        composition[0].adapter_id if len(composition) == 1 else tuple(adapter.adapter_id for adapter in composition)
    )
    _, canonical_scales = split_lora_composition(composition)
    return adapter_ids, canonical_scales


def parse_lora_adapter_spec(value: str | Mapping[str, Any]) -> WeightedLoRA:
    """Parse ``PATH``, ``PATH=SCALE``, or a mapping startup specification."""

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("LoRA adapter specification must not be empty")
        if stripped.startswith("{"):
            parsed = json.loads(stripped)
            if not isinstance(parsed, dict):
                raise ValueError("LoRA JSON specification must be an object")
            value = parsed
        else:
            path = stripped
            scale = 1.0
            maybe_path, separator, maybe_scale = stripped.rpartition("=")
            if separator:
                try:
                    scale = float(maybe_scale)
                except ValueError:
                    pass
                else:
                    path = maybe_path
            value = {"path": path, "scale": scale}

    unknown_fields = set(value) - _STARTUP_LORA_FIELDS
    if unknown_fields:
        fields = ", ".join(sorted(map(repr, unknown_fields)))
        raise ValueError(f"LoRA adapter specification contains unknown field(s): {fields}")

    path_value = _get_single_alias(value, ("path", "lora_path", "local_path"), "path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("LoRA adapter specification requires a non-empty path")
    if path_value.startswith(_REGISTERED_LORA_PATH_PREFIX):
        raise ValueError(f"LoRA adapter path uses the reserved prefix {_REGISTERED_LORA_PATH_PREFIX!r}")
    name_value = _get_single_alias(value, ("name", "lora_name"), "name", Path(path_value).stem)
    if not isinstance(name_value, str) or not name_value.strip():
        raise ValueError("LoRA adapter specification requires a non-empty name")
    name_value = name_value.strip()
    int_id_value = _get_single_alias(
        value,
        ("int_id", "lora_int_id"),
        "integer ID",
        stable_lora_int_id(path_value),
    )
    scale_value = float(_get_single_alias(value, ("scale", "lora_scale"), "scale", 1.0))
    composition = normalize_lora_composition(
        LoRARequest(
            lora_name=str(name_value),
            lora_int_id=int(int_id_value),
            lora_path=path_value,
        ),
        scale_value,
    )
    if not composition:
        raise ValueError("Startup LoRA adapter scale must be non-zero")
    return composition[0]


def parse_lora_adapter_specs(values: Sequence[str | Mapping[str, Any]] | None) -> LoRAComposition:
    if not values:
        return ()
    adapters = tuple(parse_lora_adapter_spec(value) for value in values)
    return normalize_lora_composition(
        tuple(adapter.request for adapter in adapters),
        tuple(adapter.scale for adapter in adapters),
    )


def parse_lora_registration_specs(values: Sequence[str | Mapping[str, Any]] | None) -> LoRARegistry:
    """Parse startup registrations, which deliberately have no default scale."""

    if not values:
        return ()
    for value in values:
        registration = value
        if isinstance(value, str) and value.strip().startswith("{"):
            registration = json.loads(value)
        if isinstance(registration, Mapping) and any(
            registration.get(field) is not None for field in ("int_id", "lora_int_id")
        ):
            raise ValueError(
                "Dynamic LoRA registration does not accept int_id; the request-facing name is the adapter identity."
            )

    adapters = tuple(parse_lora_adapter_spec(value) for value in values)
    weighted = [adapter for adapter in adapters if adapter.scale != 1.0]
    if weighted:
        paths = ", ".join(repr(adapter.request.lora_path) for adapter in weighted)
        raise ValueError(
            f"Dynamic LoRA registration does not accept startup scales: {paths}. Set each adapter scale in the request."
        )

    registry: dict[str, LoRARequest] = {}
    internal_ids: dict[int, str] = {}
    for adapter in adapters:
        request = adapter.request
        name = request.lora_name.strip()
        if not name:
            raise ValueError("Dynamic LoRA registration requires a non-empty name")
        previous = registry.get(name)
        if previous is not None:
            raise ValueError(
                f"Dynamic LoRA adapter name {name!r} is registered more than once: "
                f"{previous.lora_path!r} and {request.lora_path!r}"
            )
        internal_id = stable_lora_int_id(name)
        colliding_name = internal_ids.get(internal_id)
        if colliding_name is not None and colliding_name != name:
            raise ValueError(
                f"Dynamic LoRA names {colliding_name!r} and {name!r} produce the same internal ID; rename one adapter."
            )
        registry[name] = LoRARequest(
            lora_name=name,
            lora_int_id=internal_id,
            lora_path=request.lora_path,
        )
        internal_ids[internal_id] = name
    return tuple(registry[name] for name in sorted(registry))
