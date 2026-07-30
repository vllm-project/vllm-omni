# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Collection, Mapping

from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniDiffusionSamplingParams

logger = init_logger(__name__)


def _request_mapping(name: str, value: object | None) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object.")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys.")
    return dict(value)


def normalize_diffusion_request_args(
    *,
    root: object | None = None,
    nested: object | None = None,
    explicit_root_args: object | None = None,
    serving_root_fields: Collection[str] = (),
    registered_extra_fields: Collection[str] = (),
    root_field_aliases: Mapping[str, str] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Validate request sources and project diffusion consumer arguments."""
    root_args = _request_mapping("request", root)
    root_args.update(_request_mapping("explicit_root_args", explicit_root_args))
    nested_args = _request_mapping("request.extra_body", nested)

    registered = set(registered_extra_fields)
    aliases = {alias: canonical for alias, canonical in (root_field_aliases or {}).items() if alias not in registered}
    serving_fields = set(serving_root_fields) | aliases.keys()
    declared_fields = registered - serving_fields
    consumer_fields = serving_fields | declared_fields

    root_declared = {key: root_args[key] for key in declared_fields if key in root_args}
    nested_declared = {key: nested_args[key] for key in declared_fields if key in nested_args}
    canonical = _request_mapping("extra_args", root_args.get("extra_args"))
    legacy = _request_mapping("extra_params", root_args.get("extra_params"))
    nested_canonical = _request_mapping("extra_body.extra_args", nested_args.get("extra_args"))
    nested_legacy = _request_mapping("extra_body.extra_params", nested_args.get("extra_params"))

    sources_by_key: dict[str, list[str]] = {}
    sources = (
        ((root_args.keys() & serving_fields) | root_declared.keys(), "request"),
        ((nested_args.keys() & serving_fields) | nested_declared.keys(), "request.extra_body"),
        (canonical, "request.extra_args"),
        (legacy, "request.extra_params"),
        (nested_canonical, "request.extra_body.extra_args"),
        (nested_legacy, "request.extra_body.extra_params"),
    )
    for keys, prefix in sources:
        for key in sorted(keys):
            sources_by_key.setdefault(aliases.get(key, key), []).append(f"{prefix}.{key}")

    conflicts = {key: list(dict.fromkeys(paths)) for key, paths in sources_by_key.items() if len(set(paths)) > 1}
    if conflicts:
        details = "; ".join(f'"{key}": {", ".join(conflicts[key])}' for key in sorted(conflicts))
        raise ValueError(f"Diffusion request parameters were provided more than once: {details}.")

    if root_args.get("extra_params") is not None or nested_args.get("extra_params") is not None:
        logger.warning_once(
            "extra_params is deprecated; use extra_args for model-specific diffusion request parameters."
        )

    # Duplicate request keys were rejected above, so ordering is not precedence.
    normalized_extra_args = {
        **nested_declared,
        **root_declared,
        **legacy,
        **nested_legacy,
        **canonical,
        **nested_canonical,
    }
    normalized_extra_args = {key: value for key, value in normalized_extra_args.items() if value is not None}

    request_args = {
        **{key: nested_args[key] for key in consumer_fields if key in nested_args},
        **{key: root_args[key] for key in consumer_fields if key in root_args},
    }
    for alias, canonical_name in aliases.items():
        if alias in request_args:
            request_args[canonical_name] = request_args.pop(alias)
    return normalized_extra_args, request_args


def apply_normalized_diffusion_request_extra_args(
    sampling_params: OmniDiffusionSamplingParams,
    normalized_extra_args: Mapping[str, object],
) -> None:
    """Overlay request extras without discarding stage defaults."""
    if normalized_extra_args:
        sampling_params.extra_args = {
            **(sampling_params.extra_args or {}),
            **normalized_extra_args,
        }
