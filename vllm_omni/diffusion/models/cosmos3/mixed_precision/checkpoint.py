# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Checkpoint discovery for the Cosmos3 diffusion-step precision policy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, cast

from .config import Cosmos3MixedPrecisionConfig, ReasonerPolicy, _non_negative_int

PolicySource = Literal[
    "none",
    "checkpoint",
    "additional_config",
    "additional_config_disabled",
]

_COMPONENT = "transformer"
_POLICY_FIELDS = frozenset(
    {
        "schema_version",
        "type",
        "index_space",
        "scope",
        "default_mode",
        "first_steps",
        "last_steps",
        "overlap",
        "reasoner",
    }
)
_STEP_RANGE_FIELDS = frozenset({"count", "mode"})
_SUPPORTED_MODEL_OPT_CONFIGS = frozenset(
    {
        "modelopt",
        "modelopt_fp4",
    }
)


def resolve_mixed_precision_config(
    od_config: object,
) -> tuple[Cosmos3MixedPrecisionConfig | None, PolicySource]:
    """Resolve an explicit runtime override before the checkpoint default."""
    override_present, override = Cosmos3MixedPrecisionConfig.resolve_additional_config(
        getattr(od_config, "additional_config", None)
    )
    if override_present:
        source: PolicySource = "additional_config" if override is not None else "additional_config_disabled"
        return override, source

    checkpoint = read_checkpoint_policy(od_config)
    if checkpoint is not None:
        return checkpoint, "checkpoint"
    return None, "none"


def read_checkpoint_policy(od_config: object) -> Cosmos3MixedPrecisionConfig | None:
    """Read and validate a policy from ``transformer/config.json``.

    Missing metadata preserves ordinary checkpoint behavior.  Metadata that is
    present but malformed or incompatible fails closed.
    """
    tf_config = getattr(od_config, "tf_model_config", None)
    params = getattr(tf_config, "params", None)
    if not isinstance(params, Mapping):
        return None

    disk_quant_config = params.get("quantization_config")
    if not isinstance(disk_quant_config, Mapping):
        return None

    if "runtime" not in disk_quant_config:
        return None
    runtime_config = disk_quant_config["runtime"]
    if not isinstance(runtime_config, Mapping):
        raise TypeError("quantization_config.runtime must be a mapping")

    if "diffusion_step_policy" not in runtime_config:
        return None
    raw_policy = runtime_config["diffusion_step_policy"]
    if not isinstance(raw_policy, Mapping):
        raise TypeError("quantization_config.runtime.diffusion_step_policy must be a mapping")

    policy = _parse_policy(raw_policy)
    if policy is None:
        return None
    _validate_checkpoint_quantization(getattr(tf_config, "quant_config", None))
    return policy


def _parse_policy(policy: Mapping[str, object]) -> Cosmos3MixedPrecisionConfig | None:
    fields = set(policy)
    unknown = fields - _POLICY_FIELDS
    if unknown:
        raise ValueError(f"Unknown diffusion_step_policy fields: {sorted(unknown)}")
    missing = _POLICY_FIELDS - fields
    if missing:
        raise ValueError(f"Missing diffusion_step_policy fields: {sorted(missing)}")

    schema_version = policy["schema_version"]
    if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version != 1:
        raise ValueError("diffusion_step_policy.schema_version must be the integer 1")
    if policy["type"] != "first_last_n":
        raise ValueError("diffusion_step_policy.type must be 'first_last_n'")
    if policy["index_space"] != "denoising_loop_iteration":
        raise ValueError("diffusion_step_policy.index_space must be 'denoising_loop_iteration'")
    if policy["default_mode"] != "native":
        raise ValueError("diffusion_step_policy.default_mode must be 'native'")
    if policy["overlap"] != "a16":
        raise ValueError("diffusion_step_policy.overlap must be 'a16'")

    scope = policy["scope"]
    if not isinstance(scope, list) or not scope or not all(isinstance(item, str) for item in scope):
        raise TypeError("diffusion_step_policy.scope must be a non-empty list of strings")

    first_steps = _parse_step_range(policy["first_steps"], "first_steps")
    last_steps = _parse_step_range(policy["last_steps"], "last_steps")

    reasoner = policy["reasoner"]
    if reasoner not in {"native", "a16"}:
        raise ValueError("diffusion_step_policy.reasoner must be 'native' or 'a16'")

    if _COMPONENT not in scope:
        return None
    return Cosmos3MixedPrecisionConfig(
        first_steps=first_steps,
        last_steps=last_steps,
        reasoner=cast(ReasonerPolicy, reasoner),
    )


def _parse_step_range(value: object, name: str) -> int:
    if not isinstance(value, Mapping):
        raise TypeError(f"diffusion_step_policy.{name} must be a mapping")
    fields = set(value)
    unknown = fields - _STEP_RANGE_FIELDS
    if unknown:
        raise ValueError(f"Unknown diffusion_step_policy.{name} fields: {sorted(unknown)}")
    missing = _STEP_RANGE_FIELDS - fields
    if missing:
        raise ValueError(f"Missing diffusion_step_policy.{name} fields: {sorted(missing)}")
    if value["mode"] != "a16":
        raise ValueError(f"diffusion_step_policy.{name}.mode must be 'a16'")
    return _non_negative_int(value["count"], f"diffusion_step_policy.{name}.count")


def _validate_checkpoint_quantization(quant_config: object | None) -> None:
    get_name = getattr(quant_config, "get_name", None)
    name = get_name() if callable(get_name) else None
    if name not in _SUPPORTED_MODEL_OPT_CONFIGS:
        raise ValueError(
            f"diffusion_step_policy requires a serialized ModelOpt FP8 or NVFP4 checkpoint, got {name!r}"
        )

    if name == "modelopt":
        _require_serialized(quant_config, "is_checkpoint_fp8_serialized", "FP8")
        return
    if name == "modelopt_fp4":
        if getattr(quant_config, "quant_method", "NVFP4") != "NVFP4":
            raise ValueError("diffusion_step_policy requires checkpoint-native NVFP4 W4A4 linears")
        _require_serialized(quant_config, "is_checkpoint_nvfp4_serialized", "NVFP4")
        return


def _require_serialized(quant_config: object | None, flag: str, format_name: str) -> None:
    if not bool(getattr(quant_config, flag, False)):
        raise ValueError(f"diffusion_step_policy requires serialized ModelOpt {format_name} weights")
