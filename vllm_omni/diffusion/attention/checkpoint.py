# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Resolve checkpoint attention defaults before constructing concrete executors."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import AttentionConfig, TransformerConfig

PolicySource = Literal["runtime_disabled", "runtime", "default", "checkpoint"]


class CheckpointAttentionPolicyError(ValueError):
    """Invalid effective attention policy; model discovery must not suppress it."""


def resolve_checkpoint_attention_config(
    runtime_config: AttentionConfig, transformer_config: TransformerConfig
) -> tuple[AttentionConfig, PolicySource]:
    """Return a complete effective config and its source; never merge policies.

    Explicit runtime methods win over checkpoint metadata. The ignore option is
    an escape hatch for checkpoints with an unwanted or unsupported policy.
    """
    from vllm_omni.diffusion.data import AttentionConfig

    if runtime_config.checkpoint_policy == "ignore":
        return runtime_config, "runtime_disabled"
    if runtime_config.strategy is not None or runtime_config.default is not None or runtime_config.per_role:
        return runtime_config, "runtime"
    params = transformer_config.params
    if "runtime" not in params:
        return runtime_config, "default"
    runtime = params["runtime"]
    if not isinstance(runtime, Mapping):
        raise ValueError("Checkpoint runtime must be a mapping")
    if "attention_strategy" not in runtime:
        return runtime_config, "default"
    policy = runtime["attention_strategy"]
    if not isinstance(policy, Mapping) or set(policy) != {"schema_version", "config"}:
        raise ValueError("Checkpoint attention_strategy requires schema_version and config")
    if type(policy["schema_version"]) is not int or policy["schema_version"] != 1:
        raise ValueError("Checkpoint attention_strategy.schema_version must be the integer 1")
    config = policy["config"]
    if not isinstance(config, Mapping) or config.keys() - {"presets", "layout", "layouts", "schedule"}:
        raise ValueError("Checkpoint attention strategy config requires presets and layout or layouts/schedule")
    effective = AttentionConfig(**deepcopy(dict(config)))
    if effective.strategy is None:
        raise ValueError("Checkpoint attention strategy must declare an active layout or schedule")
    return effective, "checkpoint"
