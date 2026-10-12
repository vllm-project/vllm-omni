# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Strict checkpoint inventory and fused-parameter routing for Lychee-FD.

This module is intentionally tensor-free. It validates the published key
space before the model loader streams tensors and gives ``modeling_lychee`` an
unambiguous target parameter and shard id for each checkpoint tensor.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from functools import lru_cache


class LycheeWeightMappingError(ValueError):
    """The checkpoint and native Lychee model parameter inventories disagree."""


_BRANCH_LAYERS = {
    "model": 28,
    "stoken_model": 4,
    "control_model": 4,
    "merge_model": 4,
}

_DECODER_LAYER_SUFFIXES = (
    "input_layernorm.weight",
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "post_attention_layernorm.weight",
    "self_attn.k_proj.bias",
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_proj.bias",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.bias",
    "self_attn.v_proj.weight",
)

_AUDIO_BLOCK_SUFFIXES = (
    "attn.key.weight",
    "attn.out.bias",
    "attn.out.weight",
    "attn.query.bias",
    "attn.query.weight",
    "attn.value.bias",
    "attn.value.weight",
    "attn_ln.bias",
    "attn_ln.weight",
    "mlp.0.bias",
    "mlp.0.weight",
    "mlp.2.bias",
    "mlp.2.weight",
    "mlp_ln.bias",
    "mlp_ln.weight",
)

_AUDIO_ROOT_KEYS = (
    "encoder.after_norm.bias",
    "encoder.after_norm.weight",
    "encoder.conv1.bias",
    "encoder.conv1.weight",
    "encoder.conv2.bias",
    "encoder.conv2.weight",
    "encoder.positional_embedding.weight",
    "adapter.conv.bias",
    "adapter.conv.weight",
    "adapter.linear1.bias",
    "adapter.linear1.weight",
    "adapter.linear2.bias",
    "adapter.linear2.weight",
)

# The reference execution shares model.embed_tokens for all three input
# channels. These tensors exist in the published HF checkpoint but are not
# parameters of the native graph. No other checkpoint key may be ignored.
IGNORED_AUXILIARY_EMBEDDINGS = frozenset(
    {
        "stoken_model.embed_tokens.weight",
        "control_model.embed_tokens.weight",
        "merge_model.embed_tokens.weight",
    }
)

_PACKED_SEGMENTS: tuple[tuple[str, str, str | int], ...] = (
    (".self_attn.q_proj.", ".self_attn.qkv_proj.", "q"),
    (".self_attn.k_proj.", ".self_attn.qkv_proj.", "k"),
    (".self_attn.v_proj.", ".self_attn.qkv_proj.", "v"),
    (".mlp.gate_proj.", ".mlp.gate_up_proj.", 0),
    (".mlp.up_proj.", ".mlp.gate_up_proj.", 1),
)


@dataclass(frozen=True, slots=True)
class LycheeWeightRoute:
    source_name: str
    target_name: str
    shard_id: str | int | None = None


class LycheeStreamingWeightTracker:
    """Validate and account for weights without retaining checkpoint tensors."""

    def __init__(self, target_names: Collection[str]) -> None:
        self.target_names = frozenset(target_names)
        self.source_names: set[str] = set()
        self.covered_targets: set[str] = set()
        self._seen_routes: set[tuple[str, str | int | None]] = set()

    def route(self, source_name: str) -> LycheeWeightRoute | None:
        if source_name in self.source_names:
            raise LycheeWeightMappingError(f"Duplicate Lychee checkpoint key: {source_name}")
        self.source_names.add(source_name)
        route = route_checkpoint_key(source_name)
        if route is None:
            return None
        if route.target_name not in self.target_names:
            raise LycheeWeightMappingError(
                f"Checkpoint tensor {source_name} maps to absent model parameter {route.target_name}"
            )
        route_key = (route.target_name, route.shard_id)
        if route_key in self._seen_routes:
            raise LycheeWeightMappingError(f"Duplicate Lychee weight route: {route_key}")
        self._seen_routes.add(route_key)
        self.covered_targets.add(route.target_name)
        return route

    def finish(self) -> set[str]:
        validate_released_checkpoint_inventory(self.source_names)
        missing_targets = sorted(self.target_names - self.covered_targets)
        if missing_targets:
            raise LycheeWeightMappingError(
                f"Native Lychee model has {len(missing_targets)} parameters without checkpoint tensors: "
                f"{missing_targets[:5]}"
            )
        return set(self.covered_targets)


@lru_cache(maxsize=1)
def released_checkpoint_keys() -> frozenset[str]:
    """Return the exact published tensor-name inventory (982 tensors)."""

    keys = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "stoken_model.embed_tokens.weight",
        "stoken_model.norm.weight",
        "control_model.embed_tokens.weight",
        "control_model.norm.weight",
        "merge_model.embed_tokens.weight",
        "merge_model.norm.weight",
        "lm_head.weight",
        *_AUDIO_ROOT_KEYS,
    }
    for branch, layer_count in _BRANCH_LAYERS.items():
        for layer_index in range(layer_count):
            for suffix in _DECODER_LAYER_SUFFIXES:
                keys.add(f"{branch}.layers.{layer_index}.{suffix}")
    for layer_index in range(32):
        for suffix in _AUDIO_BLOCK_SUFFIXES:
            keys.add(f"encoder.blocks.{layer_index}.{suffix}")
    return frozenset(keys)


def validate_released_checkpoint_inventory(source_names: Iterable[str]) -> frozenset[str]:
    """Reject any missing, additional or duplicate released checkpoint key."""

    names = tuple(source_names)
    unique = frozenset(names)
    if len(unique) != len(names):
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        raise LycheeWeightMappingError(f"Duplicate Lychee checkpoint keys: {duplicates[:5]}")

    expected = released_checkpoint_keys()
    missing = sorted(expected - unique)
    unexpected = sorted(unique - expected)
    if missing or unexpected:
        raise LycheeWeightMappingError(
            "Lychee checkpoint inventory mismatch: "
            f"missing={missing[:5]} ({len(missing)} total), "
            f"unexpected={unexpected[:5]} ({len(unexpected)} total)"
        )
    return unique


def route_checkpoint_key(source_name: str) -> LycheeWeightRoute | None:
    """Map one validated HF key to the native vLLM parameter inventory."""

    if source_name in IGNORED_AUXILIARY_EMBEDDINGS:
        return None
    if source_name not in released_checkpoint_keys():
        raise LycheeWeightMappingError(f"Unexpected Lychee checkpoint key: {source_name}")
    for source_segment, target_segment, shard_id in _PACKED_SEGMENTS:
        if source_segment in source_name:
            return LycheeWeightRoute(
                source_name=source_name,
                target_name=source_name.replace(source_segment, target_segment, 1),
                shard_id=shard_id,
            )
    return LycheeWeightRoute(source_name=source_name, target_name=source_name)


def build_weight_load_plan(
    source_names: Iterable[str],
    target_names: Collection[str],
) -> tuple[LycheeWeightRoute, ...]:
    """Build a complete strict load plan for a native model instance."""

    sources = validate_released_checkpoint_inventory(source_names)
    targets = frozenset(target_names)
    routes: list[LycheeWeightRoute] = []
    covered_targets: set[str] = set()
    seen_routes: set[tuple[str, str | int | None]] = set()

    for source_name in sorted(sources):
        route = route_checkpoint_key(source_name)
        if route is None:
            continue
        if route.target_name not in targets:
            raise LycheeWeightMappingError(
                f"Checkpoint tensor {source_name} maps to absent model parameter {route.target_name}"
            )
        route_key = (route.target_name, route.shard_id)
        if route_key in seen_routes:
            raise LycheeWeightMappingError(f"Duplicate Lychee weight route: {route_key}")
        seen_routes.add(route_key)
        covered_targets.add(route.target_name)
        routes.append(route)

    missing_targets = sorted(targets - covered_targets)
    if missing_targets:
        raise LycheeWeightMappingError(
            f"Native Lychee model has {len(missing_targets)} parameters without checkpoint tensors: "
            f"{missing_targets[:5]}"
        )
    return tuple(routes)


__all__ = [
    "IGNORED_AUXILIARY_EMBEDDINGS",
    "LycheeWeightMappingError",
    "LycheeWeightRoute",
    "LycheeStreamingWeightTracker",
    "build_weight_load_plan",
    "released_checkpoint_keys",
    "route_checkpoint_key",
    "validate_released_checkpoint_inventory",
]
