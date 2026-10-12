# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest

from vllm_omni.model_executor.models.lychee_fd.weight_loader import (
    IGNORED_AUXILIARY_EMBEDDINGS,
    LycheeWeightMappingError,
    build_weight_load_plan,
    released_checkpoint_keys,
    route_checkpoint_key,
    validate_released_checkpoint_inventory,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _native_target_names() -> set[str]:
    return {
        route.target_name for name in released_checkpoint_keys() if (route := route_checkpoint_key(name)) is not None
    }


def test_released_inventory_is_exact_and_complete() -> None:
    keys = released_checkpoint_keys()

    assert len(keys) == 982
    assert validate_released_checkpoint_inventory(keys) == keys
    assert IGNORED_AUXILIARY_EMBEDDINGS < keys


def test_routes_qkv_and_mlp_shards() -> None:
    q = route_checkpoint_key("model.layers.0.self_attn.q_proj.weight")
    gate = route_checkpoint_key("merge_model.layers.3.mlp.gate_proj.weight")

    assert q is not None
    assert q.target_name == "model.layers.0.self_attn.qkv_proj.weight"
    assert q.shard_id == "q"
    assert gate is not None
    assert gate.target_name == "merge_model.layers.3.mlp.gate_up_proj.weight"
    assert gate.shard_id == 0


def test_only_auxiliary_branch_embeddings_are_ignored() -> None:
    for name in IGNORED_AUXILIARY_EMBEDDINGS:
        assert route_checkpoint_key(name) is None

    with pytest.raises(LycheeWeightMappingError, match="Unexpected"):
        route_checkpoint_key("merge_model.extra_projection.weight")


def test_strict_plan_rejects_missing_native_parameter() -> None:
    targets = _native_target_names()
    targets.add("model.unbacked_parameter")

    with pytest.raises(LycheeWeightMappingError, match="without checkpoint tensors"):
        build_weight_load_plan(released_checkpoint_keys(), targets)


def test_strict_plan_covers_every_native_parameter() -> None:
    targets = _native_target_names()
    plan = build_weight_load_plan(released_checkpoint_keys(), targets)

    assert {route.target_name for route in plan} == targets
