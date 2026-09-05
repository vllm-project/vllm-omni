# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ABot-World model module structure and type validation."""

from __future__ import annotations


def test_module_exports():
    """Verify the abot_world module exports the expected symbols."""
    from vllm_omni.diffusion.models.abot_world import (
        ABOT_CAMERA_ACTION_SCHEMA,
        ABotCameraControlReducer,
        ABotWorldCausalPipeline,
        get_abot_world_post_process_func,
        get_abot_world_pre_process_func,
    )

    assert ABOT_CAMERA_ACTION_SCHEMA == "abot.camera_actions.v1"
    assert ABotCameraControlReducer is not None
    assert ABotWorldCausalPipeline is not None
    assert callable(get_abot_world_post_process_func)
    assert callable(get_abot_world_pre_process_func)


def test_transformer_exports():
    """Verify the transformer module exports the expected symbols."""
    from vllm_omni.diffusion.models.abot_world.abot_world_transformer import (
        ABotAttentionCache,
        ABotCausalAttentionBlock,
        ABotCausalCrossAttention,
        ABotCausalSelfAttention,
        ABotSimpleAdapter,
        ABotTransformerCache,
        ABotWorldCausalTransformer3DModel,
        allocate_abot_cache,
    )

    assert ABotWorldCausalTransformer3DModel is not None
    assert ABotCausalSelfAttention is not None
    assert ABotCausalCrossAttention is not None
    assert ABotCausalAttentionBlock is not None
    assert ABotSimpleAdapter is not None
    assert callable(allocate_abot_cache)
    assert ABotTransformerCache is not None
    assert ABotAttentionCache is not None


def test_actions_exports():
    """Verify the actions module exports the expected symbols."""
    from vllm_omni.diffusion.models.abot_world.actions import (
        ABOT_CAMERA_ACTION_SCHEMA,
        ABotCameraControlReducer,
        parse_abot_camera_action_frames,
    )

    assert ABOT_CAMERA_ACTION_SCHEMA == "abot.camera_actions.v1"
    assert ABotCameraControlReducer is not None
    assert callable(parse_abot_camera_action_frames)


def test_registry_entry():
    """Verify ABotWorld is registered in the diffusion model registry."""
    from vllm_omni.diffusion.registry import (
        _DIFFUSION_MODELS,
        _DIFFUSION_POST_PROCESS_FUNCS,
        _DIFFUSION_PRE_PROCESS_FUNCS,
    )

    assert "ABotWorldCausalPipeline" in _DIFFUSION_MODELS
    mod_folder, mod_relname, cls_name = _DIFFUSION_MODELS["ABotWorldCausalPipeline"]
    assert mod_folder == "abot_world"
    assert mod_relname == "pipeline"
    assert cls_name == "ABotWorldCausalPipeline"

    assert "ABotWorldCausalPipeline" in _DIFFUSION_POST_PROCESS_FUNCS
    assert _DIFFUSION_POST_PROCESS_FUNCS["ABotWorldCausalPipeline"] == "get_abot_world_post_process_func"

    assert "ABotWorldCausalPipeline" in _DIFFUSION_PRE_PROCESS_FUNCS
    assert _DIFFUSION_PRE_PROCESS_FUNCS["ABotWorldCausalPipeline"] == "get_abot_world_pre_process_func"
