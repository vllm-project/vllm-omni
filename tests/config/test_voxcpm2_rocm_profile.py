# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Keep VoxCPM2's ROCm accuracy profile scoped to CI qualification."""

import pytest
import yaml

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.stage_config import _apply_platform_overrides, load_deploy_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_voxcpm2_ci_overlay_exposes_stage_anchor_to_test_modifiers() -> None:
    with open(get_deploy_config_path("ci/voxcpm2.yaml"), encoding="utf-8") as stream:
        overlay = yaml.safe_load(stream)

    assert overlay["stages"] == [{"stage_id": 0}]


def _load_for(platform: str, *, ci_profile: bool = True):
    deploy = "ci/voxcpm2.yaml" if ci_profile else "voxcpm2.yaml"
    return _apply_platform_overrides(
        load_deploy_config(get_deploy_config_path(deploy)),
        platform=platform,
    )


def test_voxcpm2_ci_profile_uses_native_attention_on_rocm() -> None:
    deploy = _load_for("rocm")

    assert deploy.stages[0].attention_backend == "ROCM_ATTN"


def _runtime_profile(deploy):
    return deploy.stages[0].engine_extras["hf_overrides"]["voxcpm2_runtime_config"]


def test_voxcpm2_ci_profile_uses_conservative_locdit_path_on_rocm() -> None:
    runtime = _runtime_profile(_load_for("rocm"))

    assert runtime["enable_torch_compile"] is False
    assert runtime["enable_loc_dit_fused_qkv"] is False
    assert runtime["enable_loc_dit_fused_mlp"] is False
    assert runtime["enable_loc_dit_zero_dt_cache"] is False
    assert runtime["enable_loc_dit_skip_qkv_contig"] is False
    # The platform override must recursively preserve unrelated production
    # settings instead of replacing the full hf_overrides subtree.
    assert runtime["inference_timesteps"] == 10
    assert runtime["unified_decode_graph_max_batch_size"] == 8


@pytest.mark.parametrize("platform", ["cuda", "npu", "xpu", "musa"])
def test_voxcpm2_ci_native_attention_override_is_rocm_only(platform: str) -> None:
    deploy = _load_for(platform)

    assert deploy.stages[0].attention_backend != "ROCM_ATTN"
    runtime = _runtime_profile(deploy)
    assert "enable_torch_compile" not in runtime
    assert runtime["enable_loc_dit_fused_qkv"] is True
    assert runtime["enable_loc_dit_fused_mlp"] is True
    assert runtime["enable_loc_dit_zero_dt_cache"] is True


def test_voxcpm2_production_profile_does_not_force_native_attention_on_rocm() -> None:
    deploy = _load_for("rocm", ci_profile=False)

    assert deploy.stages[0].attention_backend != "ROCM_ATTN"
    runtime = _runtime_profile(deploy)
    assert "enable_torch_compile" not in runtime
    assert runtime["enable_loc_dit_fused_qkv"] is True
    assert runtime["enable_loc_dit_fused_mlp"] is True
    assert runtime["enable_loc_dit_zero_dt_cache"] is True
