# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for MXFP4 quantization configs and the MXFP4 DualScale + BF16 mixed config."""

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _patch_tp_state(monkeypatch):
    """Patch TP rank/world_size so ModelWeightParameter can be instantiated on CPU
    without an initialized distributed group.  Returns TP=1 rank=0 for all tests."""
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1)


# ---------------------------------------------------------------------------
# DiffusionMXFP4Config
# ---------------------------------------------------------------------------


def test_mxfp4_config_get_name():
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    assert DiffusionMXFP4Config.get_name() == "mxfp4"


@pytest.mark.parametrize("value", [True, False, None, "2", 2.0, 1, -1, 3])
def test_mxfp4_scale_alg_rejects_invalid_values(value):
    from vllm_omni.quantization import build_quant_config

    with pytest.raises(ValueError, match="mxfp4_scale_alg"):
        build_quant_config({"method": "mxfp4", "mxfp4_scale_alg": value})


@pytest.mark.parametrize("runtime_alg", [0, 2])
@pytest.mark.parametrize("serialized", [False, True])
def test_runtime_algorithm_survives_offline_storage_rebuild(runtime_alg, serialized):
    from vllm_omni.quantization.factory import resolve_quant_config_from_disk
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    active = DiffusionMXFP4Config(is_checkpoint_mxfp4_serialized=serialized, mxfp4_scale_alg=runtime_alg)
    resolved = resolve_quant_config_from_disk(
        active,
        {
            "quant_method": "mxfp4",
            "is_checkpoint_mxfp4_serialized": True,
            "mxfp4_scale_alg": 2,
            "ignored_layers": ["proj_out"],
        },
    )
    assert resolved.is_checkpoint_mxfp4_serialized
    assert resolved.mxfp4_scale_alg == runtime_alg
    assert resolved.w4a8_fallback_layers == resolved.w4a8_fallback_steps == []


def test_mxfp4_config_from_config_defaults():
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    cfg = DiffusionMXFP4Config.from_config({})
    assert cfg.is_checkpoint_mxfp4_serialized is False
    assert cfg.ignored_layers == []
    assert cfg.w4a8_fallback_layers == []


def test_mxfp4_config_from_config_serialized():
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    cfg = DiffusionMXFP4Config.from_config({"is_checkpoint_mxfp4_serialized": True})
    assert cfg.is_checkpoint_mxfp4_serialized is True


def test_mxfp4_config_requires_offline_smooth():
    from vllm_omni.quantization import build_quant_config
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    values = {"is_checkpoint_mxfp4_serialized": True, "require_smooth_scale": True}
    assert DiffusionMXFP4Config.from_config(values).require_smooth_scale
    assert build_quant_config({"method": "mxfp4", **values}).require_smooth_scale
    with pytest.raises(ValueError, match="offline"):
        build_quant_config({"method": "mxfp4", "require_smooth_scale": True})
    with pytest.raises(ValueError, match="boolean"):
        DiffusionMXFP4Config.from_config({**values, "require_smooth_scale": "false"})


@pytest.mark.parametrize("serialized", [False, True])
def test_mxfp4_disk_smooth_requirement_preserves_step_policy(serialized):
    from vllm_omni.quantization.factory import resolve_quant_config_from_disk
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    active = DiffusionMXFP4Config(is_checkpoint_mxfp4_serialized=serialized, w4a8_fallback_steps=[1])
    resolved = resolve_quant_config_from_disk(
        active,
        {
            "quant_method": "mxfp4",
            "is_checkpoint_mxfp4_serialized": True,
            "require_smooth_scale": True,
        },
    )
    assert resolved.require_smooth_scale
    assert resolved.is_checkpoint_mxfp4_serialized
    assert resolved.w4a8_fallback_steps == [1]


@pytest.mark.parametrize("disk_requirement", [None, False, True])
def test_mxfp4_expert_rebuild_cannot_weaken_explicit_smooth_requirement(disk_requirement):
    from vllm_omni.quantization.factory import resolve_quant_config_from_disk
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    active = DiffusionMXFP4Config(is_checkpoint_mxfp4_serialized=True, require_smooth_scale=True)
    disk = {
        "quant_method": "mxfp4",
        "is_checkpoint_mxfp4_serialized": True,
        "ignored_layers": ["blocks.0.attn2.to_q"],
    }
    if disk_requirement is not None:
        disk["require_smooth_scale"] = disk_requirement
    resolved = resolve_quant_config_from_disk(active, disk)
    assert resolved.require_smooth_scale
    assert resolved.ignored_layers == disk["ignored_layers"]


@pytest.mark.parametrize("disk_qc", ["mxfp4", {"quant_method": "mxfp4"}])
def test_mxfp4_method_only_disk_config_preserves_caller_layer_policy(disk_qc):
    from vllm_omni.quantization.factory import resolve_quant_config_from_disk
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    active = DiffusionMXFP4Config(
        ignored_layers=["blocks.0.attn2.to_q"],
        w4a8_fallback_steps=[1],
        w4a8_fallback_layers=["blocks.1.attn1.to_qkv"],
    )

    resolved = resolve_quant_config_from_disk(active, disk_qc)

    assert resolved is active
    assert resolved.ignored_layers == ["blocks.0.attn2.to_q"]
    assert resolved.w4a8_fallback_steps == [1]
    assert resolved.w4a8_fallback_layers == ["blocks.1.attn1.to_qkv"]


def test_quantization_disk_string_rejects_mismatched_active_method():
    from vllm_omni.quantization import build_quant_config
    from vllm_omni.quantization.factory import resolve_quant_config_from_disk

    active = build_quant_config("mxfp4")

    with pytest.raises(ValueError, match="mxfp4_dualscale.*active quantization config is 'mxfp4'"):
        resolve_quant_config_from_disk(active, "mxfp4_dualscale")


def test_quantization_disk_string_accepts_equivalent_method_alias():
    from vllm_omni.quantization.factory import resolve_quant_config_from_disk

    class ActiveAliasConfig:
        @staticmethod
        def get_name() -> str:
            return "inc"

    active = ActiveAliasConfig()

    assert resolve_quant_config_from_disk(active, "auto-round") is active


def test_mxfp4_config_from_config_ignored_layers():
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    cfg = DiffusionMXFP4Config.from_config({"ignored_layers": ["proj_out"]})
    assert cfg.ignored_layers == ["proj_out"]


def test_mxfp4_config_from_config_modules_to_not_convert_fallback():
    """modules_to_not_convert must be accepted as an alias for ignored_layers."""
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    cfg = DiffusionMXFP4Config.from_config({"modules_to_not_convert": ["proj_out"]})
    assert cfg.ignored_layers == ["proj_out"]


@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize(
    "policy",
    [
        {"w4a8_fallback_steps": [0]},
        {"w4a8_fallback_layers": ["blocks.10.attn1.to_qkv"]},
        {"w4a8_fallback_steps": [1], "w4a8_fallback_layers": ["blocks.10.attn1.to_qkv"]},
    ],
)
def test_dualscale_rejects_w4a8_configuration(serialized, policy):
    from vllm_omni.quantization import build_quant_config
    from vllm_omni.quantization.factory import resolve_quant_config_from_disk
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4DualScaleMixedConfig

    config = {"quant_method": "mxfp4_dualscale", "is_checkpoint_serialized": serialized, **policy}
    with pytest.raises(ValueError, match="does not support W4A8"):
        DiffusionMXFP4DualScaleMixedConfig(is_checkpoint_serialized=serialized, **policy)
    with pytest.raises(ValueError, match="does not support W4A8"):
        DiffusionMXFP4DualScaleMixedConfig.from_config(config)
    with pytest.raises(ValueError, match="does not support W4A8"):
        build_quant_config({"transformer": config})
    with pytest.raises(ValueError, match="does not support W4A8"):
        resolve_quant_config_from_disk(None, config)


@pytest.mark.parametrize("method", ["mxfp4", "mxfp4_dualscale"])
@pytest.mark.parametrize("steps", [[-1], [True], [1.5], ["2"], "0,2", (0, 2)])
def test_mxfp4_rejects_invalid_fallback_steps(method, steps):
    from vllm_omni.quantization import build_quant_config

    with pytest.raises(ValueError, match="non-negative integer"):
        build_quant_config({"method": method, "w4a8_fallback_steps": steps})


def test_mxfp4_fallback_steps_follow_request_context():
    from vllm_omni.diffusion.forward_context import (
        ForwardContext,
        override_forward_context,
        set_forward_context_denoise_step_idx,
    )
    from vllm_omni.quantization import build_quant_config
    from vllm_omni.quantization.mxfp4_config import _use_w4a8

    steps = [2, 0, 2]
    cfg = build_quant_config({"method": "mxfp4", "w4a8_fallback_steps": steps})
    steps.append(1)
    assert cfg.w4a8_fallback_steps == [0, 2]
    with override_forward_context(None):
        assert not _use_w4a8([])
        with pytest.raises(RuntimeError, match="publishes its denoise step"):
            _use_w4a8(cfg.w4a8_fallback_steps)
        for _request in range(2):
            with override_forward_context(ForwardContext()):
                for step, expected in [(0, True), (1, False), (2, True), (3, False)]:
                    set_forward_context_denoise_step_idx(step)
                    # Both CFG passes and both experts read the same index.
                    assert [_use_w4a8(cfg.w4a8_fallback_steps) for _cfg_pass in range(2)] == [expected, expected]
        with pytest.raises(RuntimeError, match="publishes its denoise step"):
            _use_w4a8(cfg.w4a8_fallback_steps)


@pytest.mark.parametrize("method,flag", [("mxfp4", "is_checkpoint_mxfp4_serialized")])
@pytest.mark.parametrize("steps", [[], [0, 2]])
@pytest.mark.parametrize("already_serialized", [False, True])
def test_mxfp4_disk_rebuild_preserves_runtime_fallback(method, flag, steps, already_serialized):
    from vllm_omni.quantization import build_quant_config
    from vllm_omni.quantization.factory import resolve_quant_config_from_disk

    layers = ["blocks.10.attn1.to_qkv"] if steps else []
    active = build_quant_config(
        {
            "method": method,
            flag: already_serialized,
            "w4a8_fallback_steps": steps,
            "w4a8_fallback_layers": layers,
        }
    )
    # Separate experts may have distinct high precision layers. Both must
    # retain the requested step policy during online/offline reconciliation.
    for ignored in [["blocks.0.attn1.to_q"], ["blocks.1.attn1.to_q"]]:
        resolved = resolve_quant_config_from_disk(
            active,
            {
                "quant_method": method,
                flag: True,
                "ignored_layers": ignored,
                "w4a8_fallback_steps": [9],
                "w4a8_fallback_layers": ["blocks.9.ffn.net_2"],
            },
        )
        assert getattr(resolved, flag)
        assert resolved.ignored_layers == ignored
        assert resolved.w4a8_fallback_steps == steps
        assert resolved.w4a8_fallback_layers == layers
    assert active.ignored_layers == []


@pytest.mark.parametrize(
    "runtime_policy",
    [{}, {"w4a8_fallback_steps": [1]}, {"w4a8_fallback_layers": ["blocks.10.attn1.to_qkv"]}],
)
def test_mxfp4_explicit_omission_disables_saved_policies_for_both_experts(runtime_policy):
    from vllm_omni.quantization import build_quant_config
    from vllm_omni.quantization.factory import resolve_quant_config_from_disk

    active = build_quant_config({"method": "mxfp4", **runtime_policy})
    resolved_experts = []
    for ignored in (["blocks.0.attn2.to_q"], ["blocks.1.ffn.net_2"]):
        disk = {
            "quant_method": "mxfp4",
            "is_checkpoint_mxfp4_serialized": True,
            "ignored_layers": ignored,
            "w4a8_fallback_steps": [9],
            "w4a8_fallback_layers": ["blocks.9.ffn.net_2"],
        }
        resolved = resolve_quant_config_from_disk(active, disk)
        assert resolved.w4a8_fallback_steps == runtime_policy.get("w4a8_fallback_steps", [])
        assert resolved.w4a8_fallback_layers == runtime_policy.get("w4a8_fallback_layers", [])
        assert resolved.ignored_layers == ignored
        detected = resolve_quant_config_from_disk(None, disk)
        assert detected.w4a8_fallback_steps == [9]
        assert detected.w4a8_fallback_layers == ["blocks.9.ffn.net_2"]
        resolved_experts.append(resolved)
    assert resolved_experts[0] is not resolved_experts[1]
    assert active.is_checkpoint_mxfp4_serialized is False
    assert active.ignored_layers == []


@pytest.mark.parametrize(
    "method,flag", [("mxfp4", "is_checkpoint_mxfp4_serialized"), ("mxfp4_dualscale", "is_checkpoint_serialized")]
)
def test_serialized_mxfp4_disk_omission_clears_active_ignored_layers(method, flag):
    from vllm_omni.quantization import build_quant_config
    from vllm_omni.quantization.factory import resolve_quant_config_from_disk

    active_spec = {
        "method": method,
        flag: True,
        "ignored_layers": ["blocks.0.attn2.to_q"],
        "w4a8_fallback_steps": [1] if method == "mxfp4" else [],
        "w4a8_fallback_layers": ["blocks.1.attn1.to_qkv"] if method == "mxfp4" else [],
    }
    if method == "mxfp4":
        active_spec["require_smooth_scale"] = True
    active = build_quant_config(active_spec)

    resolved = resolve_quant_config_from_disk(active, {"quant_method": method, flag: True})

    assert resolved is not active
    assert resolved.ignored_layers == []
    assert resolved.w4a8_fallback_steps == active.w4a8_fallback_steps
    assert resolved.w4a8_fallback_layers == active.w4a8_fallback_layers
    if method == "mxfp4":
        assert resolved.require_smooth_scale


@pytest.mark.parametrize(
    "method,flag", [("mxfp4", "is_checkpoint_mxfp4_serialized"), ("mxfp4_dualscale", "is_checkpoint_serialized")]
)
@pytest.mark.parametrize("with_active", [False, True])
def test_serialized_mxfp4_disk_uses_modules_to_not_convert(method, flag, with_active):
    from vllm_omni.quantization import build_quant_config
    from vllm_omni.quantization.factory import resolve_quant_config_from_disk

    active = None
    disk_steps = [9] if method == "mxfp4" else []
    active_steps = [1] if method == "mxfp4" else []
    expected_steps = disk_steps
    if with_active:
        active = build_quant_config(
            {
                "method": method,
                flag: True,
                "ignored_layers": ["blocks.0.attn2.to_q"],
                "w4a8_fallback_steps": active_steps,
            }
        )
        expected_steps = active_steps

    resolved = resolve_quant_config_from_disk(
        active,
        {
            "quant_method": method,
            flag: True,
            "modules_to_not_convert": ["blocks.2.ffn.net_2"],
            "w4a8_fallback_steps": disk_steps,
        },
    )

    assert resolved.ignored_layers == ["blocks.2.ffn.net_2"]
    assert resolved.w4a8_fallback_steps == expected_steps


# ---------------------------------------------------------------------------
# build_quant_config integration
# ---------------------------------------------------------------------------


def test_build_quant_config_mxfp4_string():
    from vllm_omni.quantization import build_quant_config
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    cfg = build_quant_config("mxfp4")
    assert isinstance(cfg, DiffusionMXFP4Config)
    assert cfg.get_name() == "mxfp4"
    assert cfg.is_checkpoint_mxfp4_serialized is False


def test_build_quant_config_mxfp4_dict():
    from vllm_omni.quantization import build_quant_config
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    cfg = build_quant_config({"method": "mxfp4", "is_checkpoint_mxfp4_serialized": True})
    assert isinstance(cfg, DiffusionMXFP4Config)
    assert cfg.is_checkpoint_mxfp4_serialized is True


def test_build_quant_config_mxfp4_dualscale_string():
    from vllm_omni.quantization import build_quant_config
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4DualScaleMixedConfig

    cfg = build_quant_config("mxfp4_dualscale")
    assert isinstance(cfg, DiffusionMXFP4DualScaleMixedConfig)
    assert cfg.is_checkpoint_serialized is False
    assert cfg.num_bf16_fallback_layers == 5
    assert cfg.ignored_layers == []


def test_build_quant_config_mxfp4_dualscale_dict_offline():
    from vllm_omni.quantization import build_quant_config
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4DualScaleMixedConfig

    cfg = build_quant_config(
        {
            "method": "mxfp4_dualscale",
            "is_checkpoint_serialized": True,
            "ignored_layers": ["blocks.0.attn1.to_q", "blocks.0.attn1.to_k"],
        }
    )
    assert isinstance(cfg, DiffusionMXFP4DualScaleMixedConfig)
    assert cfg.is_checkpoint_serialized is True
    assert cfg.ignored_layers == ["blocks.0.attn1.to_q", "blocks.0.attn1.to_k"]


def test_build_quant_config_mxfp4_dualscale_dict_online_custom_fallback():
    from vllm_omni.quantization import build_quant_config
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4DualScaleMixedConfig

    cfg = build_quant_config({"method": "mxfp4_dualscale", "num_bf16_fallback_layers": 10})
    assert isinstance(cfg, DiffusionMXFP4DualScaleMixedConfig)
    assert cfg.num_bf16_fallback_layers == 10


# ---------------------------------------------------------------------------
# Block-index dispatch (_parse_block_idx)
# ---------------------------------------------------------------------------


def test_parse_block_idx_valid():
    from vllm_omni.quantization.mxfp4_config import _parse_block_idx

    assert _parse_block_idx("blocks.0.attn1.to_q") == 0
    assert _parse_block_idx("blocks.5.ffn.net.0.proj") == 5
    assert _parse_block_idx("blocks.40.norm1.weight") == 40


def test_parse_block_idx_non_block_prefixes():
    """Prefixes that do not start with 'blocks.N.' must return None."""
    from vllm_omni.quantization.mxfp4_config import _parse_block_idx

    assert _parse_block_idx("condition_embedder.time_embedder.linear_1") is None
    assert _parse_block_idx("proj_out.weight") is None
    assert _parse_block_idx("model.layers.0.self_attn.q_proj") is None
    assert _parse_block_idx("scale_shift_table") is None


# ---------------------------------------------------------------------------
# SUPPORTED_QUANTIZATION_METHODS
# ---------------------------------------------------------------------------


def test_supported_methods_include_mxfp4_variants():
    from vllm_omni.quantization import SUPPORTED_QUANTIZATION_METHODS

    assert "mxfp4" in SUPPORTED_QUANTIZATION_METHODS
    assert "mxfp8" in SUPPORTED_QUANTIZATION_METHODS
    assert "mxfp4_dualscale" in SUPPORTED_QUANTIZATION_METHODS


# ---------------------------------------------------------------------------
# DiffusionMXFP4DualScaleMixedConfig — config roundtrips
# ---------------------------------------------------------------------------


def test_mixed_dualscale_config_get_name():
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4DualScaleMixedConfig

    assert DiffusionMXFP4DualScaleMixedConfig.get_name() == "mxfp4_dualscale"


def test_mixed_dualscale_config_no_args_defaults():
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4DualScaleMixedConfig

    cfg = DiffusionMXFP4DualScaleMixedConfig()
    assert cfg.is_checkpoint_serialized is False
    assert cfg.ignored_layers == []
    assert cfg.num_bf16_fallback_layers == 5


def test_mixed_dualscale_config_from_config_offline():
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4DualScaleMixedConfig

    cfg = DiffusionMXFP4DualScaleMixedConfig.from_config(
        {
            "quant_method": "mxfp4_dualscale",
            "is_checkpoint_serialized": True,
            "ignored_layers": ["blocks.0.attn1.to_q", "proj_out"],
        }
    )
    assert cfg.is_checkpoint_serialized is True
    assert cfg.ignored_layers == ["blocks.0.attn1.to_q", "proj_out"]
    assert cfg.num_bf16_fallback_layers == 5  # default


def test_mixed_dualscale_config_from_config_online_custom_fallback():
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4DualScaleMixedConfig

    cfg = DiffusionMXFP4DualScaleMixedConfig.from_config({"num_bf16_fallback_layers": 10})
    assert cfg.is_checkpoint_serialized is False
    assert cfg.num_bf16_fallback_layers == 10


def test_mixed_dualscale_config_from_config_modules_to_not_convert_fallback():
    """modules_to_not_convert must be accepted as an alias for ignored_layers."""
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4DualScaleMixedConfig

    cfg = DiffusionMXFP4DualScaleMixedConfig.from_config(
        {"is_checkpoint_serialized": True, "modules_to_not_convert": ["proj_out"]}
    )
    assert cfg.ignored_layers == ["proj_out"]


# ---------------------------------------------------------------------------
# DiffusionMXFP4DualScaleMixedConfig — get_quant_method dispatch
# ---------------------------------------------------------------------------


def test_mixed_dualscale_offline_ignored_layer_returns_unquantized(
    mocker,
    monkeypatch: pytest.MonkeyPatch,
):
    """Offline: a prefix in ignored_layers must return UnquantizedLinearMethod."""
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

    from vllm_omni.platforms import current_omni_platform
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4DualScaleMixedConfig

    cfg = DiffusionMXFP4DualScaleMixedConfig(
        is_checkpoint_serialized=True,
        ignored_layers=["blocks.0.attn1.to_q"],
    )
    layer = mocker.Mock(spec=LinearBase)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)

    method = cfg.get_quant_method(layer, "blocks.0.attn1.to_q")
    assert isinstance(method, UnquantizedLinearMethod)


def test_mixed_dualscale_offline_non_ignored_returns_mxfp4(
    mocker,
    monkeypatch: pytest.MonkeyPatch,
):
    """Offline: a prefix NOT in ignored_layers must return NPUMxfp4DualScaleLinearMethod."""
    from vllm.model_executor.layers.linear import LinearBase

    from vllm_omni.platforms import current_omni_platform
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4DualScaleMixedConfig, NPUMxfp4DualScaleLinearMethod

    cfg = DiffusionMXFP4DualScaleMixedConfig(
        is_checkpoint_serialized=True,
        ignored_layers=["blocks.0.attn1.to_q"],
    )
    layer = mocker.Mock(spec=LinearBase)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)

    method = cfg.get_quant_method(layer, "blocks.1.attn1.to_q")
    assert isinstance(method, NPUMxfp4DualScaleLinearMethod)


def test_mixed_dualscale_online_fallback_block_returns_unquantized(
    mocker,
    monkeypatch: pytest.MonkeyPatch,
):
    """Online: blocks < num_bf16_fallback_layers must return UnquantizedLinearMethod."""
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

    from vllm_omni.platforms import current_omni_platform
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4DualScaleMixedConfig

    cfg = DiffusionMXFP4DualScaleMixedConfig(is_checkpoint_serialized=False, num_bf16_fallback_layers=5)
    layer = mocker.Mock(spec=LinearBase)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)

    assert isinstance(cfg.get_quant_method(layer, "blocks.0.attn1.to_q"), UnquantizedLinearMethod)
    assert isinstance(cfg.get_quant_method(layer, "blocks.4.ffn.net.0.proj"), UnquantizedLinearMethod)


def test_mixed_dualscale_online_quantized_block_returns_mxfp4(
    mocker,
    monkeypatch: pytest.MonkeyPatch,
):
    """Online: blocks >= num_bf16_fallback_layers must return NPUMxfp4DualScaleOnlineLinearMethod."""
    from vllm.model_executor.layers.linear import LinearBase

    from vllm_omni.platforms import current_omni_platform
    from vllm_omni.quantization.mxfp4_config import (
        DiffusionMXFP4DualScaleMixedConfig,
        NPUMxfp4DualScaleOnlineLinearMethod,
    )

    cfg = DiffusionMXFP4DualScaleMixedConfig(is_checkpoint_serialized=False, num_bf16_fallback_layers=5)
    layer = mocker.Mock(spec=LinearBase)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)

    assert isinstance(cfg.get_quant_method(layer, "blocks.5.attn1.to_q"), NPUMxfp4DualScaleOnlineLinearMethod)
    assert isinstance(cfg.get_quant_method(layer, "blocks.40.ffn.net.0.proj"), NPUMxfp4DualScaleOnlineLinearMethod)


def test_mixed_dualscale_online_non_block_prefix_returns_mxfp4(
    mocker,
    monkeypatch: pytest.MonkeyPatch,
):
    """Online: layers outside 'blocks.N.*' (condition_embedder etc.) always use MXFP4 online."""
    from vllm.model_executor.layers.linear import LinearBase

    from vllm_omni.platforms import current_omni_platform
    from vllm_omni.quantization.mxfp4_config import (
        DiffusionMXFP4DualScaleMixedConfig,
        NPUMxfp4DualScaleOnlineLinearMethod,
    )

    cfg = DiffusionMXFP4DualScaleMixedConfig(is_checkpoint_serialized=False, num_bf16_fallback_layers=5)
    layer = mocker.Mock(spec=LinearBase)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)

    method = cfg.get_quant_method(layer, "condition_embedder.time_embedder.linear_1")
    assert isinstance(method, NPUMxfp4DualScaleOnlineLinearMethod)


def test_mixed_dualscale_online_ignored_layers_override(
    mocker,
    monkeypatch: pytest.MonkeyPatch,
):
    """Online: explicit ignored_layers must return UnquantizedLinearMethod regardless of block index.

    A layer that is NOT in the leading-block range (block 10 >= num_bf16_fallback_layers=5)
    but IS listed in ignored_layers must still fall back to BF16.  This lets power users
    pin specific interleaved layers to BF16 during online quantization without needing an
    offline checkpoint.
    """
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

    from vllm_omni.platforms import current_omni_platform
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4DualScaleMixedConfig

    cfg = DiffusionMXFP4DualScaleMixedConfig(
        is_checkpoint_serialized=False,
        num_bf16_fallback_layers=5,
        ignored_layers=["blocks.10.attn1.to_q"],
    )
    layer = mocker.Mock(spec=LinearBase)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)

    # block 10 is above the leading-block threshold but is in ignored_layers → BF16
    assert isinstance(cfg.get_quant_method(layer, "blocks.10.attn1.to_q"), UnquantizedLinearMethod)


def test_mixed_dualscale_non_linear_returns_none(monkeypatch: pytest.MonkeyPatch):
    """Non-LinearBase layers (norms, embeddings) must return None → no quantization."""
    from vllm_omni.platforms import current_omni_platform
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4DualScaleMixedConfig

    cfg = DiffusionMXFP4DualScaleMixedConfig()
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)

    norm_layer = torch.nn.LayerNorm(64)
    assert cfg.get_quant_method(norm_layer, "blocks.0.norm1") is None


# ---------------------------------------------------------------------------
# TP=2 create_weights: parameter shapes and input_dim/output_dim
#
# Two scenarios mirror real Wan2.2 A14B linear layer types:
#   Column-parallel (to_q, ffn.net_0): output is sharded (N/TP), input is full (K).
#   Row-parallel   (to_out, ffn.net_2): input is sharded (K/TP), output is full (N).
#
# Tests verify:
#   1. Registered parameter shapes are correct for each partition configuration.
#   2. input_dim/output_dim attributes are set so RowParallelLinear.weight_loader
#      can shard scale tensors correctly (the fix for the TP>1 shape-mismatch bug).
#   3. Simulated loader slicing: slicing the full checkpoint tensor along the
#      declared input_dim produces the exact shape stored in the parameter —
#      proving the dim declaration is consistent with the allocation.
# ---------------------------------------------------------------------------

# K must be divisible by 32 (fine groups) and 512 (coarse groups).
_TP2_K, _TP2_N, _TP2 = 1024, 512, 2


class _FakeLayer(torch.nn.Module):
    """Bare nn.Module that accepts register_parameter without a real weight_loader."""


def _create_weights(method, *, input_size_per_partition, output_partition_sizes):
    layer = _FakeLayer()
    method.create_weights(
        layer=layer,
        input_size_per_partition=input_size_per_partition,
        output_partition_sizes=output_partition_sizes,
        input_size=_TP2_K,
        output_size=_TP2_N,
        params_dtype=torch.bfloat16,
    )
    return layer


def _shard(tensor, param, rank, tp, dim_attr):
    """Slice `tensor` along the dimension given by `param.<dim_attr>` for `rank`."""
    dim = getattr(param, dim_attr)
    if dim is None:
        return tensor  # not sharded along this axis
    shard_size = param.shape[dim]
    slices = [slice(None)] * tensor.ndim
    slices[dim] = slice(rank * shard_size, (rank + 1) * shard_size)
    return tensor[tuple(slices)]


# ── DualScale method ─────────────────────────────────────────────────────────


def test_dualscale_column_parallel_tp2_shapes():
    """Column-parallel TP=2: output halved, fine/coarse groups stay full, mul_scale full."""
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4DualScaleMixedConfig, NPUMxfp4DualScaleLinearMethod

    method = NPUMxfp4DualScaleLinearMethod(DiffusionMXFP4DualScaleMixedConfig())
    layer = _create_weights(method, input_size_per_partition=_TP2_K, output_partition_sizes=[_TP2_N // _TP2])

    assert layer.weight.shape == (_TP2_N // _TP2, _TP2_K)
    assert layer.weight_scale.shape == (_TP2_N // _TP2, _TP2_K // 32)
    assert layer.weight_dual_scale.shape == (_TP2_N // _TP2, _TP2_K // 512, 1)
    assert layer.mul_scale.shape == (_TP2_K,)


def test_dualscale_row_parallel_tp2_shapes():
    """Row-parallel TP=2: input halved, fine/coarse groups halved, mul_scale halved."""
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4DualScaleMixedConfig, NPUMxfp4DualScaleLinearMethod

    method = NPUMxfp4DualScaleLinearMethod(DiffusionMXFP4DualScaleMixedConfig())
    layer = _create_weights(method, input_size_per_partition=_TP2_K // _TP2, output_partition_sizes=[_TP2_N])

    assert layer.weight.shape == (_TP2_N, _TP2_K // _TP2)
    assert layer.weight_scale.shape == (_TP2_N, (_TP2_K // _TP2) // 32)
    assert layer.weight_dual_scale.shape == (_TP2_N, (_TP2_K // _TP2) // 512, 1)
    assert layer.mul_scale.shape == (_TP2_K // _TP2,)


def test_dualscale_scale_parameter_input_dims():
    """weight_scale/weight_dual_scale must have input_dim=1; mul_scale must have input_dim=0.

    RowParallelLinear.weight_loader only shards a parameter when input_dim is set.
    Without these, loading a full checkpoint tensor into a per-rank shape causes a
    shape mismatch for TP>1 row-parallel layers (to_out, ffn.net_2).
    """
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4DualScaleMixedConfig, NPUMxfp4DualScaleLinearMethod

    method = NPUMxfp4DualScaleLinearMethod(DiffusionMXFP4DualScaleMixedConfig())
    layer = _create_weights(method, input_size_per_partition=_TP2_K, output_partition_sizes=[_TP2_N])

    assert layer.weight_scale.input_dim == 1
    assert layer.weight_scale.output_dim == 0
    assert layer.weight_dual_scale.input_dim == 1
    assert layer.weight_dual_scale.output_dim == 0
    assert layer.mul_scale.input_dim == 0
    assert layer.mul_scale.output_dim is None


def test_dualscale_row_parallel_tp2_loader_simulation():
    """Slicing full checkpoint tensors along input_dim must match row-parallel parameter shapes.

    Simulates what RowParallelLinear.weight_loader does: for each scale parameter,
    take the slice at rank*shard_size:(rank+1)*shard_size along input_dim.
    The resulting shape must equal the per-rank parameter shape allocated by create_weights.
    """
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4DualScaleMixedConfig, NPUMxfp4DualScaleLinearMethod

    method = NPUMxfp4DualScaleLinearMethod(DiffusionMXFP4DualScaleMixedConfig())
    layer = _create_weights(method, input_size_per_partition=_TP2_K // _TP2, output_partition_sizes=[_TP2_N])

    # Full checkpoint tensors (what the loader reads from disk).
    ckpt_weight_scale = torch.zeros(_TP2_N, _TP2_K // 32)
    ckpt_weight_dual_scale = torch.zeros(_TP2_N, _TP2_K // 512, 1)
    ckpt_mul_scale = torch.zeros(_TP2_K)

    for rank in range(_TP2):
        assert _shard(ckpt_weight_scale, layer.weight_scale, rank, _TP2, "input_dim").shape == layer.weight_scale.shape
        assert (
            _shard(ckpt_weight_dual_scale, layer.weight_dual_scale, rank, _TP2, "input_dim").shape
            == layer.weight_dual_scale.shape
        )
        assert _shard(ckpt_mul_scale, layer.mul_scale, rank, _TP2, "input_dim").shape == layer.mul_scale.shape


def test_dualscale_column_parallel_tp2_loader_simulation():
    """Slicing full checkpoint tensors along output_dim must match column-parallel parameter shapes.

    For column-parallel layers, the loader shards along output_dim (rows).
    mul_scale has output_dim=None → not sharded (full tensor, same for all ranks).
    """
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4DualScaleMixedConfig, NPUMxfp4DualScaleLinearMethod

    method = NPUMxfp4DualScaleLinearMethod(DiffusionMXFP4DualScaleMixedConfig())
    layer = _create_weights(method, input_size_per_partition=_TP2_K, output_partition_sizes=[_TP2_N // _TP2])

    ckpt_weight_scale = torch.zeros(_TP2_N, _TP2_K // 32)
    ckpt_weight_dual_scale = torch.zeros(_TP2_N, _TP2_K // 512, 1)
    ckpt_mul_scale = torch.zeros(_TP2_K)

    for rank in range(_TP2):
        assert _shard(ckpt_weight_scale, layer.weight_scale, rank, _TP2, "output_dim").shape == layer.weight_scale.shape
        assert (
            _shard(ckpt_weight_dual_scale, layer.weight_dual_scale, rank, _TP2, "output_dim").shape
            == layer.weight_dual_scale.shape
        )
        # mul_scale: output_dim=None → no sharding → full tensor fits the column-parallel parameter
        assert _shard(ckpt_mul_scale, layer.mul_scale, rank, _TP2, "output_dim").shape == layer.mul_scale.shape


# ── Single-scale method ───────────────────────────────────────────────────────


def test_single_scale_row_parallel_tp2_shapes():
    """Row-parallel TP=2: input halved → weight_scale groups halved."""
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config, NPUMxfp4LinearMethod

    method = NPUMxfp4LinearMethod(DiffusionMXFP4Config())
    layer = _create_weights(method, input_size_per_partition=_TP2_K // _TP2, output_partition_sizes=[_TP2_N])

    assert layer.weight.shape == (_TP2_N, _TP2_K // _TP2)
    assert layer.weight_scale.shape == (_TP2_N, (_TP2_K // _TP2) // 32)


def test_single_scale_scale_parameter_input_dims():
    """Single-scale weight_scale must have input_dim=1 for RowParallel TP sharding."""
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config, NPUMxfp4LinearMethod

    method = NPUMxfp4LinearMethod(DiffusionMXFP4Config())
    layer = _create_weights(method, input_size_per_partition=_TP2_K, output_partition_sizes=[_TP2_N])

    assert layer.weight_scale.input_dim == 1
    assert layer.weight_scale.output_dim == 0


def test_single_scale_row_parallel_tp2_loader_simulation():
    """Slicing full checkpoint weight_scale along input_dim matches row-parallel parameter shape."""
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config, NPUMxfp4LinearMethod

    method = NPUMxfp4LinearMethod(DiffusionMXFP4Config())
    layer = _create_weights(method, input_size_per_partition=_TP2_K // _TP2, output_partition_sizes=[_TP2_N])

    ckpt_weight_scale = torch.zeros(_TP2_N, _TP2_K // 32)

    for rank in range(_TP2):
        assert _shard(ckpt_weight_scale, layer.weight_scale, rank, _TP2, "input_dim").shape == layer.weight_scale.shape


# ---------------------------------------------------------------------------
# ROCm MXFP4 (gfx950) — get_quant_method dispatch
#
# These run on CPU: the platform check, gcnArchName probe and aiter op
# registration are all mocked (stdlib unittest.mock + monkeypatch, so no
# pytest-mock dependency).  They cover only the dispatch + weight-allocation
# logic added by the ROCm PR — the AITER GEMM / quant kernels require real
# gfx950 hardware and are intentionally NOT exercised here.
# ---------------------------------------------------------------------------


@pytest.fixture
def _rocm_platform(monkeypatch: pytest.MonkeyPatch):
    """Make current_omni_platform report ROCm, and stub the aiter
    custom-op registration so ROCmMxfp4*Method can be constructed without aiter."""
    from vllm_omni.platforms import current_omni_platform
    from vllm_omni.quantization import mxfp4_config

    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: False)
    monkeypatch.setattr(current_omni_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(mxfp4_config, "_register_rocm_mxfp4_op", lambda: None)


def _patch_gcn_arch(monkeypatch: pytest.MonkeyPatch, arch: str) -> None:
    """Patch torch.cuda.get_device_properties(...).gcnArchName to return `arch`."""
    from types import SimpleNamespace

    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *a, **k: SimpleNamespace(gcnArchName=arch),
    )


def _fake_linear_layer():
    """A stand-in that passes isinstance(layer, LinearBase) without a real layer."""
    from unittest.mock import MagicMock

    from vllm.model_executor.layers.linear import LinearBase

    return MagicMock(spec=LinearBase)


def test_rocm_online_dispatch_returns_rocm_method(_rocm_platform, monkeypatch):
    """ROCm + gfx950 + online checkpoint must return ROCmMxfp4OnlineLinearMethod."""
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config, ROCmMxfp4OnlineLinearMethod

    _patch_gcn_arch(monkeypatch, "gfx950:sramecc+:xnack-")
    cfg = DiffusionMXFP4Config(is_checkpoint_mxfp4_serialized=False)

    method = cfg.get_quant_method(_fake_linear_layer(), "blocks.0.attn1.to_q")
    assert isinstance(method, ROCmMxfp4OnlineLinearMethod)


@pytest.mark.parametrize("steps", [[], [0, 2]])
def test_rocm_ignored_layer_returns_unquantized(_rocm_platform, monkeypatch, steps):
    """A prefix in ignored_layers must return UnquantizedLinearMethod before the gfx950 probe."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    _patch_gcn_arch(monkeypatch, "gfx950:sramecc+:xnack-")
    cfg = DiffusionMXFP4Config(
        is_checkpoint_mxfp4_serialized=False, ignored_layers=["proj_out"], w4a8_fallback_steps=steps
    )

    assert isinstance(cfg.get_quant_method(_fake_linear_layer(), "proj_out"), UnquantizedLinearMethod)


def test_rocm_rejects_w4a8_step_fallback(_rocm_platform, monkeypatch):
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    _patch_gcn_arch(monkeypatch, "gfx950:sramecc+:xnack-")
    cfg = DiffusionMXFP4Config(w4a8_fallback_steps=[0, 2])

    with pytest.raises(NotImplementedError, match="only supported on NPU"):
        cfg.get_quant_method(_fake_linear_layer(), "blocks.0.attn1.to_q")


def test_rocm_non_gfx950_raises(_rocm_platform, monkeypatch):
    """MXFP4 on ROCm requires gfx950; any other arch must raise."""
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    _patch_gcn_arch(monkeypatch, "gfx942:sramecc+:xnack-")
    cfg = DiffusionMXFP4Config(is_checkpoint_mxfp4_serialized=False)

    with pytest.raises(NotImplementedError, match="gfx950"):
        cfg.get_quant_method(_fake_linear_layer(), "blocks.0.attn1.to_q")


# ---------------------------------------------------------------------------
# ROCm MXFP4 — create_weights (online lazy meta-device placeholder)
#
# The base ROCmMxfp4LinearMethod is abstract (no create_weights); the online
# subclass gets create_weights from _LazyWeightMixin, which registers a BF16
# weight on the meta device to be materialised at load time.
# ---------------------------------------------------------------------------


def test_rocm_create_weights_column_parallel_tp2(_rocm_platform):
    """Column-parallel TP=2: meta BF16 weight has the output halved, input full."""
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config, ROCmMxfp4OnlineLinearMethod

    method = ROCmMxfp4OnlineLinearMethod(DiffusionMXFP4Config())
    layer = _create_weights(method, input_size_per_partition=_TP2_K, output_partition_sizes=[_TP2_N // _TP2])

    assert layer.weight.shape == (_TP2_N // _TP2, _TP2_K)
    assert layer.weight.dtype == torch.bfloat16
    assert layer.weight.device.type == "meta"
    assert layer.weight.input_dim == 1
    assert layer.weight.output_dim == 0
    assert layer.logical_widths == [_TP2_N // _TP2]
    assert layer.weight_block_size is None


def test_rocm_create_weights_row_parallel_tp2(_rocm_platform):
    """Row-parallel TP=2: meta BF16 weight has the input halved, output full."""
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config, ROCmMxfp4OnlineLinearMethod

    method = ROCmMxfp4OnlineLinearMethod(DiffusionMXFP4Config())
    layer = _create_weights(method, input_size_per_partition=_TP2_K // _TP2, output_partition_sizes=[_TP2_N])

    assert layer.weight.shape == (_TP2_N, _TP2_K // _TP2)
    assert layer.input_size_per_partition == _TP2_K // _TP2
    assert layer.output_size_per_partition == _TP2_N


@pytest.mark.parametrize("method", ["mxfp4", "mxfp4_dualscale"])
@pytest.mark.parametrize("layers", ["blocks.1.attn1.to_qkv", [1], [True], [""], [" a"], ["blocks.*"], ["a..b"]])
def test_mxfp4_rejects_invalid_fallback_layers(method, layers):
    from vllm_omni.quantization import build_quant_config

    with pytest.raises(ValueError, match="exact runtime Linear paths"):
        build_quant_config({"method": method, "w4a8_fallback_layers": layers})


@pytest.mark.parametrize("method", ["mxfp4"])
def test_mxfp4_layer_policy_from_checkpoint_and_component_config(method):
    from vllm_omni.quantization import build_quant_config
    from vllm_omni.quantization.component_config import resolve_component_quant_config
    from vllm_omni.quantization.factory import resolve_quant_config_from_disk

    names = ["blocks.10.attn1.to_qkv", "blocks.10.attn1.to_qkv"]
    components = build_quant_config(
        {"transformer": {"method": method, "w4a8_fallback_layers": names}, "transformer_2": {"method": method}}
    )
    names.append("blocks.11.attn1.to_qkv")
    assert resolve_component_quant_config(components, "transformer").w4a8_fallback_layers == [names[0]]
    assert resolve_component_quant_config(components, "transformer_2").w4a8_fallback_layers == []
    detected = resolve_quant_config_from_disk(None, {"quant_method": method, "w4a8_fallback_layers": [names[0]]})
    assert detected.w4a8_fallback_layers == [names[0]]
    assert type(detected).from_config({"w4a8_fallback_layers": [names[0]]}).w4a8_fallback_layers == [names[0]]


@pytest.mark.parametrize(
    "kind, serialized_flag",
    [("mxfp4", "is_checkpoint_mxfp4_serialized")],
)
@pytest.mark.parametrize("offline", [False, True])
@pytest.mark.parametrize("scale_alg", [0, 2])
def test_mxfp4_layer_and_step_union_routes_actual_methods(
    kind, serialized_flag, offline, scale_alg, monkeypatch, mocker
):
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

    from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context
    from vllm_omni.quantization import build_quant_config, mxfp4_config

    monkeypatch.setattr(mxfp4_config.current_omni_platform, "is_npu", lambda: True)
    selected = "blocks.10.attn1.to_qkv"
    other = "blocks.11.attn2.to_q"
    preserved = "blocks.12.ffn.net_2"
    config = build_quant_config(
        {
            "method": kind,
            "mxfp4_scale_alg": scale_alg,
            serialized_flag: offline,
            "w4a8_fallback_layers": [selected, preserved],
            "ignored_layers": [preserved],
            "w4a8_fallback_steps": [1],
        }
    )
    # Real config dispatch creates separate methods for selected and other layers.
    linear = mocker.Mock(spec=LinearBase)
    assert isinstance(config.get_quant_method(linear, preserved), UnquantizedLinearMethod)
    x = torch.ones(2, 64, dtype=torch.bfloat16)
    layer = torch.nn.Module()
    for name in ("weight", "weight_scale", "mul_scale"):
        layer.register_buffer(name, torch.ones(64, dtype=x.dtype))
    calls = []

    def record_a8(*args):
        calls.append("A8")
        return x

    def record_a4(*args):
        calls.append("A4")
        return x

    monkeypatch.setattr(mxfp4_config, "_npu_w4a8_matmul", record_a8)
    for name in (selected, other):
        method = config.get_quant_method(linear, name)
        assert method.w4a8_fallback_layer == (name == selected)
        scales = (x, x)
        monkeypatch.setattr(method, "_quantize_activation", lambda *args: scales)
        monkeypatch.setattr(method, "_quant_matmul", record_a4)
        for step in (0, 1, 2):
            with override_forward_context(ForwardContext(denoise_step_idx=step)):
                method.apply(layer, x)
    assert calls == ["A8", "A8", "A8", "A4", "A8", "A4"]
    # A permanently selected layer works without a request context, even if steps are also configured.
    with override_forward_context(None):
        config.get_quant_method(linear, selected).apply(layer, x)
    assert calls[-1] == "A8"


@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("scale_alg", [0, 2])
def test_single_scale_prepares_one_weight_for_both_activation_precisions(online, selected, scale_alg, monkeypatch):
    import sys
    from types import SimpleNamespace

    from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context
    from vllm_omni.quantization import mxfp4_config

    monkeypatch.setattr(torch.Tensor, "npu", lambda self: self, raising=False)
    prepared = []
    packed = []

    def quantize(x, **kwargs):
        prepared.append(kwargs)
        return x.to(torch.int8), torch.full((x.shape[0], 2), 127, dtype=torch.uint8)

    def pack(x, dtype):
        packed.append(x)
        return x.to(dtype)

    fake_npu = SimpleNamespace(
        float4_e2m1fn_x2=torch.int8,
        float8_e8m0fnu=torch.uint8,
        float8_e4m3fn=torch.float8_e4m3fn,
        npu_dtype_cast=pack,
        npu_dynamic_mx_quant=quantize,
    )
    monkeypatch.setitem(sys.modules, "torch_npu", fake_npu)
    config = mxfp4_config.DiffusionMXFP4Config(
        is_checkpoint_mxfp4_serialized=not online,
        w4a8_fallback_layers=["blocks.0.attn1.to_qkv"] if selected else [],
        w4a8_fallback_steps=[1],
        mxfp4_scale_alg=scale_alg,
    )
    cls = mxfp4_config.NPUMxfp4OnlineLinearMethod if online else mxfp4_config.NPUMxfp4LinearMethod
    method = cls(config, prefix="blocks.0.attn1.to_qkv")
    layer = torch.nn.Linear(64, 2, bias=False, dtype=torch.bfloat16)
    layer.orig_dtype = torch.bfloat16
    if not online:
        layer.register_parameter("weight_scale", torch.nn.Parameter(torch.full((2, 2), 127, dtype=torch.uint8), False))
        layer.register_parameter("mul_scale", torch.nn.Parameter(torch.linspace(0.5, 2, 64), False))
    method.process_weights_after_loading(layer)
    pointers = (layer.weight.data_ptr(), layer.weight_scale.data_ptr())
    method.process_weights_after_loading(layer)
    assert len(prepared) == int(online)
    expected = dict(dst_type=torch.int8, axis=-1, block_size=32, round_mode="rint", scale_alg=scale_alg)
    if scale_alg == 2:
        expected["dst_type_max"] = 7.25
    assert prepared == ([expected] if online else [])
    assert len(packed) == int(not online)
    branches = []

    def a4(x_q, x_scale, target, bias, dtype):
        assert (target.weight.data_ptr(), target.weight_scale.data_ptr()) == pointers
        branches.append("A4")
        return torch.zeros(x_q.shape[0], 2, dtype=dtype)

    def a8(x, weight, scale, bias, dtype):
        assert (weight.data_ptr(), scale.data_ptr()) == pointers
        branches.append("A8")
        return torch.zeros(x.shape[0], 2, dtype=dtype)

    monkeypatch.setattr(method, "_quant_matmul", a4)
    monkeypatch.setattr(mxfp4_config, "_npu_w4a8_matmul", a8)
    monkeypatch.setattr(method, "_quantize_activation", lambda x: (x, x))
    monkeypatch.setattr(fake_npu, "npu_dynamic_mx_quant", lambda *a, **kw: pytest.fail("weight requantization"))
    monkeypatch.setattr(fake_npu, "npu_dtype_cast", lambda *a, **kw: pytest.fail("weight repacking"))
    for step in (0, 1, 2):
        with override_forward_context(ForwardContext(denoise_step_idx=step)):
            method.apply(layer, torch.ones(2, 64, dtype=torch.bfloat16))
    assert branches == (["A8"] * 3 if selected else ["A4", "A8", "A4"])
    assert (layer.weight.data_ptr(), layer.weight_scale.data_ptr()) == pointers
    assert set(layer.state_dict()) == (
        {"weight", "weight_scale"} if online else {"weight", "weight_scale", "mul_scale"}
    )
    assert not hasattr(layer, "w4a8_weight")
    assert not hasattr(layer, "w4a8_weight_scale")


@pytest.mark.parametrize("online", [False, True])
def test_dualscale_without_fallback_keeps_original_a4_path(online, monkeypatch):
    import sys
    from types import SimpleNamespace

    from vllm_omni.quantization import mxfp4_config

    monkeypatch.setattr(torch.Tensor, "npu", lambda self: self, raising=False)
    original_ones = torch.ones
    monkeypatch.setattr(torch, "ones", lambda *args, **kw: original_ones(*args, **{**kw, "device": "cpu"}))
    calls = []

    def quantize_dual(x, **kw):
        calls.append(kw.get("smooth_scale"))
        return x.to(torch.int8), torch.ones(x.shape[0], 1), torch.ones(x.shape[0], 16, dtype=torch.uint8)

    fake_npu = SimpleNamespace(
        float4_e2m1fn_x2=torch.int8,
        npu_dtype_cast=lambda x, dtype: x.to(dtype),
        npu_format_cast=lambda x, *args, **kw: x,
        npu_dynamic_dual_level_mx_quant=quantize_dual,
    )
    monkeypatch.setitem(sys.modules, "torch_npu", fake_npu)
    config = mxfp4_config.DiffusionMXFP4DualScaleMixedConfig(
        is_checkpoint_serialized=not online, w4a8_fallback_steps=[], w4a8_fallback_layers=[]
    )
    cls = mxfp4_config.NPUMxfp4DualScaleOnlineLinearMethod if online else mxfp4_config.NPUMxfp4DualScaleLinearMethod
    method = cls(config)
    layer = torch.nn.Module()
    layer.input_size_per_partition = 512
    for name, value in {
        "weight": torch.ones(2, 512, dtype=torch.bfloat16),
        "weight_scale": torch.full((2, 16), 127, dtype=torch.uint8),
        "weight_dual_scale": torch.ones(2, 1, 1),
        "mul_scale": torch.ones(512),
    }.items():
        layer.register_parameter(name, torch.nn.Parameter(value, requires_grad=False))
    method.process_weights_after_loading(layer)
    method.process_weights_after_loading(layer)
    assert len(calls) == int(online)
    assert set(layer.state_dict()) == {"weight", "weight_scale", "weight_dual_scale", "mul_scale"}
    assert not hasattr(layer, "w4a8_weight")
    monkeypatch.setattr(method, "_quant_matmul", lambda *args: torch.zeros(2, 2, dtype=torch.bfloat16))
    method.apply(layer, torch.ones(2, 512, dtype=torch.bfloat16))
    assert len(calls) == 1 + int(online)
    torch.testing.assert_close(calls[-1], layer.mul_scale)
