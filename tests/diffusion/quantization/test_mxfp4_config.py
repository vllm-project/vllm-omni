# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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


def test_mxfp4_config_from_config_defaults():
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    cfg = DiffusionMXFP4Config.from_config({})
    assert cfg.is_checkpoint_mxfp4_serialized is False
    assert cfg.ignored_layers == []


def test_mxfp4_config_from_config_serialized():
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    cfg = DiffusionMXFP4Config.from_config({"is_checkpoint_mxfp4_serialized": True})
    assert cfg.is_checkpoint_mxfp4_serialized is True


def test_mxfp4_config_from_config_ignored_layers():
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    cfg = DiffusionMXFP4Config.from_config({"ignored_layers": ["proj_out"]})
    assert cfg.ignored_layers == ["proj_out"]


def test_mxfp4_config_from_config_modules_to_not_convert_fallback():
    """modules_to_not_convert must be accepted as an alias for ignored_layers."""
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    cfg = DiffusionMXFP4Config.from_config({"modules_to_not_convert": ["proj_out"]})
    assert cfg.ignored_layers == ["proj_out"]


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
