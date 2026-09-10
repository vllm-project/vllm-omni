# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise Wan's real name mapping and vLLM's QKV parameter loaders on CPU."""

import pytest
import torch
from vllm.config.load import LoadConfig
from vllm.model_executor.layers.linear import QKVParallelLinear

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformer3DModel
from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config, DiffusionMXFP4DualScaleMixedConfig

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _patch_tp_state(monkeypatch):
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr("vllm_omni.platforms.current_omni_platform.is_npu", lambda: True)
    prefix = "vllm_omni.diffusion.models.wan2_2.wan2_2_transformer"
    monkeypatch.setattr(f"{prefix}.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(f"{prefix}.get_tensor_model_parallel_world_size", lambda: 1)


class _WanCheckpoint(torch.nn.Module):
    load_weights = WanTransformer3DModel.load_weights
    _log_w4a8_fallback_load_summary = WanTransformer3DModel._log_w4a8_fallback_load_summary

    def __init__(self, config, *, num_blocks=1):
        super().__init__()
        blocks = []
        for block_idx in range(num_blocks):
            block = torch.nn.Module()
            block.attn1 = torch.nn.Module()
            block.attn1.to_qkv = QKVParallelLinear(
                hidden_size=512,
                head_size=32,
                total_num_heads=4,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=config,
                prefix=f"blocks.{block_idx}.attn1.to_qkv",
                disable_tp=True,
            )
            blocks.append(block)
        self.blocks = torch.nn.ModuleList(blocks)


def _checkpoint(mode, fused):
    if mode == "single":
        config = DiffusionMXFP4Config(is_checkpoint_mxfp4_serialized=True, require_smooth_scale=True)
    elif mode == "dual":
        config = DiffusionMXFP4DualScaleMixedConfig(is_checkpoint_serialized=True)
    else:
        config = None
    model = _WanCheckpoint(config)
    weights = {}
    expected = {}
    for name, param in model.named_parameters():
        suffix = name.rsplit(".", 1)[1]
        output_dim = getattr(param, "output_dim", None)
        if output_dim is None:
            value = torch.full_like(param, 2.5)
            shards = [value] * 3
        else:
            parts = param.chunk(3, dim=output_dim)
            shards = [
                torch.full_like(part, (121 if suffix == "weight_scale" else 1) + i) for i, part in enumerate(parts)
            ]
            value = torch.cat(shards, dim=output_dim)
        expected[name] = value
        if fused:
            weights[name] = value
        else:
            for shard_id, shard in zip(("q", "k", "v"), shards):
                weights[f"blocks.0.attn1.to_{shard_id}.{suffix}"] = shard
    return model, config, weights, expected


def _load(model, config, weights, monkeypatch):
    loader = DiffusersPipelineLoader(
        LoadConfig(), OmniDiffusionConfig(model="", dtype=torch.bfloat16, quantization_config=config)
    )
    loader.counter_before_loading_weights = 0.0
    monkeypatch.setattr(loader, "get_all_weights", lambda model: iter(weights.items()))
    loader.load_weights(model)


@pytest.mark.parametrize("mode", ["single", "dual", "bf16"])
@pytest.mark.parametrize("fused", [False, True])
def test_wan_qkv_loads_actual_weights_and_scales(mode, fused, monkeypatch):
    model, config, weights, expected = _checkpoint(mode, fused)
    _load(model, config, weights, monkeypatch)
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["single", "dual", "bf16"])
def test_wan_incomplete_qkv_weight_is_not_marked_fully_loaded(mode, monkeypatch):
    model, config, weights, _ = _checkpoint(mode, fused=False)
    weights.pop("blocks.0.attn1.to_k.weight")
    weights.pop("blocks.0.attn1.to_v.weight")
    with pytest.raises(ValueError, match="to_qkv.weight"):
        _load(model, config, weights, monkeypatch)


@pytest.mark.parametrize("mode", ["single", "dual"])
def test_wan_incomplete_qkv_scale_is_not_marked_fully_loaded(mode, monkeypatch):
    model, config, weights, _ = _checkpoint(mode, fused=False)
    weights.pop("blocks.0.attn1.to_k.weight_scale")
    weights.pop("blocks.0.attn1.to_v.weight_scale")
    with pytest.raises(ValueError, match="to_qkv.weight_scale"):
        _load(model, config, weights, monkeypatch)


@pytest.mark.parametrize("mode", ["single", "dual"])
@pytest.mark.parametrize("reverse", [False, True])
def test_wan_split_qkv_rejects_different_smooth_scales(mode, reverse, monkeypatch):
    model, config, weights, _ = _checkpoint(mode, fused=False)
    weights["blocks.0.attn1.to_k.mul_scale"] = weights["blocks.0.attn1.to_k.mul_scale"] * 2
    if reverse:
        weights = dict(reversed(weights.items()))
    with pytest.raises(ValueError, match="must share the same Smooth tensor"):
        _load(model, config, weights, monkeypatch)


@pytest.mark.parametrize("name", ["to_q.missing", "to_qkv.missing", "to_qkv_extra.weight", "to_query.weight"])
def test_wan_unknown_qkv_key_is_not_reported_as_loaded(name):
    model, _, _, _ = _checkpoint("single", fused=True)
    loaded = model.load_weights([(f"blocks.0.attn1.{name}", torch.ones(2))])
    assert loaded == set()


@pytest.mark.parametrize("fused", [False, True])
def test_wan_single_scale_rejects_relabelled_dualscale_checkpoint(fused, monkeypatch):
    model, config, weights, _ = _checkpoint("single", fused=fused)
    projection = "to_qkv" if fused else "to_q"
    weights[f"blocks.0.attn1.{projection}.weight_dual_scale"] = torch.ones(128, 1, 1)
    with pytest.raises(ValueError, match="cannot load DualScale tensor"):
        _load(model, config, weights, monkeypatch)


def test_wan_layer_policy_validates_runtime_qkv_and_bf16_override():
    config = DiffusionMXFP4Config(w4a8_fallback_layers=["blocks.0.attn1.to_qkv"])
    model = _WanCheckpoint(config)
    # Uses the same validation called at the end of WanTransformer3DModel construction.
    WanTransformer3DModel._validate_w4a8_fallback_layers(model, config)
    for bad in ("blocks.0.attn1.to_q", "blocks.0.attn1", "blocks.99.attn1.to_qkv", "transformer.blocks.0.attn1.to_qkv"):
        config.w4a8_fallback_layers = [bad]
        with pytest.raises(ValueError, match="exact runtime Linear paths"):
            WanTransformer3DModel._validate_w4a8_fallback_layers(model, config)


def test_wan_layer_policy_defers_only_other_pp_stage_paths():
    from vllm.model_executor.models.utils import PPMissingLayer

    config = DiffusionMXFP4Config(w4a8_fallback_layers=["blocks.0.attn1.to_qkv", "blocks.1.attn1.to_qkv"])
    model = _WanCheckpoint(config)
    model.blocks.append(PPMissingLayer())
    WanTransformer3DModel._validate_w4a8_fallback_layers(model, config)
    config.w4a8_fallback_layers = ["blocks.1"]
    with pytest.raises(ValueError, match="exact runtime Linear paths"):
        WanTransformer3DModel._validate_w4a8_fallback_layers(model, config)


@pytest.mark.parametrize("is_checkpoint_serialized", [False, True], ids=["online", "offline"])
def test_loader_logs_wan_expert_summaries_after_weight_processing(is_checkpoint_serialized, mocker):
    selected = "blocks.0.attn1.to_qkv"
    bf16 = "blocks.1.attn1.to_qkv"
    transformer_config = DiffusionMXFP4Config(
        is_checkpoint_mxfp4_serialized=is_checkpoint_serialized,
        w4a8_fallback_layers=[selected],
    )
    transformer_2_config = DiffusionMXFP4Config(
        is_checkpoint_mxfp4_serialized=is_checkpoint_serialized,
        ignored_layers=[bf16],
        w4a8_fallback_layers=[selected, bf16],
    )
    pipeline = torch.nn.Module()
    pipeline.transformer = _WanCheckpoint(transformer_config)
    pipeline.transformer_2 = _WanCheckpoint(transformer_2_config, num_blocks=2)
    WanTransformer3DModel._validate_w4a8_fallback_layers(pipeline.transformer, transformer_config)
    WanTransformer3DModel._validate_w4a8_fallback_layers(pipeline.transformer_2, transformer_2_config)

    loader = DiffusersPipelineLoader(
        LoadConfig(),
        OmniDiffusionConfig(model="", dtype=torch.bfloat16, quantization_config=transformer_config),
    )
    events = []
    summaries = []

    def process_weights(model, _target_device):
        events.append("process")
        for module in model.modules():
            method = getattr(module, "quant_method", None)
            if getattr(method, "w4a8_fallback_layer", False):
                module._already_called_process_weights_after_loading = True

    def capture_summary(message, *args):
        rendered = message % args
        summaries.append(rendered)
        events.append(f"summary:{args[0]}")

    loader._init_from_load_format = lambda *_args, **_kwargs: pipeline  # type: ignore[method-assign]
    loader.load_weights = lambda _model: events.append("load")  # type: ignore[method-assign]
    loader._process_weights_after_loading = process_weights  # type: ignore[method-assign]
    loader._apply_skip_softmax_calibration = lambda _model: None  # type: ignore[method-assign]
    mocker.patch(
        "vllm_omni.diffusion.models.wan2_2.wan2_2_transformer.logger.info",
        side_effect=capture_summary,
    )

    assert loader.load_model(load_device="cpu") is pipeline
    assert events == ["load", "process", "summary:transformer", "summary:transformer_2"]
    assert "component=transformer; selected=1 ['transformer.blocks.0.attn1.to_qkv']" in summaries[0]
    assert "BF16 overrides=0 []" in summaries[0]
    assert "weight/A8 processing state=ready (1/1 selected layers processed)" in summaries[0]
    assert "component=transformer_2; selected=1 ['transformer_2.blocks.0.attn1.to_qkv']" in summaries[1]
    assert "BF16 overrides=1 ['transformer_2.blocks.1.attn1.to_qkv']" in summaries[1]
    assert "weight/A8 processing state=ready (1/1 selected layers processed)" in summaries[1]


def test_wan_load_summary_does_not_claim_unprocessed_layer_is_ready(mocker):
    selected = "blocks.0.attn1.to_qkv"
    config = DiffusionMXFP4Config(w4a8_fallback_layers=[selected])
    model = _WanCheckpoint(config)
    WanTransformer3DModel._validate_w4a8_fallback_layers(model, config)
    logger_info = mocker.patch("vllm_omni.diffusion.models.wan2_2.wan2_2_transformer.logger.info")

    WanTransformer3DModel._log_w4a8_fallback_load_summary(model, "transformer")

    message, *args = logger_info.call_args.args
    rendered = message % tuple(args)
    assert "weight/A8 processing state=not-ready (0/1 selected layers processed)" in rendered
