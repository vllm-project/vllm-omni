# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L1 tests for the FP8 W8A16 blockwise config + weight-only linear method.

The tests cover target selection, recipe-gated dispatch, resident weight
allocation, and per-op dequantized GEMM behavior.
"""

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.quantization import fp8_blockwise_w8a16 as w8
from vllm_omni.quantization.fp8_blockwise_w8a16 import Fp8BlockwiseW8A16LinearMethod

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _method(block=(128, 128)):
    return Fp8BlockwiseW8A16LinearMethod(SimpleNamespace(weight_block_size=list(block)))


# --- target selection (Gated construction) ----------------------------------


@pytest.mark.parametrize(
    "prefix",
    [
        "language_model.layers.0.mlp.gate_proj",
        "language_model.layers.5.mlp.up_proj",
        "language_model.layers.35.mlp.down_proj",
        "gen_layers.0.mlp_moe_gen.gate_proj",
        "gen_layers.12.mlp_moe_gen.up_proj",
        "gen_layers.35.mlp_moe_gen.down_proj",
    ],
)
def test_is_target_prefix_matches_mlp(prefix):
    assert w8.is_target_prefix(prefix) is True


@pytest.mark.parametrize(
    "prefix",
    [
        "lm_head",
        "language_model.layers.0.self_attn.to_q",
        "language_model.layers.0.self_attn.to_out",
        "proj_in",
        "proj_out",
        "time_embedder",
        "embed_tokens",
        "action_proj_in.fc",
        # a parameter name, not a module prefix -> must NOT match.
        "language_model.layers.0.mlp.gate_proj.weight",
    ],
)
def test_is_target_prefix_excludes_nontargets(prefix):
    assert w8.is_target_prefix(prefix) is False


# --- gated build -------------------------------------------------------------


def test_maybe_build_disabled_returns_active_unchanged():
    sentinel = SimpleNamespace(name="active")
    assert w8.maybe_build_fp8_blockwise_w8a16_config(False, sentinel) is sentinel
    assert w8.maybe_build_fp8_blockwise_w8a16_config(False, None) is None


def test_maybe_build_disabled_does_not_call_build(monkeypatch):
    def boom():
        raise AssertionError("build must not be called when disabled")

    monkeypatch.setattr(w8, "build_fp8_blockwise_w8a16_config", boom)
    assert w8.maybe_build_fp8_blockwise_w8a16_config(False, None) is None


def test_maybe_build_enabled_builds_target_inclusion_config():
    cfg = w8.maybe_build_fp8_blockwise_w8a16_config(True, None)
    assert cfg.is_target_module("language_model.layers.0.mlp.gate_proj")
    assert cfg.is_target_module("gen_layers.0.mlp_moe_gen.down_proj")
    # target-inclusion: everything else is excluded -> Unquantized (BF16)
    assert cfg.is_layer_excluded("lm_head")
    assert cfg.is_layer_excluded("language_model.layers.0.self_attn.to_q")
    assert not cfg.is_layer_excluded("language_model.layers.0.mlp.gate_proj")
    assert cfg.LinearMethodCls.__name__ == "Fp8BlockwiseW8A16LinearMethod"
    assert list(cfg.weight_block_size) == [128, 128]


def test_maybe_build_enabled_never_overrides_active_config():
    # e.g. the NVFP4 W4A16 config already built by the NVFP4 hook must be kept.
    nvfp4 = SimpleNamespace(get_name=lambda: "modelopt_fp4")
    assert w8.maybe_build_fp8_blockwise_w8a16_config(True, nvfp4) is nvfp4


# --- recipe-gated default selection -----------------------------------------


def _sidecar_dir(root, recipe="fp8_blockwise_mixed"):
    """Write a minimal root quantization_config.json and return the dir path."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "quantization_config.json").write_text(json.dumps({"recipe": recipe}))
    return str(root)


def test_fp8_w8a16_forced(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_FP8_BLOCKWISE_W8A16", raising=False)
    assert w8.fp8_w8a16_forced() is False
    monkeypatch.setenv("VLLM_OMNI_FP8_BLOCKWISE_W8A16", "1")
    assert w8.fp8_w8a16_forced() is True
    monkeypatch.setenv("VLLM_OMNI_FP8_BLOCKWISE_W8A16", "0")  # only "1" opts in
    assert w8.fp8_w8a16_forced() is False


def test_fp8_w8a16_selected_default_off_for_fp8_blockwise(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_FP8_BLOCKWISE_W8A16", raising=False)
    assert w8.fp8_w8a16_selected(_sidecar_dir(tmp_path / "fp8")) is False


def test_fp8_w8a16_selected_explicit_opt_in(tmp_path, monkeypatch):
    root = _sidecar_dir(tmp_path / "fp8")
    monkeypatch.setenv("VLLM_OMNI_FP8_BLOCKWISE_W8A16", "1")
    assert w8.fp8_w8a16_selected(root) is True


def test_omni_diffusion_config_opt_in_wires_fp8_w8a16(tmp_path, monkeypatch):
    from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig

    root = _sidecar_dir(tmp_path / "fp8")
    monkeypatch.setenv("VLLM_OMNI_FP8_BLOCKWISE_W8A16", "1")

    config = OmniDiffusionConfig(model=root, tf_model_config=TransformerConfig.from_dict({}))

    assert config.quantization_config is not None
    assert config.quantization_config.LinearMethodCls.__name__ == "Fp8BlockwiseW8A16LinearMethod"


def test_fp8_w8a16_selected_declines_non_fp8_blockwise(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FP8_BLOCKWISE_W8A16", "1")
    # foreign recipe (mirrors NVFP4) -> declines (predicate is False, no raise here)
    assert w8.fp8_w8a16_selected(_sidecar_dir(tmp_path / "nv", recipe="nvfp4_blockwise_mixed_v1")) is False
    # missing sidecar (plain BF16 dir) -> declines
    (tmp_path / "bf16").mkdir()
    assert w8.fp8_w8a16_selected(str(tmp_path / "bf16")) is False
    # no path -> declines
    assert w8.fp8_w8a16_selected(None) is False


# --- linear method: residency, weight-only, JIT dequant ----------------------


def test_create_weights_registers_resident_fp8_and_block_scale(monkeypatch):
    # vLLM Parameters read the TP rank/size at construction; pin single-rank so this
    # stays a hermetic CPU test (no distributed init).
    import vllm.model_executor.parameter as vp

    monkeypatch.setattr(vp, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(vp, "get_tensor_model_parallel_world_size", lambda: 1)
    method = _method()
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=128,
        output_partition_sizes=[256],
        input_size=128,
        output_size=256,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *a, **k: None,
    )
    # resident FP8 weight (1 byte/elem), NOT dequantized to BF16
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight.element_size() == 1
    assert tuple(layer.weight.shape) == (256, 128)
    # 2D block scale grid [ceil(256/128), ceil(128/128)] = [2, 1], bf16
    assert layer.weight_scale.dtype == torch.bfloat16
    assert tuple(layer.weight_scale.shape) == (2, 1)
    # weight-only: no activation input_scale
    assert not hasattr(layer, "input_scale")
    assert layer.weight_block_size == [128, 128]


def test_apply_dequantizes_per_op_then_gemm():
    from vllm_omni.diffusion.model_loader.checkpoint_adapters.modelopt_native import (
        dequantize_weight,
    )

    method = _method()
    out, inn, batch = 128, 128, 4
    w_fp8 = (torch.randn(out, inn) * 0.1).to(torch.float8_e4m3fn)
    scale = (torch.rand(1, 1) + 0.5).to(torch.bfloat16)
    x = torch.randn(batch, inn, dtype=torch.bfloat16)
    layer = SimpleNamespace(weight=w_fp8, weight_scale=scale, weight_block_size=[128, 128])

    got = method.apply(layer, x)
    ref_w = dequantize_weight(w_fp8, scale, x.dtype, (128, 128))
    ref = F.linear(x, ref_w)

    assert tuple(got.shape) == (batch, out)  # no transpose bug
    torch.testing.assert_close(got, ref)


def test_process_weights_after_loading_passes_on_fp8_with_valid_scale_grid():
    method = _method()
    # 256x128, block 128x128 -> scale grid (ceil(256/128), ceil(128/128)) = (2, 1)
    layer = SimpleNamespace(
        weight=torch.zeros(256, 128).to(torch.float8_e4m3fn),
        weight_scale=torch.ones(2, 1, dtype=torch.bfloat16),
        output_size_per_partition=256,
        input_size_per_partition=128,
        weight_block_size=[128, 128],
    )
    method.process_weights_after_loading(layer)  # must not raise


def test_process_weights_after_loading_rejects_bf16_silent_dequant():
    method = _method()
    layer = SimpleNamespace(
        weight=torch.zeros(128, 128, dtype=torch.bfloat16),
        weight_block_size=[128, 128],
    )
    # residency check fires before the scale-grid check (no scale attrs needed)
    with pytest.raises(ValueError, match="residency"):
        method.process_weights_after_loading(layer)


def test_process_weights_after_loading_rejects_wrong_scale_grid():
    method = _method()
    # transposed grid (1, 2) != expected (2, 1) -> malformed scale rejected
    layer = SimpleNamespace(
        weight=torch.zeros(256, 128).to(torch.float8_e4m3fn),
        weight_scale=torch.ones(1, 2, dtype=torch.bfloat16),
        output_size_per_partition=256,
        input_size_per_partition=128,
        weight_block_size=[128, 128],
    )
    with pytest.raises(ValueError, match="scale-grid"):
        method.process_weights_after_loading(layer)
