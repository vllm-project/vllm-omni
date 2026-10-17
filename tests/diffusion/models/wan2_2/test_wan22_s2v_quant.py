# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for quantized Wan2.2 S2V checkpoints.

Tests cover:
- quant_config reaches every linear the S2V transformer block owns
- create_s2v_transformer_from_config forwards quant_config
- Wan22S2VPipeline resolves the checkpoint's quantization contract, including
  the sibling quantization_config.json of an original-format checkpoint
- load_weights folds an MXFP8 scale into a layer that stays plain nn.Linear
- load_weights refuses a scale it cannot apply instead of dropping it
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from pytest_mock import MockerFixture

import vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_s2v as s2v_pipeline_module
from vllm_omni.diffusion.models.wan2_2 import wan2_2_s2v_transformer as s2v
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_s2v import Wan22S2VPipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_MODULE = "vllm_omni.diffusion.models.wan2_2.wan2_2_s2v_transformer"


def _mxfp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to E4M3 values with one biased power-of-two scale per group."""
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import MXFP8_BLOCK_SIZE

    groups = weight.unflatten(-1, (-1, MXFP8_BLOCK_SIZE))
    exponent = torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-30) / 448.0))
    scale = (exponent + 127).clamp(0, 254).to(torch.uint8)
    values = (groups / torch.exp2(exponent).unsqueeze(-1)).flatten(-2).to(torch.float8_e4m3fn)
    return values, scale


# ---------------------------------------------------------------------------
# quant_config propagation: the root cause of a quantized S2V serving noise
# ---------------------------------------------------------------------------


class _FakeLinear(nn.Module):
    """Stand-in exposing the head counts WanS2VSelfAttention reads back."""

    def __init__(self, **kwargs):
        super().__init__()
        self.num_heads = kwargs.get("total_num_heads", 1)
        self.num_kv_heads = self.num_heads


class _Capture:
    """Stand-in for a layer class that records its construction kwargs."""

    def __init__(self, sink: dict):
        self.sink = sink

    def __call__(self, *args, **kwargs):
        self.sink[kwargs.get("prefix", f"positional_{len(self.sink)}")] = kwargs
        return _FakeLinear(**kwargs)


def test_transformer_block_forwards_quant_config_to_every_submodule(mocker: MockerFixture) -> None:
    captured: dict[str, dict] = {}
    for layer in ("WanS2VSelfAttention", "WanS2VCrossAttention", "WanFeedForward"):
        mocker.patch.object(s2v, layer, _Capture(captured))
    quant_config = mocker.MagicMock()

    s2v.WanS2VTransformerBlock(8, 16, 2, quant_config=quant_config, prefix="blocks.0")

    assert set(captured) == {"blocks.0.self_attn", "blocks.0.cross_attn", "blocks.0.ffn"}
    assert all(kwargs["quant_config"] is quant_config for kwargs in captured.values())


@pytest.mark.parametrize(
    ("attention_cls", "expected"),
    [
        (s2v.WanS2VSelfAttention, {"attn.to_qkv", "attn.to_out"}),
        (s2v.WanS2VCrossAttention, {"attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out"}),
    ],
)
def test_attention_forwards_quant_config_to_every_linear(
    mocker: MockerFixture, attention_cls: type, expected: set[str]
) -> None:
    captured: dict[str, dict] = {}
    for layer in ("QKVParallelLinear", "ColumnParallelLinear", "RowParallelLinear"):
        mocker.patch.object(s2v, layer, _Capture(captured))
    mocker.patch.object(s2v, "Attention", lambda **kwargs: nn.Identity())
    mocker.patch.object(s2v, "DistributedRMSNorm", lambda *args, **kwargs: nn.Identity())
    mocker.patch(f"{_MODULE}.get_tensor_model_parallel_world_size", return_value=1)
    quant_config = mocker.MagicMock()

    attention_cls(8, 2, quant_config=quant_config, prefix="attn")

    assert set(captured) == expected
    assert all(kwargs["quant_config"] is quant_config for kwargs in captured.values())


def test_create_s2v_transformer_from_config_forwards_quant_config(mocker: MockerFixture) -> None:
    captured: dict = {}

    class FakeTransformer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    mocker.patch.object(s2v, "WanS2VTransformer3DModel", FakeTransformer)
    quant_config = mocker.MagicMock()

    s2v.create_s2v_transformer_from_config({"dim": 8}, quant_config=quant_config)
    assert captured["quant_config"] is quant_config

    captured.clear()
    s2v.create_s2v_transformer_from_config({"dim": 8})
    assert "quant_config" not in captured


# ---------------------------------------------------------------------------
# Pipeline: resolving the checkpoint's quantization contract
# ---------------------------------------------------------------------------


def _pipeline_with_od_config(**od_config) -> Wan22S2VPipeline:
    pipeline = object.__new__(Wan22S2VPipeline)
    pipeline.od_config = SimpleNamespace(
        quantization_config=od_config.get("quantization_config"),
        quantization_config_is_auto_detected=od_config.get("auto_detected", False),
    )
    return pipeline


@pytest.mark.parametrize("subfolder", [".", "transformer"])
def test_create_transformer_reads_sibling_quantization_config(tmp_path, mocker: MockerFixture, subfolder: str) -> None:
    disk_qc = {"quant_method": "auto-round", "bits": 8, "data_type": "mx_fp", "group_size": 32}
    (tmp_path / subfolder).mkdir(exist_ok=True)
    (tmp_path / subfolder / "quantization_config.json").write_text(json.dumps(disk_qc))
    resolved = object()
    resolve = mocker.patch.object(s2v_pipeline_module, "resolve_wan_transformer_quant_config", return_value=resolved)
    create = mocker.patch.object(s2v_pipeline_module, "create_s2v_transformer_from_config")

    _pipeline_with_od_config()._create_transformer({"dim": 8}, str(tmp_path), subfolder)

    config, quant_config, component = resolve.call_args.args
    assert config["quantization_config"] == disk_qc
    assert quant_config is None
    assert component == "transformer"
    assert create.call_args.kwargs["quant_config"] is resolved


def test_create_transformer_prefers_the_config_json_contract(tmp_path, mocker: MockerFixture) -> None:
    (tmp_path / "quantization_config.json").write_text(json.dumps({"quant_method": "sibling"}))
    resolve = mocker.patch.object(s2v_pipeline_module, "resolve_wan_transformer_quant_config")
    mocker.patch.object(s2v_pipeline_module, "create_s2v_transformer_from_config")

    _pipeline_with_od_config()._create_transformer(
        {"dim": 8, "quantization_config": {"quant_method": "in-config"}}, str(tmp_path), "."
    )

    assert resolve.call_args.args[0]["quantization_config"] == {"quant_method": "in-config"}


def test_create_transformer_drops_auto_detected_startup_metadata(tmp_path, mocker: MockerFixture) -> None:
    resolve = mocker.patch.object(s2v_pipeline_module, "resolve_wan_transformer_quant_config")
    mocker.patch.object(s2v_pipeline_module, "create_s2v_transformer_from_config")
    startup_qc = object()

    _pipeline_with_od_config(quantization_config=startup_qc, auto_detected=True)._create_transformer(
        {"dim": 8}, str(tmp_path), "."
    )

    assert resolve.call_args.args[1] is None


# ---------------------------------------------------------------------------
# load_weights: MXFP8 scales for layers that stay plain nn.Linear
# ---------------------------------------------------------------------------

_ADAIN = "audio_injector.injector_adain_layers.0.linear"


class _AdainLinear(nn.Module):
    """Parameter layout of an unquantized ``injector_adain_layers.0.linear``."""

    def __init__(self) -> None:
        super().__init__()
        self.audio_injector = nn.Module()
        self.audio_injector.injector_adain_layers = nn.ModuleList([nn.Module()])
        self.audio_injector.injector_adain_layers[0].linear = nn.Linear(64, 4, dtype=torch.bfloat16)

    @property
    def weight(self) -> torch.Tensor:
        return self.audio_injector.injector_adain_layers[0].linear.weight


class _FusedSelfAttention(nn.Module):
    """Parameter layout of an unquantized fused ``self_attn.to_qkv``."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Module()])
        self.blocks[0].self_attn = nn.Module()
        self.blocks[0].self_attn.to_qkv = nn.Linear(64, 192, dtype=torch.bfloat16)
        # A real QKVParallelLinear parameter carries a sharding weight_loader.
        self.blocks[0].self_attn.to_qkv.weight.weight_loader = lambda param, weight, shard_id: None


def _load(module: nn.Module, weights: list[tuple[str, torch.Tensor]]) -> set[str]:
    with (
        patch(f"{_MODULE}.get_tensor_model_parallel_rank", return_value=0),
        patch(f"{_MODULE}.get_tensor_model_parallel_world_size", return_value=1),
    ):
        return s2v.WanS2VTransformer3DModel.load_weights(module, weights)


@pytest.mark.parametrize("scale_first", [False, True])
@pytest.mark.parametrize("store_as_bf16", [False, True])
def test_load_weights_folds_mxfp8_scale_into_a_plain_linear(scale_first: bool, store_as_bf16: bool) -> None:
    module = _AdainLinear()
    values, scale = _mxfp8(torch.randn(4, 64))
    # AutoRound stores the E4M3 values either as float8 or as the bf16 they
    # round-trip to; both must end up scaled.
    stored = values.to(torch.bfloat16) if store_as_bf16 else values
    tensors = [(f"{_ADAIN}.weight", stored), (f"{_ADAIN}.weight_scale", scale)]
    if scale_first:
        tensors.reverse()

    loaded = _load(module, tensors)

    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import dequant_mxfp8_to_bf16

    torch.testing.assert_close(module.weight.data, dequant_mxfp8_to_bf16(values, scale))
    assert {f"{_ADAIN}.weight", f"{_ADAIN}.weight_scale"} <= loaded


def test_load_weights_rejects_a_scale_whose_weight_never_arrives() -> None:
    _, scale = _mxfp8(torch.randn(4, 64))

    with pytest.raises(ValueError, match="no weight for the MXFP8 scales"):
        _load(_AdainLinear(), [(f"{_ADAIN}.weight_scale", scale)])


def test_load_weights_rejects_a_scale_dtype_it_cannot_read() -> None:
    values, scale = _mxfp8(torch.randn(4, 64))
    tensors = [(f"{_ADAIN}.weight", values), (f"{_ADAIN}.weight_scale", scale.to(torch.float32))]

    with pytest.raises(ValueError, match="expected MXFP8 exponents"):
        _load(_AdainLinear(), tensors)


def test_load_weights_refuses_a_quantized_checkpoint_for_an_unquantized_fused_layer() -> None:
    values, scale = _mxfp8(torch.randn(64, 64))
    tensors = [
        ("blocks.0.self_attn.q.weight", values),
        ("blocks.0.self_attn.q.weight_scale", scale),
    ]

    with pytest.raises(ValueError, match="built without a quantization config"):
        _load(_FusedSelfAttention(), tensors)
