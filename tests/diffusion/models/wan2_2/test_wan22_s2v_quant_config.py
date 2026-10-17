# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for Wan2.2 S2V quant_config propagation.

``--quantization`` used to be silently ignored for Wan22S2VPipeline: every S2V
transformer linear was built with ``quant_config=None``. These tests cover:

- both checkpoint layouts (diffusers / original) hand the resolved quant config
  to the S2V transformer, and the default (no quantization) path stays BF16
- auto-detected startup metadata defers to the checkpoint's own config.json
- the quant config reaches every transformer-block linear with its module path
  as prefix, while the audio injector, embedders and proj_out stay unquantized
- the transformer binds its packed QKV mapping to the quant config, so a real
  ignored_layers skip policy on to_q/to_k/to_v covers the fused to_qkv

All tests are pure-CPU and do not load model weights.
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped

import vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_s2v as s2v_pipeline_module
import vllm_omni.diffusion.models.wan2_2.wan2_2_s2v_transformer as s2v_transformer_module
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_s2v import Wan22S2VPipeline
from vllm_omni.diffusion.models.wan2_2.wan2_2_s2v_transformer import (
    WanS2VTransformer3DModel,
    create_s2v_transformer_from_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _RecordingQuantConfig(QuantizationConfig):
    """Records the prefix of every linear layer it is asked to quantize."""

    def __init__(self) -> None:
        super().__init__()
        self.prefixes: list[str] = []

    def get_name(self) -> str:
        return "recording"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict) -> "_RecordingQuantConfig":
        return cls()

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, LinearBase):
            self.prefixes.append(prefix)
            return UnquantizedLinearMethod()
        return None


class _QuantizedMarker(UnquantizedLinearMethod):
    """Stands in for a real quantized method so linears still build on CPU."""


class _SkipPolicyQuantConfig(_RecordingQuantConfig):
    """Applies vLLM's real ignored_layers policy, as DiffusionMXFP8Config does."""

    def __init__(self, ignored_layers: list[str]) -> None:
        super().__init__()
        self.ignored_layers = ignored_layers

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, LinearBase):
            self.prefixes.append(prefix)
            if is_layer_skipped(prefix, self.ignored_layers, fused_mapping=self.packed_modules_mapping):
                return UnquantizedLinearMethod()
            return _QuantizedMarker()
        return None


# ---------------------------------------------------------------------------
# create_s2v_transformer_from_config
# ---------------------------------------------------------------------------


def test_create_s2v_transformer_forwards_quant_config(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(s2v_transformer_module, "WanS2VTransformer3DModel", lambda **kw: captured.update(kw))

    quant_config = _RecordingQuantConfig()
    create_s2v_transformer_from_config({"patch_size": [1, 2, 2]}, quant_config=quant_config)

    assert captured["quant_config"] is quant_config


def test_create_s2v_transformer_without_quant_config_passes_none(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(s2v_transformer_module, "WanS2VTransformer3DModel", lambda **kw: captured.update(kw))

    create_s2v_transformer_from_config({"patch_size": [1, 2, 2]})

    assert "quant_config" not in captured


# ---------------------------------------------------------------------------
# Wan22S2VPipeline: both checkpoint layouts
# ---------------------------------------------------------------------------


def _init_pipeline(monkeypatch, layout: str, od_config: SimpleNamespace, transformer_config: dict) -> dict:
    """Run one format-specific init with every non-transformer loader stubbed out.

    Returns the kwargs that reached create_s2v_transformer_from_config.
    """
    for name in (
        "AutoTokenizer",
        "UMT5EncoderModel",
        "DistributedAutoencoderKLWan",
        "Wav2Vec2Processor",
        "Wav2Vec2ForCTC",
        "_load_wan_t5_as_umt5",
        "resolve_offload_strategy",
    ):
        monkeypatch.setattr(s2v_pipeline_module, name, MagicMock())

    subfolders = []

    def load_transformer_config(model_path, subfolder, local_files_only):
        subfolders.append(subfolder)
        return transformer_config

    captured: dict = {}

    def create_transformer(config, quant_config=None):
        captured.update(config=config, quant_config=quant_config)
        return torch.nn.Identity()

    monkeypatch.setattr(s2v_pipeline_module, "load_transformer_config", load_transformer_config)
    monkeypatch.setattr(s2v_pipeline_module, "create_s2v_transformer_from_config", create_transformer)

    pipeline = object.__new__(Wan22S2VPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = od_config
    pipeline.device = torch.device("cpu")
    if layout == "diffusers":
        pipeline._init_diffusers_format("/unused", torch.bfloat16)
    else:
        pipeline._init_original_format("/unused", torch.bfloat16)

    assert subfolders == ["transformer" if layout == "diffusers" else "."]
    assert captured["config"] is transformer_config
    return captured


@pytest.mark.parametrize("layout", ["diffusers", "original"])
def test_s2v_init_passes_cli_quant_config(monkeypatch, layout):
    quant_config = _RecordingQuantConfig()
    od_config = SimpleNamespace(quantization_config=quant_config)

    captured = _init_pipeline(monkeypatch, layout, od_config, {"patch_size": [1, 2, 2]})

    assert captured["quant_config"] is quant_config


@pytest.mark.parametrize("layout", ["diffusers", "original"])
def test_s2v_init_without_quantization_stays_unquantized(monkeypatch, layout):
    od_config = SimpleNamespace(quantization_config=None)

    captured = _init_pipeline(monkeypatch, layout, od_config, {"patch_size": [1, 2, 2]})

    assert captured["quant_config"] is None


@pytest.mark.parametrize("layout", ["diffusers", "original"])
def test_s2v_init_auto_detected_quant_defers_to_checkpoint(monkeypatch, layout):
    """Auto-detected startup metadata is not a user policy: the checkpoint's own
    config.json decides, as in the T2V/I2V pipeline."""
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    od_config = SimpleNamespace(
        quantization_config=_RecordingQuantConfig(),
        quantization_config_is_auto_detected=True,
    )
    transformer_config = {
        "patch_size": [1, 2, 2],
        "quantization_config": {"quant_method": "mxfp8", "is_checkpoint_mxfp8_serialized": True},
    }

    captured = _init_pipeline(monkeypatch, layout, od_config, transformer_config)

    assert isinstance(captured["quant_config"], DiffusionMXFP8Config)
    assert captured["quant_config"].is_checkpoint_mxfp8_serialized is True


# ---------------------------------------------------------------------------
# WanS2VTransformer3DModel: quant_config reaches the block linears
# ---------------------------------------------------------------------------


@pytest.fixture
def tp1_group():
    """Real single-process TP group so the parallel linear layers construct on CPU."""
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29502")
    init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
    initialize_model_parallel()
    yield
    cleanup_dist_env_and_memory()


_NUM_LAYERS = 2
_BLOCK_LINEARS = [
    "self_attn.to_qkv",
    "self_attn.to_out",
    "cross_attn.to_q",
    "cross_attn.to_k",
    "cross_attn.to_v",
    "cross_attn.to_out",
    "ffn.net_0.proj",
    "ffn.net_2",
]


def _tiny_s2v_transformer(quant_config: QuantizationConfig | None) -> WanS2VTransformer3DModel:
    return WanS2VTransformer3DModel(
        audio_dim=16,
        audio_inject_layers=[0, 1],
        motion_token_num=16,
        dim=32,
        ffn_dim=64,
        freq_dim=16,
        text_dim=16,
        text_len=8,
        num_heads=2,
        num_layers=_NUM_LAYERS,
        quant_config=quant_config,
    )


def test_s2v_quant_config_reaches_block_linears_only(tp1_group):
    quant_config = _RecordingQuantConfig()
    model = _tiny_s2v_transformer(quant_config)

    expected = [f"blocks.{i}.{name}" for i in range(_NUM_LAYERS) for name in _BLOCK_LINEARS]
    assert sorted(quant_config.prefixes) == sorted(expected)

    # Each prefix must be the layer's real module path, so ignored_layers and
    # checkpoint scale names line up with what the quant method sees.
    linear_names = {name for name, module in model.named_modules() if isinstance(module, LinearBase)}
    assert set(expected) <= linear_names
    # Audio injector / motioner / embedders keep their BF16 linears.
    assert all(name.startswith("blocks.") for name in quant_config.prefixes)


def test_s2v_without_quant_config_builds_unquantized_linears(tp1_group):
    model = _tiny_s2v_transformer(None)

    linears = [module for module in model.modules() if isinstance(module, LinearBase)]
    assert linears
    assert all(isinstance(module.quant_method, UnquantizedLinearMethod) for module in linears)


def test_s2v_binds_packed_qkv_mapping_to_quant_config(tp1_group):
    """The registry binds packed_modules_mapping to Wan22S2VPipeline, which has none;
    the transformer must bind its own so fused to_qkv resolves to to_q/to_k/to_v."""
    quant_config = _SkipPolicyQuantConfig(ignored_layers=[])
    assert quant_config.packed_modules_mapping == {}

    _tiny_s2v_transformer(quant_config)

    assert quant_config.packed_modules_mapping == WanS2VTransformer3DModel.packed_modules_mapping


def test_s2v_ignored_qkv_shards_skip_fused_to_qkv(tp1_group):
    ignored = [f"blocks.0.self_attn.{name}" for name in ("to_q", "to_k", "to_v")]
    model = _tiny_s2v_transformer(_SkipPolicyQuantConfig(ignored_layers=ignored))

    methods = {
        name: type(module.quant_method) for name, module in model.named_modules() if isinstance(module, LinearBase)
    }
    assert methods["blocks.0.self_attn.to_qkv"] is UnquantizedLinearMethod
    # Only the ignored layer is skipped: the other block and linears stay quantized.
    assert methods["blocks.1.self_attn.to_qkv"] is _QuantizedMarker
    assert methods["blocks.0.self_attn.to_out"] is _QuantizedMarker
    assert methods["blocks.0.cross_attn.to_q"] is _QuantizedMarker


def test_s2v_partially_ignored_qkv_shards_are_rejected(tp1_group):
    """Mixed precision inside the fused to_qkv cannot be honored and must fail loudly."""
    with pytest.raises(ValueError, match="some but not all shards"):
        _tiny_s2v_transformer(_SkipPolicyQuantConfig(ignored_layers=["blocks.0.self_attn.to_q"]))
