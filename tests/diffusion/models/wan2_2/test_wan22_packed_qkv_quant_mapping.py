# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests: WanTransformer3DModel binds its packed QKV mapping to the quant config.

The registry calls vLLM's ``configure_quant_config`` with the pipeline class
(Wan22Pipeline / Wan22I2VPipeline / VACE), which declares no
``packed_modules_mapping``. Without the transformer binding its own mapping,
``ignored_layers`` naming ``to_q``/``to_k``/``to_v`` did not cover the fused
``to_qkv``, which was then quantized anyway.

All tests are pure-CPU and do not load model weights.
"""

import os

import pytest
import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped

from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformer3DModel

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _QuantizedMarker(UnquantizedLinearMethod):
    """Stands in for a real quantized method so linears still build on CPU."""


class _SkipPolicyQuantConfig(QuantizationConfig):
    """Applies vLLM's real ignored_layers policy, as DiffusionMXFP8Config does."""

    def __init__(self, ignored_layers: list[str]) -> None:
        super().__init__()
        self.ignored_layers = ignored_layers

    def get_name(self) -> str:
        return "skip_policy"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict) -> "_SkipPolicyQuantConfig":
        return cls([])

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers, fused_mapping=self.packed_modules_mapping):
                return UnquantizedLinearMethod()
            return _QuantizedMarker()
        return None


@pytest.fixture
def tp1_group():
    """Single-process vllm-omni parallel groups (TP/PP/SP = 1) so the model constructs on CPU."""
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_env,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29503")
    init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
    initialize_model_parallel(backend="gloo")
    yield
    destroy_model_parallel()
    destroy_distributed_env()


# T2V (no image branch) and I2V (added K/V projections for the image tokens).
_VARIANTS = {
    "t2v": {},
    "i2v": {"image_dim": 16, "added_kv_proj_dim": 16},
}


def _tiny_wan_transformer(quant_config: QuantizationConfig | None, variant: str) -> WanTransformer3DModel:
    return WanTransformer3DModel(
        num_attention_heads=2,
        attention_head_dim=16,
        text_dim=16,
        freq_dim=16,
        ffn_dim=64,
        num_layers=2,
        quant_config=quant_config,
        **_VARIANTS[variant],
    )


@pytest.mark.parametrize("variant", sorted(_VARIANTS))
def test_wan_binds_packed_qkv_mapping_to_quant_config(tp1_group, variant):
    quant_config = _SkipPolicyQuantConfig(ignored_layers=[])
    assert quant_config.packed_modules_mapping == {}

    _tiny_wan_transformer(quant_config, variant)

    assert quant_config.packed_modules_mapping == WanTransformer3DModel.packed_modules_mapping


@pytest.mark.parametrize("variant", sorted(_VARIANTS))
def test_wan_ignored_qkv_shards_skip_fused_to_qkv(tp1_group, variant):
    ignored = [f"blocks.0.attn1.{name}" for name in ("to_q", "to_k", "to_v")]
    model = _tiny_wan_transformer(_SkipPolicyQuantConfig(ignored_layers=ignored), variant)

    methods = {
        name: type(module.quant_method) for name, module in model.named_modules() if isinstance(module, LinearBase)
    }
    assert methods["blocks.0.attn1.to_qkv"] is UnquantizedLinearMethod
    # Only the ignored layer is skipped: the other block and linears stay quantized.
    assert methods["blocks.1.attn1.to_qkv"] is _QuantizedMarker
    assert methods["blocks.0.attn1.to_out"] is _QuantizedMarker
    assert methods["blocks.0.attn2.to_q"] is _QuantizedMarker


@pytest.mark.parametrize("variant", sorted(_VARIANTS))
def test_wan_partially_ignored_qkv_shards_are_rejected(tp1_group, variant):
    """Mixed precision inside the fused to_qkv cannot be honored and must fail loudly."""
    with pytest.raises(ValueError, match="some but not all shards"):
        _tiny_wan_transformer(_SkipPolicyQuantConfig(ignored_layers=["blocks.0.attn1.to_q"]), variant)
