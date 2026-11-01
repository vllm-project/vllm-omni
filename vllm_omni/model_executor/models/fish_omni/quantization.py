# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Drama 3 checkpoint quantization: block-FP8 experts or hybrid GPTQ INT4."""

from __future__ import annotations

from typing import Any

import torch
from typing_extensions import override
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.platforms import current_platform

QUANTIZATION_METHOD = "bsgemm-moe"
DRAMA3_INT4_QUANTIZATION_METHOD = "drama3-int4"


@register_quantization_config(QUANTIZATION_METHOD)
class ExpertFp8Config(Fp8Config):
    """vLLM's block-FP8 MoE path for routed experts, unquantized linears."""

    def __init__(self, weight_block_size: list[int]) -> None:
        super().__init__(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=weight_block_size,
        )

    @classmethod
    def get_name(cls) -> str:
        return QUANTIZATION_METHOD

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ExpertFp8Config:
        if config.get("activation_scheme", "dynamic") != "dynamic":
            raise ValueError(f"{QUANTIZATION_METHOD} needs dynamic activation scales")
        block = config.get("weight_block_size")
        if not (isinstance(block, list) and len(block) == 2 and all(int(b) > 0 for b in block)):
            raise ValueError(f"{QUANTIZATION_METHOD} needs a 2-int weight_block_size, got {block!r}")
        return cls(weight_block_size=[int(b) for b in block])

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str, tid2eid: torch.Tensor | None = None
    ) -> QuantizeMethodBase | None:
        if current_platform.device_type == "npu":
            from vllm_omni.platforms.npu.quant.fish_omni import get_fish_omni_quant_method

            return get_fish_omni_quant_method(self, layer, prefix, tid2eid)
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        return super().get_quant_method(layer, prefix)


@register_quantization_config(DRAMA3_INT4_QUANTIZATION_METHOD)
class Drama3Int4Config(QuantizationConfig):
    """Hybrid INT4: GPTQ routed experts and checkpoint-selected attention projections.

    Dense targets include fused QKV/QKVZ and output projections; shared experts,
    routing gates and untargeted projections remain BF16.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        for key in ("bits", "sym"):
            if key not in config:
                raise ValueError(f"{DRAMA3_INT4_QUANTIZATION_METHOD} requires {key}")
        if config["bits"] != 4 or not config["sym"] or config.get("desc_act", False):
            raise ValueError(
                f"{DRAMA3_INT4_QUANTIZATION_METHOD} requires symmetric 4-bit weights without activation ordering"
            )
        if "group_size" not in config:
            raise ValueError(f"{DRAMA3_INT4_QUANTIZATION_METHOD} requires group_size")
        if config.get("quant_method") != DRAMA3_INT4_QUANTIZATION_METHOD:
            raise ValueError(
                f"quant_method={config.get('quant_method')!r}; expected {DRAMA3_INT4_QUANTIZATION_METHOD!r}"
            )
        for section, expected_method in (("experts", "gptq"), ("dense", "gptq_marlin")):
            section_config = config.get(section)
            if not isinstance(section_config, dict) or section_config.get("quant_method") != expected_method:
                raise ValueError(
                    f"{DRAMA3_INT4_QUANTIZATION_METHOD} requires {section}.quant_method={expected_method!r}; "
                    f"got {section_config!r}"
                )
        if "targets" not in config["dense"]:
            raise ValueError(f"{DRAMA3_INT4_QUANTIZATION_METHOD} requires dense.targets")
        # Both checkpoint sections use GPTQ weight packing. AutoGPTQ dispatches
        # dense layers through the platform kernel selector (Marlin on A100),
        # and routed experts through its WNA16 backend selector.
        gptq_config = {
            "quant_method": "gptq",
            "bits": config["bits"],
            "group_size": config["group_size"],
            "sym": config["sym"],
            "desc_act": False,
        }
        # This child config is dispatched only for the checkpoint targets below.
        # AutoGPTQ otherwise defaults to an empty (unquantized) layer list.
        self.dense_config = AutoGPTQConfig.from_config(
            {**gptq_config, "modules_in_block_to_quantize": ["self_attn", "linear_attn"]}
        )
        self.expert_config = AutoGPTQConfig.from_config({**gptq_config, "modules_in_block_to_quantize": ["experts"]})
        self.dense_targets = tuple(config["dense"]["targets"])

    @classmethod
    def get_name(cls) -> str:
        return DRAMA3_INT4_QUANTIZATION_METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # CUDA capability: 70 = Volta, 75 = Turing, 80 = Ampere (SM80).
        # This checkpoint integration requires Ampere or newer.
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Drama3Int4Config:
        return cls(config)

    @override
    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        if isinstance(layer, RoutedExperts):
            return self.expert_config.get_quant_method(layer, prefix)
        if isinstance(layer, LinearBase):
            parent, name = prefix.rsplit(".", 1)
            shards = {
                "qkv_proj": ("q_proj", "k_proj", "v_proj"),
                "in_proj_qkvz": ("in_proj_qkv", "in_proj_z"),
            }.get(name, (name,))
            if all(any(f"{parent}.{shard}".endswith(target) for target in self.dense_targets) for shard in shards):
                return self.dense_config.get_quant_method(layer, prefix)
            return UnquantizedLinearMethod()
        return None
