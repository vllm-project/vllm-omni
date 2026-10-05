# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Omni's online FP8 option backed by vLLM's online quantization API."""

from vllm.config.quantization import QuantizationConfigArgs, QuantSpec
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig


class DiffusionFp8Config(OnlineQuantizationConfig):
    """Keep the diffusion ``fp8`` option for quantizing BF16/FP16 weights.

    Upstream Fp8Config now describes serialized checkpoints only. Keeping the
    Omni method name also lets checkpoint metadata replace this online config
    through the existing serialized-checkpoint reconciliation path.
    """

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: list[str] | None = None,
        weight_block_size: list[int] | None = None,
        store_dtype: str | None = None,
    ) -> None:
        if is_checkpoint_fp8_serialized:
            raise ValueError("Serialized FP8 checkpoints require upstream Fp8Config")
        if activation_scheme != "dynamic":
            raise ValueError("Online FP8 quantization requires activation_scheme='dynamic'")
        if weight_block_size is not None:
            raise ValueError("Block-wise FP8 quantization requires a serialized checkpoint or fp8_per_block")
        if store_dtype is not None:
            raise ValueError("Online FP8 quantization does not support store_dtype")
        super().__init__(
            QuantizationConfigArgs(
                linear=QuantSpec(weight="fp8_per_tensor_static"),
                moe=QuantSpec(weight="fp8_per_tensor_static"),
                ignore=list(ignored_layers or []),
            )
        )
        self.is_checkpoint_fp8_serialized = False
        self.activation_scheme = activation_scheme
        self.weight_block_size = None
        self.store_dtype = None

    @classmethod
    def get_name(cls) -> str:
        return "fp8"
