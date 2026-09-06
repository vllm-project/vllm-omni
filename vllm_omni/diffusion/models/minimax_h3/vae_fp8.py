# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Optional FP8 backend wiring for the MiniMax H3 video VAE."""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
from vllm.model_executor.kernels.linear.scaled_mm import (
    MarlinFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.linear import LinearBase, ReplicatedLinear
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PtpcOnlineLinearMethod,
)
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import replace_parameter

H3_VAE_FP8_LINEAR_NAMES = frozenset({"attn.to_qkv", "ff.w1", "ff.w2"})

logger = init_logger(__name__)


class _H3VAEFp8LinearMethod(Fp8PtpcOnlineLinearMethod):
    """Use vLLM's PTPC weights and scaled-MM backend without ``CustomOp``."""

    def __init__(self) -> None:
        # The pipeline decodes the video VAE under FP16 autocast even when the
        # DiT uses BF16. Kernel selection must therefore not inherit the DiT's
        # dtype from the process-wide vLLM config. The upstream base initializer
        # only establishes these two fields, so declare the component contract
        # directly before its common create/process/apply lifecycle runs.
        self.input_dtype = torch.float16
        self.out_dtype = torch.float16

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """Create an already-loaded component weight without deferred loading.

        The common online method registers a layerwise checkpoint loader because
        LLM weights are normally populated after module construction. H3's remote
        video VAE is already loaded before these linears are replaced. Registering
        that loader would make Omni's finalizer quantize the weight a second time.
        """

        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                device="meta",
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=extra_weight_attrs.get("weight_loader"),
        )
        layer.register_parameter("weight", weight)

        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=self.activation_quant_key,
            weight_quant_key=self.weight_quant_key,
            weight_shape=layer.weight.shape,
            input_dtype=self.input_dtype,
            out_dtype=self.out_dtype,
            module_name=self.__class__.__name__,
        )
        if isinstance(self.fp8_linear, MarlinFP8ScaledMMLinearKernel):
            raise ValueError(
                "MiniMax H3 video-VAE FP8 requires per-token activation "
                "quantization, but the selected backend is weight-only Marlin FP8"
            )

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # H3 keeps decoder-block residuals and normalized activations in FP32.
        # Quantize that tensor directly, as the original optimized path did.
        # Calling the registered accelerator op here deliberately bypasses vLLM's
        # QuantFP8(CustomOp) wrapper: compile policy must not replace this eager
        # kernel with the much slower native decomposition.
        original_shape = x.shape
        x_quantized, input_scale = ops.scaled_fp8_quant(
            x.reshape(-1, x.shape[-1]),
            use_per_token_if_dynamic=True,
        )
        return self.fp8_linear.apply_scaled_mm(
            A=x_quantized,
            B=layer.weight,
            out_dtype=self.out_dtype,
            As=input_scale,
            Bs=layer.weight_scale,
            bias=bias,
            output_shape=[*original_shape[:-1], layer.output_size_per_partition],
        )


class _H3VAEFp8Config(Fp8Config):
    """Select the common per-token/per-channel FP8 linear backend."""

    def get_quant_method(self, layer: nn.Module, prefix: str):
        if isinstance(layer, LinearBase):
            return _H3VAEFp8LinearMethod()
        return None


def _decoder_targets(
    decoder: nn.Module,
    layers: frozenset[str],
) -> list[tuple[nn.Module, str, str, nn.Linear]] | None:
    blocks = getattr(decoder, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        return None

    targets: list[tuple[nn.Module, str, str, nn.Linear]] = []
    for index, block in enumerate(blocks):
        attention = getattr(block, "attn", None)
        feed_forward = getattr(block, "ff", None)
        candidates = {
            "attn.to_qkv": (attention, "to_qkv", getattr(attention, "to_qkv", None)),
            "ff.w1": (feed_forward, "w1", getattr(feed_forward, "w1", None)),
            "ff.w2": (feed_forward, "w2", getattr(feed_forward, "w2", None)),
        }
        if not all(
            isinstance(parent, nn.Module) and isinstance(linear, nn.Linear)
            for parent, _attribute, linear in candidates.values()
        ):
            return None

        to_qkv = candidates["attn.to_qkv"][2]
        w1 = candidates["ff.w1"][2]
        w2 = candidates["ff.w2"][2]
        if (
            to_qkv.out_features != 3 * to_qkv.in_features
            or w1.in_features != to_qkv.in_features
            or w1.out_features != 2 * w2.in_features
            or w2.out_features != to_qkv.in_features
        ):
            return None

        for name in sorted(layers):
            parent, attribute, linear = candidates[name]
            prefix = f"video_vae.decoder.transformer_blocks.{index}.{name}"
            targets.append((parent, attribute, prefix, linear))
    return targets


@torch.no_grad()
def _replace_with_fp8_linear(
    source: nn.Linear,
    *,
    prefix: str,
    execution_device: torch.device,
    storage_device: torch.device,
    quant_config: Fp8Config,
) -> ReplicatedLinear:
    with execution_device:
        target = ReplicatedLinear(
            input_size=source.in_features,
            output_size=source.out_features,
            bias=source.bias is not None,
            params_dtype=torch.float16,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=False,
            disable_tp=True,
        )

    # Online vLLM methods create a meta weight and normally materialize it via
    # the checkpoint loader. The remote VAE is already loaded, so materialize
    # that weight directly and then enter the same process/apply lifecycle.
    dense_weight = source.weight.detach().to(device=execution_device, dtype=torch.float16)
    replace_parameter(target, "weight", dense_weight)
    if source.bias is not None:
        target.bias.copy_(source.bias.detach().to(device=execution_device, dtype=torch.float16))

    target.quant_method.process_weights_after_loading(target)
    target.update_param_tp_status()
    target.train(source.training)
    if storage_device != execution_device:
        target.to(storage_device)
    return target


def install_h3_vae_fp8_quantization(
    decoder: nn.Module,
    *,
    execution_device: torch.device,
    storage_device: torch.device,
    layers: frozenset[str],
) -> None:
    """Replace selected decoder linears with vLLM's common FP8 backend."""

    unknown = layers - H3_VAE_FP8_LINEAR_NAMES
    if unknown:
        raise ValueError(f"Unsupported MiniMax H3 video-VAE FP8 layers: {sorted(unknown)}")

    installed_layers = getattr(decoder, "_omni_h3_vae_fp8_layers", None)
    if installed_layers is not None:
        if installed_layers != layers:
            raise ValueError(
                "MiniMax H3 VAE FP8 is already installed for "
                f"layers={sorted(installed_layers)}, cannot reinstall for {sorted(layers)}"
            )
        return
    if not layers:
        decoder._omni_h3_vae_fp8_layers = layers
        return

    targets = _decoder_targets(decoder, layers)
    if targets is None or any(
        linear.weight.dtype not in {torch.float16, torch.float32} for _parent, _attribute, _prefix, linear in targets
    ):
        raise ValueError(
            "MiniMax H3 video-VAE FP8 was explicitly requested, but the "
            "loaded decoder does not match the supported remote-code contract"
        )

    quant_config = _H3VAEFp8Config()
    replacements: list[tuple[nn.Module, str, nn.Linear]] = []
    try:
        for parent, attribute, prefix, source in targets:
            target = _replace_with_fp8_linear(
                source,
                prefix=prefix,
                execution_device=execution_device,
                storage_device=storage_device,
                quant_config=quant_config,
            )
            setattr(parent, attribute, target)
            replacements.append((parent, attribute, source))
    except BaseException:
        for parent, attribute, source in reversed(replacements):
            setattr(parent, attribute, source)
        raise

    decoder._omni_h3_vae_fp8_layers = layers
    logger.info(
        "Enabled user-requested MiniMax H3 video-VAE FP8 via the vLLM backend for layers=%s on %s",
        sorted(layers),
        execution_device,
    )


__all__ = [
    "H3_VAE_FP8_LINEAR_NAMES",
    "install_h3_vae_fp8_quantization",
]
