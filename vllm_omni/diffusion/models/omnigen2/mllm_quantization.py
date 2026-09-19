# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Online FP8 preparation for OmniGen2's HF language encoder."""

import torch
from torch import nn
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.quantization import ComponentQuantizationConfig


class _MllmFp8Linear(ReplicatedLinear):
    """Keep the HF activation dtype around the low-precision GEMM.

    OmniGen2's published HF encoder also loads in FP32. CUTLASS FP8 needs
    BF16/FP16 output and a matching bias; casting the entire HF model would
    change the precision of the vision tower and normalization as well.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = super().forward(x.to(self.params_dtype))
        return output.to(x.dtype)


def prepare_mllm_fp8(mllm: nn.Module, quant_config: QuantizationConfig | None) -> int:
    """Replace eligible language linears; the pipeline loader finalizes FP8.

    HF loads the original checkpoint first, retaining its version-specific name
    mapping and vision implementation. Only decoder-block linears are adapted;
    the vision tower, embeddings, output head and normalization stay untouched.
    This reduces steady-state weights, not the initial HF load peak.
    """
    # A global quantization flag historically affects only the DiT. Require an
    # explicit component entry so enabling the encoder does not change that API.
    if not isinstance(quant_config, ComponentQuantizationConfig):
        return 0
    config = quant_config.component_configs.get("mllm")
    if config is None:
        return 0
    if (
        not isinstance(config, Fp8Config)
        or config.is_checkpoint_fp8_serialized
        or config.activation_scheme != "dynamic"
    ):
        raise ValueError("OmniGen2 mllm supports dynamic online FP8 from an unquantized checkpoint only")

    # Transformers v5 nests the decoder below language_model; v4 used model.
    decoder = getattr(mllm.model, "language_model", mllm.model)
    if not hasattr(decoder, "layers"):
        raise ValueError("Cannot locate OmniGen2 mllm language decoder layers")
    decoder_prefix = "model.language_model.layers" if decoder is not mllm.model else "model.layers"
    replaced = 0
    for name, layer in list(decoder.layers.named_modules()):
        if not isinstance(layer, nn.Linear):
            continue
        prefix = f"mllm.{decoder_prefix}.{name}"
        gemm_dtype = layer.weight.dtype if layer.weight.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        # Online FP8 captures its output dtype from the construction context.
        with torch.device(layer.weight.device), set_default_torch_dtype(gemm_dtype):
            replacement = _MllmFp8Linear(
                layer.in_features,
                layer.out_features,
                bias=layer.bias is not None,
                params_dtype=gemm_dtype,
                quant_config=config,
                prefix=prefix,
                return_bias=False,
                disable_tp=True,
            )
        # Respect the backend's ignored-layer rules without adapting that layer.
        if isinstance(replacement.quant_method, UnquantizedLinearMethod):
            continue
        with torch.no_grad():
            # Use the parameter loader: current vLLM creates online FP8 weights
            # on meta and materializes/quantizes them when all weights arrive.
            # copy_ into meta would silently discard the checkpoint values.
            replacement.weight.weight_loader(replacement.weight, layer.weight)
            if layer.bias is not None:
                replacement.bias.weight_loader(replacement.bias, layer.bias)
        replacement.train(layer.training)
        parent_name, _, child_name = name.rpartition(".")
        parent = decoder.layers.get_submodule(parent_name) if parent_name else decoder.layers
        setattr(parent, child_name, replacement)
        replaced += 1
    return replaced
