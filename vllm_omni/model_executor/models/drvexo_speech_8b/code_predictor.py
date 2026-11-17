# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Experimental explicit Code Predictor precision; not enabled by model import.

Install only after original BF16 weights have loaded and before any prediction.
The native vLLM loader (or standalone probe) finalizes each quantized linear once.
Normalization, RoPE and SDPA retain the existing native implementation.
"""

from dataclasses import replace

import torch
from torch import nn
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config, ModelOptNvFp4Config

from .checkpoint import load_tensor_file as load_file


def configure_body_cuda_graphs(model, enabled):
    """Opt into the parent's full-length body graphs before lazy initialization.

    Keep sampling, output heads and request-specific RNG outside capture. Do not
    change prefix lengths or enable global Talker/Codec graph execution.
    """
    if not isinstance(enabled, bool):
        raise ValueError("drvexo_code_predictor_cuda_graphs must be boolean")
    if not enabled:
        return
    if model._compiled_model_fwd is not None or model._device_graphs:
        raise ValueError("Code Predictor graph mode must be configured before inference")
    if next(model.parameters()).device.type != "cuda":
        raise ValueError("Code Predictor body CUDA graphs require CUDA weights")
    if model._prefix_graphs_enabled:
        raise ValueError("DRVEXO body graphs do not enable prefix-length changes")
    model._wrapper_config = replace(model._wrapper_config, use_cuda_graphs=True)


class IndependentProjections(nn.Module):
    """Preserve separately calibrated q/k/v or gate/up scales."""

    def __init__(self, parts):
        super().__init__()
        self.parts = nn.ModuleList(parts)

    def forward(self, inputs):
        return torch.cat([part(inputs) for part in self.parts], dim=-1)


def fused_reference(source, num_layers):
    expected = dict(source)
    for i in range(num_layers):
        for parent, sources, target in [
            ("self_attn", ("q_proj", "k_proj", "v_proj"), "qkv_proj"),
            ("mlp", ("gate_proj", "up_proj"), "gate_up_proj"),
        ]:
            for suffix in ("weight", "bias"):
                names = [f"model.layers.{i}.{parent}.{name}.{suffix}" for name in sources]
                present = [name in expected for name in names]
                if any(present) and not all(present):
                    raise ValueError("Incomplete original projection shards")
                if all(present):
                    expected[f"model.layers.{i}.{parent}.{target}.{suffix}"] = torch.cat(
                        [expected.pop(name) for name in names], dim=0
                    )
    return expected


def install_packed_code_predictor(model, precision, packed_path, reference_path):
    """Check the exact native source, then install51 independently scaled linears."""
    if precision not in ("fp8", "nvfp4"):
        raise ValueError(f"Unsupported Code Predictor precision: {precision}")
    if model._compiled_model_fwd is not None:
        raise ValueError("Code Predictor precision cannot change after inference initialization")
    source = load_file(str(reference_path), device="cpu")
    packed = load_file(str(packed_path), device="cpu")
    expected = fused_reference(source, len(model.model.layers))
    parameters = dict(model.named_parameters())
    if set(expected) != set(parameters):
        raise ValueError("Original Code Predictor parameter names do not match calibration source")
    for name, parameter in parameters.items():
        if parameter.dtype != expected[name].dtype or not torch.equal(parameter.detach().cpu(), expected[name]):
            raise ValueError(f"Original Code Predictor differs from calibration source: {name}")
    quant = (
        ModelOptFp8Config("FP8", True, None, [])
        if precision == "fp8"
        else ModelOptNvFp4Config(is_checkpoint_nvfp4_serialized=True)
    )
    device = next(model.parameters()).device
    consumed = set()
    layers = []

    def linear(prefix):
        weight = source[prefix + ".weight"]
        with torch.device(device):
            layer = ReplicatedLinear(
                weight.shape[1],
                weight.shape[0],
                bias=prefix + ".bias" in source,
                params_dtype=torch.bfloat16,
                quant_config=quant,
                prefix="code_predictor." + prefix,
                return_bias=False,
                disable_tp=True,
            )
        parameters = dict(layer.named_parameters())
        found = {key.removeprefix(prefix + "."): value for key, value in packed.items() if key.startswith(prefix + ".")}
        if set(found) != set(parameters):
            raise ValueError(f"Packed Code Predictor tensor names mismatch: {prefix}")
        with torch.no_grad():
            for name, parameter in parameters.items():
                value = found[name]
                if value.dtype != parameter.dtype:
                    raise ValueError(f"Packed Code Predictor dtype mismatch: {prefix}.{name}")
                if name == "bias" and not torch.equal(value, source[prefix + ".bias"]):
                    raise ValueError(f"Packed Code Predictor bias changed: {prefix}")
                loader = getattr(parameter, "weight_loader", None)
                if loader is not None:
                    loader(parameter, value)
                else:
                    if value.shape != parameter.shape:
                        raise ValueError(f"Packed Code Predictor shape mismatch: {prefix}.{name}")
                    parameter.copy_(value)
                consumed.add(prefix + "." + name)
        layers.append((prefix, layer))
        return layer

    for i, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{i}."
        layer.self_attn.qkv_proj = IndependentProjections(
            [linear(prefix + "self_attn." + name) for name in ("q_proj", "k_proj", "v_proj")]
        )
        layer.self_attn.o_proj = linear(prefix + "self_attn.o_proj")
        layer.mlp.gate_up_proj = IndependentProjections(
            [linear(prefix + "mlp." + name) for name in ("gate_proj", "up_proj")]
        )
        layer.mlp.down_proj = linear(prefix + "mlp.down_proj")
    for i in range(len(model.lm_head)):
        model.lm_head[i] = linear(f"lm_head.{i}")
    model.small_to_mtp_projection = linear("small_to_mtp_projection")
    if len(layers) != 51:
        raise ValueError(f"Expected51 Code Predictor linear layers, found{len(layers)}")
    retained = set(packed) - consumed
    quantized_weights = {name + ".weight" for name, _ in layers}
    if retained != set(source) - quantized_weights - {n for n in consumed if n in source}:
        raise ValueError("Unexpected packed Code Predictor residual tensors")
    if any(not torch.equal(packed[name], source[name]) for name in retained):
        raise ValueError("Unquantized Code Predictor tensors changed")
    return layers
