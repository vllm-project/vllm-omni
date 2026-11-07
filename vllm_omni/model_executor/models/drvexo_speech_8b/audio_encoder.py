# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Explicit packed native audio-encoder linears, with unchanged BF16 frontend."""

import torch
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import LinearBase, ReplicatedLinear
from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config, ModelOptNvFp4Config

from .checkpoint import load_tensor_file as load_file


def install_packed_audio_encoder(model, precision, packed_path, reference_path):
    """Check original fused native parameters, install99 quantized linears.

    The native loader finalizes their kernels once after load_weights returns.
    All three convolutions, norms, positions and attention remain untouched.
    """
    if precision not in ("fp8", "nvfp4"):
        raise ValueError(f"Unsupported encoder precision: {precision}")
    if get_tensor_model_parallel_world_size() != 1:
        raise ValueError("Packed DRVEXO audio encoder currently requires TP1")
    source = load_file(str(reference_path), device="cpu")
    packed = load_file(str(packed_path), device="cpu")
    parameters = dict(model.named_parameters())
    if set(source) != set(parameters):
        raise ValueError("Original native encoder parameter names differ from calibration source")
    for name, parameter in parameters.items():
        if parameter.dtype != source[name].dtype or not torch.equal(parameter.detach().cpu(), source[name]):
            raise ValueError(f"Original audio encoder differs from calibration source: {name}")
    originals = [(n, m) for n, m in model.named_modules() if isinstance(m, LinearBase)]
    if len(originals) != 99:
        raise ValueError(f"Expected99 native audio encoder linears, got{len(originals)}")
    if any(m.skip_bias_add for _, m in originals):
        raise ValueError("Encoder calibration expects bias applied inside each linear")
    quant = (
        ModelOptFp8Config("FP8", True, None, [])
        if precision == "fp8"
        else ModelOptNvFp4Config(is_checkpoint_nvfp4_serialized=True)
    )
    consumed = set()
    replacements = []
    for name, original in originals:
        weight = source[name + ".weight"]
        with torch.device(original.weight.device):
            layer = ReplicatedLinear(
                weight.shape[1],
                weight.shape[0],
                bias=original.bias is not None,
                params_dtype=torch.bfloat16,
                quant_config=quant,
                prefix="audio_tower." + name,
                return_bias=original.return_bias,
                skip_bias_add=False,
                disable_tp=True,
            )
        target = dict(layer.named_parameters())
        found = {key.removeprefix(name + "."): value for key, value in packed.items() if key.startswith(name + ".")}
        if set(target) != set(found):
            raise ValueError(f"Packed audio encoder tensor names mismatch: {name}")
        with torch.no_grad():
            for suffix, parameter in target.items():
                value = found[suffix]
                if parameter.dtype != value.dtype:
                    raise ValueError(f"Packed audio encoder dtype mismatch: {name}.{suffix}")
                if suffix == "bias" and not torch.equal(value, source[name + ".bias"]):
                    raise ValueError(f"Packed audio encoder bias changed: {name}")
                loader = getattr(parameter, "weight_loader", None)
                if loader is not None:
                    loader(parameter, value)
                else:
                    if parameter.shape != value.shape:
                        raise ValueError(f"Packed audio encoder shape mismatch: {name}.{suffix}")
                    parameter.copy_(value)
                consumed.add(name + "." + suffix)
        replacements.append((name, layer))
    retained = set(packed) - consumed
    quantized_weights = {n + ".weight" for n, _ in originals}
    if retained != set(source) - quantized_weights - (consumed & set(source)):
        raise ValueError("Unexpected packed encoder residual tensors")
    if any(not torch.equal(packed[n], source[n]) for n in retained):
        raise ValueError("Unquantized encoder tensors changed")
    for name, layer in replacements:
        parent, _, key = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, key, layer)
    return replacements
