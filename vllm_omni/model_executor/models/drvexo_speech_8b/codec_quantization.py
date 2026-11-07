# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Optional packed Codec linears with explicit surrounding compute dtype.

The compute dtype selects convolution, normalization, SnakeBeta and codebook precision.
The outer vLLM loader must finalize the installed linear kernels exactly once.
"""

import torch
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config, ModelOptNvFp4Config

from .checkpoint import load_tensor_file as load_file


class CodecLinear(torch.nn.Module):
    def __init__(self, original, quant, prefix, compute_dtype=torch.float32):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.linear = ReplicatedLinear(
            original.in_features,
            original.out_features,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=quant,
            prefix=prefix,
            return_bias=False,
            disable_tp=True,
        )
        self.register_buffer(
            "bias", None if original.bias is None else original.bias.detach().to(compute_dtype).clone()
        )

    def forward(self, value):
        # The Codec transposes convolution outputs into B,T,C views. Native
        # quantized GEMMs require contiguous features; nn.Linear had accepted
        # these strided views, hiding the boundary requirement in eager probes.
        output = self.linear(value.to(torch.bfloat16).contiguous()).to(self.compute_dtype)
        return output if self.bias is None else output + self.bias


def install_packed_codec_linears(model, precision, packed_path, reference_path, compute_dtype=torch.float32):
    """Strictly check the actual native FP32 state, then replace62 linears."""
    if precision not in ("fp8", "nvfp4"):
        raise ValueError(f"Unsupported Codec precision: {precision}")
    if compute_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Codec compute dtype must be float32 or bfloat16")
    if get_tensor_model_parallel_world_size() != 1:
        raise ValueError("Packed DRVEXO Codec currently requires TP1")
    source = load_file(str(reference_path), device="cpu")
    packed = load_file(str(packed_path), device="cpu")
    current = model.state_dict()
    if set(current) != set(source):
        raise ValueError("Codec native state names differ from calibration source")
    for name, value in current.items():
        if value.dtype != source[name].dtype or not torch.equal(value.detach().cpu(), source[name]):
            raise ValueError(f"Codec native value differs from calibration source: {name}")
    originals = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    if len(originals) != 62:
        raise ValueError(f"Expected62 Codec linears, got{len(originals)}")
    quant = (
        ModelOptFp8Config("FP8", True, None, [])
        if precision == "fp8"
        else ModelOptNvFp4Config(is_checkpoint_nvfp4_serialized=True)
    )
    replacements, consumed = [], set()
    for name, original in originals:
        if original.weight.dtype != torch.float32:
            raise ValueError("Codec precision adapter expects native FP32 execution")
        with torch.device(original.weight.device):
            wrapper = CodecLinear(original, quant, "decoder." + name, compute_dtype)
        target = dict(wrapper.linear.named_parameters())
        found = {
            k.removeprefix(name + "."): v for k, v in packed.items() if k.startswith(name + ".") and k != name + ".bias"
        }
        if set(target) != set(found):
            raise ValueError(f"Packed Codec tensor names mismatch: {name}")
        with torch.no_grad():
            for suffix, parameter in target.items():
                value = found[suffix]
                if value.dtype != parameter.dtype:
                    raise ValueError(f"Packed Codec dtype mismatch: {name}.{suffix}")
                loader = getattr(parameter, "weight_loader", None)
                if loader is not None:
                    loader(parameter, value)
                else:
                    if value.shape != parameter.shape:
                        raise ValueError(f"Packed Codec shape mismatch: {name}.{suffix}")
                    parameter.copy_(value)
                consumed.add(name + "." + suffix)
        replacements.append((name, wrapper))
    retained = set(source) - {n + ".weight" for n, _ in originals}
    if set(packed) - consumed != retained:
        raise ValueError("Unexpected retained Codec tensors")
    for name in retained:
        if packed[name].dtype != source[name].dtype or not torch.equal(packed[name], source[name]):
            raise ValueError(f"Unquantized Codec value changed: {name}")
    # Convert only the original decoder. Casting after installing packed
    # modules would destroy FP8 weights and their FP32 calibration scales.
    model.to(dtype=compute_dtype)
    for name, wrapper in replacements:
        parent, _, key = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, key, wrapper)
    return replacements
