# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Prepare the official BFL FLUX.2 NVFP4 checkpoint for the native loader."""

from collections.abc import Iterator

import torch

_ROOT = {
    "img_in": "x_embedder",
    "txt_in": "context_embedder",
    "time_in.in_layer": "time_guidance_embed.timestep_embedder.linear_1",
    "time_in.out_layer": "time_guidance_embed.timestep_embedder.linear_2",
    "guidance_in.in_layer": "time_guidance_embed.guidance_embedder.linear_1",
    "guidance_in.out_layer": "time_guidance_embed.guidance_embedder.linear_2",
    "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
    "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
    "single_stream_modulation.lin": "single_stream_modulation.linear",
    "final_layer.linear": "proj_out",
    "final_layer.adaLN_modulation.1": "norm_out.linear",
}
_DOUBLE = {
    "img_attn.qkv": "attn.to_qkv",
    "txt_attn.qkv": "attn.add_kv_proj",
    "img_attn.proj": "attn.to_out.0",
    "txt_attn.proj": "attn.to_add_out",
    "img_attn.norm.query_norm": "attn.norm_q",
    "img_attn.norm.key_norm": "attn.norm_k",
    "txt_attn.norm.query_norm": "attn.norm_added_q",
    "txt_attn.norm.key_norm": "attn.norm_added_k",
    "img_mlp.0": "ff.linear_in",
    "img_mlp.2": "ff.linear_out",
    "txt_mlp.0": "ff_context.linear_in",
    "txt_mlp.2": "ff_context.linear_out",
}
_SINGLE = {
    "linear1": "attn.to_qkv_mlp_proj",
    "linear2": "attn.to_out",
    "norm.query_norm": "attn.norm_q",
    "norm.key_norm": "attn.norm_k",
}


def map_bfl_name(name: str) -> str:
    """Map one exact BFL parameter name, retaining serialized scale suffixes."""
    module, _, parameter = name.rpartition(".")
    if module.startswith(("double_blocks.", "single_blocks.")):
        group, index, inner = module.split(".", 2)
        table = _DOUBLE if group == "double_blocks" else _SINGLE
        target = "transformer_blocks" if group == "double_blocks" else "single_transformer_blocks"
        module = f"{target}.{index}.{table[inner]}"
    else:
        module = _ROOT[module]
    if parameter == "scale":
        parameter = "weight"
    return f"{module}.{parameter}"


def map_bfl_weight(name: str, weight: torch.Tensor) -> tuple[str, torch.Tensor]:
    name = map_bfl_name(name)
    if name in ("norm_out.linear.weight", "norm_out.linear.bias"):
        scale, shift = weight.chunk(2, dim=0)
        weight = torch.cat((shift, scale), dim=0)
    return name, weight


def quantized_layer_names(metadata: dict[str, object]) -> Iterator[str]:
    layers = metadata["layers"]
    if not isinstance(layers, dict):
        raise ValueError("NVFP4 metadata must contain a layer mapping")
    for name, spec in layers.items():
        if not isinstance(name, str) or spec != {"format": "nvfp4"}:
            raise ValueError("Only the official uniform NVFP4 layer format is supported")
        yield map_bfl_name(name + ".weight").removesuffix(".weight")
