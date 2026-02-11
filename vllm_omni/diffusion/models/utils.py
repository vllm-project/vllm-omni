from typing import Literal, TYPE_CHECKING

import torch
import torch.nn as nn

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.quantization import get_vllm_quant_config_for_layers
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.models.utils import maybe_prefix

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )


Style = Literal["colwise", "colwise_rep", "rowwise", "rowwise_rep", "replicate"]


def replace_linear_class(
    linear: nn.Linear,
    style: Style = "replicate",
    quant_config: "QuantizationConfig | None" = None,
    *,
    prefix: str = "",
) -> ColumnParallelLinear | RowParallelLinear | ReplicatedLinear:
    """
    Replace nn.Linear with one of vLLM's tensor parallel linear classes.

    Args:
        linear: `nn.Linear` to be replaced.
        style: Tensor parallel style of the new linear, e.g. "colwise".
        quant_config: Quantization config for the new linear.
    Returns:
        The new linear.
    """

    if not isinstance(style, str):
        raise ValueError(f"Unsupported parallel style type {type(style)}, expected str")

    vllm_linear_cls, vllm_linear_kwargs = {
        "colwise": (ColumnParallelLinear, {}),
        "colwise_rep": (ColumnParallelLinear, {"gather_output": True}),
        "rowwise": (RowParallelLinear, {}),
        "rowwise_rep": (RowParallelLinear, {"input_is_parallel": False}),
        "replicate": (ReplicatedLinear, {}),
    }.get(style, (ReplicatedLinear, {}))

    return vllm_linear_cls(
        input_size=linear.in_features,
        output_size=linear.out_features,
        bias=linear.bias is not None,
        quant_config=quant_config,
        prefix=prefix,
        return_bias=False,
        **vllm_linear_kwargs,
    )


def recursive_replace_linear(model: nn.Module, od_config: OmniDiffusionConfig):
    """Recursively replace modules in the model as needed.
    Currently, this replaces:
    - `nn.Linear` with vLLM's tensor parallel linear classes
    - `*RMSNorm` with vLLM's `RMSNorm`
    """
    # Prefix the patterns because we always start from `self.model`
    quant_config = get_vllm_quant_config_for_layers(od_config.quantization_config)
    def _recursive_replace(module: nn.Module, prefix: str):
        for child_name, child_module in module.named_children():
            new_module = child_module
            qual_name = maybe_prefix(prefix, child_name)
            # Replace modules as needed
            if isinstance(child_module, nn.Linear):
                style = "replicate"
                new_module = replace_linear_class(
                    child_module, style, quant_config, prefix=qual_name
                )
            else:
                _recursive_replace(child_module, prefix=qual_name)
            if new_module is not child_module:
                setattr(module, child_name, new_module)
    _recursive_replace(model, prefix="")


def init_parameters(module: nn.Module, dtype: torch.dtype | None, device: torch.device | None = None,):
    for name, param in module.named_parameters(recurse=False):
        if param.device == torch.device("meta"):
            new_param = nn.Parameter(
                torch.empty_like(
                    param.data,
                    dtype=dtype,
                    device=device,
                )
            )
            setattr(module, name, new_param)
    for child in module.children():
        init_parameters(child, dtype, device)