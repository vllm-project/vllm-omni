# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import gguf
import numpy as np
import torch

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.model_loader.diffusers_loader import (
        DiffusersPipelineLoader,
    )


@dataclass
class MappedTensor:
    name: str
    tensor: Any
    tensor_type: Any
    row_slice: slice | None = None
    swap_scale_shift: bool = False


class GGUFAdapter(ABC):
    """Base class for model-specific GGUF adapters."""

    def __init__(
        self,
        gguf_file: str,
        model: torch.nn.Module,
        source: DiffusersPipelineLoader.ComponentSource,
        od_config: OmniDiffusionConfig,
    ) -> None:
        self.gguf_file = gguf_file
        self.model = model
        self.source = source
        self.od_config = od_config

    @staticmethod
    def is_compatible(
        od_config: OmniDiffusionConfig,
        model: torch.nn.Module,
        source: DiffusersPipelineLoader.ComponentSource,
    ) -> bool:
        return False

    @abstractmethod
    def weights_iterator(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        raise NotImplementedError


# FIXME(Isotr0py): Sync implemnentation with upstream vLLM?
def gguf_quant_weights_iterator(gguf_file: str) -> Generator[tuple[str, torch.Tensor]]:
    """
    Iterate over the quant weights in the model gguf files and convert
    them to torch tensors.
    Be careful of the order of yielding weight types and weights data,
    we have to yield all weight types first before yielding any weights.
    Otherwise it would cause issue when loading weights with for packed
    layer with different quant types.
    """

    reader = gguf.GGUFReader(gguf_file)

    for tensor in reader.tensors:
        weight_type = tensor.tensor_type
        name = tensor.name

        if weight_type.name not in ("F32", "F16"):
            weight_type_name = name.replace("weight", "qweight_type")
            weight_type = torch.tensor(weight_type)
            yield weight_type_name, weight_type

    for tensor in reader.tensors:
        weight = tensor.data
        weight_type = tensor.tensor_type
        name = tensor.name
        if weight_type.name not in ("F32", "F16"):
            name = name.replace("weight", "qweight")
        if weight_type.name == "BF16" and tensor.data.dtype == np.uint8:
            # BF16 is currently the only "quantization" type that isn't
            # actually quantized but is read as a raw byte tensor.
            # Reinterpret as `torch.bfloat16` tensor.
            weight = weight.view(np.uint16)
            if reader.byte_order == "S":
                # GGUF endianness != system endianness
                weight = weight.byteswap()
            param = torch.tensor(weight).view(torch.bfloat16)
        else:
            param = torch.tensor(weight)
        yield name, param
