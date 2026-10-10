# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import NamedTuple, cast

import numpy as np
import torch

from vllm_omni.outputs.output_modality import OutputModalityNames

MediaToTensor = Callable[[object], torch.Tensor]
TensorToMedia = Callable[[torch.Tensor, object], object]


class ConverterPair(NamedTuple):
    """Utils for converting a given modality to tensors & back."""

    to_tensor: MediaToTensor
    restore: TensorToMedia


def media_to_tensor(data: object, converter: MediaToTensor) -> torch.Tensor:
    """Convert nested media values to one tensor."""
    if isinstance(data, list):
        return torch.stack([media_to_tensor(item, converter) for item in data])
    return converter(data)


def restore_media(data: torch.Tensor, source: object, converter: TensorToMedia) -> object:
    """Restore a tensor to its nested media value types."""
    if isinstance(source, list):
        return [restore_media(item, template, converter) for item, template in zip(data, source, strict=True)]
    return converter(data, source)


def array_to_tensor(data: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Try to convert raw multimodal data to a tensor (modality agnostic)."""
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    raise TypeError(f"unsupported output type: {type(data).__name__}")


def restore_array(data: torch.Tensor, source: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Try to restore a data tensor to its original type (modality agnostic)."""
    if isinstance(source, torch.Tensor):
        return data
    if isinstance(source, np.ndarray):
        return data.numpy()
    raise TypeError(f"unsupported output type: {type(source).__name__}")


MEDIA_CONVERTERS: Mapping[OutputModalityNames, ConverterPair] = {
    OutputModalityNames.AUDIO: ConverterPair(cast(MediaToTensor, array_to_tensor), cast(TensorToMedia, restore_array)),
}
