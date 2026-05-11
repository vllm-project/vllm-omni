# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from torch import nn

from .modelopt import ModelOptFp8CheckpointAdapter


def get_checkpoint_adapter(
    model: nn.Module,
    source: object,
    quant_config: object | None,
    use_safetensors: bool,
) -> ModelOptFp8CheckpointAdapter | None:
    if ModelOptFp8CheckpointAdapter.is_compatible(source, quant_config, use_safetensors):
        return ModelOptFp8CheckpointAdapter(model, source)
    return None


__all__ = [
    "ModelOptFp8CheckpointAdapter",
    "get_checkpoint_adapter",
]
