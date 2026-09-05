# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import hashlib
from typing import Any

import torch
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

_PARAMETER_SUFFIXES = (
    "language_model.model.layers.0.self_attn.qkv_proj.weight",
    "language_model.model.layers.0.self_attn.o_proj.weight",
    "language_model.model.layers.0.mlp.gate_up_proj.weight",
    "language_model.model.layers.0.mlp.down_proj.weight",
)


class LlamaOmni2ValidationWorkerExtension:
    def llama_omni2_parameter_shapes(self) -> dict[str, Any]:
        parameters = dict(self.model_runner.model.named_parameters())
        matched = {
            suffix: {
                "name": name,
                "shape": list(parameter.shape),
                "numel": parameter.numel(),
                "sha256": hashlib.sha256(
                    parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
                ).hexdigest(),
            }
            for suffix in _PARAMETER_SUFFIXES
            for name, parameter in parameters.items()
            if name.endswith(suffix)
        }
        if set(matched) != set(_PARAMETER_SUFFIXES):
            missing = sorted(set(_PARAMETER_SUFFIXES) - set(matched))
            raise RuntimeError(f"LLaMA-Omni 2 validation worker is missing parameters: {missing}")
        return {
            "tp_rank": get_tensor_model_parallel_rank(),
            "tp_world_size": get_tensor_model_parallel_world_size(),
            "parameters": matched,
        }
