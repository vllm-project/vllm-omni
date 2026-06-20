# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from .pipeline_janus import JanusPipeline, get_janus_post_process_func
from .pipeline_janus_vq import JanusVQDecodePipeline


def _register_janus_hf_classes() -> None:
    """Register ``multi_modality`` with Transformers.

    The public ``deepseek-ai/Janus-1.3B`` repo ships weights + JSON only (no ``modeling_*.py``),
    so ``trust_remote_code`` cannot load the architecture. We vendor DeepSeek's registration
    from https://github.com/deepseek-ai/Janus (Apache-2.0 / MIT licensed code).
    """
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if "multi_modality" in CONFIG_MAPPING:
        return
    from vllm_omni.diffusion.models.deepseek_janus._janus_hf_vendor import modeling_vlm  # noqa: F401


def _load_param(module: torch.nn.Module, name: str, tensor: torch.Tensor) -> None:
    """Load a single parameter into a module, supporting dotted paths and integer indices."""
    parts = name.split(".")
    obj = module
    for part in parts[:-1]:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    final = parts[-1]
    if final.isdigit():
        target = obj[int(final)]
    else:
        target = getattr(obj, final)
    if isinstance(target, torch.nn.Parameter):
        target.data.copy_(tensor)
    elif isinstance(target, torch.Tensor):
        target.copy_(tensor)
    else:
        raise TypeError(f"Cannot load weight into {type(target)} at {name}")


__all__ = [
    "JanusPipeline",
    "JanusVQDecodePipeline",
    "get_janus_post_process_func",
    "_load_param",
    "_register_janus_hf_classes",
]
