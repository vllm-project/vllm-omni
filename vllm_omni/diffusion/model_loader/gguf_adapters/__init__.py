# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .base import GGUFAdapter
from .flux2_klein import Flux2KleinGGUFAdapter
from .z_image import ZImageGGUFAdapter

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.model_loader.diffusers_loader import (
        DiffusersPipelineLoader,
    )


def get_gguf_adapter(
    gguf_file: str,
    model: torch.nn.Module,
    source: DiffusersPipelineLoader.ComponentSource,
    od_config: OmniDiffusionConfig,
) -> GGUFAdapter:
    adapter_classes = (ZImageGGUFAdapter, Flux2KleinGGUFAdapter)
    for adapter_cls in adapter_classes:
        if adapter_cls.is_compatible(od_config, model, source):
            return adapter_cls(gguf_file, model, source, od_config)
    model_type = None
    if od_config.tf_model_config is not None:
        model_type = od_config.tf_model_config.get("model_type")
    supported = ", ".join(cls.__name__ for cls in adapter_classes)
    raise ValueError(
        "No GGUF adapter matched diffusion model "
        f"(model_class_name={od_config.model_class_name!r}, model_type={model_type!r}). "
        f"Supported adapters: {supported}."
    )


__all__ = [
    "GGUFAdapter",
    "Flux2KleinGGUFAdapter",
    "ZImageGGUFAdapter",
    "get_gguf_adapter",
]
