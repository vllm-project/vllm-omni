# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterable

import torch
from vllm.model_executor.models.utils import WeightsMapper

from .base import GGUFAdapter, gguf_quant_weights_iterator

Z_IMAGE_KEYS_RENAME_DICT = {
    "final_layer.": "all_final_layer.2-1.",
    "x_embedder.": "all_x_embedder.2-1.",
    ".attention.qkv": ".attention.to_qkv",
    ".attention.k_norm": ".attention.norm_k",
    ".attention.q_norm": ".attention.norm_q",
    ".attention.out": ".attention.to_out.0",
    "model.diffusion_model.": "",
}


class ZImageGGUFAdapter(GGUFAdapter):
    """GGUF adapter for Z-Image models with QKV/FFN shard support."""

    @staticmethod
    def is_compatible(od_config, model: torch.nn.Module, source) -> bool:
        model_class = od_config.model_class_name or ""
        if model_class.startswith("ZImage"):
            return True
        cfg = od_config.tf_model_config
        if cfg is not None:
            model_type = str(cfg.get("model_type", "")).lower()
            if model_type in {"z_image", "zimage", "z-image"}:
                return True
        return False

    gguf_to_hf_mapper = WeightsMapper(
        orig_to_new_substr=Z_IMAGE_KEYS_RENAME_DICT,
    )

    def weights_iterator(self) -> Iterable[tuple[str, torch.Tensor]]:
        weights = gguf_quant_weights_iterator(self.gguf_file)
        yield from self.gguf_to_hf_mapper.apply(weights)
