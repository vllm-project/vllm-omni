# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterable

import torch

from .base import GGUFAdapter, gguf_quant_weights_iterator


class QwenImageGGUFAdapter(GGUFAdapter):
    """GGUF adapter for the Qwen-Image transformer family."""

    @staticmethod
    def is_compatible(od_config, model: torch.nn.Module, source) -> bool:
        model_class = od_config.model_class_name or ""
        if model_class.startswith("QwenImage"):
            return True
        cfg = od_config.tf_model_config
        if cfg is not None:
            model_type = str(cfg.get("model_type", "")).lower()
            if model_type.startswith("qwen_image"):
                return True
        return False

    def weights_iterator(self) -> Iterable[tuple[str, torch.Tensor]]:
        yield from gguf_quant_weights_iterator(self.gguf_file)
