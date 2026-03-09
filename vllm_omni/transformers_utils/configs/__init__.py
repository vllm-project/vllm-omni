# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Custom model configs that cannot be expressed via HuggingFace Transformers alone,
following the same pattern as vllm.transformers_utils.configs.
"""

from __future__ import annotations

import importlib

_CLASS_TO_MODULE: dict[str, str] = {
    "Mammothmoda2Config": "vllm_omni.transformers_utils.configs.mammoth_moda2",
    "Mammothmoda2Qwen2_5_VLConfig": "vllm_omni.transformers_utils.configs.mammoth_moda2",
    "Mammothmoda2Qwen2_5_VLTextConfig": "vllm_omni.transformers_utils.configs.mammoth_moda2",
    "Mammothmoda2Qwen2_5_VLVisionConfig": "vllm_omni.transformers_utils.configs.mammoth_moda2",
}

__all__ = [
    "Mammothmoda2Config",
    "Mammothmoda2Qwen2_5_VLConfig",
    "Mammothmoda2Qwen2_5_VLTextConfig",
    "Mammothmoda2Qwen2_5_VLVisionConfig",
]


def __getattr__(name: str):
    if name in _CLASS_TO_MODULE:
        module_name = _CLASS_TO_MODULE[name]
        module = importlib.import_module(module_name)
        return getattr(module, name)

    raise AttributeError(f"module 'vllm_omni.transformers_utils.configs' has no attribute '{name}'")


def __dir__():
    return sorted(list(__all__))


# Eagerly import all config modules so their AutoConfig.register() side-effects
# run as soon as `vllm_omni.transformers_utils.configs` is imported.
from vllm_omni.transformers_utils.configs import mammoth_moda2 as _mammoth_moda2  # noqa: F401, E402
