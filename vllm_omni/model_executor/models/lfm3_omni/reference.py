# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Load checkpoint-defined encoders with the same revision as its weights."""

import importlib
from types import ModuleType

from transformers.dynamic_module_utils import get_class_from_dynamic_module
from vllm.config import ModelConfig


def reference_module(model_config: ModelConfig, module: str, name: str) -> ModuleType:
    if not model_config.trust_remote_code:
        raise ValueError("LFM3-Omni requires trust_remote_code=True for its checkpoint-defined encoders and codec")
    cls = get_class_from_dynamic_module(
        f"{module}.{name}",
        model_config.model,
        revision=model_config.revision,
        code_revision=model_config.code_revision,
    )
    return importlib.import_module(cls.__module__)
