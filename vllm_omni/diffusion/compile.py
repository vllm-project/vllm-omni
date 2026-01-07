# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch.nn as nn
from vllm.logger import init_logger

logger = init_logger(__name__)


def regionally_compile(model: nn.Module, *args: Any, **kwargs: dict[str, Any]) -> nn.Module:
    """
    Apply regional compilation to a PyTorch model.

    Args:
        model: The PyTorch model instance to compile
        *args, **kwargs: Arguments passed to torch.compile

    Returns:
        The same model instance (modified in-place)
    """
    # Get the list of repeated blocks from the model
    repeated_blocks = getattr(model, "_repeated_blocks", None)

    if not repeated_blocks:
        logger.warning("Regional compilation skipped because the model does not define `_repeated_blocks`.")
        return model

    # Check if we have modules with the specified class names
    has_compiled_region = False
    for submod in model.modules():
        if submod.__class__.__name__ in repeated_blocks:
            # Compile this submodule
            submod.compile(*args, **kwargs)
            has_compiled_region = True

    if not has_compiled_region:
        logger.warning(f"Regional compilation skipped because {repeated_blocks} classes are not found in the model.")

    return model
