# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib

from vllm.logger import init_logger

from vllm_omni.diffusion.models.dreamzero.transform.base import TRANSFORMS

logger = init_logger(__name__)

DEFAULT_EMBODIMENT = "roboarena"
_BUILTIN_TRANSFORM_MODULES = (
    "vllm_omni.diffusion.models.dreamzero.transform.droid",
    "vllm_omni.diffusion.models.dreamzero.transform.roboarena",
)


def ensure_transforms_loaded() -> None:
    """Import DreamZero transform modules and verify registration."""
    for module_name in _BUILTIN_TRANSFORM_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            logger.exception("Failed to import DreamZero transform module %s", module_name)
            raise RuntimeError(f"Failed to import DreamZero transform module '{module_name}'.") from exc

    if DEFAULT_EMBODIMENT not in TRANSFORMS:
        raise RuntimeError(f"Built-in DreamZero transform '{DEFAULT_EMBODIMENT}' is not registered after import.")
