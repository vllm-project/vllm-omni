"""
Utilities for resolving real models to their tiny model equivalents.
"""

import logging

from tests.model_tests.diffusion.model_settings import DIFFUSION_TEST_SETTINGS
from vllm_omni.diffusion.data import resolve_model_class_name

logger = logging.getLogger(__name__)


def resolve_tiny_model_path(model: str) -> str:
    """Given a real model name/path, resolve it to a tiny model path.

    Raises ValueError if the pipeline class cannot be determined (invalid
    model). Returns the original model path if no tiny builder exists yet."""
    pipeline_class = resolve_model_class_name(model)
    if pipeline_class is None:
        raise ValueError(
            f"Cannot resolve pipeline class for model: {model}. The model path may be invalid or its config unreadable."
        )

    test_opts = DIFFUSION_TEST_SETTINGS.get(pipeline_class)
    if test_opts is None:
        logger.warning(
            "No tiny model builder for pipeline %s (model: %s). Using original model.",
            pipeline_class,
            model,
        )
        return model

    return test_opts.builder()
