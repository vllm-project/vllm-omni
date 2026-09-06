# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Helpers for resolving a model reference to a locally-readable config source."""

from __future__ import annotations

import functools

from vllm.logger import init_logger
from vllm.transformers_utils.runai_utils import ObjectStorageModel, is_runai_obj_uri

logger = init_logger(__name__)


@functools.cache
def materialize_object_storage_configs(model: str) -> str:
    """Materialize an object-storage model URI's config files locally.

    vLLM's Run:AI streamer keeps ``s3://``/``gs://``/``az://`` URIs opaque until
    each stage builds its ``ModelConfig``; parent-process resolution (HF config
    lookup, pipeline/pipeline-key matching) would instead hand the URI to
    ``huggingface_hub`` helpers, which reject it with ``HFValidationError``.
    Pull the lightweight files once into vLLM's deterministic
    ``model_streamer/<hash>`` directory so config reads work here, and so the
    stage processes' own pull lands in that same directory.

    Returns the input unchanged for non object-storage paths.
    """
    if not is_runai_obj_uri(model):
        return model
    object_storage_model = ObjectStorageModel(url=model)
    object_storage_model.pull_files(model, allow_pattern=["*.model", "*.py", "*.json"])
    logger.info("Materialized object-storage configs for %s at %s", model, object_storage_model.dir)
    return object_storage_model.dir
