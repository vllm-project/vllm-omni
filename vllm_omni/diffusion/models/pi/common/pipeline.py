# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stateless construction helpers shared by Pi-family pipelines."""

import os
from numbers import Integral
from typing import Any

import torch

_CHECKPOINT_ALLOW_PATTERNS = ("*.json", "*.safetensors", "*.model", "tokenizer*")


def resolve_num_inference_steps(sampling_params: Any) -> int | None:
    """Read and validate the top-level per-request denoising-step override.

    ``sampling_params.num_inference_steps`` is the single source of truth;
    legacy values under ``extra_args`` are deliberately ignored.
    """
    value = getattr(sampling_params, "num_inference_steps", None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"num_inference_steps must be a positive integer, got {value!r}.")
    return int(value)


def identity_post_process(value):
    """Return pipeline output unchanged across the process boundary."""
    return value


def get_identity_post_process_func(od_config: Any):
    """Return the module-level identity function expected by the registry."""
    del od_config
    return identity_post_process


def resolve_model_dir(model: str | None) -> str | None:
    """Return a local checkpoint directory, downloading an HF repo if needed."""
    if not model:
        return None
    if os.path.isdir(model):
        return model

    # Use vLLM-Omni's shared HfApi so downloads carry the project user agent.
    from vllm_omni.transformers_utils.repo_utils import hf_api

    return hf_api().snapshot_download(
        repo_id=model,
        allow_patterns=list(_CHECKPOINT_ALLOW_PATTERNS),
    )


def resolve_tokenizer_source(model_dir: str | None, fallback: str) -> str:
    """Prefer checkpoint tokenizer metadata, otherwise use ``fallback``."""
    if model_dir and os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, "tokenizer_config.json")):
        return model_dir
    return fallback


def resolve_device() -> torch.device:
    """Use the diffusion worker's local device with a standalone fallback."""
    from vllm_omni.diffusion.distributed.utils import get_local_device

    try:
        return get_local_device()
    except Exception:  # noqa: BLE001
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def has_safetensors_checkpoint(model_dir: str | None) -> bool:
    """Return whether ``model_dir`` contains the self-loaded weight file."""
    return bool(model_dir) and os.path.exists(os.path.join(model_dir, "model.safetensors"))
