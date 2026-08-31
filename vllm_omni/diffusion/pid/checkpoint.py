# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Simple checkpoint loader for PiD model weights."""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from pathlib import Path

import torch

from vllm_omni.diffusion.pid.config import _PID_HF_REPO, PID_CHECKPOINT_REGISTRY
from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)

logger = logging.getLogger(__name__)

# Prefixes of training-only modules removed from the inference-only model;
# their checkpoint weights are filtered out before loading.
_DROPPED_PREFIXES = ("lq_proj.lq_aux_rgb_head",)


def _download_pid_file(repo_id: str, filename: str) -> str:
    hf_folder = download_weights_from_hf_specific(
        model_name_or_path=repo_id,
        cache_dir=None,
        allow_patterns=[filename],
        require_all=True,
    )
    return os.path.join(hf_folder, filename)


def resolve_pid_checkpoint_path(
    checkpoint_path: str | None,
    backbone: str | None = None,
) -> str:
    if checkpoint_path:
        if Path(checkpoint_path).is_file():
            return checkpoint_path
        parts = checkpoint_path.split("/")
        if len(parts) >= 3 and "." in parts[-1]:
            repo_id = "/".join(parts[:2])
            filename = "/".join(parts[2:])
            return _download_pid_file(repo_id, filename)
        raise ValueError(
            f"pid_checkpoint must be a local .pth path or an HF reference "
            f"'<repo_id>/<subfolder>/<file>' (got {checkpoint_path!r})"
        )

    if backbone is None:
        raise ValueError(
            "No --pid-checkpoint configured and no backbone given to auto-select a default PiD checkpoint."
        )
    try:
        _, in_repo_path, _ = PID_CHECKPOINT_REGISTRY[backbone]
    except KeyError:
        raise ValueError(
            f"No default PiD checkpoint registered for backbone {backbone!r}; set --pid-checkpoint explicitly."
        ) from None
    return _download_pid_file(_PID_HF_REPO, in_repo_path)


def load_pid_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | None,
    backbone: str | None = None,
) -> None:
    """Load PiD checkpoint, stripping 'net.' prefix from state dict keys.

    PiD checkpoints saved via ``PidDistillModel.state_dict()`` have all keys
    prefixed with ``"net."``. We strip this to match ``PidInferenceModel.net``
    (a ``PidNet`` instance).

    Missing LQ-projection keys are expected when loading a checkpoint that
    was fine-tuned from a base T2I model (LQ modules are zero-init anyway).
    """
    local_path = resolve_pid_checkpoint_path(checkpoint_path, backbone)
    state_dict = torch.load(local_path, map_location="cpu", weights_only=True)

    net_sd = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("net.") and not k.startswith("net_ema."):
            net_sd[k[len("net.") :]] = v

    # Drop keys of training-only modules that were removed from the
    # inference-only model (e.g. the aux RGB supervision head).
    for prefix in _DROPPED_PREFIXES:
        for k in [k for k in net_sd if k.startswith(prefix)]:
            del net_sd[k]

    missing, unexpected = model.net.load_state_dict(net_sd, strict=False)

    lq_missing = [k for k in missing if "lq_proj" in k or "pit_lq" in k]
    other_missing = [k for k in missing if "lq_proj" not in k and "pit_lq" not in k]

    if lq_missing:
        logger.info(
            "Expected missing LQ keys (%d keys) -- LQ modules are zero-init.",
            len(lq_missing),
        )
    if other_missing:
        logger.warning("Missing keys: %s", other_missing)
    if unexpected:
        logger.warning("Unexpected keys: %s", unexpected)
