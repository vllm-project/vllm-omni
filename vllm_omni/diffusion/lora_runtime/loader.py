# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from pathlib import Path

from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)

from .types import DiffusionLoRADeployment


def resolve_diffusion_lora_artifact(deployment: DiffusionLoRADeployment) -> Path:
    """Resolve a deployment path once during worker startup."""

    local_path = Path(deployment.path).expanduser()
    if local_path.exists():
        # Keep the snapshot-facing filename. Hugging Face cache files are
        # symlinks to extensionless blobs, while model loaders may use the
        # published suffix to select and validate the checkpoint format.
        return local_path.absolute()

    resolved = download_weights_from_hf_specific(
        deployment.path,
        cache_dir=None,
        allow_patterns=["*.safetensors"],
    )
    return Path(resolved)
