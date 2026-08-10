# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve Omni-Diffusion side-component model directories."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download
from vllm.logger import init_logger

OMNI_DIFFUSION_IMAGE_TOKENIZER_REPO_ID = "showlab/magvitv2"
OMNI_DIFFUSION_SENSEVOICE_REPO_ID = "FunAudioLLM/SenseVoiceSmall"
OMNI_DIFFUSION_FLOW_DECODER_REPO_ID = "THUDM/glm-4-voice-decoder"

logger = init_logger(__name__)


def resolve_omni_diffusion_component_path(
    configured_path: object,
    *,
    config_key: str,
    default_repo_id: str,
) -> str:
    """Resolve a local override or download the default component snapshot.

    A non-empty config value is always interpreted as an explicit local
    directory. ``None`` or an empty string selects the component's official
    Hugging Face repository and returns its cached snapshot directory.
    """
    if configured_path is not None and not isinstance(configured_path, str):
        raise TypeError(
            f"Omni-Diffusion {config_key} must be a local directory path or null, got {type(configured_path)!r}."
        )

    if isinstance(configured_path, str) and configured_path.strip():
        path = Path(configured_path).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(
                f"Omni-Diffusion {config_key} must be an existing local directory, got {str(path)!r}."
            )
        return str(path.resolve())

    logger.info(
        "Omni-Diffusion %s is unset; downloading or reusing Hugging Face snapshot %s.",
        config_key,
        default_repo_id,
    )
    return snapshot_download(repo_id=default_repo_id)
