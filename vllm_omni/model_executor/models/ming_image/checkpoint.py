# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Checkpoint resolution for the Ming-Image MLLM stage."""

from pathlib import Path

from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)

_MLLM_REQUIRED_PATTERNS = [
    "mllm/**",
    "mlp/**",
]


def resolve_ming_image_model_root(
    model: str,
    revision: str | None,
    task_type: str | None,
) -> str:
    """Materialize the MLLM and its sibling query-token checkpoint.

    The autoregressive weights and tokenizer live under mllm/,
    while the learned image-query tokens consumed by that stage live under mlp/.
    Stage subdirectory resolution runs after this hook, so return the repository
    root and let model_subdir/tokenizer_subdir select mllm/.
    """
    del task_type
    path = Path(model)
    if path.is_dir():
        if path.name == "mllm" and (path / "config.json").is_file():
            return str(path.parent)
        return str(path)

    return download_weights_from_hf_specific(
        model_name_or_path=model,
        cache_dir=None,
        allow_patterns=_MLLM_REQUIRED_PATTERNS,
        revision=revision,
        require_all=True,
    )


__all__ = ["resolve_ming_image_model_root"]
