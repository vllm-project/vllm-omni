# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 checkpoint resolution for the AR text-encoder stage."""

from pathlib import Path

from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)

MINIMAX_H3_TEXT_ENCODER_DOWNLOAD_PATTERNS = {
    "fl2va": ["FL2VA/text_encoder/**"],
    "ref2va": ["Ref2VA/text_encoder/**"],
}


def resolve_minimax_h3_model_root(
    model: str,
    revision: str | None,
    task_type: str | None,
) -> str:
    task = str(task_type or "auto").lower()
    path = Path(model)
    if task == "auto" and path.is_dir() and path.name in {"FL2VA", "Ref2VA"}:
        partition = path.name.lower()
    elif task in {"auto", "combined", "t2va", "fl2va"}:
        partition = "fl2va"
    elif task == "ref2va":
        partition = "ref2va"
    else:
        raise ValueError(
            f"MiniMax-H3 task_type must be one of auto, combined, t2va, fl2va, or ref2va; got {task_type!r}"
        )

    if path.is_dir():
        if path.name == "text_encoder" and (path / "config.json").is_file():
            return str(path)
        if path.name in {"FL2VA", "Ref2VA"}:
            path = path.parent
        subdir = "Ref2VA" if partition == "ref2va" else "FL2VA"
        return str(path / subdir / "text_encoder")
    snapshot = download_weights_from_hf_specific(
        model_name_or_path=model,
        cache_dir=None,
        allow_patterns=MINIMAX_H3_TEXT_ENCODER_DOWNLOAD_PATTERNS[partition],
        revision=revision,
        require_all=True,
    )

    subdir = "Ref2VA" if partition == "ref2va" else "FL2VA"
    return str(Path(snapshot) / subdir / "text_encoder")


__all__ = ["resolve_minimax_h3_model_root"]
