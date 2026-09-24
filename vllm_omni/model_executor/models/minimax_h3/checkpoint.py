# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax H3 checkpoint resolution for the AR text-encoder stage."""

from pathlib import Path

from vllm_omni.diffusion.utils.hf_utils import get_diffusion_model_index
from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)

_FASTH3_MODULAR_MODEL_PREFIX = "FastVideo/"

MINIMAX_H3_ENCODER_DOWNLOAD_PATTERNS = {
    partition: [
        f"{subdir}/text_encoder/**",
        f"{subdir}/video_vae/**",
        f"{subdir}/audio_vae/**",
    ]
    for partition, subdir in {"fl2va": "FL2VA", "ref2va": "Ref2VA"}.items()
}
MINIMAX_H3_ENCODER_DOWNLOAD_PATTERNS["combined"] = MINIMAX_H3_ENCODER_DOWNLOAD_PATTERNS["fl2va"]


def is_minimax_h3_modular(model: str, revision: str | None = None) -> bool:
    path = Path(model)
    if path.is_dir():
        # Native MiniMax-H3 snapshots can advertise the Diffusers modular
        # class in ``model_index.json`` while retaining the legacy FL2VA /
        # Ref2VA layout.  FastH3 modular releases carry this marker file.
        return (path / "modular_model_index.json").is_file() or (path / "fastvideo_inference.json").is_file()
    # The FastVideo release is the modular checkpoint handled by this
    # resolver.  Do not classify the native MiniMax-H3 Hub repository from
    # its class name alone; its partition files are still required here.
    if not model.startswith(_FASTH3_MODULAR_MODEL_PREFIX):
        return False
    index = get_diffusion_model_index(model, revision=revision) or {}
    return index.get("_class_name") == "MiniMaxH3ModularPipeline"


def resolve_minimax_h3_partition(
    model: str,
    task_type: str | None,
    *,
    auto_partition: str,
) -> str:
    task = str(task_type or "auto").lower()
    if task not in {"auto", "combined", "t2va", "fl2va", "ref2va"}:
        raise ValueError(
            f"MiniMax-H3 task_type must be one of auto, combined, t2va, fl2va, or ref2va; got {task_type!r}"
        )
    path = Path(model)
    if task == "auto" and path.is_dir() and path.name in {"FL2VA", "Ref2VA"}:
        return path.name.lower()
    if task in {"auto", "combined"}:
        return auto_partition
    return "ref2va" if task == "ref2va" else "fl2va"


def resolve_minimax_h3_encoder_model_root(
    model: str,
    revision: str | None,
    task_type: str | None,
) -> str:
    path = Path(model)
    partition = resolve_minimax_h3_partition(model, task_type, auto_partition="fl2va")
    modular = is_minimax_h3_modular(model, revision)

    if path.is_dir():
        if path.name == "text_encoder" and (path / "config.json").is_file():
            return str(path)
        if modular:
            return str(path / "text_encoder")
        if path.name in {"FL2VA", "Ref2VA"}:
            path = path.parent
        subdir = "Ref2VA" if partition == "ref2va" else "FL2VA"
        return str(path / subdir / "text_encoder")
    snapshot = download_weights_from_hf_specific(
        model_name_or_path=model,
        cache_dir=None,
        allow_patterns=["text_encoder/**"] if modular else MINIMAX_H3_ENCODER_DOWNLOAD_PATTERNS[partition],
        revision=revision,
        require_all=True,
    )

    if modular:
        return str(Path(snapshot) / "text_encoder")
    subdir = "Ref2VA" if partition == "ref2va" else "FL2VA"
    return str(Path(snapshot) / subdir / "text_encoder")


resolve_minimax_h3_model_root = resolve_minimax_h3_encoder_model_root


__all__ = [
    "resolve_minimax_h3_encoder_model_root",
    "resolve_minimax_h3_model_root",
    "resolve_minimax_h3_partition",
]
