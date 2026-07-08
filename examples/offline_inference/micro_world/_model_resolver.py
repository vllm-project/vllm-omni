# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Resolve AMD Micro-World model layouts for the offline examples.

AMD/Micro-World stores the Micro-World-specific transformer and LoRA weights
under T2W/ and I2W/. The Wan2.1 base components remain in Wan-AI repositories.
These helpers compose an internal local diffusers-style directory so
vLLM-Omni can load Micro-World from a single user-facing model id.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

MICRO_WORLD_REPO = "AMD/Micro-World"

_VARIANTS = {
    "T2W": {
        "base_repo": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "class_name": "MicroWorldT2WPipeline",
        "transformer": "MicroWorldControlNetTransformer",
        "base_subfolders": ("scheduler", "tokenizer", "text_encoder", "vae"),
        "model_index": {
            "scheduler": ["diffusers", "UniPCMultistepScheduler"],
            "text_encoder": ["transformers", "UMT5EncoderModel"],
            "tokenizer": ["transformers", "T5TokenizerFast"],
            "transformer": ["diffusers", "MicroWorldControlNetTransformer"],
            "vae": ["diffusers", "AutoencoderKLWan"],
        },
    },
    "I2W": {
        "base_repo": "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        "class_name": "MicroWorldI2WPipeline",
        "transformer": "MicroWorldAdaLNTransformer",
        "base_subfolders": ("scheduler", "tokenizer", "text_encoder", "image_processor", "image_encoder", "vae"),
        "model_index": {
            "scheduler": ["diffusers", "UniPCMultistepScheduler"],
            "text_encoder": ["transformers", "UMT5EncoderModel"],
            "tokenizer": ["transformers", "T5TokenizerFast"],
            "image_encoder": ["transformers", "CLIPVisionModel"],
            "image_processor": ["transformers", "CLIPImageProcessor"],
            "transformer": ["diffusers", "MicroWorldAdaLNTransformer"],
            "vae": ["diffusers", "AutoencoderKLWan"],
        },
    },
}


def _safe_name(value: str) -> str:
    return value.replace("/", "--")


def _download_repo(repo_id: str, *, allow_patterns: list[str]) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo_id, allow_patterns=allow_patterns))


def _link_or_keep(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src, target_is_directory=src.is_dir())


def _write_model_index(path: Path, variant: str) -> None:
    spec = _VARIANTS[variant]
    model_index = {
        "_class_name": spec["class_name"],
        "_diffusers_version": "0.38.0",
        **spec["model_index"],
    }
    index_path = path / "model_index.json"
    if index_path.exists():
        return
    index_path.write_text(json.dumps(model_index, indent=2) + "\n")


def resolve_micro_world_model(model: str, variant: str) -> str:
    """Return a loadable local model path for a Micro-World example.

    The consolidated AMD/Micro-World repo is expanded into an internal local
    load directory containing Micro-World transformer/LoRA files plus symlinks
    to the Wan2.1 base components needed by the selected variant.
    """
    variant = variant.upper()
    if variant not in _VARIANTS:
        raise ValueError(f"Unknown Micro-World variant {variant!r}; expected one of {sorted(_VARIANTS)}")

    spec = _VARIANTS[variant]
    micro_world_root = _download_repo(
        model,
        allow_patterns=[f"{variant}/transformer/*", f"{variant}/lora_diffusion_pytorch_model.safetensors"],
    )

    base_root = _download_repo(
        spec["base_repo"],
        allow_patterns=[f"{subfolder}/*" for subfolder in spec["base_subfolders"]],
    )

    micro_world_variant = micro_world_root / variant
    cache_root = Path(
        os.environ.get(
            "VLLM_OMNI_MICRO_WORLD_CACHE",
            Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "vllm-omni" / "micro_world",
        )
    )
    combined = cache_root / f"{_safe_name(model)}-{variant}-{micro_world_root.name}-{base_root.name}"
    combined.mkdir(parents=True, exist_ok=True)

    _link_or_keep(micro_world_variant / "transformer", combined / "transformer")
    _link_or_keep(
        micro_world_variant / "lora_diffusion_pytorch_model.safetensors",
        combined / "lora_diffusion_pytorch_model.safetensors",
    )
    for subfolder in spec["base_subfolders"]:
        _link_or_keep(base_root / subfolder, combined / subfolder)
    _write_model_index(combined, variant)

    return str(combined)
