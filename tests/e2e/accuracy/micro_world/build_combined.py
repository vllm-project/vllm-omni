"""Build Micro-World T2W and I2W "combined" model directories.

Downloads the AMD Micro-World transformer+LoRA from the consolidated
AMD/Micro-World repository and the matching Wan2.1 base components, then
assembles a directory with symlinks and the custom
``model_index.json`` so that vllm-omni routes to ``MicroWorld{T2W,I2W}Pipeline``
instead of stock Wan.

Usage::

    python -m tests.e2e.accuracy.micro_world.build_combined
    # prints MICRO_WORLD_T2W_MODEL=... and MICRO_WORLD_I2W_MODEL=... to stdout

Override the cache location with ``ROOT=/path``. Re-runs are idempotent.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import snapshot_download

_T2W_MODEL_INDEX = {
    "_class_name": "MicroWorldT2WPipeline",
    "_diffusers_version": "0.34.0",
    "scheduler": ["diffusers", "UniPCMultistepScheduler"],
    "text_encoder": ["transformers", "UMT5EncoderModel"],
    "tokenizer": ["transformers", "T5TokenizerFast"],
    "transformer": ["diffusers", "MicroWorldControlNetTransformer"],
    "vae": ["diffusers", "AutoencoderKLWan"],
}

_I2W_MODEL_INDEX = {
    "_class_name": "MicroWorldI2WPipeline",
    "_diffusers_version": "0.34.0",
    "scheduler": ["diffusers", "UniPCMultistepScheduler"],
    "text_encoder": ["transformers", "UMT5EncoderModel"],
    "tokenizer": ["transformers", "T5TokenizerFast"],
    "image_encoder": ["transformers", "CLIPVisionModel"],
    "image_processor": ["transformers", "CLIPImageProcessor"],
    "transformer": ["diffusers", "MicroWorldAdaLNTransformer"],
    "vae": ["diffusers", "AutoencoderKLWan"],
}


@dataclass(frozen=True)
class Layout:
    name: str  # e.g. "T2W"
    amd_repo: str
    base_repo: str
    amd_files: tuple[str, ...]  # files/dirs to symlink from the variant snapshot
    model_index: dict
    # Subdirs symlinked from the base snapshot. Drop "scheduler" and `model_index`-
    # only entries; derived automatically from model_index keys minus AMD-provided.
    base_subdirs: tuple[str, ...] = field(default_factory=tuple)


def _base_subdirs_from_index(model_index: dict, amd_provided: set[str]) -> tuple[str, ...]:
    skip = {"_class_name", "_diffusers_version", "scheduler"}
    return tuple(k for k in model_index if k not in skip and k not in amd_provided)


_AMD_PROVIDED = ("transformer", "lora_diffusion_pytorch_model.safetensors")

T2W = Layout(
    name="T2W",
    amd_repo="AMD/Micro-World",
    base_repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    amd_files=_AMD_PROVIDED,
    model_index=_T2W_MODEL_INDEX,
    base_subdirs=_base_subdirs_from_index(_T2W_MODEL_INDEX, {"transformer"}),
)
I2W = Layout(
    name="I2W",
    amd_repo="AMD/Micro-World",
    base_repo="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
    amd_files=_AMD_PROVIDED,
    model_index=_I2W_MODEL_INDEX,
    base_subdirs=_base_subdirs_from_index(_I2W_MODEL_INDEX, {"transformer"}),
)


def _hf_download(repo_id: str, local_dir: Path, allow_patterns: list[str] | None = None) -> Path:
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir), allow_patterns=allow_patterns)
    return local_dir


def _link(target: Path, link: Path) -> None:
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)


def build(layout: Layout, root: Path) -> Path:
    amd_dir = root / Path(layout.amd_repo).name
    base_dir = root / Path(layout.base_repo).name
    variant_dir = amd_dir / layout.name
    combined = root / f"Micro-World-{layout.name}-combined"

    if not (variant_dir / "transformer").exists():
        _hf_download(
            layout.amd_repo,
            amd_dir,
            allow_patterns=[
                f"{layout.name}/transformer/*",
                f"{layout.name}/lora_diffusion_pytorch_model.safetensors",
            ],
        )
    if not all((base_dir / sub).exists() for sub in layout.base_subdirs):
        _hf_download(layout.base_repo, base_dir, allow_patterns=[f"{sub}/*" for sub in layout.base_subdirs])

    combined.mkdir(parents=True, exist_ok=True)
    for entry in layout.amd_files:
        _link(variant_dir / entry, combined / entry)
    for sub in layout.base_subdirs:
        _link(base_dir / sub, combined / sub)
    (combined / "model_index.json").write_text(json.dumps(layout.model_index, indent=2) + "\n")
    return combined


def main() -> int:
    root_env = (
        os.environ.get("ROOT") or f"{os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))}/micro_world"
    )
    root = Path(root_env)
    root.mkdir(parents=True, exist_ok=True)
    t2w = build(T2W, root)
    i2w = build(I2W, root)
    print(f"MICRO_WORLD_T2W_MODEL={t2w}")
    print(f"MICRO_WORLD_I2W_MODEL={i2w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
