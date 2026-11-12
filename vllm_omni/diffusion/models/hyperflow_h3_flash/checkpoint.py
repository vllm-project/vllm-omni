# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""HyperFlow H3 Flash hub layout and base MiniMax-H3 component resolution."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm_omni.diffusion.models.hyperflow_h3_flash.schedule import (
    AUDIO_SHIFT,
    DEFAULT_STEPS,
    SIGMA_GRIDS,
    VIDEO_SHIFT,
    hyperflow_schedule,
)
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

HYPERFLOW_H3_FLASH_MODEL_ID = "VideoRebirthAI/hyperflow-h3-flash"
HYPERFLOW_BASE_MODEL = "MiniMaxAI/MiniMax-H3"

PARTITIONS: dict[str, str] = {
    "t2va": "fl2va_int8_convrot",
    "fl2va": "fl2va_int8_convrot",
    "ref2va": "ref2va_int8_convrot",
}

# RFC / upstream model-card defaults when config.json is unavailable offline.
DEFAULT_DROPPED_BLOCKS: dict[str, tuple[int, ...]] = {
    "fl2va": (6, 21, 22),
    "ref2va": (9,),
}


def _normalize_model_id(model: str) -> str:
    normalized = str(model).strip().rstrip("/")
    hub_prefix = "https://huggingface.co/"
    if normalized.lower().startswith(hub_prefix):
        normalized = normalized[len(hub_prefix) :]
        normalized = normalized.split("/tree/", 1)[0]
    return normalized


def looks_like_hyperflow_h3_flash(model: str | None) -> bool:
    """Detect the Hub id or a local snapshot with HyperFlow INT8 partitions."""
    if not model:
        return False
    normalized = _normalize_model_id(model)
    if normalized.lower() == HYPERFLOW_H3_FLASH_MODEL_ID.lower():
        return True
    path = Path(normalized)
    if not path.is_dir():
        return False
    return any((path / subfolder / "config.json").is_file() for subfolder in PARTITIONS.values()) or (
        path / "config.json"
    ).is_file() and _config_looks_hyperflow(path / "config.json")


def _config_looks_hyperflow(config_path: Path) -> bool:
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(cfg, Mapping) and (
        "dropped_blocks" in cfg or "lowrank_ranks" in cfg or cfg.get("quantization", {}).get("method") == "int8_convrot"
    )


def resolve_hyperflow_partition(task_type: str | None) -> str:
    """Map ``--task-type`` to fl2va / ref2va (t2va aliases fl2va)."""
    task = str(task_type or "fl2va").lower()
    if task in {"auto", "combined", "t2va", "fl2va"}:
        return "fl2va"
    if task == "ref2va":
        return "ref2va"
    raise ValueError(
        "HyperFlow H3 Flash --task-type must be one of auto/combined/t2va/fl2va/ref2va, "
        f"got {task_type!r}"
    )


def resolve_hyperflow_subfolder(partition: str) -> str:
    return PARTITIONS["ref2va" if partition == "ref2va" else "fl2va"]


@dataclass(frozen=True)
class HyperFlowH3FlashLayout:
    """Resolved on-disk roots for the HyperFlow DiT and base MiniMax TE/VAE."""

    weights_root: Path
    transformer_subfolder: str
    base_partition_root: Path
    partition: str
    transformer_config: dict[str, Any]

    @property
    def transformer_root(self) -> Path:
        return self.weights_root / self.transformer_subfolder

    def release_metadata(self, *, steps: int = DEFAULT_STEPS) -> dict[str, Any]:
        schedule = hyperflow_schedule(steps)
        tasks = ["ref2va"] if self.partition == "ref2va" else ["t2va", "fl2va"]
        return {
            "partition": self.partition,
            "tasks": tasks,
            "sigma_shift_scales": {"video": VIDEO_SHIFT, "audio": AUDIO_SHIFT},
            "base_schedule": list(schedule.base_schedule),
            "hyperflow": {
                "default_steps": DEFAULT_STEPS,
                "supported_steps": sorted(SIGMA_GRIDS),
                "cfg": False,
            },
        }

    def dropped_blocks(self) -> tuple[int, ...]:
        raw = self.transformer_config.get("dropped_blocks")
        if raw is None:
            return DEFAULT_DROPPED_BLOCKS[self.partition]
        return tuple(int(x) for x in raw)

    def hyperflow_gate(self) -> float:
        return float(self.transformer_config.get("hyperflow_gate", 0.25))

    def lowrank_ranks(self) -> dict[str, int]:
        raw = self.transformer_config.get("lowrank_ranks") or {}
        return {str(k): int(v) for k, v in raw.items()}

    def quantization(self) -> dict[str, Any] | None:
        quant = self.transformer_config.get("quantization")
        return dict(quant) if isinstance(quant, Mapping) else None


def _download_hyperflow_transformer(model: str, revision: str | None, subfolder: str) -> Path:
    path = Path(model)
    if path.is_dir():
        return path
    return Path(
        download_weights_from_hf_specific(
            model_name_or_path=model,
            cache_dir=None,
            allow_patterns=[f"{subfolder}/**"],
            revision=revision,
            require_all=True,
        )
    )


def _download_base_partition(
    *,
    base_model: str,
    revision: str | None,
    partition: str,
    load_text_encoder: bool,
) -> Path:
    native = "Ref2VA" if partition == "ref2va" else "FL2VA"
    patterns = [
        f"{native}/model_index.json",
        f"{native}/video_vae/**",
        f"{native}/audio_vae/**",
    ]
    if load_text_encoder:
        patterns += [
            f"{native}/text_encoder/**",
            f"{native}/tokenizer/**",
            f"{native}/processor/**",
        ]
    path = Path(base_model)
    if path.is_dir():
        return path / native
    return (
        Path(
            download_weights_from_hf_specific(
                model_name_or_path=base_model,
                cache_dir=None,
                allow_patterns=patterns,
                revision=revision,
                require_all=True,
            )
        )
        / native
    )


def resolve_hyperflow_layout(
    model: str,
    *,
    task_type: str | None,
    revision: str | None = None,
    base_model: str | None = None,
    base_revision: str | None = None,
    load_text_encoder: bool = True,
    subfolder: str | None = None,
) -> HyperFlowH3FlashLayout:
    """Download HyperFlow INT8 DiT + MiniMax-H3 TE/VAE without the 31B base transformer."""
    partition = resolve_hyperflow_partition(task_type)
    transformer_subfolder = subfolder or resolve_hyperflow_subfolder(partition)
    weights_root = _download_hyperflow_transformer(model, revision, transformer_subfolder)
    config_path = weights_root / transformer_subfolder / "config.json"
    if not config_path.is_file():
        # Local directory may already be the partition root.
        if (weights_root / "config.json").is_file():
            transformer_subfolder = ""
            config_path = weights_root / "config.json"
        else:
            raise FileNotFoundError(f"HyperFlow transformer config not found under {weights_root}")
    transformer_config = json.loads(config_path.read_text(encoding="utf-8"))
    base = base_model or os.environ.get("HYPERFLOW_BASE_MODEL") or HYPERFLOW_BASE_MODEL
    base_partition_root = _download_base_partition(
        base_model=base,
        revision=base_revision,
        partition=partition,
        load_text_encoder=load_text_encoder,
    )
    return HyperFlowH3FlashLayout(
        weights_root=weights_root,
        transformer_subfolder=transformer_subfolder,
        base_partition_root=base_partition_root,
        partition=partition,
        transformer_config=transformer_config,
    )


def write_omni_model_index(layout: HyperFlowH3FlashLayout, dest: Path, *, steps: int = DEFAULT_STEPS) -> Path:
    """Write a MiniMax-H3-compatible ``model_index.json`` for Omni path reuse."""
    dest.mkdir(parents=True, exist_ok=True)
    release = layout.release_metadata(steps=steps)
    index = {
        "_class_name": "MiniMaxH3Pipeline",
        "_diffusers_version": "0.40.0",
        "_minimax_h3": release,
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "transformer": ["diffusers", "MiniMaxH3Transformer3DModel"],
        "video_vae": ["diffusers", "AutoencoderKL"],
        "audio_vae": ["diffusers", "AutoencoderKL"],
    }
    path = dest / "model_index.json"
    path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = [
    "DEFAULT_DROPPED_BLOCKS",
    "HYPERFLOW_BASE_MODEL",
    "HYPERFLOW_H3_FLASH_MODEL_ID",
    "HyperFlowH3FlashLayout",
    "PARTITIONS",
    "looks_like_hyperflow_h3_flash",
    "resolve_hyperflow_layout",
    "resolve_hyperflow_partition",
    "resolve_hyperflow_subfolder",
    "write_omni_model_index",
]
