# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Inspection and resolution helpers for ComfyUI MiniMax-H3 checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlparse

import regex as re
import torch
from safetensors import safe_open

from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)
from vllm_omni.quantization.int8_convrot_config import Int8ConvRotLayerConfig


@dataclass(frozen=True)
class MiniMaxH3ComfyCheckpoint:
    path: Path
    partition: Literal["fl2va", "ref2va"]
    layer_configs: dict[str, Int8ConvRotLayerConfig]
    adaln_curve_grid: int | None
    adaln_curve_dim: int | None

    @property
    def arch_overrides(self) -> dict[str, int]:
        if self.adaln_curve_grid is None or self.adaln_curve_dim is None:
            return {}
        return {
            "adaln_curve_grid": self.adaln_curve_grid,
            "adaln_curve_dim": self.adaln_curve_dim,
        }


def resolve_comfy_checkpoint_path(value: str, *, cache_dir: str | None = None) -> Path:
    """Resolve a local file/directory or a Hugging Face resolve URL."""
    local = Path(value).expanduser()
    if local.is_file():
        # Preserve a Hub snapshot symlink's user-facing filename.  Resolving it
        # to ``blobs/<sha>`` drops the ``.safetensors`` suffix, which makes the
        # generic diffusion loader misclassify the file as a PyTorch checkpoint.
        return local.absolute()
    if local.is_dir():
        candidates = sorted(local.glob("*.safetensors"))
        if len(candidates) != 1:
            raise ValueError(
                f"MiniMax-H3 transformer directory {local} must contain exactly one "
                f".safetensors file, found {len(candidates)}."
            )
        return candidates[0].absolute()

    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or parsed.netloc not in {
        "huggingface.co",
        "www.huggingface.co",
    }:
        raise FileNotFoundError(
            f"MiniMax-H3 ConvRot checkpoint {value!r} is not a local path or supported Hugging Face URL."
        )
    parts = [unquote(part) for part in parsed.path.strip("/").split("/")]
    try:
        marker_index = next(i for i, part in enumerate(parts) if part in {"resolve", "blob"})
    except StopIteration as exc:
        raise ValueError(
            f"Hugging Face checkpoint URL lacks /resolve/<revision>/ or /blob/<revision>/: {value}"
        ) from exc
    if marker_index != 2 or len(parts) <= marker_index + 2:
        raise ValueError(f"Unsupported Hugging Face checkpoint URL: {value}")
    repo_id = "/".join(parts[:marker_index])
    revision = parts[marker_index + 1]
    filename = "/".join(parts[marker_index + 2 :])
    snapshot = download_weights_from_hf_specific(
        model_name_or_path=repo_id,
        cache_dir=cache_dir,
        allow_patterns=[filename],
        revision=revision,
        require_all=True,
    )
    return Path(snapshot) / filename


def _decode_marker(marker: torch.Tensor, *, name: str) -> dict[str, object]:
    if marker.dtype != torch.uint8 or marker.dim() != 1:
        raise ValueError(
            f"{name} must be a one-dimensional uint8 JSON tensor, got {marker.dtype} {tuple(marker.shape)}."
        )
    try:
        decoded = json.loads(bytes(marker.tolist()).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} does not contain valid UTF-8 JSON.") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must decode to a JSON object.")
    return decoded


def _checkpoint_partition(path: Path) -> Literal["fl2va", "ref2va"]:
    """Read the partition contract carried by official Comfy filenames.

    The published MiniMax-H3 safetensors files do not contain safetensors
    ``__metadata__`` identifying their partition, and FL2VA/Ref2VA use the
    same tensor names and shapes.  The official filenames therefore provide
    the only inspectable contract.  Refuse ambiguous renamed files instead of
    silently loading one partition's weights into the other model.
    """
    matches = set(re.findall(r"(?:^|[._-])(fl2va|ref2va)(?=$|[._-])", path.name.lower()))
    if len(matches) != 1:
        raise ValueError(
            f"MiniMax-H3 ConvRot checkpoint filename {path.name!r} must identify exactly one "
            "partition with an 'fl2va' or 'ref2va' token."
        )
    return "fl2va" if "fl2va" in matches else "ref2va"


def inspect_comfy_checkpoint(
    path: str | Path,
    *,
    expected_partition: Literal["fl2va", "ref2va"] | None = None,
) -> MiniMaxH3ComfyCheckpoint:
    """Validate markers and derive mixed-precision/curve architecture metadata.

    Only tiny marker tensors and the optional AdaLN curve table are read.  INT8
    weights remain memory-mapped for the ordinary streaming loader.
    """
    checkpoint_path = Path(path)
    partition = _checkpoint_partition(checkpoint_path)
    if expected_partition is not None and partition != expected_partition:
        raise ValueError(
            f"{partition.upper()} checkpoint {checkpoint_path.name!r} cannot serve "
            f"the {expected_partition.upper()} partition."
        )
    layer_configs: dict[str, Int8ConvRotLayerConfig] = {}
    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        marker_names = sorted(name for name in keys if name.endswith(".comfy_quant"))
        if not marker_names:
            raise ValueError(f"{checkpoint_path} contains no Comfy .comfy_quant layer metadata.")

        for marker_name in marker_names:
            prefix = marker_name.removesuffix(".comfy_quant")
            weight_name = f"{prefix}.weight"
            scale_name = f"{prefix}.weight_scale"
            if weight_name not in keys or scale_name not in keys:
                raise ValueError(f"{marker_name} requires companion tensors {weight_name!r} and {scale_name!r}.")
            marker = _decode_marker(checkpoint.get_tensor(marker_name), name=marker_name)
            layer_config = Int8ConvRotLayerConfig.from_mapping(marker)
            weight_slice = checkpoint.get_slice(weight_name)
            scale_slice = checkpoint.get_slice(scale_name)
            weight_shape = tuple(weight_slice.get_shape())
            scale_shape = tuple(scale_slice.get_shape())
            if weight_slice.get_dtype() != "I8" or len(weight_shape) != 2:
                raise ValueError(
                    f"{weight_name} must be a two-dimensional I8 tensor, got {weight_slice.get_dtype()} {weight_shape}."
                )
            if scale_slice.get_dtype() != "F32" or scale_shape not in {(weight_shape[0],), (weight_shape[0], 1)}:
                raise ValueError(
                    f"{scale_name} must be F32 with one value per output row, got "
                    f"{scale_slice.get_dtype()} {scale_shape} for weight {weight_shape}."
                )
            if layer_config.convrot and weight_shape[1] % layer_config.convrot_groupsize:
                raise ValueError(
                    f"{weight_name} input width {weight_shape[1]} is not divisible by "
                    f"ConvRot group size {layer_config.convrot_groupsize}."
                )
            layer_configs[prefix] = layer_config

        marked_weights = {f"{prefix}.weight" for prefix in layer_configs}
        unmarked_int8_weights = []
        for name in sorted(keys):
            if not name.endswith(".weight") or name in marked_weights:
                continue
            tensor_slice = checkpoint.get_slice(name)
            if tensor_slice.get_dtype() == "I8" and len(tensor_slice.get_shape()) == 2:
                unmarked_int8_weights.append(name)
        if unmarked_int8_weights:
            raise ValueError(
                "Every two-dimensional INT8 weight in a ConvRot checkpoint must have matching "
                f".comfy_quant metadata; missing markers for {unmarked_int8_weights[:5]}."
            )

        if "adaln_t_table" in keys:
            table = checkpoint.get_tensor("adaln_t_table")
            if table.dtype != torch.float32 or table.dim() != 2 or table.shape[0] < 2:
                raise ValueError(
                    "adaln_t_table must be a two-dimensional FP32 table with at least two rows, "
                    f"got {table.dtype} {tuple(table.shape)}."
                )
            curve_grid, curve_dim = (int(table.shape[0]), int(table.shape[1]))
        else:
            curve_grid = curve_dim = None

    return MiniMaxH3ComfyCheckpoint(
        path=checkpoint_path,
        partition=partition,
        layer_configs=layer_configs,
        adaln_curve_grid=curve_grid,
        adaln_curve_dim=curve_dim,
    )


__all__ = [
    "MiniMaxH3ComfyCheckpoint",
    "inspect_comfy_checkpoint",
    "resolve_comfy_checkpoint_path",
]
