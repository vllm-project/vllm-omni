# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""H3 integration for native checkpoints using the shared MXFP4 methods."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


@dataclass(frozen=True)
class H3TransformerCheckpoint:
    path: Path
    config: dict[str, Any]
    release: dict[str, Any]


def resolve_h3_transformer_checkpoint(
    path: str | None,
    *,
    partition: str,
    modular: bool,
    runtime_lora: bool,
    base_config: dict[str, Any],
) -> H3TransformerCheckpoint | None:
    """Use a merged native transformer while retaining official components.

    The adjacent partition index belongs to the delivered transformer, so its
    distilled schedule cannot silently inherit the official 50-step defaults.
    """
    if path is None:
        return None
    if modular or partition not in {"fl2va", "ref2va"}:
        raise ValueError("transformer_checkpoint requires a single native H3 partition; set task_type explicitly")
    if runtime_lora:
        raise ValueError("transformer_checkpoint contains merged adapters; remove runtime LoRA options")
    if not isinstance(path, str) or not path:
        raise ValueError("transformer_checkpoint must be a local transformer directory")
    root = Path(path).expanduser().resolve()
    config_path, index_path = root / "config.json", root.parent / "model_index.json"
    if not config_path.is_file() or not index_path.is_file() or not any(root.glob("*.safetensors")):
        raise ValueError(
            "transformer_checkpoint requires config.json, safetensors weights and the adjacent parent model_index.json"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    release = index.get("_minimax_h3") if isinstance(index, dict) else None
    if not isinstance(config, dict) or not isinstance(release, dict):
        raise ValueError("transformer_checkpoint requires object config and _minimax_h3 release metadata")
    from .minimax_h3_blocks import MiniMaxH3DiTArchConfig

    if MiniMaxH3DiTArchConfig.from_mapping(config) != MiniMaxH3DiTArchConfig.from_mapping(base_config) or float(
        config.get("rope_theta", 10000.0)
    ) != float(base_config.get("rope_theta", 10000.0)):
        raise ValueError("transformer_checkpoint architecture must match the selected official H3 model")
    tasks = release.get("tasks")
    allowed_tasks = {"ref2va"} if partition == "ref2va" else {"t2va", "fl2va"}
    if (
        str(release.get("partition", "")).lower() != partition
        or not isinstance(tasks, list)
        or not tasks
        or not all(isinstance(task, str) and task.lower() in allowed_tasks for task in tasks)
    ):
        raise ValueError("transformer_checkpoint release partition/tasks do not match the selected H3 partition")
    from vllm_omni.diffusion.sched.sigma_schedule import DMD2SigmaSchedule

    if DMD2SigmaSchedule.from_metadata(release) is None:
        raise ValueError("transformer_checkpoint must declare its distilled base_schedule")
    shifts = release.get("sigma_shift_scales")
    if not isinstance(shifts, dict) or any(
        type(shifts.get(key)) not in (int, float) or not math.isfinite(shifts[key]) or shifts[key] <= 0
        for key in ("video", "audio")
    ):
        raise ValueError("transformer_checkpoint must declare finite positive video/audio sigma_shift_scales")
    return H3TransformerCheckpoint(root, config, release)


def is_serialized_mxfp4(config: "QuantizationConfig | None") -> bool:
    if config is None:
        return False
    return (config.get_name() == "mxfp4" and bool(getattr(config, "is_checkpoint_mxfp4_serialized", False))) or (
        config.get_name() == "mxfp4_dualscale" and bool(getattr(config, "is_checkpoint_serialized", False))
    )


def resolve_h3_transformer_quant_config(
    config: "QuantizationConfig | None",
    disk_config: dict[str, Any] | str | None,
    *,
    diffusers_weights: bool,
    runtime_lora: bool,
) -> "QuantizationConfig | None":
    """Resolve transformer storage metadata without changing other components.

    This initial serialized MX path consumes native H3 tensor names and the
    shared methods' numeric-FP4/scale layouts. It never dequantizes and
    requantizes an exported checkpoint or guesses its training recipe.
    """
    from vllm_omni.quantization.factory import resolve_quant_config_from_disk

    if config is not None and config.get_name() == "mxfp4_fake":
        if runtime_lora:
            raise ValueError("MiniMax-H3 fake quantization requires LoRA to be merged before loading")
        if isinstance(disk_config, str) and disk_config not in {"mxfp4", "mxfp4_fake"}:
            raise ValueError("MiniMax-H3 fake quantization requires a floating MXFP4 QAT checkpoint")
        if isinstance(disk_config, dict):
            if disk_config.get("is_checkpoint_mxfp4_serialized") or disk_config.get("is_checkpoint_serialized"):
                raise ValueError("MiniMax-H3 fake quantization requires floating weights, not a serialized checkpoint")
            method = disk_config.get("quant_method")
            if method not in (None, "mxfp4", "mxfp4_fake"):
                raise ValueError("MiniMax-H3 fake quantization requires a floating MXFP4 QAT checkpoint")
            ignored = disk_config.get("ignored_layers")
            if ignored is not None and ignored != config.ignored_layers:
                raise ValueError("mxfp4_fake ignored_layers must explicitly match the checkpoint policy")
        return config

    if config is not None and config.get_name() == "mxfp4" and getattr(config, "mxfp4_recipe", None) == "mxfp4_c7":
        if runtime_lora:
            raise ValueError("native C7 requires LoRA to be merged before loading")
        if isinstance(disk_config, dict):
            if disk_config.get("is_checkpoint_mxfp4_serialized") or disk_config.get("is_checkpoint_serialized"):
                raise ValueError("native C7 requires floating checkpoint weights")
            if disk_config.get("quant_method") not in (None, "mxfp4", "mxfp4_fake"):
                raise ValueError("native C7 requires a floating MXFP4 QAT checkpoint")
            ignored = disk_config.get("ignored_layers")
            if ignored is not None and ignored != config.ignored_layers:
                raise ValueError("native C7 ignored_layers must match the checkpoint policy")
        elif disk_config not in (None, "mxfp4", "mxfp4_fake"):
            raise ValueError("native C7 requires a floating MXFP4 QAT checkpoint")
        return config

    if isinstance(disk_config, dict) and (
        disk_config.get("quant_scheme") == "mxfp4_c7" or disk_config.get("use_torch_fake_quant")
    ):
        if config is None:
            # A fake-quant training config does not serialize low-bit weights.
            # Keep a merged BF16 baseline dense unless quantization is requested.
            return None
        raise ValueError(
            "MiniMax-H3 training fake-quant metadata is not a runtime MXFP4 contract; "
            "provide the matched quantization recipe/config instead of assuming mxfp4_c7 equals OCP or UOS"
        )
    disk_method = disk_config.get("quant_method") if isinstance(disk_config, dict) else disk_config
    resolved = (
        resolve_quant_config_from_disk(config, disk_config) if disk_method in {"mxfp4", "mxfp4_dualscale"} else config
    )
    if is_serialized_mxfp4(resolved):
        if diffusers_weights:
            raise ValueError(
                "MiniMax-H3 serialized MXFP4 requires native H3 tensor names and grouped-by-head QKV; "
                "export a native checkpoint before loading Diffusers QAT weights"
            )
        if runtime_lora:
            raise ValueError(
                "MiniMax-H3 serialized MXFP4 requires LoRA to be merged before quantization export; "
                "remove runtime LoRA options"
            )
    return resolved
