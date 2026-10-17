# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Checkpoint configuration and output geometry for Kandinsky 6 VSR."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

TARGET_RESOLUTIONS = {"hd": (720, 1280), "fhd": (1080, 1920), "fullhd": (1080, 1920)}


def transformer_config(config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "in_visual_dim",
        "out_visual_dim",
        "time_dim",
        "patch_size",
        "model_dim",
        "ff_dim",
        "num_visual_blocks",
        "axes_dims",
        "scale_factor",
        "nabla_threshold",
        "nabla_window",
        "tile_sizes",
    )
    result = {key: config[key] for key in keys if key in config}
    if config.get("use_text", False) or config.get("instruct_type", "hybrid_anchor") != "hybrid_anchor":
        raise ValueError("Kandinsky VSR requires a text-free hybrid-anchor transformer")
    if config.get("attribute_overrides"):
        raise ValueError("Kandinsky VSR does not support checkpoint attribute_overrides")
    sr = config.get("sr_params", {})
    if "scale_factor" not in result and "scale_factor" in sr:
        result["scale_factor"] = sr["scale_factor"][str(sr["visual_size"][0])]
    attention = config.get("attention_params", {}).get("512", {})
    if attention:
        if attention.get("type") != "nabla":
            raise ValueError("Kandinsky VSR requires the checkpoint's NABLA sparse attention")
        result["nabla_threshold"] = attention["P"]
        result["nabla_window"] = (attention["wT"], attention["wH"], attention["wW"])
    return result


def vae_config(config: dict[str, Any]) -> dict[str, Any]:
    if "ch" not in config:
        return {key: value for key, value in config.items() if not key.startswith("_")}
    if config.get("norm_type") != "rms_norm" or not config.get("fix_pxs") or config.get("downsample_version") != 2:
        raise ValueError("Unsupported Kandinsky KVAE architecture")
    return {
        "in_channels": config["in_channels"],
        "out_channels": config["out_channels"],
        "latent_channels": config["z_channels"],
        "encoder_block_out_channels": tuple(int(config["ch"] * value) for value in config["ch_mult"]),
        "decoder_block_out_channels": tuple(int(config["decoder_ch"] * value) for value in config["decoder_ch_mult"]),
        "layers_per_block": config["num_res_blocks"],
        "temporal_compression_ratio": config["temporal_compress_times"],
        "temporal_compression_start_level": config["temporal_compress_start_level"],
        "scaling_factor": config["scaling_factor"],
    }


def upscaler_config(config: dict[str, Any], weight_names: list[str] | None = None) -> dict[str, Any]:
    if "models" not in config:
        return {key: value for key, value in config.items() if not key.startswith("_")}
    models = config["models"]
    first = models[0]["model"]
    for entry in models:
        model = entry["model"]
        if model["architecture"] != "multi_scale" or model.get("upsample_mode") != "pxs_v2":
            raise ValueError("Unsupported Kandinsky latent-upscaler architecture")
        for key in ("stage_channels", "in_channels", "num_pre_blocks", "num_mid_blocks", "num_post_blocks"):
            if model[key] != first[key]:
                raise ValueError("Kandinsky latent-upscaler bank requires matching branch widths and depths")
    scales = tuple(int(entry["target_scale"].removesuffix("x")) for entry in models)
    if weight_names is not None:
        # Released legacy bundles list branches as (4,2), but store tensors as
        # (2,4). Identify the x2 entry from its dedicated branch parameters.
        indices = sorted({int(name.split(".")[1]) for name in weight_names if name.startswith("_models.")})
        scales = tuple(
            2 if any(name.startswith(f"_models.{i}.x2_branch.") for name in weight_names) else 4 for i in indices
        )
        if sorted(scales) != sorted(int(entry["target_scale"].removesuffix("x")) for entry in models):
            raise ValueError("Latent-upscaler checkpoint branches do not match the configured scales")
    return {
        "in_channels": first["in_channels"],
        "stage_channels": first["stage_channels"],
        "num_pre_blocks": first["num_pre_blocks"],
        "num_mid_blocks": first["num_mid_blocks"],
        "num_post_blocks": first["num_post_blocks"],
        "num_x2_adapter_blocks": next(
            (entry["model"].get("x2_adapter_blocks", 2) for entry in models if entry["target_scale"] == "2x"), 2
        ),
        "scales": scales,
        "scaling_factor": config["scaling_factor"],
    }


def default_steps(config: dict[str, Any]) -> int:
    if config.get("_class_name") == "PiflowScheduler":
        return int(config.get("nfe", 2))
    if config.get("_class_name") == "FlowMatchEulerDiscreteScheduler":
        return 4
    raise ValueError(f"Unsupported Kandinsky VSR scheduler: {config.get('_class_name')}")


@dataclass(frozen=True)
class OutputPlan:
    height: int
    width: int
    content_height: int
    content_width: int
    prepared_height: int
    prepared_width: int
    tiling_scale: int


def output_plan(
    source_height: int,
    source_width: int,
    height: int | None,
    width: int | None,
    extra: dict[str, Any],
) -> OutputPlan:
    scale = float(extra.get("resolution_scale", 2.25))
    if scale not in (2.0, 2.25, 4.0):
        raise ValueError("resolution_scale must be 2, 2.25 or 4")
    target = extra.get("target_resolution")
    if target is not None:
        if target not in TARGET_RESOLUTIONS:
            raise ValueError("target_resolution must be hd, fhd or fullhd")
        target_height, target_width = TARGET_RESOLUTIONS[target]
        if height not in (None, target_height) or width not in (None, target_width):
            raise ValueError("Explicit dimensions conflict with target_resolution")
        height, width = target_height, target_width
    if (height is None) != (width is None):
        raise ValueError("Supply both height and width for Kandinsky VSR")
    if height is None:
        height, width = round(source_height * scale), round(source_width * scale)
    assert width is not None
    if min(height, width) < 2 or height % 2 or width % 2:
        raise ValueError("Kandinsky VSR output dimensions must be positive even numbers")
    tiling_scale = 4 if scale == 4 else 2
    content_height, content_width = math.ceil(height / tiling_scale), math.ceil(width / tiling_scale)
    base_height, base_width = min(
        ((512, 512), (512, 768), (768, 512)), key=lambda size: abs(size[1] / size[0] - width / height)
    )
    return OutputPlan(
        height,
        width,
        content_height,
        content_width,
        max(math.ceil(content_height / 16) * 16, base_height // tiling_scale),
        max(math.ceil(content_width / 16) * 16, base_width // tiling_scale),
        tiling_scale,
    )
