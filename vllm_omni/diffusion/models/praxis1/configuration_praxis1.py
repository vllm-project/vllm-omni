# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validate the supported DROID checkpoint's mathematical and robot-I/O contract.

Dimensions used to construct the native modules stay in the checkpoint dict.
Reference kernel/backend tuning is metadata; Omni selects its own execution.
"""

import json
import math
from collections.abc import Mapping
from pathlib import Path

EXPERT_TRUNK_ROPE = {
    "kind": "trunk",
    "dim": 64,
    "effective_dim": 64,
    "base": 26000000,
    "table_size": 1000000,
    "order": "before_norm",
}

# These values are fixed in the implemented math or observation processor.
# Shape-dependent dimensions are consumed directly by the native model instead.
_REQUIRED_VALUES = {
    "format_version": 2,
    "model_type": "praxis",
    "embodiment": "single_arm",
    "pose_frame": "flange",
    "video.width": 832,
    "video.height": 224,
    "video.tile_height": 224,
    "video.views": ["exterior", "wrist"],
    "action_space.dim": 10,
    "action_expert.action_patch_dim": 40,
    "action_expert.proprio_patch_dim": 10,
    "backbone.layout": "chunk_causal",
    "backbone.norm_eps": 1e-6,
    "backbone.norm_compute_dtype": "float32",
    "backbone.attention.gated": True,
    "backbone.attention.qk_norm_eps": 1e-6,
    "backbone.attention.qk_norm_compute_dtype": "float32",
    "backbone.attention.rope_after_norm": False,
    "backbone.rope.ref_placement": "group",
    "backbone.rope.n_chunks": None,
    "backbone.moe.shared_expert": True,
    "backbone.moe.router_activation": "sigmoid",
    "backbone.moe.router_dtype": "float32",
    "backbone.moe.routed_weight_layout": "expert_first",
    "action_expert.absolute_positions": True,
    "action_expert.action_pos_table": False,
    "action_expert.future_chunk_offset": "zero",
    "action_expert.rope": EXPERT_TRUNK_ROPE,
    "action_expert.global_order": ["register", "anchor", "text"],
    "action_expert.head_dim": 64,
    "chunking.n_chunks": 4,
    "chunking.action_chunk": 24,
    "chunking.action_frames_per_token": 4,
    "chunking.latents_per_chunk": 6,
    "chunking.grid_hz": 15.0,
    "video.antialias": True,
    "video.canvas": "tiles",
    "video.keep_aspect": True,
    "video.tile_width": 416,
    "video.latent_channels": 48,
    "video.vae_stride": [4, 16, 16],
    "video.patch_size": [1, 2, 2],
    "text_encoder.hidden_layer": -3,
    "text_encoder.pad_to_multiple_of": 32,
    "text_encoder.trim_padding": True,
    "text_encoder.hidden_dim": 2048,
    "text_encoder.uncond": "blank_caption",
}


class CheckpointError(ValueError):
    """The checkpoint requests math or robot conventions this model cannot run."""


def validate_config(config: Mapping) -> None:
    """Reject incompatible mathematical conventions before allocating weights."""
    embodiment = config.get("embodiment")
    if embodiment != "single_arm":
        raise CheckpointError(f"Unsupported Praxis embodiment: {embodiment!r}")
    dim, height, views = 10, 224, ["exterior", "wrist"]
    rig_values = {
        "embodiment": embodiment,
        "pose_frame": "flange",
        "video.width": 416 * len(views),
        "video.height": height,
        "video.tile_height": height,
        "video.views": views,
        "action_space.dim": dim,
        "action_expert.action_patch_dim": 4 * dim,
        "action_expert.proprio_patch_dim": dim,
    }
    for path, expected in {**_REQUIRED_VALUES, **rig_values}.items():
        actual = config
        for key in path.split("."):
            if not isinstance(actual, Mapping) or key not in actual:
                raise CheckpointError(f"Praxis checkpoint is missing {path}")
            actual = actual[key]
        if actual != expected or (type(expected) in (int, bool) and type(actual) is not type(expected)):
            raise CheckpointError(f"Unsupported Praxis {path}: {actual!r}; expected {expected!r}")
    if not isinstance(config["text_encoder"].get("blank_caption"), str):
        raise CheckpointError("Praxis text_encoder.blank_caption must be a string")
    tol = config["video"].get("aspect_tol", 0.03)
    if isinstance(tol, bool) or not isinstance(tol, int | float) or not 0 <= tol < 1:
        raise CheckpointError("Praxis video.aspect_tol must be in [0, 1)")
    ex = config["action_expert"]
    if ex.get("rope_cross") or ex.get("rope_self"):
        raise CheckpointError("Praxis action_expert.rope excludes rope_cross and rope_self")
    if "head_plus_zero" in ex and not isinstance(ex["head_plus_zero"], bool):
        raise CheckpointError("Praxis action_expert.head_plus_zero must be Boolean")
    if ex.get("layout_stamp", 20260924) != 20260924:
        raise CheckpointError("Unsupported Praxis action_expert.layout_stamp")
    norm = config.get("normalization")
    if not isinstance(norm, Mapping):
        raise CheckpointError("Praxis normalization must be an object")
    for key in ("action_q01", "action_q99", "proprio_mean", "proprio_std"):
        values = norm.get(key)
        if (
            not isinstance(values, list)
            or len(values) != dim
            or not all(isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v) for v in values)
        ):
            raise CheckpointError(f"Praxis normalization.{key} must contain {dim} finite numbers")
    dims = norm.get("q99_normed_dims", [i + offset for offset in range(0, dim, 10) for i in (0, 1, 2, 9)])
    if not isinstance(dims, list) or not all(type(d) is int and 0 <= d < dim for d in dims):
        raise CheckpointError(f"Praxis normalization.q99_normed_dims must name dimensions in [0, {dim})")
    if any(norm["action_q99"][d] <= norm["action_q01"][d] for d in dims):
        raise CheckpointError("Praxis action_q99 must exceed action_q01 on normalized dimensions")
    if any(v <= 0 for v in norm["proprio_std"]):
        raise CheckpointError("Praxis proprio_std must be positive")


def load_config(path: str | Path) -> dict:
    """Read an unmodified native checkpoint; ignore reference kernel metadata."""
    config = json.loads((Path(path) / "config.json").read_text())
    if not isinstance(config, Mapping):
        raise CheckpointError("Praxis config.json must contain an object")
    validate_config(config)
    return config
