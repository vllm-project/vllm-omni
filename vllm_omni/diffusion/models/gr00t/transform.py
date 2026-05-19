# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Robot-observation transform for GR00T-N1.7 — server-side pre/post-process.

OpenPI clients send raw observations:

    {
        "embodiment": "<embodiment_tag>",
        "modality_config": {
            "state":  {"<key>": {"start": int, "end": int}, ...},
            "action": {"<key>": {"start": int, "end": int}, ...},
        },
        "state":  {"<key>": list[float], ...},   # RAW, un-normalized
        "prompt": "<language instruction>",
        "images": {"<camera_name>": ndarray, ...},
    }

The server runs :func:`encode` to turn that into model-ready tensors,
:func:`Gr00tN1d7Pipeline.forward` dispatches the GR00T model, and
:func:`decode` turns the normalized RELATIVE 132-dim output into an
absolute per-action-key dict for the client to consume.

Clients see **only** raw obs in and **absolute** actions out — no
checkpoint stats, no Qwen3VLProcessor, no SE(3) math on the client side.

Reference: ``Isaac-GR00T/gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py``
(``EMBODIMENT_TAG_TO_PROJECTOR_INDEX`` lines 56–72).  Modality-config slice
semantics mirror ``demo_data/*/meta/modality.json``
(``{"<key>": {"start": int, "end": int}}``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


# Verbatim from Isaac-GR00T/gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py
# (lines 56-72).  Used to compute the per-embodiment integer ID the action
# head's category-specific MLPs select on.
EMBODIMENT_TAG_TO_PROJECTOR_INDEX: dict[str, int] = {
    # Pretrain embodiment ids
    "oxe_droid_relative_eef_relative_joint": 24,
    "xdof_relative_eef_relative_joint": 27,
    "xdof_relative_eef_relative_joint_subtask": 27,
    "real_g1_relative_eef_relative_joints": 25,
    "real_r1_pro_sharpa_relative_eef": 26,
    "real_r1_pro_sharpa_relative_eef_human": 26,
    "real_r1_pro_sharpa_relative_eef_maxinsights": 26,
    "real_r1_pro_sharpa_relative_eef_mecka": 26,
    # Posttrain embodiment ids
    "unitree_g1_full_body_with_waist_height_nav_cmd": 25,
    "simpler_env_google": 0,
    "simpler_env_widowx": 1,
    "libero_sim": 2,
    "new_embodiment": 10,
}


# Action keys whose model output represents an ABSOLUTE quantity (not
# composed against the current state).  The convention mirrors DreamZero's
# `relative_action_keys` setup, just inverted.  Used by :func:`decode`.
_ABSOLUTE_ACTION_KEYS: frozenset[str] = frozenset({"gripper_position"})

# Action keys that use the SE(3) xyz+rot6d format (relative pose deltas
# that need matrix composition, not element-wise add).
_EEF_XYZ_ROT6D_KEYS: frozenset[str] = frozenset({"eef_9d"})


# ---------------------------------------------------------------------------
# Embodiment + modality helpers
# ---------------------------------------------------------------------------


def embodiment_id_for_tag(embodiment_tag: str) -> int:
    """Lookup the projector index for an embodiment tag."""
    if embodiment_tag not in EMBODIMENT_TAG_TO_PROJECTOR_INDEX:
        raise KeyError(
            f"Unknown embodiment tag {embodiment_tag!r}; supported tags: "
            f"{sorted(EMBODIMENT_TAG_TO_PROJECTOR_INDEX)}"
        )
    return EMBODIMENT_TAG_TO_PROJECTOR_INDEX[embodiment_tag]


def pack_proprio_state(
    state_dict: Mapping[str, list[float] | np.ndarray],
    state_modality: Mapping[str, Mapping[str, int]],
    max_state_dim: int,
) -> np.ndarray:
    """Flatten a per-joint proprio_state dict into a single float32 vector.

    Each modality key writes into ``packed[start:end]`` per ``state_modality``;
    everything else is zero-padded.
    """
    packed = np.zeros(max_state_dim, dtype=np.float32)
    for key, segment in state_modality.items():
        if key not in state_dict:
            continue
        start = int(segment["start"])
        end = int(segment["end"])
        if end > max_state_dim:
            raise ValueError(
                f"State key {key!r} segment [{start}:{end}] exceeds "
                f"max_state_dim={max_state_dim}."
            )
        values = np.asarray(state_dict[key], dtype=np.float32).flatten()
        expected = end - start
        if values.size != expected:
            raise ValueError(
                f"State key {key!r} expects {expected} dims but got "
                f"{values.size}."
            )
        packed[start:end] = values
    return packed


def unpack_actions(
    actions: np.ndarray,
    action_modality: Mapping[str, Mapping[str, int]],
) -> dict[str, np.ndarray]:
    """Split a packed action trajectory ``[horizon, max_action_dim]`` into a
    per-key dict ``{key: ndarray[horizon, key_dim]}``."""
    out: dict[str, np.ndarray] = {}
    for key, segment in action_modality.items():
        start = int(segment["start"])
        end = int(segment["end"])
        out[key] = actions[..., start:end].astype(np.float32, copy=False)
    return out


# ---------------------------------------------------------------------------
# q99 normalize / denormalize
# ---------------------------------------------------------------------------


def _q99_norm(real: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Forward q99-mode normalize, clipped to ``[-1, 1]``."""
    rng = np.maximum(q99 - q01, 1e-12)
    return np.clip(2.0 * (real - q01) / rng - 1.0, -1.0, 1.0)


def _q99_denorm(normalized: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_q99_norm`."""
    return (normalized + 1.0) / 2.0 * (q99 - q01) + q01


# ---------------------------------------------------------------------------
# SE(3) composition for xyz+rot6d EEF actions
# ---------------------------------------------------------------------------
#
# 6D rotation rep (Zhou et al. 2019) stores the first two columns of a 3x3
# rotation matrix; the third column is recovered as ``cross(col0, col1)``
# after Gram-Schmidt orthonormalization.  EEF actions in GR00T are stored
# as ``[x, y, z, rot6d_0, ..., rot6d_5]``; relative-to-absolute requires
# proper SE(3) composition ``T_abs = T_ref @ T_rel`` rather than
# element-wise addition.


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert ``(..., 6)`` rot6d to ``(..., 3, 3)`` rotation matrix via
    Gram-Schmidt orthonormalization of the first two columns."""
    a = rot6d[..., 0:3]
    b = rot6d[..., 3:6]
    a_norm = a / np.maximum(
        np.linalg.norm(a, axis=-1, keepdims=True), 1e-12
    )
    b_orth = b - (a_norm * b).sum(axis=-1, keepdims=True) * a_norm
    b_norm = b_orth / np.maximum(
        np.linalg.norm(b_orth, axis=-1, keepdims=True), 1e-12
    )
    c = np.cross(a_norm, b_norm)
    return np.stack([a_norm, b_norm, c], axis=-1)  # columns


def _matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_rot6d_to_matrix` — take the first two columns."""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def _xyz_rot6d_to_se3(arr9: np.ndarray) -> np.ndarray:
    """Convert ``(..., 9)`` xyz+rot6d to ``(..., 4, 4)`` SE(3) matrix."""
    pos = arr9[..., 0:3]
    R = _rot6d_to_matrix(arr9[..., 3:9])
    T = np.zeros((*arr9.shape[:-1], 4, 4), dtype=arr9.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = pos
    T[..., 3, 3] = 1.0
    return T


def _se3_to_xyz_rot6d(T: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_xyz_rot6d_to_se3`."""
    pos = T[..., :3, 3]
    rot6d = _matrix_to_rot6d(T[..., :3, :3])
    return np.concatenate([pos, rot6d], axis=-1)


def _compose_eef_relative_to_absolute(
    rel_chunk: np.ndarray, ref_state: np.ndarray
) -> np.ndarray:
    """Compose a 9-D xyz+rot6d delta chunk against a 9-D reference state.

    ``T_absolute = T_ref @ T_relative`` for each step in the chunk.

    Args:
        rel_chunk: ``[horizon, 9]`` relative xyz+rot6d.
        ref_state: ``[9]`` reference xyz+rot6d.

    Returns:
        ``[horizon, 9]`` absolute xyz+rot6d.
    """
    T_ref = _xyz_rot6d_to_se3(ref_state.astype(np.float64))  # (4, 4)
    T_rel = _xyz_rot6d_to_se3(rel_chunk.astype(np.float64))  # (H, 4, 4)
    T_abs = T_ref @ T_rel  # broadcasted matmul
    return _se3_to_xyz_rot6d(T_abs).astype(np.float32)


# ---------------------------------------------------------------------------
# Public encode / decode
# ---------------------------------------------------------------------------


def encode(
    robot_obs: Mapping[str, Any],
    *,
    gr00t_config: Any,
    processor: Any,
    state_norm_stats: Mapping[str, Mapping[str, Any]],
    embodiment_name_to_id: Mapping[str, int],
) -> dict[str, Any]:
    """Turn a raw OpenPI observation into model-ready CPU tensors.

    The returned dict is the boundary between the server's transform layer
    and the GR00T model code (``self.backbone`` + ``self.action_head``):

      ``input_ids``       : ``[1, S]``  long tensor — text + image markers
      ``attention_mask``  : ``[1, S]``  long tensor
      ``pixel_values``    : ``[1, ...]`` float tensor (optional)
      ``image_grid_thw``  : ``[1, 3]``  long tensor (optional)
      ``state``           : ``[1, state_history_length, max_state_dim]``
                            float tensor — q99-normalized
      ``embodiment_id``   : ``[1]``  long tensor — projector index
      ``embodiment_tag``  : str — passed through for :func:`decode`
      ``modality``        : dict — passed through for :func:`decode`
      ``raw_state``       : dict[str, ndarray] — passed through so
                            :func:`decode` can rebuild absolute poses

    Tensors are CPU + default dtype; the pipeline moves them to device.
    """
    embodiment_tag = robot_obs.get("embodiment")
    if embodiment_tag is None:
        raise KeyError("robot_obs is missing the 'embodiment' tag")
    if embodiment_tag in embodiment_name_to_id:
        embodiment_idx = int(embodiment_name_to_id[embodiment_tag])
    else:
        embodiment_idx = embodiment_id_for_tag(embodiment_tag)

    raw_state_dict = robot_obs.get("state", {})
    if not isinstance(raw_state_dict, Mapping):
        raise TypeError(
            f"robot_obs['state'] must be a dict; got {type(raw_state_dict).__name__}"
        )
    modality = robot_obs.get("modality_config") or {}
    state_modality = modality.get("state", {})

    # Stash raw arrays for decode's relative→absolute step.
    raw_state_np: dict[str, np.ndarray] = {
        k: np.asarray(v, dtype=np.float32).flatten()
        for k, v in raw_state_dict.items()
    }

    # q99-normalize each state key into the packed [max_state_dim] vector.
    norm_state_dict: dict[str, np.ndarray] = {}
    for key, raw in raw_state_np.items():
        s = state_norm_stats.get(key)
        if s is None:
            norm_state_dict[key] = raw
            continue
        q01 = np.asarray(s["q01"], dtype=np.float32)
        q99 = np.asarray(s["q99"], dtype=np.float32)
        norm_state_dict[key] = _q99_norm(raw, q01, q99)
    packed = pack_proprio_state(
        norm_state_dict, state_modality, gr00t_config.max_state_dim
    )
    state_tensor = torch.from_numpy(packed).view(1, 1, -1).to(dtype=torch.float32)

    # Text + image tokenization via Qwen3VLProcessor.
    prompt = robot_obs.get("prompt", "") or ""
    images = robot_obs.get("images") or {}
    images_list = (
        list(images.values()) if isinstance(images, Mapping) else list(images)
    )
    content: list[dict[str, Any]] = [{"type": "image"} for _ in images_list]
    content.append({"type": "text", "text": prompt})
    chat = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(chat, add_generation_prompt=True)
    encoded = processor(
        text=[text],
        images=images_list if images_list else None,
        return_tensors="pt",
        padding=True,
    )

    out: dict[str, Any] = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "state": state_tensor,
        "embodiment_id": torch.tensor([embodiment_idx], dtype=torch.long),
        "embodiment_tag": embodiment_tag,
        "modality": modality,
        "raw_state": raw_state_np,
    }
    if "pixel_values" in encoded:
        out["pixel_values"] = encoded["pixel_values"]
    if "image_grid_thw" in encoded:
        out["image_grid_thw"] = encoded["image_grid_thw"]
    return out


def decode(
    actions_packed: np.ndarray,
    *,
    raw_state: Mapping[str, np.ndarray],
    modality: Mapping[str, Any],
    action_norm_stats: Mapping[str, Mapping[str, Any]],
    relative_action_norm_stats: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, np.ndarray]:
    """Turn the model's ``[horizon, max_action_dim]`` normalized RELATIVE
    output into an absolute per-action-key dict for OpenPI clients.

    For each action key listed in ``modality["action"]``:

      1. Slice ``actions_packed[..., start:end]``.
      2. Denormalize with q01/q99 — per-horizon-step from
         ``relative_action_norm_stats`` for relative keys, static from
         ``action_norm_stats`` for absolute keys (and as a fallback).
      3. For relative keys, reconstruct the absolute action:
         - ``eef_9d`` (xyz+rot6d): SE(3) composition with ``raw_state[key]``
         - other relative keys: element-wise add ``raw_state[key]``

    Absolute keys (``gripper_position``) skip step 3.
    """
    action_modality = modality.get("action") if modality else None
    if not action_modality:
        # No modality config supplied — return packed as a single entry.
        return {"actions": actions_packed.astype(np.float32, copy=False)}

    horizon = actions_packed.shape[0]
    out: dict[str, np.ndarray] = {}
    for key, segment in action_modality.items():
        start = int(segment["start"])
        end = int(segment["end"])
        arr = actions_packed[..., start:end].astype(np.float32, copy=False)

        is_absolute = key in _ABSOLUTE_ACTION_KEYS
        relative_stats_for_key = None
        if (
            not is_absolute
            and relative_action_norm_stats is not None
            and key in relative_action_norm_stats
        ):
            relative_stats_for_key = relative_action_norm_stats[key]

        if relative_stats_for_key is not None:
            q01 = np.asarray(relative_stats_for_key["q01"], dtype=np.float32)
            q99 = np.asarray(relative_stats_for_key["q99"], dtype=np.float32)
            if q01.ndim == 2:  # per-horizon-step quantiles [H, dim]
                q01 = q01[:horizon]
                q99 = q99[:horizon]
            denorm = _q99_denorm(arr, q01, q99)
        elif key in action_norm_stats:
            q01 = np.asarray(action_norm_stats[key]["q01"], dtype=np.float32)
            q99 = np.asarray(action_norm_stats[key]["q99"], dtype=np.float32)
            denorm = _q99_denorm(arr, q01, q99)
        else:
            denorm = arr

        if not is_absolute and key in raw_state:
            ref = raw_state[key]
            if key in _EEF_XYZ_ROT6D_KEYS and denorm.shape[-1] == 9 and ref.shape[-1] == 9:
                denorm = _compose_eef_relative_to_absolute(denorm, ref)
            else:
                denorm = denorm + ref[None, :]

        out[key] = denorm.astype(np.float32, copy=False)
    return out


# ---------------------------------------------------------------------------
# Backward-compat thin wrapper kept so existing unit tests stay green.
# ---------------------------------------------------------------------------


class Gr00tTransform:
    """Stateful convenience wrapper around the stateless :func:`pack_proprio_state`
    / :func:`unpack_actions` helpers.

    Server code paths use the top-level :func:`encode` / :func:`decode`
    functions directly; this class is preserved for older callers and
    unit tests that exercise the pack / unpack pieces in isolation.
    """

    def __init__(self, embodiment_tag: str, modality_config: Mapping[str, Any]):
        self.embodiment_tag = embodiment_tag
        self.embodiment_id = embodiment_id_for_tag(embodiment_tag)
        self.modality_config = modality_config

    @property
    def state_modality(self) -> Mapping[str, Mapping[str, int]]:
        return self.modality_config.get("state", {})

    @property
    def action_modality(self) -> Mapping[str, Mapping[str, int]]:
        return self.modality_config.get("action", {})

    def transform_input(
        self, robot_obs: Mapping[str, Any], *, max_state_dim: int
    ) -> dict[str, Any]:
        raw_state = robot_obs.get("state", {})
        if not isinstance(raw_state, Mapping):
            raise TypeError(
                f"robot_obs['state'] must be a dict; got {type(raw_state).__name__}"
            )
        packed_state = pack_proprio_state(
            raw_state, self.state_modality, max_state_dim
        )
        return {
            "state": packed_state[None, None, :],
            "embodiment_id": np.array([self.embodiment_id], dtype=np.int64),
            "prompt": robot_obs.get("prompt", ""),
            "images": robot_obs.get("images"),
            "input_ids": robot_obs.get("input_ids"),
            "attention_mask": robot_obs.get("attention_mask"),
            "pixel_values": robot_obs.get("pixel_values"),
            "image_grid_thw": robot_obs.get("image_grid_thw"),
        }

    def transform_action_output(self, actions: np.ndarray) -> dict[str, np.ndarray]:
        if not self.action_modality:
            return {"actions": actions}
        return unpack_actions(actions, self.action_modality)


def get_transform(
    embodiment_tag: str, modality_config: Mapping[str, Any]
) -> Gr00tTransform:
    """Convenience factory matching the historical public API."""
    return Gr00tTransform(embodiment_tag=embodiment_tag, modality_config=modality_config)
