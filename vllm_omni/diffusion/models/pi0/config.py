# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config surface for the π0 (Pi-Zero) VLA model in vllm-omni.

Only the parameters that actually shape runtime behaviour live here; the
transformer dimensions (hidden size, head count, ...) are derived from
``paligemma_variant`` / ``action_expert_variant`` inside the model via
``get_gemma_config`` (see ``modeling_pi0``).

The resolver reads the **raw LeRobot ``config.json``** (the field surface of
``lerobot.policies.pi0.PI0Config``): only the recognized dataclass fields are
used; any other keys in the file are ignored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

# LeRobot / OpenPI observation key conventions.
ACTION = "action"
OBS_STR = "observation"
OBS_STATE = OBS_STR + ".state"
OBS_IMAGES = OBS_STR + ".images"


@dataclass
class Pi0Config:
    """π0 VLA config (dataclass, not an HF ``PretrainedConfig``).

    Mirrors the runtime-relevant subset of LeRobot ``PI0Config`` plus a couple
    of serving-side knobs (``max_cameras``, ``image_feature_keys``,
    ``image_key_map``, ``norm_stats``).
    """

    # Backbone variants — mapped to Gemma dimensions by ``get_gemma_config``.
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"

    # Action chunk shape.
    chunk_size: int = 50
    max_action_dim: int = 32
    max_state_dim: int = 32

    # Flow-matching denoising schedule.
    num_inference_steps: int = 10
    # Sinusoidal timestep embedding periods (must match OpenPI/LeRobot).
    min_period: float = 4e-3
    max_period: float = 4.0

    # Image preprocessing. π0/SigLIP only support square inputs.
    image_resolution: tuple[int, int] = (224, 224)
    tokenizer_max_length: int = 48
    # Number of camera slots the model attends to (real + padded).
    max_cameras: int = 3

    # Weight dtype the checkpoint was saved in; parity passes this through to
    # match LeRobot's ``PI0Config.dtype``.
    dtype: str = "float32"

    # Per-dataset normalization stats, optional (schema matches LeRobot's
    # ``NormalizerProcessorStep``; consumed by ``_build_norm_buffers``). ``None``
    # means identity / pass-through (the ``lerobot/pi0_base`` case).
    norm_stats: dict | None = None

    # Ordered list of image feature keys (e.g. ``observation.images.base_0_rgb``)
    # taken from the checkpoint's ``input_features``. This is the **camera order**
    # that must be reproduced for LeRobot parity.
    image_feature_keys: list[str] | None = None

    # Optional map from raw OpenPI obs keys → ``image_feature_keys`` entries, so
    # serving can translate wire obs (e.g. ``observation/exterior_image_0_left``)
    # into the checkpoint's camera identities. Empty → identity match on key.
    image_key_map: dict[str, str] = field(default_factory=dict)

    # Stored for reference / camera-order derivation; not used at inference.
    input_features: dict[str, Any] = field(default_factory=dict)
    output_features: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Coerce list → tuple (JSON has no tuples) and validate squareness.
        res = self.image_resolution
        if not isinstance(res, (tuple, list)) or len(res) != 2 or res[0] != res[1]:
            raise ValueError(f"π0 expects a square image_resolution (H == W); got {res!r}.")
        self.image_resolution = (int(res[0]), int(res[1]))

        # Derive the camera order from input_features if not given explicitly.
        if self.image_feature_keys is None and self.input_features:
            self.image_feature_keys = [key for key in self.input_features if key.startswith(OBS_IMAGES + ".")]

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, checkpoint_dir: str | Path) -> Pi0Config:
        """Build from a checkpoint directory's ``config.json``."""
        checkpoint_dir = Path(checkpoint_dir)
        config_path = checkpoint_dir / "config.json"
        if not config_path.exists():
            return cls()
        with open(config_path, encoding="utf-8") as f:
            raw = json.load(f)
        return cls.from_model_config(raw)

    @classmethod
    def from_model_config(cls, model_config: dict[str, Any] | None) -> Pi0Config:
        """Build from a config dict (LeRobot ``config.json`` or deploy yaml).

        Keeps only the recognized dataclass fields (the LeRobot config also
        carries many training-only keys) and coerces ``image_resolution`` to a
        tuple.
        """
        if not model_config:
            return cls()

        raw = dict(model_config)
        if "image_resolution" in raw:
            raw["image_resolution"] = tuple(raw["image_resolution"])

        allowed = {item.name for item in dataclass_fields(cls)}
        filtered = {key: value for key, value in raw.items() if key in allowed}
        return cls(**filtered)
