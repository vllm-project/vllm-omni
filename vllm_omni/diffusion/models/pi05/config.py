# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Config surface for the π0.5 VLA model in vllm-omni.

Deliberately shaped exactly like ``pi0/config.py``: a small dataclass that
consumes the raw LeRobot ``config.json`` (the field surface of
``lerobot.policies.pi05.PI05Config``) and keeps only the runtime-relevant
fields. Transformer dimensions are derived from ``paligemma_variant`` /
``action_expert_variant`` inside the model via ``get_gemma_config``.

What π0.5 adds on top of the π0 config surface:

* ``tokenizer_max_length = 200`` (π0 uses 48).
* ``state_num_bins`` — state is discretized into language tokens instead of
  going through a ``state_proj`` layer.
* ``use_relative_actions`` / ``relative_exclude_joints`` /
  ``action_feature_names`` — the relative-action contract. See
  ``processor_pi05.Pi05RelativeActions``.
* Quantile normalization stats. LeRobot's π0.5 defaults ``STATE`` and
  ``ACTION`` to ``NormalizationMode.QUANTILES`` where π0 uses ``MEAN_STD``.

**Checkpoint boundary rule.** A capability that the checkpoint *declares* but
this implementation does not *consume* must raise, not be silently dropped.
π0.5 checkpoints can declare MEM (short-horizon observation memory) and RTC
(real-time chunking); neither is supported here, and serving such a checkpoint
anyway would produce plausible-looking wrong actions. See
``_reject_unsupported_capabilities``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Serving dtypes, by name. Kept in sync with ``Pi05Pipeline.SUPPORTED_DTYPES``,
# which guards the path that performs the actual cast. Defined here rather than
# imported to keep config.py free of a pipeline import cycle.
SUPPORTED_DTYPE_NAMES = frozenset({"float32", "bfloat16"})

# LeRobot / OpenPI observation key conventions.
ACTION = "action"
OBS_STR = "observation"
OBS_STATE = OBS_STR + ".state"
OBS_IMAGES = OBS_STR + ".images"


class UnsupportedCheckpointCapabilityError(ValueError):
    """A checkpoint declares a capability this implementation does not consume.

    Raised at load time rather than silently ignored: every one of these
    capabilities changes what a *correct* action chunk looks like, and none of
    them is visible in the weights alone.
    """


def resolve_excluded_action_indices(
    exclude_joints: list[str] | None,
    action_names: list[str] | None,
) -> list[int]:
    """Map ``relative_exclude_joints`` names onto action-vector indices.

    Matching is exact name first, then substring (a checkpoint may name the
    gripper dimension ``gripper_position`` while the config just says
    ``gripper``). Single source of truth for both the config-time validation
    and the runtime mask in ``processor_pi05.Pi05RelativeActions``.

    Returns an empty list when there is nothing to exclude. Raises when a name
    cannot be resolved — an unresolvable exclusion would otherwise silently
    become "make this dimension relative too".
    """
    if not [name for name in (exclude_joints or []) if name]:
        return []
    if not action_names:
        raise UnsupportedCheckpointCapabilityError(
            f"Cannot resolve relative_exclude_joints={exclude_joints!r} without action_feature_names."
        )

    # LeRobot walks the action names and excludes each one any token equals or is
    # contained in, both lowercased. Walking the tokens instead and stopping at the
    # first exact hit misses a second name the token is a substring of.
    tokens = [str(name).lower() for name in exclude_joints if name]
    lowered = [str(name).lower() for name in action_names]
    indices = [i for i, name in enumerate(lowered) if any(t == name or t in name for t in tokens)]
    unresolved = [t for t in tokens if not any(t == name or t in name for name in lowered)]

    if unresolved:
        raise UnsupportedCheckpointCapabilityError(
            f"relative_exclude_joints entries {unresolved!r} match no entry in action_feature_names={action_names!r}."
        )
    return sorted(set(indices))


@dataclass
class Pi05Config:
    """π0.5 VLA config (dataclass, not an HF ``PretrainedConfig``)."""

    # Backbone variants — mapped to Gemma dimensions by ``get_gemma_config``.
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"

    # Action chunk shape.
    chunk_size: int = 50
    # The stateless OpenPI endpoint returns one complete predicted chunk.
    n_action_steps: int = 50
    max_action_dim: int = 32
    max_state_dim: int = 32

    # Flow-matching denoising schedule.
    num_inference_steps: int = 10
    # Sinusoidal timestep embedding periods (must match OpenPI/LeRobot).
    min_period: float = 4e-3
    max_period: float = 4.0

    # Image preprocessing. π0.5/SigLIP only support square inputs.
    image_resolution: tuple[int, int] = (224, 224)
    # π0.5 pads text to 200 tokens (π0 uses 48). The prompt now also carries the
    # serialized state, which is why it is so much longer.
    tokenizer_max_length: int = 200
    # Number of camera slots the model attends to (real + padded).
    max_cameras: int = 3

    # Bins the normalized state is discretized into before it enters the prompt.
    # Ref: openpi PaliGemmaTokenizer.tokenize().
    state_num_bins: int = 256

    # Weight dtype the checkpoint was saved in.
    dtype: str = "float32"

    # ── Relative actions ──────────────────────────────────────────────
    # True when the checkpoint was trained on actions relative to the current
    # state. Its ``norm_stats`` are then in relative space, so serving it without
    # the transform silently yields wrong actions — the weights look identical
    # either way.
    use_relative_actions: bool = False
    # Joint names kept absolute (gripper open/close is an absolute quantity).
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    # Needed to resolve ``relative_exclude_joints``; there is no dataset to
    # fall back on at serving time.
    action_feature_names: list[str] | None = None

    # Per-dataset normalization stats (schema matches LeRobot's
    # ``NormalizerProcessorStep``). ``None`` means identity / pass-through.
    # π0.5 defaults to quantile mode; see ``processor_pi05.build_norm_stats``.
    norm_stats: dict | None = None

    # Ordered camera-slot identities. Missing keys retain their slot and are
    # represented by an empty image with a false mask during preprocessing.
    image_feature_keys: list[str] | None = None
    # Optional map from raw OpenPI obs keys → ``image_feature_keys`` entries.
    image_key_map: dict[str, str] = field(default_factory=dict)

    # Checkpoint feature schemas. The input schema determines camera order and
    # the state width serialized into the prompt; the output schema determines
    # the unpadded action width returned on the wire.
    input_features: dict[str, Any] = field(default_factory=dict)
    output_features: dict[str, Any] = field(default_factory=dict)
    # OpenPI handshake metadata from the deploy config. Construction validates
    # it against the resolved checkpoint contract.
    policy_server_config: dict[str, Any] = field(default_factory=dict)

    # Derived from output_features[ACTION].shape; never accepted as a second
    # source of truth in config.json.
    action_dim: int = field(init=False)
    # Derived from input_features[OBS_STATE].shape — the number of values the
    # prompt serializes, not a padding target. See processor_pi05.as_state_vector.
    state_dim: int = field(init=False)

    def __post_init__(self) -> None:
        # Coerce list → tuple (JSON has no tuples). Squareness is a real
        # constraint: SigLIP only takes square inputs.
        res = self.image_resolution
        if not isinstance(res, (tuple, list)) or len(res) != 2 or res[0] != res[1]:
            raise ValueError(f"π0.5 expects a square image_resolution (H == W); got {res!r}.")
        self.image_resolution = (int(res[0]), int(res[1]))

        if self.n_action_steps != self.chunk_size:
            raise UnsupportedCheckpointCapabilityError(
                "The stateless OpenPI serving path returns one complete action chunk and "
                f"does not support n_action_steps={self.n_action_steps} with "
                f"chunk_size={self.chunk_size}."
            )
        # Mirrors Pi05Pipeline.SUPPORTED_DTYPES. This field records what the
        # *checkpoint* declares; the dtype the weights are actually cast to comes
        # from the top-level OmniDiffusionConfig and is validated there.
        if self.dtype not in SUPPORTED_DTYPE_NAMES:
            raise ValueError(f"dtype must be one of {sorted(SUPPORTED_DTYPE_NAMES)}, got {self.dtype!r}.")

        # Derive the camera order from input_features if not given explicitly.
        if self.image_feature_keys is None and self.input_features:
            self.image_feature_keys = [key for key in self.input_features if key.startswith(OBS_IMAGES + ".")]

        if self.max_cameras < 1:
            raise ValueError(f"max_cameras must be at least 1, got {self.max_cameras!r}.")
        if self.num_inference_steps < 1:
            # ``dt = -1 / num_steps`` in the Euler loop.
            raise ValueError(f"num_inference_steps must be at least 1, got {self.num_inference_steps!r}.")

        self._derive_widths()
        self._validate_norm_stats()
        self._validate_policy_server_config()
        self._validate_relative_actions()

    def _validate_norm_stats(self) -> None:
        """Statistics have to cover the widths the checkpoint declares."""
        for key, width in (("state", self.state_dim), ("action", self.action_dim)):
            for name, value in (self.norm_stats or {}).get(key, {}).items():
                if name != "mode" and len(value) != width:
                    raise ValueError(f"norm_stats[{key}][{name}] has {len(value)} entries, expected {width}.")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _derive_widths(self) -> None:
        """Read the real state and action widths off the checkpoint schema.

        π0.5 serializes the state into the prompt at its real width, so a 7-joint
        checkpoint must not be padded out to ``max_state_dim``. The action chunk
        is cropped to ``action_dim`` on the way out.
        """
        state = self.input_features.get(OBS_STATE) or {}
        action = self.output_features.get(ACTION) or {}
        if state.get("type", "STATE").upper() != "STATE":
            raise ValueError(f"input_features[{OBS_STATE!r}].type must be STATE, got {state['type']!r}.")
        if action.get("type", "ACTION").upper() != "ACTION":
            raise ValueError(f"output_features[{ACTION!r}].type must be ACTION, got {action['type']!r}.")
        self.state_dim = int(state["shape"][0]) if state else self.max_state_dim
        self.action_dim = int(action["shape"][0]) if action else self.max_action_dim
        if self.state_dim > self.max_state_dim or self.action_dim > self.max_action_dim:
            raise ValueError(
                f"Checkpoint declares state_dim={self.state_dim} / action_dim={self.action_dim}, "
                f"over max_state_dim={self.max_state_dim} / max_action_dim={self.max_action_dim}."
            )

    def _validate_policy_server_config(self) -> None:
        """The OpenPI handshake has to advertise the resolved contract."""
        expected = {
            "action_horizon": self.chunk_size,
            "action_dim": self.action_dim,
            "max_action_dim": self.max_action_dim,
            "max_cameras": self.max_cameras,
            "image_resolution": list(self.image_resolution),
        }
        for key, value in expected.items():
            declared = (self.policy_server_config or {}).get(key)
            if declared is None:
                continue
            if key == "image_resolution":
                declared = list(declared)
            if declared != value:
                raise ValueError(
                    f"policy_server_config.{key}={declared!r} does not match the resolved value {value!r}."
                )

    def _validate_relative_actions(self) -> None:
        """Resolve the excluded joints at load, so a bad name fails before serving."""
        if not self.use_relative_actions or not self.relative_exclude_joints:
            return
        if not self.action_feature_names:
            raise UnsupportedCheckpointCapabilityError(
                f"relative_exclude_joints={self.relative_exclude_joints!r} needs action_feature_names, "
                "which the checkpoint does not carry."
            )
        resolve_excluded_action_indices(self.relative_exclude_joints, self.action_feature_names)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, checkpoint_dir: str | Path) -> Pi05Config:
        """Build from a checkpoint directory's ``config.json``.

        Normalization stats are *not* in ``config.json``. LeRobot keeps them in
        the processor sidecar, so they are loaded separately and backfilled
        here — see :func:`load_lerobot_norm_stats`.
        """
        checkpoint_dir = Path(checkpoint_dir)
        config_path = checkpoint_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"π0.5 checkpoint is missing required config: {config_path}.")
        with open(config_path, encoding="utf-8") as f:
            raw = json.load(f)
        config = cls.from_model_config(raw)

        if config.norm_stats is None:
            stats = load_lerobot_norm_stats(checkpoint_dir)
            if stats:
                config.norm_stats = stats
                config._validate_norm_stats()
        return config

    @classmethod
    def from_model_config(cls, model_config: dict[str, Any] | None) -> Pi05Config:
        """Build from a config dict (LeRobot ``config.json`` or deploy yaml)."""
        if not model_config:
            return cls()

        raw = dict(model_config)
        _reject_unsupported_capabilities(raw)

        model_type = raw.pop("type", "pi05")
        if model_type != "pi05":
            raise ValueError(f"Expected a π0.5 checkpoint (type='pi05'), got type={model_type!r}.")

        if "image_resolution" in raw:
            raw["image_resolution"] = tuple(raw["image_resolution"])

        # A config.json carries the whole training recipe; none of it reaches
        # inference. Capabilities we cannot serve are rejected above instead.
        allowed = {item.name for item in dataclass_fields(cls) if item.init}
        filtered = {key: value for key, value in raw.items() if key in allowed}
        dropped = sorted(set(raw) - allowed)
        if dropped:
            logger.debug("π0.5 config: ignoring %d non-runtime key(s): %s", len(dropped), dropped)
        return cls(**filtered)


# ----------------------------------------------------------------------
# LeRobot normalization-stats sidecar
# ----------------------------------------------------------------------
# Stats live beside config.json: ``policy_preprocessor.json`` names a
# safetensors file per stateful step, keyed ``"<feature_name>.<stat_name>"``.
_PREPROCESSOR_JSON = "policy_preprocessor.json"

# The mode must come from ``norm_map``: a state_dict carries all of
# mean/std/min/max/q01/q99 regardless, so guessing from the present keys would
# read a QUANTILES checkpoint as mean_std and apply a wrong affine map.
_NORM_MODE_FROM_LEROBOT: dict[str, tuple[str, str, str] | None] = {
    "IDENTITY": None,
    "MEAN_STD": ("mean_std", "mean", "std"),
    "MIN_MAX": ("min_max", "min", "max"),
    "QUANTILES": ("quantile", "q01", "q99"),
}
# Renaming happens in an earlier processor step, so these names are canonical
# here. VISUAL is absent by design: images are normalized in the processor.
_NORM_STATS_FEATURES = {OBS_STATE: ("STATE", "state"), ACTION: ("ACTION", "action")}


def load_lerobot_norm_stats(checkpoint_dir: str | Path) -> dict[str, dict[str, Any]] | None:
    """Load normalization stats from a LeRobot checkpoint's processor sidecar.

    Returns a ``norm_stats``-shaped dict (``{"state": {...}, "action": {...}}``)
    carrying an **explicit** ``mode``, or ``None`` when the checkpoint ships no
    stats. ``lerobot/pi05_base`` is the latter case: its normalizer step has no
    ``state_file``, i.e. normalization is identity and the client is expected to
    send an already-normalized state.

    Raises on a mode we cannot reproduce rather than serving the checkpoint with
    the wrong transform, which fails silently — a wrongly normalized state still
    yields a plausible-looking action chunk.
    """
    checkpoint_dir = Path(checkpoint_dir)
    preprocessor_path = checkpoint_dir / _PREPROCESSOR_JSON
    if not preprocessor_path.exists():
        return None
    with open(preprocessor_path, encoding="utf-8") as f:
        steps = json.load(f).get("steps", [])

    step = next((s for s in steps if s.get("registry_name") == "normalizer_processor"), {})
    state_file = step.get("state_file")
    if not state_file:
        logger.info(
            "π0.5 config: %s declares no normalizer state file; the state passes through.",
            _PREPROCESSOR_JSON,
        )
        return None
    state_path = checkpoint_dir / state_file
    if not state_path.exists():
        raise FileNotFoundError(f"{preprocessor_path} declares normalizer state {state_file!r}, which is missing.")

    norm_map = {str(k).upper(): str(v).upper() for k, v in ((step.get("config") or {}).get("norm_map") or {}).items()}

    import safetensors.torch

    flat = safetensors.torch.load_file(str(state_path))

    stats: dict[str, dict[str, Any]] = {}
    for feature_name, (feature_type, stats_key) in _NORM_STATS_FEATURES.items():
        lerobot_mode = norm_map.get(feature_type, "IDENTITY")
        if lerobot_mode not in _NORM_MODE_FROM_LEROBOT:
            raise ValueError(
                f"π0.5 cannot reproduce normalization mode {lerobot_mode!r} for {feature_type}. "
                f"Expected one of {sorted(_NORM_MODE_FROM_LEROBOT)}."
            )
        selected = _NORM_MODE_FROM_LEROBOT[lerobot_mode]
        if selected is None:
            continue

        mode, *stat_names = selected
        missing = [name for name in stat_names if f"{feature_name}.{name}" not in flat]
        if missing:
            raise ValueError(
                f"{feature_type} declares {lerobot_mode} normalization but the state file is missing {missing}."
            )
        stats[stats_key] = {"mode": mode} | {name: flat[f"{feature_name}.{name}"].tolist() for name in stat_names}

    if not stats:
        return None
    logger.info(
        "π0.5 config: loaded normalization stats from %s — %s.",
        state_file,
        ", ".join(f"{key}={value['mode']}" for key, value in sorted(stats.items())),
    )
    return stats


# ----------------------------------------------------------------------
# Checkpoint-boundary rule
# ----------------------------------------------------------------------
# Each entry: config key → (is this value *enabled*?, why we cannot serve it).
# Every capability here changes what a *correct* action chunk looks like, and
# none is visible in the weights, so a checkpoint that enables one is refused
# rather than served plausibly wrong. See recipes/lerobot/Pi05.md for the
# longer explanations.
#
# The predicate is per key because these are not all flags: ``n_obs_steps`` is
# a count whose supported value is 1, and an empty ``rtc_config`` still selects
# RTC — LeRobot's ``RTCConfig`` defaults ``enabled=True``.
_UNSUPPORTED: dict[str, tuple[Callable[[Any], bool], str]] = {
    "use_peft": (bool, "adapter-aware loading; this loader takes a merged checkpoint only"),
    "empty_cameras": (bool, "mutates the input schema; declare camera features explicitly instead"),
    "use_visual_memory": (bool, "MEM needs a per-session observation history, which this path does not keep"),
    "use_proprioceptive_memory": (
        bool,
        "MEM sends the state as a projected token, not the discretized prompt built here",
    ),
    "rtc_config": (
        lambda value: value is not None,
        "RTC needs prefix guidance in the denoising loop and per-request chunk carry-over",
    ),
    "n_obs_steps": (
        lambda value: bool(value) and value > 1,
        "an observation history; this path is first-order Markov",
    ),
}


def _reject_unsupported_capabilities(raw: dict[str, Any]) -> None:
    """Raise if the checkpoint enables something we do not consume.

    Scope note: this validates that we correctly consume *the checkpoint we were
    handed*. Choosing which checkpoint to load belongs to the caller.
    """
    problems = [
        f"  - {key}={raw[key]!r}: {why}"
        for key, (is_enabled, why) in _UNSUPPORTED.items()
        if key in raw and is_enabled(raw[key])
    ]
    if problems:
        raise UnsupportedCheckpointCapabilityError(
            "This checkpoint declares capabilities the vllm-omni π0.5 implementation "
            "does not support:\n" + "\n".join(problems) + "\nServing it anyway would "
            "produce plausible-looking but wrong actions."
        )

    # Not an error: it records that the checkpoint was trained with clean action
    # prefixes sampled in, which is a property of training rather than a request
    # to run RTC. Such a checkpoint is still correct to serve without RTC.
    delay = raw.get("rtc_training_max_delay")
    if delay:
        logger.info(
            "π0.5 config: checkpoint was trained with rtc_training_max_delay=%s. "
            "Real-Time Chunking is not implemented here; serving proceeds without it.",
            delay,
        )
