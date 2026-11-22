# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Strict schema-v1 artifact contract for Cosmos3-Nano-Sim-Auto.

The imaginaire4 exporter (``export_model.py --cosmos3-nano-sim-auto`` followed
by ``convert_model_to_diffusers.py``) embeds one ``cosmos3_nano_sim_auto``
object in ``transformer/config.json``. It records everything the rolling
causal multiview student needs at inference that is not an architecture field:
the sensor clock and chunk partition, the KV ring, the replay visibility
policy, the segmented prompt policy, mRoPE constants, the fixed-step student
schedule and the optional LiDAR tokenizer contract. Validation is exact: unknown
or missing fields fail, and deployment YAML may not override artifact data.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

from vllm_omni.diffusion.models.cosmos3.lidar import validate_lidar_config
from vllm_omni.diffusion.models.cosmos3.multiview_packing import spatial_patch_hw
from vllm_omni.diffusion.models.cosmos3.multiview_prompts import MADS_CAMERA_ATTRIBUTES

COSMOS3_NANO_SIM_AUTO_SCHEMA_VERSION = 1
COSMOS3_NANO_SIM_AUTO_ARTIFACT_KEY = "cosmos3_nano_sim_auto"

ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "checkpoint_id",
        "checkpoint_iteration",
        "checkpoint_hash",
        "attention_mode",
        "video_temporal_causal",
        "cameras",
        "rgb",
        "lidar",
        "lidar_latent_patch_size_hw",
        "chunk",
        "ring",
        "replay",
        "prompt",
        "mrope",
        "fixed_step_sampler_config",
        "inference_defaults",
    }
)
RGB_FIELDS = frozenset(
    {"latent_channels", "latent_patch_size", "vae_spatial_compression_factor", "temporal_compression_factor"}
)
CHUNK_FIELDS = frozenset({"frames_per_chunk"})
RING_FIELDS = frozenset({"cache_chunks", "sink_chunks"})
REPLAY_FIELDS = frozenset(
    {
        "control_visibility",
        "controls_read_strict_past_clean_rgb",
        "clean_pass_causality",
        "attention_scope",
        "cross_view_past_window_seconds",
        "deduplicate_cross_view",
        "lidar_attends_captions",
    }
)
PROMPT_FIELDS = frozenset(
    {
        "policy",
        "text_sink_tokens",
        "text_position_mode",
        "context_before",
        "context_after",
        "media_text_extent",
        "media_rope_scale",
        "sink_age_cap_seconds",
        "use_first_chunk_only",
        "system_prompt_rgb",
        "system_prompt_joint",
        "fractional_duration",
        "emphasize_control_in_prompt",
        "text_cache_max_len",
    }
)
MROPE_FIELDS = frozenset({"temporal_modality_margin", "reset_spatial_ids", "base_fps", "enable_fps_modulation"})
SAMPLER_FIELDS = frozenset({"t_list", "sample_type", "num_train_timesteps"})
DEFAULTS_FIELDS = frozenset({"resolution", "fps", "guidance", "control_guidance"})

#: Reference system prompt identifiers; the pipeline maps them to the exact prompt strings.
SYSTEM_PROMPT_IDS = frozenset({"av_multiview_transfer", "av_joint_camera_lidar_transfer"})

_DEPLOY_ARTIFACT_ENVELOPES = frozenset({COSMOS3_NANO_SIM_AUTO_ARTIFACT_KEY, "multiview", "cosmos3_nano_sim_bimanual"})


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        return converted if isinstance(converted, dict) else {}
    params = getattr(value, "params", None)
    return params if isinstance(params, dict) else {}


def _deployment_roots(config: Any) -> list[tuple[str, dict[str, Any]]]:
    roots: list[tuple[str, dict[str, Any]]] = []
    for attr in ("custom_pipeline_args", "model_config"):
        value = _mapping(getattr(config, attr, None))
        if value:
            roots.append((attr, value))
    return roots


def deploy_option(config: Any, key: str, default: Any = None) -> Any:
    """Read one deployment option without inspecting artifact envelopes."""
    direct = getattr(config, key, None)
    if direct is not None:
        return direct
    for _, root in _deployment_roots(config):
        if root.get(key) is not None:
            return root[key]
    return default


def _exact_fields(payload: Any, allowed: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"Cosmos3-Nano-Sim-Auto artifact {name} must be an object, got {type(payload).__name__}.")
    missing = sorted(allowed - payload.keys())
    unknown = sorted(payload.keys() - allowed)
    if missing or unknown:
        raise ValueError(
            f"Cosmos3-Nano-Sim-Auto artifact {name} field set mismatch: missing={missing}, unknown={unknown}. "
            "Re-export the checkpoint with the matching imaginaire4 exporter."
        )
    return payload


def _int(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Cosmos3-Nano-Sim-Auto artifact {name} must be an integer, got {value!r}.")
    if value < minimum:
        raise ValueError(f"Cosmos3-Nano-Sim-Auto artifact {name} must be >= {minimum}, got {value}.")
    return value


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Cosmos3-Nano-Sim-Auto artifact {name} must be a boolean, got {value!r}.")
    return value


def _float(value: Any, name: str, *, minimum: float | None = None, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"Cosmos3-Nano-Sim-Auto artifact {name} must be a finite number, got {value!r}.")
    result = float(value)
    if positive and result <= 0:
        raise ValueError(f"Cosmos3-Nano-Sim-Auto artifact {name} must be positive, got {value!r}.")
    if minimum is not None and result < minimum:
        raise ValueError(f"Cosmos3-Nano-Sim-Auto artifact {name} must be >= {minimum}, got {value!r}.")
    return result


def _str(value: Any, name: str, *, choices: frozenset[str] | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Cosmos3-Nano-Sim-Auto artifact {name} must be a non-empty string, got {value!r}.")
    if choices is not None and value not in choices:
        raise ValueError(f"Cosmos3-Nano-Sim-Auto artifact {name}={value!r} is not one of {sorted(choices)}.")
    return value


@dataclass(frozen=True)
class Cosmos3NanoSimAutoManifest:
    """Immutable, validated schema-v1 artifact of a Cosmos3-Nano-Sim-Auto checkpoint."""

    checkpoint_id: str
    checkpoint_iteration: int
    checkpoint_hash: str
    cameras: tuple[str, ...]
    # RGB tokenizer / transformer geometry.
    latent_channels: int
    latent_patch_size: int
    vae_spatial_compression_factor: int
    temporal_compression_factor: int
    # Sensor clock.
    frames_per_chunk: int
    # KV ring.
    cache_chunks: int
    sink_chunks: int
    # Replay visibility policy.
    cross_view_past_window_seconds: float
    lidar_attends_captions: bool
    # Segmented prompting.
    text_sink_tokens: int
    use_first_chunk_only: bool
    system_prompt_rgb: str
    system_prompt_joint: str
    fractional_duration: bool
    emphasize_control_in_prompt: bool
    text_cache_max_len: int
    # mRoPE.
    temporal_modality_margin: int
    base_fps: float
    enable_fps_modulation: bool
    # Student schedule.
    t_list: tuple[float, ...]
    sample_type: str
    num_train_timesteps: int
    # Defaults.
    default_resolution: str
    default_fps: float
    # Optional LiDAR tokenizer contract (validated by the shared LiDAR validator).
    lidar: dict[str, Any] | None = None
    lidar_latent_patch_size_hw: tuple[int, int] | None = None
    schema_version: int = COSMOS3_NANO_SIM_AUTO_SCHEMA_VERSION
    attention_mode: str = "multiview"
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    # -- derived ---------------------------------------------------------

    @property
    def joint(self) -> bool:
        return self.lidar is not None

    @property
    def rgb_seconds_per_frame(self) -> float:
        return self.temporal_compression_factor / self.default_fps

    @property
    def chunk_seconds(self) -> float:
        return self.frames_per_chunk * self.rgb_seconds_per_frame

    @property
    def lidar_fps(self) -> float | None:
        return None if self.lidar is None else float(self.lidar["fps"])

    @property
    def lidar_seconds_per_frame(self) -> float | None:
        if self.lidar is None:
            return None
        return float(self.lidar["temporal_compression_factor"]) / float(self.lidar["fps"])

    @property
    def digest(self) -> str:
        """Stable fingerprint of the artifact contract for session fingerprints."""
        return hashlib.sha256(json.dumps(self.raw, sort_keys=True, default=str).encode()).hexdigest()

    # -- construction ----------------------------------------------------

    @classmethod
    def from_artifact(cls, artifact: Any) -> Cosmos3NanoSimAutoManifest:
        payload = _exact_fields(_mapping(artifact) or artifact, ARTIFACT_FIELDS, "root")
        schema = _int(payload["schema_version"], "schema_version")
        if schema != COSMOS3_NANO_SIM_AUTO_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported Cosmos3-Nano-Sim-Auto artifact schema_version={schema}; "
                f"expected {COSMOS3_NANO_SIM_AUTO_SCHEMA_VERSION}."
            )
        attention_mode = _str(payload["attention_mode"], "attention_mode", choices=frozenset({"multiview"}))
        if _bool(payload["video_temporal_causal"], "video_temporal_causal") is not True:
            raise ValueError("Cosmos3-Nano-Sim-Auto requires video_temporal_causal=true.")
        cameras = payload["cameras"]
        if (
            not isinstance(cameras, list)
            or not cameras
            or any(not isinstance(camera, str) or camera not in MADS_CAMERA_ATTRIBUTES for camera in cameras)
            or len(set(cameras)) != len(cameras)
        ):
            raise ValueError("Cosmos3-Nano-Sim-Auto cameras must be unique MADS camera keys.")
        rgb = _exact_fields(payload["rgb"], RGB_FIELDS, "rgb")
        chunk = _exact_fields(payload["chunk"], CHUNK_FIELDS, "chunk")
        ring = _exact_fields(payload["ring"], RING_FIELDS, "ring")
        replay = _exact_fields(payload["replay"], REPLAY_FIELDS, "replay")
        prompt = _exact_fields(payload["prompt"], PROMPT_FIELDS, "prompt")
        mrope = _exact_fields(payload["mrope"], MROPE_FIELDS, "mrope")
        sampler = _exact_fields(payload["fixed_step_sampler_config"], SAMPLER_FIELDS, "fixed_step_sampler_config")
        defaults = _exact_fields(payload["inference_defaults"], DEFAULTS_FIELDS, "inference_defaults")

        # The replay policy the planner implements is fixed; anything else is a different model.
        expected_replay = {
            "control_visibility": "causal",
            "controls_read_strict_past_clean_rgb": True,
            "clean_pass_causality": "chunk",
            "attention_scope": "decomposed",
            "deduplicate_cross_view": True,
        }
        for key, expected in expected_replay.items():
            if replay[key] != expected:
                raise ValueError(
                    f"Cosmos3-Nano-Sim-Auto replay.{key}={replay[key]!r} is unsupported; expected {expected!r}."
                )
        window = _float(replay["cross_view_past_window_seconds"], "replay.cross_view_past_window_seconds", minimum=0.0)
        lidar_attends_captions = _bool(replay["lidar_attends_captions"], "replay.lidar_attends_captions")
        if not lidar_attends_captions:
            raise ValueError("Cosmos3-Nano-Sim-Auto requires replay.lidar_attends_captions=true.")

        expected_prompt = {
            "policy": "segmented",
            "text_position_mode": "fixed",
            "context_before": 0,
            "context_after": 0,
            "media_text_extent": None,
            "media_rope_scale": 1.0,
            "sink_age_cap_seconds": None,
        }
        for key, expected in expected_prompt.items():
            if prompt[key] != expected:
                raise ValueError(
                    f"Cosmos3-Nano-Sim-Auto prompt.{key}={prompt[key]!r} is unsupported; expected {expected!r}."
                )
        cache_chunks = _int(ring["cache_chunks"], "ring.cache_chunks", minimum=2)
        sink_chunks = _int(ring["sink_chunks"], "ring.sink_chunks", minimum=1)
        if sink_chunks >= cache_chunks:
            raise ValueError("Cosmos3-Nano-Sim-Auto ring must retain at least one non-sink chunk.")
        raw_t_list = sampler["t_list"]
        if not isinstance(raw_t_list, list) or not raw_t_list:
            raise ValueError("Cosmos3-Nano-Sim-Auto fixed_step_sampler_config.t_list must be a non-empty list.")
        t_list = tuple(_float(value, "fixed_step_sampler_config.t_list entry") for value in raw_t_list)
        if (
            t_list[0] != 1.0
            or any(later >= earlier for earlier, later in zip(t_list[:-1], t_list[1:]))
            or t_list[-1] <= 0
        ):
            raise ValueError("Cosmos3-Nano-Sim-Auto t_list must start at 1.0 and be strictly decreasing and positive.")
        sample_type = _str(
            sampler["sample_type"], "fixed_step_sampler_config.sample_type", choices=frozenset({"sde", "ode"})
        )
        resolution = _str(defaults["resolution"], "inference_defaults.resolution", choices=frozenset({"480", "720"}))

        lidar = payload["lidar"]
        lidar_patch = payload["lidar_latent_patch_size_hw"]
        if lidar is None:
            if lidar_patch is not None:
                raise ValueError("Cosmos3-Nano-Sim-Auto lidar_latent_patch_size_hw requires a lidar block.")
        else:
            lidar = dict(_mapping(lidar) or lidar)
            validate_lidar_config(lidar)
            lidar_patch = tuple(spatial_patch_hw(lidar_patch))
        return cls(
            schema_version=schema,
            checkpoint_id=_str(payload["checkpoint_id"], "checkpoint_id"),
            checkpoint_iteration=_int(payload["checkpoint_iteration"], "checkpoint_iteration", minimum=0),
            checkpoint_hash=_str(payload["checkpoint_hash"], "checkpoint_hash"),
            attention_mode=attention_mode,
            cameras=tuple(cameras),
            latent_channels=_int(rgb["latent_channels"], "rgb.latent_channels"),
            latent_patch_size=_int(rgb["latent_patch_size"], "rgb.latent_patch_size"),
            vae_spatial_compression_factor=_int(
                rgb["vae_spatial_compression_factor"], "rgb.vae_spatial_compression_factor"
            ),
            temporal_compression_factor=_int(rgb["temporal_compression_factor"], "rgb.temporal_compression_factor"),
            frames_per_chunk=_int(chunk["frames_per_chunk"], "chunk.frames_per_chunk"),
            cache_chunks=cache_chunks,
            sink_chunks=sink_chunks,
            cross_view_past_window_seconds=window,
            lidar_attends_captions=lidar_attends_captions,
            text_sink_tokens=_int(prompt["text_sink_tokens"], "prompt.text_sink_tokens", minimum=0),
            use_first_chunk_only=_bool(prompt["use_first_chunk_only"], "prompt.use_first_chunk_only"),
            system_prompt_rgb=_str(prompt["system_prompt_rgb"], "prompt.system_prompt_rgb", choices=SYSTEM_PROMPT_IDS),
            system_prompt_joint=_str(
                prompt["system_prompt_joint"], "prompt.system_prompt_joint", choices=SYSTEM_PROMPT_IDS
            ),
            fractional_duration=_bool(prompt["fractional_duration"], "prompt.fractional_duration"),
            emphasize_control_in_prompt=_bool(
                prompt["emphasize_control_in_prompt"], "prompt.emphasize_control_in_prompt"
            ),
            text_cache_max_len=_int(prompt["text_cache_max_len"], "prompt.text_cache_max_len"),
            temporal_modality_margin=_int(
                mrope["temporal_modality_margin"], "mrope.temporal_modality_margin", minimum=0
            ),
            base_fps=_float(mrope["base_fps"], "mrope.base_fps", positive=True),
            enable_fps_modulation=_bool(mrope["enable_fps_modulation"], "mrope.enable_fps_modulation"),
            t_list=t_list,
            sample_type=sample_type,
            num_train_timesteps=_int(sampler["num_train_timesteps"], "fixed_step_sampler_config.num_train_timesteps"),
            default_resolution=resolution,
            default_fps=_float(defaults["fps"], "inference_defaults.fps", positive=True),
            lidar=lidar,
            lidar_latent_patch_size_hw=lidar_patch,
            raw=json.loads(json.dumps(payload, default=str)),
        )

    @classmethod
    def from_od_config(cls, od_config: Any) -> Cosmos3NanoSimAutoManifest:
        for attr, root in _deployment_roots(od_config):
            envelopes = sorted(_DEPLOY_ARTIFACT_ENVELOPES & set(root))
            if envelopes:
                raise ValueError(
                    f"Cosmos3-Nano-Sim-Auto artifact envelopes are not supported in deploy {attr}: {envelopes}. "
                    f"The artifact must come from transformer/config.json['{COSMOS3_NANO_SIM_AUTO_ARTIFACT_KEY}']."
                )
            fields_present = sorted(ARTIFACT_FIELDS & set(root))
            if fields_present:
                raise ValueError(
                    f"Cosmos3-Nano-Sim-Auto artifact fields must not be placed at deploy root {attr}: {fields_present}."
                )
        transformer_config = _mapping(getattr(od_config, "tf_model_config", None))
        artifact = transformer_config.get(COSMOS3_NANO_SIM_AUTO_ARTIFACT_KEY)
        if artifact is None:
            raise ValueError(
                f"Cosmos3-Nano-Sim-Auto requires transformer/config.json['{COSMOS3_NANO_SIM_AUTO_ARTIFACT_KEY}']. "
                "Export the checkpoint through both imaginaire4 stages with --cosmos3-nano-sim-auto."
            )
        if transformer_config.get("backbone_type") != "cosmos3_multiview":
            raise ValueError(
                "Cosmos3-Nano-Sim-Auto requires transformer/config.json backbone_type='cosmos3_multiview', "
                f"got {transformer_config.get('backbone_type')!r}."
            )
        return cls.from_artifact(artifact)

    def validate_request_cameras(self, cameras: list[str]) -> None:
        if not cameras or len(set(cameras)) != len(cameras):
            raise ValueError("Cosmos3-Nano-Sim-Auto requests need a non-empty list of unique cameras.")
        unknown = [camera for camera in cameras if camera not in self.cameras]
        if unknown:
            raise ValueError(
                f"Cosmos3-Nano-Sim-Auto cameras {unknown} are not part of the exported rig {list(self.cameras)}."
            )


def validate_sim_auto_parallel_config(config: Any) -> None:
    """v1 runs TP only: no SP/PP/DP/EP, no VAE patch parallelism, no CFG parallelism."""
    parallel = getattr(config, "parallel_config", None)
    unsupported = {
        "sequence_parallel_size": "the causal ring and maskless passes are head-parallel only in v1",
        "ulysses_degree": "the causal ring and maskless passes are head-parallel only in v1",
        "ring_degree": "ring attention is not supported",
        "cfg_parallel_size": "the distilled student runs guidance 1 with a single branch",
        "pipeline_parallel_size": "all GEN layers execute on every rank",
        "data_parallel_size": "use stage num_replicas for independent workers",
        "vae_patch_parallel_size": "streaming per-camera decode bypasses distributed VAE execution",
    }
    for name, reason in unsupported.items():
        if (getattr(parallel, name, 1) or 1) > 1:
            raise ValueError(f"Cosmos3-Nano-Sim-Auto does not support {name}>1: {reason}.")
    if getattr(parallel, "enable_expert_parallel", False):
        raise ValueError("Cosmos3-Nano-Sim-Auto uses dense MLP blocks; there are no experts to distribute.")


__all__ = [
    "COSMOS3_NANO_SIM_AUTO_ARTIFACT_KEY",
    "COSMOS3_NANO_SIM_AUTO_SCHEMA_VERSION",
    "Cosmos3NanoSimAutoManifest",
    "deploy_option",
    "validate_sim_auto_parallel_config",
]
