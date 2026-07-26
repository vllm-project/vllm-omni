# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

ACTION = "action"
OBS_STR = "observation"
OBS_PREFIX = OBS_STR + "."
OBS_STATE = OBS_STR + ".state"
OBS_TASK = OBS_STR + ".task"
OBS_IMAGE = OBS_STR + ".image"
OBS_IMAGES = OBS_IMAGE + "s"

DEFAULT_INTERNVL_PROCESSOR = os.getenv(
    "GO1_AIR_PROCESSOR_DIR",
    "OpenGVLab/InternVL3-2B-hf",
)

DEFAULT_GO1_AIR_POLICY_SERVER_CONFIG = {
    "image_resolution": [448, 448],
    "n_external_cameras": 1,
    "needs_wrist_camera": False,
    "needs_stereo_camera": False,
    "needs_session_id": True,
    "action_space": "joint_position",
    "action_horizon": 30,
    "action_dim": 16,
}


@dataclass
class Go1AirConfig:
    """Flattened GO-1-Air configuration.

    The upstream config.json is hierarchical (vision_config / llm_config /
    action_config / noise_scheduler_config). We pull the load-bearing fields
    into a flat dataclass so the rest of the pipeline can read them without
    walking the dict; advanced consumers can still inspect ``raw_config``.
    """

    type: str = "go1_air"
    dtype: str = "bfloat16"
    device: str = "cuda"

    # Action / state shape (action_chunk_size from config.json).
    chunk_size: int = 30
    n_action_steps: int = 30
    max_action_dim: int = 16
    max_state_dim: int = 16
    state_token_num: int = 3

    # Diffusion (noise_scheduler_config).
    num_inference_steps: int = 5
    num_train_timesteps: int = 1000
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "sample"
    clip_sample: bool = False

    # Vision (vision_config -> intern_vit_6b).
    image_resolution: tuple[int, int] = (448, 448)
    vision_hidden_size: int = 1024
    vision_intermediate_size: int = 4096
    vision_num_hidden_layers: int = 24
    vision_num_attention_heads: int = 16
    vision_patch_size: int = 14
    vision_qkv_bias: bool = True
    vision_qk_normalization: bool = False
    vision_norm_type: str = "layer_norm"
    vision_drop_path_rate: float = 0.1

    # Vision -> LLM bridge.
    select_layer: int = -1
    downsample_ratio: float = 0.5
    pixel_shuffle_version: str = "v2"
    img_context_token_id: int = 92546

    # Language model (llm_config -> InternLM2 GO-1 variant).
    llm_vocab_size: int = 92553
    llm_hidden_size: int = 2048
    llm_intermediate_size: int = 8192
    llm_num_hidden_layers: int = 24
    llm_num_attention_heads: int = 16
    llm_num_key_value_heads: int = 8
    llm_max_position_embeddings: int = 32768
    llm_rope_theta: float = 1000000.0
    llm_rope_scaling_factor: float = 2.0
    llm_rope_scaling_type: str = "dynamic"
    llm_rms_norm_eps: float = 1e-5
    llm_tie_word_embeddings: bool = False
    llm_pad_token_id: int = 2
    llm_bos_token_id: int = 1
    llm_eos_token_id: int = 2

    # Action expert (action_config).
    act_hidden_size: int = 1024
    act_input_hidden_size: int = 2048
    act_intermediate_size: int = 4096
    act_num_hidden_layers: int = 24
    act_num_attention_heads: int = 16
    act_num_key_value_heads: int = 8
    act_head_dim: int = 64
    act_max_position_embeddings: int = 32768
    act_rope_theta: float = 1000000.0
    act_rope_scaling_factor: float = 2.0
    act_rope_scaling_type: str = "dynamic"
    act_rms_norm_eps: float = 1e-5
    act_tie_word_embeddings: bool = True
    act_pad_token_id: int = 2

    # Latent planner: GO-1-Air ships disabled. Kept here as a plain flag so the
    # forward path can branch without loading the LP submodule weights.
    latent_planning: bool = False

    # Conversation template.
    template: str = "internlm2-chat"

    # Runtime knobs (mirroring internvla_a1).
    attn_implementation: str = "eager"
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    enable_regional_compile: bool = False
    regional_compile_dynamic: bool = True
    gradient_checkpointing: bool = False
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False

    # Full upstream config.json kept verbatim for fields we don't flatten.
    raw_config: dict[str, Any] = field(default_factory=dict)

    # Metadata consumed by the OpenPI-compatible robot serving endpoint.
    policy_server_config: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_GO1_AIR_POLICY_SERVER_CONFIG))

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str | Path) -> Go1AirConfig:
        checkpoint_dir = Path(checkpoint_dir)
        with open(checkpoint_dir / "config.json", encoding="utf-8") as f:
            raw = json.load(f)
        return cls.from_model_config(raw)

    @classmethod
    def from_model_config(cls, model_config: dict[str, Any] | None) -> Go1AirConfig:
        if not model_config:
            return cls()

        raw = dict(model_config)
        flat: dict[str, Any] = {}

        # Top-level passthroughs.
        if "action_chunk_size" in raw:
            flat["chunk_size"] = int(raw["action_chunk_size"])
            flat["n_action_steps"] = int(raw["action_chunk_size"])
        if "force_image_size" in raw:
            size = int(raw["force_image_size"])
            flat["image_resolution"] = (size, size)
        if "select_layer" in raw:
            flat["select_layer"] = int(raw["select_layer"])
        if "downsample_ratio" in raw:
            flat["downsample_ratio"] = float(raw["downsample_ratio"])
        if "ps_version" in raw:
            flat["pixel_shuffle_version"] = str(raw["ps_version"])
        if "img_context_token_id" in raw:
            flat["img_context_token_id"] = int(raw["img_context_token_id"])
        if "template" in raw:
            flat["template"] = str(raw["template"])
        if "torch_dtype" in raw and isinstance(raw["torch_dtype"], str):
            flat["dtype"] = raw["torch_dtype"]
        if "latent_planning" in raw:
            flat["latent_planning"] = bool(raw["latent_planning"])
        if "pad_token_id" in raw:
            flat["llm_pad_token_id"] = int(raw["pad_token_id"])
        if "policy_server_config" in raw:
            flat["policy_server_config"] = dict(raw["policy_server_config"])

        # Nested vision_config.
        vision_cfg = raw.get("vision_config") or {}
        if vision_cfg:
            flat["vision_hidden_size"] = int(vision_cfg.get("hidden_size", 1024))
            flat["vision_intermediate_size"] = int(vision_cfg.get("intermediate_size", 4096))
            flat["vision_num_hidden_layers"] = int(vision_cfg.get("num_hidden_layers", 24))
            flat["vision_num_attention_heads"] = int(vision_cfg.get("num_attention_heads", 16))
            flat["vision_patch_size"] = int(vision_cfg.get("patch_size", 14))
            flat["vision_qkv_bias"] = bool(vision_cfg.get("qkv_bias", True))
            flat["vision_qk_normalization"] = bool(vision_cfg.get("qk_normalization", False))
            flat["vision_norm_type"] = str(vision_cfg.get("norm_type", "layer_norm"))
            flat["vision_drop_path_rate"] = float(vision_cfg.get("drop_path_rate", 0.1))
            if "image_size" in vision_cfg:
                size = int(vision_cfg["image_size"])
                flat["image_resolution"] = (size, size)

        # Nested llm_config.
        llm_cfg = raw.get("llm_config") or {}
        if llm_cfg:
            flat["llm_vocab_size"] = int(llm_cfg.get("vocab_size", 92553))
            flat["llm_hidden_size"] = int(llm_cfg.get("hidden_size", 2048))
            flat["llm_intermediate_size"] = int(llm_cfg.get("intermediate_size", 8192))
            flat["llm_num_hidden_layers"] = int(llm_cfg.get("num_hidden_layers", 24))
            flat["llm_num_attention_heads"] = int(llm_cfg.get("num_attention_heads", 16))
            flat["llm_num_key_value_heads"] = int(llm_cfg.get("num_key_value_heads", 8))
            flat["llm_max_position_embeddings"] = int(llm_cfg.get("max_position_embeddings", 32768))
            flat["llm_rope_theta"] = float(llm_cfg.get("rope_theta", 1000000.0))
            flat["llm_rms_norm_eps"] = float(llm_cfg.get("rms_norm_eps", 1e-5))
            flat["llm_tie_word_embeddings"] = bool(llm_cfg.get("tie_word_embeddings", False))
            rope_scaling = llm_cfg.get("rope_scaling") or {}
            if rope_scaling:
                flat["llm_rope_scaling_factor"] = float(rope_scaling.get("factor", 2.0))
                flat["llm_rope_scaling_type"] = str(rope_scaling.get("type", "dynamic"))
            for tok in ("pad_token_id", "bos_token_id", "eos_token_id"):
                if tok in llm_cfg and llm_cfg[tok] is not None:
                    flat[f"llm_{tok}"] = int(llm_cfg[tok])

        # Nested action_config.
        act_cfg = raw.get("action_config") or {}
        if act_cfg:
            flat["act_hidden_size"] = int(act_cfg.get("hidden_size", 1024))
            flat["act_input_hidden_size"] = int(act_cfg.get("input_hidden_size", 2048))
            flat["act_intermediate_size"] = int(act_cfg.get("intermediate_size", 4096))
            flat["act_num_hidden_layers"] = int(act_cfg.get("num_hidden_layers", 24))
            flat["act_num_attention_heads"] = int(act_cfg.get("num_attention_heads", 16))
            flat["act_num_key_value_heads"] = int(act_cfg.get("num_key_value_heads", 8))
            flat["act_head_dim"] = int(act_cfg.get("head_dim", 64))
            flat["act_max_position_embeddings"] = int(act_cfg.get("max_position_embeddings", 32768))
            flat["act_rope_theta"] = float(act_cfg.get("rope_theta", 1000000.0))
            flat["act_rms_norm_eps"] = float(act_cfg.get("rms_norm_eps", 1e-5))
            flat["act_tie_word_embeddings"] = bool(act_cfg.get("tie_word_embeddings", True))
            if "action_dim" in act_cfg:
                flat["max_action_dim"] = int(act_cfg["action_dim"])
            if "state_dim" in act_cfg:
                flat["max_state_dim"] = int(act_cfg["state_dim"])
            if "state_token_num" in act_cfg:
                flat["state_token_num"] = int(act_cfg["state_token_num"])
            rope_scaling = act_cfg.get("rope_scaling") or {}
            if rope_scaling:
                flat["act_rope_scaling_factor"] = float(rope_scaling.get("factor", 2.0))
                flat["act_rope_scaling_type"] = str(rope_scaling.get("type", "dynamic"))
            if "pad_token_id" in act_cfg and act_cfg["pad_token_id"] is not None:
                flat["act_pad_token_id"] = int(act_cfg["pad_token_id"])

        # Noise scheduler.
        sched_cfg = raw.get("noise_scheduler_config") or {}
        if sched_cfg:
            flat["num_inference_steps"] = int(sched_cfg.get("num_inference_timesteps", 5))
            flat["num_train_timesteps"] = int(sched_cfg.get("num_train_timesteps", 1000))
            flat["beta_schedule"] = str(sched_cfg.get("beta_schedule", "squaredcos_cap_v2"))
            flat["prediction_type"] = str(sched_cfg.get("prediction_type", "sample"))
            flat["clip_sample"] = bool(sched_cfg.get("clip_sample", False))

        flat["raw_config"] = raw

        allowed = {item.name for item in dataclass_fields(cls)}
        filtered = {key: value for key, value in flat.items() if key in allowed}
        return cls(**filtered)


@dataclass
class Go1AirTrainMetadata:
    """Optional sidecar metadata for offline inference scripts."""

    action_mode: str = "delta"
    processor_model_name: str = DEFAULT_INTERNVL_PROCESSOR

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str | Path) -> Go1AirTrainMetadata:
        checkpoint_dir = Path(checkpoint_dir)
        path = checkpoint_dir / "train_config.json"
        if not path.exists():
            return cls()

        with open(path, encoding="utf-8") as f:
            raw = json.load(f)

        return cls(
            action_mode=raw.get("action_mode", "delta"),
            processor_model_name=raw.get("processor_model_name", DEFAULT_INTERNVL_PROCESSOR),
        )
