# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Realtime-Venus configuration without the checkpoint's Transformers 4 API."""

import hashlib
from pathlib import Path

import yaml
from transformers import AutoConfig, PretrainedConfig, Qwen3Config, WhisperConfig


class RealtimeVenusOmniConfig(Qwen3Config):
    model_type = "realtime_venus_omni"
    sub_configs = {
        "vision_config": PretrainedConfig,
        "audio_config": WhisperConfig,
        "tts_config": PretrainedConfig,
        "slice_config": PretrainedConfig,
    }

    def __init__(self, vision_config=None, audio_config=None, tts_config=None, slice_config=None, **kwargs):
        super().__init__(**kwargs)
        for name, value in (
            ("vision_config", vision_config),
            ("audio_config", audio_config),
            ("tts_config", tts_config),
            ("slice_config", slice_config),
        ):
            setattr(
                self,
                name,
                self.sub_configs[name](**(value or {})) if isinstance(value, dict) or value is None else value,
            )

    @classmethod
    def get_config_dict(cls, pretrained_model_name_or_path, **kwargs):
        config, remaining = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
        return cls._expand_config(config, pretrained_model_name_or_path, kwargs), remaining

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        config_dict = cls._expand_config(config_dict, kwargs.get("name_or_path", ""), kwargs)
        return super().from_dict(config_dict, **kwargs)

    @classmethod
    def _expand_config(cls, config, pretrained_model_name_or_path, kwargs):
        ref = config.get("venus_config_ref")
        if ref is None:
            return config
        if not isinstance(ref, dict) or set(ref) != {"repo_id", "repo_type", "filename", "revision", "sha256", "model"}:
            raise ValueError("Invalid Realtime-Venus configuration reference")
        if (
            ref["repo_id"] != "inclusionAI/Realtime-Venus"
            or ref["repo_type"] != "model"
            or ref["filename"] != "config.yaml"
            or ref["model"] != "omni"
        ):
            raise ValueError("Realtime-Venus configuration reference identity mismatch")
        for key, length in (("revision", 40), ("sha256", 64)):
            value = ref[key]
            if not isinstance(value, str) or len(value) != length or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"Realtime-Venus requires a pinned {key}")
        model_path = Path(pretrained_model_name_or_path)
        root = None
        if model_path.is_dir():
            root = (model_path / kwargs.get("subfolder", "")).parent / "config.yaml"
        elif model_path.is_file():
            root = model_path.parent.parent / "config.yaml"
        raw = root.read_bytes() if root is not None and root.is_file() else None
        if raw is None:
            raise OSError(
                "Realtime-Venus requires its pinned config.yaml beside the model directory. "
                "Download the checkpoint subdirectory and root config.yaml together, then serve the local model path."
            )
        if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
            raise ValueError("Realtime-Venus config.yaml SHA-256 mismatch")
        document = yaml.safe_load(raw)
        if (
            not isinstance(document, dict)
            or document.get("schema_version") != 3
            or document.get("name") != "Realtime-Venus"
        ):
            raise ValueError("Invalid Realtime-Venus root configuration")
        models = document.get("models")
        entry = models.get("omni") if isinstance(models, dict) else None
        full = entry.get("hf_config") if isinstance(entry, dict) else None
        if not isinstance(entry, dict) or not isinstance(full, dict):
            raise ValueError("Missing Realtime-Venus Omni configuration")
        if (
            entry.get("path") != "Realtime-Venus-Omni"
            or full.get("model_type") != cls.model_type
            or config.get("model_type") != cls.model_type
            or "venus_config_ref" in full
            or full.get("architectures") != config.get("architectures")
        ):
            raise ValueError("Realtime-Venus root configuration model identity mismatch")
        for name in cls.sub_configs:
            if not isinstance(full.get(name), dict):
                raise ValueError(f"Missing Realtime-Venus {name}")

        def auto_map_identity(mapping):
            if not isinstance(mapping, dict) or not all(isinstance(v, str) for v in mapping.values()):
                raise ValueError("Invalid Realtime-Venus auto_map")
            return {k: v.rsplit("--", 1)[-1] for k, v in mapping.items()}

        if auto_map_identity(full.get("auto_map", {})) != auto_map_identity(config.get("auto_map", {})):
            raise ValueError("Realtime-Venus AutoClass identity mismatch")
        if "auto_map" in config:
            full["auto_map"] = config["auto_map"]
        # Preserve the checkpoint revision, rather than the separate YAML revision.
        if "_commit_hash" in config:
            full["_commit_hash"] = config["_commit_hash"]
        return full


AutoConfig.register(RealtimeVenusOmniConfig.model_type, RealtimeVenusOmniConfig)
