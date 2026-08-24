# SPDX-License-Identifier: Apache-2.0
"""Shared loading mechanisms for TTS model capability metadata."""

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from transformers.utils.hub import cached_file
from vllm.logger import init_logger

from vllm_omni.utils.speaker_cache import iter_custom_voice_profiles, load_validated_profile_tensors

logger = init_logger(__name__)


def load_supported_speakers(engine_client: Any, config: Any = None) -> set[str]:
    """Extract supported speaker names from a model configuration."""
    try:
        if config is None:
            config = getattr(engine_client.model_config.hf_config, "talker_config", None)
            if config is None:
                return set()

        for attr_name in ("spk_id", "speaker_id"):
            speakers = config.get(attr_name) if isinstance(config, dict) else getattr(config, attr_name, None)
            if speakers and isinstance(speakers, dict):
                return {speaker.lower() for speaker in speakers}

        logger.warning("No speakers found in config (checked spk_id and speaker_id)")
    except Exception as exc:
        logger.warning("Could not load speakers from model config: %s", exc)
    return set()


def load_codec_frame_rate(engine_client: Any) -> float | None:
    """Load codec frame rate from speech-tokenizer or HF configuration."""
    try:
        model_path = engine_client.model_config.model
        config_path = os.path.join(model_path, "speech_tokenizer", "config.json")
        if not os.path.exists(config_path):
            config_path = cached_file(model_path, "speech_tokenizer/config.json")
        if config_path is not None and os.path.exists(config_path):
            with open(config_path) as file:
                config = json.load(file)
            output_sr = config.get("output_sample_rate")
            downsample = config.get("encode_downsample_rate")
            if output_sr and downsample and downsample > 0:
                rate = float(output_sr) / float(downsample)
                logger.info(
                    "Loaded codec frame rate: %.1f Hz (output_sample_rate=%s, encode_downsample_rate=%s)",
                    rate,
                    output_sr,
                    downsample,
                )
                return rate
    except Exception as exc:
        logger.warning("Failed to load codec frame rate from speech tokenizer config: %s", exc)

    try:
        rate = getattr(engine_client.model_config.hf_config, "codec_frame_rate_hz", None)
        if rate is not None:
            logger.info("Using codec frame rate from hf_config: %s Hz", rate)
            return float(rate)
    except Exception:
        pass
    return None


def load_precomputed_speakers(
    engine_client: Any,
    *,
    expected_model_type: str,
    validate_profile: Callable[[dict[str, Any], dict[str, torch.Tensor]], str | None],
) -> dict[str, dict[str, Any]]:
    """Load and validate precomputed profiles from ``custom_voice_dir``."""
    try:
        custom_voice_dir = getattr(engine_client.model_config.hf_config, "custom_voice_dir", None)
    except AttributeError:
        return {}
    if isinstance(custom_voice_dir, os.PathLike):
        custom_voice_dir = os.fspath(custom_voice_dir)
    if not isinstance(custom_voice_dir, str) or not custom_voice_dir:
        return {}

    profiles: dict[str, dict[str, Any]] = {}
    for profile in iter_custom_voice_profiles(custom_voice_dir, expected_model_type=expected_model_type):
        tensors = load_validated_profile_tensors(
            profile,
            expected_model_type=expected_model_type,
            validate_profile=validate_profile,
        )
        if tensors is not None:
            profiles[profile["voice_name_lower"]] = profile
    if profiles:
        logger.info(
            "Loaded %d precomputed %s voice profile(s) from %s",
            len(profiles),
            expected_model_type,
            Path(custom_voice_dir).expanduser(),
        )
    return profiles
