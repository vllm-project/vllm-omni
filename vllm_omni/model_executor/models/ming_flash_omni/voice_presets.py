# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Copyright (c) Ant Group. All rights reserved.
# Adapted from:
# https://github.com/inclusionAI/Ming/blob/e58533db227031990c5a6864dcf5f08fb53ed0d2/modeling_bailing_talker.py

from __future__ import annotations

import json
import os
from functools import cached_property
from typing import TYPE_CHECKING, Any

import soundfile as sf
import torch
from transformers.utils.hub import cached_file
from vllm.logger import init_logger

from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

from .spk_embedding import SpkembExtractor
from .talker_module import resample

if TYPE_CHECKING:
    from .audio_vae import AudioVAE
    from .talker_module import Aggregator

logger = init_logger(__name__)


class InvalidPromptWavError(ValueError):
    """Prompt wav failed local validation and can be skipped in list mode."""


class VoicePresetRegistry:
    """Loader and registry for Ming voice presets."""

    def __init__(
        self,
        *,
        talker_dir: str,
        model_path: str,
        download_dir: str | None,
        audio_vae: AudioVAE | None,
        aggregator: Aggregator,
        spk_head: torch.nn.Module,
        patch_size: int,
    ) -> None:
        self._talker_dir = talker_dir
        self._model_path = model_path
        self._download_dir = download_dir
        self._audio_vae = audio_vae
        self._aggregator = aggregator
        self._spk_head = spk_head
        self._patch_size = patch_size

        self.registered: dict[str, dict[str, Any]] = {}

    def __contains__(self, voice_name: str) -> bool:
        return voice_name in self.registered

    def get(self, voice_name: str) -> dict[str, Any] | None:
        return self.registered.get(voice_name)

    def register(
        self,
        voice_name: str,
        prompt_wav_path: str | list[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Register a voice preset from one or more reference wav files.

        Args:
            voice_name: Key under which to store the preset.
            prompt_wav_path: Single wav path or a list (multi-clip mode skips
                invalid entries with a warning instead of raising).
            device: Target device for cached prompt latents / projected
                speaker embeddings.
            dtype: Target dtype for the projected speaker embedding head.
        """
        paths = self._normalize_paths(voice_name, prompt_wav_path)
        allow_partial = len(paths) > 1

        vae_sr = int(self._audio_vae.config.sample_rate) if self._audio_vae else 44100
        if self._audio_vae is None:
            logger.warning(
                "Voice preset '%s' being registered without AudioVAE features",
                voice_name,
            )

        speech_chunks: list[torch.Tensor] = []
        spk_emb_list: list[torch.Tensor] = []
        for wav_path in paths:
            try:
                speech_for_vae, raw_emb = self._load_single_wav(voice_name, wav_path, vae_sr)
            except (FileNotFoundError, InvalidPromptWavError) as e:
                if allow_partial:
                    logger.warning(
                        "Voice preset '%s': skipping invalid prompt wav %s: %s",
                        voice_name,
                        wav_path,
                        e,
                    )
                    continue
                raise
            speech_chunks.append(speech_for_vae)
            if raw_emb is not None:
                projected = self._spk_head(raw_emb.to(device=device, dtype=dtype))
                spk_emb_list.append(projected)

        if not speech_chunks:
            raise RuntimeError(f"Failed to register voice preset '{voice_name}': no valid prompt wavs remained")
        if not spk_emb_list and self._audio_vae is None:
            raise RuntimeError(
                f"Failed to register voice preset '{voice_name}': neither speaker "
                "embeddings nor AudioVAE prompt features are available"
            )

        prompt_wav_lat, prompt_wav_emb = self._build_wav_embeddings(
            voice_name, torch.cat(speech_chunks, dim=-1), device=device
        )

        if voice_name in self.registered:
            logger.warning("Voice preset '%s' is being overwritten", voice_name)
        self.registered[voice_name] = {
            "prompt_wav_lat": prompt_wav_lat,
            "prompt_wav_emb": prompt_wav_emb,
            "spk_emb": spk_emb_list,
        }
        logger.info("Registered voice preset '%s' from %s", voice_name, paths)

    def load_presets_from_manifest(self, *, device: torch.device, dtype: torch.dtype) -> None:
        """Resolve voice_name.json on disk or HF hub and register all entries.

        Each entry is registered onto the supplied device and dtype.
        """
        voice_json_path, base_dir = self._locate_manifest()
        if voice_json_path is None:
            logger.info("No voice_name.json found; voice presets unavailable")
            return

        with open(voice_json_path) as f:
            voice_dict = json.load(f)

        for name, info in voice_dict.items():
            wav_path = info.get("prompt_wav_path", "")
            prompt_text = info.get("prompt_text", "")
            if not wav_path:
                logger.warning("Voice preset '%s' has no prompt_wav_path, skipping", name)
                continue
            if not os.path.isabs(wav_path):
                wav_path = os.path.join(base_dir, wav_path)
            if not os.path.isfile(wav_path):
                logger.warning("Voice preset '%s': wav not found at %s, skipping", name, wav_path)
                continue
            try:
                self.register(name, wav_path, device=device, dtype=dtype)
                self.registered[name]["prompt_text"] = prompt_text
            except Exception as e:  # pragma: no cover — manifest is best-effort
                logger.warning("Failed to register voice preset '%s': %s", name, e)

    @cached_property
    def _spkemb_extractor(self) -> SpkembExtractor:
        """Lazily resolve the CAMPPlus ONNX extractor."""
        for candidate in (self._talker_dir, self._model_path):
            path = os.path.join(candidate, "campplus.onnx")
            if os.path.isfile(path):
                extractor = SpkembExtractor(path)
                logger.info("Initialized SpkembExtractor from %s", path)
                return extractor
        try:
            path = cached_file(self._model_path, "campplus.onnx", subfolder="talker")
        except Exception as e:
            raise RuntimeError("campplus.onnx not found. Expected at <model_path>/talker/campplus.onnx") from e
        extractor = SpkembExtractor(path)
        logger.info("Initialized SpkembExtractor from %s", path)
        return extractor

    @staticmethod
    def _normalize_paths(voice_name: str, prompt_wav_path: str | list[str]) -> list[str]:
        if not isinstance(voice_name, str) or not voice_name.strip():
            raise ValueError("voice_name must be a non-empty string")
        if isinstance(prompt_wav_path, str):
            paths = [prompt_wav_path]
        elif isinstance(prompt_wav_path, list):
            paths = list(prompt_wav_path)
        else:
            raise TypeError("prompt_wav_path must be a string path or a list of string paths")
        paths = [p.strip() for p in paths]
        if not paths or any(not p for p in paths):
            raise ValueError("Provided audio path is invalid")
        return paths

    def _load_single_wav(self, voice_name: str, wav_path: str, vae_sr: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(speech_for_vae, raw_spk_emb_or_none)``.

        Stays device-agnostic — both returned tensors live on CPU; the caller
        moves them to the target device when projecting / encoding.
        """
        if not os.path.isfile(wav_path):
            raise FileNotFoundError(f"prompt wav not found: {wav_path}")

        data, sample_rate = sf.read(wav_path, dtype="float32")
        speech_tmp = torch.from_numpy(data)
        if speech_tmp.ndim == 1:
            speech_tmp = speech_tmp.unsqueeze(0)
        elif speech_tmp.ndim == 2:
            num_channels = speech_tmp.shape[1]
            if num_channels > 1:
                logger.warning(
                    "Voice preset '%s': downmixing %d-channel audio at %s to mono",
                    voice_name,
                    num_channels,
                    wav_path,
                )
            speech_tmp = speech_tmp.mean(dim=1, keepdim=True).T
        else:
            raise InvalidPromptWavError(f"unsupported audio shape {tuple(speech_tmp.shape)} for {wav_path}")

        if not torch.isfinite(speech_tmp).all():
            raise InvalidPromptWavError(f"audio file contains NaN or Inf samples: {wav_path}")

        speech_for_vae = resample(speech_tmp, sample_rate, vae_sr)

        # Speaker embedding (16 kHz CAMPPlus). If the extractor fails to
        # resolve (missing ONNX model), skip embedding extraction rather than
        # blocking VAE-only registration.
        raw_emb: torch.Tensor | None = None
        try:
            extractor = self._spkemb_extractor
            speech_for_spk = resample(speech_tmp, sample_rate, 16000)
            raw_emb = extractor(speech_for_spk)
        except RuntimeError:
            raw_emb = None
        return speech_for_vae, raw_emb

    def _build_wav_embeddings(
        self,
        voice_name: str,
        speech: torch.Tensor,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self._audio_vae is None:
            return None, None

        patch_pt = self._audio_vae.encoder.hop_size * max(1, self._audio_vae.encoder.patch_size) * self._patch_size
        if speech.shape[-1] % patch_pt != 0:
            pad_len = (speech.shape[-1] + patch_pt - 1) // patch_pt * patch_pt
            pad_speech = torch.zeros((speech.shape[0], pad_len), dtype=speech.dtype, device=speech.device)
            pad_speech[:, -speech.shape[-1] :] = speech
            speech = pad_speech

        prompt_wav_lat, _ = self._audio_vae.encode_latent(
            speech.to(dtype=torch.bfloat16, device=device),
            torch.tensor([speech.size(1)], dtype=torch.long, device=device),
        )
        assert prompt_wav_lat.shape[1] % self._patch_size == 0, (
            f"AudioVAE latent length is incompatible with patch_size for voice preset '{voice_name}'"
        )
        prompt_wav_lat = prompt_wav_lat.reshape(-1, self._patch_size, prompt_wav_lat.shape[-1])
        prompt_wav_emb = self._aggregator(prompt_wav_lat)
        prompt_wav_lat = prompt_wav_lat.reshape(1, -1, prompt_wav_lat.shape[-1])
        prompt_wav_emb = prompt_wav_emb.reshape(1, -1, prompt_wav_emb.shape[-1])
        return prompt_wav_lat, prompt_wav_emb

    def _locate_manifest(self) -> tuple[str | None, str | None]:
        for candidate in (self._talker_dir, self._model_path):
            path = os.path.join(candidate, "data", "voice_name.json")
            if os.path.isfile(path):
                return path, candidate

        if not os.path.isdir(self._model_path):
            try:
                hf_root = download_weights_from_hf_specific(
                    self._model_path,
                    self._download_dir,
                    allow_patterns=["talker/data/**"],
                    require_all=True,
                )
                candidate = os.path.join(hf_root, "talker", "data", "voice_name.json")
                if os.path.isfile(candidate):
                    return candidate, os.path.join(hf_root, "talker")
            except Exception as e:  # pragma: no cover
                logger.info("Could not download voice presets from HF: %s", e)

        return None, None
