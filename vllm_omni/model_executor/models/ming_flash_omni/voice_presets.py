# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Copyright (c) Ant Group. All rights reserved.
# Adapted from:
# https://github.com/inclusionAI/Ming/blob/e58533db227031990c5a6864dcf5f08fb53ed0d2/modeling_bailing_talker.py

from __future__ import annotations

import json
import os
from functools import cached_property, lru_cache
from typing import TYPE_CHECKING, Any

import soundfile as sf
import torch
from transformers.utils.hub import cached_file
from vllm.logger import init_logger

from vllm_omni.model_executor.model_loader.weight_utils import resolve_model_to_local_path
from vllm_omni.model_executor.models.common.ming.spk_embedding import SpkembExtractor
from vllm_omni.transformers_utils.configs.ming_flash_omni import resolve_ming_talker_config

from .talker_module import (
    ming_prompt_wav_len,
    resample,
    resolve_audio_vae_config,
)

if TYPE_CHECKING:
    from vllm_omni.model_executor.models.common.ming.aggregator import Aggregator
    from vllm_omni.model_executor.models.common.ming.audio_vae import AudioVAE

logger = init_logger(__name__)


class InvalidPromptWavError(ValueError):
    """Prompt wav failed local validation and can be skipped in list mode."""


def _locate_voice_manifest(talker_dir: str, model_path: str, download_dir: str | None) -> tuple[str | None, str | None]:
    """Return ``(voice_name.json path, base dir)``, or ``(None, None)`` if absent."""
    for candidate in (talker_dir, model_path):
        path = os.path.join(candidate, "data", "voice_name.json")
        if os.path.isfile(path):
            return path, candidate

    try:
        hf_root = resolve_model_to_local_path(
            model_path,
            allow_download=True,
            allow_patterns=["talker/data/**"],
            cache_dir=download_dir,
        )
    except Exception as e:
        logger.info("Could not download voice presets from HF: %s", e)
        return None, None

    candidate = os.path.join(hf_root, "talker", "data", "voice_name.json")
    if os.path.isfile(candidate):
        return candidate, os.path.join(hf_root, "talker")

    return None, None


def _load_voice_manifest_entries(voice_json_path: str, base_dir: str | None) -> list[tuple[str, list[str], str]]:
    """Read voice_name.json into ``(name, absolute wav paths, prompt_text)`` rows.

    ``prompt_wav_path`` may be a single path or a list; multi-clip presets are
    concatenated by ``register``. Parsing here rather than in each consumer
    keeps the registry and the derived metadata over the same wav set.
    """
    with open(voice_json_path) as f:
        voice_dict = json.load(f)

    entries: list[tuple[str, list[str], str]] = []
    for name, info in voice_dict.items():
        raw = info.get("prompt_wav_path") or []
        if isinstance(raw, str):
            raw = [raw]
        # ``register`` strips paths, so strip here too or the two sides would
        # resolve a whitespace-padded manifest entry differently.
        paths = [p.strip() for p in raw if isinstance(p, str) and p.strip()]
        paths = [p if os.path.isabs(p) or not base_dir else os.path.join(base_dir, p) for p in paths]
        if not paths:
            logger.warning("Voice preset '%s' has no prompt_wav_path, skipping", name)
            continue
        entries.append((name, paths, info.get("prompt_text", "") or ""))
    return entries


def _resolve_prompt_wav_geometry(talker_dir: str, model_path: str) -> tuple[int, int, int, int] | None:
    """Return ``(vae_sample_rate, hop_size, vae_patch_size, patch_size)`` from config JSON.

    Uses the talker's own config resolvers so both processes read the same
    files; no weights are loaded. The talker prefers an in-memory talker config
    when the root config wraps one, which this cannot see — a divergence there
    is caught by ``_verify_derived_meta`` at init.
    """
    talker_config = resolve_ming_talker_config(None, talker_dir, model_path)
    if talker_config is None:
        logger.info("Could not resolve talker config for prompt-wav geometry")
        return None

    resolved = resolve_audio_vae_config(talker_config.audio_vae_path, talker_dir, model_path)
    if resolved is None:
        logger.info("Could not resolve AudioVAE config for prompt-wav geometry")
        return None
    vae_config, _ = resolved

    enc_kwargs = vae_config.enc_kwargs or {}
    return (
        int(vae_config.sample_rate),
        int(enc_kwargs.get("hop_size", enc_kwargs.get("input_dim", 320))),
        int(getattr(vae_config, "patch_size", 4)),
        int(talker_config.patch_size),
    )


def _resampled_sample_count(frames: int, orig_sr: int, target_sr: int) -> int:
    """Sample count after ``talker_module.resample``, which truncates rather than rounds."""
    if orig_sr == target_sr:
        return int(frames)
    return int(frames * (target_sr / orig_sr))


def _resolve_spkemb_path(talker_dir: str, model_path: str) -> str | None:
    """Resolve campplus.onnx locally or from the HF hub; ``None`` when absent.

    Shared by the extractor and the derived metadata: whether this resolves
    decides whether a preset carries speaker embeddings at all, and the two
    sides must agree or the reserved spk slots won't match the injected ones.
    """
    for candidate in (talker_dir, model_path):
        path = os.path.join(candidate, "campplus.onnx")
        if os.path.isfile(path):
            return path
    try:
        return cached_file(model_path, "campplus.onnx", subfolder="talker") or None
    except Exception:
        return None


@lru_cache(maxsize=8)
def resolve_voice_preset_meta(model_path: str | None, download_dir: str | None = None) -> dict[str, dict[str, Any]]:
    """Derive ``{voice_name: {prompt_text, prompt_wav_len, spk_emb_count}}`` from disk.

    Lets the stage input processor size prompt-KV slots from the manifest and
    wav headers alone, with no state passed from the talker; the talker checks
    its registrations against it in ``_verify_derived_meta``. Empty when no
    manifest or config resolves. Cached and shared — treat as read-only.
    """
    if not model_path:
        return {}
    talker_dir = os.path.join(model_path, "talker") if os.path.isdir(os.path.join(model_path, "talker")) else model_path

    manifest_path, base_dir = _locate_voice_manifest(talker_dir, model_path, download_dir)
    if manifest_path is None:
        return {}
    # Locating the manifest resolves a local snapshot directory even when
    # model_path is a bare repo id. Probe configs there rather than by repo id:
    # every from_pretrained candidate on a repo id is an HF hub round-trip, and
    # the talker worker pays them again at init (~3 min on a mirrored hub).
    local_talker_dir = base_dir or talker_dir
    geometry = _resolve_prompt_wav_geometry(local_talker_dir, model_path)
    if geometry is None:
        return {}
    vae_sr, hop_size, vae_patch_size, patch_size = geometry
    spkemb_available = _resolve_spkemb_path(local_talker_dir, model_path) is not None

    meta: dict[str, dict[str, Any]] = {}
    for name, paths, prompt_text in _load_voice_manifest_entries(manifest_path, base_dir):
        # ``register`` concatenates the resampled clips and pads the whole once,
        # so the frame count comes from the summed per-clip sample counts, and
        # each readable clip contributes one speaker embedding. Clips whose
        # header cannot be read are skipped on both sides.
        num_samples = 0
        num_clips = 0
        for wav_path in paths:
            try:
                header = sf.info(wav_path)
            except Exception as e:
                logger.warning("Voice preset '%s': cannot read wav header at %s: %s", name, wav_path, e)
                continue
            num_samples += _resampled_sample_count(header.frames, header.samplerate, vae_sr)
            num_clips += 1
        if not num_clips:
            continue
        meta[name] = {
            "prompt_text": prompt_text,
            "prompt_wav_len": ming_prompt_wav_len(
                num_samples,
                hop_size=hop_size,
                vae_patch_size=vae_patch_size,
                patch_size=patch_size,
            ),
            "spk_emb_count": num_clips if spkemb_available else 0,
        }
    return meta


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

    @torch.no_grad()
    def register(
        self,
        voice_name: str,
        prompt_wav_path: str | list[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Register a voice preset from one or more reference wav files.

        Runs under ``torch.no_grad()``: the spk-head / AudioVAE / aggregator
        forwards here touch model weights that vLLM loads as inference-mode
        tensors, and tracking them through autograd raises "Inference tensors
        cannot be saved for backward" on newer torch/vLLM.

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
        voice_json_path, base_dir = _locate_voice_manifest(self._talker_dir, self._model_path, self._download_dir)
        if voice_json_path is None:
            logger.info("No voice_name.json found; voice presets unavailable")
            return

        for name, paths, prompt_text in _load_voice_manifest_entries(voice_json_path, base_dir):
            try:
                self.register(name, paths, device=device, dtype=dtype)
                self.registered[name]["prompt_text"] = prompt_text
            except Exception as e:  # pragma: no cover — manifest is best-effort
                logger.warning("Failed to register voice preset '%s': %s", name, e)

        self._verify_derived_meta()

    def _verify_derived_meta(self) -> None:
        """Cross-check registrations against ``resolve_voice_preset_meta``.

        Every voice that sizes prompt-KV slots must be registered with exactly
        the derived geometry, so a mismatch fails here instead of resurfacing as
        a prefill length error on the first request naming that voice.
        """
        if self._audio_vae is None:
            logger.warning("AudioVAE unavailable; skipping voice preset geometry cross-check")
            return

        for name, expected in resolve_voice_preset_meta(self._model_path, self._download_dir).items():
            info = self.registered.get(name)
            if info is None:
                raise RuntimeError(
                    f"Voice preset '{name}' is listed in voice_name.json but failed to register; "
                    "the stage input processor would still reserve prompt-KV slots for it"
                )
            emb = info.get("prompt_wav_emb")
            actual_len = int(emb.size(1)) if emb is not None else 0
            spk = info.get("spk_emb")
            actual_spk = len(spk) if isinstance(spk, list) else (0 if spk is None else 1)
            if actual_len != expected["prompt_wav_len"] or actual_spk != expected["spk_emb_count"]:
                raise RuntimeError(
                    f"Voice preset '{name}' geometry mismatch: registered "
                    f"(prompt_wav_len={actual_len}, spk_emb_count={actual_spk}), but the "
                    f"config-derived values used to size prompt-KV slots are "
                    f"(prompt_wav_len={expected['prompt_wav_len']}, "
                    f"spk_emb_count={expected['spk_emb_count']})"
                )

    @cached_property
    def _spkemb_extractor(self) -> SpkembExtractor:
        """Lazily resolve the CAMPPlus ONNX extractor."""
        path = _resolve_spkemb_path(self._talker_dir, self._model_path)
        if path is None:
            raise RuntimeError("campplus.onnx not found. Expected at <model_path>/talker/campplus.onnx")
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

        expected_len = ming_prompt_wav_len(
            speech.shape[-1],
            hop_size=self._audio_vae.encoder.hop_size,
            vae_patch_size=self._audio_vae.encoder.patch_size,
            patch_size=self._patch_size,
        )

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
        assert prompt_wav_emb.size(1) == expected_len, (
            f"voice preset '{voice_name}': AudioVAE produced {prompt_wav_emb.size(1)} prompt-wav "
            f"frames but ming_prompt_wav_len derived {expected_len}"
        )
        return prompt_wav_lat, prompt_wav_emb
