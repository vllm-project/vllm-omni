# SPDX-License-Identifier: Apache-2.0
"""Raon voice-clone cache and voices API helpers."""

from __future__ import annotations

import base64
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class RaonVoiceClonePromptItem:
    """One voice-clone prompt (embedding + mode flags + optional ref text)."""

    ref_spk_embedding: torch.Tensor  # (D,) ECAPA-TDNN speaker embedding
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: str | None = None


# ------------------------------------------------------------------
# Voices API helpers (upload / TTS-time lookup / delete)
# ------------------------------------------------------------------

_ecapa_backend: Any = None
_ecapa_lock = threading.Lock()


def _get_ecapa_backend() -> Any:
    """Lazy singleton for the ECAPA-TDNN speaker backend (CPU)."""
    global _ecapa_backend
    if _ecapa_backend is None:
        with _ecapa_lock:
            if _ecapa_backend is None:
                from vllm_omni.model_executor.models.raon.raon_speaker_encoder import (
                    EcapaSpeakerBackend,
                    _load_ecapa_from_hf,
                )

                ecapa = _load_ecapa_from_hf("speechbrain/spkrec-ecapa-voxceleb")
                _ecapa_backend = EcapaSpeakerBackend(ecapa)
    return _ecapa_backend


def _safetensors_path(audio_path: str) -> Path:
    return Path(audio_path).with_suffix(".safetensors")


def extract_and_cache_speaker_embedding(
    audio_file_path: str,
    *,
    ref_text: str | None = None,
) -> bool:
    """Extract ECAPA-TDNN embedding from audio and save as safetensors.

    Returns True on success, False on graceful failure.
    """
    file_path = Path(audio_file_path)
    if not file_path.exists():
        logger.warning("Audio file not found for embedding extraction: %s", audio_file_path)
        return False

    try:
        import soundfile

        audio_np, sr = soundfile.read(str(file_path), dtype="float32", always_2d=True)
    except Exception as e:
        logger.warning("Failed to read audio for embedding extraction: %s", e)
        return False

    audio_tensor = (
        torch.from_numpy(
            np.asarray(audio_np, dtype=np.float32),
        )
        .transpose(0, 1)
        .contiguous()
    )
    if audio_tensor.shape[0] > 1:
        audio_tensor = audio_tensor.mean(dim=0, keepdim=True)

    try:
        import torchaudio.functional

        backend = _get_ecapa_backend()
        target_sr = 16000
        if sr != target_sr:
            audio_16k = torchaudio.functional.resample(
                audio_tensor.float(),
                orig_freq=int(sr),
                new_freq=target_sr,
            )
        else:
            audio_16k = audio_tensor.float()

        with torch.inference_mode():
            raw_embedding = backend.encode_batch(audio_16k)  # [1, 1, 192]
        raw_embedding = raw_embedding.squeeze()  # (192,)
    except Exception as e:
        logger.warning("Speaker encoder unavailable; skipping embedding extraction: %s", e)
        return False

    x_vector_only = ref_text is None or not ref_text.strip()
    tensors: dict[str, torch.Tensor] = {
        "ref_spk_embedding": raw_embedding.detach().cpu(),
        "x_vector_only_mode": torch.tensor(int(x_vector_only), dtype=torch.int8),
    }
    metadata: dict[str, str] = {}
    if not x_vector_only and ref_text:
        metadata["ref_text"] = ref_text

    cache_path = _safetensors_path(audio_file_path)
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".safetensors", dir=str(cache_path.parent))
        os.close(tmp_fd)
        try:
            save_file(tensors, tmp_path, metadata=metadata)
            os.rename(tmp_path, str(cache_path))
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        logger.info("Cached speaker embedding: %s", cache_path)
        return True
    except Exception as e:
        logger.warning("Failed to save speaker embedding cache: %s", e)
        return False


def resolve_uploaded_voice(
    voice_name: str,
    uploaded_speakers: dict[str, dict],
) -> dict[str, Any] | None:
    """Look up an uploaded voice and return artifacts for ``additional_information``.

    Fast path: load cached embedding from safetensors.
    Slow path: return audio as data URL for model-level extraction.
    """
    voice_key = voice_name.lower()
    speaker_info = uploaded_speakers.get(voice_key)
    if speaker_info is None:
        return None

    file_path = speaker_info.get("file_path")
    if not file_path:
        return None

    cache_path = _safetensors_path(file_path)
    if cache_path.exists():
        try:
            with safe_open(str(cache_path), framework="pt", device="cpu") as f:
                embedding = f.get_tensor("ref_spk_embedding")
                meta = f.metadata()
            result: dict[str, Any] = {"cached_spk_embedding": [embedding]}
            ref_text = meta.get("ref_text") if meta else None
            if ref_text:
                result["ref_text"] = ref_text
            return result
        except Exception as e:
            logger.warning("Failed to load cached embedding for '%s': %s", voice_name, e)

    if not Path(file_path).exists():
        logger.warning("No cache or audio file for uploaded voice: %s", voice_name)
        return None

    try:
        with open(file_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        mime_type = speaker_info.get("mime_type", "audio/wav")
        data_url = f"data:{mime_type};base64,{audio_b64}"
        logger.info("Falling back to raw audio data URL for voice: %s", voice_name)
        return {"ref_audio": data_url}
    except Exception as e:
        logger.warning("Failed to read audio for uploaded voice %s: %s", voice_name, e)
        return None


def cleanup_voice_cache(file_path: str | None) -> None:
    """Remove the safetensors cache file associated with an audio upload."""
    if not file_path:
        return
    cache_path = _safetensors_path(file_path)
    if cache_path.exists():
        try:
            cache_path.unlink()
            logger.info("Deleted Raon voice cache: %s", cache_path)
        except Exception as e:
            logger.warning("Failed to delete Raon voice cache %s: %s", cache_path, e)
