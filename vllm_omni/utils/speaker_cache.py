"""Process-wide thread-safe LRU cache for speaker extraction artifacts.

Keyed by ``(model_type, speaker_name, created_at)`` so each upload generation
has its own slot. Access via :func:`get_speaker_cache`.
"""

from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_MAX_BYTES = 512 * 1024**2  # 512 MiB
_CUSTOM_VOICE_MANIFEST = "custom_voice_manifest.json"
_CUSTOM_VOICE_SCHEMA_VERSION = 1

_SINGLETON: SpeakerEmbeddingCache | None = None
_SINGLETON_LOCK = threading.Lock()


def _estimate_tensor_bytes(obj: object) -> int:
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(_estimate_tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_estimate_tensor_bytes(item) for item in obj)
    return 0


def _as_voice_map(voices: object) -> dict[str, dict[str, Any]]:
    if not isinstance(voices, dict):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for name, info in voices.items():
        entry = dict(info) if isinstance(info, dict) else {}
        entry.setdefault("name", str(name))
        result[str(name)] = entry
    return result


def _load_custom_voice_manifest(custom_voice_dir: str | os.PathLike[str] | None) -> dict[str, Any] | None:
    if not custom_voice_dir:
        return None
    manifest_path = Path(custom_voice_dir).expanduser() / _CUSTOM_VOICE_MANIFEST
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load custom voice manifest %s: %s", manifest_path, exc)
        return None


def iter_custom_voice_profiles(
    custom_voice_dir: str | os.PathLike[str] | None,
    *,
    expected_model_type: str | None = None,
) -> list[dict[str, Any]]:
    manifest = _load_custom_voice_manifest(custom_voice_dir)
    if manifest is None:
        return []

    manifest_model_type = manifest.get("model_type")
    if expected_model_type and manifest_model_type and manifest_model_type != expected_model_type:
        logger.warning(
            "Skipping custom voice manifest for model_type=%s while loading %s",
            manifest_model_type,
            expected_model_type,
        )
        return []

    root = Path(custom_voice_dir).expanduser().resolve()  # type: ignore[arg-type]
    profiles: list[dict[str, Any]] = []
    for raw_name, raw_info in _as_voice_map(manifest.get("voices", {})).items():
        name = str(raw_info.get("name") or raw_name).strip()
        if not name:
            continue
        lower = name.lower()
        info = dict(raw_info)
        for stale_key in ("model_revision", "audio_codec_version", "audio_codec_config_hash"):
            info.pop(stale_key, None)
        filename = info.get("file") or info.get("filename") or f"{lower}.safetensors"
        file_path = (root / str(filename)).resolve()
        if not file_path.is_relative_to(root):
            logger.warning("Skipping custom voice %s: file path escapes custom_voice_dir", name)
            continue
        info.update(
            {
                "name": name,
                "voice_name_lower": lower,
                "file": str(filename),
                "file_path": str(file_path),
                "custom_voice_dir": str(root),
                "model_type": manifest_model_type or expected_model_type,
                "schema_version": manifest.get("schema_version", _CUSTOM_VOICE_SCHEMA_VERSION),
            }
        )
        profiles.append(info)
    return profiles


def _load_profile_tensors(profile: dict[str, Any]) -> dict[str, torch.Tensor] | None:
    try:
        from safetensors.torch import load_file
    except ImportError:
        logger.warning("safetensors is required to load custom voice profiles")
        return None

    file_path = profile.get("file_path")
    if not file_path:
        return None
    try:
        return load_file(str(file_path))
    except Exception as exc:
        logger.warning("Failed to load custom voice profile %s: %s", file_path, exc)
        return None


def _first_dim(tensor: torch.Tensor | None) -> int:
    if tensor is None or tensor.ndim == 0:
        return 0
    return int(tensor.shape[0])


def _validate_qwen3_tts_profile(
    profile: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    *,
    expected_embedding_dim: int | None,
) -> str | None:
    # Successful validation also normalizes/augments `profile` for serving metadata.
    speaker_embedding = tensors.get("speaker_embedding")
    if not isinstance(speaker_embedding, torch.Tensor):
        return "missing required tensor `speaker_embedding`"

    embedding_dim = int(speaker_embedding.reshape(-1).numel())
    if expected_embedding_dim and embedding_dim != expected_embedding_dim:
        return f"speaker_embedding dim={embedding_dim}, expected={expected_embedding_dim}"

    mode = str(profile.get("mode") or "xvec").lower()
    if mode not in ("xvec", "icl"):
        return f"invalid mode={mode!r}; expected 'xvec' or 'icl'"
    profile["mode"] = mode
    profile["embedding_dim"] = embedding_dim

    if mode == "icl":
        ref_code = tensors.get("ref_code")
        if not isinstance(ref_code, torch.Tensor):
            return "mode='icl' requires tensor `ref_code`"
        ref_code_length = _first_dim(ref_code)
        if ref_code_length <= 0:
            return "mode='icl' requires non-empty tensor `ref_code`"
        profile["ref_code_length"] = ref_code_length

    return None


def _validate_voxcpm2_profile(profile: dict[str, Any], tensors: dict[str, torch.Tensor]) -> str | None:
    # Successful validation also normalizes/augments `profile` for serving metadata.
    ref_audio_feat = tensors.get("ref_audio_feat")
    audio_feat = tensors.get("audio_feat")
    has_ref_audio_feat = isinstance(ref_audio_feat, torch.Tensor)
    has_audio_feat = isinstance(audio_feat, torch.Tensor)
    if not has_ref_audio_feat and not has_audio_feat:
        return "missing required tensor `ref_audio_feat` or `audio_feat`"

    mode = str(profile.get("mode") or "").lower()
    if not mode:
        if has_ref_audio_feat and has_audio_feat:
            mode = "ref_continuation"
        elif has_ref_audio_feat:
            mode = "reference"
        else:
            mode = "continuation"
    if mode not in ("reference", "continuation", "ref_continuation"):
        return f"invalid mode={mode!r}; expected 'reference', 'continuation', or 'ref_continuation'"
    if mode in ("reference", "ref_continuation") and not has_ref_audio_feat:
        return f"mode={mode!r} requires tensor `ref_audio_feat`"
    if mode in ("continuation", "ref_continuation") and not has_audio_feat:
        return f"mode={mode!r} requires tensor `audio_feat`"

    ref_audio_feat_len = _first_dim(ref_audio_feat)
    audio_feat_len = _first_dim(audio_feat)
    if mode in ("reference", "ref_continuation") and ref_audio_feat_len <= 0:
        return f"mode={mode!r} requires non-empty tensor `ref_audio_feat`"
    if mode in ("continuation", "ref_continuation") and audio_feat_len <= 0:
        return f"mode={mode!r} requires non-empty tensor `audio_feat`"

    profile["mode"] = mode
    if has_ref_audio_feat:
        profile["ref_audio_feat_len"] = ref_audio_feat_len
    if has_audio_feat:
        profile["audio_feat_len"] = audio_feat_len
    return None


def load_validated_profile_tensors(
    profile: dict[str, Any],
    *,
    expected_model_type: str,
    qwen3_embedding_dim: int | None = None,
) -> dict[str, torch.Tensor] | None:
    tensors = _load_profile_tensors(profile)
    error = None
    if tensors is None:
        error = "safetensors file could not be loaded"
    elif str(profile.get("model_type") or expected_model_type or "") != expected_model_type:
        error = f"model_type={profile.get('model_type')!r}, expected={expected_model_type!r}"
    elif expected_model_type == "qwen3_tts":
        error = _validate_qwen3_tts_profile(profile, tensors, expected_embedding_dim=qwen3_embedding_dim)
    elif expected_model_type == "voxcpm2":
        error = _validate_voxcpm2_profile(profile, tensors)
    else:
        error = f"unsupported custom voice model_type={expected_model_type!r}"

    if error is not None:
        logger.warning("Skipping custom %s voice %s: %s", expected_model_type, profile.get("name"), error)
        return None
    return tensors


class SpeakerEmbeddingCache:
    """Thread-safe in-memory LRU cache for speaker extraction artifacts."""

    def __init__(self, *, max_bytes: int = _MAX_BYTES):
        self._cache: OrderedDict[tuple[str, str, int], dict[str, Any]] = OrderedDict()
        self._sizes: dict[tuple[str, str, int], int] = {}
        self._total_bytes = 0
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._max_bytes = max_bytes
        logger.info("Speaker cache ready (max_bytes=%d)", self._max_bytes)

    @staticmethod
    def make_cache_key(speaker_name: str, model_type: str, created_at: int = 0) -> tuple[str, str, int]:
        """Build a cache key. ``created_at=0`` for built-in speakers (no upload).

        Names are normalized (stripped + lowercased) so delete/clear paths that
        normalize to lowercase match entries put with mixed-case names.
        """
        if not speaker_name or not speaker_name.strip():
            raise ValueError("speaker_name is required")
        if not model_type:
            raise ValueError("model_type is required")
        return (model_type, speaker_name.strip().lower(), int(created_at))

    def get(self, key: tuple[str, str, int]) -> dict[str, Any] | None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: tuple[str, str, int], artifacts: dict[str, Any]) -> None:
        with self._lock:
            self._insert_locked(key, artifacts)

    def _insert_locked(self, key: tuple[str, str, int], artifacts: dict[str, Any]) -> None:
        size = _estimate_tensor_bytes(artifacts)
        if size > self._max_bytes:
            logger.warning("Speaker cache skip: entry %s size=%dB exceeds max_bytes=%dB", key, size, self._max_bytes)
            return
        if key in self._cache:
            self._total_bytes -= self._sizes.pop(key, 0)
            del self._cache[key]
        self._cache[key] = artifacts
        self._sizes[key] = size
        self._total_bytes += size
        self._cache.move_to_end(key)
        while self._cache and self._total_bytes > self._max_bytes:
            evict_key, _ = self._cache.popitem(last=False)
            self._total_bytes -= self._sizes.pop(evict_key, 0)
            logger.debug("Speaker cache EVICT: key=%s", evict_key)

    def clear(self, speaker_name: str | None = None) -> int:
        """Remove entries. With a name, drops matches across model types and generations."""
        with self._lock:
            if speaker_name is None:
                removed = len(self._cache)
                self._cache.clear()
                self._sizes.clear()
                self._total_bytes = 0
                self._hits = 0
                self._misses = 0
                return removed

            if not speaker_name or not speaker_name.strip():
                raise ValueError("speaker_name cannot be an empty string")
            normalized = speaker_name.strip().lower()
            removed = 0
            for k in list(self._cache.keys()):
                if isinstance(k, tuple) and len(k) >= 2 and k[1] == normalized:
                    self._total_bytes -= self._sizes.pop(k, 0)
                    del self._cache[k]
                    removed += 1
            return removed

    def memory_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._cache),
                "memory_bytes": self._total_bytes,
                "max_bytes": self._max_bytes,
                "memory_mb": round(self._total_bytes / (1024 * 1024), 2),
                "hits": self._hits,
                "misses": self._misses,
            }


def get_speaker_cache() -> SpeakerEmbeddingCache:
    """Return the process-wide speaker cache singleton."""
    global _SINGLETON
    if _SINGLETON is None:
        with _SINGLETON_LOCK:
            if _SINGLETON is None:
                _SINGLETON = SpeakerEmbeddingCache()
    return _SINGLETON
