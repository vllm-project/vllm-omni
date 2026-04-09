"""Caching utilities for voice and speaker extraction artifacts.

Keyed by voice name + extraction mode (e.g. ``"alice:icl"``).
Only named voices are cached; inline ``ref_audio`` without a voice
name is not cached.

Usage::

    key = VoiceEmbeddingCache.make_cache_key("alice", xvec_only=False)
    cached = cache.get(key)
    if cached is None:
        # ... extract ...
        cache.put(key, {"artifact": result})
"""

import asyncio
import os
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_MAX_ENTRIES = 128
_PROCESSING_TIMEOUT_S = 300  # 5 minutes


def is_path_within_directory(file_path: Path, directory: Path) -> bool:
    """Return whether *file_path* resolves under *directory*."""
    try:
        file_path_resolved = file_path.resolve()
        directory_resolved = directory.resolve()
        return directory_resolved in file_path_resolved.parents or directory_resolved == file_path_resolved
    except Exception:
        return False


class VoiceEmbeddingCache:
    """LRU cache for voice extraction outputs.

    Each entry stores a ``dict[str, Any]`` whose contents are model-specific.
    Thread-safe via a lightweight ``threading.Lock``.
    """

    def __init__(self, max_entries: int | None = None):
        if max_entries is None:
            max_entries = int(os.environ.get("VOICE_CACHE_MAX_ENTRIES", _DEFAULT_MAX_ENTRIES))
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        logger.info("Voice embedding cache initialized (max_entries=%d)", max_entries)

    @staticmethod
    def make_cache_key(voice_name: str, xvec_only: bool, created_at: float = 0.0) -> str:
        """Build a cache key from a voice name, upload timestamp, and extraction mode.

        Args:
            voice_name: The speaker/voice name (case-insensitive, lowered
                by the caller).
            xvec_only: True for speaker-embedding-only mode, False for
                ICL mode (speaker embedding + ref_code).
            created_at: Upload timestamp from metadata. Prevents stale cache
                hits after a voice is deleted and re-uploaded with the same
                name but different audio.
        """
        mode = "xvec" if xvec_only else "icl"
        return f"{voice_name}:{created_at:.6f}:{mode}"

    def get(self, key: str) -> dict[str, Any] | None:
        """Return cached artifacts or ``None`` on miss.  Promotes to MRU on hit."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                logger.debug("Voice cache HIT (key=%s, hits=%d)", key, self._hits)
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, artifacts: dict[str, Any]) -> None:
        """Store *artifacts* under *key*, evicting the LRU entry if full."""
        with self._lock:
            self._cache[key] = artifacts
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_entries:
                evicted_key, _ = self._cache.popitem(last=False)
                logger.debug("Voice cache EVICT (key=%s)", evicted_key)

    def stats(self) -> dict[str, int]:
        """Return cache statistics."""
        with self._lock:
            return {
                "entries": len(self._cache),
                "max_entries": self._max_entries,
                "hits": self._hits,
                "misses": self._misses,
            }


class VoiceCacheManager:
    """Shared manager for persisted speaker prompt caches.

    Handles the in-memory prompt memoization, speaker cache state machine,
    safetensors persistence, and atomic file replacement for uploaded
    speaker prompts. Serving layers provide the model-specific extraction
    callback that turns a source audio file into a plain-Python prompt
    payload.
    """

    def __init__(self, cache_dir: str | Path):
        self._cache_dir = Path(cache_dir)
        self._speaker_prompt_cache: dict[str, dict[str, Any]] = {}
        self._speaker_locks: dict[str, asyncio.Lock] = {}

    def _get_speaker_lock(self, speaker_key: str) -> asyncio.Lock:
        lock = self._speaker_locks.get(speaker_key)
        if lock is None:
            lock = asyncio.Lock()
            self._speaker_locks[speaker_key] = lock
        return lock

    @staticmethod
    def cache_file_path_for_speaker(speaker_info: dict[str, Any]) -> Path | None:
        cache_file = speaker_info.get("cache_file")
        if isinstance(cache_file, str) and cache_file:
            return Path(cache_file)
        file_path = speaker_info.get("file_path")
        if isinstance(file_path, str) and file_path:
            return Path(file_path).with_suffix(".safetensors")
        return None

    def invalidate_speaker_prompt_cache(self, speaker_key: str) -> None:
        self._speaker_prompt_cache.pop(speaker_key, None)

    def load_cached_speaker_prompt(self, speaker_name: str, speaker_info: dict[str, Any]) -> dict[str, Any] | None:
        speaker_key = speaker_name.lower()
        if speaker_info.get("cache_status") != "ready":
            return None

        cached = self._speaker_prompt_cache.get(speaker_key)
        if cached is not None:
            return cached

        cache_file_path = self.cache_file_path_for_speaker(speaker_info)
        if cache_file_path is None:
            return None
        if not is_path_within_directory(cache_file_path, self._cache_dir):
            logger.error("Illegal cache path outside voice samples directory: %s", cache_file_path)
            return None
        if not cache_file_path.is_file() or cache_file_path.suffix != ".safetensors":
            return None

        try:
            from safetensors import safe_open

            with safe_open(cache_file_path, framework="pt", device="cpu") as f:
                metadata = f.metadata()
                ref_spk_embedding = f.get_tensor("item_0_ref_spk_embedding")
                has_ref_code = bool(f.get_tensor("item_0_has_ref_code").item())
                ref_code = f.get_tensor("item_0_ref_code") if has_ref_code else None
                payload = {
                    "ref_spk_embedding": ref_spk_embedding.tolist(),
                    "ref_code": ref_code.tolist() if ref_code is not None else None,
                    "x_vector_only_mode": bool(f.get_tensor("item_0_x_vector_only_mode").item()),
                    "icl_mode": bool(f.get_tensor("item_0_icl_mode").item()),
                    "ref_text": metadata.get("item_0_ref_text"),
                }
                self._speaker_prompt_cache[speaker_key] = payload
                return payload
        except Exception as e:
            self.invalidate_speaker_prompt_cache(speaker_key)
            logger.warning("Failed to load speaker cache for %s: %s", speaker_name, e)
            return None

    def save_speaker_cache(
        self,
        speaker_key: str,
        speaker_info: dict[str, Any],
        audio_file_path: Path,
        payload: dict[str, Any],
    ) -> bool:
        try:
            import torch
            from safetensors.torch import save_file
        except ImportError:
            raise ValueError("safetensors is required for speaker cache generation")

        cache_file_path = audio_file_path.with_suffix(".safetensors")
        if not is_path_within_directory(cache_file_path, self._cache_dir):
            raise ValueError("Illegal cache path outside voice samples directory")

        tensors: dict[str, torch.Tensor] = {
            "__len__": torch.tensor(1, dtype=torch.int64),
            "item_0_ref_spk_embedding": torch.tensor(payload["ref_spk_embedding"], dtype=torch.float32),
            "item_0_has_ref_code": torch.tensor(int(payload["ref_code"] is not None), dtype=torch.int8),
            "item_0_x_vector_only_mode": torch.tensor(int(payload["x_vector_only_mode"]), dtype=torch.int8),
            "item_0_icl_mode": torch.tensor(int(payload["icl_mode"]), dtype=torch.int8),
        }
        if payload["ref_code"] is not None:
            tensors["item_0_ref_code"] = torch.tensor(payload["ref_code"], dtype=torch.long)

        metadata: dict[str, str] = {}
        if payload.get("ref_text") is not None:
            metadata["item_0_ref_text"] = str(payload["ref_text"])

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=cache_file_path.parent,
                prefix=f"{cache_file_path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                temp_path = Path(tmp.name)

            save_file(tensors, str(temp_path), metadata=metadata)
            os.replace(temp_path, cache_file_path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)

        speaker_info["cache_status"] = "ready"
        speaker_info["cache_file"] = str(cache_file_path)
        speaker_info["cache_generated_at"] = time.time()
        self._speaker_prompt_cache[speaker_key] = {
            "ref_spk_embedding": payload["ref_spk_embedding"],
            "ref_code": payload["ref_code"],
            "x_vector_only_mode": payload["x_vector_only_mode"],
            "icl_mode": payload["icl_mode"],
            "ref_text": payload.get("ref_text"),
        }
        return True

    async def create_speaker_cache(
        self,
        speaker_name: str,
        speaker_info: dict[str, Any],
        build_speaker_prompt,
        force: bool = False,
    ) -> dict[str, Any]:
        speaker_key = speaker_name.lower()
        async with self._get_speaker_lock(speaker_key):
            current_status = speaker_info.get("cache_status")

            if current_status == "processing" and not force:
                started_at = speaker_info.get("cache_generated_at")
                try:
                    started_at = float(started_at) if started_at is not None else None
                except (TypeError, ValueError):
                    started_at = None
                    logger.warning("Invalid cache_generated_at for speaker %s, treating as stale", speaker_name)
                if started_at is not None and (time.time() - started_at) < _PROCESSING_TIMEOUT_S:
                    return {
                        "cache_status": "processing",
                        "message": "Cache generation in progress",
                    }
                logger.warning("Processing state for speaker %s timed out or stale, allowing rebuild", speaker_name)

            if current_status == "ready" and not force:
                cached = self.load_cached_speaker_prompt(speaker_name, speaker_info)
                if cached is not None:
                    return {
                        "cache_status": "ready",
                        "message": "Cache already exists and is valid",
                    }
                logger.warning("Cache for speaker %s is invalid, rebuilding", speaker_name)

            previous_status = current_status or "pending"
            previous_cache_generated_at = speaker_info.get("cache_generated_at")
            self.invalidate_speaker_prompt_cache(speaker_key)
            speaker_info["cache_status"] = "processing"
            speaker_info["cache_generated_at"] = time.time()

            save_attempted = False
            try:
                file_path_str = speaker_info.get("file_path")
                if not file_path_str:
                    raise ValueError(
                        f"Metadata for voice '{speaker_name}' has no file_path. "
                        f"Delete this voice via DELETE /v1/audio/voices/{speaker_name} and re-upload."
                    )
                audio_file_path = Path(file_path_str)
                if not audio_file_path.is_file():
                    raise ValueError(
                        f"Audio file for voice '{speaker_name}' is missing from disk. "
                        f"Delete this voice via DELETE /v1/audio/voices/{speaker_name} "
                        "then re-upload via POST /v1/audio/voices before generating cache."
                    )

                payload = await build_speaker_prompt(audio_file_path, speaker_info)

                save_attempted = True
                if not self.save_speaker_cache(speaker_key, speaker_info, audio_file_path, payload):
                    raise ValueError(f"Failed to save voice cache for '{speaker_name}'")
            except Exception:
                if save_attempted:
                    speaker_info["cache_status"] = "failed"
                else:
                    speaker_info["cache_status"] = previous_status
                    speaker_info["cache_generated_at"] = previous_cache_generated_at
                raise

            return {"cache_status": "ready"}
