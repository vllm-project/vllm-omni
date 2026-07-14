# Copyright 2025 vLLM-Omni Team
"""Global storage for Kimi Audio tokens."""

import threading

# Thread-safe storage for audio tokens per request
_audio_tokens_storage = {}
_storage_lock = threading.Lock()


def store_audio_tokens(request_id: str, audio_tokens: list[int]) -> None:
    """Store audio tokens for a request."""
    with _storage_lock:
        _audio_tokens_storage[request_id] = audio_tokens


def get_audio_tokens(request_id: str) -> list[int] | None:
    """Get audio tokens for a request."""
    with _storage_lock:
        return _audio_tokens_storage.get(request_id)


def clear_audio_tokens(request_id: str) -> None:
    """Clear audio tokens for a request."""
    with _storage_lock:
        _audio_tokens_storage.pop(request_id, None)


def get_all_audio_tokens() -> dict[str, list[int]]:
    """Get all stored audio tokens (for debugging)."""
    with _storage_lock:
        return _audio_tokens_storage.copy()
