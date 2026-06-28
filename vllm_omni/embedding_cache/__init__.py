# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-process content-addressed embedding cache (RFC #3427, P0)."""

from vllm_omni.embedding_cache.cache import EmbeddingCache
from vllm_omni.embedding_cache.hasher import hash_audio_features, hash_image_pixels, hash_video_pixels

__all__ = [
    "EmbeddingCache",
    "hash_audio_features",
    "hash_image_pixels",
    "hash_video_pixels",
]
