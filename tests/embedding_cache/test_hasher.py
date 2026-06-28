"""Tests for embedding_cache.hasher."""
import torch

from vllm_omni.embedding_cache.hasher import (
    hash_audio_features,
    hash_image_pixels,
    hash_video_pixels,
)


def _rand(*shape) -> torch.Tensor:
    return torch.randn(*shape)


def test_same_tensor_same_hash():
    t = _rand(4, 80, 3000)
    assert hash_audio_features(t) == hash_audio_features(t)


def test_different_tensor_different_hash():
    a = _rand(4, 80, 3000)
    b = _rand(4, 80, 3000)
    assert hash_audio_features(a) != hash_audio_features(b)


def test_modality_prefixes_differ():
    t = _rand(16, 3, 224, 224)
    audio_key = hash_audio_features(t)
    image_key = hash_image_pixels(t)
    video_key = hash_video_pixels(t)
    assert audio_key.startswith("a:")
    assert image_key.startswith("i:")
    assert video_key.startswith("v:")
    # same bytes, different prefix → different keys
    assert audio_key != image_key
    assert image_key != video_key


def test_copy_same_hash():
    t = _rand(8, 128)
    tc = t.clone()
    assert hash_image_pixels(t) == hash_image_pixels(tc)


def test_key_length():
    t = _rand(2, 2)
    key = hash_audio_features(t)
    # "a:" + 16 hex chars
    assert len(key) == 18
