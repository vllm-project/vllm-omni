# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniCPM-o 4.5 audio clips are padded to their own length, not to 30 s."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import WhisperFeatureExtractor

from vllm_omni.model_executor.models.minicpmo_4_5.audio_feature_padding import (
    _longest_waveform,
    trim_whisper_max_length_padding,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_SR = 16000
# The arguments processing_minicpmo.py passes for every audio clip.
_PROCESSOR_KWARGS = dict(sampling_rate=_SR, return_tensors="pt", padding="max_length", return_attention_mask=True)


def _extractor() -> WhisperFeatureExtractor:
    return WhisperFeatureExtractor(feature_size=80, sampling_rate=_SR, hop_length=160, chunk_length=30, n_fft=400)


def _trimmed() -> WhisperFeatureExtractor:
    extractor = _extractor()
    trim_whisper_max_length_padding(SimpleNamespace(audio_processor=extractor))
    return extractor


def _clip(seconds: float, seed: int = 0) -> np.ndarray:
    n = int(seconds * _SR)
    rng = np.random.default_rng(seed)
    tone = np.sin(2 * np.pi * 220.0 * np.arange(n) / _SR)
    return (0.3 * tone + 0.05 * rng.standard_normal(n)).astype(np.float32)


def _kept(features: torch.Tensor, attention_mask: torch.Tensor) -> list[torch.Tensor]:
    lens = attention_mask.sum(dim=1).tolist()
    return [feat[:, :length] for feat, length in zip(features, lens)]


# 1.3707 s (21931 samples) is not a multiple of hop_length; 1.0 s is.
@pytest.mark.parametrize("seconds", [0.05, 1.0, 1.3707, 6.0, 29.9])
@pytest.mark.parametrize("batched", [False, True], ids=["1d", "list"])
def test_kept_frames_match_30s_padding_exactly(seconds: float, batched: bool) -> None:
    clip = _clip(seconds)
    speech = [clip] if batched else clip

    stock = _extractor()(speech, **_PROCESSOR_KWARGS)
    trimmed = _trimmed()(speech, **_PROCESSOR_KWARGS)

    assert stock["input_features"].shape[-1] == 3000
    assert trimmed["input_features"].shape[-1] < 3000
    assert torch.equal(stock["attention_mask"].sum(dim=1), trimmed["attention_mask"].sum(dim=1))
    for want, got in zip(
        _kept(stock["input_features"], stock["attention_mask"]),
        _kept(trimmed["input_features"], trimmed["attention_mask"]),
        strict=True,
    ):
        assert torch.equal(want, got)


def test_batch_of_unequal_clips_pads_to_the_longest() -> None:
    clips = [_clip(1.0, seed=1), _clip(2.53, seed=2), _clip(0.4, seed=3)]

    stock = _extractor()(clips, **_PROCESSOR_KWARGS)
    trimmed = _trimmed()(clips, **_PROCESSOR_KWARGS)

    assert trimmed["input_features"].shape[-1] == (len(clips[1]) + 400) // 160
    for want, got in zip(
        _kept(stock["input_features"], stock["attention_mask"]),
        _kept(trimmed["input_features"], trimmed["attention_mask"]),
        strict=True,
    ):
        assert torch.equal(want, got)


def test_batched_ndarray_pads_to_the_sample_count() -> None:
    clips = np.stack([_clip(1.0, seed=1), _clip(1.0, seed=2)])

    stock = _extractor()(clips, **_PROCESSOR_KWARGS)
    trimmed = _trimmed()(clips, **_PROCESSOR_KWARGS)

    assert trimmed["input_features"].shape[-1] == (clips.shape[-1] + 400) // 160
    for want, got in zip(
        _kept(stock["input_features"], stock["attention_mask"]),
        _kept(trimmed["input_features"], trimmed["attention_mask"]),
        strict=True,
    ):
        assert torch.equal(want, got)


def test_multichannel_input_is_declined() -> None:
    """Anything but mono waveforms declines: a wrong length silently truncates."""
    assert _longest_waveform([np.zeros((2, 16000), dtype=np.float32)]) is None
    assert _longest_waveform(np.zeros((2, 16000, 3), dtype=np.float32)) is None
    assert _longest_waveform([]) is None


def test_the_dynamic_floor_sees_the_same_maximum() -> None:
    """A frame the mask drops can hold the per-clip max that sets the floor.

    Whisper clamps each clip's log-mel at ``max - 8``. For a near-silent clip
    with a loud transient in its last samples, the argmax is the frame at index
    ``mask_sum`` -- the first frame the mask drops -- so a layout that stops
    computing at the waveform's end moves the floor into the kept frames.
    """
    rng = np.random.default_rng(0)
    clip = (1e-5 * rng.standard_normal(16159)).astype(np.float32)
    clip[-1] = 1.0

    stock = _extractor()(clip, **_PROCESSOR_KWARGS)
    trimmed = _trimmed()(clip, **_PROCESSOR_KWARGS)
    (want,) = _kept(stock["input_features"], stock["attention_mask"])
    (got,) = _kept(trimmed["input_features"], trimmed["attention_mask"])
    assert torch.equal(want, got)

    # Stopping a hop_length past the waveform keeps every frame the mask keeps,
    # but not the dropped frame that holds the maximum.
    short = _extractor()(clip, **{**_PROCESSOR_KWARGS, "max_length": len(clip) + 160})
    (under,) = _kept(short["input_features"], short["attention_mask"])
    assert under.shape == want.shape
    assert not torch.equal(under, want)


def test_padding_to_the_bare_length_would_not_be_exact() -> None:
    """Guards the ``+ n_fft``: the bare length loses a frame or changes the tail."""
    ragged = _clip(1.3707)
    stock = _extractor()(ragged, **_PROCESSOR_KWARGS)
    bare = _extractor()(ragged, **{**_PROCESSOR_KWARGS, "max_length": len(ragged)})
    assert bare["attention_mask"].sum() == stock["attention_mask"].sum() - 1

    # On a hop multiple the frame count survives, but the centred STFT reflects
    # real samples into the last frame instead of reading zeros.
    even = _clip(1.0)
    stock = _extractor()(even, **_PROCESSOR_KWARGS)
    bare = _extractor()(even, **{**_PROCESSOR_KWARGS, "max_length": len(even)})
    (want,) = _kept(stock["input_features"], stock["attention_mask"])
    (got,) = _kept(bare["input_features"], bare["attention_mask"])
    assert want.shape == got.shape
    assert not torch.equal(want[:, -1], got[:, -1])


@pytest.mark.parametrize(
    "kwargs",
    [
        # Streaming mel path in processing_minicpmo.py.
        dict(sampling_rate=_SR, return_tensors="pt", padding=False),
        # Caller chose its own length.
        {**_PROCESSOR_KWARGS, "max_length": 2 * _SR},
        # Without the mask the caller may consume the full 30 s layout.
        dict(sampling_rate=_SR, return_tensors="pt", padding="max_length"),
    ],
    ids=["no-padding", "explicit-max-length", "no-attention-mask"],
)
def test_other_calls_are_unchanged(kwargs: dict) -> None:
    clip = _clip(1.0)
    want = _extractor()(clip, **kwargs)["input_features"]
    got = _trimmed()(clip, **kwargs)["input_features"]
    assert torch.equal(want, got)


def test_clip_reaching_30s_keeps_stock_padding_and_truncation() -> None:
    clip = _clip(31.0)
    want = _extractor()(clip, **_PROCESSOR_KWARGS)
    got = _trimmed()(clip, **_PROCESSOR_KWARGS)
    assert got["input_features"].shape[-1] == 3000
    assert torch.equal(want["input_features"], got["input_features"])


@pytest.mark.parametrize("attr", ["audio_processor", "feature_extractor"])
def test_wrap_is_idempotent_and_keeps_the_extractor_type(attr: str) -> None:
    extractor = _extractor()
    processor = SimpleNamespace(**{attr: extractor})

    trim_whisper_max_length_padding(processor)
    wrapped = type(extractor)
    trim_whisper_max_length_padding(processor)

    assert type(extractor) is wrapped
    assert wrapped is not WhisperFeatureExtractor
    assert isinstance(extractor, WhisperFeatureExtractor)
    # to_dict and save_pretrained write feature_extractor_type from the class
    # name, so the subclass has to keep the stock name and module.
    assert wrapped.__name__ == WhisperFeatureExtractor.__name__
    assert wrapped.__module__ == WhisperFeatureExtractor.__module__
    assert extractor.to_dict() == _extractor().to_dict()
    # Other extractors of the same class keep the stock behaviour.
    assert type(_extractor()) is WhisperFeatureExtractor


def test_processor_without_whisper_extractor_is_left_alone() -> None:
    trim_whisper_max_length_padding(SimpleNamespace())
    other = SimpleNamespace(audio_processor=SimpleNamespace(sampling_rate=_SR))
    trim_whisper_max_length_padding(other)
    assert type(other.audio_processor) is SimpleNamespace

    # A look-alike must not reach the class swap: assigning __class__ raises
    # for an object whose layout differs, which would fail the model load.
    class LooksLikeWhisper:
        __slots__ = ("n_fft", "n_samples")

        def __init__(self) -> None:
            self.n_samples = 30 * _SR
            self.n_fft = 400

    lookalike = LooksLikeWhisper()
    trim_whisper_max_length_padding(SimpleNamespace(audio_processor=lookalike))
    assert type(lookalike) is LooksLikeWhisper


def test_a_copy_keeps_the_trim_and_the_kept_frames() -> None:
    clip = _clip(1.0)
    restored = copy.deepcopy(_trimmed())

    assert isinstance(restored, WhisperFeatureExtractor)
    stock = _extractor()(clip, **_PROCESSOR_KWARGS)
    got = restored(clip, **_PROCESSOR_KWARGS)
    assert got["input_features"].shape[-1] < 3000
    (want,) = _kept(stock["input_features"], stock["attention_mask"])
    (mine,) = _kept(got["input_features"], got["attention_mask"])
    assert torch.equal(want, mine)


def test_reduction_names_the_stock_class_and_keeps_state() -> None:
    """The subclass has no importable name, so the reduction names the base.

    Asserted on the reduction contract directly rather than through a
    round-trip, because this repository forbids importing pickle.
    """
    extractor = _trimmed()
    extractor.some_marker = 7
    rebuild, args = extractor.__reduce__()
    cls, state = args

    assert cls is WhisperFeatureExtractor
    assert state["some_marker"] == 7
    assert state["n_fft"] == extractor.n_fft
    restored = rebuild(cls, state)
    assert type(restored) is type(extractor)
    assert restored.some_marker == 7
