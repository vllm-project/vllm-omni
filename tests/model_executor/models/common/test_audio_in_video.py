# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The tensor-reduction interleave check agrees with vLLM's per-token loop."""

import random

import pytest
import torch

from vllm_omni.model_executor.models.common.audio_in_video import check_interleaved_audio_video

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

TEXT, VIDEO, AUDIO, IMAGE, BOUNDARY = 0, 1, 2, 3, 4


def _reference():
    from vllm.model_executor.models.qwen2_5_omni_thinker import check_interleaved_audio_video as reference

    return reference


def _masks(tokens):
    ids = torch.tensor(tokens, dtype=torch.long)
    return ids == VIDEO, ids == AUDIO


def _check(fn, tokens):
    is_video, is_audio = _masks(tokens)
    return fn(is_video, is_audio, int(is_video.sum()), int(is_audio.sum()))


def _random_request(rng):
    """One prompt: text, images, and videos with separate or interleaved audio, as the processors lay them out."""
    tokens = [TEXT] * rng.randint(0, 4)
    for _ in range(rng.randint(1, 4)):
        kind = rng.choice(["image", "video", "audio", "video_then_audio", "audio_in_video"])
        if kind == "image":
            tokens += [IMAGE] * rng.randint(1, 6)
        elif kind == "video":
            tokens += [VIDEO] * rng.randint(1, 8)
        elif kind == "audio":
            tokens += [AUDIO] * rng.randint(1, 8)
        elif kind == "video_then_audio":
            tokens += [VIDEO] * rng.randint(1, 6) + [AUDIO] * rng.randint(1, 6)
        else:
            tokens.append(BOUNDARY)
            for _ in range(rng.randint(1, 4)):
                tokens += [VIDEO] * rng.randint(0, 5) + [AUDIO] * rng.randint(0, 3)
            tokens.append(BOUNDARY)
        tokens += [rng.choice([TEXT, BOUNDARY])] * rng.randint(0, 2)
    return tokens


@pytest.mark.parametrize("seed", range(20))
def test_matches_vllm_on_random_batches(seed):
    rng = random.Random(seed)
    reference = _reference()
    outcomes = set()
    for _ in range(50):
        # The thinker checks a whole scheduled batch at once, so concatenate several prompts.
        tokens: list[int] = sum((_random_request(rng) for _ in range(rng.randint(1, 4))), [])
        expected = _check(reference, tokens)
        assert _check(check_interleaved_audio_video, tokens) == expected, tokens
        outcomes.add(expected)
    assert outcomes == {True, False}


def _audio_in_video(num_video, num_audio, chunk_video=600, chunk_audio=50):
    tokens = [BOUNDARY]
    while num_video or num_audio:
        step = min(num_video, chunk_video)
        tokens += [VIDEO] * step
        num_video -= step
        step = min(num_audio, chunk_audio)
        tokens += [AUDIO] * step
        num_audio -= step
    return tokens + [BOUNDARY]


@pytest.mark.parametrize(
    "tokens, expected",
    [
        # Daily-Omni sizes: one video with audio in it, and one with its audio as a separate item.
        ([TEXT] * 50 + _audio_in_video(6600, 750) + [TEXT] * 20, True),
        ([TEXT] * 50 + [BOUNDARY] + [VIDEO] * 6600 + [BOUNDARY, BOUNDARY] + [AUDIO] * 750 + [BOUNDARY], False),
    ],
)
def test_request_sized_inputs(tokens, expected):
    assert _check(check_interleaved_audio_video, tokens) is expected
    assert _check(_reference(), tokens) is expected


@pytest.mark.parametrize(
    "tokens, expected",
    [
        ([TEXT, VIDEO, VIDEO, AUDIO, AUDIO, TEXT], False),  # one span, video before audio
        ([VIDEO, AUDIO, VIDEO], True),
        ([AUDIO, VIDEO, AUDIO], True),
        ([BOUNDARY, VIDEO, AUDIO, BOUNDARY, VIDEO, AUDIO, BOUNDARY], False),  # two videos, each with audio after
        ([VIDEO, VIDEO, BOUNDARY, AUDIO, VIDEO, AUDIO], True),  # only the second span interleaves
        ([VIDEO, IMAGE, AUDIO, IMAGE, VIDEO], False),  # images split the spans
        ([TEXT, VIDEO, VIDEO, AUDIO, VIDEO, AUDIO, TEXT], True),  # two prompts touching, no boundary token between
        ([VIDEO] * 5, False),
        ([AUDIO] * 5, False),
        ([TEXT] * 5, False),
    ],
)
def test_span_rules(tokens, expected):
    assert _check(check_interleaved_audio_video, tokens) is expected
    assert _check(_reference(), tokens) is expected


@pytest.mark.parametrize(
    "module",
    [
        "vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker",
        "vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_thinker",
    ],
)
def test_thinkers_use_this_check(module):
    import importlib

    assert importlib.import_module(module).check_interleaved_audio_video is check_interleaved_audio_video


def _host_reads(monkeypatch, tokens):
    is_video, is_audio = _masks(tokens)
    num_video, num_audio = int(is_video.sum()), int(is_audio.sum())
    calls = []

    def counting(name):
        method = getattr(torch.Tensor, name)

        def wrapper(self, *args, **kwargs):
            calls.append(name)
            return method(self, *args, **kwargs)

        return wrapper

    with monkeypatch.context() as patch:
        for name in ("item", "tolist", "__bool__", "__int__", "__index__", "__float__"):
            patch.setattr(torch.Tensor, name, counting(name))
        result = check_interleaved_audio_video(is_video, is_audio, num_video, num_audio)
    return result, calls


def test_host_reads_do_not_grow_with_the_prompt(monkeypatch):
    # The vLLM loop calls .item() twice per placeholder token; this check reads back a fixed number of values.
    short, short_calls = _host_reads(monkeypatch, _audio_in_video(3000, 250))
    long, long_calls = _host_reads(monkeypatch, _audio_in_video(6000, 500) * 2)
    assert short is True and long is True
    assert short_calls == long_calls
    assert len(long_calls) <= 1
