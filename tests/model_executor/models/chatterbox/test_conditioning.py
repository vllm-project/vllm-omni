# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import numpy as np
import pytest
import torch

from vllm_omni.data_entry_keys import to_struct
from vllm_omni.model_executor.models.chatterbox.conditioning import (
    VoiceConditioning,
    build_prompt,
    normalize_loudness,
    punc_norm,
    trim_silence,
    voice_encoder_mel,
)
from vllm_omni.model_executor.models.chatterbox.voice_encoder import VoiceEncConfig
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def conditioning(cond_tokens: int, prompt_tokens: int) -> VoiceConditioning:
    return VoiceConditioning(
        cond_tokens=torch.randint(0, 6561, (1, cond_tokens)),
        speaker_emb=torch.randn(1, 256),
        prompt_token=torch.randint(0, 6561, (1, prompt_tokens)),
        prompt_feat=torch.randn(1, 2 * prompt_tokens, 80),
        embedding=torch.randn(1, 192),
    )


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("hello world", "Hello world."),
        ("Wait…  what:  now", "Wait,  what, now."),
        ("It’s “fine”", 'It\'s "fine".'),
        ("Done!", "Done!"),
        ("", "You need to add some text for me to talk."),
    ],
)
def test_punc_norm_matches_upstream_001(raw: str, expected: str) -> None:
    assert punc_norm(raw) == expected


def test_trim_silence_removes_leading_and_trailing_silence_001() -> None:
    sr = 16000
    tone = 0.5 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32)
    padded = np.concatenate([np.zeros(sr, np.float32), tone, np.zeros(sr, np.float32)])
    assert abs(len(trim_silence(padded)) - len(tone)) < 2048 + 512


def test_normalize_loudness_hits_the_target_and_keeps_float32_001() -> None:
    import pyloudnorm

    sr = 24000
    wav = (0.05 * np.random.default_rng(0).standard_normal(sr * 6)).astype(np.float32)
    out = normalize_loudness(wav, sr, -27.0)
    assert out.dtype == np.float32
    assert abs(pyloudnorm.Meter(sr).integrated_loudness(out) - (-27.0)) < 0.5


def test_normalize_loudness_returns_silence_unchanged_001() -> None:
    wav = np.zeros(24000, dtype=np.float32)
    assert np.array_equal(normalize_loudness(wav, 24000, -27.0), wav)


def test_voice_encoder_mel_shape_and_power_001() -> None:
    hp = VoiceEncConfig()
    wav = (0.1 * np.random.default_rng(0).standard_normal(16000)).astype(np.float32)
    mel = voice_encoder_mel(wav, hp)
    assert mel.shape == (1 + 16000 // hp.hop_size, hp.num_mels)
    assert mel.min() >= 0


def test_build_prompt_follows_the_conditioning_length_001() -> None:
    """A clip shorter than the 15 s window gives fewer than 375 prompt tokens."""
    cfg = ChatterboxConfig()
    for cond_tokens in (150, 375):
        prompt = build_prompt([5, 6, 7], conditioning(cond_tokens, 125), cfg)
        assert prompt["prompt_token_ids"] == [cfg.start_speech_token] * (1 + cond_tokens + 3 + 1)
        info = prompt["additional_information"]
        assert info["ids"]["prompt"] == [5, 6, 7]
        assert len(info["ids"]["speech_token"]) == cond_tokens


def test_additional_information_is_a_valid_stage_payload_001() -> None:
    """The keys must be ones the payload schema knows, or stage 1 rejects them."""
    info = conditioning(375, 250).additional_information([1, 2, 3])
    payload = to_struct(info)
    assert payload.embed.speech_feat.shape == (1, 500, 80)
    assert payload.embed.voice.shape == (1, 256)
    assert payload.ids.prompt == [1, 2, 3]
