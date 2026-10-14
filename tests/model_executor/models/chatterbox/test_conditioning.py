# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from vllm_omni.data_entry_keys import to_struct
from vllm_omni.model_executor.models.chatterbox.conditioning import (
    VoiceConditioner,
    VoiceConditioning,
    build_prompt,
    load_text_tokenizer,
    normalize_loudness,
    punc_norm,
    tokenize_text,
    trim_silence,
    voice_encoder_mel,
)
from vllm_omni.model_executor.models.chatterbox.voice_encoder import VoiceEncConfig, VoiceEncoder
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
def test_punc_norm_matches_upstream(raw: str, expected: str) -> None:
    assert punc_norm(raw) == expected


def test_trim_silence_removes_leading_and_trailing_silence() -> None:
    sr = 16000
    tone = 0.5 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32)
    padded = np.concatenate([np.zeros(sr, np.float32), tone, np.zeros(sr, np.float32)])
    assert abs(len(trim_silence(padded)) - len(tone)) < 2048 + 512


def test_normalize_loudness_hits_the_target_and_keeps_float32() -> None:
    import pyloudnorm

    sr = 24000
    wav = (0.05 * np.random.default_rng(0).standard_normal(sr * 6)).astype(np.float32)
    out = normalize_loudness(wav, sr, -27.0)
    assert out.dtype == np.float32
    assert abs(pyloudnorm.Meter(sr).integrated_loudness(out) - (-27.0)) < 0.5


def test_normalize_loudness_returns_silence_unchanged() -> None:
    wav = np.zeros(24000, dtype=np.float32)
    assert np.array_equal(normalize_loudness(wav, 24000, -27.0), wav)


def test_voice_encoder_mel_shape_and_power() -> None:
    hp = VoiceEncConfig()
    wav = (0.1 * np.random.default_rng(0).standard_normal(16000)).astype(np.float32)
    mel = voice_encoder_mel(wav, hp)
    assert mel.shape == (1 + 16000 // hp.hop_size, hp.num_mels)
    assert mel.min() >= 0


def test_build_prompt_follows_the_conditioning_length() -> None:
    """A clip shorter than the 15 s window gives fewer than 375 prompt tokens."""
    cfg = ChatterboxConfig()
    for cond_tokens in (150, 375):
        prompt = build_prompt([5, 6, 7], conditioning(cond_tokens, 125), cfg)
        assert prompt["prompt_token_ids"] == [cfg.start_speech_token] * (1 + cond_tokens + 3 + 1)
        info = prompt["additional_information"]
        assert info["ids"]["prompt"] == [5, 6, 7]
        assert len(info["ids"]["speech_token"]) == cond_tokens


def test_additional_information_is_a_valid_stage_payload() -> None:
    """The keys must be ones the payload schema knows, or stage 1 rejects them."""
    info = conditioning(375, 250).additional_information([1, 2, 3])
    payload = to_struct(info)
    assert payload.embed.speech_feat.shape == (1, 500, 80)
    assert payload.embed.voice.shape == (1, 256)
    assert payload.ids.prompt == [1, 2, 3]


def test_voice_encoder_padded_batch_matches_individual_references() -> None:
    """Exercise the actual partial-window encoder across unequal lengths."""
    torch.manual_seed(7)
    encoder = VoiceEncoder(VoiceEncConfig()).eval()
    mels = [torch.rand(length, encoder.hp.num_mels) for length in (90, 205, 317)]
    with torch.inference_mode():
        individual = torch.cat([encoder.inference(mel[None], [len(mel)], batch_size=32, rate=1.3) for mel in mels])
        batched = encoder.inference(
            torch.nn.utils.rnn.pad_sequence(mels, batch_first=True),
            [len(mel) for mel in mels],
            batch_size=32,
            rate=1.3,
        )
    assert batched.shape == (3, 256)
    torch.testing.assert_close(batched.norm(dim=1), torch.ones(3), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(batched, individual, atol=1e-6, rtol=1e-5)


def test_tokenize_batch_uses_one_quantization_and_isolates_rows() -> None:
    """The package supplies the real padding primitive; no weights are needed."""
    import s3tokenizer

    class Quantizer:
        def __init__(self):
            self.calls = 0

        def quantize(self, mels, lengths):
            self.calls += 1
            assert mels.shape[0] == 2
            return torch.arange(mels.shape[-1]).tile(2, 1), lengths // 4

    conditioner = object.__new__(VoiceConditioner)
    conditioner.s3tokenizer = s3tokenizer
    conditioner.tokenizer = Quantizer()
    conditioner.device = torch.device("cpu")
    wavs = [torch.randn(length) for length in (16000, 24000)]
    tokens = conditioner.tokenize_batch(wavs, 30)
    assert conditioner.tokenizer.calls == 1
    assert [token.shape for token in tokens] == [(1, 25), (1, 30)]
    tokens[0].fill_(-1)
    assert tokens[1].min() == 0


def test_prepare_batch_is_bounded_and_keeps_order() -> None:
    from vllm_omni.model_executor.models.chatterbox.conditioning import CONDITIONING_BATCH_SIZE

    class Conditioner(VoiceConditioner):
        def __init__(self):
            self.sizes = []

        def resample(self, wav, sample_rate):
            return wav, wav

        def from_resampled_batch(self, clips):
            self.sizes.append(len(clips))
            return [int(wav[0]) for wav, _ in clips]

    conditioner = Conditioner()
    count = CONDITIONING_BATCH_SIZE * 2 + 1
    references = [(np.array([index]), 16000) for index in range(count)]
    assert conditioner.prepare_batch(references) == list(range(count))
    assert conditioner.sizes == [CONDITIONING_BATCH_SIZE, CONDITIONING_BATCH_SIZE, 1]
    assert conditioner.prepare_batch([]) == []


def test_original_prompt_rejects_text_beyond_learned_positions():
    with pytest.raises(ValueError, match="2048 text tokens"):
        build_prompt([5] * 2049, conditioning(10, 125), ChatterboxConfig("original"))


def test_original_text_helper_loads_local_tokenizer_and_normalizes_spaces(tmp_path):
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "Hello[SPACE]world.": 7}, unk_token="[UNK]"))
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    loaded = load_text_tokenizer(str(tmp_path), "original")
    assert tokenize_text("hello world", loaded, "original") == [7]
