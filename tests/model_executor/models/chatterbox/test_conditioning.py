# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import re
import sys
from types import SimpleNamespace

import numpy as np
import pyloudnorm
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
from vllm_omni.model_executor.models.cosyvoice3.utils import mel_spectrogram
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
    # librosa's frames of 2048 samples every 512, centred: the first frame
    # louder than -20 dB is 30 (it reaches into the tone at 16000), the last
    # is 64, so the cut is [30 * 512, 65 * 512).
    assert np.array_equal(trim_silence(padded), padded[15360:33280])


def test_normalize_loudness_hits_the_target_and_keeps_float32() -> None:
    sr = 24000
    meter = pyloudnorm.Meter(sr)
    wav = (0.05 * np.random.default_rng(0).standard_normal(sr * 6)).astype(np.float32)
    # Far enough from the target that a gain in the wrong units would miss it.
    assert abs(meter.integrated_loudness(wav) - (-27.0)) > 3
    out = normalize_loudness(wav, sr, -27.0)
    assert out.dtype == np.float32
    # Measured: within 2.2e-8 LU; the gain is exact up to float32 rounding.
    assert abs(meter.integrated_loudness(out) - (-27.0)) < 1e-3


def test_normalize_loudness_names_the_extra_that_installs_its_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pyloudnorm", None)
    with pytest.raises(ImportError, match=re.escape("pip install 'vllm-omni[chatterbox]'")):
        normalize_loudness(np.ones(24000, dtype=np.float32), 24000, -27.0)


def test_normalize_loudness_returns_silence_unchanged() -> None:
    wav = np.zeros(24000, dtype=np.float32)
    assert np.array_equal(normalize_loudness(wav, 24000, -27.0), wav)


def test_voice_encoder_mel_shape_and_power() -> None:
    """Power magnitudes through a linear basis, with no log: doubling the signal quadruples every bin."""
    hp = VoiceEncConfig()
    wav = (0.1 * np.random.default_rng(0).standard_normal(16000)).astype(np.float32)
    mel = voice_encoder_mel(wav, hp)
    assert mel.shape == (1 + 16000 // hp.hop_size, hp.num_mels)
    assert mel.min() >= 0
    assert torch.equal(voice_encoder_mel(2 * wav, hp), 4 * mel)


def test_voice_encoder_mel_puts_a_1_khz_tone_in_its_slaney_band() -> None:
    """Forty Slaney bands over 0 to 8 kHz: 1 kHz falls in band 13, counting from 0."""
    hp = VoiceEncConfig()
    tone = (0.5 * np.sin(2 * np.pi * 1000 * np.arange(16000) / 16000)).astype(np.float32)
    assert int(voice_encoder_mel(tone, hp).mean(dim=0).argmax()) == 13


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
    # One partial window, then several; 520 frames is a clip just over five seconds.
    mels = [torch.rand(length, encoder.hp.num_mels) for length in (90, 205, 317, 520)]
    with torch.inference_mode():
        individual = torch.cat([encoder.inference(mel[None], [len(mel)], batch_size=32, rate=1.3) for mel in mels])
        batched = encoder.inference(
            torch.nn.utils.rnn.pad_sequence(mels, batch_first=True),
            [len(mel) for mel in mels],
            batch_size=32,
            rate=1.3,
        )
    assert batched.shape == (4, 256)
    torch.testing.assert_close(batched.norm(dim=1), torch.ones(4), atol=1e-5, rtol=1e-5)
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


def pass_through_conditioner(variant: str) -> VoiceConditioner:
    """The conditioner short of its checkpoint, with a resampler that returns its input."""
    conditioner = object.__new__(VoiceConditioner)
    conditioner.config = ChatterboxConfig(variant)
    conditioner.device = torch.device("cpu")
    resampler = SimpleNamespace(resample=lambda wav, orig_sr: wav)
    conditioner.to_24k = conditioner.to_16k = resampler
    return conditioner


def test_a_reference_that_is_not_mono_is_refused_by_its_shape() -> None:
    with pytest.raises(ValueError, match=re.escape("got shape (2, 24000)")):
        pass_through_conditioner("turbo").resample(np.zeros((2, 24000), dtype=np.float32), 24000)


def test_turbo_needs_a_reference_longer_than_five_seconds_and_normalizes_it() -> None:
    """Upstream asserts the bound and normalizes the clip to -27 LUFS before encoding it."""
    conditioner = pass_through_conditioner("turbo")
    wav = (0.05 * np.random.default_rng(1).standard_normal(5 * 24000 + 1)).astype(np.float32)
    with pytest.raises(ValueError, match="longer than 5 s, got 5.00 s"):
        conditioner.resample(wav[:-1], 24000)

    wav24, wav16 = conditioner.resample(wav, 24000)

    assert abs(pyloudnorm.Meter(24000).integrated_loudness(wav24) - (-27.0)) < 1e-3
    assert np.array_equal(wav16, wav24)


def test_original_takes_a_short_reference_as_it_is() -> None:
    """Upstream's Original neither bounds the clip's length nor normalizes its loudness."""
    wav = (0.05 * np.random.default_rng(2).standard_normal(24000)).astype(np.float32)

    wav24, _ = pass_through_conditioner("original").resample(wav, 24000)

    assert np.array_equal(wav24, wav)


@pytest.mark.parametrize("extra", [1, -1], ids=["one token more than the mel", "one token fewer"])
def test_the_prompt_mel_is_cut_to_two_frames_per_prompt_token(extra: int) -> None:
    """``flow_mels`` writes the prompt mel over the first ``2 * P`` frames, so the two must agree.

    The encoders are stand-ins; the mel is the real one. The tokenizer gives
    one token more, or one fewer, than the mel's frames hold.
    """
    conditioner = pass_through_conditioner("turbo")
    config = conditioner.config
    conditioner.voice_encoder = SimpleNamespace(
        hp=VoiceEncConfig(), inference=lambda mels, mel_lens, batch_size, rate: torch.zeros(len(mel_lens), 256)
    )
    conditioner.campplus = lambda fbank: torch.zeros(fbank.shape[0], 192)
    rng = np.random.default_rng(3)
    wav24 = (0.1 * rng.standard_normal(6 * config.sample_rate)).astype(np.float32)
    wav16 = (0.1 * rng.standard_normal(6 * config.s3_sample_rate)).astype(np.float32)
    mel = mel_spectrogram(torch.from_numpy(wav24)[None], **config.mel).transpose(1, 2)
    tokens = torch.arange(mel.shape[1] // config.token_mel_ratio + extra)[None]
    conditioner.tokenize_batch = lambda wavs, max_tokens: [tokens[:, :max_tokens] for _ in wavs]

    conditioning = conditioner.from_resampled(wav24, wav16)

    kept = min(tokens.shape[1], mel.shape[1] // config.token_mel_ratio)
    assert torch.equal(conditioning.prompt_token, tokens[:, :kept])
    assert torch.equal(conditioning.prompt_feat, mel[:, : config.token_mel_ratio * kept])
    assert torch.equal(conditioning.cond_tokens, tokens[:, : config.cond_prompt_len])
