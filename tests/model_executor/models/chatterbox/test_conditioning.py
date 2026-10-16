# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import re
import sys

import numpy as np
import pyloudnorm
import pytest
import torch
import torchaudio
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch import nn

from tests.model_executor.models.chatterbox.helpers import (
    random_voice,
    write_model_dir,
    write_s3gen_checkpoint,
    write_voice_encoder,
)
from vllm_omni.data_entry_keys import to_struct
from vllm_omni.model_executor.models.chatterbox.conditioning import (
    VoiceConditioner,
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
from vllm_omni.utils.audio import mel_filter_bank

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(scope="module")
def conditioners(tmp_path_factory: pytest.TempPathFactory) -> dict[str, VoiceConditioner]:
    """Each checkpoint's reference encoders, built by the conditioner from a model directory.

    The directory holds what the conditioner reads, in the released files'
    layout: ``ve.safetensors`` and the variant's S3Gen file, whose
    ``tokenizer.*`` and ``speaker_encoder.*`` tensors come from the module
    definitions with seeded initial weights. The S3 tokenizer is the
    ``s3tokenizer`` package's own, which needs no download.
    """
    directory = write_model_dir(tmp_path_factory.mktemp("conditioner"))
    write_voice_encoder(directory, seed=0)
    built = {}
    for variant in ("turbo", "original"):
        config = ChatterboxConfig(variant)
        write_s3gen_checkpoint(directory, config, seed=1)
        built[variant] = VoiceConditioner(directory, config, torch.device("cpu"))
    return built


def tokenizer_calls(conditioner: VoiceConditioner) -> tuple[list[int], torch.utils.hooks.RemovableHandle]:
    """The batch size of every S3 tokenizer quantization, from its encoder, which each one runs once."""
    calls: list[int] = []

    def encoder_input(module: nn.Module, inputs: tuple) -> None:
        calls.append(inputs[0].shape[0])

    return calls, conditioner.tokenizer.encoder.register_forward_pre_hook(encoder_input)


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
    (mel,) = voice_encoder_mel([wav], hp, torch.device("cpu"))
    assert mel.shape == (1 + 16000 // hp.hop_size, hp.num_mels)
    assert mel.min() >= 0
    assert torch.equal(voice_encoder_mel([2 * wav], hp, torch.device("cpu"))[0], 4 * mel)


def test_voice_encoder_mel_rows_are_each_clips_own_centered_stft() -> None:
    """Each row of a batch of unequal clips is the clip's own ``torch.stft(center=True)`` mel, then zeros."""
    hp = VoiceEncConfig()
    rng = np.random.default_rng(6)
    wavs = [(0.1 * rng.standard_normal(length)).astype(np.float32) for length in (16000, 9137, 24001)]

    batched = voice_encoder_mel(wavs, hp, torch.device("cpu"))

    basis = mel_filter_bank(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels, fmin=hp.fmin, fmax=hp.fmax)
    for row, wav in zip(batched, wavs, strict=True):
        spectrum = torch.stft(
            torch.from_numpy(wav),
            n_fft=hp.n_fft,
            hop_length=hp.hop_size,
            win_length=hp.win_size,
            window=torch.hann_window(hp.win_size),
            center=True,
            pad_mode="reflect",
            return_complex=True,
        ).abs()
        alone = (basis @ spectrum**hp.mel_power).T
        torch.testing.assert_close(row[: alone.shape[0]], alone, atol=1e-5, rtol=1e-5)
        assert not row[alone.shape[0] :].any()


def test_a_batch_of_unequal_clips_conditions_each_as_it_is_alone(conditioners: dict[str, VoiceConditioner]) -> None:
    """Clips shorter and longer than the conditioning windows, encoded together and one at a time."""
    conditioner = conditioners["original"]
    rng = np.random.default_rng(7)
    clips = []
    for seconds in (1.3, 7.2, 12.5):
        wav24 = (0.1 * rng.standard_normal(round(seconds * 24000))).astype(np.float32)
        clips.append((wav24, torchaudio.functional.resample(torch.from_numpy(wav24), 24000, 16000).numpy()))

    batched = conditioner.from_resampled_batch(clips)

    for voice, clip in zip(batched, clips, strict=True):
        (alone,) = conditioner.from_resampled_batch([clip])
        assert torch.equal(voice.cond_tokens, alone.cond_tokens)
        assert torch.equal(voice.prompt_token, alone.prompt_token)
        torch.testing.assert_close(voice.prompt_feat, alone.prompt_feat, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(voice.embedding, alone.embedding, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(voice.speaker_emb, alone.speaker_emb, atol=1e-6, rtol=1e-5)


def test_voice_encoder_mel_puts_a_1_khz_tone_in_its_slaney_band() -> None:
    """Forty Slaney bands over 0 to 8 kHz: 1 kHz falls in band 13, counting from 0."""
    hp = VoiceEncConfig()
    tone = (0.5 * np.sin(2 * np.pi * 1000 * np.arange(16000) / 16000)).astype(np.float32)
    assert int(voice_encoder_mel([tone], hp, torch.device("cpu"))[0].mean(dim=0).argmax()) == 13


def test_build_prompt_follows_the_conditioning_length() -> None:
    """A clip shorter than the 15 s window gives fewer than 375 prompt tokens."""
    cfg = ChatterboxConfig()
    for cond_tokens in (150, 375):
        prompt = build_prompt([5, 6, 7], random_voice(cond_tokens, 125), cfg)
        assert prompt["prompt_token_ids"] == [cfg.start_speech_token] * (1 + cond_tokens + 3 + 1)
        info = prompt["additional_information"]
        assert info["ids"]["prompt"] == [5, 6, 7]
        assert len(info["ids"]["speech_token"]) == cond_tokens


def test_additional_information_is_a_valid_stage_payload() -> None:
    """The keys must be ones the payload schema knows, or stage 1 rejects them."""
    info = random_voice(375, 250).additional_information([1, 2, 3])
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
        individual = torch.cat([encoder.inference(mel[None], [len(mel)], rate=1.3) for mel in mels])
        batched = encoder.inference(
            torch.nn.utils.rnn.pad_sequence(mels, batch_first=True), [len(mel) for mel in mels], rate=1.3
        )
    assert batched.shape == (4, 256)
    torch.testing.assert_close(batched.norm(dim=1), torch.ones(4), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(batched, individual, atol=1e-6, rtol=1e-5)


def test_tokenize_batch_uses_one_quantization_and_isolates_rows(conditioners: dict[str, VoiceConditioner]) -> None:
    """Both clips go through one quantization; each keeps its own length, and its own tensor."""
    conditioner = conditioners["turbo"]
    wavs = [torch.randn(length) for length in (16000, 24000)]
    calls, hook = tokenizer_calls(conditioner)
    try:
        tokens = conditioner.tokenize_batch(wavs, 30)
    finally:
        hook.remove()

    assert calls == [2]
    # 25 tokens a second; the second clip's 38 are cut to 30.
    assert [token.shape for token in tokens] == [(1, 25), (1, 30)]
    tokens[0].fill_(-1)
    assert tokens[1].min() >= 0


def test_prepare_batch_encodes_every_reference_together_and_keeps_order(
    conditioners: dict[str, VoiceConditioner],
) -> None:
    """Seventeen references go through the encoders as one batch, and each result is its own reference's.

    Each clip is 40 ms (one prompt token) longer than the one before, so the
    prompt lengths show the order; one clip encoded alone shows a result in
    the batch is that clip's own.
    """
    conditioner = conditioners["turbo"]
    rng = np.random.default_rng(4)
    count = 17
    references = [
        ((0.05 * rng.standard_normal(6 * 24000 + 960 * index)).astype(np.float32), 24000) for index in range(count)
    ]
    calls, hook = tokenizer_calls(conditioner)
    try:
        voices = conditioner.prepare_batch(references)
    finally:
        hook.remove()
    (alone,) = conditioner.prepare_batch(references[3:4])

    # One batch: the conditioning tokens, then the prompt tokens.
    assert calls == [count, count]
    lengths = [voice.prompt_token.shape[1] for voice in voices]
    assert lengths == list(range(lengths[0], lengths[0] + count))
    assert torch.equal(voices[3].cond_tokens, alone.cond_tokens)
    assert torch.equal(voices[3].prompt_token, alone.prompt_token)
    # The batched STFT is not bit-identical to the clip's own on every BLAS.
    torch.testing.assert_close(voices[3].prompt_feat, alone.prompt_feat, atol=1e-5, rtol=1e-5)
    assert torch.equal(voices[3].embedding, alone.embedding)
    torch.testing.assert_close(voices[3].speaker_emb, alone.speaker_emb, atol=1e-6, rtol=1e-5)
    assert conditioner.prepare_batch([]) == []


def test_a_refused_reference_leaves_the_rest_of_its_batch(conditioners: dict[str, VoiceConditioner]) -> None:
    """A clip the conditioner refuses gets its error in its place; the others are encoded in one batch."""
    conditioner = conditioners["turbo"]
    rng = np.random.default_rng(5)
    good = [((0.05 * rng.standard_normal(6 * 24000 + 960 * index)).astype(np.float32), 24000) for index in range(2)]
    stereo = (np.zeros((6 * 24000, 2), dtype=np.float32), 24000)
    calls, hook = tokenizer_calls(conditioner)
    try:
        first, refused, last = conditioner.prepare_batch([good[0], stereo, good[1]])
    finally:
        hook.remove()

    assert isinstance(refused, ValueError) and "mono" in str(refused)
    assert calls == [2, 2]
    assert torch.equal(first.prompt_token, conditioner.prepare(*good[0]).prompt_token)
    assert torch.equal(last.prompt_token, conditioner.prepare(*good[1]).prompt_token)
    with pytest.raises(ValueError, match="mono"):
        conditioner.prepare(*stereo)
    with pytest.raises(ValueError, match="samples in it"):
        conditioner.prepare(np.zeros(0, dtype=np.float32), 24000)


def test_original_prompt_rejects_text_beyond_learned_positions():
    with pytest.raises(ValueError, match="2048 text tokens"):
        build_prompt([5] * 2049, random_voice(10, 125), ChatterboxConfig("original"))


@pytest.mark.parametrize("control", ["exaggeration", "cfg_weight"])
@pytest.mark.parametrize("value", [-1, float("inf"), float("nan"), "0.5"])
def test_original_prompt_rejects_invalid_guidance_controls(control: str, value: object) -> None:
    """The serving adapter passes a request's ``extra_params`` through unchecked; this is their one check."""
    with pytest.raises(ValueError, match=f"{control} must be a finite nonnegative number"):
        build_prompt(
            [5],
            random_voice(10, 125),
            ChatterboxConfig("original"),
            **{"exaggeration": 0.5, "cfg_weight": 0.5, control: value},
        )


@pytest.mark.parametrize("missing", ["exaggeration", "cfg_weight"])
def test_original_prompt_requires_both_controls(missing: str) -> None:
    controls = {name: 0.5 for name in ("exaggeration", "cfg_weight") if name != missing}
    with pytest.raises(ValueError, match=f"Chatterbox Original prompts require {missing}"):
        build_prompt([5], random_voice(10, 125), ChatterboxConfig("original"), **controls)


def test_original_text_helper_loads_local_tokenizer_and_normalizes_spaces(tmp_path):
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "Hello[SPACE]world.": 7}, unk_token="[UNK]"))
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    loaded = load_text_tokenizer(str(tmp_path), "original")
    assert tokenize_text("hello world", loaded, "original") == [7]


def test_a_reference_that_is_not_mono_is_refused_by_its_shape(conditioners: dict[str, VoiceConditioner]) -> None:
    with pytest.raises(ValueError, match=re.escape("got shape (2, 24000)")):
        conditioners["turbo"].resample(np.zeros((2, 24000), dtype=np.float32), 24000)


def test_turbo_repeats_a_reference_of_five_seconds_or_less_and_normalizes_it(
    conditioners: dict[str, VoiceConditioner],
) -> None:
    """Upstream refuses a clip of five seconds or less; here it is repeated past five seconds.

    Every clip is then normalized to -27 LUFS, as upstream normalizes its clips.
    """
    conditioner = conditioners["turbo"]
    wav = (0.05 * np.random.default_rng(1).standard_normal(5 * 24000 + 1)).astype(np.float32)

    wav24, wav16 = conditioner.resample(wav, 24000)
    assert wav24.shape == wav.shape
    assert abs(pyloudnorm.Meter(24000).integrated_loudness(wav24) - (-27.0)) < 1e-3
    assert wav16.shape == (round(wav24.shape[0] * 16000 / 24000),)

    # Exactly five seconds is repeated once; two seconds, three times.
    five, _ = conditioner.resample(wav[:-1], 24000)
    assert five.shape == (2 * 5 * 24000,) and np.array_equal(five[: 5 * 24000], five[5 * 24000 :])
    two, _ = conditioner.resample(wav[: 2 * 24000], 24000)
    assert two.shape == (3 * 2 * 24000,) and np.array_equal(two[: 2 * 24000], two[4 * 24000 :])
    assert abs(pyloudnorm.Meter(24000).integrated_loudness(two) - (-27.0)) < 1e-3


def test_original_takes_a_short_reference_as_it_is(conditioners: dict[str, VoiceConditioner]) -> None:
    """Upstream's Original neither bounds the clip's length nor normalizes its loudness."""
    wav = (0.05 * np.random.default_rng(2).standard_normal(24000)).astype(np.float32)

    wav24, _ = conditioners["original"].resample(wav, 24000)

    assert np.array_equal(wav24, wav)


# 24 kHz samples: the tokenizer gives one token more than the mel's frames
# hold at the first two lengths and as many at the third, which is past the
# ten seconds both are cut to. It never gives fewer (measured over two
# seconds of lengths in 37-sample steps).
@pytest.mark.parametrize(
    ("samples", "extra"),
    [(144259, 1), (239999, 1), (264000, 0)],
    ids=["one token more", "one token more at the ten-second cut", "as many, past the cut"],
)
def test_the_prompt_mel_is_cut_to_two_frames_per_prompt_token(
    conditioners: dict[str, VoiceConditioner], samples: int, extra: int
) -> None:
    """``flow_mels`` writes the prompt mel over the first ``2 * P`` frames, so the two must agree.

    The tokenizer and the mel are the real ones on the clip's first ten
    seconds; the prompt keeps whole token pairs of mel frames.
    """
    conditioner = conditioners["turbo"]
    config = conditioner.config
    rng = np.random.default_rng(3)
    wav24 = (0.1 * rng.standard_normal(samples)).astype(np.float32)
    wav16 = (0.1 * rng.standard_normal(samples * 2 // 3)).astype(np.float32)
    clip = torch.from_numpy(wav24[: config.dec_cond_seconds * config.sample_rate])
    mel = mel_spectrogram(clip[None], **config.mel).transpose(1, 2)
    (tokens,) = conditioner.tokenize_batch([torchaudio.functional.resample(clip, 24000, 16000)], None)
    assert tokens.shape[1] - mel.shape[1] // config.token_mel_ratio == extra

    (conditioning,) = conditioner.from_resampled_batch([(wav24, wav16)])

    kept = min(tokens.shape[1], mel.shape[1] // config.token_mel_ratio)
    assert conditioning.prompt_feat.shape[1] == config.token_mel_ratio * conditioning.prompt_token.shape[1]
    assert torch.equal(conditioning.prompt_token, tokens[:, :kept])
    assert torch.equal(conditioning.prompt_feat, mel[:, : config.token_mel_ratio * kept])
    (cond_tokens,) = conditioner.tokenize_batch(
        [torch.from_numpy(wav16[: config.enc_cond_seconds * config.s3_sample_rate])], config.cond_prompt_len
    )
    assert torch.equal(conditioning.cond_tokens, cond_tokens)
