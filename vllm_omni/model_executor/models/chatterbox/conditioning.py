# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reference conditioning and prompt construction for Chatterbox.

``VoiceConditioner`` mirrors ``ChatterboxTurboTTS.prepare_conditionals``
(chatterbox 0.1.7) step for step, without ``librosa`` and without the model
object it hangs off upstream. The result travels with the request in
``additional_information``: stage 0 reads ``ids.prompt``, ``ids.speech_token``
and ``embed.voice`` in its ``preprocess``; stage 1 reads ``embed.speech_token``,
``embed.speech_feat`` and ``embed.embedding``, the names the CosyVoice3
async-chunk processor forwards. The serving adapter and offline callers both
build prompts with ``build_prompt``.
"""

import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
from safetensors.torch import load_file
from vllm.inputs import tokens_input
from vllm.multimodal.parse import MultiModalDataParser

from vllm_omni.model_executor.models.chatterbox.s3gen_core.xvector import CAMPPlus
from vllm_omni.model_executor.models.chatterbox.voice_encoder import VoiceEncConfig, VoiceEncoder
from vllm_omni.model_executor.models.cosyvoice3.utils import mel_spectrogram
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig
from vllm_omni.transformers_utils.repo_utils import hf_api
from vllm_omni.utils.audio import mel_filter_bank

PUNCTUATION_REPLACEMENTS = (
    ("…", ", "),
    (":", ","),
    ("—", "-"),
    ("–", "-"),
    (" ,", ","),
    ("“", '"'),
    ("”", '"'),
    ("‘", "'"),
    ("’", "'"),
)
SENTENCE_ENDERS = (".", "!", "?", "-", ",")
# Not parameters the checkpoint carries: the S3Tokenizer subclass upstream
# registers them itself, and the s3tokenizer package computes its own.
TOKENIZER_BUFFERS = ("tokenizer._mel_filters", "tokenizer.window")


def punc_norm(text: str) -> str:
    """Port of ``chatterbox.tts_turbo.punc_norm``."""
    if len(text) == 0:
        return "You need to add some text for me to talk."
    if text[0].islower():
        text = text[0].upper() + text[1:]
    text = " ".join(text.split())
    for old, new in PUNCTUATION_REPLACEMENTS:
        text = text.replace(old, new)
    text = text.rstrip(" ")
    if not text.endswith(SENTENCE_ENDERS):
        text += "."
    return text


def normalize_loudness(wav: np.ndarray, sample_rate: int, target_lufs: float) -> np.ndarray:
    """Port of ``ChatterboxTurboTTS.norm_loudness``.

    ``integrated_loudness`` returns a ``numpy.float64``; under numpy 2 the
    product silently becomes float64, which the S3 tokenizer's float32 mel
    filters reject. Upstream has that defect; the cast here is the fix.

    Args:
        wav: Float32 mono samples.
        sample_rate: Sample rate of ``wav``.
        target_lufs: Integrated loudness to reach.

    Returns:
        The scaled samples, float32. Silence is returned unchanged.
    """
    try:
        import pyloudnorm
    except ImportError as error:
        raise ImportError(
            "Chatterbox reference conditioning needs 'pyloudnorm'; "
            "install it with `pip install 'vllm-omni[chatterbox]'`."
        ) from error

    loudness = pyloudnorm.Meter(sample_rate).integrated_loudness(wav)
    gain = 10.0 ** ((target_lufs - loudness) / 20.0)
    if not math.isfinite(gain) or gain <= 0.0:
        return wav
    return (wav * gain).astype(np.float32)


def trim_silence(wav: np.ndarray, top_db: float = 20.0) -> np.ndarray:
    """Port of ``librosa.effects.trim`` as ``VoiceEncoder.embeds_from_wavs`` calls it.

    Frames whose mean-square power is more than ``top_db`` below the loudest
    frame are silent; the signal is cut to the first and last other frame.
    """
    frame_length, hop_length = 2048, 512
    padded = np.pad(wav, frame_length // 2, mode="constant")
    n_frames = 1 + (len(padded) - frame_length) // hop_length
    index = np.arange(frame_length)[None, :] + hop_length * np.arange(n_frames)[:, None]
    power = np.mean(padded[index] ** 2, axis=1)
    decibels = 10.0 * np.log10(np.maximum(power, 1e-10) / max(power.max(), 1e-10))
    loud = np.flatnonzero(decibels > -top_db)
    if loud.size == 0:
        return wav
    return wav[int(loud[0] * hop_length) : min(len(wav), int((loud[-1] + 1) * hop_length))]


def voice_encoder_mel(wav16: np.ndarray, hp: VoiceEncConfig) -> torch.Tensor:
    """Port of ``chatterbox.models.voice_encoder.melspec.melspectrogram``, transposed.

    Power STFT magnitudes through a Slaney mel basis; the shipped config uses
    no log, no normalization and no pre-emphasis.

    Returns:
        Mel of shape (T, 40).
    """
    spectrum = torch.stft(
        torch.from_numpy(np.asarray(wav16, dtype=np.float32)),
        n_fft=hp.n_fft,
        hop_length=hp.hop_size,
        win_length=hp.win_size,
        window=torch.hann_window(hp.win_size),
        center=True,
        pad_mode="reflect",
        return_complex=True,
    ).abs()
    basis = mel_filter_bank(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels, fmin=hp.fmin, fmax=hp.fmax)
    return (basis @ spectrum**hp.mel_power).T.contiguous()


@dataclass
class VoiceConditioning:
    """One reference clip, as both stages consume it.

    Attributes:
        cond_tokens: Shape (1, C), S3 tokens of the first fifteen seconds, C <= 375.
        speaker_emb: Shape (1, 256), the voice encoder's embedding.
        prompt_token: Shape (1, P), S3 tokens of the first ten seconds, P <= 250.
        prompt_feat: Shape (1, 2P, 80), 24 kHz mel of the same ten seconds.
        embedding: Shape (1, 192), CAMPPlus x-vector of the same ten seconds.
    """

    cond_tokens: torch.Tensor
    speaker_emb: torch.Tensor
    prompt_token: torch.Tensor
    prompt_feat: torch.Tensor
    embedding: torch.Tensor

    def additional_information(self, text_ids: list[int]) -> dict:
        """The request payload: stage 0's prompt and stage 1's reference.

        Args:
            text_ids: GPT-2 ids of the normalized text.

        Returns:
            A nested dict that validates as an ``OmniPayloadStruct``.
        """
        return {
            "ids": {"prompt": text_ids, "speech_token": self.cond_tokens[0].tolist()},
            "embed": {
                "voice": self.speaker_emb,
                "speech_token": self.prompt_token,
                "speech_feat": self.prompt_feat,
                "embedding": self.embedding,
            },
        }


class VoiceConditioner:
    """The reference encoders, loaded once per process.

    Lives in the API process (the adapter builds it) or in an offline caller,
    on the CPU by default so the GPU is left to the two stages.
    """

    def __init__(self, model: str, config: ChatterboxConfig, device: torch.device) -> None:
        import s3tokenizer
        from s3tokenizer.model_v2 import S3TokenizerV2

        self.config = config
        self.device = device
        self.s3tokenizer = s3tokenizer
        model_dir = model
        if not os.path.isdir(model):
            model_dir = hf_api().snapshot_download(model, allow_patterns=[config.s3gen_weights, config.ve_weights])

        self.voice_encoder = VoiceEncoder(VoiceEncConfig())
        self.voice_encoder.load_state_dict(load_file(os.path.join(model_dir, config.ve_weights)))
        self.voice_encoder.to(device).eval()

        # The checkpoint's own tokenizer weights, not the package defaults:
        # the defaults produce a different codebook and cloning turns to noise.
        weights = load_file(os.path.join(model_dir, config.s3gen_weights))
        self.tokenizer = S3TokenizerV2(config.s3_tokenizer_name)
        self.tokenizer.load_state_dict(
            {
                name.removeprefix("tokenizer."): tensor
                for name, tensor in weights.items()
                if name.startswith("tokenizer.") and name not in TOKENIZER_BUFFERS
            },
            strict=True,
        )
        self.tokenizer.to(device).eval()

        self.campplus = CAMPPlus(memory_efficient=False)
        self.campplus.load_state_dict(
            {
                name.removeprefix("speaker_encoder."): tensor
                for name, tensor in weights.items()
                if name.startswith("speaker_encoder.")
            },
            strict=True,
        )
        self.campplus.to(device).eval()

        # soxr's default quality is the resampler librosa.load and
        # librosa.resample use upstream; vLLM's default (torchaudio) differs
        # enough to move a tenth of the S3 tokens.
        self.to_24k = MultiModalDataParser(target_sr=config.sample_rate, audio_resample_method="soxr").audio_resampler
        self.to_16k = MultiModalDataParser(
            target_sr=config.s3_sample_rate, audio_resample_method="soxr"
        ).audio_resampler

    def prepare(self, wav: np.ndarray, sample_rate: int) -> VoiceConditioning:
        """Condition on a reference clip.

        Args:
            wav: Float32 mono samples.
            sample_rate: Sample rate of ``wav``.

        Returns:
            The five tensors, on the CPU.

        Raises:
            ValueError: If the clip is not longer than five seconds, the
                bound upstream asserts.
        """
        config = self.config
        wav24 = self.to_24k.resample(np.asarray(wav, dtype=np.float32), orig_sr=sample_rate).astype(np.float32)
        seconds = len(wav24) / config.sample_rate
        if seconds <= config.min_ref_seconds:
            raise ValueError(
                f"Chatterbox needs a reference clip longer than {config.min_ref_seconds:g} s, got {seconds:.2f} s"
            )
        wav24 = normalize_loudness(wav24, config.sample_rate, config.loudness_target_lufs)
        wav16 = self.to_16k.resample(wav24, orig_sr=config.sample_rate).astype(np.float32)
        return self.from_resampled(wav24, wav16)

    @torch.inference_mode()
    def from_resampled(self, wav24: np.ndarray, wav16: np.ndarray) -> VoiceConditioning:
        """Condition on a clip already normalized and at both rates.

        Split from ``prepare`` so the port can be compared with upstream from
        identical samples, independent of the resampler.

        Args:
            wav24: The loudness-normalized clip at 24 kHz.
            wav16: The same clip at 16 kHz.
        """
        config = self.config
        mel = voice_encoder_mel(trim_silence(wav16), self.voice_encoder.hp)
        speaker_emb = self.voice_encoder.inference(
            mel.unsqueeze(0).to(self.device), mel_lens=[mel.shape[0]], batch_size=32, rate=1.3
        )

        encoder_clip = torch.from_numpy(wav16[: config.enc_cond_seconds * config.s3_sample_rate])
        cond_tokens = self.tokenize(encoder_clip, config.cond_prompt_len)

        # S3Gen.embed_ref resamples its own ten seconds with torchaudio.
        decoder_clip = torch.from_numpy(wav24[: config.dec_cond_seconds * config.sample_rate])
        decoder_clip_16 = torchaudio.functional.resample(decoder_clip, config.sample_rate, config.s3_sample_rate)
        prompt_token = self.tokenize(decoder_clip_16, None)
        prompt_feat = mel_spectrogram(decoder_clip.unsqueeze(0), **config.mel).transpose(1, 2)
        frames = min(prompt_feat.shape[1] // config.token_mel_ratio, prompt_token.shape[1])
        embedding = self.campplus.inference([decoder_clip_16.to(self.device)])

        return VoiceConditioning(
            cond_tokens=cond_tokens.cpu(),
            speaker_emb=speaker_emb.reshape(1, -1).float().cpu(),
            prompt_token=prompt_token[:, :frames].cpu(),
            prompt_feat=prompt_feat[:, : config.token_mel_ratio * frames].float().cpu(),
            embedding=embedding.float().cpu(),
        )

    def tokenize(self, wav16: torch.Tensor, max_tokens: int | None) -> torch.Tensor:
        """Port of ``chatterbox.models.s3tokenizer.S3Tokenizer.forward`` for one clip.

        Returns:
            Speech tokens of shape (1, T), long; ``T <= max_tokens`` when given.
        """
        mel = self.s3tokenizer.log_mel_spectrogram(wav16.float())
        if max_tokens is not None:
            mel = mel[..., : max_tokens * 4]
        mels, mel_lens = self.s3tokenizer.padding([mel])
        codes, code_lens = self.tokenizer.quantize(mels.to(self.device), mel_lens.to(self.device))
        return codes[:, : int(code_lens[0])].long()


def build_prompt(text_ids: list[int], conditioning: VoiceConditioning, config: ChatterboxConfig) -> dict:
    """The engine prompt for one utterance.

    Every prompt id is the start-of-speech token, whose logit is always
    ``-inf``: vLLM applies the repetition penalty to prompt ids too, and this
    keeps it off real speech tokens. Stage 0 replaces the placeholders with
    ``[speaker | prompt tokens | text | start-of-speech]`` embeddings.

    Args:
        text_ids: GPT-2 ids of ``punc_norm(text)``.
        conditioning: The reference clip's conditioning.
        config: The model config.

    Returns:
        A token prompt carrying the conditioning in ``additional_information``.
    """
    length = 1 + conditioning.cond_tokens.shape[1] + len(text_ids) + 1
    prompt = tokens_input(prompt_token_ids=[config.start_speech_token] * length)
    prompt["additional_information"] = conditioning.additional_information(text_ids)
    return prompt
