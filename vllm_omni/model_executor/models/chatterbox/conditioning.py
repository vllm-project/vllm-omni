# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reference conditioning and prompt construction for Chatterbox.

``VoiceConditioner`` mirrors ``ChatterboxTurboTTS.prepare_conditionals`` and,
for Original, ``ChatterboxTTS.prepare_conditionals`` (chatterbox 0.1.7) step
for step, without ``librosa`` and without the model object it hangs off
upstream, with one deliberate difference: the decoder's prompt mel is cut to
exactly two frames per prompt token (see
``VoiceConditioner.from_resampled_batch``). The result travels with the
request in ``additional_information``: stage 0 reads ``ids.prompt``, ``ids.speech_token``
and ``embed.voice`` in its ``preprocess``; stage 1 reads ``embed.speech_token``,
``embed.speech_feat`` and ``embed.embedding``, which the stage handoff
(``stage_input_processors.chatterbox``) forwards to it. The serving adapter
and offline callers both build prompts with ``build_prompt``.
"""

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
from safetensors.torch import load_file
from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm.inputs import tokens_input
from vllm.multimodal.parse import MultiModalDataParser

from vllm_omni.model_executor.models.chatterbox.original_heads import (
    ORIGINAL_MAX_TEXT_TOKENS,
    ORIGINAL_PROMPT_EXTRA_ROWS,
)
from vllm_omni.model_executor.models.chatterbox.voice_encoder import VoiceEncConfig, VoiceEncoder
from vllm_omni.model_executor.models.cosyvoice3.utils import mel_spectrogram
from vllm_omni.model_executor.models.indextts2.preprocess_utils import compute_fbank
from vllm_omni.model_executor.models.indextts2.utils.campplus.dtdnn import CAMPPlus
from vllm_omni.transformers_utils.configs.chatterbox import CFG_UNCOND_SUFFIX, ChatterboxConfig
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


def punc_norm(text: str, variant: str = "turbo") -> str:
    """Port of ``chatterbox.tts_turbo.punc_norm``."""
    if len(text) == 0:
        return "You need to add some text for me to talk."
    if text[0].islower():
        text = text[0].upper() + text[1:]
    text = " ".join(text.split())
    if variant == "original":
        for old, new in (("...", ", "), (" - ", ", "), (";", ", ")):
            text = text.replace(old, new)
    for old, new in PUNCTUATION_REPLACEMENTS:
        text = text.replace(old, new)
    text = text.rstrip(" ")
    if not text.endswith(SENTENCE_ENDERS):
        text += "."
    return text


def load_text_tokenizer(model: str, variant: str) -> PreTrainedTokenizerBase | Tokenizer:
    """Load the checkpoint tokenizer, including Original's config-less JSON."""
    if variant == "turbo":
        return AutoTokenizer.from_pretrained(model, tokenizer_type="gpt2")
    if variant != "original":
        raise ValueError(f"Unsupported Chatterbox variant: {variant}")
    path = (
        os.path.join(model, "tokenizer.json")
        if os.path.isdir(model)
        else hf_api().hf_hub_download(model, "tokenizer.json")
    )
    return Tokenizer.from_file(path)


def tokenize_text(text: str, tokenizer: PreTrainedTokenizerBase | Tokenizer, variant: str) -> list[int]:
    """Encode normalized text with the variant's released tokenizer contract."""
    normalized = punc_norm(text, variant)
    if variant == "original":
        if not isinstance(tokenizer, Tokenizer):
            raise TypeError("Original requires a tokenizers.Tokenizer")
        return tokenizer.encode(normalized.replace(" ", "[SPACE]")).ids
    if variant != "turbo":
        raise ValueError(f"Unsupported Chatterbox variant: {variant}")
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise TypeError("Turbo requires a Transformers tokenizer")
    return tokenizer.encode(normalized, add_special_tokens=False)


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
    power = np.lib.stride_tricks.sliding_window_view(padded**2, frame_length)[::hop_length].mean(axis=1)
    decibels = 10.0 * np.log10(np.maximum(power, 1e-10) / max(power.max(), 1e-10))
    loud = np.flatnonzero(decibels > -top_db)
    if loud.size == 0:
        return wav
    return wav[int(loud[0] * hop_length) : min(len(wav), int((loud[-1] + 1) * hop_length))]


def to_device_batch(arrays: list[np.ndarray], device: torch.device) -> torch.Tensor:
    """Zero-pad float32 clips to one batch and copy it to ``device`` once.

    Returns:
        Shape (B, N), N the longest clip.
    """
    return torch.nn.utils.rnn.pad_sequence(
        [torch.from_numpy(np.asarray(array, dtype=np.float32)) for array in arrays], batch_first=True
    ).to(device)


def voice_encoder_mel(wavs16: list[np.ndarray], hp: VoiceEncConfig, device: torch.device) -> torch.Tensor:
    """Port of ``chatterbox.models.voice_encoder.melspec.melspectrogram`` for a batch, transposed.

    Power STFT magnitudes through a Slaney mel basis; the shipped config uses
    no log, no normalization and no pre-emphasis. Each clip gets its own
    reflected edges on the host, as ``torch.stft``'s centering would give it
    alone, so the batch's zero padding reaches none of its frames.

    Args:
        wavs16: Clips at the encoder's rate, of any lengths.
        hp: The encoder's config.
        device: Where the STFT and the mel run.

    Returns:
        Shape (B, T, 40), each clip's ``1 + len // hop_size`` frames followed by zeros.
    """
    edge = hp.n_fft // 2
    batch = to_device_batch([np.pad(wav, edge, mode="reflect") for wav in wavs16], device)
    spectrum = torch.stft(
        batch,
        n_fft=hp.n_fft,
        hop_length=hp.hop_size,
        win_length=hp.win_size,
        window=torch.hann_window(hp.win_size, device=device),
        center=False,
        return_complex=True,
    ).abs()
    basis = mel_filter_bank(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels, fmin=hp.fmin, fmax=hp.fmax)
    mels = (basis.to(device) @ spectrum**hp.mel_power).transpose(1, 2)
    frames = torch.tensor([1 + len(wav) // hp.hop_size for wav in wavs16], device=device)
    return mels * (torch.arange(mels.shape[1], device=device) < frames[:, None]).unsqueeze(-1)


@dataclass
class VoiceConditioning:
    """One reference clip, as both stages consume it.

    Attributes:
        cond_tokens: Shape (1, C), S3 tokens of the clip's first ``enc_cond_seconds``,
            C <= ``cond_prompt_len`` (15 s and 375 for Turbo, 6 s and 150 for Original).
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

    @classmethod
    def from_builtin(cls, model: str) -> "VoiceConditioning":
        """Load the checkpoint's built-in voice without constructing encoders."""
        model_dir = model
        if not os.path.isdir(model):
            model_dir = hf_api().snapshot_download(model, allow_patterns=["conds.pt"])
        conditionals = torch.load(os.path.join(model_dir, "conds.pt"), map_location="cpu", weights_only=True)
        t3, gen = conditionals["t3"], conditionals["gen"]
        return cls(
            cond_tokens=t3["cond_prompt_speech_tokens"],
            speaker_emb=t3["speaker_emb"],
            prompt_token=gen["prompt_token"],
            prompt_feat=gen["prompt_feat"],
            embedding=gen["embedding"],
        )

    def additional_information(self, text_ids: list[int]) -> dict:
        """The request payload: stage 0's prompt and stage 1's reference.

        Args:
            text_ids: Text tokenizer ids of the normalized text.

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

        # The same CAMPPlus as IndexTTS2's, at S3Gen's 192-dim x-vector.
        self.campplus = CAMPPlus(feat_dim=80, embedding_size=192, memory_efficient=False)
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
        # librosa.resample use upstream; vLLM's default is torchaudio.
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
            ValueError: If ``wav`` is not one-dimensional, or the clip is not
                longer than the checkpoint's minimum, see ``resample``.
        """
        (voice,) = self.prepare_batch([(wav, sample_rate)])
        if isinstance(voice, ValueError):
            raise voice
        return voice

    def prepare_batch(self, references: list[tuple[np.ndarray, int]]) -> list[VoiceConditioning | ValueError]:
        """Encode every reference together, preserving input order.

        The references are resampled concurrently on the CPU, then encoded
        as one batch: the caller's admission bound, not this method, decides
        how many fit the device. A reference ``resample`` refuses gets its
        error in its place and leaves the others' batch.
        """

        def resample(reference: tuple[np.ndarray, int]) -> tuple[np.ndarray, np.ndarray] | ValueError:
            try:
                return self.resample(*reference)
            except ValueError as error:
                return error

        with ThreadPoolExecutor(thread_name_prefix="chatterbox-resample") as pool:
            clips = list(pool.map(resample, references))
        voices = iter(self.from_resampled_batch([clip for clip in clips if not isinstance(clip, ValueError)]))
        return [clip if isinstance(clip, ValueError) else next(voices) for clip in clips]

    def resample(self, wav: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        """Validate, normalize and resample one reference at both model rates.

        A clip no longer than the checkpoint's ``min_ref_seconds`` is repeated
        until it is longer. Upstream Turbo refuses such a clip; repeating it
        keeps the voice and gives the conditioning windows the length the
        checkpoint was tuned for.
        """
        if wav.ndim != 1:
            raise ValueError(f"Chatterbox needs a mono reference clip of shape (samples,), got shape {wav.shape}")
        wav24 = self.to_24k.resample(np.asarray(wav, dtype=np.float32), orig_sr=sample_rate).astype(np.float32)
        if len(wav24) == 0:
            raise ValueError("Chatterbox needs a reference clip with samples in it")
        minimum = int(self.config.min_ref_seconds * self.config.sample_rate)
        if len(wav24) <= minimum:
            wav24 = np.tile(wav24, minimum // len(wav24) + 1)
        if self.config.loudness_target_lufs is not None:
            wav24 = normalize_loudness(wav24, self.config.sample_rate, self.config.loudness_target_lufs)
        wav16 = self.to_16k.resample(wav24, orig_sr=self.config.sample_rate).astype(np.float32)
        return wav24, wav16

    @torch.inference_mode()
    @torch.backends.cudnn.flags(
        enabled=None, benchmark=None, benchmark_limit=None, deterministic=None, allow_tf32=False
    )
    def from_resampled_batch(self, clips: list[tuple[np.ndarray, np.ndarray]]) -> list[VoiceConditioning]:
        """Encode clips with FP32 speaker kernels and unpadded CAMPPlus statistics.

        Each signal family goes to the device as one padded batch, its
        features are computed there, and the results come back once. Every
        clip's features are its own: its edges are reflected from its own
        samples on the host, and lengths come from each clip.

        cuDNN's TF32 RNN/convolution kernels change speaker embeddings with
        batch shape. The native flags context restores the caller's precision.
        """
        if not clips:
            return []
        config, device = self.config, self.device
        hp = self.voice_encoder.hp
        trimmed = [trim_silence(wav16) for _, wav16 in clips]
        # inference selects partial windows using mel_lens and reads zeros past them.
        speaker_embs = self.voice_encoder.inference(
            voice_encoder_mel(trimmed, hp, device), mel_lens=[1 + len(wav) // hp.hop_size for wav in trimmed], rate=1.3
        )
        encoder_clips = [wav16[: config.enc_cond_seconds * config.s3_sample_rate] for _, wav16 in clips]
        encoder_batch = to_device_batch(encoder_clips, device)
        cond_tokens = self.tokenize_batch(
            [encoder_batch[index, : len(clip)] for index, clip in enumerate(encoder_clips)], config.cond_prompt_len
        )

        decoder_clips = [wav24[: config.dec_cond_seconds * config.sample_rate] for wav24, _ in clips]
        lengths = torch.tensor([len(clip) for clip in decoder_clips], device=device)
        # mel_spectrogram reflects (n_fft - hop) / 2 samples on each side; each
        # clip carries its own right reflection, so the batch padding never
        # stands in for it.
        edge = (config.mel["n_fft"] - config.mel["hop_size"]) // 2
        decoder_batch = to_device_batch(
            [np.concatenate([clip, clip[-2 : -edge - 2 : -1]]) for clip in decoder_clips], device
        )
        prompt_feats = mel_spectrogram(decoder_batch, **config.mel).transpose(1, 2)
        # S3Gen.embed_ref resamples its own ten seconds with torchaudio; zeros,
        # not the reflection, follow each clip, as they would alone.
        unpadded = decoder_batch * (torch.arange(decoder_batch.shape[1], device=device) < lengths[:, None])
        decoder_batch_16 = torchaudio.functional.resample(unpadded, config.sample_rate, config.s3_sample_rate)
        lengths_16 = [math.ceil(config.s3_sample_rate * len(clip) / config.sample_rate) for clip in decoder_clips]
        decoder_clips_16 = [decoder_batch_16[index, :length] for index, length in enumerate(lengths_16)]
        prompt_tokens = self.tokenize_batch(decoder_clips_16, None)
        # CAMPPlus uses unmasked temporal statistics and attention. Equal
        # waveform lengths give equal fbank lengths and require no padding.
        groups: dict[int, list[int]] = {}
        for index, length in enumerate(lengths_16):
            groups.setdefault(length, []).append(index)
        order = [index for indices in groups.values() for index in indices]
        embeddings = torch.cat(
            [
                self.campplus(torch.cat([compute_fbank(decoder_clips_16[index], device) for index in indices]))
                for indices in groups.values()
            ]
        )
        embeddings = embeddings.float().cpu()[torch.tensor(order).argsort()]
        prompt_feats = prompt_feats.float().cpu()
        results = []
        for index, (clip, prompt_token) in enumerate(zip(decoder_clips, prompt_tokens, strict=True)):
            # Keep exactly token_mel_ratio frames per prompt token so that a
            # leftover prompt frame cannot overlap generated-token positions.
            frames = min(len(clip) // config.mel["hop_size"] // config.token_mel_ratio, prompt_token.shape[1])
            results.append(
                VoiceConditioning(
                    cond_tokens=cond_tokens[index],
                    speaker_emb=speaker_embs[index : index + 1].float().clone(),
                    prompt_token=prompt_token[:, :frames].clone(),
                    prompt_feat=prompt_feats[index : index + 1, : config.token_mel_ratio * frames].clone(),
                    embedding=embeddings[index : index + 1].clone(),
                )
            )
        return results

    def tokenize_batch(self, wavs16: list[torch.Tensor], max_tokens: int | None) -> list[torch.Tensor]:
        """Quantize a padded mel batch, retaining each clip's code length.

        The tokenizer's log mel normalizes by its clip's own maximum, so it is
        computed per clip, on the clip's device.

        Returns:
            Per clip, its codes of shape (1, T), long, on the host.
        """
        mels = [self.s3tokenizer.log_mel_spectrogram(wav.float()) for wav in wavs16]
        if max_tokens is not None:
            mels = [mel[..., : max_tokens * 4] for mel in mels]
        padded, mel_lens = self.s3tokenizer.padding(mels)
        codes, code_lens = self.tokenizer.quantize(padded.to(self.device), mel_lens.to(self.device))
        codes = codes.long().cpu()
        return [codes[index : index + 1, : int(length)].clone() for index, length in enumerate(code_lens.tolist())]


def build_prompt(
    text_ids: list[int],
    conditioning: VoiceConditioning,
    config: ChatterboxConfig,
    *,
    exaggeration: float | None = None,
    cfg_weight: float | None = None,
) -> dict:
    """The engine prompt for one utterance.

    Every prompt id is the start-of-speech token, whose logit is always
    ``-inf``: vLLM applies the repetition penalty to prompt ids too, and this
    keeps it off real speech tokens. Stage 0 replaces the placeholders with
    the variant's layout: ``[speaker | prompt tokens | text | start-of-speech]``
    for Turbo, ``[speaker | perceiver queries | emotion | text with BOS and EOS
    | two start-of-speech]`` for Original.

    Args:
        text_ids: Text tokenizer ids of ``punc_norm(text)``.
        conditioning: The reference clip's conditioning.
        config: The model config.
        exaggeration: Original's emotion control; Turbo has none. The
            serving default is the deploy file's.
        cfg_weight: Original's guidance weight, 0 for no guidance; Turbo
            has none. The serving default is the deploy file's.

    Returns:
        A token prompt carrying the conditioning in ``additional_information``.

    Raises:
        ValueError: For Original, if the text is longer than its position
            table or a control is missing or not a finite nonnegative number.
    """
    if config.variant == "original":
        if len(text_ids) > ORIGINAL_MAX_TEXT_TOKENS:
            raise ValueError(f"Chatterbox Original supports at most {ORIGINAL_MAX_TEXT_TOKENS} text tokens")
        for name, value in (("exaggeration", exaggeration), ("cfg_weight", cfg_weight)):
            if value is None:
                raise ValueError(f"Chatterbox Original prompts require {name}")
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
    length = (
        len(text_ids) + ORIGINAL_PROMPT_EXTRA_ROWS
        if config.variant == "original"
        else 1 + conditioning.cond_tokens.shape[1] + len(text_ids) + 1
    )
    prompt = tokens_input(prompt_token_ids=[config.start_speech_token] * length)
    prompt["additional_information"] = conditioning.additional_information(text_ids)
    if config.variant == "original":
        prompt["additional_information"]["chatterbox"] = {"exaggeration": exaggeration, "cfg_weight": cfg_weight}
        if cfg_weight:
            prompt["additional_information"]["cfg_group"] = {"role": "cond", "uncond_suffix": CFG_UNCOND_SUFFIX}
    return prompt
