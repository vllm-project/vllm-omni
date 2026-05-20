# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FunCineForge utility functions.

Audio processing, feature extraction, speaker embedding, and text tokenization
helpers.  Mirrors the CosyVoice3 utils pattern but adapted for FunCineForge's
24 kHz / 25 Hz token rate / 50 Hz mel rate pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from vllm_omni.model_executor.models.cosyvoice3.utils import (
    extract_speech_token,
    extract_spk_embedding,
    log_mel_spectrogram,
)
from vllm_omni.utils.audio import mel_filter_bank

__all__ = [
    "extract_speech_token",
    "extract_spk_embedding",
    "log_mel_spectrogram",
]


def mel_spectrogram(
    wav: np.ndarray | torch.Tensor,
    *,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    sampling_rate: int = 24000,
    n_mel_channels: int = 80,
    mel_fmin: float = 0.0,
    mel_fmax: float | None = None,
    center: bool = False,
) -> torch.Tensor:
    if isinstance(wav, np.ndarray):
        wav = torch.from_numpy(wav).float()
    if not isinstance(wav, torch.Tensor):
        wav = torch.tensor(wav, dtype=torch.float32)
    if wav.ndim > 1:
        wav = wav.mean(dim=-1)

    mel_basis = mel_filter_bank(
        sr=sampling_rate,
        n_fft=n_fft,
        n_mels=n_mel_channels,
        fmin=mel_fmin,
        fmax=mel_fmax,
    ).to(wav.device)

    window = torch.hann_window(win_length, device=wav.device)
    pad_size = (n_fft - hop_length) // 2
    wav_padded = torch.nn.functional.pad(wav.unsqueeze(0), (pad_size, pad_size), mode="reflect").squeeze(0)
    spec = torch.stft(
        wav_padded,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    magnitudes = spec.abs()
    mel = mel_basis @ magnitudes
    mel = torch.log(torch.clamp(mel, min=1e-5))
    return mel.unsqueeze(0)


# ---------------------------------------------------------------------------
# Speech feature extraction (mel)
# ---------------------------------------------------------------------------


def extract_speech_feat(
    audio: tuple[Any, int],
    feat_extractor: Any,
    device: str = "cpu",
    target_sr: int = 24000,
) -> tuple[torch.Tensor, int]:
    """Extract mel features from audio.

    Args:
        audio: (wav_samples_or_np, sample_rate) tuple.
        feat_extractor: mel_spectrogram function or callable.
        target_sr: Expected sample rate for mel extraction.

    Returns:
        (mel_tensor, feat_len) where tensor shape is (1, n_mels, T).
    """
    wav, sr = audio
    if isinstance(wav, list):
        wav = np.asarray(wav, dtype=np.float32)
    if isinstance(wav, torch.Tensor):
        wav = wav.cpu().numpy()
    wav = np.asarray(wav, dtype=np.float32)

    if sr != target_sr:
        import torchaudio

        wav_t = torch.from_numpy(wav).float()
        if wav_t.ndim == 1:
            wav_t = wav_t.unsqueeze(0)
        wav_t = torchaudio.functional.resample(wav_t, sr, target_sr)
        wav = wav_t.squeeze(0).numpy()

    if callable(feat_extractor):
        feat = feat_extractor(wav)
    else:
        feat = mel_spectrogram(wav)

    feat_len = feat.shape[-1]
    return feat, feat_len


# ---------------------------------------------------------------------------
# Text token extraction
# ---------------------------------------------------------------------------


def extract_text_token(
    text: str,
    tokenizer: Any,
    allowed_special: set[str] | None = None,
) -> tuple[torch.Tensor, int]:
    """Tokenize text using the FunCineForge tokenizer.

    Args:
        text: Input text string.
        tokenizer: Tokenizer with encode() method.
        allowed_special: Set of special tokens to allow (unused for now).

    Returns:
        (token_ids_tensor, token_len) where tensor shape is (1, T).
    """
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(text)
    else:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    token_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    token_len = len(ids)
    return token_ids, token_len


# ---------------------------------------------------------------------------
# Concat text with prompt IDs
# ---------------------------------------------------------------------------


def concat_text_with_prompt_ids(
    text_token: torch.Tensor,
    text_token_len: int,
    prompt_text_token: torch.Tensor,
    prompt_text_token_len: int,
    *,
    sos: int = 6561,
    turn_of_speech: int = 6563,
    type_id: int = 1502,
    timespk_ids: list[int] | None = None,
    startofclue_token: int = 151646,
    endofclue_token: int = 151647,
    lm_use_prompt: bool = True,
) -> tuple[torch.Tensor, int]:
    """Build the full FunCineForge LM input sequence.

    The sequence layout (from FunCineForge's ``load_data``) is::

        [SOS, startofclue, clue_ids, endofclue, text_ids, type_id, timespk_ids..., turn_of_speech]

    When ``lm_use_prompt`` is True the clue (prompt_text) is prepended to
    the text.  When False, only the text is used.

    Args:
        text_token: Text token IDs, shape (1, T_text).
        text_token_len: Valid length of text_token.
        prompt_text_token: Clue/prompt token IDs, shape (1, T_clue).
        prompt_text_token_len: Valid length of prompt_text_token.
        sos: SOS token ID (default 6561).
        turn_of_speech: Turn-of-speech token ID (default 6563).
        type_id: Speech type tag (default 1502 = duihua/对话).
        timespk_ids: Optional list of timespeaker tag IDs (e.g. speaker IDs).
        startofclue_token: Special token wrapping the clue.
        endofclue_token: Special token wrapping the clue.
        lm_use_prompt: Whether to prepend clue to text.

    Returns:
        (input_ids, total_len) where input_ids shape is (1, T).
    """
    parts: list[torch.Tensor] = []

    # SOS
    parts.append(torch.tensor([sos], dtype=torch.long))

    # Clue (prompt text) wrapped with startofclue / endofclue
    if lm_use_prompt and prompt_text_token_len > 0:
        parts.append(torch.tensor([startofclue_token], dtype=torch.long))
        parts.append(prompt_text_token[0, :prompt_text_token_len])
        parts.append(torch.tensor([endofclue_token], dtype=torch.long))

    # Text
    parts.append(text_token[0, :text_token_len])

    # Type ID (speech type tag: duihua, pangbai, dubai, etc.)
    parts.append(torch.tensor([type_id], dtype=torch.long))

    # Timespeaker IDs (speaker tags, gender/age, etc.)
    if timespk_ids is not None:
        parts.append(torch.tensor(timespk_ids, dtype=torch.long))

    # Turn of speech (signals start of codec token generation)
    parts.append(torch.tensor([turn_of_speech], dtype=torch.long))

    concat = torch.cat(parts, dim=0).unsqueeze(0)
    total_len = concat.shape[1]
    return concat, total_len


def speech_type_to_id(speech_type: str | None, config: Any) -> int:
    type_map = {
        "旁白": config.pangbai,
        "独白": config.dubai,
        "对话": config.duihua,
        "多人": config.duoren,
    }
    return int(type_map.get(speech_type or "", config.pangbai))


def dialogue_to_timespk_ids(dialogue: list[dict[str, Any]] | None, config: Any) -> list[int]:
    """Convert FunCineForge demo dialogue metadata to timespeaker token IDs."""
    if not dialogue:
        return []

    gender_map = {
        "男": config.male,
        "male": config.male,
        "女": config.female,
        "female": config.female,
    }
    age_map = {
        "儿童": config.child,
        "child": config.child,
        "青年": config.youth,
        "teenager": config.youth,
        "中年": config.adult,
        "adult": config.adult,
        "中老年": config.middle,
        "middle-aged": config.middle,
        "老年": config.elderly,
        "elderly": config.elderly,
    }

    ids: list[int] = []
    for part in dialogue:
        start = float(part.get("start", 0.0))
        duration = float(part.get("duration", 0.0))
        speaker = int(part.get("spk", 1))
        ids.extend(
            [
                int(start * 25 + 1),
                int(config.speaker_id_start + speaker - 1),
                int(gender_map.get(part.get("gender"), config.male)),
                int(age_map.get(part.get("age"), config.adult)),
                int((start + duration) * 25 + 1),
            ]
        )
    return ids


# ---------------------------------------------------------------------------
# Face embedding loading
# ---------------------------------------------------------------------------


def load_face_embedding(face_path: str, speech_len: int, face_size: int = 512) -> torch.Tensor:
    """Load face embeddings from a ``.npz`` file.

    Args:
        face_path: Path to npz file with ``embeddings`` and ``faceI`` arrays.
        speech_len: Length of the speech sequence (for zero-padding).
        face_size: Dimension of each face embedding vector.

    Returns:
        Tensor of shape (1, speech_len, face_size).
    """
    face_embs = torch.zeros((speech_len, face_size), dtype=torch.float32)
    data = np.load(face_path, allow_pickle=False)
    embeddings = data["embeddings"]
    face_indices = data["faceI"]
    for emb, frame_idx in zip(embeddings, face_indices):
        fi = int(frame_idx)
        if 0 <= fi < speech_len:
            end = min(fi + 5, speech_len)
            face_embs[fi:end] = torch.from_numpy(np.asarray(emb)).expand(end - fi, -1)
    return face_embs.unsqueeze(0)
