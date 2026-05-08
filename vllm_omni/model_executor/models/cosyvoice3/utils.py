# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
from functools import cache, lru_cache

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from vllm_omni.utils.audio import mel_filter_bank

logger = logging.getLogger(__name__)

IGNORE_ID = -1


def dynamic_range_compression_torch(x, c=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * c)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


@lru_cache
def _get_mel_basis(
    sampling_rate: int,
    n_fft: int,
    num_mels: int,
    fmin: float,
    fmax: float | None,
    device_str: str,
) -> torch.Tensor:
    return mel_filter_bank(
        sr=sampling_rate,
        n_fft=n_fft,
        n_mels=num_mels,
        fmin=fmin,
        fmax=fmax,
    ).to(torch.device(device_str))


@lru_cache
def _get_hann_window(win_size: int, device_str: str) -> torch.Tensor:
    return torch.hann_window(win_size).to(torch.device(device_str))


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    device_str = str(y.device)
    mel = _get_mel_basis(
        int(sampling_rate),
        int(n_fft),
        int(num_mels),
        float(fmin),
        fmax,
        device_str,
    )
    window = _get_hann_window(int(win_size), device_str)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=window,
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel, spec)
    spec = spectral_normalize_torch(spec)

    return spec


def load_wav(wav, target_sr, min_sr=16000):
    if not isinstance(wav, tuple):
        speech, sample_rate = torchaudio.load(wav, backend="soundfile")
    else:
        speech, sample_rate = wav
        if isinstance(speech, np.ndarray):
            speech = torch.from_numpy(speech).float().unsqueeze(0)

    if sample_rate != target_sr:
        if sample_rate < min_sr:
            raise ValueError(
                f"Audio sample rate {sample_rate} Hz is too low. "
                f"Minimum required: {min_sr} Hz, target: {target_sr} Hz. "
                f"Please provide audio with sample rate >= {min_sr} Hz."
            )
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)

    speech = speech.to(dtype=torch.float32)
    return speech


def extract_speech_feat(prompt_wav, feat_extractor, device):
    speech = load_wav(prompt_wav, 24000)
    speech_feat = feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(device)
    speech_feat = speech_feat.unsqueeze(dim=0)
    speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(device)
    return speech_feat, speech_feat_len


# Adopted from https://github.com/openai/whisper/blob/main/whisper/audio.py


def exact_div(x, y):
    assert x % y == 0
    return x // y


@cache
def mel_filters(device, n_mels: int) -> torch.Tensor:
    """Compute mel filterbank matrix for projecting STFT into a Mel spectrogram."""
    return mel_filter_bank(sr=16000, n_fft=400, n_mels=n_mels).to(device)


def log_mel_spectrogram(
    audio: str | np.ndarray | torch.Tensor,
    n_mels: int = 80,
    padding: int = 0,
    device: str | torch.device | None = None,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 and 128 are supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (n_mels, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    N_FFT = 400
    HOP_LENGTH = 160

    if not torch.is_tensor(audio):
        raise TypeError(f"audio is not tensor {type(audio)}")

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def extract_speech_token(prompt_wav, speech_tokenizer_session, device):
    speech = load_wav(prompt_wav, 16000)
    assert speech.shape[1] / 16000 <= 30, "do not support extract speech token for audio longer than 30s"
    feat = log_mel_spectrogram(speech, n_mels=128)
    speech_token = (
        speech_tokenizer_session.run(
            None,
            {
                speech_tokenizer_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                speech_tokenizer_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32),
            },
        )[0]
        .flatten()
        .tolist()
    )
    speech_token = torch.tensor([speech_token], dtype=torch.int32).to(device)
    speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(device)
    return speech_token, speech_token_len


def extract_spk_embedding(prompt_wav, campplus_session, device):
    speech = load_wav(prompt_wav, 16000)
    feat = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    embedding = (
        campplus_session.run(None, {campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0]
        .flatten()
        .tolist()
    )
    embedding = torch.tensor([embedding]).to(device)
    return embedding


def extract_text_token(text, tokenizer, allowed_special):
    text_token = tokenizer.encode(text, allowed_special=allowed_special)
    text_token = torch.tensor([text_token], dtype=torch.int32)
    text_token_len = text_token.shape[1]
    return text_token, text_token_len


def concat_text_with_prompt_ids(
    text: torch.Tensor,
    text_len: torch.Tensor,
    prompt_text: torch.Tensor,
    prompt_text_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    text = torch.concat([prompt_text, text], dim=1)
    text_len = text_len + prompt_text_len
    return text, text_len


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask
