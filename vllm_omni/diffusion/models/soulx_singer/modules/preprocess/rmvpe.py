"""RMVPE F0 extractor.

All model code is inlined (originally from RickyL-2000/ROSVOT
``modules/pe/rmvpe/``) with two minor substitutions:

* ``librosa.filters.mel``  → ``torchaudio.functional.melscale_fbanks``
  (already used elsewhere in vllm-omni);
* ``librosa.sequence.viterbi`` → hand-written numpy viterbi (non-default path).
* ``pyworld`` (deprecated) → removed.
"""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.functional import melscale_fbanks
from torchaudio.transforms import Resample
from vllm.logger import init_logger

from vllm_omni.diffusion.models.interface import SupportAudioInput, SupportsComponentDiscovery

logger = init_logger(__name__)

RMVPE_SAMPLE_RATE = 16000
_N_CLASS = 360
_N_MELS = 128
_MEL_FMIN = 30
_MEL_FMAX = 8000
_WINDOW_LENGTH = 1024
_CONST = 1997.3794084376191


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, momentum: float = 0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))
            self._has_shortcut = True
        else:
            self._has_shortcut = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._has_shortcut:
            return self.conv(x) + self.shortcut(x)
        return self.conv(x) + x


class ResEncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, n_blocks: int = 1, momentum: float = 0.01):
        super().__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        for i in range(self.n_blocks):
            x = self.conv[i](x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride, n_blocks: int = 1, momentum: float = 0.01):
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=out_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x: torch.Tensor, concat_tensor: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for i in range(self.n_blocks):
            x = self.conv2[i](x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_size: int,
        n_encoders: int,
        kernel_size,
        n_blocks: int,
        out_channels: int = 16,
        momentum: float = 0.01,
    ):
        super().__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        self.latent_channels: list[list[int]] = []
        for _ in range(self.n_encoders):
            self.layers.append(ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum=momentum))
            self.latent_channels.append([out_channels, in_size])
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        concat_tensors: list[torch.Tensor] = []
        x = self.bn(x)
        for i in range(self.n_encoders):
            _c, x = self.layers[i](x)
            concat_tensors.append(_c)
        return x, concat_tensors


class Intermediate(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_inters: int, n_blocks: int, momentum: float = 0.01):
        super().__init__()
        self.n_inters = n_inters
        self.layers = nn.ModuleList()
        self.layers.append(ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum))
        for _ in range(n_inters - 1):
            self.layers.append(ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.n_inters):
            x = self.layers[i](x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels: int, n_decoders: int, stride, n_blocks: int, momentum: float = 0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        self.n_decoders = n_decoders
        for _ in range(self.n_decoders):
            out_channels = in_channels // 2
            self.layers.append(ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum))
            in_channels = out_channels

    def forward(self, x: torch.Tensor, concat_tensors: list[torch.Tensor]) -> torch.Tensor:
        for i in range(self.n_decoders):
            x = self.layers[i](x, concat_tensors[-1 - i])
        return x


class TimbreFilter(nn.Module):
    def __init__(self, latent_rep_channels: list[list[int]]):
        super().__init__()
        self.layers = nn.ModuleList()
        for ch in latent_rep_channels:
            self.layers.append(ConvBlockRes(ch[0], ch[0]))

    def forward(self, x_tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        return [layer(t) for layer, t in zip(self.layers, x_tensors)]


class DeepUnet0(nn.Module):
    def __init__(
        self,
        kernel_size,
        n_blocks: int,
        en_de_layers: int = 5,
        inter_layers: int = 4,
        in_channels: int = 1,
        en_out_channels: int = 16,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, _N_MELS, en_de_layers, kernel_size, n_blocks, en_out_channels)
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks
        )
        self.tf = TimbreFilter(self.encoder.latent_channels)
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class BiGRU(nn.Module):
    def __init__(self, input_features: int, hidden_features: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gru(x)[0]


class E2E0(nn.Module):
    """E2E0 pitch estimator network (DeepUnet0 + CNN + BiGRU + classifier)."""

    def __init__(
        self,
        n_blocks: int,
        n_gru: int,
        kernel_size,
        en_de_layers: int = 5,
        inter_layers: int = 4,
        in_channels: int = 1,
        en_out_channels: int = 16,
    ):
        super().__init__()
        self.unet = DeepUnet0(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * _N_MELS, 256, n_gru),
                nn.Linear(512, _N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * _N_MELS, _N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class MelSpectrogram(nn.Module):
    """Mel-spectrogram extractor matching ROSVOT's MelSpectrogram.

    Log-mel spectrogram via torch.stft + adaptive filterbank.
    """

    def __init__(
        self,
        n_mel_channels: int = _N_MELS,
        sampling_rate: int = RMVPE_SAMPLE_RATE,
        win_length: int = _WINDOW_LENGTH,
        hop_length: int = 160,
        n_fft: int | None = None,
        mel_fmin: float = _MEL_FMIN,
        mel_fmax: float = _MEL_FMAX,
        clamp: float = 1e-5,
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hann_window: dict[str, torch.Tensor] = {}
        mel_basis = melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=float(mel_fmin),
            f_max=float(mel_fmax) if mel_fmax is not None else 8000.0,
            n_mels=n_mel_channels,
            sample_rate=int(sampling_rate),
            norm="slaney",
            mel_scale="htk",
        ).T  # (n_freqs, n_mels) → (n_mels, n_freqs) to match librosa/layernorm convention
        self.register_buffer("mel_basis", mel_basis)  # (n_mels, n_freqs)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp

    def forward(self, audio: torch.Tensor, keyshift: float = 0, speed: float = 1, center: bool = True) -> torch.Tensor:
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))

        key = str(keyshift) + "_" + str(audio.device)
        if key not in self.hann_window:
            self.hann_window[key] = torch.hann_window(win_length_new).to(audio.device)

        if center:
            pad_left = win_length_new // 2
            pad_right = (win_length_new + 1) // 2
            audio = F.pad(audio, (pad_left, pad_right))

        fft = torch.stft(
            audio,
            n_fft=n_fft_new,
            hop_length=hop_length_new,
            win_length=win_length_new,
            window=self.hann_window[key],
            center=False,
            return_complex=True,
        )
        magnitude = fft.abs()

        if keyshift != 0:
            size = self.n_fft // 2 + 1
            if magnitude.size(1) < size:
                magnitude = F.pad(magnitude, (0, 0, 0, size - magnitude.size(1)))
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new

        # register_buffer confuses type checkers; cast to Tensor explicitly
        mel_basis: torch.Tensor = self.mel_basis  # type: ignore[assignment]
        mel_output = torch.matmul(mel_basis, magnitude)
        log_mel = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel


def to_local_average_f0(hidden: torch.Tensor, center: torch.Tensor | None = None, thread: float = 0.03) -> np.ndarray:
    """Decode hidden representation to F0 via local weighted average."""
    idx = torch.arange(_N_CLASS, device=hidden.device)[None, None, :]
    idx_cents = idx * 20 + _CONST
    if center is None:
        center = torch.argmax(hidden, dim=2, keepdim=True)
    start = torch.clip(center - 4, min=0)
    end = torch.clip(center + 5, max=_N_CLASS)
    idx_mask = (idx >= start) & (idx < end)
    weights = hidden * idx_mask
    product_sum = torch.sum(weights * idx_cents, dim=2)
    weight_sum = torch.sum(weights, dim=2)
    cents = product_sum / (weight_sum + (weight_sum == 0))
    f0 = 10 * 2 ** (cents / 1200)
    uv = hidden.max(dim=2)[0] < thread
    f0 = f0 * ~uv
    return f0.cpu().numpy()


def numpy_viterbi(prob: np.ndarray, transition: np.ndarray) -> np.ndarray:
    """Viterbi decoding on log-probabilities (numpy).

    Args:
        prob: Emission probabilities, shape (n_states, n_obs).
        transition: Transition matrix, shape (n_states, n_states), row-stochastic.

    Returns:
        path: Most likely state sequence, shape (n_obs,).
    """
    n_states, n_obs = prob.shape
    log_emit = np.log(prob + 1e-12)
    log_trans = np.log(transition + 1e-12)

    delta = log_emit[:, 0] + np.zeros(n_states)  # initial: log(pi) omitted (uniform)
    psi = np.zeros((n_obs, n_states), dtype=np.int64)

    for t in range(1, n_obs):
        scores = delta[:, None] + log_trans  # (n_states, n_states)
        psi[t] = np.argmax(scores, axis=0)
        delta = scores[psi[t], np.arange(n_states)] + log_emit[:, t]

    # Backtrack
    path = np.zeros(n_obs, dtype=np.int64)
    path[-1] = np.argmax(delta)
    for t in range(n_obs - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


def to_viterbi_f0(hidden: torch.Tensor, thread: float = 0.03) -> np.ndarray:
    """Decode hidden representation to F0 via Viterbi.

    Uses a hand-written numpy Viterbi decoder (replacing
    ``librosa.sequence.viterbi``) to avoid the librosa dependency.
    """
    if not hasattr(to_viterbi_f0, "_transition"):
        xx, yy = np.meshgrid(range(_N_CLASS), range(_N_CLASS))
        transition = np.maximum(30 - abs(xx - yy), 0)
        transition = transition / transition.sum(axis=1, keepdims=True)
        to_viterbi_f0._transition = transition  # type: ignore[attr-defined]

    prob = hidden.squeeze(0).cpu().numpy()
    prob = prob.T
    prob = prob / prob.sum(axis=0, keepdims=True)

    path = numpy_viterbi(prob, to_viterbi_f0._transition)  # type: ignore[attr-defined]
    center = torch.from_numpy(path).unsqueeze(0).unsqueeze(-1).to(hidden.device)
    return to_local_average_f0(hidden, center=center, thread=thread)


def resample_align_curve(
    points: np.ndarray, original_timestep: float, target_timestep: float, align_length: int = -1
) -> np.ndarray:
    """Align a curve from one time grid to another via linear interpolation."""
    t_max = (len(points) - 1) * original_timestep
    curve_interp = np.interp(
        np.arange(0, t_max, target_timestep),
        original_timestep * np.arange(len(points)),
        points,
    ).astype(points.dtype)
    if align_length > 0:
        if len(curve_interp) > align_length:
            curve_interp = curve_interp[:align_length]
        elif len(curve_interp) < align_length:
            curve_interp = np.concatenate(
                [curve_interp, np.full(align_length - len(curve_interp), fill_value=curve_interp[-1])],
                axis=0,
            )
    return curve_interp


class RMVPE(nn.Module, SupportAudioInput, SupportsComponentDiscovery):
    """RMVPE F0 extractor (SoulX-Singer).

    Wraps the E2E0 model with mel-spectrogram front-end and decoding
    post-processing.
    """

    support_audio_input: ClassVar[bool] = True
    _dit_modules: ClassVar[list[str]] = []
    _encoder_modules: ClassVar[list[str]] = []
    _vae_modules: ClassVar[list[str]] = []
    _resident_modules: ClassVar[list[str]] = ["."]
    _layerwise_offload_blocks_attrs: ClassVar[list[str]] = []

    def __init__(self, model_path: str | None = None, hop_length: int = 160, device: str | None = None):
        super().__init__()
        self.resample_kernel: dict[str, Resample] = {}
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = E2E0(4, 1, (2, 2)).eval()
        if model_path is not None:
            ckpt = torch.load(model_path, map_location=self.device)
            state_dict = ckpt if not isinstance(ckpt, dict) else ckpt.get("model", ckpt)
            self.model.load_state_dict(state_dict, strict=False)
            logger.info("Loaded RMVPE weights from %s", model_path)
        self.model = self.model.to(self.device)

        self.mel_extractor = MelSpectrogram(
            _N_MELS,
            RMVPE_SAMPLE_RATE,
            _WINDOW_LENGTH,
            hop_length,
            None,
            _MEL_FMIN,
            _MEL_FMAX,
        ).to(self.device)
        self.hop_length = hop_length

    # ---- internal helpers ----

    @torch.no_grad()
    def mel2hidden(self, mel: torch.Tensor) -> torch.Tensor:
        n_frames = mel.shape[-1]
        pad_len = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        mel = F.pad(mel, (0, pad_len), mode="constant")
        hidden = self.model(mel)
        return hidden[:, :n_frames]

    def decode(self, hidden: torch.Tensor, thread: float = 0.03, use_viterbi: bool = False) -> np.ndarray:
        if use_viterbi:
            return to_viterbi_f0(hidden, thread=thread)
        return to_local_average_f0(hidden, thread=thread)

    def postprocess(self, f0: np.ndarray, fmin: float = 50, fmax: float = 1000, min_gap: int = 2) -> np.ndarray:
        f0[f0 < fmin] = 0
        f0[f0 > fmax] = 0
        # Eliminate glitch: if positive f0 values span fewer than min_gap frames
        # between zeros, zero them out.
        for idx in range(f0.shape[0] - min_gap - 1):
            if f0[idx] == 0 and f0[idx + min_gap + 1] == 0 and np.sum(f0[idx : idx + min_gap + 2]) > 0:
                f0[idx : idx + min_gap + 2] = 0
        return f0

    # ---- single audio inference ----

    def infer_from_audio(
        self, audio: np.ndarray, sample_rate: int = 16000, thread: float = 0.03, use_viterbi: bool = False
    ) -> np.ndarray:
        audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        if sample_rate == RMVPE_SAMPLE_RATE:
            audio_res = audio_t
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, RMVPE_SAMPLE_RATE, lowpass_filter_width=128)
            self.resample_kernel[key_str] = self.resample_kernel[key_str].to(self.device)
            audio_res = self.resample_kernel[key_str](audio_t)
        mel = self.mel_extractor(audio_res, center=True)
        hidden = self.mel2hidden(mel)
        f0 = self.decode(hidden, thread=thread, use_viterbi=use_viterbi).squeeze(0)
        return f0

    def get_pitch(
        self,
        waveform,
        sample_rate: int,
        hop_size: int,
        length: int,
        interp_uv: bool = False,
        fmin: float = 50,
        fmax: float = 1000,
    ) -> tuple[np.ndarray, np.ndarray]:
        f0 = self.infer_from_audio(waveform, sample_rate=sample_rate)
        f0 = self.postprocess(f0, fmin, fmax)
        uv = f0 == 0
        time_step = hop_size / sample_rate
        f0_res = resample_align_curve(f0, 0.01, time_step, length)
        uv_res = resample_align_curve(uv.astype(np.float32), 0.01, time_step, length) > 0.5
        if not interp_uv:
            f0_res[uv_res] = 0
        return f0_res, uv_res

    # ---- batch audio inference ----

    def infer_from_audio_batch(
        self, audios, sample_rate: int = 16000, thread: float = 0.03, use_viterbi: bool = False
    ) -> list[np.ndarray]:
        sizes: list[int] | None = None
        if isinstance(audios, list):
            audios = [torch.from_numpy(a).float() for a in audios]
            sizes = [math.ceil((a.shape[0] + 1) / self.hop_length) for a in audios]
            audios = nn.utils.rnn.pad_sequence(audios, batch_first=True, padding_value=0.0).to(self.device)
        elif isinstance(audios, torch.Tensor):
            if audios.device != self.device:
                audios = audios.to(self.device)
        else:
            raise NotImplementedError(f"Unsupported audio type: {type(audios)}")

        if sample_rate == RMVPE_SAMPLE_RATE:
            audios_res = audios
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, RMVPE_SAMPLE_RATE, lowpass_filter_width=128)
            self.resample_kernel[key_str] = self.resample_kernel[key_str].to(self.device)
            audios_res = self.resample_kernel[key_str](audios)

        mels = self.mel_extractor(audios_res, center=True)
        hiddens = self.mel2hidden(mels)
        f0 = self.decode(hiddens, thread=thread, use_viterbi=use_viterbi)

        f0s: list[np.ndarray] = []
        for i in range(f0.shape[0]):
            f0s.append(f0[i, : sizes[i]] if sizes is not None else f0[i, :])
        return f0s

    def get_pitch_batch(
        self,
        waveforms,
        sample_rate: int,
        hop_size: int,
        lengths: list[int],
        interp_uv: bool = False,
        fmin: float = 50,
        fmax: float = 1000,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        f0s = self.infer_from_audio_batch(waveforms, sample_rate=sample_rate)
        f0s_res: list[np.ndarray] = []
        uvs_res: list[np.ndarray] = []
        for idx, f0 in enumerate(f0s):
            f0 = self.postprocess(f0, fmin, fmax, min_gap=6)
            uv = f0 == 0
            length = lengths[idx]
            time_step = hop_size / sample_rate
            f0_res = resample_align_curve(f0, 0.01, time_step, length)
            uv_res = resample_align_curve(uv.astype(np.float32), 0.01, time_step, length) > 0.5
            if not interp_uv:
                f0_res[uv_res] = 0
            f0s_res.append(f0_res)
            uvs_res.append(uv_res)
        return f0s_res, uvs_res

    def release_cuda(self):
        self.model = self.model.cpu()
        self.mel_extractor = self.mel_extractor.cpu()
