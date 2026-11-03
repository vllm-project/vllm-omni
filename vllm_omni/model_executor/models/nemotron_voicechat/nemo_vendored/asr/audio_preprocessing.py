# SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Vendored from NVIDIA-NeMo/Speech (branch nemotron-labs-voicechat) nemo/collections/asr/modules/audio_preprocessing.py; training-only code removed.
# Kept: AudioPreprocessor, AudioToMelSpectrogramPreprocessor (+ its config dataclass).
# Removed: AudioToMFCCPreprocessor, SpectrogramAugmentation, MaskedPatchAugmentation,
# CropOrPadSpectrogramAugmentation and numba/spec-augment imports (training-only).

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
from packaging import version

from ..compat import (
    Exportable,
    NeuralModule,
    logging,
    logging_mode,
    typecheck,
)
from .features import FilterbankFeatures, FilterbankFeaturesTA

try:
    import torchaudio
    import torchaudio.functional
    import torchaudio.transforms

    TORCHAUDIO_VERSION = version.parse(torchaudio.__version__)
    TORCHAUDIO_VERSION_MIN = version.parse("0.5")

    HAVE_TORCHAUDIO = True
except ModuleNotFoundError:
    HAVE_TORCHAUDIO = False

__all__ = [
    "AudioToMelSpectrogramPreprocessor",
]


class AudioPreprocessor(NeuralModule, ABC):
    """
    An interface for Neural Modules that performs audio pre-processing,
    transforming the wav files to features.
    """

    def __init__(self, win_length, hop_length):
        super().__init__()

        self.win_length = win_length
        self.hop_length = hop_length

        self.torch_windows = {
            "hann": torch.hann_window,
            "hamming": torch.hamming_window,
            "blackman": torch.blackman_window,
            "bartlett": torch.bartlett_window,
            "ones": torch.ones,
            None: torch.ones,
        }

        # Normally, when you call to(dtype) on a torch.nn.Module, all
        # floating point parameters and buffers will change to that
        # dtype, rather than being float32. The AudioPreprocessor
        # classes, uniquely, don't actually have any parameters or
        # buffers from what I see. In addition, we want the input to
        # the preprocessor to be float32, but need to create the
        # output in appropriate precision. We have this empty tensor
        # here just to detect which dtype tensor this module should
        # output at the end of execution.
        self.register_buffer("dtype_sentinel_tensor", torch.tensor((), dtype=torch.float32), persistent=False)

    @typecheck()
    @torch.no_grad()
    def forward(self, input_signal, length):
        if input_signal.dtype != torch.float32:
            logging.warning(
                f"AudioPreprocessor received an input signal of dtype {input_signal.dtype}, rather than torch.float32. In sweeps across multiple datasets, we have found that the preprocessor is not robust to low precision  mathematics. As such, it runs in float32. Your input will be cast to float32, but this is not necessarily enough to recovery full accuracy. For example, simply casting input_signal from torch.float32 to torch.bfloat16, then back to torch.float32 before running AudioPreprocessor causes drops in absolute WER of up to 0.1%. torch.bfloat16 simply does not have enough mantissa bits to represent enough values in the range [-1.0,+1.0] correctly.",
                mode=logging_mode.ONCE,
            )
        processed_signal, processed_length = self.get_features(input_signal.to(torch.float32), length)
        processed_signal = processed_signal.to(self.dtype_sentinel_tensor.dtype)
        return processed_signal, processed_length

    @abstractmethod
    def get_features(self, input_signal, length):
        # Called by forward(). Subclasses should implement this.
        pass


class AudioToMelSpectrogramPreprocessor(AudioPreprocessor, Exportable):
    """Featurizer module that converts wavs to mel spectrograms.

    Args:
        sample_rate (int): Sample rate of the input audio data.
            Defaults to 16000
        window_size (float): Size of window for fft in seconds
            Defaults to 0.02
        window_stride (float): Stride of window for fft in seconds
            Defaults to 0.01
        n_window_size (int): Size of window for fft in samples
            Defaults to None. Use one of window_size or n_window_size.
        n_window_stride (int): Stride of window for fft in samples
            Defaults to None. Use one of window_stride or n_window_stride.
        window (str): Windowing function for fft. can be one of ['hann',
            'hamming', 'blackman', 'bartlett']
            Defaults to "hann"
        normalize (str): Can be one of ['per_feature', 'all_features']; all
            other options disable feature normalization. 'all_features'
            normalizes the entire spectrogram to be mean 0 with std 1.
            'pre_features' normalizes per channel / freq instead.
            Defaults to "per_feature"
        n_fft (int): Length of FT window. If None, it uses the smallest power
            of 2 that is larger than n_window_size.
            Defaults to None
        preemph (float): Amount of pre emphasis to add to audio. Can be
            disabled by passing None.
            Defaults to 0.97
        features (int): Number of mel spectrogram freq bins to output.
            Defaults to 64
        lowfreq (int): Lower bound on mel basis in Hz.
            Defaults to 0
        highfreq  (int): Lower bound on mel basis in Hz.
            Defaults to None
        log (bool): Log features.
            Defaults to True
        log_zero_guard_type(str): Need to avoid taking the log of zero. There
            are two options: "add" or "clamp".
            Defaults to "add".
        log_zero_guard_value(float, or str): Add or clamp requires the number
            to add with or clamp to. log_zero_guard_value can either be a float
            or "tiny" or "eps". torch.finfo is used if "tiny" or "eps" is
            passed.
            Defaults to 2**-24.
        dither (float): Amount of white-noise dithering.
            Defaults to 1e-5
        pad_to (int): Ensures that the output size of the time dimension is
            a multiple of pad_to.
            Defaults to 16
        frame_splicing (int): Defaults to 1
        exact_pad (bool): If True, sets stft center to False and adds padding, such that num_frames = audio_length
            // hop_length. Defaults to False.
        pad_value (float): The value that shorter mels are padded with.
            Defaults to 0
        mag_power (float): The power that the linear spectrogram is raised to
            prior to multiplication with mel basis.
            Defaults to 2 for a power spec
        rng : Random number generator
        nb_augmentation_prob (float) : Probability with which narrowband augmentation would be applied to
            samples in the batch.
            Defaults to 0.0
        nb_max_freq (int) : Frequency above which all frequencies will be masked for narrowband augmentation.
            Defaults to 4000
        use_torchaudio: Whether to use the `torchaudio` implementation.
        mel_norm: Normalization used for mel filterbank weights.
            Defaults to 'slaney' (area normalization)
        stft_exact_pad: Deprecated argument, kept for compatibility with older checkpoints.
        stft_conv: Deprecated argument, kept for compatibility with older checkpoints.
    """

    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass

    def __init__(
        self,
        sample_rate=16000,
        window_size=0.02,
        window_stride=0.01,
        n_window_size=None,
        n_window_stride=None,
        window="hann",
        normalize="per_feature",
        n_fft=None,
        preemph=0.97,
        features=64,
        lowfreq=0,
        highfreq=None,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2**-24,
        dither=1e-5,
        pad_to=16,
        frame_splicing=1,
        exact_pad=False,
        pad_value=0,
        mag_power=2.0,
        rng=None,
        nb_augmentation_prob=0.0,
        nb_max_freq=4000,
        use_torchaudio: bool = False,
        mel_norm="slaney",
        stft_exact_pad=False,  # Deprecated arguments; kept for config compatibility
        stft_conv=False,  # Deprecated arguments; kept for config compatibility
    ):
        super().__init__(n_window_size, n_window_stride)

        self._sample_rate = sample_rate
        if window_size and n_window_size:
            raise ValueError(f"{self} received both window_size and n_window_size. Only one should be specified.")
        if window_stride and n_window_stride:
            raise ValueError(f"{self} received both window_stride and n_window_stride. Only one should be specified.")
        if window_size:
            n_window_size = int(window_size * self._sample_rate)
        if window_stride:
            n_window_stride = int(window_stride * self._sample_rate)

        # Given the long and similar argument list, point to the class and instantiate it by reference
        if not use_torchaudio:
            featurizer_class = FilterbankFeatures
        else:
            featurizer_class = FilterbankFeaturesTA
        self.featurizer = featurizer_class(
            sample_rate=self._sample_rate,
            n_window_size=n_window_size,
            n_window_stride=n_window_stride,
            window=window,
            normalize=normalize,
            n_fft=n_fft,
            preemph=preemph,
            nfilt=features,
            lowfreq=lowfreq,
            highfreq=highfreq,
            log=log,
            log_zero_guard_type=log_zero_guard_type,
            log_zero_guard_value=log_zero_guard_value,
            dither=dither,
            pad_to=pad_to,
            frame_splicing=frame_splicing,
            exact_pad=exact_pad,
            pad_value=pad_value,
            mag_power=mag_power,
            rng=rng,
            nb_augmentation_prob=nb_augmentation_prob,
            nb_max_freq=nb_max_freq,
            mel_norm=mel_norm,
            stft_exact_pad=stft_exact_pad,  # Deprecated arguments; kept for config compatibility
            stft_conv=stft_conv,  # Deprecated arguments; kept for config compatibility
        )

    def get_features(self, input_signal, length):
        return self.featurizer(input_signal, length)
