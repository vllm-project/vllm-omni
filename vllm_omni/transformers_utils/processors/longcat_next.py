# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 Meituan.
"""LongCat-Next processor for image, audio, and text inputs.

Vendored from the checkpoint's own ``processing_longcat_next.py``
(meituan-longcat/LongCat-Next), trimmed to what vLLM-Omni's multimodal
processor actually calls: ``LongcatNextProcessor.from_pretrained(...)`` for
its ``image_processor``/``audio_processor``/``tokenizer`` sub-attributes, and
``LongcatNextAudioProcessor``'s feature-extraction methods. The checkpoint's
own ``__call__``/``process``/``load_audio_waveform`` methods are path-based
(read audio/image files from disk) and are not used here — the multimodal
processor in ``longcat_next_processor.py`` drives feature extraction
directly on in-memory data.
"""

import torch
from transformers import AutoFeatureExtractor
from transformers.audio_utils import mel_filter_bank
from transformers.configuration_utils import PretrainedConfig
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.processing_utils import ProcessorMixin


class LongcatNextAudioProcessor(FeatureExtractionMixin):
    """Fbank feature extraction for LongCat-Next's audio encoder.

    Config fields (``n_fft``, ``num_mel_bins``, ``sampling_rate``,
    ``max_audio_seconds``, ``hop_length``, ``kernel_size``, ``stride_size``,
    ``split_overlap``, ``avg_pooler``) are loaded from the checkpoint's
    ``preprocessor_config.json`` via ``FeatureExtractionMixin.__init__``.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + self.n_fft // 2,
            num_mel_filters=self.num_mel_bins,
            min_frequency=0.0,
            max_frequency=self.sampling_rate / 2.0,
            sampling_rate=self.sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        self.window = torch.hann_window(self.n_fft)

    def split_with_overlap(self, waveform: torch.Tensor) -> list[torch.Tensor]:
        channels, wave_samples = waveform.shape
        max_audio_samples = self.max_audio_seconds * self.sampling_rate
        if wave_samples <= max_audio_samples or self.split_overlap < 0:
            return [waveform]

        split_waveform, start = [], 0
        while start < wave_samples:
            if start > int(self.sampling_rate * self.split_overlap):
                start -= int(self.sampling_rate * self.split_overlap)
            end = min(start + max_audio_samples, wave_samples)
            if end - start >= self.n_fft:
                split_waveform.append(waveform[:, start:end])
            start = end
        return split_waveform

    @classmethod
    def inference_output_length(
        cls, input_length: int, kernel_size: int, stride_size: int, avg_pooler: int
    ) -> tuple[int, int]:
        encoder_length = (input_length + 2 * (kernel_size // 2) - kernel_size) // 1 + 1
        encoder_length = (encoder_length + 2 * (kernel_size // 2) - kernel_size) // stride_size + 1
        bridge_length = encoder_length // avg_pooler if avg_pooler > 1 else encoder_length
        return encoder_length, bridge_length

    def extract_fbank_features(self, waveform: torch.Tensor):
        channels, wave_samples = waveform.shape
        assert wave_samples >= self.n_fft
        valid_frame_nums = min(
            self.max_audio_seconds * self.sampling_rate // self.hop_length,
            wave_samples // self.hop_length + 1,
        )
        if wave_samples < self.max_audio_seconds * self.sampling_rate:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.max_audio_seconds * self.sampling_rate - wave_samples)
            )
        else:
            waveform = waveform[:, : self.max_audio_seconds * self.sampling_rate]

        stft = torch.stft(waveform, self.n_fft, self.hop_length, window=self.window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2

        mel_filters = torch.from_numpy(self.mel_filters).type(torch.float32)
        mel_spec = mel_filters.T @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        if waveform.dim() == 2:
            max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
            log_spec = torch.maximum(log_spec, max_val - 8.0)
        else:
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        log_spec = log_spec[0].numpy()
        log_spec[:, valid_frame_nums:] = 0.0
        return log_spec, valid_frame_nums


class LongcatNextAudioProcessorConfig(PretrainedConfig):
    pass


# ProcessorMixin.from_pretrained resolves audio_processor_class="LongcatNextAudioProcessor"
# by searching AutoFeatureExtractor's registry for a matching class name (not present in
# transformers by default, since this is a checkpoint-specific extractor); register it so
# that lookup succeeds instead of raising ValueError.
AutoFeatureExtractor.register(LongcatNextAudioProcessorConfig, LongcatNextAudioProcessor)


class LongcatNextProcessor(ProcessorMixin):
    """Wraps the image/audio processors and tokenizer for LongCat-Next.

    Special-token strings/ids are fixed by the checkpoint and read directly
    off the tokenizer rather than via ``tokenizer.init_kwargs`` (which moved
    to a nested ``model_specific_special_tokens`` dict in newer transformers
    versions) — callers needing the ids should resolve them from the
    tokenizer's vocab instead of a processor attribute.
    """

    attributes = ["image_processor", "audio_processor", "tokenizer"]
    image_processor_class = "Qwen2VLImageProcessor"
    audio_processor_class = "LongcatNextAudioProcessor"
    tokenizer_class = "AutoTokenizer"
