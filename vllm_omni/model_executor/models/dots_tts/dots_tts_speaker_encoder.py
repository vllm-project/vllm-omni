# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The vLLM-Omni team.
# Copyright (c) rednote-hilab. All rights reserved.
# Portions Copyright 3D-Speaker
# (https://github.com/alibaba-damo-academy/3D-Speaker), Apache-2.0.
# Adapted from:
# https://github.com/rednote-hilab/dots.tts/tree/a393d2e/src/dots_tts/modules/speaker
# (whose CAM++ implementation is itself adapted from 3D-Speaker).
#
# Merged from four upstream files (preserved as-is, inference behaviour
# unchanged; only internal cross-module imports were folded inline):
#   - modules/speaker/encoder.py         -> SpeakerXVectorFeatures
#   - modules/speaker/fbank.py           -> fbank constants + extract_speaker_fbank
#   - modules/speaker/campplus.py        -> FCM, CAMPPlus
#   - modules/speaker/campplus_layers.py -> get_nonlinear, statistics_pooling,
#                                           StatsPool, TDNNLayer, CAMLayer,
#                                           CAMDenseTDNNLayer, CAMDenseTDNNBlock,
#                                           TransitLayer, DenseLayer, BasicResBlock
#
# Plus two tiny helpers folded from utils/audio.py (used by
# extract_speaker_fbank):
#   - high_quality_resample, extract_fbank
#
# Nothing in this file diverges from upstream behaviour.

from __future__ import annotations

import math
import random
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import torchaudio
import torchaudio.compliance.kaldi as Kaldi
import torchaudio.functional as AF
from torch.nn.utils.rnn import pad_sequence

# ============================================================================
# Constants (from modules/speaker/fbank.py)
# ============================================================================

_SPEAKER_FBANK_SAMPLE_RATE = 16000
_SPEAKER_FBANK_N_MELS = 80
_SPEAKER_FBANK_MEAN_NORM = True
_SPEAKER_FBANK_DITHER = 0.0


# ============================================================================
# Audio helpers (folded from utils/audio.py — used by extract_speaker_fbank)
# ============================================================================


def high_quality_resample(x, orig_sr, target_sr):
    return AF.resample(
        x,
        orig_freq=orig_sr,
        new_freq=target_sr,
        lowpass_filter_width=64,
        rolloff=0.95,
        resampling_method="sinc_interp_kaiser",
    )


def extract_fbank(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    n_mels: int,
    dither: float = 0.0,
    mean_norm: bool = False,
) -> torch.Tensor:
    if waveform.ndim == 1:
        feature_input = waveform.unsqueeze(0)
    elif waveform.ndim == 2:
        feature_input = waveform if waveform.size(0) == 1 else waveform[0:1, :]
    else:
        raise ValueError(f"FBank expects a 1D or 2D waveform, got shape {tuple(waveform.shape)}.")
    features = Kaldi.fbank(
        feature_input,
        num_mel_bins=n_mels,
        sample_frequency=sample_rate,
        dither=dither,
    )
    if mean_norm:
        features = features - features.mean(dim=0, keepdim=True)
    return features


# ============================================================================
# Speaker fbank entry (from modules/speaker/fbank.py)
# ============================================================================


def extract_speaker_fbank(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
) -> torch.Tensor:
    feature_input = waveform
    if sample_rate != _SPEAKER_FBANK_SAMPLE_RATE:
        feature_input = high_quality_resample(
            waveform,
            orig_sr=sample_rate,
            target_sr=_SPEAKER_FBANK_SAMPLE_RATE,
        )
    return extract_fbank(
        feature_input,
        sample_rate=_SPEAKER_FBANK_SAMPLE_RATE,
        n_mels=_SPEAKER_FBANK_N_MELS,
        dither=_SPEAKER_FBANK_DITHER,
        mean_norm=_SPEAKER_FBANK_MEAN_NORM,
    )


# ============================================================================
# CAM++ building blocks (from modules/speaker/campplus_layers.py)
# ----------------------------------------------------------------------------
# 3D-Speaker, Apache-2.0.
# ============================================================================


def get_nonlinear(config_str, channels):
    nonlinear = nn.Sequential()
    for name in config_str.split("-"):
        if name == "relu":
            nonlinear.add_module("relu", nn.ReLU(inplace=True))
        elif name == "prelu":
            nonlinear.add_module("prelu", nn.PReLU(channels))
        elif name == "batchnorm":
            nonlinear.add_module("batchnorm", nn.BatchNorm1d(channels))
        elif name == "batchnorm_":
            nonlinear.add_module("batchnorm", nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError(f"Unexpected module ({name}).")
    return nonlinear


def statistics_pooling(x, dim=-1, keepdim=False, unbiased=True, _eps=1e-2):
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    stats = torch.cat([mean, std], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


class StatsPool(nn.Module):
    def forward(self, x):
        return statistics_pooling(x)


class TDNNLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
    ):
        super().__init__()
        if padding < 0:
            assert kernel_size % 2 == 1, f"Expect equal paddings, but got even kernel size ({kernel_size})"
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        x = self.linear(x)
        return self.nonlinear(x)


class CAMLayer(nn.Module):
    def __init__(
        self,
        bn_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        bias,
        reduction=2,
    ):
        super().__init__()
        self.linear_local = nn.Conv1d(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))
        return y * m

    def seg_pooling(self, x, seg_len=100, stype="avg"):
        if stype == "avg":
            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        elif stype == "max":
            seg = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        else:
            raise ValueError("Wrong segment pooling type.")
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
        return seg[..., : x.shape[-1]]


class CAMDenseTDNNLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bn_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
        memory_efficient=False,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, f"Expect equal paddings, but got even kernel size ({kernel_size})"
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def bn_function(self, x):
        return self.linear1(self.nonlinear1(x))

    def forward(self, x):
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x)
        else:
            x = self.bn_function(x)
        return self.cam_layer(self.nonlinear2(x))


class CAMDenseTDNNBlock(nn.ModuleList):
    def __init__(
        self,
        num_layers,
        in_channels,
        out_channels,
        bn_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
        memory_efficient=False,
    ):
        super().__init__()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.add_module(f"tdnnd{i + 1}", layer)

    def forward(self, x):
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, config_str="batchnorm-relu"):
        super().__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.nonlinear(x)
        return self.linear(x)


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, config_str="batchnorm-relu"):
        super().__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        return self.nonlinear(x)


class BasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=(stride, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


# ============================================================================
# CAM++ model (from modules/speaker/campplus.py)
# ----------------------------------------------------------------------------
# 3D-Speaker, Apache-2.0.
# ============================================================================


class FCM(nn.Module):
    def __init__(
        self,
        block=BasicResBlock,
        num_blocks=(2, 2),
        m_channels=32,
        feat_dim=_SPEAKER_FBANK_N_MELS,
    ):
        super().__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)

        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[1], stride=2)

        self.conv2 = nn.Conv2d(m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))

        shape = out.shape
        return out.reshape(shape[0], shape[1] * shape[2], shape[3])


class CAMPPlus(nn.Module):
    _TDNN_KERNEL_SIZE = 5
    _TDNN_STRIDE = 2
    _TDNN_PADDING = 2

    def __init__(
        self,
        feat_dim=_SPEAKER_FBANK_N_MELS,
        embedding_size=512,
        growth_rate=32,
        bn_size=4,
        init_channels=128,
        config_str="batchnorm-relu",
        memory_efficient=True,
    ):
        super().__init__()

        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels

        self.xvector = nn.Sequential(
            OrderedDict(
                [
                    (
                        "tdnn",
                        TDNNLayer(
                            channels,
                            init_channels,
                            self._TDNN_KERNEL_SIZE,
                            stride=self._TDNN_STRIDE,
                            dilation=1,
                            padding=-1,
                            config_str=config_str,
                        ),
                    ),
                ]
            )
        )
        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(zip((12, 24, 16), (3, 3, 3), (1, 2, 2), strict=True)):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.xvector.add_module(f"block{i + 1}", block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(
                f"transit{i + 1}",
                TransitLayer(channels, channels // 2, bias=False, config_str=config_str),
            )
            channels //= 2

        self.xvector.add_module("out_nonlinear", get_nonlinear(config_str, channels))

        self.xvector.add_module("stats", StatsPool())
        self.xvector.add_module("dense", DenseLayer(channels * 2, embedding_size, config_str="batchnorm_"))

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _conv_output_lengths(lengths, kernel_size, stride=1, padding=0, dilation=1):
        return (
            torch.div(
                lengths + 2 * padding - dilation * (kernel_size - 1) - 1,
                stride,
                rounding_mode="floor",
            )
            + 1
        )

    @staticmethod
    def _make_length_mask(lengths, max_len, device):
        lengths = lengths.to(device=device, dtype=torch.long).clamp(min=0, max=max_len)
        return torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)

    def _masked_stats_pooling(self, x, lengths, unbiased=True, eps=1e-2):
        lengths = lengths.to(device=x.device, dtype=torch.long).clamp(min=1, max=x.size(-1))
        mask = self._make_length_mask(lengths, x.size(-1), x.device).unsqueeze(1)
        mask = mask.to(dtype=x.dtype)

        denom = lengths.to(dtype=x.dtype).view(-1, 1).clamp_min(1.0)
        mean = (x * mask).sum(dim=-1) / denom

        centered = (x - mean.unsqueeze(-1)) * mask
        var_denom = (lengths - 1).clamp_min(1).to(dtype=x.dtype).view(-1, 1) if unbiased else denom
        var = centered.pow(2).sum(dim=-1) / var_denom
        std = torch.sqrt(var.clamp_min(eps))
        return torch.cat([mean, std], dim=1)

    def forward(self, x, lengths=None):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)
        if lengths is not None:
            lengths = lengths.to(device=x.device, dtype=torch.long).clamp(min=1)

        for name, module in self.xvector.named_children():
            if name == "stats":
                x = self._masked_stats_pooling(x, lengths) if lengths is not None else module(x)
                continue

            x = module(x)
            if name == "tdnn" and lengths is not None:
                lengths = self._conv_output_lengths(
                    lengths,
                    kernel_size=self._TDNN_KERNEL_SIZE,
                    stride=self._TDNN_STRIDE,
                    padding=self._TDNN_PADDING,
                )

        return x


# ============================================================================
# SpeakerXVectorFeatures wrapper (from modules/speaker/encoder.py)
# ============================================================================


class SpeakerXVectorFeatures(nn.Module):
    """Speaker embedding extractor based on 3D-Speaker CAM++."""

    def __init__(
        self,
        sample_rate=_SPEAKER_FBANK_SAMPLE_RATE,
        campplus_embedding_size=512,
        max_audio_seconds=10.0,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.max_audio_seconds = float(max_audio_seconds)
        self.model = CAMPPlus(
            feat_dim=_SPEAKER_FBANK_N_MELS,
            embedding_size=campplus_embedding_size,
        )
        self.resample = None
        if self.sample_rate != _SPEAKER_FBANK_SAMPLE_RATE:
            self.resample = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=_SPEAKER_FBANK_SAMPLE_RATE,
            )

        for param in self.model.parameters():
            param.requires_grad = False

    @staticmethod
    def _normalize_lengths(lengths, batch_size, max_length, device, *, min_length):
        if lengths is None:
            return torch.full(
                (batch_size,),
                max_length,
                device=device,
                dtype=torch.long,
            )
        return lengths.to(device=device, dtype=torch.long).clamp(
            min=min_length,
            max=max_length,
        )

    def _crop_audio(self, audio, audio_lengths=None):
        original_lengths = self._normalize_lengths(
            audio_lengths,
            audio.size(0),
            audio.size(-1),
            audio.device,
            min_length=0,
        )
        if self.max_audio_seconds <= 0:
            return audio, original_lengths, original_lengths, torch.zeros_like(original_lengths)

        max_input_length = round(self.sample_rate * self.max_audio_seconds)
        cropped_audio = []
        cropped_lengths = []
        starts = []

        for index, total_length_tensor in enumerate(original_lengths):
            total_length = int(total_length_tensor.item())
            cropped_length = min(total_length, max_input_length)
            start = random.randint(0, total_length - cropped_length) if total_length > cropped_length else 0
            cropped_audio.append(audio[index, start : start + cropped_length])
            cropped_lengths.append(cropped_length)
            starts.append(start)

        return (
            pad_sequence(
                cropped_audio,
                batch_first=True,
                padding_value=0.0,
            ),
            original_lengths,
            torch.tensor(
                cropped_lengths,
                device=audio.device,
                dtype=torch.long,
            ),
            torch.tensor(starts, device=audio.device, dtype=torch.long),
        )

    def _crop_fbank(
        self,
        fbank,
        fbank_lengths,
        original_audio_lengths,
        cropped_audio_lengths,
        starts,
    ):
        original_fbank_lengths = self._normalize_lengths(
            fbank_lengths,
            fbank.size(0),
            fbank.size(1),
            fbank.device,
            min_length=1,
        )
        cropped_fbank = []
        cropped_fbank_lengths = []

        for index, total_feat_length_tensor in enumerate(original_fbank_lengths):
            total_audio_length = int(original_audio_lengths[index].item())
            total_feat_length = int(total_feat_length_tensor.item())
            start_audio = int(starts[index].item())
            end_audio = start_audio + int(cropped_audio_lengths[index].item())

            if total_audio_length > 0:
                start_feat = math.floor(start_audio * total_feat_length / total_audio_length)
                end_feat = math.ceil(end_audio * total_feat_length / total_audio_length)
            else:
                start_feat = 0
                end_feat = 1

            start_feat = min(start_feat, total_feat_length - 1)
            end_feat = min(max(end_feat, start_feat + 1), total_feat_length)
            cropped_fbank.append(fbank[index, start_feat:end_feat])
            cropped_fbank_lengths.append(end_feat - start_feat)

        return pad_sequence(
            cropped_fbank,
            batch_first=True,
            padding_value=0.0,
        ), torch.tensor(
            cropped_fbank_lengths,
            device=fbank.device,
            dtype=torch.long,
        )

    def _extract_fbank_batch(self, audio, audio_lengths):
        if self.resample is not None:
            audio = self.resample(audio)
            audio_lengths = torch.ceil(audio_lengths.float() * (_SPEAKER_FBANK_SAMPLE_RATE / self.sample_rate)).long()

        audio_cpu = audio.detach().cpu()
        features = []

        for index, valid_length_tensor in enumerate(audio_lengths):
            valid_length = int(valid_length_tensor.item())
            waveform = audio_cpu[index, :valid_length]
            if waveform.numel() == 0:
                waveform = audio_cpu.new_zeros(1)
            features.append(
                extract_speaker_fbank(
                    waveform,
                    sample_rate=_SPEAKER_FBANK_SAMPLE_RATE,
                )
            )

        fbank_lengths = torch.tensor(
            [feature.size(0) for feature in features],
            device=audio.device,
            dtype=torch.long,
        )
        fbank = pad_sequence(
            features,
            batch_first=True,
            padding_value=0.0,
        ).to(device=audio.device, dtype=audio.dtype)
        return fbank, fbank_lengths

    @torch.no_grad()
    @torch.autocast(enabled=False, device_type="cuda")
    def forward(self, audio, audio_lengths=None, fbank=None, fbank_lengths=None, **_kwargs):
        self.model.eval()
        audio = audio.float()
        if audio.dim() == 3:
            if audio.size(1) != 1:
                raise ValueError(f"Speaker encoder expects mono audio, got shape {tuple(audio.shape)}.")
            audio = audio[:, 0]
        elif audio.dim() != 2:
            raise ValueError(f"Speaker encoder expects a 2D or 3D audio tensor, got shape {tuple(audio.shape)}.")

        audio, original_audio_lengths, cropped_audio_lengths, starts = self._crop_audio(
            audio,
            audio_lengths=audio_lengths,
        )

        if fbank is None:
            fbank, fbank_lengths = self._extract_fbank_batch(
                audio,
                cropped_audio_lengths,
            )
        else:
            if not isinstance(fbank, torch.Tensor):
                raise TypeError("Speaker encoder expects `fbank` to be a torch.Tensor.")
            if fbank.dim() != 3 or fbank.size(0) != audio.size(0):
                raise ValueError(
                    f"Speaker encoder expects `fbank` with shape (B, T, F) and matching batch size, got {tuple(fbank.shape)}."  # noqa: E501
                )
            fbank, fbank_lengths = self._crop_fbank(
                fbank.to(device=audio.device, dtype=torch.float32),
                fbank_lengths,
                original_audio_lengths,
                cropped_audio_lengths,
                starts,
            )

        return self.model(fbank, lengths=fbank_lengths)
