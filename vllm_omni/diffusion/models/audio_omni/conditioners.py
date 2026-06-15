# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)

_F5_VOCAB_PATH = Path(__file__).with_name("f5_vocab.txt")


class AudioOmniRMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)


class OmniConditioner(nn.Module):
    def __init__(
        self,
        qwen_omni_model_name: str = "Qwen/Qwen2.5-Omni-3B",
        pad_length: int = 2048,
        layer_idx: int = -2,
        **kwargs,
    ):
        super().__init__()
        from transformers import AutoConfig, Qwen2_5OmniProcessor
        from transformers.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration

        self.pad_length = pad_length
        self.layer_idx = layer_idx

        model_path = os.environ.get("QWEN_OMNI_MODEL_PATH", qwen_omni_model_name)
        cfg_src = model_path if os.path.isdir(model_path) else qwen_omni_model_name
        qwen_config = AutoConfig.from_pretrained(cfg_src)
        self.processor = Qwen2_5OmniProcessor.from_pretrained(cfg_src)

        self.model = nn.Module()
        self.model.thinker = Qwen2_5OmniThinkerForConditionalGeneration(qwen_config.thinker_config)
        self.model.thinker.to(torch.bfloat16).eval().requires_grad_(False)

        qwen_feature_dim = 2048
        self.proj_features = nn.Linear(qwen_feature_dim, 768)
        self.norm = AudioOmniRMSNorm()

    @torch.no_grad()
    def forward(self, texts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        pad_length = self.pad_length
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        qwen_mask = inputs["attention_mask"].to(device)
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.autocast(device_type="cuda", enabled=False):
            out = self.model.thinker(**inputs, output_hidden_states=True, return_dict=True)
            features = out.hidden_states[self.layer_idx].bfloat16()

        seq_len = features.shape[1]
        if qwen_mask.shape[1] != seq_len:
            if qwen_mask.shape[1] > seq_len:
                qwen_mask = qwen_mask[:, :seq_len]
            else:
                qwen_mask = F.pad(qwen_mask, (0, seq_len - qwen_mask.shape[1]), value=0)

        padded, real_lengths = [], []
        for b in range(features.shape[0]):
            real = features[b][qwen_mask[b].bool()]
            if real.shape[0] > pad_length:
                real = (
                    F.interpolate(
                        real.transpose(0, 1).unsqueeze(0), size=pad_length, mode="linear", align_corners=False
                    )
                    .squeeze(0)
                    .transpose(0, 1)
                )
            real_lengths.append(real.shape[0])
            padded.append(F.pad(real, (0, 0, 0, pad_length - real.shape[0]), value=0))
        features = torch.stack(padded, dim=0)

        mask = torch.zeros(features.shape[0], pad_length, dtype=torch.bool, device=features.device)
        for b, n in enumerate(real_lengths):
            mask[b, :n] = True

        features = self.norm(features)
        features = self.proj_features(features.to(self.proj_features.weight.dtype))
        return features, mask


class _GRN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=1, keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class _ConvNeXtV2Block(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.grn = _GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv2(self.grn(self.act(self.pwconv1(x))))
        return residual + x


def _precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1)


class _F5TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds: int, text_dim: int, conv_layers: int = 4, conv_mult: int = 2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)
        self.precompute_max_pos = 4096
        self.register_buffer("freqs_cis", _precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
        self.text_blocks = nn.Sequential(
            *[_ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
        )

    def forward(self, text: torch.Tensor, seq_len: int) -> torch.Tensor:
        text = text.to(torch.long) + 1
        text = text[:, :seq_len]
        text = F.pad(text, (0, seq_len - text.shape[1]), value=0)
        text_mask = text == 0

        x = self.text_embed(text)
        pos = torch.arange(seq_len, device=self.freqs_cis.device, dtype=torch.long).clamp(
            max=self.precompute_max_pos - 1
        )
        x = x + self.freqs_cis[pos].to(x.device)
        fill = text_mask.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = x.masked_fill(fill, 0.0)
        for block in self.text_blocks:
            x = block(x)
            x = x.masked_fill(fill, 0.0)
        return x


class TTSConditioner(nn.Module):
    def __init__(
        self,
        vocab_file: str | None = None,
        seq_len: int = 2584,
        proj_seq_len: int = 256,
        **kwargs,
    ):
        super().__init__()
        vocab_path = vocab_file if vocab_file and os.path.exists(vocab_file) else str(_F5_VOCAB_PATH)
        with open(vocab_path, encoding="utf-8") as f:
            self.vocab_char_map = {line[:-1]: i for i, line in enumerate(f)}
        self.seq_len = seq_len

        text_dim = 512
        self.text_embed = _F5TextEmbedding(len(self.vocab_char_map), text_dim, conv_layers=4)
        self.proj_features = nn.Linear(text_dim, 768)
        self.proj_seq_len = nn.Linear(seq_len, proj_seq_len)
        self.empty_speech_feat = nn.Parameter(torch.zeros(1, proj_seq_len, 768), requires_grad=False)
        self.norm_features = AudioOmniRMSNorm()
        self.norm = AudioOmniRMSNorm()

    @staticmethod
    def _convert_char_to_pinyin(text_list: list[str]) -> list[list[str]]:
        import jieba
        from pypinyin import Style, lazy_pinyin

        if jieba.dt.initialized is False:
            jieba.default_logger.setLevel(50)
            jieba.initialize()

        custom_trans = str.maketrans({";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"})

        def is_chinese(c: str) -> bool:
            return "㄀" <= c <= "鿿"

        out = []
        for text in text_list:
            char_list: list[str] = []
            text = text.translate(custom_trans)
            for seg in jieba.cut(text):
                seg_byte_len = len(bytes(seg, "UTF-8"))
                if seg_byte_len == len(seg):
                    if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                        char_list.append(" ")
                    char_list.extend(seg)
                elif seg_byte_len == 3 * len(seg):
                    seg_pinyin = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                    for i, c in enumerate(seg):
                        if is_chinese(c):
                            char_list.append(" ")
                        char_list.append(seg_pinyin[i])
                else:
                    for c in seg:
                        if ord(c) < 256:
                            char_list.extend(c)
                        elif is_chinese(c):
                            char_list.append(" ")
                            char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                        else:
                            char_list.append(c)
            out.append(char_list)
        return out

    @torch.no_grad()
    def forward(self, texts: list[str | None], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(texts)
        is_none = [t is None for t in texts]
        if all(is_none):
            empty = self.empty_speech_feat.expand(batch_size, -1, -1).to(device)
            return empty, torch.ones(empty.shape[0], empty.shape[1], device=device)

        char_lists = self._convert_char_to_pinyin([t if t is not None else "" for t in texts])
        idx = [torch.tensor([self.vocab_char_map.get(c, 0) for c in chars]) for chars in char_lists]
        text = pad_sequence(idx, padding_value=-1, batch_first=True).to(device)

        x = self.text_embed(text, self.seq_len)
        x = self.norm_features(x)
        x = self.proj_features(x)
        x = self.proj_seq_len(x.transpose(1, 2)).transpose(1, 2)

        empty = self.empty_speech_feat.expand(batch_size, -1, -1).to(device)
        none_mask = torch.tensor(is_none, device=device).view(batch_size, 1, 1)
        x = torch.where(none_mask, empty, x)
        x = self.norm(x)
        return x, torch.ones(x.shape[0], x.shape[1], device=device)


class AudioMelConditioner(nn.Module):
    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mel_channels: int = 100,
        target_sample_rate: int = 44100,
        seq_len: int = 236,
        **kwargs,
    ):
        super().__init__()
        self._mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mel_channels,
            power=1,
            center=True,
            normalized=False,
            norm=None,
        )
        self.proj_features = nn.Linear(n_mel_channels, 768)
        self.proj_sequence_features = nn.Linear(1723, seq_len)
        self.empty_audio_feat = nn.Parameter(torch.zeros(1, seq_len, 768), requires_grad=False)
        self.norm = AudioOmniRMSNorm()

    @torch.no_grad()
    def forward(self, wavs: list[torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        batch = torch.stack(wavs, dim=0).to(device).float()
        self._mel.to(device)
        mels = self._mel(batch).clamp(min=1e-5).log()
        mels = self.proj_sequence_features(mels)
        mels = mels.transpose(1, 2)
        emb = self.norm(self.proj_features(mels))
        return emb, torch.ones(emb.shape[0], emb.shape[1], device=device)


class SynchformerConditioner(nn.Module):
    def __init__(self, sync_seq_dim: int = 240, sync_output_dim: int = 128, input_dim: int = 768, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, input_dim)
        self.dim_project = nn.Linear(sync_seq_dim, sync_output_dim)

    @torch.no_grad()
    def forward(self, sync_features: list[torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat(sync_features, dim=0).to(device)
        x = self.proj(self.norm(x))
        x = self.dim_project(x.transpose(1, 2)).transpose(1, 2)
        return x, torch.ones(x.shape[0], x.shape[1], device=device)
