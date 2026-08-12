"""Vendored TADA acoustic encoder (from MIT-licensed hume-tada), adapted for OFFLINE
local use. Encodes a reference waveform + transcript into per-token acoustic features
(the voice) and a token→frame alignment (durations) for voice-cloning prompts.

Used ONLY offline by the example to build a prompt; it is never imported by the
serving worker (so its torch/torchaudio-heavy deps stay out of the engine). Differences
vs. upstream ``tada/modules/encoder.py``:
  * ``Snake1d`` is vendored inline (drops the external ``dac`` dependency);
  * weights/tokenizer are loaded from local paths (no HuggingFace fetch);
  * only the transcript-provided path is kept (no parakeet ASR).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Literal

import torch
import torchaudio

from .aligner import Aligner


class Snake1d(torch.nn.Module):
    """Snake activation (vendored from descript-audio-codec to avoid the ``dac`` dep)."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(shape[0], shape[1], -1)
        x = x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)
        return x.reshape(shape)


def WNConv1d(*args, **kwargs):
    return torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(*args, **kwargs))


class ResidualUnit(torch.nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = torch.nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(torch.nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = torch.nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(dim // 2, dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)),
        )

    def forward(self, x):
        return self.block(x)


class WavEncoder(torch.nn.Module):
    def __init__(self, d_model: int = 64, strides: list | None = None, d_latent: int = 64):
        super().__init__()
        strides = strides or [2, 4, 8, 8]
        block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            block += [EncoderBlock(d_model, stride=stride)]
        block += [Snake1d(d_model), WNConv1d(d_model, d_latent, kernel_size=3, padding=1)]
        self.block = torch.nn.Sequential(*block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


def _create_segment_attention_mask(text_token_mask: torch.Tensor, version: str = "v2") -> torch.Tensor:
    """Block-wise attention mask from token-boundary markers (upstream v1/v2).

    Returns a [B, L, L] boolean mask where True == cannot attend.
    """
    if version == "v1":
        block_ids = torch.cumsum(text_token_mask, dim=1) - text_token_mask
        block_ids_i = block_ids.unsqueeze(2)
        block_ids_j = block_ids.unsqueeze(1)
        same_block = block_ids_j == block_ids_i
        block_ids_j_excl_last = torch.where(text_token_mask.bool(), -10, block_ids_j[:, 0, :]).unsqueeze(1)
        next_block = block_ids_j_excl_last == (block_ids_i + 1)
        return ~(same_block | next_block)
    elif version == "v2":
        block_ids = torch.cumsum(text_token_mask, dim=1)
        block_ids_i = block_ids.unsqueeze(2)
        block_ids_j = block_ids.unsqueeze(1)
        same_block = block_ids_i == block_ids_j
        is_marked_i = text_token_mask.unsqueeze(2).bool()
        is_marked_j = text_token_mask.unsqueeze(1).bool()
        same_block_valid = same_block & (~is_marked_j | (is_marked_i & same_block))
        prev_block = block_ids_j == (block_ids_i - 1)
        prev_block_valid = prev_block & ~is_marked_j
        can_attend = same_block_valid | (is_marked_i & prev_block_valid)
        return ~can_attend
    raise ValueError(f"Unknown version: {version}")


class LocalSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1, max_seq_len: int = 8192):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = torch.nn.Linear(d_model, 3 * d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(d_model)
        self.max_seq_len = max_seq_len
        self.register_buffer("rope_freqs", self._compute_rope_freqs(self.head_dim, max_seq_len), persistent=False)

    def _compute_rope_freqs(self, head_dim: int, max_seq_len: int) -> torch.Tensor:
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        return torch.stack([freqs.cos(), freqs.sin()], dim=-1)

    def _apply_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        b, h, s, d = x.shape
        freqs = self.rope_freqs[:seq_len]
        fc = freqs[..., 0]
        fs = freqs[..., 1]
        xr = x.reshape(b, h, s, d // 2, 2)
        x0 = xr[..., 0]
        x1 = xr[..., 1]
        r0 = x0 * fc.unsqueeze(0).unsqueeze(0) - x1 * fs.unsqueeze(0).unsqueeze(0)
        r1 = x0 * fs.unsqueeze(0).unsqueeze(0) + x1 * fc.unsqueeze(0).unsqueeze(0)
        return torch.stack([r0, r1], dim=-1).reshape(b, h, s, d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, s, _ = x.shape
        qkv = self.qkv(x).reshape(b, s, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self._apply_rope(q, s)
        k = self._apply_rope(k, s)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            if mask.dim() == 2:
                attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            elif mask.dim() == 3:
                attn = attn.masked_fill(mask.unsqueeze(1), float("-inf"))
        w = torch.softmax(attn, dim=-1)
        out = torch.matmul(w, v).transpose(1, 2).reshape(b, s, self.d_model)
        out = self.dropout(self.out_proj(out))
        return self.layer_norm(x + out)


class LocalAttentionEncoderLayer(torch.nn.Module):
    def __init__(
        self, d_model: int, num_heads: int = 8, d_ff: int | None = None, dropout: float = 0.1, max_seq_len: int = 8192
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attn = LocalSelfAttention(d_model, num_heads=num_heads, dropout=dropout, max_seq_len=max_seq_len)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_ff, d_model),
            torch.nn.Dropout(dropout),
        )
        self.norm = torch.nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.self_attn(x, mask=mask)
        return self.norm(x + self.ffn(x))


class LocalAttentionEncoder(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int | None = None,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
        d_input: int | None = None,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                LocalAttentionEncoderLayer(
                    d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout, max_seq_len=max_seq_len
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = torch.nn.LayerNorm(d_model)
        self.input_proj = (
            torch.nn.Linear(d_input, d_model) if d_input is not None and d_input != d_model else torch.nn.Identity()
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.final_norm(x)


@dataclass
class EncoderOutput:
    audio: torch.Tensor
    audio_len: torch.Tensor
    text: list[str]
    token_positions: torch.Tensor
    token_values: torch.Tensor
    sample_rate: int = 24000
    text_tokens: torch.Tensor | None = None
    text_tokens_len: torch.Tensor | None = None
    token_masks: torch.Tensor | None = None


class EncoderConfig:
    hidden_dim = 1024
    embed_dim = 512
    strides = [6, 5, 4, 4]
    num_attn_layers = 6
    num_attn_heads = 8
    attn_dim_feedforward = 4096
    attn_dropout = 0.1
    block_attention: Literal["none", "v1", "v2"] = "v2"
    num_frames_per_second = 50
    std = 0.5
    acoustic_mean = 0.0
    acoustic_std = 1.5


class Encoder(torch.nn.Module):
    """Acoustic encoder: reference waveform -> per-token acoustic features + alignment."""

    def __init__(self, config: EncoderConfig, aligner: Aligner):
        super().__init__()
        self.config = config
        self.wav_encoder = WavEncoder(d_model=64, strides=config.strides, d_latent=config.hidden_dim)
        self.local_attention_encoder = LocalAttentionEncoder(
            d_model=config.hidden_dim,
            num_layers=config.num_attn_layers,
            num_heads=config.num_attn_heads,
            d_ff=config.attn_dim_feedforward,
            dropout=config.attn_dropout,
            max_seq_len=8192,
        )
        self.hidden_linear = (
            torch.nn.Linear(config.hidden_dim, config.embed_dim)
            if config.hidden_dim != config.embed_dim
            else torch.nn.Identity()
        )
        self.pos_emb = torch.nn.Embedding(2, config.hidden_dim)
        self._aligner = aligner

    @property
    def tokenizer(self):
        return self._aligner.tokenizer

    @classmethod
    def from_local(
        cls, codec_path: str, tokenizer_path: str, device: torch.device | str = "cpu", dtype=torch.float32
    ) -> Encoder:
        """Build encoder + aligner and load weights from local ``<codec_path>``."""
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        aligner = Aligner.from_local(codec_path, tokenizer, device=device, dtype=dtype)
        self = cls(EncoderConfig(), aligner)
        weights = load_file(os.path.join(codec_path, "encoder", "model.safetensors"))
        missing, unexpected = self.load_state_dict(weights, strict=False)
        real_missing = [k for k in missing if "rope_freqs" not in k and not k.startswith("_aligner")]
        if real_missing:
            raise RuntimeError(f"Encoder: missing weights: {real_missing[:8]} ...")
        return self.to(device=device, dtype=dtype).eval()

    def get_encoder_outputs(self, audio: torch.Tensor, token_masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        enc_out = self.wav_encoder(torch.nn.functional.pad(audio.unsqueeze(1), (0, 960), value=0)).transpose(1, 2)
        seq_len = enc_out.shape[1]
        padded_token_masks = torch.nn.functional.pad(token_masks, (0, seq_len - token_masks.shape[1]), value=0)
        enc_out = enc_out + self.pos_emb(padded_token_masks)
        attn_mask = _create_segment_attention_mask(padded_token_masks, version=self.config.block_attention)
        enc_out = self.local_attention_encoder(enc_out, mask=attn_mask)
        enc_out = self.hidden_linear(enc_out)
        return enc_out, padded_token_masks

    @torch.no_grad()
    def forward(
        self, audio: torch.Tensor, text: list[str] | str, sample_rate: int = 24000, sample: bool = False
    ) -> EncoderOutput:
        if isinstance(text, str):
            text = [text]
        device = audio.device
        if sample_rate != 24000:
            audio = torchaudio.functional.resample(audio, sample_rate, 24000)
            sample_rate = 24000
        audio_length = torch.tensor([audio.shape[-1]], device=device)

        text_tokens = [self.tokenizer.encode(t, add_special_tokens=False, return_tensors="pt") for t in text]
        text_token_len = torch.tensor([t.shape[-1] for t in text_tokens], device=device)
        text_tokens = torch.nn.utils.rnn.pad_sequence(
            [t.squeeze(0) for t in text_tokens], batch_first=True, padding_value=self.tokenizer.eos_token_id
        ).to(device)

        align = self._aligner(audio, text_tokens=text_tokens, audio_length=audio_length, sample_rate=sample_rate)
        token_positions, token_masks = align.token_positions, align.token_masks

        enc_out, token_masks = self.get_encoder_outputs(
            audio.to(
                self.hidden_linear.weight.dtype if isinstance(self.hidden_linear, torch.nn.Linear) else audio.dtype
            ),
            token_masks,
        )
        encoded_expanded = torch.where(token_masks.unsqueeze(-1) == 0, torch.zeros_like(enc_out), enc_out)
        token_values = torch.gather(
            encoded_expanded,
            1,
            (token_positions - 1).clamp(min=0).unsqueeze(-1).expand(-1, -1, encoded_expanded.shape[-1]),
        )
        token_values = (token_values - self.config.acoustic_mean) / self.config.acoustic_std

        return EncoderOutput(
            audio=audio,
            audio_len=audio_length,
            text=text,
            text_tokens=text_tokens,
            text_tokens_len=text_token_len,
            token_positions=token_positions,
            token_masks=token_masks,
            token_values=token_values,
        )
