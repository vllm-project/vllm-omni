import math
from typing import Dict
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from htrain.util import instantiate_from_config
from nemo.collections.asr.parts.convolution_layers import create_pad_mask
from .decoding import decode as decode_function


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int, dropout: float, n_kv: int = 0):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.qkv_same_dim = n_kv == n_state
        self.key = nn.Linear(n_kv if n_kv else n_state, n_state, bias=False)
        self.value = nn.Linear(n_kv if n_kv else n_state, n_state)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(n_state, n_state)

        self.reset_parameters()

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.query.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.key.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.value.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.query.weight)
            nn.init.xavier_uniform_(self.key.weight)
            nn.init.xavier_uniform_(self.value.weight)

        nn.init.xavier_uniform_(self.out.weight)
        if self.out.bias is not None:
            nn.init.constant_(self.out.bias, 0.0)
        # if self.value.bias is not None:
        #     nn.init.xavier_normal_(self.value.bias)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        self_attn_mask: Optional[Tensor] = None,
        xattn_padding_mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None:
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # decoding
            if xa is None:
                # for self-attention, calculate keys and values and concat with previous keys and values
                k = self.key(x)
                v = self.value(x)
                prev_kv = kv_cache.get(self)
                if prev_kv:
                    prev_k, prev_v = prev_kv
                    k = torch.cat([prev_k, k], dim=1)
                    v = torch.cat([prev_v, v], dim=1)
                kv_cache[self] = k.detach(), v.detach()
            else:
                # for cross-attention, calculate keys and values once and reuse in subsequent calls.
                kv = kv_cache.get(self)
                if kv:
                    k, v = kv
                else:
                    k = self.key(xa)
                    v = self.value(xa)
                    kv_cache[self] = k.detach(), v.detach()

        wv = self.qkv_attention(q, k, v, self_attn_mask, xattn_padding_mask)
        return self.out(wv)

    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, self_attn_mask: Optional[Tensor] = None, xattn_padding_mask: Optional[Tensor] = None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        # [B, n_head, q_len, k_len]
        qk = q @ k

        if self_attn_mask is not None:
            qk = qk + self_attn_mask

        if xattn_padding_mask is not None:
            # don't attend to padding symbols
            qk = qk.masked_fill(xattn_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf"))

        w = F.softmax(qk.float(), dim=-1).to(q.dtype)
        w = self.dropout(w)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, n_mlp: int, dropout: float, dropout_attn: float, dropout_mlp: float,
                 *, act_fn='gelu', cross_attn: bool = False, n_xattn_kv: int = 0):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head, dropout_attn)
        self.attn_ln = nn.LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(n_state, n_head, dropout_attn, n_xattn_kv) if cross_attn else None
        self.cross_attn_ln = nn.LayerNorm(n_state) if cross_attn else None

        self.dropout = nn.Dropout(dropout)

        if act_fn == 'gelu':
            act = nn.GELU()
        else:
            assert act_fn == 'relu'
            act = nn.ReLU()
        self.mlp = nn.Sequential(nn.Linear(n_state, n_mlp), act, nn.Dropout(dropout_mlp), nn.Linear(n_mlp, n_state))
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        self_attn_mask: Optional[Tensor] = None,
        xattn_padding_mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.dropout(
            self.attn(self.attn_ln(x), self_attn_mask=self_attn_mask, kv_cache=kv_cache)
        )
        if self.cross_attn:
            x = x + self.dropout(self.cross_attn(self.cross_attn_ln(x), xa, xattn_padding_mask=xattn_padding_mask, kv_cache=kv_cache))
        x = x + self.dropout(self.mlp(self.mlp_ln(x)))
        return x


class TextDecoder(nn.Module):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_mlp: int, n_layer: int,
                 n_encoder_state: int, dropout: float, dropout_attn: float, dropout_mlp: float, layer_drop: float,
                 act_fn: str = 'gelu'):
        super().__init__()

        token_embedding = nn.Embedding(n_vocab, n_state)
        nn.init.normal_(token_embedding.weight, mean=0, std=n_state ** -0.5)
        self.token_embedding = token_embedding
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state).normal_(mean=0, std=n_state ** -0.5))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(
                n_state, n_head, n_mlp, dropout=dropout, dropout_attn=dropout_attn, dropout_mlp=dropout_mlp,
                cross_attn=True, n_xattn_kv=n_encoder_state, act_fn=act_fn
            )
             for _ in range(n_layer)]
        )
        self.layer_drop = layer_drop
        self.ln = nn.LayerNorm(n_state)

        self._future_mask_buffer = torch.empty(0)

    def forward(self, x: Tensor, x_len: Tensor, xa: Tensor, xa_len: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        if kv_cache:
            offset = next(iter(kv_cache.values()))[0].shape[1]
        else:
            offset = 0
        x = self.token_embedding(x) + self.positional_embedding[offset: offset + x.shape[-1]]
        x = x.to(xa.dtype)

        future_mask = self.get_future_mask(x)
        encoder_padding_mask = create_pad_mask(xa_len, xa.shape[1])
        for block in self.blocks:
            if self.training and self.layer_drop > 0 and np.random.random() < self.layer_drop:
                continue
            x = block(x, xa, self_attn_mask=future_mask, xattn_padding_mask=encoder_padding_mask, kv_cache=kv_cache)

        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

        return logits

    def get_future_mask(self, tensor):
        dim = tensor.size(1)
        if self._future_mask_buffer.size(0) < dim:
            self._future_mask_buffer = torch.empty(dim, dim).fill_(float('-inf')).triu_(1)
        self._future_mask_buffer = self._future_mask_buffer.to(tensor)
        return self._future_mask_buffer[:dim, :dim]


class Whisper(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = instantiate_from_config(encoder)
        self.decoder = TextDecoder(**decoder)

    def embed_audio(self, mel: torch.Tensor, mel_len: torch.Tensor):
        feat, feat_len, _ = self.encoder.forward(mel, mel_len)
        # [B, D, T] => [B, T, D]
        feat = torch.transpose(feat, 1, 2)
        return feat, feat_len

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder.forward(tokens, audio_features)

    def forward(self, mel: torch.Tensor, mel_len: torch.Tensor, tokens: torch.Tensor, tokens_len: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat, feat_len, _ = self.encoder(mel, mel_len)
        # [B, D, T] => [B, T, D]
        feat = torch.transpose(feat, 1, 2)

        output = self.decoder(tokens, tokens_len, feat, feat_len)
        return output

    @property
    def device(self):
        return next(self.parameters()).device

    decode = decode_function
