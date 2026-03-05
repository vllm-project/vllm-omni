from typing import Optional

import torch

from nemo.collections.asr.modules.conv_transformer_encoder import init_weights
from nemo.collections.common.parts import rnn


class TextEmbed(torch.nn.Module):

    def __init__(self,
                 embed_num: int,
                 embed_size: int,
                 embed_dropout: float,
                 embed_proj_size: int,
                 mask_label_prob: float = 0.0,
                 mask_label_id: Optional[int] = None,
                 norm_embed: bool = False,
                 ln_eps: float = 1e-5,
                 init_mode='xavier_uniform', bias_init_mode='zero', embedding_init_mode='xavier_uniform'):
        super().__init__()

        self.embed = torch.nn.Embedding(embed_num, embed_size)
        self.embed_size = embed_size

        if embed_proj_size > 0:
            self.embed_proj = torch.nn.Linear(embed_size, embed_proj_size)
            self.embed_size = embed_proj_size
        else:
            self.embed_proj = identity

        if norm_embed:
            self.embed_proj_norm = torch.nn.LayerNorm(embed_proj_size if embed_proj_size else embed_size, eps=ln_eps)
        else:
            self.embed_proj_norm = identity

        self.embed_drop = torch.nn.Dropout(embed_dropout)

        self.mask_label_prob = mask_label_prob
        self.mask_label_id = mask_label_id
        assert 0.0 <= self.mask_label_prob < 1.0
        if self.mask_label_prob > 0:
            assert self.mask_label_id is not None and self.mask_label_id >= 0
            assert self.mask_label_id not in [self.sos_idx, self.blank_idx]

        self.apply(lambda x: init_weights(x, mode=init_mode, bias_mode=bias_init_mode,
                                          embedding_mode=embedding_init_mode))

    def forward(self, texts):
        y = rnn.label_collate(texts)

        if self.mask_label_prob > 0 and self.training:
            y = random_replace(y, rep_prob=self.mask_label_prob, rep_id=self.mask_label_id)

        h = self.embed(y)
        h = self.embed_proj(h)
        h = self.embed_proj_norm(h)
        h = self.embed_drop(h)

        return h


def random_replace(inputs: torch.Tensor, rep_prob, rep_id):
    mask = torch.bernoulli(torch.full(inputs.size(), rep_prob, device=inputs.device)).type(inputs.dtype)
    return mask * rep_id + (1 - mask) * inputs


def identity(x):
    return x
