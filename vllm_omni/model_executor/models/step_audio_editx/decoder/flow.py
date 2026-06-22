import logging
import math
from collections.abc import Iterable

import torch
import torch.nn as nn
from torch.nn import functional as F
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
)
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.layers import PreLookaheadLayer
from vllm_omni.model_executor.models.cosyvoice3.utils import make_pad_mask

from ..audio_tokenizer.transformer_utils import PositionwiseFeedForward

logger = logging.getLogger(__name__)


class DualCodebookEmbedding(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        input_size: int,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, input_size // 2)

    def forward(self, token: torch.Tensor):
        embed1 = self.embedding(token[..., 0])
        embed2 = self.embedding(token[..., 1])
        return torch.cat([embed1, embed2], dim=-1)


class BaseSubsampling(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def position_encoding(self, offset: int | torch.Tensor, size: int) -> torch.Tensor:
        return self.pos_enc.position_encoding(offset, size)


class LinearNoSubsampling(BaseSubsampling):
    def __init__(self, idim: int, odim: int, pos_enc_class: torch.nn.Module):
        super().__init__()
        self.out = torch.nn.Sequential(
            torch.nn.Linear(idim, odim),
            torch.nn.LayerNorm(odim, eps=1e-5),
        )
        self.pos_enc = pos_enc_class

    def forward(
        self, x: torch.Tensor, offset: int | torch.Tensor = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb


class RelPositionMultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
    """

    def __init__(self, n_head: int, n_feat: int, key_bias: bool = True):
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = ReplicatedLinear(n_feat, n_feat)
        self.linear_k = ReplicatedLinear(n_feat, n_feat, bias=key_bias)
        self.linear_v = ReplicatedLinear(n_feat, n_feat)
        self.linear_out = ReplicatedLinear(n_feat, n_feat)
        self.linear_pos = ReplicatedLinear(n_feat, n_feat, bias=False)

        # # linear transformation for positional encoding
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.empty(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.empty(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)
        # torch.nn.init.xavier_uniform_(self.pos_bias_u)
        # torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Compute relative positional encoding."""

        zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(x.size()[0], x.size()[1], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[:, :, :, : x.size(-1) // 2 + 1]  # only keep the positions from 0 to time2
        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, time2, size).
            cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        """
        n_batch = query.size(0)

        q, _ = self.linear_q(query)
        k, _ = self.linear_k(key)
        v, _ = self.linear_v(value)

        q = q.view(n_batch, -1, self.h, self.d_k)  # (batch, time1, head, d_k)
        k = k.view(n_batch, -1, self.h, self.d_k).transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.view(n_batch, -1, self.h, self.d_k).transpose(1, 2)  # (batch, head, time2, d_k)

        if cache is not None and cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        # NOTE(xcsong): We do cache slicing in encoder.forward_chunk, since it's
        #   non-trivial to calculate `next_cache_start` here.
        new_cache = torch.cat((k, v), dim=-1)

        n_batch_pos = pos_emb.size(0)
        p, _ = self.linear_pos(pos_emb)
        p = p.view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        # NOTE(Xiang Lyu): Keep rel_shift since espnet rel_pos_emb is used
        if matrix_ac.shape != matrix_bd.shape:
            matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)  # (batch, head, time1, time2)

        n_batch = v.size(0)
        if mask.size(2) > 0:  # time2 > 0
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            # For last chunk, time2 might be larger than scores.size(-1)
            mask = mask[:, :, :, : scores.size(-1)]  # (batch, 1, *, time2)
            scores = scores.masked_fill(mask, -float("inf"))
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        x = torch.matmul(attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        output, _ = self.linear_out(x)
        return output, new_cache


class EspnetRelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding module (new implementation).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    See : Appendix B in https://arxiv.org/abs/1901.02860

    Args:
        d_model (int): Embedding dimension.
        max_len (int): Maximum input length.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: torch.Tensor):
        """Reset the positional encodings."""
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vector and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor, offset: int | torch.Tensor = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).

        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.position_encoding(size=x.size(1), offset=offset)
        return x, pos_emb

    def position_encoding(self, offset: int | torch.Tensor, size: int) -> torch.Tensor:
        """For getting encoding in a streaming fashion
        Args:
            offset (int or torch.tensor): start offset
            size (int): required size of position encoding

        Returns:
            torch.Tensor: Corresponding encoding
        """
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - size + 1 : self.pe.size(1) // 2 + size,
        ]
        return pos_emb


class ConformerEncoderLayer(nn.Module):
    """Encoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module
             instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: use layer_norm after each sub-block.
        enable_cuda_graph (bool): Control whether to enable CUDA Graph.
    """

    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: nn.Module | None = None,
        feed_forward_macaron: nn.Module | None = None,
        conv_module: nn.Module | None = None,
        normalize_before: bool = True,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = nn.LayerNorm(size, eps=1e-12)  # for the FNN module
        self.norm_mha = nn.LayerNorm(size, eps=1e-12)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-12)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = nn.LayerNorm(size, eps=1e-12)  # for the CNN module
            self.norm_final = nn.LayerNorm(size, eps=1e-12)  # for the final output of the block
        self.size = size
        self.normalize_before = normalize_before

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): (#batch, time, size)
            mask (torch.Tensor): Mask tensor for the input (#batch, time，time),
                (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): positional encoding, must not be None
                for ConformerEncoderLayer.
            mask_pad (torch.Tensor): batch padding mask used for conv module.
                (#batch, 1，time), (0, 0, 0) means fake mask.
            att_cache (torch.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (torch.Tensor): Convolution cache in conformer layer
                (#batch=1, size, cache_t2)
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time, time).
            torch.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            torch.Tensor: cnn_cache tensor (#batch, size, cache_t2).
        """
        return self._forward_impl(x, mask, pos_emb, mask_pad, att_cache, cnn_cache)

    def _forward_impl(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.feed_forward_macaron(x)
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        # att_cache: (b, head, cache_t, d_k*2)
        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb, att_cache)
        x = residual + x_att
        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        # Fake new cnn cache here, and then change it in conv_module
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + x

            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)

        x = residual + self.ff_scale * self.feed_forward(x)
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)
        return x, mask, new_att_cache, new_cnn_cache


class Upsample1D(nn.Module):
    """A 1D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
    """

    def __init__(self, channels: int, out_channels: int, stride: int = 2, scale_factor: float = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.stride = stride
        # In this mode, first repeat interpolate, than conv with stride=1
        self.conv = nn.Conv1d(self.channels, self.out_channels, stride * 2 + 1, stride=1, padding=0)
        self.scale_factor = float(self.stride) if scale_factor is None else float(scale_factor)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor):
        outputs = F.interpolate(inputs, scale_factor=self.scale_factor, mode="nearest")
        outputs = F.pad(outputs, (self.stride * 2, 0), value=0.0)
        outputs = self.conv(outputs)
        return outputs, input_lengths * self.stride

    def forward_chunk(
        self, inputs: torch.Tensor, input_lengths: torch.Tensor, cache: torch.Tensor = torch.zeros((0, 0, 0))
    ):
        """
        Args:
            inputs(torch.Tensor): shape (b, c, t)
            input_length(torch.Tensor): shape (b), can be None
            cache(torch.Tensor): shape (b, c, cache_t), where cache_t = stride * 2
        """
        outputs = F.interpolate(inputs, scale_factor=self.scale_factor, mode="nearest")

        if cache is None:
            cache = inputs.new_zeros(inputs.shape[0], inputs.shape[1], self.stride * 2)
        outputs = torch.cat([cache, outputs], dim=2)
        new_cache = outputs[..., -self.stride * 2 :]
        outputs = self.conv(outputs)

        if input_lengths is not None:
            input_lengths = input_lengths * self.stride
        return outputs, input_lengths, new_cache


class StepPreLookaheadLayer(PreLookaheadLayer):
    def forward_chunk(self, inputs: torch.Tensor, cache: torch.Tensor = None):
        """
        Args:
            inputs(torch.Tensor): shape (b, t, c)
            cache(torch.Tensor): shape (b, c, cache_t=2), c = channels
        """
        outputs = inputs.transpose(1, 2).contiguous()
        outputs = F.leaky_relu(self.conv1(outputs))
        # the length of outputs is input length - pre_lookahead_len
        if cache is None:
            cache = outputs.new_zeros(outputs.shape[0], outputs.shape[1], 2)
        # NOTE
        new_cache = outputs[..., -2:]
        outputs = torch.cat([cache, outputs], dim=2)
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2).contiguous()
        # residual connection
        outputs = outputs + inputs[:, : -self.pre_lookahead_len]
        return outputs, new_cache


class UpsampleConformerEncoderV2(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        pre_lookahead_len: int = 3,
        num_blocks: int = 6,
        num_up_blocks: int = 4,
        up_stride: int = 2,
        up_scale_factor: float = 2,
        attention_heads: int = 4,
        key_bias: bool = True,
        linear_units: int = 2048,
        normalize_before: bool = True,
        **kwargs,
    ):
        super().__init__()
        self._output_size = output_size
        self.embed = LinearNoSubsampling(
            input_size,
            output_size,
            EspnetRelPositionalEncoding(
                output_size,
            ),
        )

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-5)
        activation = nn.SiLU()
        # self-attention module definition
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            key_bias,
        )
        # feed-forward module definition
        positionwise_layer_args = (
            output_size,
            linear_units,
            activation,
        )
        self.pre_lookahead_layer = StepPreLookaheadLayer(
            in_channels=output_size, channels=output_size, pre_lookahead_len=pre_lookahead_len
        )
        self.encoders = torch.nn.ModuleList(
            [
                ConformerEncoderLayer(
                    output_size,
                    RelPositionMultiHeadedAttention(*encoder_selfattn_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args),
                    None,
                    None,
                    normalize_before,
                )
                for _ in range(num_blocks)
            ]
        )
        self.up_layer = Upsample1D(
            channels=output_size, out_channels=output_size, stride=up_stride, scale_factor=up_scale_factor
        )
        self.up_embed = LinearNoSubsampling(
            input_size,
            output_size,
            EspnetRelPositionalEncoding(
                output_size,
            ),
        )
        self.up_encoders = torch.nn.ModuleList(
            [
                ConformerEncoderLayer(
                    output_size,
                    RelPositionMultiHeadedAttention(*encoder_selfattn_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args),
                    None,
                    None,
                    normalize_before,
                )
                for _ in range(num_up_blocks)
            ]
        )

        self.inference_buffers_encoder = {}
        self.inference_buffers_up_encoder = {}
        self.max_static_time = 1500

    # @torch.compile(dynamic=True,backend="eager")
    def _forward_impl_encoder(self, x: torch.Tensor, mask: torch.Tensor, pos_emb: torch.Tensor):
        for layer in self.encoders:
            x, _, _, _ = layer(x, mask, pos_emb)
        return x

    # @torch.compile(dynamic=True,backend="eager")
    def _forward_impl_up_encoder(self, x: torch.Tensor, mask: torch.Tensor, pos_emb: torch.Tensor):
        for layer in self.up_encoders:
            x, _, _, _ = layer(x, mask, pos_emb)
        return x

    def output_size(self) -> int:
        return self._output_size

    # @torch.compile(dynamic=True,backend="eager")
    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # (sfy) chunk training strategy should not be open-sourced
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        xs, pos_emb = self.embed(xs)

        # lookahead
        xs = self.pre_lookahead_layer(xs)
        # conformer block

        xs = self._forward_impl_encoder(xs, masks, pos_emb)
        # upsample
        xs = xs.transpose(1, 2).contiguous()
        xs, xs_lens = self.up_layer(xs, xs_lens)
        xs = xs.transpose(1, 2).contiguous()

        # 2nd conformer block
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        xs, pos_emb = self.up_embed(xs)

        xs = self._forward_impl_up_encoder(xs, masks, pos_emb)
        # post norm
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks

    @torch.compile(dynamic=True, backend="eager")
    def forward_chunk(
        self,
        xs: torch.Tensor,
        last_chunk: bool = False,
        cnn_cache: torch.Tensor = None,
        att_cache: torch.Tensor = None,
    ):
        """
        Args:
            xs: shape (b, dt, c)
            last_chunk: bool. If last chunk, will pad input with lookaheads
            att_cache: shape (depth1+depth2, b, nh, 2*t1, c).
            cnn_cache: shape (b, c, t1+t2). Where t1=2 (pre_lookahead_layer), t2=4 (up_layer)
        """
        if att_cache is not None:
            assert att_cache.shape[3] % 2 == 0, att_cache.shape
        if cnn_cache is not None:
            assert cnn_cache.shape[2] == 2 + self.up_layer.stride * 2, cnn_cache.shape

        # unpack caches
        offset1 = att_cache.shape[3] // 2 if att_cache is not None else 0
        att_cache1 = (
            att_cache[: len(self.encoders), :, :, :offset1] if att_cache is not None else [None] * len(self.encoders)
        )
        att_cache2 = att_cache[len(self.encoders) :] if att_cache is not None else [None] * len(self.encoders)
        cnn_cache1 = cnn_cache[:, :, :2] if cnn_cache is not None else None
        cnn_cache2 = cnn_cache[:, :, 2:] if cnn_cache is not None else None
        xs, _ = self.embed(xs)
        if last_chunk:
            xs = F.pad(xs, (0, 0, 0, self.pre_lookahead_layer.pre_lookahead_len))

        # this_cnn_cache: shape (b=1, c=512, t=2)
        xs, new_cnn_cache1 = self.pre_lookahead_layer.forward_chunk(xs, cache=cnn_cache1)

        # remake pos_emb, offset param is ignored by position_encoding
        pos_emb = self.embed.position_encoding(offset=None, size=offset1 + xs.shape[1])

        # first conformer
        chunk_masks = torch.zeros((0, 0, 0))
        new_att_cache1 = []

        for idx, layer in enumerate(self.encoders):
            # this_att_cache: shape (b, nh, t, c * 2)
            xs, _, this_new_att_cache1, _ = layer(xs, chunk_masks, pos_emb, att_cache=att_cache1[idx])
            new_att_cache1.append(this_new_att_cache1)
        new_att_cache1 = torch.stack(new_att_cache1, dim=0)

        # upsample + conformer encoder, xs: (b, t, c) -> (b, c, t)
        xs = xs.transpose(1, 2).contiguous()
        # this_cnn_cache: shape (b=1, c=512, t=2*2)
        xs, _, new_cnn_cache2 = self.up_layer.forward_chunk(xs, None, cache=cnn_cache2)
        xs = xs.transpose(1, 2).contiguous()

        # at this time, xs are doubled in length
        xs, _ = self.up_embed(xs)

        # remake pos_emb
        pos_emb = self.embed.position_encoding(offset=None, size=offset1 * self.up_layer.stride + xs.shape[1])

        # second conformer
        chunk_masks = torch.zeros((0, 0, 0), dtype=torch.bfloat16)
        new_att_cache2 = []

        for idx, layer in enumerate(self.up_encoders):
            xs, _, this_new_att_cache2, _ = layer(xs, chunk_masks, pos_emb, att_cache=att_cache2[idx])
            new_att_cache2.append(this_new_att_cache2)
        new_att_cache2 = torch.stack(new_att_cache2, dim=0)

        if self.normalize_before:
            xs = self.after_norm(xs)

        # pack new cache
        new_att_cache = torch.cat([new_att_cache1.repeat(1, 1, 1, 2, 1), new_att_cache2], dim=0)
        new_cnn_cache = torch.cat([new_cnn_cache1, new_cnn_cache2], dim=2)

        return xs, new_cnn_cache, new_att_cache


class MLP(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = ReplicatedLinear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = ReplicatedLinear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x, _ = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x, _ = self.fc2(x)
        return x


class Attention(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.scale = head_dim**-0.5

        # Keep DiT attention unfused: QKVParallelLinear/RowParallelLinear causes
        # measurable decoder drift from the official Step-Audio implementation.
        self.to_q = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(self.inner_dim, dim)

    def get_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, _ = x.shape
        q = self.to_q(x).view(b, t, self.num_heads, self.head_dim)
        k = self.to_k(x).view(b, t, self.num_heads, self.head_dim)
        v = self.to_v(x).view(b, t, self.num_heads, self.head_dim)
        q = q.reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.get_qkv(x)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn_mask = attn_mask.unsqueeze(1)
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
        )  # (b, nh, t, c)
        x = x.transpose(1, 2).reshape(b, t, -1)
        return self.proj(x)

    def forward_chunk(self, x: torch.Tensor, att_cache: torch.Tensor = None, attn_mask: torch.Tensor = None):
        b, t, c = x.shape
        q, k, v = self.get_qkv(x)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if att_cache is not None:
            if attn_mask is not None:
                k_cache, v_cache = att_cache.chunk(2, dim=3)
                k = torch.cat([k, k_cache], dim=2)
                v = torch.cat([v, v_cache], dim=2)

            else:
                k_cache, v_cache = att_cache.chunk(2, dim=3)
                k = torch.cat([k, k_cache], dim=2)
                v = torch.cat([v, v_cache], dim=2)

        # new {k,v}_cache
        new_att_cache = torch.cat([k, v], dim=3)
        # attn_mask = torch.ones((b, 1, t, t1), dtype=torch.bool, device=x.device)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # (b, nh, t, c)
        x = x.transpose(1, 2).reshape(b, t, -1)
        return self.proj(x), new_att_cache


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        # from SinusoidalPosEmb
        self.scale = 1000

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half) / half).to(t)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t * self.scale, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class Transpose(torch.nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor):
        x = torch.transpose(x, self.dim0, self.dim1)
        return x


class CausalConv1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size)
        self.causal_padding = (kernel_size - 1, 0)

    def forward(self, x: torch.Tensor):
        x = F.pad(x, self.causal_padding)
        x = super().forward(x)
        return x

    def forward_chunk(self, x: torch.Tensor, cnn_cache: torch.Tensor = None):
        if cnn_cache is None:
            cnn_cache = x.new_zeros((x.shape[0], self.in_channels, self.causal_padding[0]))
        x = torch.cat([cnn_cache, x], dim=2)
        new_cnn_cache = x[..., -self.causal_padding[0] :]
        x = super().forward(x)
        return x, new_cnn_cache


class CausalConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.block = torch.nn.Sequential(
            Transpose(1, 2),
            CausalConv1d(in_channels, out_channels, kernel_size),
            Transpose(1, 2),
            nn.LayerNorm(out_channels),
            nn.Mish(),
            Transpose(1, 2),
            CausalConv1d(out_channels, out_channels, kernel_size),
            Transpose(1, 2),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: shape (b, t, c)
            mask: shape (b, t, 1)
        """
        if mask is not None:
            x = x * mask
        x = self.block(x)
        if mask is not None:
            x = x * mask
        return x

    def forward_chunk(self, x: torch.Tensor, cnn_cache: torch.Tensor = None):
        """
        Args:
            x: shape (b, dt, c)
            cnn_cache: shape (b, c1+c2, 2)
        """
        if cnn_cache is not None:
            cnn_cache1, cnn_cache2 = cnn_cache.split((self.in_channels, self.out_channels), dim=1)
        else:
            cnn_cache1, cnn_cache2 = None, None
        x = self.block[0](x)
        x, new_cnn_cache1 = self.block[1].forward_chunk(x, cnn_cache1)
        x = self.block[2:6](x)
        x, new_cnn_cache2 = self.block[6].forward_chunk(x, cnn_cache2)
        x = self.block[7](x)
        new_cnn_cache = torch.cat((new_cnn_cache1, new_cnn_cache2), dim=1)
        return x, new_cnn_cache


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, head_dim=head_dim, qkv_bias=True, qk_norm=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        self.mlp = MLP(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.conv = CausalConvBlock(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor, attn_mask: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_conv, scale_conv, gate_conv = (
            self.adaLN_modulation(c).chunk(9, dim=-1)
        )
        attn_in = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.attn(attn_in, attn_mask)
        x = x + gate_conv * self.conv(modulate(self.norm3(x), shift_conv, scale_conv))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    def forward_chunk(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        cnn_cache: torch.Tensor = None,
        att_cache: torch.Tensor = None,
        mask: torch.Tensor = None,
    ):
        """
        Args:
            x: shape (b, dt, c)
            c: shape (b, 1, c)
            cnn_cache: shape (b, c1+c2, 2)
            att_cache: shape (b, nh, t, c * 2)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_conv, scale_conv, gate_conv = (
            self.adaLN_modulation(c).chunk(9, dim=-1)
        )
        x_att, new_att_cache = self.attn.forward_chunk(modulate(self.norm1(x), shift_msa, scale_msa), att_cache, mask)
        x = x + gate_msa * x_att
        x_conv, new_cnn_cache = self.conv.forward_chunk(modulate(self.norm3(x), shift_conv, scale_conv), cnn_cache)
        x = x + gate_conv * x_conv
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x, new_cnn_cache, new_att_cache


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mlp_ratio: float = 4.0,
        depth: int = 28,
        num_heads: int = 8,
        head_dim: int = 64,
        hidden_size: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.in_proj = nn.Linear(in_channels, hidden_size)

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, head_dim, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, self.out_channels)

        self.inference_buffers_chunk = {}
        self.max_size_chunk = {}

        self.register_buffer("att_cache_buffer", torch.zeros((16, 2, 8, 1000, 128)), persistent=False)
        self.register_buffer("cnn_cache_buffer", torch.zeros((16, 2, 1024, 2)), persistent=False)

    def forward(self, x, mask, mu, t, spks=None, cond=None):
        """Args:
        x: shape (b, c, t)
        mask: shape (b, 1, t)
        t: shape (b,)
        spks: shape (b, c)
        cond: shape (b, c, t)
        """
        t = self.t_embedder(t).unsqueeze(1)  # (b, 1, c)
        x = torch.cat([x, mu], dim=1)
        if spks is not None:
            spks = spks.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            x = torch.cat([x, spks], dim=1)
        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        return self.blocks_forward(x, t, mask)

    def blocks_forward(self, x, t, mask):
        x = x.transpose(1, 2)
        attn_mask = mask.bool()
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x, t, attn_mask)
        x = self.final_layer(x, t)
        x = x.transpose(1, 2)
        return x

    def forward_chunk(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        cnn_cache: torch.Tensor = None,
        att_cache: torch.Tensor = None,
    ):
        """
        Args:
            x: shape (b, dt, c)
            mu: shape (b, dt, c)
            t: shape (b,)
            spks: shape (b, c)
            cond: shape (b, dt, c)
            cnn_cache: shape (depth, b, c1+c2, 2)
            att_cache: shape (depth, b, nh, t, c * 2)
        """
        t = self.t_embedder(t).unsqueeze(1)  # (b, 1, c)
        x = torch.cat([x, mu], dim=1)
        if spks is not None:
            spks = spks.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            x = torch.cat([x, spks], dim=1)
        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        # create fake cache
        if cnn_cache is None:
            cnn_cache = [None] * len(self.blocks)
        if att_cache is None:
            att_cache = [None] * len(self.blocks)
        if att_cache[0] is not None:
            last_att_len = att_cache.shape[3]
        else:
            last_att_len = 0
        chunk_size = x.shape[2]
        mask = torch.ones(x.shape[0], chunk_size, last_att_len + chunk_size, dtype=torch.bool, device=x.device)
        mask = None
        x = self.blocks_forward_chunk(x, t, mask, cnn_cache, att_cache, self.cnn_cache_buffer, self.att_cache_buffer)
        new_cnn_cache = self.cnn_cache_buffer
        new_att_cache = self.att_cache_buffer[:, :, :, : last_att_len + chunk_size, :]

        return x, new_cnn_cache, new_att_cache

    def blocks_forward_chunk(
        self, x, t, mask, cnn_cache=None, att_cache=None, cnn_cache_buffer=None, att_cache_buffer=None
    ):
        x = x.transpose(1, 2)
        x = self.in_proj(x)
        for b_idx, block in enumerate(self.blocks):
            x, this_new_cnn_cache, this_new_att_cache = block.forward_chunk(
                x, t, cnn_cache[b_idx], att_cache[b_idx], mask
            )
            cnn_cache_buffer[b_idx] = this_new_cnn_cache
            att_cache_buffer[b_idx][:, :, : this_new_att_cache.shape[2], :] = this_new_att_cache
        x = self.final_layer(x, t)
        x = x.transpose(1, 2)
        return x


class CausalMaskedDiffWithXvec(torch.nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        input_embedding,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: str = "mel",
        vocab_size: int = 5121,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_embedding = input_embedding
        self.encoder = encoder
        self.decoder = decoder
        self.pre_lookahead_len = int(self.encoder.pre_lookahead_layer.pre_lookahead_len)
        self.up_rate = int(self.encoder.up_layer.stride)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)

    @torch.inference_mode()
    def inference(
        self,
        token,
        token_len,
        prompt_token,
        prompt_token_len,
        prompt_feat,
        prompt_feat_len,
        embedding,
        n_timesteps: int = 10,
    ):
        assert token.shape[0] == 1

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        token_len = prompt_token_len + token_len
        token = torch.concat([prompt_token, token], dim=1)

        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # token encode
        h, _ = self.encoder.forward(token, token_len)
        h = self.encoder_proj(h)

        # condition
        mel_len1 = prompt_feat.shape[1]
        mel_len2 = h.shape[1] - prompt_feat.shape[1]

        conds = torch.zeros_like(h)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2).contiguous()

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat, _ = self.decoder.forward(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=n_timesteps,
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat

    @torch.inference_mode()
    def setup_cache(
        self,
        token: torch.Tensor,
        mel: torch.Tensor,
        spk: torch.Tensor,
        n_timesteps: int = 10,
    ):
        """
        Args:
            token: shape (b, t), with look ahead tokens
            mel: shape (b, t, c), groundtruth mel
            spk: shape (b, 192), speaker embedding
        Returns:
            cache: dict {
                'conformer': {'cnn_cache': xxx, 'att_cache': xxx},
                'estimator': {'cnn_cache': xxx, 'att_cache': xxx}
            }
        """
        # check if look ahead token included
        assert (token.shape[1] - self.pre_lookahead_len) * self.up_rate == mel.shape[1], (token.shape, mel.shape)

        # xvec projection
        spk = F.normalize(spk, dim=1)
        spk = self.spk_embed_affine_layer(spk)

        token = self.input_embedding(token)
        # NOTE encoder.forward_chunk will strip the look ahead part
        h, conformer_cnn_cache, conformer_att_cache = self.encoder.forward_chunk(
            xs=token,
            last_chunk=False,
            cnn_cache=None,
            att_cache=None,
        )
        h = self.encoder_proj(h)

        feat, estimator_cnn_cache, estimator_att_cache = self.decoder.forward_chunk(
            mu=h.transpose(1, 2).contiguous(),
            spks=spk,
            cond=mel.transpose(1, 2).contiguous(),
            n_timesteps=n_timesteps,
            temperature=1.0,
            cnn_cache=None,
            att_cache=None,
        )

        cache = {
            "conformer_cnn_cache": conformer_cnn_cache,
            "conformer_att_cache": conformer_att_cache,
            "estimator_cnn_cache": estimator_cnn_cache,
            "estimator_att_cache": estimator_att_cache,
        }
        return cache

    @torch.inference_mode()
    def inference_chunk(
        self,
        token: torch.Tensor,
        spk: torch.Tensor,
        cache: dict,
        last_chunk: bool = False,
        n_timesteps: int = 10,
    ):
        """
        Args:
            token: shape (b, t), with look ahead tokens
            spk: shape (b, 192), speaker embedding
            cache: dict {
                'conformer_cnn_cache': xxx,
                ...
            }
        """
        # unpack cache
        conformer_cnn_cache = cache["conformer_cnn_cache"]
        conformer_att_cache = cache["conformer_att_cache"]
        estimator_cnn_cache = cache["estimator_cnn_cache"]
        estimator_att_cache = cache["estimator_att_cache"]

        # xvec projection
        spk = F.normalize(spk, dim=1)
        spk = self.spk_embed_affine_layer(spk)

        token = self.input_embedding(token)
        # if not the last chunk, h is shorter than xs for a length of lookahead_length * stride (6)
        h, conformer_cnn_cache, conformer_att_cache = self.encoder.forward_chunk(
            xs=token,
            last_chunk=last_chunk,
            cnn_cache=conformer_cnn_cache,
            att_cache=conformer_att_cache,
        )
        h = self.encoder_proj(h)

        cond = torch.zeros_like(h)
        # forward estimator
        feat, estimator_cnn_cache, estimator_att_cache = self.decoder.forward_chunk(
            mu=h.transpose(1, 2).contiguous(),
            spks=spk,
            cond=cond.transpose(1, 2).contiguous(),
            n_timesteps=n_timesteps,
            temperature=1.0,
            cnn_cache=estimator_cnn_cache,
            att_cache=estimator_att_cache,
        )

        new_cache = {
            "conformer_cnn_cache": conformer_cnn_cache,
            "conformer_att_cache": conformer_att_cache,
            "estimator_cnn_cache": estimator_cnn_cache,
            "estimator_att_cache": estimator_att_cache,
        }

        return feat, new_cache

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)
