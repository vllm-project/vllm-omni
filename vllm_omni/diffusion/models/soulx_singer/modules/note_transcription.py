import math

import numpy as np
import packaging.version as version
import torch
import torch.nn as nn

# ---------------------- note transcription common layers ----------------------


class LayerNorm(nn.LayerNorm):
    def __init__(self, nout: int, dim: int = -1, eps: float = 1e-5):
        """Construct an LayerNorm object."""
        super().__init__(nout, eps=eps)
        self.dim = dim

    def forward(self, x):
        if self.dim == -1:
            return super().forward(x)
        return super().forward(x.transpose(1, -1)).transpose(1, -1)


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.permute(self.args)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class ResidualBlock(nn.Module):
    """
    Implements a residual block pattern:
    [Norm] → [Conv1d] → [Scale] → [Activation] → [Conv1d]
    repeated n times, each as a residual unit.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        n: int = 2,
        c_multiple: int = 2,
        ln_eps: float = 1e-12,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    LayerNorm(channels, dim=1, eps=ln_eps),
                    nn.Conv1d(
                        channels,
                        c_multiple * channels,
                        kernel_size,
                        dilation=dilation,
                        padding=(dilation * (kernel_size - 1)) // 2,
                    ),
                    LambdaLayer(lambda x: x * kernel_size**-0.5),  # scaling for stability
                    nn.LeakyReLU(),
                    nn.Conv1d(c_multiple * channels, channels, kernel_size=1, dilation=dilation),
                )
                for _ in range(n)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nonpadding = (x.abs().sum(1) > 0).float()[:, None, :]
        for block in self.blocks:
            x_res = block(x)
            x = (x + x_res) * nonpadding
        return x


class ConvBlocks(nn.Module):
    """
    Decodes the expanded phoneme encoding into spectrograms.
    """

    def __init__(
        self,
        hidden_size,
        out_dims,
        dilations,
        kernel_size,
        layers_in_block=2,
        c_multiple=2,
        ln_eps=1e-5,
        is_btc=True,
        post_net_kernel=3,
    ):
        super().__init__()
        self.is_btc = is_btc

        res_blocks = [
            ResidualBlock(
                hidden_size,
                kernel_size=kernel_size,
                dilation=1,
                n=layers_in_block,
                c_multiple=c_multiple,
                ln_eps=ln_eps,
            )
            for _ in range(1)
        ]  # default 1
        self.res_blocks = nn.Sequential(*res_blocks)

        # Output normalization and conv
        self.last_norm = LayerNorm(hidden_size, dim=1, eps=ln_eps)
        self.post_net1 = nn.Conv1d(hidden_size, out_dims, kernel_size=post_net_kernel, padding=post_net_kernel // 2)

    def forward(self, x: torch.Tensor, nonpadding: torch.Tensor | None = None) -> torch.Tensor:
        if self.is_btc:
            x = x.transpose(1, 2)
        if nonpadding is None:
            nonpadding = (x.abs().sum(1) > 0).float()[:, None, :]
        elif self.is_btc:
            nonpadding = nonpadding.transpose(1, 2)
        x = self.res_blocks(x) * nonpadding
        x = self.last_norm(x) * nonpadding
        x = self.post_net1(x) * nonpadding
        if self.is_btc:
            x = x.transpose(1, 2)
        return x


def Linear(in_features: int, out_features: int, bias: bool = True, init_type: str = "xavier"):
    m = nn.Linear(in_features, out_features, bias)
    if init_type == "xavier":
        nn.init.xavier_uniform_(m.weight)
    elif init_type == "kaiming":
        nn.init.kaiming_normal_(m.weight, mode="fan_in")
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def Embedding(num_embeddings, embedding_dim, padding_idx=None, init_type="normal"):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    if init_type == "normal":
        nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    elif init_type == "kaiming":
        nn.init.kaiming_normal_(m.weight, mode="fan_in")
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


# ---------------------- note transcription espnet attention ----------------------


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, n_head, n_feat, dropout_rate, flash=False):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)
        self.dropout_rate = dropout_rate
        self.flash = flash

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.
        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).
        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).
        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = float(np.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value)
        if version.parse(torch.__version__) >= version.parse("2.0") and self.flash:
            n_batch = value.size(0)
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask.unsqueeze(1) if mask is not None else None, dropout_p=self.dropout_rate
            )
            x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
            return self.linear_out(x)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            return self.forward_attention(v, scores, mask)


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x, zero_triu=False):
        """Compute relative positinal encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, size).
            zero_triu (bool): If true, return the lower triangular part of the matrix.
        Returns:
            torch.Tensor: Output tensor.
        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            pos_emb (torch.Tensor): Positional embedding tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
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
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)


# ---------------------- note transcription positional encoding ----------------------


class PositionalEncoding(torch.nn.Module):
    """Positional encoding.
    Args:
        d_model (int): Embedding dimension.
        max_len (int): Maximum input length.
        reverse (bool): Whether to reverse the input position.
    """

    def __init__(self, d_model, max_len=5000, reverse=False):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.pe = None

        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        if self.reverse:
            position = torch.arange(x.size(1) - 1, -1, -1.0, dtype=torch.float32).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return x


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        max_len (int): Maximum input length.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__(d_model, max_len, reverse=True)

    def forward(self, x):
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding tensor (1, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.pe[:, : x.size(1)]
        return x, pos_emb


# ---------------------- note transcription conformer layers ----------------------


class ConvolutionModule(nn.Module):
    def __init__(self, channels: int, kernel_size: int, activation: nn.Module = nn.ReLU(), bias: bool = True):
        super().__init__()
        if (kernel_size - 1) % 2 != 0:
            raise ValueError("kernel_size should be a odd number for 'SAME' padding")

        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, kernel_size=1, bias=bias)
        self.depthwise_conv = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, groups=channels, bias=bias
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=bias)
        self.activation = activation

    def forward(self, x):
        # (batch, time, channels) -> (batch, channels, time)
        x = x.transpose(1, 2)

        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=1)

        x = self.depthwise_conv(x)
        x = self.norm(x)
        x = self.activation(x)

        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        return x


class MultiLayeredConv1d(nn.Module):
    """
    Multi-layered Conv1d that replaces feed-forward in Transformer block.

    Reference:
        FastSpeech: Fast, Robust and Controllable Text to Speech
        https://arxiv.org/pdf/1905.09263.pdf
    """

    def __init__(self, in_chans, hidden_chans, kernel_size):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.w_1 = nn.Conv1d(in_chans, hidden_chans, kernel_size, padding=padding)
        self.w_2 = nn.Conv1d(hidden_chans, in_chans, kernel_size, padding=padding)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (batch, time, in_chans)
        Returns:
            torch.Tensor: (batch, time, in_chans)
        """
        # Conv1d expects (batch, channels, time)
        x = x.transpose(1, 2)
        x = torch.relu(self.w_1(x))
        x = self.w_2(x)
        x = x.transpose(1, 2)
        return x


class Swish(nn.Module):
    """Swish activation function."""

    def forward(self, x):
        return x * torch.sigmoid(x)


class EncoderLayer(nn.Module):
    """
    A single encoder layer: optional macaron FFN, self-attn, conv, FFN, norms.
    Args:
        size (int): Input dimension.
        self_attn (nn.Module): Self-attention module.
        feed_forward (nn.Module): Standard feed-forward network.
        feed_forward_macaron (nn.Module): Optional, second FFN in macaron structure.
        conv_module (nn.Module): Optional, convolution module.
        normalize_before (bool): If True, use pre-layer norm. Else, post-layer norm.
        concat_after (bool): If True, concat attn in/out, else residual.
    """

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        feed_forward_macaron,
        conv_module,
        normalize_before=True,
        concat_after=False,
    ):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module

        self.norm_ff = LayerNorm(size)
        self.norm_mha = LayerNorm(size)

        self.ff_scale = 0.5 if feed_forward_macaron is not None else 1.0
        self.norm_ff_macaron = LayerNorm(size) if feed_forward_macaron is not None else None

        if conv_module is not None:
            self.norm_conv = LayerNorm(size)
            self.norm_final = LayerNorm(size)
        else:
            self.norm_conv = None
            self.norm_final = None

        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(2 * size, size)

    def forward(self, x_input, mask, cache=None) -> tuple:
        """
        Forward pass for the encoder layer.

        Args:
            x_input (Tensor or Tuple): Input tensor of shape (batch, time, size)
                or tuple (x, pos_emb) where pos_emb is the positional embedding.
            mask (Tensor): Mask tensor of shape (batch, time) or (batch, 1, time).
            cache (Tensor, optional): Input cache for streaming (can be None).

        Returns:
            Tuple[Tensor, Tensor]: (outputs, mask), where outputs either includes
                pos_emb or not, depending on input.
        """
        # Unpack possible (x, pos_emb) tuple input
        if isinstance(x_input, tuple):
            x, pos_emb = x_input
        else:
            x, pos_emb = x_input, None

        # --- Macaron Feed-Forward (first FFN) ---
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.feed_forward_macaron(x)
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # --- Self-Attention ---
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        # Handle cache for streaming
        if cache is not None:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = mask[:, -1:, :] if mask is not None else None
        else:
            x_q = x

        # Apply self-attention (with or without positional embeddings)
        if pos_emb is not None:
            attn_out = self.self_attn(x_q, x, x, pos_emb, mask)
        else:
            attn_out = self.self_attn(x_q, x, x, mask)

        if self.concat_after:
            x = torch.cat((x, attn_out), dim=-1)
            x = residual + self.concat_linear(x)
        else:
            x = residual + attn_out
        if not self.normalize_before:
            x = self.norm_mha(x)

        # --- Convolution Module ---
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x = residual + self.conv_module(x)
            if not self.normalize_before:
                x = self.norm_conv(x)

        # --- Feed-Forward (second FFN) ---
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.feed_forward(x)
        if not self.normalize_before:
            x = self.norm_ff(x)

        # --- Final normalization, if present ---
        if self.norm_final is not None:
            x = self.norm_final(x)

        # --- Append to cache if streaming ---
        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        # --- Final return, with or without positional embedding ---
        if pos_emb is not None:
            return (x, pos_emb), mask
        else:
            return x, mask


class ConformerLayers(nn.Module):
    def __init__(self, hidden_size, num_layers, kernel_size=9, num_heads=4, use_last_norm=True, save_hidden=False):
        super().__init__()
        self.use_last_norm = use_last_norm
        self.layers = nn.ModuleList()
        positionwise_layer = MultiLayeredConv1d
        positionwise_layer_args = (hidden_size, hidden_size * 4, 1)
        self.pos_embed = RelPositionalEncoding(hidden_size)
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    hidden_size,
                    RelPositionMultiHeadedAttention(num_heads, hidden_size, 0.0),
                    positionwise_layer(*positionwise_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    ConvolutionModule(hidden_size, kernel_size, Swish()),
                )
                for _ in range(num_layers)
            ]
        )
        if self.use_last_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = nn.Linear(hidden_size, hidden_size)
        self.save_hidden = save_hidden
        if save_hidden:
            self.hiddens = []

    def forward(self, x, padding_mask=None):
        """

        :param x: [B, T, H]
        :param padding_mask: [B, T]
        :return: [B, T, H]
        """
        self.hiddens = []
        nonpadding_mask = x.abs().sum(-1) > 0
        x = self.pos_embed(x)
        for layer in self.encoder_layers:
            x, mask = layer(x, nonpadding_mask[:, None, :])
            if self.save_hidden:
                self.hiddens.append(x[0])
        x = x[0]
        x = self.layer_norm(x) * nonpadding_mask.float()[:, :, None]
        return x


# ---------------------- note transcription backbone layers ----------------------


class UnetDown(nn.Module):
    def __init__(
        self,
        hidden_size,
        n_layers,
        kernel_size,
        down_rates,
        channel_multiples=None,
        is_btc=True,
        constant_channels=False,
    ):
        super().__init__()
        assert n_layers == len(down_rates)  # downs, down sample rate
        down_rates = [int(i) for i in down_rates]
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.is_btc = is_btc
        channel_multiples = channel_multiples if channel_multiples is not None else down_rates
        self.layers = nn.ModuleList()
        self.downs = nn.ModuleList()
        in_channels = hidden_size
        for i in range(self.n_layers):
            out_channels = int(in_channels * channel_multiples[i]) if not constant_channels else in_channels
            self.layers.append(
                nn.Sequential(
                    ResidualBlock(in_channels, kernel_size, dilation=1, n=1, c_multiple=1, ln_eps=1e-5),
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2),
                    ResidualBlock(out_channels, kernel_size, dilation=1, n=1, c_multiple=1, ln_eps=1e-5),
                )
            )
            self.downs.append(nn.Sequential(nn.AvgPool1d(down_rates[i])))
            in_channels = out_channels
        self.last_norm = LayerNorm(out_channels, dim=1, eps=1e-6)
        self.post_net = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x, **kwargs):
        # x [B, T, C]
        if self.is_btc:
            x = x.transpose(1, 2)  # [B, C, T]
        skip_xs = []
        for i in range(self.n_layers):
            skip_x = self.layers[i](x)
            x = self.downs[i](skip_x)
            if self.is_btc:
                skip_xs.append(skip_x.transpose(1, 2))  # [B, T, C]
            else:
                skip_xs.append(skip_x)
        x = self.post_net(self.last_norm(x))
        if self.is_btc:
            x = x.transpose(1, 2)
        return x, skip_xs


class UnetMid(nn.Module):
    def __init__(self, hidden_size, kernel_size, n_layers=None, in_dims=None, out_dims=None, is_btc=True, net=None):
        super().__init__()
        in_dims = in_dims if in_dims is not None else hidden_size
        out_dims = out_dims if out_dims is not None else hidden_size
        self.pre = nn.Conv1d(in_dims, hidden_size, kernel_size, padding=kernel_size // 2)
        self.post = nn.Conv1d(hidden_size, out_dims, kernel_size, padding=kernel_size // 2)
        self.is_btc = is_btc
        if net is not None:
            self.net = net
        else:
            self.net = ConvBlocks(
                hidden_size,
                out_dims=hidden_size,
                dilations=None,
                kernel_size=kernel_size,
                layers_in_block=2,
                c_multiple=2,
                post_net_kernel=3,
                is_btc=is_btc,
            )

    def forward(self, x, cond=None, **kwargs):
        # x [B, T, C]
        if self.is_btc:
            x = self.pre(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = self.pre(x)
        if cond is None:
            cond = 0
        x = self.net(x + cond)
        if self.is_btc:
            x = self.post(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = self.post(x)
        return x


class UnetUp(nn.Module):
    def __init__(
        self,
        hidden_size,
        n_layers,
        kernel_size,
        up_rates,
        channel_multiples=None,
        is_btc=True,
        constant_channels=False,
        use_skip_layer=False,
        skip_scale=1.0,
    ):
        super().__init__()
        assert n_layers == len(up_rates)  # this is reversed in up module, from the output to the interface with middle
        up_rates = [int(i) for i in up_rates]
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.is_btc = is_btc
        self.skip_scale = skip_scale
        channel_multiples = channel_multiples if channel_multiples is not None else up_rates
        # in_channels = int(np.cumprod(channel_multiples)[-1] * hidden_size) if not constant_channels else hidden_size
        self.in_channels_lst = (
            (np.cumprod([1] + channel_multiples) * hidden_size).astype(int)
            if not constant_channels
            else [hidden_size for _ in range(self.n_layers + 1)]
        )
        in_channels = self.in_channels_lst[-1]
        self.ups = nn.ModuleList()
        self.skip_layers = nn.ModuleList()
        self.layers = nn.ModuleList()
        for i in range(self.n_layers - 1, -1, -1):
            out_channels = self.in_channels_lst[i] if not constant_channels else in_channels
            self.ups.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        in_channels,
                        in_channels,
                        kernel_size=kernel_size,
                        stride=up_rates[i],
                        padding=kernel_size // 2,
                        output_padding=up_rates[i] - 1,
                    ),
                    LayerNorm(in_channels, dim=1, eps=1e-6),
                    nn.LeakyReLU(),
                )
            )
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels * 2, out_channels, kernel_size, padding=(kernel_size - 1) // 2),
                    ResidualBlock(out_channels, kernel_size, dilation=1, n=1, c_multiple=1, ln_eps=1e-5),
                )
            )
            if use_skip_layer:
                self.skip_layers.append(
                    ResidualBlock(in_channels, kernel_size, dilation=1, n=1, c_multiple=1, ln_eps=1e-5)
                )
            else:
                self.skip_layers.append(nn.Identity())

            in_channels = out_channels
        self.out_channels = out_channels
        self.last_norm = LayerNorm(out_channels, dim=1, eps=1e-6)
        self.post_net = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x, skips, **kwargs):
        # x [B, T, C]
        if self.is_btc:
            x = x.transpose(1, 2)  # [B, C, T]
        for i in range(self.n_layers):
            x = self.ups[i](x)
            skip_x = (
                skips[self.n_layers - i - 1] if not self.is_btc else skips[self.n_layers - i - 1].transpose(1, 2)
            )  # [B, T, C] -> [B, C, T]
            skip_x = self.skip_layers[i](skip_x) * self.skip_scale
            x = torch.cat((x, skip_x), dim=1)  # [B, C, T]
            x = self.layers[i](x)
        x = self.post_net(self.last_norm(x))
        if self.is_btc:
            x = x.transpose(1, 2)
        return x


class Unet(nn.Module):
    def __init__(
        self,
        hidden_size,
        down_layers,
        up_layers,
        kernel_size,
        updown_rates,
        mid_layers=None,
        channel_multiples=None,
        is_btc=True,
        constant_channels=False,
        mid_net=None,
        use_skip_layer=False,
        skip_scale=1.0,
    ):
        super().__init__()
        assert len(updown_rates) == down_layers == up_layers, f"{len(updown_rates)}, {down_layers}, {up_layers}"
        if channel_multiples is not None:
            assert len(channel_multiples) == len(updown_rates)
        else:
            channel_multiples = updown_rates
        self.down = UnetDown(
            hidden_size, down_layers, kernel_size, updown_rates, channel_multiples, is_btc, constant_channels
        )
        down_out_dims = int(np.cumprod(channel_multiples)[-1] * hidden_size) if not constant_channels else hidden_size
        self.mid = UnetMid(
            hidden_size,
            kernel_size,
            mid_layers,
            in_dims=down_out_dims,
            out_dims=down_out_dims,
            is_btc=is_btc,
            net=mid_net,
        )
        self.up = UnetUp(
            hidden_size,
            up_layers,
            kernel_size,
            updown_rates,
            channel_multiples,
            is_btc,
            constant_channels,
            use_skip_layer,
            skip_scale,
        )

    def forward(self, x, mid_cond=None, **kwargs):
        x, skips = self.down(x)
        x = self.mid(x, mid_cond)
        x = self.up(x, skips)
        return x


class BackboneNet(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hidden_size = hidden_size = hparams["hidden_size"]
        updown_rates = [2, 2, 2]
        channel_multiples = [1, 1, 1]
        if hparams.get("updown_rates", None) is not None:
            updown_rates = [int(i) for i in hparams.get("updown_rates", None).split("-")]
        if hparams.get("channel_multiples", None) is not None:
            channel_multiples = [float(i) for i in hparams.get("channel_multiples", None).split("-")]
        assert len(updown_rates) == len(channel_multiples)
        # convs
        if hparams.get("bkb_net", "conv") == "conv":
            self.net = Unet(
                hidden_size,
                down_layers=len(updown_rates),
                mid_layers=hparams.get("bkb_layers", 12),
                up_layers=len(updown_rates),
                kernel_size=3,
                updown_rates=updown_rates,
                channel_multiples=channel_multiples,
                is_btc=True,
                constant_channels=False,
                mid_net=None,
                use_skip_layer=hparams.get("unet_skip_layer", False),
            )
        # conformer
        elif hparams.get("bkb_net", "conv") == "conformer":
            mid_net = ConformerLayers(
                hidden_size,
                num_layers=hparams.get("bkb_layers", 12),
                kernel_size=hparams.get("conformer_kernel", 9),
                num_heads=4,
            )
            self.net = Unet(
                hidden_size,
                down_layers=len(updown_rates),
                up_layers=len(updown_rates),
                kernel_size=3,
                updown_rates=updown_rates,
                channel_multiples=channel_multiples,
                is_btc=True,
                constant_channels=False,
                mid_net=mid_net,
                use_skip_layer=hparams.get("unet_skip_layer", False),
            )

    def forward(self, x):
        return self.net(x)
