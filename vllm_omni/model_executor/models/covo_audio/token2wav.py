# Copyright 2026 Tencent.
# token2wav: audio token codes -> waveform (inference only)
# Pipeline: Token -> Latent (flow matching) -> Waveform (BigVGAN)

import math
from collections import OrderedDict, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import pow, sin
from torch.nn import Parameter
from torch.nn.utils import weight_norm

# --------------- Utilities ---------------


class JsonHParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = JsonHParams(**v)
            if isinstance(v, str) and v.lower() in ["non", "none", "nil", "null"]:
                v = None
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

    def get(self, key, default=None):
        return self.__dict__.get(key, default)


def exists(val):
    return val is not None


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@torch.autocast(enabled=False, device_type="cuda")
def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()


if "sinc" in dir(torch):
    sinc = torch.sinc
else:

    def sinc(x: torch.Tensor):
        return torch.where(
            x == 0, torch.tensor(1.0, device=x.device, dtype=x.dtype), torch.sin(math.pi * x) / math.pi / x
        )


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
        filter_ /= filter_.sum()

    return filter_.view(1, 1, kernel_size)


# --------------- Basic Layers ---------------


class Dropout(nn.Module):
    def __init__(self, p: float = 0.5, inplace: bool = False, force_drop: bool = False, **kwargs):
        super().__init__()
        if p < 0.0 or p > 1.0:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.inplace = inplace
        self.force_drop = force_drop

    def forward(self, x, **kwargs):
        return F.dropout(x, p=self.p, training=True if self.force_drop else self.training, inplace=self.inplace)


class EmbeddingTable(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, pad_id=-1, **kwargs):
        super().__init__(num_embeddings, embedding_dim, **kwargs)
        nn.init.normal_(self.weight, 0.0, embedding_dim**-0.5)
        self.pad_id = pad_id
        self.output_dim = embedding_dim

    def forward(self, x):
        if self.pad_id is not None:
            mask = x == self.pad_id
            x = x.masked_fill(mask, 0)
        outputs = super().forward(x)
        if self.pad_id is not None:
            outputs = outputs.masked_fill(mask.unsqueeze(-1), 0.0)
        return outputs


class Linear(nn.Linear):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        w_init_gain: str = "linear",
        activation=None,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, bias=bias)
        self.activation = activation if activation is not None else nn.Identity()
        self.output_dim = out_channels
        if w_init_gain is not None:
            if isinstance(w_init_gain, str):
                gain = nn.init.calculate_gain(w_init_gain)
            else:
                gain = w_init_gain
            nn.init.xavier_uniform_(self.weight, gain=gain)
        if bias:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x, **kwargs):
        return self.activation(super().forward(x))


class Conv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        bias: bool = True,
        padding=None,
        causal: bool = False,
        bn: bool = False,
        activation=None,
        w_init_gain=None,
        input_transpose: bool = False,
        **kwargs,
    ):
        self.causal = causal
        if padding is None:
            if causal:
                padding = 0
                self.left_padding = dilation * (kernel_size - 1)
            else:
                padding = get_padding(kernel_size, dilation)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            bias=bias,
        )

        self.in_channels = in_channels
        self.transpose = input_transpose
        self.bn = nn.BatchNorm1d(out_channels) if bn else nn.Identity()
        self.activation = activation if activation is not None else nn.Identity()
        if w_init_gain is not None:
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        if self.transpose or x.size(1) != self.in_channels:
            assert x.size(2) == self.in_channels
            x = x.transpose(1, 2)
            self.transpose = True

        if self.causal:
            x = F.pad(x.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)

        outputs = self.activation(self.bn(super().forward(x)))
        return outputs.transpose(1, 2) if self.transpose else outputs


class ConvTranspose1d(nn.ConvTranspose1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding=None,
        padding_mode: str = "zeros",
        causal: bool = False,
        input_transpose: bool = False,
        **kwargs,
    ):
        if padding is None:
            padding = 0 if causal else (kernel_size - stride) // 2
        if causal:
            assert padding == 0, "padding is not allowed in causal ConvTranspose1d."
            assert kernel_size == 2 * stride, "kernel_size must be equal to 2*stride in Causal ConvTranspose1d."

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
        )

        self.causal = causal
        self.stride = stride
        self.transpose = input_transpose

    def forward(self, x):
        if self.transpose or x.size(1) != self.in_channels:
            assert x.size(2) == self.in_channels
            x = x.transpose(1, 2)
            self.transpose = True

        x = super().forward(x)
        if self.causal:
            x = x[:, :, : -self.stride]
        return x.transpose(1, 2) if self.transpose else x


class ConvPositionEmbed(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int = 31, groups: int = 16):
        super().__init__()
        self.conv = Conv1d(
            hidden_size, hidden_size, kernel_size, groups=groups, input_transpose=True, activation=nn.GELU()
        )

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)
        x = self.conv(x)
        if mask is not None:
            x = x.masked_fill(~mask, 0.0)
        return x


# --------------- Anti-aliasing (BigVGAN) ---------------


class LowPassFilter1d(nn.Module):
    def __init__(
        self,
        cutoff=0.5,
        half_width=0.6,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "replicate",
        kernel_size: int = 12,
    ):
        super().__init__()
        if cutoff < -0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter)

    def forward(self, x):
        _, C, _ = x.shape
        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        out = F.conv1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)
        return out


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size)
        self.register_buffer("filter", filter)

    def forward(self, x):
        _, C, _ = x.shape
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)
        x = x[..., self.pad_left : -self.pad_right]
        return x


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, stride=ratio, kernel_size=self.kernel_size
        )

    def forward(self, x):
        return self.lowpass(x)


class Activation1d(nn.Module):
    def __init__(
        self, activation, up_ratio: int = 2, down_ratio: int = 2, up_kernel_size: int = 12, down_kernel_size: int = 12
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x


# --------------- Activations ---------------


class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable
        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)
        return x


# --------------- Attention ---------------

AttentionConfig = namedtuple("AttentionConfig", ["enable_flash", "enable_math", "enable_mem_efficient"])


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=50000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    @property
    def device(self):
        return self.inv_freq.device

    @torch.autocast(enabled=False, device_type="cuda")
    def forward(self, t):
        if not torch.is_tensor(t):
            t = torch.arange(t, device=self.device)
        t = t.type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        dropout: float = 0.0,
        max_position_embeddings: int = 4096,
        norm_layer: nn.Module = nn.LayerNorm,
        rotary_bias: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.rotary_bias = rotary_bias

        self.q_proj = Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.k_proj = Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.v_proj = Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = Dropout(attn_drop)
        self.o_proj = Linear(hidden_size, hidden_size)
        self.o_dropout = Dropout(dropout)

        self.cpu_config = AttentionConfig(True, True, True)
        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))
        if device_properties.major == 8 and device_properties.minor == 0:
            self.cuda_config = AttentionConfig(True, True, True)
        else:
            self.cuda_config = AttentionConfig(False, True, True)

        if self.rotary_bias:
            self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, q, k=None, v=None, mask=None):
        k = k or q
        v = v or q
        B, L, C = q.shape
        B, S, C = v.shape
        if mask is not None:
            if mask.ndim == 2:
                assert L == S
                mask = rearrange(mask, "b j -> b 1 1 j")
                mask = mask.expand(-1, self.num_heads, L, -1)
            elif mask.ndim == 3:
                assert mask.size(1) == L and mask.size(2) == S
                mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        q, k = self.q_norm(q), self.k_norm(k)

        config = self.cuda_config if q.is_cuda else self.cpu_config
        attn_bias = torch.zeros(B, self.num_heads, L, S, dtype=q.dtype, device=q.device)

        if self.rotary_bias:
            if L == S:
                rotary_emb = self.rotary(L)
                q, k = map(lambda x: apply_rotary_pos_emb(rotary_emb, x), (q, k))
            else:
                q_rotary_emb = self.rotary(L)
                k_rotary_emb = self.rotary(S)
                q = apply_rotary_pos_emb(q_rotary_emb, q)
                k = apply_rotary_pos_emb(k_rotary_emb, k)

        if mask is not None:
            attn_bias.masked_fill_(mask.logical_not(), float("-inf"))

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias, dropout_p=self.attn_drop.p if self.training else 0.0
            )

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.o_dropout(self.o_proj(out))
        return out


# --------------- DiT Modules ---------------


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = Linear(hidden_size, output_size, bias=True)

    def forward(self, x, c, mask=None):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


# --------------- Transformer ---------------


class Mlp(nn.Module):
    def __init__(self, hidden_size, ffn_hidden_size=4096, act_layer=nn.GELU, dropout=0.0, **kwargs):
        super().__init__()
        self.fc1 = Linear(hidden_size, ffn_hidden_size)
        self.act = act_layer()
        self.fc2 = Linear(ffn_hidden_size, hidden_size)
        self.drop = Dropout(dropout)

    def forward(self, x, **kwargs):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        ffn: nn.Module,
        hidden_size: int = 1024,
        modulation: bool = False,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=not modulation, eps=eps)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=not modulation, eps=eps)
        self.attn = attention
        self.ffn = ffn
        self.modulation = modulation
        if modulation:
            self.modulation_layer = nn.Sequential(nn.SiLU(), Linear(hidden_size, 6 * hidden_size, bias=True))
            nn.init.constant_(self.modulation_layer[-1].weight, 0.0)
            nn.init.constant_(self.modulation_layer[-1].bias, 0.0)

    def forward(self, x, condition=None, mask=None):
        if condition is None:
            assert not self.modulation, "Without global condition, must set modulation to False"
        else:
            assert self.modulation, "With global condition, must set modulation to True"
            shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = self.modulation_layer(condition).chunk(
                6, dim=1
            )

        if condition is not None:
            x = x + gate_attn.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_attn, scale_attn), mask=mask)
        else:
            x = x + self.attn(self.norm1(x), mask=mask)

        if condition is not None:
            x = x + gate_ffn.unsqueeze(1) * self.ffn(modulate(self.norm2(x), shift_ffn, scale_ffn))
        else:
            x = x + self.ffn(self.norm2(x), mask=mask)
        return x


# --------------- Flow Matching: Token -> Latent ---------------


class Token2latentFlowMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_dim = config.transformer.hidden_size
        self.token_input_dim = config.get("token_input_dim", self.model_dim)
        self.target_dim = config.z_dim

        self.spkr_embed_dim = config.get("spkr_embed_dim", 512)
        self.cond_proj = Linear(self.model_dim + self.spkr_embed_dim, self.model_dim)
        self.token_pad_id = -1
        if config.upsample_factor > 1:
            self.token_proj = nn.Sequential(
                Linear(self.token_input_dim, self.model_dim, bias=True),
                ConvTranspose1d(
                    self.model_dim,
                    self.model_dim,
                    stride=config.upsample_factor,
                    kernel_size=config.upsample_factor * 2,
                ),
            )
        else:
            self.token_proj = Linear(self.token_input_dim, self.model_dim)

        self.transformer_input_proj = Linear(self.model_dim + self.target_dim * 2, self.model_dim)
        self.time_embedder = TimestepEmbedder(self.model_dim)
        self.conv_embed = ConvPositionEmbed(hidden_size=self.model_dim, kernel_size=31, groups=16)
        self.blocks = nn.ModuleList()
        for _ in range(config.transformer.num_layers):
            attn_block = MultiHeadAttention(**config.transformer)
            ffn_block = Mlp(act_layer=lambda: nn.GELU(approximate="tanh"), **config.transformer)
            self.blocks.append(TransformerBlock(attn_block, ffn_block, **config.transformer))
        self.output_layer = FinalLayer(config.transformer.hidden_size, self.target_dim)

    def cond_mask_spkr_embed(self, x, spkr_embed):
        b, device = x.size(0), x.device
        if not exists(spkr_embed):
            spkr_embed = torch.zeros(b, self.spkr_embed_dim, device=device, dtype=x.dtype)
        return spkr_embed

    def vectorfield_forward(self, inputs, times, self_attn_mask, g_cond=None):
        t = self.time_embedder(times)
        if g_cond is None:
            c = t
        else:
            cond_inp = torch.cat([t, g_cond], dim=-1)
            c = self.cond_proj(cond_inp)

        ut = self.transformer_input_proj(inputs)
        ut = self.conv_embed(ut, mask=self_attn_mask) + ut

        for block in self.blocks:
            ut = block(ut, c, mask=self_attn_mask)
        ut = self.output_layer(ut, c)
        return ut

    @eval_decorator
    @torch.no_grad()
    def inference(
        self,
        *,
        token: torch.Tensor,
        prefix_target: torch.Tensor | None = None,
        spkr_embed: torch.Tensor | None = None,
        s_steps: int | None = 10,
        cfg_alpha: float | None = 2.0,
        rescale_logits: bool = False,
        **kwargs,
    ):
        b, device = token.size(0), token.device
        assert b == 1, "Only support batch_size == 1 when inference"

        token = self.token_proj(token)
        ge = self.cond_mask_spkr_embed(token, spkr_embed)
        tgt_lens = token.size(1)
        latent_cond = torch.zeros(b, tgt_lens, self.target_dim).to(device)
        if prefix_target is not None:
            latent_cond[:, : prefix_target.size(1), :] = prefix_target
        sample, trajectory = self.sample(
            tokens=token, audio=latent_cond, steps=s_steps, alpha=cfg_alpha, g_cond=ge, rescale_logits=rescale_logits
        )
        return sample

    def sample(self, tokens, audio, steps, alpha=None, g_cond=None, rescale_logits=False):
        noise = torch.randn([audio.size(0), audio.shape[1], self.target_dim], device=audio.device)
        times = torch.linspace(0, 1, steps, device=audio.device)

        def solver(t, z):
            if alpha is None:
                output = torch.cat([tokens, audio, z], dim=-1)
                return self.vectorfield_forward(inputs=output, times=t.unsqueeze(0), self_attn_mask=None, g_cond=g_cond)
            tokens_empty = torch.zeros(*audio.shape[:2], self.model_dim, device=tokens.device, dtype=tokens.dtype)
            audio_empty = audio
            tokens_t = torch.cat([tokens_empty, tokens], dim=0)
            audio_t = torch.cat([audio_empty, audio], dim=0)
            audio_noizy_t = torch.cat([z, z], dim=0)
            t_t = torch.stack([t, t], dim=0)
            c = g_cond
            if g_cond is not None:
                c = torch.cat([g_cond, g_cond], dim=0)
            output = torch.cat([tokens_t, audio_t, audio_noizy_t], dim=-1)
            predicted_mix = self.vectorfield_forward(inputs=output, times=t_t, self_attn_mask=None, g_cond=c)
            predicted_conditioned = predicted_mix[1].unsqueeze(0)
            predicted_unconditioned = predicted_mix[0].unsqueeze(0)

            prediction = predicted_conditioned + (predicted_conditioned - predicted_unconditioned) * alpha
            prediction_rescaled = predicted_conditioned.std() * (prediction / prediction.std())
            if rescale_logits:
                return prediction_rescaled
            else:
                return prediction

        try:
            from torchdiffeq import odeint
        except ImportError:
            raise ImportError("Covo-Audio code2wav requires `torchdiffeq`. Install it with: pip install torchdiffeq")
        trajectory = odeint(solver, noise, times, atol=1e-5, rtol=1e-5, method="midpoint")
        return trajectory[-1], trajectory


class Token2latentFlowMatchingWithEmbed(Token2latentFlowMatching):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.vocab_size = config.token_vocab_size
        self.token_embedding = EmbeddingTable(
            num_embeddings=self.vocab_size, embedding_dim=self.token_input_dim, pad_id=self.token_pad_id
        )

    def inference(
        self, *, token, prefix_target=None, spkr_embed=None, s_steps=10, cfg_alpha=2, rescale_logits=False, **kwargs
    ):
        token = self.token_embedding(token)
        return super().inference(
            token=token,
            prefix_target=prefix_target,
            spkr_embed=spkr_embed,
            s_steps=s_steps,
            cfg_alpha=cfg_alpha,
            rescale_logits=rescale_logits,
            **kwargs,
        )


# --------------- Vocoder: Latent -> Waveform (BigVGAN) ---------------


class AMPBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5), causal=True):
        super().__init__()
        self.h = h

        self.convs1 = nn.ModuleList(
            [
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], causal=causal)),
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], causal=causal)),
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], causal=causal)),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, causal=causal)),
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, causal=causal)),
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, causal=causal)),
            ]
        )
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(self.convs2)

        self.activations = nn.ModuleList(
            [
                Activation1d(activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale))
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x


class BigVGANFlowVAE(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.config = h
        self.h = h
        causal = h.causal
        self.hop_size = np.prod(h.downsample_rates)

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        self.conv_pre = weight_norm(Conv1d(h.latent_dim, h.upsample_initial_channel, 7, 1, causal=False))

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                h.upsample_initial_channel // (2**i),
                                h.upsample_initial_channel // (2 ** (i + 1)),
                                k,
                                u,
                                causal=causal,
                            )
                        )
                    ]
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(AMPBlock1(h, ch, k, d, causal=causal))

        activation_post = SnakeBeta(ch, alpha_logscale=h.snake_logscale)
        self.activation_post = Activation1d(activation=activation_post)

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, causal=causal))

        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

    def inference_from_latents(self, x, do_sample=True, noise_scale=1.0):
        if self.h.use_vae and do_sample:
            assert x.size(1) == self.h.latent_dim * 2, f"Input must be like [B, D, H], got {x.shape}"
            m_q, logs_q = torch.split(x, self.h.latent_dim, dim=1)
            x = m_q + torch.randn_like(m_q) * torch.exp(logs_q) * noise_scale
        else:
            assert x.size(1) == self.h.latent_dim, f"Input must be like [B, D, H], got {x.shape}"

        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x


# --------------- Token2WavDecoder ---------------


class Token2WavDecoder(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.wavegan = BigVGANFlowVAE(config.wavegan)
        self.wavegan_hop_size = np.prod(config.wavegan.downsample_rates)
        self.global_mean_var = getattr(config, "global_mean_var", None)
        if self.global_mean_var is not None:
            mean_var_data = torch.from_numpy(np.load(self.global_mean_var)).float().squeeze()
            global_mean, global_var = mean_var_data.chunk(2, 0)
            self.register_buffer("global_mean", global_mean)
            self.register_buffer("global_var", global_var)

        self.token2latent = Token2latentFlowMatchingWithEmbed(config.token2latent)

        self.upsample_factor = self.token2latent.config.get("upsample_factor", 1)
        self.wav_input_sr = config.get("wav_input_sr", 24000)

        self.trainable_module = ["wavegan", "token2latent"]

    def state_dict(self):
        param_dict = OrderedDict()
        for name in self.trainable_module:
            state = self.get_submodule(name).state_dict(prefix=f"{name}.")
            param_dict.update(state)
        return param_dict

    def load_state_dict(self, param_dict):
        for name in self.trainable_module:
            module_state = OrderedDict()
            name_len = len(name)
            for k, v in param_dict.items():
                if k.startswith(f"{name}."):
                    new_k = k[name_len + 1 :]
                    module_state[new_k] = v
            self.get_submodule(name).load_state_dict(module_state, strict=False)

    @torch.no_grad()
    def preprocess_infer_data(self, data):
        zero_spkr = data.get("zero_spkr", False)

        token = data["target_token"]
        prefix_target = None
        res_prefix_len = 0
        prompt_token = data.get("prompt_token", None)
        prompt_latent = data.get("prompt_latent", None)
        if exists(prompt_token):
            token = torch.cat([prompt_token, token], dim=1)
        if exists(prompt_latent):
            prefix_target = prompt_latent
            res_prefix_len = prompt_latent.shape[1]

        spkr_embed = None
        if not zero_spkr:
            spkr_embed = data.get("spkr_embed", None)
        res = {"token": token, "prefix_target": prefix_target, "spkr_embed": spkr_embed}
        return res, res_prefix_len

    @eval_decorator
    @torch.no_grad()
    def inference(self, data, **kwargs):
        infer_data, prefix_len = self.preprocess_infer_data(data)
        prefix_target = infer_data["prefix_target"]
        res_latents = self.token2latent.inference(**infer_data, **kwargs)
        if self.global_mean_var is not None:
            res_latents = self.global_mean + res_latents * torch.sqrt(self.global_var)
            if exists(prefix_target):
                prefix_target = self.global_mean + prefix_target * torch.sqrt(self.global_var)
        if exists(prefix_target):
            res_latents[:, : prefix_target.shape[1]] = prefix_target
        res_latents = res_latents[:, prefix_len:]
        audio = self.wavegan.inference_from_latents(res_latents.transpose(1, 2), do_sample=False)
        return audio
