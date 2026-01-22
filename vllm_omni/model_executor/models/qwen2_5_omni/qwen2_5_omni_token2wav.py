############################
#      Start Token2Wav     #
############################

import math
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniBigVGANConfig,
    Qwen2_5OmniDiTConfig,
    Qwen2_5OmniToken2WavConfig,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniPreTrainedModel

# Bring in HF base classes, configs and utilities used below
from transformers.utils.logging import get_logger as _hf_get_logger
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import AutoWeightsLoader as _Vllm_AutoWeightsLoader
from vllm.model_executor.models.utils import WeightsMapper as _Vllm_WeightsMapper
from vllm.model_executor.models.utils import init_vllm_registered_model as _vllm_init_vllm_registered_model
from vllm.model_executor.models.utils import maybe_prefix as _vllm_maybe_prefix
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.qwen2_5_omni.audio_length import cap_and_align_mel_length, resolve_max_mel_frames
from vllm_omni.utils.platform_utils import is_npu


# Provide a no-op auto_docstring decorator to satisfy annotations if missing
def auto_docstring(func=None, **_kwargs):
    if func is None:

        def wrapper(f):
            return f

        return wrapper
    return func


# HF logger alias
logger = _hf_get_logger(__name__)


# Using custom RoPE, will use LlamaRotaryEmbedding next version
class Qwen2_5OmniDiTRotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        t = torch.arange(seq_len, device=x.device)
        device_type = x.device.type
        device_type = device_type if device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = t.unsqueeze(1).float() @ self.inv_freq.unsqueeze(0).float()
            freqs = torch.stack((freqs, freqs), dim=-1)
            freqs = freqs.reshape(*freqs.shape[:-2], -1)
            freqs = freqs.repeat(batch_size, *([1] * freqs.dim()))
            cos = freqs.cos()
            sin = freqs.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class TimeDelayNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding="same",
            padding_mode="reflect",
        )
        self.activation = nn.ReLU()

    def forward(self, hidden_states: torch.Tensor):
        return self.activation(self.conv(hidden_states))


class Res2NetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1):
        super().__init__()

        in_channel = in_channels // scale
        hidden_channel = out_channels // scale

        self.blocks = nn.ModuleList(
            [
                TimeDelayNetBlock(
                    in_channel,
                    hidden_channel,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
                for i in range(scale - 1)
            ]
        )
        self.scale = scale

    def forward(self, hidden_states):
        outputs = []
        for i, hidden_part in enumerate(torch.chunk(hidden_states, self.scale, dim=1)):
            if i == 0:
                output_part = hidden_part
            elif i == 1:
                output_part = self.blocks[i - 1](hidden_part)
            else:
                output_part = self.blocks[i - 1](hidden_part + output_part)
            outputs.append(output_part)
        output = torch.cat(outputs, dim=1)
        return output


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels, se_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=se_channels,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            in_channels=se_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_states):
        hidden_states_mean = hidden_states.mean(dim=2, keepdim=True)

        hidden_states_mean = self.relu(self.conv1(hidden_states_mean))
        hidden_states_mean = self.sigmoid(self.conv2(hidden_states_mean))

        return hidden_states * hidden_states_mean


class AttentiveStatisticsPooling(nn.Module):
    """This class implements an attentive statistic pooling layer for each channel.
    It returns the concatenated mean and std of the input tensor.
    """

    def __init__(self, channels, attention_channels=128):
        super().__init__()

        self.eps = 1e-12
        self.tdnn = TimeDelayNetBlock(channels * 3, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(
            in_channels=attention_channels,
            out_channels=channels,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )

    def _length_to_mask(self, length, max_len=None, dtype=None, device=None):
        """Creates a binary mask for each sequence.

        Reference: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397/3  # noqa: E501

        Arguments
        ---------
        length : torch.LongTensor
            Containing the length of each sequence in the batch. Must be 1D.
        max_len : int
            Max length for the mask, also the size of the second dimension.
        dtype : torch.dtype, default: None
            The dtype of the generated mask.
        device: torch.device, default: None
            The device to put the mask variable.

        Returns
        -------
        mask : tensor
            The binary mask.
        """

        if max_len is None:
            max_len = length.max().long().item()  # using arange to generate mask
        mask = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(
            len(length), max_len
        ) < length.unsqueeze(1)

        mask = torch.as_tensor(mask, dtype=dtype, device=device)
        return mask

    def _compute_statistics(self, x, m, dim=2):
        mean = (m * x).sum(dim)
        std = torch.sqrt((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(self.eps))
        return mean, std

    def forward(self, hidden_states):
        seq_length = hidden_states.shape[-1]
        lengths = torch.ones(hidden_states.shape[0], device=hidden_states.device)

        # Make binary mask of shape [N, 1, L]
        mask = self._length_to_mask(
            lengths * seq_length,
            max_len=seq_length,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        mask = mask.unsqueeze(1)

        # Expand the temporal context of the pooling layer by allowing the
        # self-attention to look at global properties of the utterance.
        total = mask.sum(dim=2, keepdim=True)

        mean, std = self._compute_statistics(hidden_states, mask / total)
        mean = mean.unsqueeze(2).repeat(1, 1, seq_length)
        std = std.unsqueeze(2).repeat(1, 1, seq_length)
        attention = torch.cat([hidden_states, mean, std], dim=1)

        # Apply layers
        attention = self.conv(self.tanh(self.tdnn(attention)))

        # Filter out zero-paddings
        attention = attention.masked_fill(mask == 0, float("-inf"))

        attention = F.softmax(attention, dim=2)
        mean, std = self._compute_statistics(hidden_states, attention)
        # Append mean and std of the batch
        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(2)

        return pooled_stats


class SqueezeExcitationRes2NetBlock(nn.Module):
    """An implementation of building block in ECAPA-TDNN, i.e.,
    TDNN-Res2Net-TDNN-SqueezeExcitationBlock.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        res2net_scale=8,
        se_channels=128,
        kernel_size=1,
        dilation=1,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TimeDelayNetBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
        )
        self.res2net_block = Res2NetBlock(out_channels, out_channels, res2net_scale, kernel_size, dilation)
        self.tdnn2 = TimeDelayNetBlock(
            out_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
        )
        self.se_block = SqueezeExcitationBlock(out_channels, se_channels, out_channels)

    def forward(self, hidden_state):
        residual = hidden_state

        hidden_state = self.tdnn1(hidden_state)
        hidden_state = self.res2net_block(hidden_state)
        hidden_state = self.tdnn2(hidden_state)
        hidden_state = self.se_block(hidden_state)

        return hidden_state + residual


class ECAPA_TimeDelayNet(torch.nn.Module):
    """An implementation of the speaker embedding model in a paper.
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification" (https://huggingface.co/papers/2005.07143).
    """

    def __init__(self, config: Qwen2_5OmniDiTConfig):
        super().__init__()
        if len(config.enc_channels) != len(config.enc_kernel_sizes) or len(config.enc_channels) != len(
            config.enc_dilations
        ):
            raise ValueError("enc_channels, enc_kernel_sizes and enc_dilations should have same length")
        self.channels = config.enc_channels
        self.blocks = nn.ModuleList()

        # The initial TDNN layer
        self.blocks.append(
            TimeDelayNetBlock(
                config.mel_dim,
                config.enc_channels[0],
                config.enc_kernel_sizes[0],
                config.enc_dilations[0],
            )
        )

        # SE-Res2Net layers
        for i in range(1, len(config.enc_channels) - 1):
            self.blocks.append(
                SqueezeExcitationRes2NetBlock(
                    config.enc_channels[i - 1],
                    config.enc_channels[i],
                    res2net_scale=config.enc_res2net_scale,
                    se_channels=config.enc_se_channels,
                    kernel_size=config.enc_kernel_sizes[i],
                    dilation=config.enc_dilations[i],
                )
            )

        # Multi-layer feature aggregation
        self.mfa = TimeDelayNetBlock(
            config.enc_channels[-1],
            config.enc_channels[-1],
            config.enc_kernel_sizes[-1],
            config.enc_dilations[-1],
        )

        # Attentive Statistical Pooling
        self.asp = AttentiveStatisticsPooling(
            config.enc_channels[-1],
            attention_channels=config.enc_attention_channels,
        )

        # Final linear transformation
        self.fc = nn.Conv1d(
            in_channels=config.enc_channels[-1] * 2,
            out_channels=config.enc_dim,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )

    def forward(self, hidden_states):
        # Minimize transpose for efficiency
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states_list = []
        for layer in self.blocks:
            hidden_states = layer(hidden_states)
            hidden_states_list.append(hidden_states)

        # Multi-layer feature aggregation
        hidden_states = torch.cat(hidden_states_list[1:], dim=1)
        hidden_states = self.mfa(hidden_states)

        # Attentive Statistical Pooling
        hidden_states = self.asp(hidden_states)

        # Final linear transformation
        hidden_states = self.fc(hidden_states)

        hidden_states = hidden_states.squeeze(-1)
        return hidden_states


class DiTInputEmbedding(nn.Module):
    def __init__(self, config: Qwen2_5OmniDiTConfig):
        super().__init__()
        self.proj = nn.Linear(
            config.mel_dim + config.enc_dim + config.enc_emb_dim + config.emb_dim,
            config.hidden_size,
        )
        self.spk_encoder = ECAPA_TimeDelayNet(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        speaker_embedding: torch.Tensor,
        condition_vector: torch.Tensor,
        code_embed: torch.Tensor,
        drop_audio_cond: bool | None = False,
        code_embed_uncond: bool | None = None,
        apply_cfg: bool | None = True,
    ):
        if apply_cfg:
            hidden_states = torch.cat([hidden_states, hidden_states], dim=0)
            speaker_embedding = torch.cat([speaker_embedding, torch.zeros_like(speaker_embedding)], dim=0)
            condition_vector = torch.cat([condition_vector, torch.zeros_like(condition_vector)], dim=0)
            code_embed = torch.cat([code_embed, code_embed_uncond], dim=0)
        elif drop_audio_cond:  # cfg for cond audio
            condition_vector = torch.zeros_like(condition_vector)
            speaker_embedding = torch.zeros_like(speaker_embedding)
        condition_vector = self.spk_encoder(condition_vector).unsqueeze(1).repeat(1, hidden_states.size(1), 1)
        hidden_states = self.proj(torch.cat((hidden_states, condition_vector, code_embed, speaker_embedding), dim=-1))

        return hidden_states


# Transformer backbone using DiT blocks
class DiTCodecEmbedding(nn.Module):
    def __init__(self, codec_num_embeds, codec_dim, repeats):
        super().__init__()
        self.repeats = repeats
        self.codec_embed = nn.Embedding(codec_num_embeds + 1, codec_dim)

    def forward(self, code, drop_code=False):
        if drop_code:
            code = torch.zeros_like(code)
        code_embed = self.codec_embed(code)

        code_embed = torch.repeat_interleave(code_embed, repeats=self.repeats, dim=1)
        return code_embed


# AdaLayerNormZero
# return with modulated x for attn input, and params for later mlp modulation
class Qwen2_5_OmniAdaLayerNormZero(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, hidden_states, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)

        hidden_states = self.norm(hidden_states) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp


# AdaLayerNormZero for final layer
# return only with modulated x for attn input, cuz no more mlp modulation
class Qwen2_5_OmniAdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, hidden_states, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)

        hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        return hidden_states


# FeedForward
class DiTMLP(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)

        self.ff = nn.ModuleList(
            [
                nn.Linear(dim, inner_dim),
                nn.GELU(approximate="tanh"),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim),
            ]
        )

    def forward(self, hidden_states):
        for layer in self.ff:
            hidden_states = layer(hidden_states)
        return hidden_states


# Modified from Llama with a different rotate function, will fixed in next release
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to
            unsqueeze cos[position_ids] and sin[position_ids] so that they can be
            properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape
            [batch_size, seq_len, head_dim]. Then, if q and k have the shape
            [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1
            makes cos[position_ids] and sin[position_ids] broadcastable to the
            shapes of q and k. Similarly, if q and k have the shape
            [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated
        using the Rotary Position Embedding.
    """

    def rotate_half_codec(x):
        # x = rearrange(x, "... (d r) -> ... d r", r=2)
        x = x.reshape(*x.shape[:-1], -1, 2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return x.reshape(*x.shape[:-2], -1)

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half_codec(q) * sin)
    k_embed = (k * cos) + (rotate_half_codec(k) * sin)
    return q_embed, k_embed


class DiTAttention(nn.Module):
    def __init__(self, config: Qwen2_5OmniDiTConfig, prefix: str = ""):
        super().__init__()

        self.config = config
        self.dim = config.hidden_size
        self.heads = config.num_attention_heads
        self.inner_dim = config.head_dim * config.num_attention_heads
        self.dropout = config.dropout
        self.is_causal = False

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.dim,
            head_size=config.head_dim,
            total_num_heads=self.heads,
            bias=True,
            prefix=f"{prefix}.qkv_proj",
            disable_tp=True,
            return_bias=False,
        )
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, self.dim), nn.Dropout(config.dropout)])

    def forward(
        self,
        hidden_states,  # noised input x
        position_embeddings=None,  # rotary position embedding for x
        attention_mask=None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        qkv = self.qkv_proj(hidden_states)
        query, key, value = qkv.split([self.inner_dim, self.inner_dim, self.inner_dim], dim=-1)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        # apply rotary position embedding
        # Due to training process, only first head is applied with RoPE,
        # will be fixed at next release
        cos, sin = position_embeddings
        query[:, :1], key[:, :1] = apply_rotary_pos_emb(query[:, :1], key[:, :1], cos, sin)

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attention_weights, _ = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask=attention_mask,
            is_causal=False,
        )

        # mask. e.g. inference got a batch with different target durations,
        # mask out the padding
        attention_weights = attention_weights.reshape(batch_size, -1, self.heads * head_dim)
        attention_weights = attention_weights.to(query.dtype)

        # linear proj
        attention_output = self.to_out[0](attention_weights)
        attention_output = self.to_out[1](attention_output)

        return attention_output


# time step conditioning embedding
class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, hidden_states, scale=1000):
        device = hidden_states.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * hidden_states.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.type_as(hidden_states)


class DiTTimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.ModuleList([nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim)])

    def forward(self, timestep):  # noqa: F821
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        for layer in self.time_mlp:
            time_hidden = layer(time_hidden)  # b d
        return time_hidden


class DiTDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2_5OmniDiTConfig, look_ahead_block=0, look_backward_block=0):
        super().__init__()
        self.attn_norm = Qwen2_5_OmniAdaLayerNormZero(config.hidden_size)

        self.attn = DiTAttention(config)
        self.look_ahead_block = look_ahead_block
        self.look_backward_block = look_backward_block
        self.ff_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = DiTMLP(dim=config.hidden_size, mult=config.ff_mult, dropout=config.dropout)

    def forward(
        self, hidden_states, timestep, position_embeddings=None, block_diff=None
    ):  # x: noised input, t: time embedding
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(hidden_states, emb=timestep)

        # attention
        attn_output = self.attn(
            hidden_states=norm,
            position_embeddings=position_embeddings,
            attention_mask=(block_diff >= -float(self.look_backward_block))
            & (block_diff <= float(self.look_ahead_block)),
        )

        # process attention output for input x
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output

        norm = self.ff_norm(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        return hidden_states


class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude
    of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper
          by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://huggingface.co/papers/2006.08195
    """

    def __init__(self, in_features, alpha=1.0):
        super().__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha = Parameter(torch.zeros(in_features) * alpha)
        self.beta = Parameter(torch.zeros(in_features) * alpha)

        self.no_div_by_zero = 0.000000001

    def forward(self, hidden_states):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        alpha = torch.exp(alpha)
        beta = torch.exp(beta)
        hidden_states = hidden_states + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(
            torch.sin(hidden_states * alpha), 2
        )

        return hidden_states


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    """Generates a 1D Kaiser-windowed sinc filter.

    Args:
        cutoff (float): Normalized cutoff frequency (0 to 0.5).
        half_width (float): Transition bandwidth.
        kernel_size (int): Number of filter taps.

    Returns:
        torch.Tensor: A tensor of shape (1, 1, kernel_size) representing the filter.
    """
    is_even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # Compute Kaiser window parameters
    delta_f = 4 * half_width
    attenuation = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95

    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0

    # TODO: When torch.kaiser_window supports NPU, remove the device="cpu" argument
    if is_npu():
        kaiser_window = torch.kaiser_window(
            kernel_size, beta=beta, periodic=False, dtype=torch.float32, device="cpu"
        ).to("npu")
    else:
        kaiser_window = torch.kaiser_window(kernel_size, beta=beta, periodic=False, dtype=torch.float32)

    # Compute time indices
    if is_even:
        time_indices = torch.arange(-half_size, half_size) + 0.5
    else:
        time_indices = torch.arange(kernel_size) - half_size

    # Compute sinc filter
    if cutoff == 0:
        return torch.zeros((1, 1, kernel_size), dtype=torch.float32)

    sinc_filter = torch.sinc(2 * cutoff * time_indices)
    normalized_filter = 2 * cutoff * kaiser_window * sinc_filter

    # Normalize to ensure sum = 1 (avoid leakage of constant component)
    normalized_filter /= normalized_filter.sum()

    return normalized_filter.view(1, 1, kernel_size)


def replication_pad_1d(hidden_states: torch.Tensor, pad_left: int, pad_right: int) -> torch.Tensor:
    """
    Manual replicate padding to avoid replication_pad1d kernel limits on NPU.
    TODO: remove when F.pad supports replicate mode on NPU.
    """
    # NOTE: a immature implementation for running in NPU. Need to discuss.
    if pad_left == 0 and pad_right == 0:
        return hidden_states

    segments = []
    if pad_left > 0:
        left = hidden_states[..., :1].expand(*hidden_states.shape[:-1], pad_left)
        segments.append(left)

    segments.append(hidden_states)

    if pad_right > 0:
        right = hidden_states[..., -1:].expand(*hidden_states.shape[:-1], pad_right)
        segments.append(right)

    return torch.cat(segments, dim=-1)


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
        self.register_buffer("filter", filter, persistent=False)

    def forward(self, hidden_states):
        channels = hidden_states.shape[1]
        if is_npu():
            # TODO: When F.pad supports replicate mode on NPU, remove this branch
            input_dtype = hidden_states.dtype
            # F.pad in NPU doesn't support BF16 when mode is replicate.
            # To ensure the accuracy, manually pad the input tensor.
            hidden_states = replication_pad_1d(hidden_states.to(self.filter.dtype), self.pad, self.pad)
            filter_convert_dtype = self.filter.to(hidden_states.dtype)
            hidden_states = self.ratio * F.conv_transpose1d(
                hidden_states,
                filter_convert_dtype.expand(channels, -1, -1),
                stride=self.stride,
                groups=channels,
            ).to(input_dtype)
        else:
            hidden_states_dtype = hidden_states.dtype
            hidden_states = F.pad(hidden_states, (self.pad, self.pad), mode="replicate").to(self.filter.dtype)
            hidden_states = self.ratio * F.conv_transpose1d(
                hidden_states,
                self.filter.expand(channels, -1, -1),
                stride=self.stride,
                groups=channels,
            ).to(hidden_states_dtype)
        hidden_states = hidden_states[..., self.pad_left : -self.pad_right]

        return hidden_states


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        cutoff = 0.5 / ratio
        half_width = 0.6 / ratio

        if cutoff < 0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")

        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = ratio
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter, persistent=False)

    def forward(self, hidden_states):
        channels = hidden_states.shape[1]
        if is_npu():
            input_dtype = hidden_states.dtype
            # F.pad in NPU doesn't support BF16 when mode is replicate.
            # To ensure the accuracy, manually pad the input tensor.
            hidden_states = replication_pad_1d(hidden_states.to(self.filter.dtype), self.pad_left, self.pad_right)
            filter_on_device = self.filter.to(device=hidden_states.device, dtype=hidden_states.dtype)
            out = F.conv1d(
                hidden_states,
                filter_on_device.expand(channels, -1, -1),
                stride=self.stride,
                groups=channels,
            ).to(input_dtype)
        else:
            hidden_states_dtype = hidden_states.dtype
            hidden_states = F.pad(hidden_states, (self.pad_left, self.pad_right), mode="replicate").to(
                self.filter.dtype
            )
            out = F.conv1d(
                hidden_states,
                self.filter.expand(channels, -1, -1),
                stride=self.stride,
                groups=channels,
            ).to(hidden_states_dtype)
        return out


class TorchActivation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        if not callable(activation):
            raise TypeError("Activation function must be callable")
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, hidden_states):
        hidden_states = self.upsample(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.downsample(hidden_states)

        return hidden_states


class AMPBlock(torch.nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=(1, 3, 5),
    ):
        super().__init__()

        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding=self._get_padding(kernel_size, dilation[0]),
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding=self._get_padding(kernel_size, dilation[1]),
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[2],
                    padding=self._get_padding(kernel_size, dilation[2]),
                ),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=self._get_padding(kernel_size, 1),
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=self._get_padding(kernel_size, 1),
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=self._get_padding(kernel_size, 1),
                ),
            ]
        )

        self.num_layers = len(self.convs1) + len(self.convs2)  # total number of conv layers

        self.activations = nn.ModuleList(
            [TorchActivation1d(activation=SnakeBeta(channels)) for _ in range(self.num_layers)]
        )

    def _get_padding(self, kernel_size, dilation=1):
        return int((kernel_size * dilation - dilation) / 2)

    def forward(self, hidden_states):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, acts1, acts2):
            residual = hidden_states
            hidden_states = act1(hidden_states)
            hidden_states = conv1(hidden_states)
            hidden_states = act2(hidden_states)
            hidden_states = conv2(hidden_states)
            hidden_states = residual + hidden_states

        return hidden_states


@auto_docstring(
    custom_intro="""
    The full Qwen2.5Omni Token2WavBigVGAN model. Which take mel spectrogram
    as input and predict waveform.
    """
)
class Qwen2_5OmniToken2WavBigVGANModel(Qwen2_5OmniPreTrainedModel):
    config: Qwen2_5OmniBigVGANConfig

    def __init__(self, config: Qwen2_5OmniBigVGANConfig):
        super().__init__(config)
        self.num_residual_blocks = len(config.resblock_kernel_sizes)
        self.num_upsample_layers = len(config.upsample_rates)

        self.conv_pre = nn.Conv1d(config.mel_dim, config.upsample_initial_channel, 7, 1, padding=3)

        # Removing extra ModuleList breaks official state dict
        ups = [
            nn.ModuleList(
                [
                    nn.ConvTranspose1d(
                        config.upsample_initial_channel // (2**layer_idx),
                        config.upsample_initial_channel // (2 ** (layer_idx + 1)),
                        kernel_size,
                        stride,
                        padding=(kernel_size - stride) // 2,
                    )
                ]
            )
            for layer_idx, (stride, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes))
        ]
        self.ups = nn.ModuleList(ups)

        self.resblocks = nn.ModuleList(
            [
                AMPBlock(
                    config.upsample_initial_channel // (2 ** (layer_idx + 1)),
                    kernel_size,
                    dilation,
                )
                for layer_idx in range(self.num_upsample_layers)
                for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes)
            ]
        )

        self.activation_post = TorchActivation1d(
            activation=SnakeBeta(config.upsample_initial_channel // (2**self.num_upsample_layers))
        )
        self.conv_post = nn.Conv1d(
            config.upsample_initial_channel // (2**self.num_upsample_layers),
            1,
            7,
            1,
            padding=3,
            bias=False,
        )

    def normalize_spectrogram(self, spectrogram, max_value, min_db):
        return torch.clamp(
            (2 * max_value) * ((spectrogram - min_db) / (-min_db)) - max_value,
            -max_value,
            max_value,
        )

    def amplitude_to_db(self, amplitude, min_db_level):
        min_level = torch.exp(
            torch.tensor(
                min_db_level / 20.0 * np.log(10),
                device=amplitude.device,
                dtype=amplitude.dtype,
            )
        )
        return 20 * torch.log10(torch.clamp(amplitude, min=min_level))

    def process_mel_spectrogram(self, mel_spectrogram):
        amplitude_spectrum = torch.exp(mel_spectrogram)
        decibel_spectrum = self.amplitude_to_db(amplitude_spectrum, -115) - 20
        return self.normalize_spectrogram(decibel_spectrum, 1, -115)

    def forward(self, mel_spectrogram):
        processed_spectrogram = self.process_mel_spectrogram(mel_spectrogram)
        hidden_representation = self.conv_pre(processed_spectrogram)

        for layer_index in range(self.num_upsample_layers):
            hidden_representation = self.ups[layer_index][0](hidden_representation)
            residual_output = sum(
                self.resblocks[layer_index * self.num_residual_blocks + block_index](hidden_representation)
                for block_index in range(self.num_residual_blocks)
            )
            residual_output = residual_output / self.num_residual_blocks
            hidden_representation = residual_output

        hidden_representation = self.activation_post(hidden_representation)
        output_waveform = self.conv_post(hidden_representation)
        return torch.clamp(output_waveform, min=-1.0, max=1.0).squeeze().cpu()


class RungeKutta4ODESolver:
    def __init__(self, function, initial_value):
        self.function = function
        self.initial_value = initial_value

        self._one_third = 1 / 3
        self._two_thirds = 2 / 3

    def _rk4_step(
        self,
        function,
        time_start,
        time_step,
        time_end,
        value_start,
        function_value_start=None,
    ):
        k1 = function_value_start if function_value_start is not None else function(time_start, value_start)
        k2 = function(
            time_start + time_step * self._one_third,
            value_start + time_step * k1 * self._one_third,
        )
        k3 = function(
            time_start + time_step * self._two_thirds,
            value_start + time_step * (k2 - k1 * self._one_third),
        )
        k4 = function(time_end, value_start + time_step * (k1 - k2 + k3))
        return (k1 + 3 * (k2 + k3) + k4) * time_step / 8

    def _compute_step(self, function, time_start, time_step, time_end, value_start):
        function_value_start = function(time_start, value_start)
        return (
            self._rk4_step(
                function,
                time_start,
                time_step,
                time_end,
                value_start,
                function_value_start=function_value_start,
            ),
            function_value_start,
        )

    def _linear_interpolation(self, time_start, time_end, value_start, value_end, time_point):
        if time_point == time_start:
            return value_start
        if time_point == time_end:
            return value_end
        weight = (time_point - time_start) / (time_end - time_start)
        return value_start + weight * (value_end - value_start)

    def integrate(self, time_points):
        solution = torch.empty(
            len(time_points),
            *self.initial_value.shape,
            dtype=self.initial_value.dtype,
            device=self.initial_value.device,
        )
        solution[0] = self.initial_value

        current_index = 1
        current_value = self.initial_value
        for time_start, time_end in zip(time_points[:-1], time_points[1:]):
            time_step = time_end - time_start
            delta_value, _ = self._compute_step(self.function, time_start, time_step, time_end, current_value)
            next_value = current_value + delta_value

            while current_index < len(time_points) and time_end >= time_points[current_index]:
                solution[current_index] = self._linear_interpolation(
                    time_start,
                    time_end,
                    current_value,
                    next_value,
                    time_points[current_index],
                )
                current_index += 1

            current_value = next_value

        return solution


@auto_docstring(
    custom_intro="""
    The full Qwen2.5Omni Token2WavDiT model. Which take speech tokens as
    input and predict mel spectrogram.
    """
)
class Qwen2_5OmniToken2WavDiTModel(Qwen2_5OmniPreTrainedModel):
    config: Qwen2_5OmniDiTConfig
    _no_split_modules = ["DiTDecoderLayer"]

    def __init__(self, config: Qwen2_5OmniDiTConfig):
        super().__init__(config)
        self.mel_dim = config.mel_dim
        self.repeats = config.repeats
        self.time_embed = DiTTimestepEmbedding(config.hidden_size)

        self.text_embed = DiTCodecEmbedding(config.num_embeds, config.emb_dim, config.repeats)
        self.input_embed = DiTInputEmbedding(config)

        self.rotary_embed = Qwen2_5OmniDiTRotaryEmbedding(config.head_dim)

        self.hidden_size = config.hidden_size
        self.layers = config.num_hidden_layers
        self.block_size = config.block_size
        self.num_attention_heads = config.num_attention_heads

        self.transformer_blocks = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            self.transformer_blocks.append(
                DiTDecoderLayer(
                    config,
                    look_ahead_block=1 if i in config.look_ahead_layers else 0,
                    look_backward_block=1 if i in config.look_backward_layers else 0,
                )
            )

        self.norm_out = Qwen2_5_OmniAdaLayerNormZero_Final(config.hidden_size)  # final modulation
        self.proj_out = nn.Linear(config.hidden_size, config.mel_dim)

    def _create_block_diff(self, hidden_states):
        batch, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        block_indices = torch.arange(seq_len, device=hidden_states.device) // self.block_size  # [seq_length]

        block_i = block_indices.unsqueeze(1)  # [seq_length, 1]
        block_j = block_indices.unsqueeze(0)  # [1, seq_length]
        block_diff = block_j - block_i  # (n, n)

        return block_diff.expand(batch, self.num_attention_heads, seq_len, seq_len)

    def forward(
        self,
        hidden_states,
        condition_vector,
        speaker_embedding,
        quantized_code,
        time_step,
        drop_audio_conditioning=False,
        drop_code=False,
        apply_cfg=True,
    ):
        batch_size = hidden_states.shape[0]
        if time_step.ndim == 0:
            time_step = time_step.repeat(batch_size)

        # Compute embeddings
        time_embedding = self.time_embed(time_step)
        text_embedding = self.text_embed(quantized_code, drop_code=False if apply_cfg else drop_code)
        text_embedding_unconditioned = self.text_embed(quantized_code, drop_code=True) if apply_cfg else None

        hidden_states = self.input_embed(
            hidden_states,
            speaker_embedding,
            condition_vector,
            text_embedding,
            drop_audio_cond=drop_audio_conditioning,
            code_embed_uncond=text_embedding_unconditioned,
            apply_cfg=apply_cfg,
        )

        # Compute positional encodings
        position_embeddings = self.rotary_embed(hidden_states)
        blockwise_difference = self._create_block_diff(hidden_states)

        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(
                hidden_states,
                time_embedding,
                position_embeddings=position_embeddings,
                block_diff=blockwise_difference,
            )

        hidden_states = self.norm_out(hidden_states, time_embedding)
        output = self.proj_out(hidden_states)

        return output

    @torch.no_grad()
    def sample(
        self,
        conditioning_vector,
        reference_mel_spectrogram,
        quantized_code,
        num_steps=10,
        guidance_scale=0.5,
        sway_coefficient=-1.0,
        max_mel_frames: int | None = None,
    ):
        max_mel_frames = resolve_max_mel_frames(max_mel_frames, default=30000)
        target_code_len, target_duration = cap_and_align_mel_length(
            code_len=int(quantized_code.shape[1]),
            repeats=int(self.repeats),
            max_mel_frames=max_mel_frames,
        )
        if int(quantized_code.shape[1]) != target_code_len:
            quantized_code = quantized_code[:, :target_code_len]

        initial_state = torch.randn(
            [1, target_duration, self.mel_dim],
            dtype=reference_mel_spectrogram.dtype,
            device=quantized_code.device,
        )
        batch_size = reference_mel_spectrogram.shape[0]
        conditioning_vector = conditioning_vector.unsqueeze(1).repeat(1, target_duration, 1)

        if batch_size != 1:
            raise ValueError("Only batch size = 1 is currently supported")

        def ode_function(time_step, hidden_states):
            if guidance_scale < 1e-5:
                prediction = self(
                    hidden_states=hidden_states,
                    speaker_embedding=conditioning_vector,
                    condition_vector=reference_mel_spectrogram,
                    quantized_code=quantized_code,
                    time_step=time_step,
                    drop_audio_conditioning=False,
                    drop_code=False,
                    apply_cfg=False,
                )
                return prediction

            model_output = self(
                hidden_states=hidden_states,
                quantized_code=quantized_code,
                speaker_embedding=conditioning_vector,
                condition_vector=reference_mel_spectrogram,
                time_step=time_step,
                apply_cfg=True,
            )
            guided_prediction, null_prediction = torch.chunk(model_output, 2, dim=0)
            return guided_prediction + (guided_prediction - null_prediction) * guidance_scale

        initial_time = 0
        time_embedding = torch.linspace(
            initial_time,
            1,
            num_steps,
            device=quantized_code.device,
            dtype=conditioning_vector.dtype,
        )

        if sway_coefficient is not None:
            time_embedding += sway_coefficient * (torch.cos(torch.pi / 2 * time_embedding) - 1 + time_embedding)

        ode_solver = RungeKutta4ODESolver(function=ode_function, initial_value=initial_state)
        solution_trajectory = ode_solver.integrate(time_embedding)

        generated_waveform = solution_trajectory[-1]
        generated_mel_spectrogram = generated_waveform.permute(0, 2, 1)
        return generated_mel_spectrogram

    @torch.no_grad()
    def fast_block_sample(
        self,
        conditioning_vector: torch.Tensor,
        reference_mel_spectrogram: torch.Tensor,
        quantized_code: torch.Tensor,
        y0: torch.Tensor,
        num_steps: int = 10,
        guidance_scale: float = 0.5,
        sway_coefficient: float | None = -1.0,
    ) -> torch.Tensor:
        """
        Block-wise ODE sampling starting from provided initial state y0.

        Args:
            conditioning_vector: (B, enc_emb_dim)
            reference_mel_spectrogram: (B, T_ref, mel_dim)
            quantized_code: (B, T_code)
            y0: (B, T_target, mel_dim) initial state for ODE
        Returns:
            mel: (B, mel_dim, T_target)
        """
        initial_state = y0.to(quantized_code.device)
        batch_size = reference_mel_spectrogram.shape[0]
        conditioning_vector = conditioning_vector.unsqueeze(1).repeat(1, initial_state.shape[1], 1)

        if batch_size != 1:
            raise ValueError("Only batch size = 1 is currently supported")

        def ode_function(time_step, hidden_states):
            if guidance_scale < 1e-5:
                prediction = self(
                    hidden_states=hidden_states,
                    speaker_embedding=conditioning_vector,
                    condition_vector=reference_mel_spectrogram,
                    quantized_code=quantized_code,
                    time_step=time_step,
                    drop_audio_conditioning=False,
                    drop_code=False,
                )
                return prediction

            model_output = self(
                hidden_states=hidden_states,
                quantized_code=quantized_code,
                speaker_embedding=conditioning_vector,
                condition_vector=reference_mel_spectrogram,
                time_step=time_step,
                apply_cfg=True,
            )
            guided_prediction, null_prediction = torch.chunk(model_output, 2, dim=0)
            return guided_prediction + (guided_prediction - null_prediction) * guidance_scale

        initial_time = 0
        time_embedding = torch.linspace(
            initial_time,
            1,
            num_steps,
            device=quantized_code.device,
            dtype=conditioning_vector.dtype,
        )

        if sway_coefficient is not None:
            time_embedding += sway_coefficient * (torch.cos(torch.pi / 2 * time_embedding) - 1 + time_embedding)

        ode_solver = RungeKutta4ODESolver(function=ode_function, initial_value=initial_state)
        solution_trajectory = ode_solver.integrate(time_embedding)

        generated_waveform = solution_trajectory[-1]
        generated_mel_spectrogram = generated_waveform.permute(0, 2, 1)
        return generated_mel_spectrogram

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # self-attn
            (".qkv_proj", ".to_q", "q"),
            (".qkv_proj", ".to_k", "k"),
            (".qkv_proj", ".to_v", "v"),
        ]

        params_dict = dict(self.named_parameters())

        loaded_params = set[str]()
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


@auto_docstring(
    custom_intro="""
    The full Qwen2.5Omni Token2Wav model. Consists a DiT model take speech
    tokens as input and predict mel spectrogram and a BigVGAN vocoder take
    mel spectrogram as input and predict waveform.
    """
)
class Qwen2_5OmniToken2WavModel(Qwen2_5OmniPreTrainedModel):
    config: Qwen2_5OmniToken2WavConfig
    base_model_prefix = "model"
    _no_split_modules = [
        "Qwen2_5OmniToken2WavDiTModel",
        "Qwen2_5OmniToken2WavBigVGANModel",
    ]

    def __init__(self, config: Qwen2_5OmniToken2WavConfig):
        super().__init__(config)
        attn_impl = config._attn_implementation
        if config._attn_implementation == "flash_attention_2":
            logger.warning_once(
                "Qwen2_5OmniToken2WavModel must inference with fp32, but "
                "flash_attention_2 only supports fp16 and bf16, "
                "attention implementation of Qwen2_5OmniToken2WavModel will "
                "fallback to sdpa."
            )
            attn_impl = "sdpa"
        elif config._attn_implementation == "eager":
            logger.warning_once(
                "Qwen2_5OmniToken2WavModel does not support eager attention implementation, fall back to sdpa"
            )
            attn_impl = "sdpa"
        self.code2wav_dit_model = Qwen2_5OmniToken2WavDiTModel._from_config(
            config.dit_config, attn_implementation=attn_impl
        )
        self.code2wav_bigvgan_model = Qwen2_5OmniToken2WavBigVGANModel._from_config(
            config.bigvgan_config, attn_implementation=attn_impl
        )

        # Streaming-related parameters aligned with Qwen2Code2wav
        self.factor = self.code2wav_dit_model.repeats  # 50Hz=2, 200Hz=4
        # default bs_mel depends on factor
        self.bs_mel = 24 if self.factor == 2 else 32
        self.bs_codec = self.bs_mel // self.factor
        self.past_cache_size = self.bs_mel * self.factor
        self.future_cache_size = self.bs_mel * 1
        self.batched_chunk = 3
        self.chunk_size = self.bs_mel * self.batched_chunk
        self.future_size = 20 if self.factor == 2 else 13

        # codec embedding size for masking EOS out-of-range
        try:
            self.codec_embed_size = self.code2wav_dit_model.text_embed.codec_embed.weight.size(0)
        except Exception:
            self.codec_embed_size = -1

        # vocoder hop length inferred from upsample rates
        try:
            ups = self.code2wav_bigvgan_model.config.upsample_rates
            hop = 1
            for r in ups:
                hop *= int(r)
            self.vocoder_hop = int(hop)
        except Exception:
            # fallback to commonly used value
            self.vocoder_hop = 240

    def forward(
        self,
        code,
        conditioning,
        reference_mel,
        num_steps=10,
        guidance_scale=0.5,
        sway_coefficient=-1.0,
        max_mel_frames: int | None = None,
        **kwargs,
    ):
        """Generates a waveform from input code and conditioning parameters."""

        mel_spectrogram = self.code2wav_dit_model.sample(
            conditioning,
            reference_mel,
            code,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            sway_coefficient=sway_coefficient,
            max_mel_frames=max_mel_frames,
        ).to(self.code2wav_bigvgan_model.dtype)

        waveform = self.code2wav_bigvgan_model(mel_spectrogram).to(self.dtype)

        return waveform

    # ============== Chunked processing helpers (compat with qwen2_code2wav_dit) ==============  # noqa: E501
    @torch.inference_mode()
    def process_chunk_dit_batch(
        self,
        conditioning: torch.Tensor,
        reference_mel: torch.Tensor,
        code: torch.Tensor,
        y0: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        """
        Block-wise DiT: generate mel from initial state y0 for the given code slice.
        """
        # prevent codec out-of-range (eos)
        if self.codec_embed_size > 0:
            code = code.clone()
            code[code >= self.codec_embed_size] = 0
        mel = self.code2wav_dit_model.fast_block_sample(
            conditioning_vector=conditioning,
            reference_mel_spectrogram=reference_mel,
            quantized_code=code,
            y0=y0,
            num_steps=steps,
        )
        return mel.to(self.code2wav_bigvgan_model.dtype)

    @torch.inference_mode()
    def process_chunk_bigvgan_batch(self, mel_batch: torch.Tensor) -> torch.Tensor:
        """Vocoder batch: mel -> waveform."""
        return self.code2wav_bigvgan_model(mel_batch)

    @torch.inference_mode()
    def process_little_chunk(
        self,
        conditioning: torch.Tensor,
        reference_mel: torch.Tensor,
        codec_all: torch.Tensor,
        y_all: torch.Tensor,
        i: int,
        steps: int,
        prev_generated: torch.Tensor,
        finished: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Streaming per small chunk: returns (mel_or_None, audio_slice)."""
        start_index = max(i * self.chunk_size - self.past_cache_size, 0)
        end_index = min(
            (i + 1) * self.chunk_size + self.future_cache_size,
            codec_all.shape[1] * self.factor,
        )

        y0 = y_all[:, start_index:end_index].reshape(1, -1, self.code2wav_dit_model.mel_dim).contiguous()
        codec = codec_all[:, start_index // self.factor : end_index // self.factor].reshape(1, -1).contiguous()

        # generate mel for current window (B, mel_dim, T)
        generated = self.process_chunk_dit_batch(
            conditioning=conditioning,
            reference_mel=reference_mel,
            code=codec,
            y0=y0,
            steps=steps,
        )

        # splice and vocode with 50Hz-style rules
        return self._process_chunk_for_50hz(
            i=i,
            start_index=start_index,
            end_index=end_index,
            finished=finished,
            prev_generated=prev_generated,
            generated=generated,
        )

    @torch.inference_mode()
    def process_chunk(
        self,
        conditioning: torch.Tensor,
        reference_mel: torch.Tensor,
        codec_all: torch.Tensor,
        y_all: torch.Tensor,
        i: int,
        steps: int,
        prev_generated: torch.Tensor | list[torch.Tensor],
        finished: bool = False,
    ) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor]:
        """High-level chunk API aligning to qwen2_code2wav_dit signature."""
        if not isinstance(prev_generated, torch.Tensor):
            prev_generated = prev_generated[0] if len(prev_generated) > 0 else None
        _mel, audio = self.process_little_chunk(
            conditioning=conditioning,
            reference_mel=reference_mel,
            codec_all=codec_all,
            y_all=y_all,
            i=i,
            steps=steps,
            prev_generated=prev_generated,
            finished=finished,
        )
        return _mel if _mel is not None else prev_generated, audio

    @torch.inference_mode()
    def _process_chunk_for_50hz(
        self,
        i: int,
        start_index: int,
        end_index: int,
        finished: bool,
        prev_generated: torch.Tensor | None,
        generated: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Align mel and audio boundaries for 50Hz-like streaming.

        Shapes:
          - generated: (B, mel_dim, T_window)
          - prev_generated: (B, mel_dim, T_prev)
        Returns:
          - mel_chunk: (B, mel_dim, T_chunk)
          - audio_slice: (T_audio_chunk,)
        """
        # Normalize dtype
        generated = generated.to(torch.float32)
        if i == 0:
            mel = generated[:, :, : self.chunk_size]
        elif finished:
            mel_trim = generated[:, :, self.past_cache_size :]
            mel = torch.cat([prev_generated[:, :, -self.future_size * 2 :], mel_trim], dim=2)
        else:
            if start_index == 0:
                mel_trim = generated[:, :, i * self.chunk_size : -self.future_cache_size]
            else:
                mel_trim = generated[:, :, self.past_cache_size : -self.future_cache_size]
            mel = torch.cat([prev_generated[:, :, -self.future_size * 2 :], mel_trim], dim=2)

        audio = self.code2wav_bigvgan_model(mel)
        if i == 0:
            audio_output = audio[: -self.future_size * self.vocoder_hop]
        elif finished:
            audio_output = audio[self.future_size * self.vocoder_hop :]
        else:
            audio_output = audio[self.future_size * self.vocoder_hop : -self.future_size * self.vocoder_hop]
        return mel, audio_output


# ================= vLLM-style wrapper for Token2Wav =================


class Qwen2_5OmniToken2WavForConditionalGenerationVLLM(nn.Module, SupportsPP):
    logger = init_logger(__name__)

    # Map HF weights -> vLLM module names
    hf_to_vllm_mapper = _Vllm_WeightsMapper(
        orig_to_new_prefix={
            # HF root is 'model.'
            "model.": "token2wav_model.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        # Expect hf_config to be Token2Wav config
        self.config = vllm_config.model_config.hf_config

        # Initialize underlying HF Token2Wav model via registry
        self.token2wav = _vllm_init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=_vllm_maybe_prefix(prefix, "token2wav_model"),
            hf_config=self.config,
            architectures=["Qwen2_5OmniToken2WavDiTModel"],
        )

        # Provide placeholder to align with vLLM runner expectations
        def _empty_intermediate_tensors():
            return None

        self.make_empty_intermediate_tensors = _empty_intermediate_tensors

    def get_language_model(self) -> torch.nn.Module:
        return self.token2wav

    @property
    def sampler(self):
        # Token2Wav does not use sampler; return vLLM default for API parity
        return Sampler()

    def forward(
        self,
        code: torch.Tensor,
        conditioning: torch.Tensor,
        reference_mel: torch.Tensor,
        num_steps: int = 10,
        guidance_scale: float = 0.5,
        sway_coefficient: float = -1.0,
        intermediate_tensors: IntermediateTensors | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # Delegate to HF token2wav model
        return self.token2wav(
            code=code,
            conditioning=conditioning,
            reference_mel=reference_mel,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            sway_coefficient=sway_coefficient,
            **kwargs,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        # Token2Wav outputs waveform; logits are not applicable
        return hidden_states

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        return None

    def load_weights_without_buffers(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = _Vllm_AutoWeightsLoader(self)
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        # Log load summary
        try:
            total_bytes = 0
            for _, param in self.named_parameters():
                if param is not None and param.data is not None:
                    total_bytes += param.data.numel() * param.data.element_size()
            device = next(self.parameters()).device
            self.logger.info(
                "[Model Loaded] name=%s, success=%s, size=%.2f MB, device=%s",
                self.__class__.__name__,
                True,
                total_bytes / (1024**2),
                str(device),
            )
        except Exception:
            pass
        return loaded

    def find_all_registers(self):
        """
        Find all registered buffers in a PyTorch model.

        Args:
        Returns:
            dict: Dictionary with buffer names as keys and their properties as values
        """
        registers = {}

        # Get all named buffers
        for name, buf in self.named_buffers():
            if name in self.state_dict():
                registers[name] = {"name": name, "buffer": buf}
        return registers

    # remove buffers from the weights and reload them after loading weights
    def remove_buffers_from_weights(self, weights: Iterable[tuple[str, torch.Tensor]], buffers: dict):
        weights_to_load = []
        for key, value in weights:
            if key in buffers:
                buffers[key]["buffer"] = value
                continue
            weights_to_load.append((key, value))
        return weights_to_load

    def reload_buffers_to_model(self, buffers: dict):
        """
        reload stored buffers from weights to model
        """
        loaded_buffers = set()
        for name, buf_val in self.named_buffers():
            if name in buffers:
                buf_val.copy_(buffers[name]["buffer"])
                loaded_buffers.add(name)
        return loaded_buffers

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]], spk_dict_path: str) -> set[str]:
        buffers = self.find_all_registers()
        weights_to_load = self.remove_buffers_from_weights(weights, buffers)
        loaded = self.load_weights_without_buffers(weights_to_load)
        loaded_buffers = self.reload_buffers_to_model(buffers)
        # merge loaded and loaded_buffers
        loaded.update(loaded_buffers)
        self.spk_dict = torch.load(spk_dict_path)
        return loaded

    # ============== Optional chunked helpers for API parity ==============
    @torch.inference_mode()
    def process_chunk_dit_batch(
        self,
        conditioning: torch.Tensor,
        reference_mel: torch.Tensor,
        code: torch.Tensor,
        y0: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        return self.token2wav(
            code=code,
            conditioning=conditioning,
            reference_mel=reference_mel,
            num_steps=steps,
        )

    @torch.inference_mode()
    def process_chunk_bigvgan_batch(self, mel_batch: torch.Tensor) -> torch.Tensor | None:
        # BigVGAN is not part of this wrapper; return None for parity.
        return None

    @torch.inference_mode()
    def process_little_chunk(
        self,
        conditioning: torch.Tensor,
        reference_mel: torch.Tensor,
        codec_all: torch.Tensor,
        y_all: torch.Tensor,
        i: int,
        steps: int,
        prev_generated: torch.Tensor,
        finished: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        mel = self.token2wav(
            code=codec_all,
            conditioning=conditioning,
            reference_mel=reference_mel,
            num_steps=steps,
        )
        return None, mel

    @torch.inference_mode()
    def process_chunk(
        self,
        conditioning: torch.Tensor,
        reference_mel: torch.Tensor,
        codec_all: torch.Tensor,
        y_all: torch.Tensor,
        i: int,
        steps: int,
        prev_generated: torch.Tensor | list[torch.Tensor],
        finished: bool = False,
    ) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor]:
        _mel, out = self.process_little_chunk(
            conditioning=conditioning,
            reference_mel=reference_mel,
            codec_all=codec_all,
            y_all=y_all,
            i=i,
            steps=steps,
            prev_generated=(prev_generated if isinstance(prev_generated, torch.Tensor) else None),
            finished=finished,
        )
        return _mel if _mel is not None else prev_generated, out
