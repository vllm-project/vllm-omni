# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Incremental Phonon Mimi with persistent slot-indexed state."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .mimi_attention import ring_attention_chunk


def causal_conv_step(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    history_pool: torch.Tensor,
    slots: torch.Tensor,
    *,
    dilation: int = 1,
    groups: int = 1,
    scratch: torch.Tensor | None = None,
    reset: torch.Tensor | None = None,
) -> torch.Tensor:
    """Stride-one causal convolution with alias-safe history update.

    Inputs are [batch, channels, time]. History is [slots, channels,
    dilation * (kernel_size - 1)]. No tensor retained by convolution is
    overwritten when the new history is installed.
    """
    history_length = history_pool.shape[-1]
    history = history_pool.index_select(0, slots)
    if reset is not None:
        history = torch.where(reset[:, None, None], 0, history)
    if scratch is None:
        combined = torch.cat((history, inputs), dim=-1)
    else:
        combined = scratch[: inputs.shape[0]]
        if history_length:
            combined[..., :history_length].copy_(history)
        combined[..., history_length:].copy_(inputs)
    output = F.conv1d(combined, weight, bias, dilation=dilation, groups=groups)
    if history_length:
        history_pool[slots] = combined[..., -history_length:]
    return output


class StreamingConv(nn.Module):
    def __init__(self, inputs, outputs, kernel, slots, *, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(inputs, outputs, kernel, dilation=dilation)
        self.register_buffer("history", torch.zeros(slots, inputs, dilation * (kernel - 1)), persistent=False)
        self.register_buffer("scratch", torch.empty(0), persistent=False)

    def forward(self, inputs, slots, reset=None):
        return causal_conv_step(
            inputs,
            self.conv.weight,
            self.conv.bias,
            self.history,
            slots,
            dilation=self.conv.dilation[0],
            reset=reset,
            scratch=self.scratch if self.scratch.shape[-1] == self.history.shape[-1] + inputs.shape[-1] else None,
        )


class StreamingTranspose(nn.Module):
    def __init__(self, inputs, outputs, stride, slots, *, groups=1, bias=True):
        super().__init__()
        self.convtr = nn.ConvTranspose1d(inputs, outputs, stride * 2, stride=stride, groups=groups, bias=bias)
        self.register_buffer("history", torch.zeros(slots, outputs, stride), persistent=False)

    def forward(self, inputs, slots, reset=None):
        output = self.convtr(inputs)
        overlap = self.history.shape[-1]
        history = self.history[slots]
        if reset is not None:
            history = torch.where(reset[:, None, None], 0, history)
        output[..., :overlap] += history
        tail = output[..., -overlap:]
        if self.convtr.bias is not None:
            tail = tail - self.convtr.bias[None, :, None]
        self.history[slots] = tail
        return output[..., :-overlap]


class ResidualBlock(nn.Module):
    def __init__(self, width, kernel, dilation, compress, slots):
        super().__init__()
        self.block = nn.ModuleList(
            [
                nn.ELU(),
                StreamingConv(width, width // compress, kernel, slots, dilation=dilation),
                nn.ELU(),
                StreamingConv(width // compress, width, 1, slots),
            ]
        )

    def forward(self, inputs, slots, reset=None):
        output = inputs
        for layer in self.block:
            output = layer(output, slots, reset) if isinstance(layer, StreamingConv) else layer(output)
        return inputs + output


class MimiAttention(nn.Module):
    def __init__(self, config, slots):
        super().__init__()
        width = config["transformer_d_model"]
        self.heads = config["transformer_num_heads"]
        self.in_proj = nn.Linear(width, width * 3, bias=False)
        self.out_proj = nn.Linear(width, width, bias=False)
        self.window = config["transformer_context"]
        chunk_frames = round(config["sample_rate"] / math.prod(config["ratios"]) / config["frame_rate"])
        ring_capacity = self.window + max(50, chunk_frames - 1)
        shape = (slots, ring_capacity, self.heads, width // self.heads)
        self.register_buffer("key_pool", torch.zeros(shape), persistent=False)
        self.register_buffer("value_pool", torch.zeros(shape), persistent=False)

    def forward(self, inputs, slots, positions, rotary):
        batch, frames, width = inputs.shape
        packed = self.in_proj(inputs).view(batch, frames, 3, self.heads, -1)
        query, key, value = packed.unbind(2)
        cosine, sine = rotary

        def rotate(tensor):
            even, odd = tensor[..., 0::2], tensor[..., 1::2]
            rotated = torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), -1)
            return rotated.flatten(-2).to(tensor.dtype)

        attended = ring_attention_chunk(
            rotate(query), rotate(key), value, self.key_pool, self.value_pool, slots, positions, self.window
        )
        return self.out_proj(attended.flatten(2))


class LayerScale(nn.Module):
    def __init__(self, width, scale):
        super().__init__()
        self.scale = nn.Parameter(torch.full((width,), scale))

    def forward(self, inputs):
        return inputs * self.scale


class MimiLayer(nn.Module):
    def __init__(self, config, slots):
        super().__init__()
        width = config["transformer_d_model"]
        self.self_attn = MimiAttention(config, slots)
        self.norm1 = nn.LayerNorm(width, eps=1e-5)
        self.norm2 = nn.LayerNorm(width, eps=1e-5)
        self.linear1 = nn.Linear(width, config["transformer_dim_feedforward"], bias=False)
        self.linear2 = nn.Linear(config["transformer_dim_feedforward"], width, bias=False)
        self.layer_scale_1 = LayerScale(width, config["transformer_layer_scale"])
        self.layer_scale_2 = LayerScale(width, config["transformer_layer_scale"])

    def forward(self, inputs, slots, positions, rotary):
        inputs = inputs + self.layer_scale_1(self.self_attn(self.norm1(inputs), slots, positions, rotary))
        return inputs + self.layer_scale_2(self.linear2(F.gelu(self.linear1(self.norm2(inputs)))))


class MimiStreamingDecoder(nn.Module):
    """Consumes denormalized, projected codec latents [batch, channels, frames]."""

    def __init__(self, config: dict, num_slots: int):
        super().__init__()
        self.config = config
        dimension = config["dimension"]
        width = config["transformer_d_model"]
        decoder_width = config.get("decoder_dimension", dimension)
        self.quantizer_projection = nn.Conv1d(
            config["quantizer_dimension"], config["quantizer_output_dimension"], 1, bias=False
        )
        ratio_product = math.prod(config["ratios"])
        upsample_ratio = config["sample_rate"] / ratio_product / config["frame_rate"]
        if abs(upsample_ratio - round(upsample_ratio)) > 1e-6:
            raise ValueError("Mimi requires an integral upsample ratio")
        self.upsample_ratio = round(upsample_ratio)
        self.upsample = (
            StreamingTranspose(dimension, dimension, self.upsample_ratio, num_slots, groups=dimension, bias=False)
            if self.upsample_ratio > 1
            else None
        )
        self.input_projection = nn.Linear(dimension, width, bias=False) if dimension != width else nn.Identity()
        self.layers = nn.ModuleList([MimiLayer(config, num_slots) for _ in range(config["transformer_num_layers"])])
        head_dim = width // config["transformer_num_heads"]
        frequencies = config["transformer_max_period"] ** (-torch.arange(0, head_dim, 2).float() / head_dim)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.output_projection = (
            nn.Linear(width, decoder_width, bias=False) if decoder_width != width else nn.Identity()
        )
        multiplier = 2 ** len(config["ratios"])
        channels = config["n_filters"] * multiplier
        decoder = [StreamingConv(decoder_width, channels, config["kernel_size"], num_slots)]
        for ratio in config["ratios"]:
            decoder.extend([nn.ELU(), StreamingTranspose(channels, channels // 2, ratio, num_slots)])
            channels //= 2
            for layer in range(config["n_residual_layers"]):
                decoder.append(
                    ResidualBlock(
                        channels,
                        config["residual_kernel_size"],
                        config["dilation_base"] ** layer,
                        config["compress"],
                        num_slots,
                    )
                )
        decoder.extend([nn.ELU(), StreamingConv(channels, config["channels"], config["last_kernel_size"], num_slots)])
        self.decoder = nn.ModuleList(decoder)
        self.samples_per_frame = round(config["sample_rate"] / config["frame_rate"])
        frames = self.upsample_ratio
        for layer in self.decoder:
            if isinstance(layer, StreamingTranspose):
                frames *= layer.convtr.stride[0]
            for module in layer.modules():
                if isinstance(module, StreamingConv):
                    module.scratch = torch.empty(num_slots, module.history.shape[1], module.history.shape[-1] + frames)

    def forward(self, latent, slots, positions):
        reset = positions == 0
        inputs = self.quantizer_projection(latent)
        if self.upsample is not None:
            inputs = self.upsample(inputs, slots, reset)
        inputs = self.input_projection(inputs.transpose(1, 2))
        position = positions[:, None] + torch.arange(inputs.shape[1], device=positions.device)
        angles = position[:, :, None, None].float() * self.frequencies
        rotary = angles.cos(), angles.sin()
        for layer in self.layers:
            inputs = layer(inputs, slots, positions, rotary)
        inputs = self.output_projection(inputs).transpose(1, 2)
        for layer in self.decoder:
            inputs = layer(inputs) if isinstance(layer, nn.ELU) else layer(inputs, slots, reset)
        return inputs

    def load_official_weights(self, weights):
        mapped = {}
        for name, tensor in weights.items():
            if not name.startswith("mimi."):
                continue
            target = name.removeprefix("mimi.")
            target = target.replace("quantizer.output_proj.", "quantizer_projection.")
            target = target.replace("upsample.convtr.convtr.", "upsample.convtr.")
            target = target.replace("decoder_transformer.transformer.layers.", "layers.")
            target = target.replace("decoder_transformer.input_proj.", "input_projection.")
            target = target.replace("decoder_transformer.output_proj.0.", "output_projection.")
            target = target.replace("decoder.model.", "decoder.")
            mapped[target] = tensor
        self.load_state_dict(mapped, strict=True)
