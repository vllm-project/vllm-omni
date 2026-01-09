# Copyright (c) 2024 NVIDIA CORPORATION.
# Licensed under the MIT license.
#
# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
# Portions from https://github.com/NVIDIA/BigVGAN under the MIT license.
# See NOTICE file for license details.

import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm

from vllm_omni.diffusion.models.hyperclovax_audio.activations import Activation1d, SnakeBeta
from vllm_omni.diffusion.models.hyperclovax_audio.ecapa_tdnn import ECAPA_TDNN


# Dataclass for model hyper-parameters
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_hparams_from_json(path) -> AttrDict:
    with open(path) as f:
        data = f.read()
    return AttrDict(json.loads(data))


# Functions for model initialization
def init_weights(m, mean=0.0, std=0.01):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class CausalConv1d(nn.Module):
    """1D causal convloution w/ 1-side padding."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        pad_buffer=None,
    ):
        super().__init__()
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        )
        self.stride = stride
        self.pad_length = (kernel_size - 1) * dilation

        # TODO: deprecate pad_buffer and inference. Remove in the future
        if pad_buffer is None:
            pad_buffer = torch.zeros(1, in_channels, self.pad_length)
        self.register_buffer("pad_buffer", pad_buffer)

    def forward(self, x, hidden_states=None):
        if hidden_states is None:
            x = nn.functional.pad(x, (self.pad_length, 0), "constant", value=0.0)
        else:
            assert hidden_states.shape[-1] >= self.pad_length
            hidden_states = hidden_states[:, :, -self.pad_length :]
            x = torch.cat((hidden_states, x), -1)
        return self.conv(x), x[:, :, -self.pad_length :].detach()


class CausalConvTranspose1d(nn.Module):
    """1D causal transpose convloution."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        pad_buffer=None,
    ):
        super().__init__()
        self.deconv = weight_norm(
            nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                output_padding=0,
                groups=groups,
                bias=bias,
            )
        )
        self.stride = stride
        self.pad_length = math.ceil(kernel_size / stride) - 1
        self.pad = nn.ReplicationPad1d((self.pad_length, 0))

        # TODO: deprecate pad_buffer and inference. Remove in the future
        if pad_buffer is None:
            pad_buffer = torch.zeros(1, in_channels, self.pad_length)
        self.register_buffer("pad_buffer", pad_buffer)

    def forward(self, x, hidden_states=None):
        if hidden_states is None:
            x = self.pad(x)
        else:
            assert hidden_states.shape[-1] >= self.pad_length
            hidden_states = hidden_states[:, :, -self.pad_length :]
            x = torch.cat((hidden_states, x), -1)
        return (
            self.deconv(x)[:, :, self.stride : -self.stride],
            x[:, :, -self.pad_length :].detach(),
        )


class NonCausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        **kwargs,
    ):
        super().__init__()
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                **kwargs,
            )
        )
        self.pad_length = ((kernel_size - 1) * dilation) // 2

    def forward(self, x, hidden_states=None):
        if hidden_states is None:
            out = self.conv(x)
        else:
            assert hidden_states.shape[-1] >= self.pad_length
            hidden_states = hidden_states[:, :, -self.pad_length :]
            x_ = torch.cat((hidden_states, x), -1)
            out = self.conv(x_)[:, :, self.pad_length :]
        return out, x[:, :, -self.pad_length :].detach()


class NonCausalConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        **kwargs,
    ):
        super().__init__()
        self.deconv = weight_norm(
            nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                groups=groups,
                bias=bias,
                **kwargs,
            )
        )
        self.stride = stride
        self.pad_length = (kernel_size - stride) // 2

    def forward(self, x, hidden_states=None):
        if hidden_states is None:
            out = self.deconv(x)
        else:
            assert hidden_states.shape[-1] >= self.pad_length
            hidden_states = hidden_states[:, :, -self.pad_length :]
            x_ = torch.cat((hidden_states, x), -1)
            out = self.deconv(x_)[:, :, self.pad_length * self.stride :]
        return out, x[:, :, -self.pad_length :].detach()


class AMPBlock1(torch.nn.Module):
    """
    AMPBlock applies Snake / SnakeBeta activation functions with trainable parameters
    that control periodicity, defined for each layer.
    AMPBlock1 has additional self.convs2 that contains additional Conv1d layers with a fixed dilation=1
    followed by each layer in self.convs1

    Args:
        h (AttrDict): Hyperparameters.
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'. Default is None.
    """

    def __init__(
        self,
        h: AttrDict,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = None,
        causal: bool = False,
    ):
        super().__init__()
        conv1d = CausalConv1d if causal else NonCausalConv1d

        self.h = h

        self.convs1 = nn.ModuleList(
            [
                conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=d,
                    padding=get_padding(kernel_size, d),
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                )
                for _ in range(len(dilation))
            ]
        )
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(
            self.convs2
        )  # Total number of conv layers

        # Activation functions
        if activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale),
                        causal=False,
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x, hidden_states=None):
        if hidden_states is None:
            hidden_states = [(None, None, None, None)] * len(self.convs1)

        hidden_states_new = []
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2, (h_a1, h_c1, h_a2, h_c2) in zip(
            self.convs1, self.convs2, acts1, acts2, hidden_states
        ):
            xt, ht_a1 = a1(x, h_a1)
            xt, ht_c1 = c1(xt, h_c1)
            xt, ht_a2 = a2(xt, h_a2)
            xt, ht_c2 = c2(xt, h_c2)
            x = xt + x
            hidden_states_new.append((ht_a1, ht_c1, ht_a2, ht_c2))

        return x, hidden_states_new

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class HyperCLOVAXAudioDecoderModel(nn.Module):
    """
    HyperCLOVAXAudioDecoderModel is a neural vocoder model that applies anti-aliased periodic activation for residual blocks (resblocks).

    Args:
        h (AttrDict): Hyperparameters.

    Note:
        Ensure that the activation function is correctly specified in the hyperparameters (h.activation).
    """

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        h: AttrDict
    ):
        super().__init__()

        self.h = h
        self.od_config = od_config

        self.causal = h.get("causal", True)
        conv1d = CausalConv1d if self.causal else NonCausalConv1d
        convtranspose1d = (
            CausalConvTranspose1d if self.causal else NonCausalConvTranspose1d
        )

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        self.finetune = getattr(h, "finetune", False)
        # Speaker embedding
        if not self.finetune:
            self.spk_emb = ECAPA_TDNN(
                in_channel=h.num_mels,
                hidden_channel=h.spk_hidden_dim,
                emb_dim=h.spk_emb_dim,
                global_context_att=h.global_context_att,
            )
        else:
            self.spk_emb = nn.Embedding(h.num_spk, h.spk_emb_dim)

        # Unit embedding
        self.unit_emb = nn.Embedding(h.num_units, h.unit_emb_dim)
        self.unit_emb_dim = h.unit_emb_dim

        # Pre-conv
        self.conv_pre = conv1d(
            h.unit_emb_dim + h.spk_emb_dim, h.upsample_initial_channel, 7, 1, padding=3
        )

        # Define which AMPBlock to use. BigVGAN uses AMPBlock1 as default
        if h.resblock == "1":
            resblock_class = AMPBlock1
        else:
            raise ValueError(
                f"Incorrect resblock class specified in hyperparameters. Got {h.resblock}"
            )

        # Transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        convtranspose1d(
                            h.upsample_initial_channel // (2**i),
                            h.upsample_initial_channel // (2 ** (i + 1)),
                            k,
                            u,
                            padding=math.ceil((k - u) / 2),
                            output_padding=(k - u) % 2,
                        )
                    ]
                )
            )

        # Residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)
            ):
                self.resblocks.append(
                    resblock_class(
                        h, ch, k, d, activation=h.activation, causal=self.causal
                    )
                )

        # Post-conv
        activation_post = (
            SnakeBeta(ch, alpha_logscale=h.snake_logscale)
            if h.activation == "snakebeta"
            else None
        )
        if activation_post is None:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

        self.activation_post = Activation1d(activation=activation_post, causal=False)

        # Whether to use bias for the final conv_post. Default to True for backward compatibility
        self.use_bias_at_final = h.get("use_bias_at_final", True)
        self.conv_post = conv1d(ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final)

        # Weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

        # Final tanh activation. Defaults to True for backward compatibility
        self.use_tanh_at_final = h.get("use_tanh_at_final", True)

        self.num_layers = self.num_upsamples + self.num_upsamples * self.num_kernels + 3

    def forward_with_spk_emb(self, x, spk_or_ref, hidden_states=None):
        spk_emb = self.spk_emb(spk_or_ref)
        return self(x, spk_emb, hidden_states=hidden_states)

    def forward(self, x, spk_emb, hidden_states=None):
        if hidden_states is None:
            hidden_states = [None] * self.num_layers
        else:
            assert len(hidden_states) == self.num_layers, (
                f"Expected hidden_states to have {self.num_layers} elements, but got {len(hidden_states)}."
            )

        hidden_state_iter = iter(hidden_states)
        hidden_states_new = []

        # Unit and speaker embedding
        x = self.unit_emb(x).transpose(1, 2) * (self.unit_emb_dim**-0.5)
        if self.finetune:
            spk_emb = spk_emb.transpose(1, 2).expand(-1, -1, x.shape[-1])
        else:
            spk_emb = spk_emb.unsqueeze(2).expand(-1, -1, x.shape[-1])

        x = torch.cat([x, spk_emb], dim=1)
        x, h = self.conv_pre(x, next(hidden_state_iter))
        hidden_states_new.append(h)

        for i, up_layers in enumerate(self.ups):
            # Upsampling
            for up_layer in up_layers:
                x, h = up_layer(x, next(hidden_state_iter))
                hidden_states_new.append(h)
            # AMP blocks
            resblock_outputs = [
                self.resblocks[i * self.num_kernels + j](x, next(hidden_state_iter))
                for j in range(self.num_kernels)
            ]
            x = sum(o for o, _ in resblock_outputs) / self.num_kernels
            hidden_states_new.extend([h for _, h in resblock_outputs])

        # Post-conv
        x, h = self.activation_post(x, next(hidden_state_iter))
        hidden_states_new.append(h)
        x, h = self.conv_post(x, next(hidden_state_iter))
        hidden_states_new.append(h)
        # Final tanh activation
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)  # Bound the output to [-1, 1]

        return x, hidden_states_new

    def remove_weight_norm(self):
        try:
            print("Removing weight norm...")
            for l in self.ups:
                for l_i in l:
                    remove_weight_norm(l_i)
            for l in self.resblocks:
                l.remove_weight_norm()
            remove_weight_norm(self.conv_pre)
            remove_weight_norm(self.conv_post)
        except ValueError:
            print("[INFO] Model already removed weight norm. Skipping!")
            pass

    @classmethod
    def from_pretrained(
        cls,
        ckpt_path: str,
        config_path: Optional[str] = None,
        map_location: str = "cpu",  # Additional argument
    ):
        """Load Pytorch pretrained weights and return the loaded model."""

        # Load hyperparameters (h) used by BigVGAN
        if config_path is None:
            print("Loading config.json from local directory")
            config_path = Path(ckpt_path).with_name("config.json")

        h = load_hparams_from_json(config_path)

        # instantiate BigVGAN using h
        model = cls(h)

        # Load pretrained generator weight
        print("Loading weights from local directory")
        checkpoint_dict = torch.load(ckpt_path, map_location=map_location)

        try:
            model.load_state_dict(checkpoint_dict["generator"])
        except RuntimeError:
            print(
                "[INFO] the pretrained checkpoint does not contain weight norm. "
                "Loading the checkpoint after removing weight norm!"
            )
            model.remove_weight_norm()
            model.load_state_dict(checkpoint_dict["generator"])

        return model