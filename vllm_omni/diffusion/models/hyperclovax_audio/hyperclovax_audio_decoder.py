# Copyright (c) 2024 NVIDIA CORPORATION.
# Licensed under the MIT license.
#
# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
# Portions from https://github.com/NVIDIA/BigVGAN under the MIT license.
# See NOTICE file for license details.

import inspect
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.hyperclovax_audio.activations import (
    Activation1d,
    SnakeBeta,
)
from vllm_omni.diffusion.models.hyperclovax_audio.ecapa_tdnn import ECAPA_TDNN

logger = init_logger(__name__)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def load_hparams_from_json(path) -> AttrDict:
    with open(path) as f:
        data = f.read()
    return AttrDict(json.loads(data))


def init_weights(m, mean=0.0, std=0.01):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class CausalConv1d(nn.Module):
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
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, **kwargs
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
    """Anti-aliased Multi-Periodicity residual block with SnakeBeta activations."""

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
                conv1d(channels, channels, kernel_size, stride=1, dilation=d, padding=get_padding(kernel_size, d))
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                conv1d(channels, channels, kernel_size, stride=1, dilation=1, padding=get_padding(kernel_size, 1))
                for _ in range(len(dilation))
            ]
        )
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(self.convs2)

        if activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    Activation1d(activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale), causal=False)
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
        for c1, c2, a1, a2, (h_a1, h_c1, h_a2, h_c2) in zip(self.convs1, self.convs2, acts1, acts2, hidden_states):
            xt, ht_a1 = a1(x, h_a1)
            xt, ht_c1 = c1(xt, h_c1)
            xt, ht_a2 = a2(xt, h_a2)
            xt, ht_c2 = c2(xt, h_c2)
            x = xt + x
            hidden_states_new.append((ht_a1, ht_c1, ht_a2, ht_c2))

        return x, hidden_states_new

    def remove_weight_norm(self):
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class HyperCLOVAXAudioDecoderModel(nn.Module):
    """Unit-BigVGAN vocoder: discrete audio codes → 24kHz waveform.

    Args:
        od_config: OmniDiffusionConfig containing model hyperparameters.
    """

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        resblock: str = "1",
        causal: bool = False,
        finetune: bool = True,
        upsample_rates: list[int] = [5, 4, 4, 3, 2, 2],
        upsample_kernel_sizes: list[int] = [10, 8, 8, 6, 4, 4],
        upsample_initial_channel: int = 1536,
        resblock_kernel_sizes: list[int] = [3, 7, 11],
        resblock_dilation_sizes: list[list[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        use_tanh_at_final: bool = False,
        use_bias_at_final: bool = False,
        activation: str = "snakebeta",
        snake_logscale: bool = True,
        num_units: int = 6561,
        unit_emb_dim: int = 1280,
        num_mels: int = 100,
        n_fft: int = 1024,
        hop_size: int = 256,
        win_size: int = 1024,
        spk_emb_dim: int = 256,
        spk_hidden_dim: int = 512,
        global_context_att: bool = False,
        sampling_rate: int = 24000,
        fmin: int = 0,
        fmax: int = 8000,
        num_spk: int = 26,
    ):
        super().__init__()

        self.h = AttrDict(
            {
                "resblock": resblock,
                "causal": causal,
                "finetune": finetune,
                "upsample_rates": upsample_rates,
                "upsample_kernel_sizes": upsample_kernel_sizes,
                "upsample_initial_channel": upsample_initial_channel,
                "resblock_kernel_sizes": resblock_kernel_sizes,
                "resblock_dilation_sizes": resblock_dilation_sizes,
                "use_tanh_at_final": use_tanh_at_final,
                "use_bias_at_final": use_bias_at_final,
                "activation": activation,
                "snake_logscale": snake_logscale,
                "num_units": num_units,
                "unit_emb_dim": unit_emb_dim,
                "num_mels": num_mels,
                "n_fft": n_fft,
                "hop_size": hop_size,
                "win_size": win_size,
                "spk_emb_dim": spk_emb_dim,
                "spk_hidden_dim": spk_hidden_dim,
                "global_context_att": global_context_att,
                "sampling_rate": sampling_rate,
                "fmin": fmin,
                "fmax": fmax,
                "num_spk": num_spk,
            }
        )

        self.causal = self.h.get("causal", True)
        conv1d = CausalConv1d if self.causal else NonCausalConv1d
        convtranspose1d = CausalConvTranspose1d if self.causal else NonCausalConvTranspose1d

        self.num_kernels = len(self.h.resblock_kernel_sizes)
        self.num_upsamples = len(self.h.upsample_rates)

        self.finetune = getattr(self.h, "finetune", False)
        if not self.finetune:
            self.spk_emb = ECAPA_TDNN(
                in_channel=self.h.num_mels,
                hidden_channel=self.h.spk_hidden_dim,
                emb_dim=self.h.spk_emb_dim,
                global_context_att=self.h.global_context_att,
            )
        else:
            self.spk_emb = nn.Embedding(self.h.num_spk, self.h.spk_emb_dim)

        self.unit_emb = nn.Embedding(self.h.num_units, self.h.unit_emb_dim)
        self.unit_emb_dim = self.h.unit_emb_dim

        self.conv_pre = conv1d(
            self.h.unit_emb_dim + self.h.spk_emb_dim, self.h.upsample_initial_channel, 7, 1, padding=3
        )

        if self.h.resblock == "1":
            resblock_class = AMPBlock1
        else:
            raise ValueError(f"Incorrect resblock class specified. Got {self.h.resblock}")

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(self.h.upsample_rates, self.h.upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        convtranspose1d(
                            self.h.upsample_initial_channel // (2**i),
                            self.h.upsample_initial_channel // (2 ** (i + 1)),
                            k,
                            u,
                            padding=math.ceil((k - u) / 2),
                            output_padding=(k - u) % 2,
                        )
                    ]
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = self.h.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(self.h.resblock_kernel_sizes, self.h.resblock_dilation_sizes):
                self.resblocks.append(
                    resblock_class(self.h, ch, k, d, activation=self.h.activation, causal=self.causal)
                )

        activation_post = (
            SnakeBeta(ch, alpha_logscale=self.h.snake_logscale) if self.h.activation == "snakebeta" else None
        )
        if activation_post is None:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

        self.activation_post = Activation1d(activation=activation_post, causal=False)
        self.use_bias_at_final = self.h.get("use_bias_at_final", True)
        self.conv_post = conv1d(ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final)

        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

        self.use_tanh_at_final = self.h.get("use_tanh_at_final", True)
        self.num_layers = self.num_upsamples + self.num_upsamples * self.num_kernels + 3

    def forward_with_spk_emb(self, x, spk_or_ref, hidden_states=None):
        spk_emb = self.spk_emb(spk_or_ref)
        return self(x, spk_emb, hidden_states=hidden_states)

    def forward(self, x, spk_emb, hidden_states=None):
        if hidden_states is None:
            hidden_states = [None] * self.num_layers
        else:
            assert len(hidden_states) == self.num_layers

        hidden_state_iter = iter(hidden_states)
        hidden_states_new = []

        x = self.unit_emb(x).transpose(1, 2) * (self.unit_emb_dim**-0.5)
        if self.finetune:
            spk_emb = spk_emb.transpose(1, 2).expand(-1, -1, x.shape[-1])
        else:
            spk_emb = spk_emb.unsqueeze(2).expand(-1, -1, x.shape[-1])

        x = torch.cat([x, spk_emb], dim=1)
        x, h = self.conv_pre(x, next(hidden_state_iter))
        hidden_states_new.append(h)

        for i, up_layers in enumerate(self.ups):
            for up_layer in up_layers:
                x, h = up_layer(x, next(hidden_state_iter))
                hidden_states_new.append(h)
            resblock_outputs = [
                self.resblocks[i * self.num_kernels + j](x, next(hidden_state_iter)) for j in range(self.num_kernels)
            ]
            x = sum(o for o, _ in resblock_outputs) / self.num_kernels
            hidden_states_new.extend([h for _, h in resblock_outputs])

        x, h = self.activation_post(x, next(hidden_state_iter))
        hidden_states_new.append(h)
        x, h = self.conv_post(x, next(hidden_state_iter))
        hidden_states_new.append(h)

        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)

        return x, hidden_states_new

    def remove_weight_norm(self):
        try:
            logger.info("Removing weight norm...")
            for layer in self.ups:
                for l_i in layer:
                    remove_weight_norm(l_i)
            for layer in self.resblocks:
                layer.remove_weight_norm()
            remove_weight_norm(self.conv_pre)
            remove_weight_norm(self.conv_post)
        except ValueError:
            logger.info("Model already removed weight norm. Skipping!")

    @classmethod
    def from_pretrained(cls, ckpt_path: str, config_path: str | None = None, map_location: str = "cpu"):
        """Load pretrained weights and return the model.

        BUG FIX: Original PR #869 passed AttrDict to cls(h) but __init__
        expects OmniDiffusionConfig. Fixed to unpack AttrDict as kwargs.
        """
        if config_path is None:
            logger.info("Loading config.json from local directory")
            config_path = Path(ckpt_path).with_name("config.json")

        h = load_hparams_from_json(config_path)

        # Keep only kwargs supported by __init__; MAR configs contain
        # training-only fields (e.g., num_gpus) that are irrelevant here.
        init_sig = inspect.signature(cls.__init__)
        valid_params = set(init_sig.parameters.keys()) - {"self", "od_config"}
        init_kwargs = {k: v for k, v in dict(h).items() if k in valid_params}

        model = cls(od_config=None, **init_kwargs)

        logger.info("Loading weights from local directory")
        checkpoint_dict = torch.load(ckpt_path, map_location=map_location)

        try:
            model.load_state_dict(checkpoint_dict["generator"])
        except RuntimeError:
            logger.info("Pretrained checkpoint does not contain weight norm. Loading after removing weight norm.")
            model.remove_weight_norm()
            model.load_state_dict(checkpoint_dict["generator"])

        return model
