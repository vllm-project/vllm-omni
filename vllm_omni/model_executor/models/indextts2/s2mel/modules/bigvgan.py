# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

import json
import os

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

from vllm_omni.model_executor.models.common.alias_free_activation import AliasFreeActivation1d
from vllm_omni.model_executor.models.common.snake_activation import Snake, SnakeBeta
from vllm_omni.model_executor.models.indextts2.s2mel.modules.commons import AttrDict

# ---------------------------------------------------------------------------
# Helpers (inlined from env.py / utils.py)
# ---------------------------------------------------------------------------


def load_hparams_from_json(path) -> AttrDict:
    with open(path) as f:
        data = f.read()
    return AttrDict(json.loads(data))


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


# ---------------------------------------------------------------------------
# AMP Residual Blocks
# ---------------------------------------------------------------------------


class AMPBlock1(torch.nn.Module):
    """AMPBlock with Snake/SnakeBeta activations and extra convs2 (dilation=1)."""

    def __init__(
        self,
        h: AttrDict,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = None,
    ):
        super().__init__()

        self.h = h

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for _ in range(len(dilation))
            ]
        )
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(self.convs2)

        if activation == "snake":
            self.activations = nn.ModuleList(
                [
                    AliasFreeActivation1d(activation=Snake(channels, alpha_logscale=h.snake_logscale))
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    AliasFreeActivation1d(activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale))
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
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

    def remove_weight_norm(self):
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class AMPBlock2(torch.nn.Module):
    """AMPBlock with Snake/SnakeBeta activations, no extra convs2."""

    def __init__(
        self,
        h: AttrDict,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = None,
    ):
        super().__init__()

        self.h = h

        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs.apply(init_weights)

        self.num_layers = len(self.convs)

        if activation == "snake":
            self.activations = nn.ModuleList(
                [
                    AliasFreeActivation1d(activation=Snake(channels, alpha_logscale=h.snake_logscale))
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    AliasFreeActivation1d(activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale))
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        for c, a in zip(self.convs, self.activations):
            xt = a(x)
            xt = c(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for layer in self.convs:
            remove_weight_norm(layer)


# ---------------------------------------------------------------------------
# BigVGAN Generator
# ---------------------------------------------------------------------------


class BigVGAN(
    torch.nn.Module,
    PyTorchModelHubMixin,
    library_name="bigvgan",
    repo_url="https://github.com/NVIDIA/BigVGAN",
    docs_url="https://github.com/NVIDIA/BigVGAN/blob/main/README.md",
    pipeline_tag="audio-to-audio",
    license="mit",
    tags=["neural-vocoder", "audio-generation", "arxiv:2206.04658"],
):
    """BigVGAN neural vocoder with anti-aliased periodic activation."""

    def __init__(self, h: AttrDict, use_cuda_kernel: bool = False):
        super().__init__()
        self.h = h

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        # Pre-conv
        self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))

        # Define which AMPBlock to use. BigVGAN uses AMPBlock1 as default
        if h.resblock == "1":
            resblock_class = AMPBlock1
        elif h.resblock == "2":
            resblock_class = AMPBlock2
        else:
            raise ValueError(f"Incorrect resblock class specified in hyperparameters. Got {h.resblock}")

        # Transposed conv-based upsamplers. does not apply anti-aliasing
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
                                padding=(k - u) // 2,
                            )
                        )
                    ]
                )
            )

        # Residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock_class(h, ch, k, d, activation=h.activation))

        # Post-conv
        activation_post = (
            Snake(ch, alpha_logscale=h.snake_logscale)
            if h.activation == "snake"
            else (SnakeBeta(ch, alpha_logscale=h.snake_logscale) if h.activation == "snakebeta" else None)
        )
        if activation_post is None:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

        self.activation_post = AliasFreeActivation1d(activation=activation_post)

        # Whether to use bias for the final conv_post
        self.use_bias_at_final = h.get("use_bias_at_final", True)
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final))

        # Weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

        # Final tanh activation
        self.use_tanh_at_final = h.get("use_tanh_at_final", True)

    def forward(self, x):
        # Pre-conv
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            # Upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # Post-conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        # Final tanh activation
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)

        return x

    def remove_weight_norm(self):
        try:
            for up_layer in self.ups:
                for sub in up_layer:
                    remove_weight_norm(sub)
            for block in self.resblocks:
                block.remove_weight_norm()
            remove_weight_norm(self.conv_pre)
            remove_weight_norm(self.conv_post)
        except ValueError:
            pass

    @classmethod
    def _from_pretrained(cls, *, model_id: str, map_location: str = "cpu", **kwargs):
        def _resolve(filename):
            if os.path.isdir(model_id):
                return os.path.join(model_id, filename)
            return hf_hub_download(
                repo_id=model_id,
                filename=filename,
                **{
                    k: kwargs[k]
                    for k in (
                        "revision",
                        "cache_dir",
                        "force_download",
                        "proxies",
                        "resume_download",
                        "token",
                        "local_files_only",
                    )
                    if k in kwargs
                },
            )

        h = load_hparams_from_json(_resolve("config.json"))
        model = cls(h)

        checkpoint_dict = torch.load(_resolve("bigvgan_generator.pt"), map_location=map_location)
        try:
            model.load_state_dict(checkpoint_dict["generator"], strict=False)
        except RuntimeError:
            model.remove_weight_norm()
            model.load_state_dict(checkpoint_dict["generator"], strict=False)

        return model
