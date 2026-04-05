"""VibeVoice acoustic tokenizer decoder for Stage 1.

Receives continuous acoustic latent vectors from Stage 0 and decodes them
to PCM waveforms using a ConvNeXt-like causal decoder.

Architecture ported from microsoft/VibeVoice modular_vibevoice_tokenizer.py.
"""

from __future__ import annotations

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

logger = init_logger(__name__)


CONV_NORMALIZATIONS = frozenset(["none", "weight_norm", "spectral_norm",
                                  "time_layer_norm", "layer_norm", "time_group_norm"])


# ---------------------------------------------------------------------------
# Normalization modules
# ---------------------------------------------------------------------------

class ConvRMSNorm(nn.Module):
    """Channel-last RMSNorm for conv sequences (B, C, T)."""

    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> transpose -> normalize -> transpose back
        x = x.transpose(1, 2)
        output = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        output = output.type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output.transpose(1, 2)


class ConvLayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.layer_norm(x.float(), self.normalized_shape, self.weight.float(),
                         self.bias.float() if self.bias is not None else None, self.eps).type_as(x)
        return x.transpose(1, 2)


# ---------------------------------------------------------------------------
# Conv primitives
# ---------------------------------------------------------------------------

def _apply_parametrization_norm(module: nn.Module, norm: str = "none") -> nn.Module:
    if norm == "weight_norm":
        return nn.utils.parametrizations.weight_norm(module)
    elif norm == "spectral_norm":
        return nn.utils.spectral_norm(module)
    return module


def _get_norm_module(module: nn.Module, causal: bool, norm: str) -> nn.Module:
    if norm == "layer_norm":
        return ConvLayerNorm(module.out_channels)
    elif norm == "time_group_norm":
        return nn.GroupNorm(1, module.out_channels)
    return nn.Identity()


def _get_extra_padding_for_conv1d(x: torch.Tensor, kernel_size: int,
                                   stride: int, padding_total: int = 0) -> int:
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def _pad1d(x: torch.Tensor, paddings: tuple[int, int],
           mode: str = "constant", value: float = 0.0) -> torch.Tensor:
    length = x.shape[-1]
    padding_left, padding_right = paddings
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    return F.pad(x, paddings, mode, value)


class NormConv1d(nn.Module):
    def __init__(self, *args, causal: bool = False, norm: str = "none", **kwargs) -> None:
        super().__init__()
        self.conv = _apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = _get_norm_module(self.conv, causal, norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))


class NormConvTranspose1d(nn.Module):
    def __init__(self, *args, causal: bool = False, norm: str = "none", **kwargs) -> None:
        super().__init__()
        self.convtr = _apply_parametrization_norm(nn.ConvTranspose1d(*args, **kwargs), norm)
        self.norm = _get_norm_module(self.convtr, causal, norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.convtr(x))


class SConv1d(nn.Module):
    """Causal Conv1d with built-in padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, dilation: int = 1, groups: int = 1,
                 bias: bool = True, causal: bool = False, norm: str = "none",
                 pad_mode: str = "constant") -> None:
        super().__init__()
        self.conv = NormConv1d(in_channels, out_channels, kernel_size,
                               stride, dilation=dilation, groups=groups,
                               bias=bias, causal=causal, norm=norm)
        self.causal = causal
        self.pad_mode = pad_mode
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding_total = (kernel_size - 1) * dilation - (stride - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        extra_padding = _get_extra_padding_for_conv1d(x, self.kernel_size,
                                                       self.stride, self.padding_total)
        if self.causal:
            x = _pad1d(x, (self.padding_total, int(extra_padding)),
                        mode=self.pad_mode, value=0)
        else:
            pr = self.padding_total // 2
            pl = self.padding_total - pr
            x = _pad1d(x, (pl, pr + int(extra_padding)), mode=self.pad_mode)
        return self.conv(x)


class SConvTranspose1d(nn.Module):
    """Causal ConvTranspose1d with built-in unpadding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, causal: bool = False, norm: str = "none",
                 trim_right_ratio: float = 1.0, bias: bool = True) -> None:
        super().__init__()
        self.convtr = NormConvTranspose1d(in_channels, out_channels, kernel_size,
                                          stride, causal=causal, norm=norm, bias=bias)
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        self.padding_total = kernel_size - stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.convtr(x)
        if self.causal:
            padding_right = math.ceil(self.padding_total * self.trim_right_ratio)
            padding_left = self.padding_total - padding_right
        else:
            padding_right = self.padding_total // 2
            padding_left = self.padding_total - padding_right
        if padding_left + padding_right > 0:
            end = y.shape[-1] - padding_right if padding_right > 0 else y.shape[-1]
            y = y[..., padding_left:end]
        return y


# ---------------------------------------------------------------------------
# Block1D (ConvNeXt-like)
# ---------------------------------------------------------------------------

class FFN(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim, bias=bias)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.gelu(self.linear1(x)))


class Block1D(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 7, layer_scale_init_value: float = 1e-6,
                 **kwargs) -> None:
        super().__init__()
        layernorm = kwargs.get("layernorm", "RMSNorm")
        eps = kwargs.get("eps", 1e-6)
        causal = kwargs.get("causal", True)
        pad_mode = kwargs.get("pad_mode", "constant")
        norm = kwargs.get("norm", "none")
        bias = kwargs.get("bias", True)
        mixer_layer = kwargs.get("mixer_layer", "depthwise_conv")

        if layernorm == "RMSNorm":
            affine = kwargs.get("layernorm_elementwise_affine", True)
            NormCls = partial(ConvRMSNorm, elementwise_affine=affine)
        else:
            NormCls = partial(ConvLayerNorm)

        self.norm = NormCls(dim, eps=eps)
        self.ffn_norm = NormCls(dim, eps=eps)

        groups = dim if mixer_layer == "depthwise_conv" else kwargs.get("groups", 1)
        self.mixer = SConv1d(dim, dim, kernel_size, groups=groups, pad_mode=pad_mode,
                             norm=norm, causal=causal, bias=bias)

        ffn_expansion = kwargs.get("ffn_expansion", 4)
        self.ffn = FFN(dim, ffn_expansion * dim, bias=bias)

        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))
            self.ffn_gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        else:
            self.gamma = None
            self.ffn_gamma = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.mixer(x)
        if self.gamma is not None:
            x = x * self.gamma.unsqueeze(-1)
        x = residual + x

        residual = x
        x = self.ffn_norm(x)
        x = x.permute(0, 2, 1)
        x = self.ffn(x)
        x = x.permute(0, 2, 1)
        if self.ffn_gamma is not None:
            x = x * self.ffn_gamma.unsqueeze(-1)
        return residual + x


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class TokenizerDecoder(nn.Module):
    """ConvNeXt-based decoder: latent (B, D, T) -> waveform (B, 1, T*downsample)."""

    def __init__(self, config) -> None:
        super().__init__()
        self.dimension = config["dimension"]
        self.channels = config["channels"]
        self.n_filters = config["n_filters"]
        self.ratios = config["ratios"]
        self.depths = config["depths"]
        self.causal = config.get("causal", True)

        kernel_size = config.get("kernel_size", 7)
        norm = config.get("norm", "none")
        pad_mode = config.get("pad_mode", "constant")
        bias = config.get("bias", True)
        layernorm = config.get("layernorm", "RMSNorm")
        layernorm_eps = config.get("layernorm_eps", 1e-6)
        layernorm_elementwise_affine = config.get("layernorm_elementwise_affine", True)
        mixer_layer = config.get("mixer_layer", "depthwise_conv")
        layer_scale_init_value = config.get("layer_scale_init_value", 0)
        disable_last_norm = config.get("disable_last_norm", False)
        trim_right_ratio = config.get("trim_right_ratio", 1.0)

        num_stages = len(self.depths)
        # Stem: project from latent dim to widest filter count
        self.stem = SConv1d(self.dimension, self.n_filters * 2 ** (num_stages - 1),
                            kernel_size, norm=norm, causal=self.causal,
                            pad_mode=pad_mode, bias=bias)

        block_kwargs = dict(
            mixer_layer=mixer_layer, layernorm=layernorm, eps=layernorm_eps,
            causal=self.causal, pad_mode=pad_mode, norm=norm, bias=bias,
            layer_scale_init_value=layer_scale_init_value,
            layernorm_elementwise_affine=layernorm_elementwise_affine,
        )

        self.stages = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        for i in range(num_stages):
            in_ch = self.n_filters * 2 ** (num_stages - 1 - i)
            out_ch = in_ch // 2 if i < num_stages - 1 else in_ch
            stage = nn.Sequential(*[
                Block1D(dim=in_ch, kernel_size=kernel_size, **block_kwargs)
                for _ in range(self.depths[i])
            ])
            self.stages.append(stage)
            if i < len(self.ratios):
                self.upsample_layers.append(
                    SConvTranspose1d(in_ch, out_ch, kernel_size=self.ratios[i] * 2,
                                     stride=self.ratios[i], causal=self.causal,
                                     norm=norm, trim_right_ratio=trim_right_ratio, bias=bias)
                )

        final_ch = self.n_filters
        if not disable_last_norm:
            if layernorm == "RMSNorm":
                self.final_norm = ConvRMSNorm(final_ch, eps=layernorm_eps,
                                              elementwise_affine=layernorm_elementwise_affine)
            else:
                self.final_norm = ConvLayerNorm(final_ch, eps=layernorm_eps)
        else:
            self.final_norm = nn.Identity()

        self.head = SConv1d(final_ch, self.channels, kernel_size=7,
                            causal=self.causal, pad_mode=pad_mode, norm=norm, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D, T) latent -> (B, channels, T_audio)"""
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.upsample_layers):
                x = self.upsample_layers[i](x)
        x = self.final_norm(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# vLLM wrapper
# ---------------------------------------------------------------------------

def _build_decoder_config(hf_config) -> dict:
    """Build decoder config dict from the HF VibeVoiceTTSConfig."""
    ac = hf_config.acoustic_tokenizer_config
    encoder_ratios = ac.encoder_ratios if hasattr(ac, "encoder_ratios") else [8, 5, 5, 4, 2, 2]
    decoder_ratios = ac.decoder_ratios if hasattr(ac, "decoder_ratios") else list(reversed(encoder_ratios))

    encoder_depths_str = ac.encoder_depths if hasattr(ac, "encoder_depths") else "3-3-3-3-3-3-8"
    encoder_depths = [int(d) for d in encoder_depths_str.split("-")]
    if hasattr(ac, "decoder_depths") and ac.decoder_depths is not None:
        decoder_depths = [int(d) for d in ac.decoder_depths.split("-")]
    else:
        decoder_depths = list(reversed(encoder_depths))

    return {
        "dimension": ac.vae_dim,
        "channels": ac.channels if hasattr(ac, "channels") else 1,
        "n_filters": ac.decoder_n_filters if hasattr(ac, "decoder_n_filters") else 32,
        "ratios": decoder_ratios,
        "depths": decoder_depths,
        "causal": ac.causal if hasattr(ac, "causal") else True,
        "norm": ac.conv_norm if hasattr(ac, "conv_norm") else "none",
        "pad_mode": ac.pad_mode if hasattr(ac, "pad_mode") else "constant",
        "bias": ac.conv_bias if hasattr(ac, "conv_bias") else True,
        "layernorm": ac.layernorm if hasattr(ac, "layernorm") else "RMSNorm",
        "layernorm_eps": ac.layernorm_eps if hasattr(ac, "layernorm_eps") else 1e-5,
        "layernorm_elementwise_affine": getattr(ac, "layernorm_elementwise_affine", True),
        "mixer_layer": ac.mixer_layer if hasattr(ac, "mixer_layer") else "depthwise_conv",
        "layer_scale_init_value": ac.layer_scale_init_value if hasattr(ac, "layer_scale_init_value") else 1e-6,
        "disable_last_norm": ac.disable_last_norm if hasattr(ac, "disable_last_norm") else True,
        "kernel_size": 7,
    }


class VibeVoiceTTSAudioTokenizer(nn.Module):
    """Stage 1 model: acoustic tokenizer decoder that converts continuous
    latent frames to PCM waveforms.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        decoder_cfg = _build_decoder_config(config)
        self.decoder = TokenizerDecoder(decoder_cfg)
        self.sampling_rate = getattr(config, "sampling_rate", 24000)
        self._vae_dim = config.acoustic_vae_dim

        encoder_ratios = config.acoustic_tokenizer_config.encoder_ratios
        self.downsample_factor = int(np.prod(encoder_ratios))

    @property
    def frame_rate(self) -> float:
        return self.sampling_rate / self.downsample_factor

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent frames to audio.

        Args:
            latents: (B, T, D) continuous acoustic latent frames.

        Returns:
            (B, 1, T_audio) waveform tensor.
        """
        x = latents.transpose(1, 2)
        target_dtype = next(self.decoder.parameters()).dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)
        return self.decoder(x)

    def decode_batch(self, latent_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """Decode a list of variable-length latent tensors.

        Args:
            latent_list: list of (T_i, D) tensors.

        Returns:
            List of 1-D waveform tensors.
        """
        if not latent_list:
            return []

        device = next(self.decoder.parameters()).device
        dtype = next(self.decoder.parameters()).dtype

        max_len = max(lt.shape[0] for lt in latent_list)
        batch = torch.zeros(len(latent_list), max_len, self._vae_dim, device=device, dtype=dtype)
        lengths = []
        for i, lt in enumerate(latent_list):
            batch[i, :lt.shape[0]] = lt.to(device=device, dtype=dtype)
            lengths.append(lt.shape[0])

        with torch.no_grad():
            waveforms = self.decode(batch)  # (B, 1, T_audio)

        results = []
        for i, n_frames in enumerate(lengths):
            n_samples = n_frames * self.downsample_factor
            wav = waveforms[i, 0, :n_samples]
            results.append(wav)
        return results

    @staticmethod
    def _remap_weight_name(name: str) -> str:
        """Remap checkpoint weight names to match our model structure.

        The original Microsoft checkpoint uses a slightly different layout:
        - upsample_layers[0] is the stem (regular conv), our model uses a
          separate ``stem`` attribute.
        - Each upsample layer is wrapped in nn.Sequential (extra ``.0.``).
        - Mixer conv has an extra level of wrapping (``.conv.conv.conv``
          vs our ``.conv.conv``).
        """
        import re
        # upsample_layers.0.0.conv.conv -> stem.conv.conv (stem projection)
        name = re.sub(
            r"^(decoder\.)upsample_layers\.0\.0\.",
            r"\1stem.",
            name,
        )
        # upsample_layers.{N+1}.0. -> upsample_layers.{N}. (shift index, strip .0.)
        m = re.match(r"^(decoder\.)upsample_layers\.(\d+)\.0\.(.*)", name)
        if m:
            prefix, idx_str, rest = m.group(1), int(m.group(2)), m.group(3)
            name = f"{prefix}upsample_layers.{idx_str - 1}.{rest}"
        # mixer.conv.conv.conv -> mixer.conv.conv (strip extra wrapper level)
        name = name.replace(".mixer.conv.conv.conv.", ".mixer.conv.conv.")
        return name

    def load_weight(self, weight: tuple[str, torch.Tensor]) -> str:
        params_dict = dict(self.named_parameters())
        name, loaded_weight = weight
        name = self._remap_weight_name(name)
        if name not in params_dict:
            logger.warning("Audio tokenizer weight %s not found (UNUSED)", name)
            return name
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)
        return name

    def load_weights(self, weights) -> set[str]:
        loaded = set()
        for name, w in weights:
            loaded_name = self.load_weight((name, w))
            loaded.add(loaded_name)
        return loaded
