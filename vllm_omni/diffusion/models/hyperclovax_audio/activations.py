# Implementation adapted from https://github.com/EdwardDixon/snake under the MIT license.
# See NOTICE file for license details.

import math

import torch
import torch.nn.functional as F
from torch import nn, pow, sin
from torch.nn import Parameter

if "sinc" in dir(torch):
    sinc = torch.sinc
else:
    # This code is adopted from adefossez's julius.core.sinc under the MIT License
    # https://adefossez.github.io/julius/julius/core.html
    # See NOTICE file for license details.
    def sinc(x: torch.Tensor):
        """
        Implementation of sinc, i.e. sin(pi * x) / (pi * x)
        __Warning__: Different to julius.sinc, the input is multiplied by `pi`!
        """
        return torch.where(
            x == 0,
            torch.tensor(1.0, device=x.device, dtype=x.dtype),
            torch.sin(math.pi * x) / math.pi / x,
        )


# This code is adopted from adefossez's julius.lowpass.LowPassFilters under the MIT License
# https://adefossez.github.io/julius/julius/lowpass.html
# See NOTICE file for license details.
def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):  # return filter [1,1,kernel_size]
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # For kaiser window
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    # ratio = 0.5/cutoff -> 2 * cutoff = 1 / ratio
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
        """
        Normalize filter to have sum = 1,
        otherwise we will have a small leakage of the constant component in the input signal.
        """
        filter_ /= filter_.sum()
        filter = filter_.view(1, 1, kernel_size)

    return filter


class LowPassFilter1d(nn.Module):
    def __init__(
        self,
        cutoff=0.5,
        half_width=0.6,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "replicate",
        kernel_size: int = 12,
        causal: bool = False,
    ):
        """
        kernel_size should be even number for stylegan3 setup, in this implementation, odd number is also possible.
        """
        super().__init__()
        if cutoff < -0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        self.causal = causal
        if self.causal:
            self.pad_left = kernel_size - 1
            self.pad_right = 0
        else:
            self.even = kernel_size % 2 == 0
            self.pad_left = kernel_size // 2 - int(self.even)
            self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter)

    # Input [B, C, T]
    def forward(self, x, hidden_states=None):
        _, C, _ = x.shape
        hs = x[..., -self.pad_left :]

        if self.padding:
            if self.causal:
                if hidden_states is not None:
                    assert hidden_states.shape[-1] >= self.pad_left
                    hidden_states = hidden_states[..., -self.pad_left :]
                    x = torch.cat([hidden_states, x], dim=-1)
                else:
                    x = F.pad(x, (self.pad_left, 0), mode="constant", value=0.0)
            else:
                x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        out = F.conv1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)

        return out, hs


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None, causal=False):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.causal = causal

        self.half_left = (kernel_size - ratio) // 2
        self.half_right = (kernel_size - ratio + 1) // 2

        filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size)
        self.register_buffer("filter", filter)

    # x: [B, C, T]
    def forward(self, x, hidden_states=None):
        _, C, _ = x.shape
        hs = x[..., -self.pad :]

        pad_left = self.pad
        pad_right = 0 if self.causal else self.pad

        if hidden_states is not None:
            assert hidden_states.shape[-1] >= self.pad
            hidden_states = hidden_states[..., -self.pad :]
            x = torch.cat([hidden_states, x], dim=-1)
            if pad_right > 0:
                x = F.pad(x, (0, pad_right), mode="replicate")
        else:
            x = F.pad(x, (pad_left, pad_right), mode="replicate")

        x = self.ratio * F.conv_transpose1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)

        crop_left = pad_left * self.stride + self.half_left
        crop_right = pad_right * self.stride + self.half_right
        if crop_right > 0:
            x = x[..., crop_left:-crop_right]
        else:
            x = x[..., crop_left:]

        return x, hs.detach()


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None, causal=False):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
            causal=causal,
        )

    def forward(self, x, hidden_states=None):
        xx, hs = self.lowpass(x, hidden_states)

        return xx, hs.detach()


class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper
          by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
          https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        """
        super().__init__()
        self.in_features = in_features

        # Initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # Log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:  # Linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # Line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x


class Activation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
        causal: bool = False,
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size, causal)
        self.downsample = DownSample1d(down_ratio, down_kernel_size, causal)

    # x: [B,C,T]
    def forward(self, x, hidden_states=None):
        if hidden_states is None:
            hidden_states = [None] * 2
        else:
            assert len(hidden_states) == 2

        x, h_up = self.upsample(x, hidden_states[0])
        x = self.act(x)
        x, h_down = self.downsample(x, hidden_states[-1])

        return x, (h_up.detach(), h_down.detach())
