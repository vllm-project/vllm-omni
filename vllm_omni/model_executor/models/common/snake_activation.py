# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
"""Shared SnakeBeta activation for speech decoders (Qwen3-TTS, Qwen3-Omni Code2Wav)."""

import torch
from torch import nn
from torch.nn import Parameter
from vllm.logger import init_logger

logger = init_logger(__name__)


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
          https://huggingface.co/papers/2006.08195
    """

    _triton_kernel = None  # None = untried, False = unavailable, callable = ready
    _TRITON_MAX_BLOCK_T = 4096  # upper bound for time-axis tile size

    @staticmethod
    def _init_triton():
        """Load and JIT-compile the fused Triton kernel (once)."""
        if SnakeBeta._triton_kernel is not None:
            return SnakeBeta._triton_kernel is not False
        try:
            import triton
            import triton.language as tl
        except ImportError:
            SnakeBeta._triton_kernel = False
            return False

        @triton.jit
        def _kernel(  # noqa: N803
            x_ptr,
            exp_alpha_ptr,
            inv_beta_ptr,
            out_ptr,
            stride_b,
            stride_c,
            t_len,
            block_t: tl.constexpr,
        ):
            """Fused SnakeBeta using precomputed exp(α) and 1/(exp(β)+ε)."""
            bid = tl.program_id(0)
            cid = tl.program_id(1)
            t_off = tl.program_id(2) * block_t + tl.arange(0, block_t)
            mask = t_off < t_len

            x = tl.load(x_ptr + bid * stride_b + cid * stride_c + t_off, mask=mask, other=0.0)
            ea = tl.load(exp_alpha_ptr + cid)
            ib = tl.load(inv_beta_ptr + cid)
            sin_val = tl.sin(x * ea)
            result = x + ib * sin_val * sin_val

            tl.store(out_ptr + bid * stride_b + cid * stride_c + t_off, result, mask=mask)

        SnakeBeta._triton_kernel = _kernel
        return True

    def __init__(self, in_features, alpha=1.0):
        super().__init__()
        self.in_features = in_features

        self.alpha = Parameter(torch.zeros(in_features) * alpha)
        self.beta = Parameter(torch.zeros(in_features) * alpha)

        self.no_div_by_zero = 0.000000001

        # Precomputed buffers (populated by precompute_exp_cache)
        self.register_buffer("_exp_alpha", None, persistent=False)
        self.register_buffer("_inv_beta", None, persistent=False)

    def precompute_exp_cache(self):
        """Materialize exp(alpha) and 1/(exp(beta)+eps) as frozen buffers."""
        with torch.no_grad():
            self._exp_alpha = torch.exp(self.alpha).contiguous()
            self._inv_beta = (1.0 / (torch.exp(self.beta) + self.no_div_by_zero)).contiguous()

    @property
    def _cached(self):
        return self._exp_alpha is not None

    def forward(self, hidden_states):
        """SnakeBeta := x + 1/b * sin^2(x*a)"""
        if hidden_states.is_cuda and not torch.is_grad_enabled() and self._init_triton():
            try:
                return self._triton_forward(hidden_states)
            except Exception:
                logger.warning("Triton SnakeBeta failed, falling back to eager", exc_info=True)
                SnakeBeta._triton_kernel = False
        return self._eager_forward(hidden_states)

    def _eager_forward(self, hidden_states):
        if self._cached:
            exp_alpha = self._exp_alpha.unsqueeze(0).unsqueeze(-1)
            inv_beta = self._inv_beta.unsqueeze(0).unsqueeze(-1)
        else:
            exp_alpha = torch.exp(self.alpha).unsqueeze(0).unsqueeze(-1)
            inv_beta = (1.0 / (torch.exp(self.beta) + self.no_div_by_zero)).unsqueeze(0).unsqueeze(-1)
        hidden_states = hidden_states + inv_beta * torch.pow(torch.sin(hidden_states * exp_alpha), 2)
        return hidden_states

    def _triton_forward(self, x):
        import triton

        if not self._cached:
            self.precompute_exp_cache()

        x = x.contiguous()
        B, C, T = x.shape
        out = torch.empty_like(x)
        block_t = min(triton.next_power_of_2(T), self._TRITON_MAX_BLOCK_T)
        self._triton_kernel[(B, C, triton.cdiv(T, block_t))](
            x,
            self._exp_alpha,
            self._inv_beta,
            out,
            x.stride(0),
            x.stride(1),
            t_len=T,
            block_t=block_t,
        )
        return out
