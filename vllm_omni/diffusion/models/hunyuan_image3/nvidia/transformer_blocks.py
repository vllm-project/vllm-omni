# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DiT ``ResBlock`` for HunyuanImage3 -- NVIDIA CUDA + Triton implementation.

Two fusions over the default block:

* ``in_layers``: ``GroupNorm -> SiLU`` becomes one :func:`fused_group_norm_silu`
  kernel instead of two ops.
* ``out_layers``: the adaptive GroupNorm ``norm(h) * (1 + scale) + shift``
  becomes one :func:`fused_adaptive_group_norm` kernel instead of three.

Only ``forward`` is overridden -- ``__init__`` is inherited, so the submodule
layout and therefore every state_dict key (``in_layers.0/.2``,
``out_layers.0/.3``, ...) is identical to the default block and checkpoints
load unchanged. Both fused ops fall back to native PyTorch when Triton is
missing, so this class is still correct if it is ever selected on a machine
without a working Triton.
"""

import torch

from vllm_omni.diffusion.models.hunyuan_image3.transformer_blocks import (
    ResBlock as _DefaultResBlock,
)
from vllm_omni.model_executor.models.common.ops import (
    fused_adaptive_group_norm,
    fused_group_norm_silu,
)


class ResBlock(_DefaultResBlock):
    __doc__ = _DefaultResBlock.__doc__

    def forward(self, x, emb) -> torch.Tensor:
        # ``in_layers``/``out_layers`` stay nn.Sequential and are indexed into
        # rather than unpacked into named submodules -- that is what keeps the
        # state_dict keys (``in_layers.0/.2``, ``out_layers.0/.3``) identical to
        # the unfused block, so checkpoints load unchanged.
        in_norm, in_conv = self.in_layers[0], self.in_layers[-1]

        # GroupNorm -> SiLU, one kernel instead of two.
        h = fused_group_norm_silu(x, in_norm.weight, in_norm.bias, num_groups=in_norm.num_groups, eps=in_norm.eps)
        if self.updown:
            h = self.h_upd(h)
            x = self.x_upd(x)
        h = in_conv(h)

        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        # Adaptive Group Normalization: GroupNorm -> mul -> add, one kernel
        # instead of three.
        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = fused_adaptive_group_norm(
            h,
            out_norm.weight,
            out_norm.bias,
            scale,
            shift,
            num_groups=out_norm.num_groups,
            eps=out_norm.eps,
        )
        h = out_rest(h)

        return self.skip_connection(x) + h


__all__ = ["ResBlock"]
