# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project and the xDiT authors.
#
# This module is adapted from xDiT (https://github.com/xdit-project/xdit)

from functools import wraps

import torch
from torch.nn import functional as F
from vllm.model_executor.layers.conv import Conv3dLayer as Conv3dLayerVLLM

from vllm_omni.diffusion.distributed.pipefusion.pipefusion_runtime import (
    get_pipefusion_runtime,
    is_pipefusion_initialized,
)


class PipeFusionConvMixin:
    """
    Mixin for Conv layers in PipeFusion.

    In patch mode, maintains an activation cache so that convolution
    at patch boundaries can access neighboring patches' activations.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        init = getattr(cls, "__init__")

        @wraps(init)
        def wrapped_init(self, *args, **kwargs):
            init(self, *args, **kwargs)
            if is_pipefusion_initialized():
                self.forward = self.pipefusion_forward
            else:
                self.forward = self.orig_forward

        cls.__init__ = wrapped_init

    def pipefusion_conv3d_enabled(self) -> bool:
        """Whether this conv layer should use PipeFusion patch-wise execution.

        Only needed when kernel != stride (overlapping convolutions that
        require boundary data from neighbouring patches).  When kernel == stride
        (e.g. the patch embedding), each output position depends on exactly one
        non-overlapping input block, so the direct conv on the patch is correct.
        """
        runtime = get_pipefusion_runtime()
        return runtime.patch_mode and runtime.num_pipeline_patch > 1 and self.kernel_size != self.stride

    def pipefusion_reset_cache(self) -> None:
        self.activation_cache = None

    def pipefusion_conv3d_forward(self, x: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
        """
        Forward pass for Conv3d with PipeFusion patch support.

        In patch mode, caches activations and performs sliced convolution
        to handle boundary conditions correctly. Call only when
        `pipefusion_conv3d_enabled()` returns True.

        Args:
            x: Input tensor for the current patch.
            dims: Original full input dimensions.

        Returns:
            Convolution output for the current patch.
        """
        runtime = get_pipefusion_runtime()
        if getattr(self, "activation_cache", None) is None:
            self.activation_cache = torch.zeros(dims, dtype=x.dtype, device=x.device)

        patch_idx = runtime.pipeline_patch_idx
        start, end = runtime.pp_patches_start_end_idx[patch_idx]
        out_start, out_end = runtime.pp_patches_post_start_end_idx[patch_idx]
        self.activation_cache[:, :, :, start:end, :] = x
        return self._sliced_conv3d_forward(self.activation_cache, out_start, out_end)

    def _sliced_conv3d_forward(self, x: torch.Tensor, out_start: int, out_end: int) -> torch.Tensor:
        """
        Compute convolution on a slice of the input that produces output for [out_start:out_end].

        Args:
            x: Full input tensor with all patches cached.
            out_start, out_end: Output slice range (post-patch space).
        """
        b, c, t, h, w = x.shape
        pad_t, pad_h, pad_w = self.padding
        stride_h = self.stride[1] if isinstance(self.stride, tuple) else self.stride

        # Calculate input range needed to produce output [out_start:out_end]
        # For strided conv: out_pos = (in_pos + pad - kernel_size) // stride + 1
        # Inverse: in_pos = out_pos * stride - pad (approximately)
        in_start = out_start * stride_h
        in_end = (out_end - 1) * stride_h + self.kernel_size[1]  # Need full kernel for last output

        # Expand to include padding context from neighbors
        h_begin = max(0, in_start - pad_h)
        h_end = min(h, in_end + pad_h)

        # Determine padding needed at boundaries
        pad_top = max(0, pad_h - in_start) if h_begin == 0 else 0
        pad_bottom = max(0, in_end + pad_h - h) if h_end == h else 0

        sliced_input = x[:, :, :, h_begin:h_end, :]
        padded_input = F.pad(sliced_input, (pad_w, pad_w, pad_top, pad_bottom, pad_t, pad_t), mode="constant")

        output = F.conv3d(
            padded_input,
            self.weight,
            self.bias,
            stride=self.stride,
            padding="valid",
            dilation=self.dilation,
            groups=self.groups,
        )

        # Extract only the output rows we need (in case we computed extra)
        expected_out_height = out_end - out_start
        if output.shape[3] > expected_out_height:
            output = output[:, :, :, :expected_out_height, :]
        return output


class Conv3dLayer(Conv3dLayerVLLM, PipeFusionConvMixin):
    def pipefusion_forward(self, x: torch.Tensor, dims=None) -> torch.Tensor:
        if self.pipefusion_conv3d_enabled():
            return self.pipefusion_conv3d_forward(x, dims)
        return super().forward(x)

    def orig_forward(self, x: torch.Tensor, dims=None) -> torch.Tensor:
        return super().forward(x)
