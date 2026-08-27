# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
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

    def pipefusion_reset_cache(self, request_id: str | None = None, sequence_id: int | None = None) -> None:
        # Signature matches PipeFusionSelfAttentionMixin so
        # PipeFusionPipelineMixin._reset_pipefusion_caches can pass request
        # scope. Conv activations are not request-keyed.
        del request_id, sequence_id
        self.activation_cache = None

    @staticmethod
    def _as_3tuple(value) -> tuple[int, int, int]:
        if isinstance(value, tuple):
            return value
        return (value, value, value)

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
        if runtime.split_dim == "temporal":
            self.activation_cache[:, :, start:end, :, :] = x
        elif runtime.split_dim == "height":
            self.activation_cache[:, :, :, start:end, :] = x
        else:
            raise ValueError(
                f"Unsupported PipeFusion Conv3D split_dim={runtime.split_dim!r}; expected 'height' or 'temporal'."
            )
        return self._sliced_conv3d_forward(self.activation_cache, out_start, out_end, runtime.split_dim)

    def _sliced_conv3d_forward(
        self,
        x: torch.Tensor,
        out_start: int,
        out_end: int,
        split_dim: str,
    ) -> torch.Tensor:
        """
        Compute convolution on a slice of the input that produces output for [out_start:out_end].

        Args:
            x: Full input tensor with all patches cached.
            out_start, out_end: Output slice range (post-patch space).
            split_dim: PipeFusion patch split dimension.
        """
        _, _, t, h, _ = x.shape
        pad_t, pad_h, pad_w = self._as_3tuple(self.padding)
        stride_t, stride_h, _ = self._as_3tuple(self.stride)
        kernel_t, kernel_h, _ = self._as_3tuple(self.kernel_size)
        dilation_t, dilation_h, _ = self._as_3tuple(self.dilation)

        if split_dim == "temporal":
            split_axis = 2
            full_extent = t
            pad_along_split = pad_t
            stride_along_split = stride_t
            kernel_along_split = kernel_t
            dilation_along_split = dilation_t
        elif split_dim == "height":
            split_axis = 3
            full_extent = h
            pad_along_split = pad_h
            stride_along_split = stride_h
            kernel_along_split = kernel_h
            dilation_along_split = dilation_h
        else:
            raise ValueError(f"Unsupported PipeFusion Conv3D split_dim={split_dim!r}.")

        # Calculate input range needed to produce output [out_start:out_end]
        # For strided conv: out_pos = (in_pos + pad - kernel_size) // stride + 1
        # Inverse: in_pos = out_pos * stride - pad (approximately)
        effective_kernel = dilation_along_split * (kernel_along_split - 1) + 1
        in_start = out_start * stride_along_split
        in_end = (out_end - 1) * stride_along_split + effective_kernel  # Need full kernel for last output

        # Expand to include padding context from neighbors
        slice_begin = max(0, in_start - pad_along_split)
        slice_end = min(full_extent, in_end + pad_along_split)

        # Determine padding needed at boundaries
        pad_before = max(0, pad_along_split - in_start) if slice_begin == 0 else 0
        pad_after = max(0, in_end + pad_along_split - full_extent) if slice_end == full_extent else 0

        if split_dim == "temporal":
            sliced_input = x[:, :, slice_begin:slice_end, :, :]
            padding = (pad_w, pad_w, pad_h, pad_h, pad_before, pad_after)
        else:
            sliced_input = x[:, :, :, slice_begin:slice_end, :]
            padding = (pad_w, pad_w, pad_before, pad_after, pad_t, pad_t)
        padded_input = F.pad(sliced_input, padding, mode="constant")

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
        expected_out_extent = out_end - out_start
        if output.shape[split_axis] > expected_out_extent:
            output = output.narrow(split_axis, 0, expected_out_extent)
        return output


class Conv3dLayer(Conv3dLayerVLLM, PipeFusionConvMixin):
    def pipefusion_forward(self, x: torch.Tensor, dims=None) -> torch.Tensor:
        if self.pipefusion_conv3d_enabled():
            return self.pipefusion_conv3d_forward(x, dims)
        return super().forward(x)

    def orig_forward(self, x: torch.Tensor, dims=None) -> torch.Tensor:
        return super().forward(x)
