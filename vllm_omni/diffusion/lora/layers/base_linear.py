# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA


def get_global_output_sizes(lora_layer: object) -> tuple[int, ...]:
    """Return checkpoint-visible output sizes for a possibly TP-sharded layer."""

    n_slices = int(getattr(lora_layer, "n_slices", 1))
    base_layer = getattr(lora_layer, "base_layer", None)
    output_sizes = getattr(lora_layer, "output_sizes", None)
    if output_sizes is None:
        output_sizes = getattr(base_layer, "output_sizes", None)
    if n_slices > 1 and output_sizes is not None and len(output_sizes) == n_slices:
        return tuple(int(size) for size in output_sizes)

    output_size = getattr(base_layer, "output_size", None)
    if output_size is None:
        output_size = getattr(lora_layer, "output_size", None)
    if output_size is None and output_sizes is not None:
        output_size = sum(output_sizes)
    if n_slices == 1 and output_size is not None:
        return (int(output_size),)

    output_slices = tuple(int(size) for size in getattr(lora_layer, "output_slices", ()))
    if len(output_slices) != n_slices:
        raise RuntimeError(f"LoRA output slice metadata mismatch: got {len(output_slices)}, expected {n_slices}")
    return output_slices


class DiffusionBaseLinearLayerWithLoRA(BaseLinearLayerWithLoRA):
    """
    Diffusion-specific base that overrides apply() to use direct torch matmul
    instead of punica_wrapper.

    punica_wrapper is used to hold multiple LoRA slots and slices efficiently.

    This matches the semantics of PunicaWrapperGPU.add_lora_linear():
    - Shrink: buffer = (x @ lora_a.T)
    - Expand: y += buffer @ lora_b.T

    All other functionality (weight management, TP slicing, forward logic)
    is inherited from vLLM's BaseLinearLayerWithLoRA.
    """

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config,
        model_config=None,
    ) -> None:
        super().create_lora_weights(max_loras, lora_config, model_config)
        # Keep a direct reference for attribute forwarding: `base_layer` is a
        # registered submodule (stored under `_modules`), so direct access via
        # `object.__getattribute__` will not find it. We stash a ref in
        # `__dict__` for robust lookups in `__getattr__`.
        modules = object.__getattribute__(self, "_modules")
        base_layer = modules.get("base_layer") or object.__getattribute__(self, "__dict__").get("base_layer")
        object.__setattr__(self, "_diffusion_base_layer_ref", base_layer)
        n_slices = getattr(self, "n_slices", 1)
        self._diffusion_lora_active_slices = (False,) * int(n_slices)
        self._diffusion_additive_bias = (None,) * int(n_slices)

    def reset_lora(self, index: int):
        super().reset_lora(index)
        n_slices = getattr(self, "n_slices", 1)
        self._diffusion_lora_active_slices = (False,) * int(n_slices)
        self._diffusion_additive_bias = (None,) * int(n_slices)

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list[torch.Tensor | None],
        lora_b: torch.Tensor | list[torch.Tensor | None],
    ):
        super().set_lora(index, lora_a, lora_b)  # type: ignore[arg-type]

        n_slices = getattr(self, "n_slices", 1)
        if isinstance(lora_a, list) or isinstance(lora_b, list):
            assert isinstance(lora_a, list)
            assert isinstance(lora_b, list)
            active_slices = []
            for a_i, b_i in zip(lora_a[:n_slices], lora_b[:n_slices]):
                active_slices.append(a_i is not None and b_i is not None)
            if len(active_slices) < n_slices:
                active_slices.extend([False] * (n_slices - len(active_slices)))
            self._diffusion_lora_active_slices = tuple(active_slices)
        else:
            # Single-slice layer.
            self._diffusion_lora_active_slices = (True,)

    @staticmethod
    def _validate_bias_shapes(
        bias_slices: list[torch.Tensor | None],
        output_sizes: tuple[int, ...],
    ) -> None:
        for slice_idx, (bias, output_size) in enumerate(zip(bias_slices, output_sizes, strict=True)):
            if bias is not None and tuple(bias.shape) != (output_size,):
                raise ValueError(
                    f"Additive bias shape mismatch for slice {slice_idx}: "
                    f"got {tuple(bias.shape)}, expected {(output_size,)}"
                )

    def set_additive_bias(
        self,
        bias: torch.Tensor | list[torch.Tensor | None] | None,
    ) -> None:
        n_slices = int(getattr(self, "n_slices", 1))
        if bias is None:
            self._diffusion_additive_bias = (None,) * n_slices
            return

        bias_slices = bias if isinstance(bias, list) else [bias]
        if len(bias_slices) != n_slices:
            raise ValueError(f"Additive bias slice mismatch: got {len(bias_slices)}, expected {n_slices}")
        self._validate_bias_shapes(bias_slices, get_global_output_sizes(self))
        if self.tp_size > 1:
            shaped_bias = [bias.unsqueeze(1) if bias is not None else None for bias in bias_slices]
            if n_slices == 1:
                assert shaped_bias[0] is not None
                sliced_bias = [self.slice_lora_b(shaped_bias[0])]
            else:
                sliced_bias = self.slice_lora_b(shaped_bias)
            bias_slices = [bias.squeeze(1) if bias is not None else None for bias in sliced_bias]
        self._validate_bias_shapes(bias_slices, tuple(int(size) for size in self.output_slices))

        device = self.lora_b_stacked[0].device
        dtype = self.lora_b_stacked[0].dtype
        self._diffusion_additive_bias = tuple(
            bias.to(device=device, dtype=dtype, non_blocking=True) if bias is not None else None for bias in bias_slices
        )

    def move_lora_runtime_to(self, device: torch.device) -> None:
        """Keep request-dependent LoRA state resident on its execution device."""

        self.lora_a_stacked = tuple(tensor.to(device=device) for tensor in self.lora_a_stacked)
        self.lora_b_stacked = tuple(tensor.to(device=device) for tensor in self.lora_b_stacked)
        self._diffusion_additive_bias = tuple(
            bias.to(device=device) if bias is not None else None
            for bias in getattr(self, "_diffusion_additive_bias", ())
        )
        self.device = device

    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """
        override: Use simple matmul instead of punica_wrapper.add_lora_linear().

        This matches the exact computation in PunicaWrapperGPU.add_lora_linear()
        for the single-LoRA case. For packed projections (e.g. fused QKV), we
        apply LoRA per-slice using `output_slices`.
        """
        quant_method = getattr(self.base_layer, "quant_method", None)
        if quant_method is None:
            output = F.linear(x, self.base_layer.weight, bias)
        else:
            output = quant_method.apply(self.base_layer, x, bias)

        additive_bias = getattr(self, "_diffusion_additive_bias", ())
        if any(bias is not None for bias in additive_bias):
            output_offset = 0
            for slice_size, bias in zip(self.output_slices, additive_bias, strict=True):
                if bias is not None:
                    output[..., output_offset : output_offset + slice_size] += bias.to(
                        device=output.device, non_blocking=True
                    )
                output_offset += slice_size

        if not hasattr(self, "lora_a_stacked") or not hasattr(self, "lora_b_stacked"):
            return output
        if not self.lora_a_stacked or not self.lora_b_stacked:
            return output
        # Fast path: if no LoRA is active for this layer, skip matmuls.
        active_slices = getattr(self, "_diffusion_lora_active_slices", None)
        if active_slices is not None and not any(active_slices):
            return output

        # In fully-sharded LoRA mode, vLLM uses an all-gather between shrink and
        # expand for ColumnParallelLinear variants. This diffusion path doesn't
        # implement that communication yet.
        if getattr(self, "lora_config", None) is not None:
            if self.lora_config.fully_sharded_loras and self.tp_size > 1:
                raise NotImplementedError(
                    "Diffusion LoRA apply() does not support fully_sharded_loras with tensor parallelism yet."
                )

        original_shape = output.shape
        x_flat = x.reshape(-1, x.shape[-1])
        y_flat = output.reshape(-1, output.shape[-1])

        output_slices = getattr(self, "output_slices", None)
        if output_slices is None:
            # Fallback: infer slice sizes from the allocated tensors.
            output_slices = tuple(lora_b.shape[2] for lora_b in self.lora_b_stacked)

        if len(output_slices) != len(self.lora_a_stacked) or len(output_slices) != len(self.lora_b_stacked):
            raise RuntimeError(
                "LoRA slice metadata mismatch: "
                f"output_slices={len(output_slices)}, "
                f"lora_a_stacked={len(self.lora_a_stacked)}, "
                f"lora_b_stacked={len(self.lora_b_stacked)}"
            )

        offset = 0
        for slice_idx, slice_size in enumerate(output_slices):
            if active_slices is not None and slice_idx < len(active_slices) and not active_slices[slice_idx]:
                offset += slice_size
                continue

            A = self.lora_a_stacked[slice_idx][0, 0, :, :].to(device=x_flat.device, non_blocking=True)  # (rank, in_dim)
            B = self.lora_b_stacked[slice_idx][0, 0, :, :].to(
                device=x_flat.device, non_blocking=True
            )  # (out_dim, rank)

            if A.numel() == 0 or B.numel() == 0:
                offset += slice_size
                continue

            # LoRA shrink & expand as in add_lora_linear():
            #   buffer = (x @ A.T)
            #   y += buffer @ B.T
            delta = (x_flat @ A.t()) @ B.t()
            y_flat[:, offset : offset + slice_size] = y_flat[:, offset : offset + slice_size] + delta
            offset += slice_size

        return y_flat.view(original_shape)

    def __getattr__(self, name: str):
        # The diffusion model implementations may access attributes directly
        # from linear layers (e.g. QKVParallelLinear.num_heads). vLLM's LoRA
        # wrappers don't forward these attributes by default, so we delegate
        # missing attribute lookups to the underlying base_layer.
        try:
            return super().__getattr__(name)
        except AttributeError as exc:
            base_layer = object.__getattribute__(self, "__dict__").get("_diffusion_base_layer_ref")
            if base_layer is None:
                base_layer = object.__getattribute__(self, "_modules").get("base_layer")
            if base_layer is None:
                raise exc
            try:
                return getattr(base_layer, name)
            except AttributeError:
                raise exc
