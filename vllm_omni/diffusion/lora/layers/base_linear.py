# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch
from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA


class DiffusionBaseLinearLayerWithLoRA(BaseLinearLayerWithLoRA):
    """
    Diffusion-specific base that overrides apply() to use direct torch matmul
    instead of punica_wrapper.

    punica_wrapper is used to hold multiple LoRA slots and slices efficiently.

    This matches the semantics of PunicaWrapperGPU.add_lora_linear():
    - Shrink: buffer = (x @ lora_a.T)
    - Expand: y += buffer @ lora_b.T

    Multi-LoRA composition: multiple adapters can be active simultaneously,
    each in its own slot; apply() accumulates deltas from all active slots.

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
        self._n_active_adapters: int = 0
        n_slices = getattr(self, "n_slices", 1)
        # Per-adapter, per-slice active tracking: list of tuples
        self._diffusion_lora_active_slices: list[tuple[bool, ...]] = [
            (False,) * int(n_slices) for _ in range(max_loras)
        ]

    def reset_lora(self, index: int):
        super().reset_lora(index)
        n_slices = getattr(self, "n_slices", 1)
        active_slices = getattr(self, "_diffusion_lora_active_slices", None)
        if active_slices is not None and index < len(active_slices):
            active_slices[index] = (False,) * int(n_slices)

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list[torch.Tensor | None],
        lora_b: torch.Tensor | list[torch.Tensor | None],
    ):
        super().set_lora(index, lora_a, lora_b)  # type: ignore[arg-type]

        n_slices = getattr(self, "n_slices", 1)
        active_slices = getattr(self, "_diffusion_lora_active_slices", None)
        if active_slices is None or index >= len(active_slices):
            return

        if isinstance(lora_a, list) or isinstance(lora_b, list):
            assert isinstance(lora_a, list)
            assert isinstance(lora_b, list)
            slot_active = []
            for a_i, b_i in zip(lora_a[:n_slices], lora_b[:n_slices]):
                slot_active.append(a_i is not None and b_i is not None)
            if len(slot_active) < n_slices:
                slot_active.extend([False] * (n_slices - len(slot_active)))
            active_slices[index] = tuple(slot_active)
        else:
            # Single-slice layer.
            active_slices[index] = (True,)

    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """
        override: Use simple matmul instead of punica_wrapper.add_lora_linear().

        Supports multi-LoRA composition by accumulating deltas from all active
        adapter slots. For packed projections (e.g. fused QKV), LoRA is applied
        per-slice using `output_slices`.
        """
        output = self.base_layer.quant_method.apply(self.base_layer, x, bias)

        if not hasattr(self, "lora_a_stacked") or not hasattr(self, "lora_b_stacked"):
            return output
        if not self.lora_a_stacked or not self.lora_b_stacked:
            return output

        n_active = getattr(self, "_n_active_adapters", 0)
        if n_active == 0:
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

        active_slices_list = getattr(self, "_diffusion_lora_active_slices", None)

        for adapter_idx in range(n_active):
            adapter_active_slices = (
                active_slices_list[adapter_idx]
                if active_slices_list is not None and adapter_idx < len(active_slices_list)
                else None
            )

            offset = 0
            for slice_idx, slice_size in enumerate(output_slices):
                if (
                    adapter_active_slices is not None
                    and slice_idx < len(adapter_active_slices)
                    and not adapter_active_slices[slice_idx]
                ):
                    offset += slice_size
                    continue

                A = self.lora_a_stacked[slice_idx][adapter_idx, 0, :, :]  # (rank, in_dim)
                B = self.lora_b_stacked[slice_idx][adapter_idx, 0, :, :]  # (out_dim, rank)

                if A.numel() == 0 or B.numel() == 0:
                    offset += slice_size
                    continue

                # LoRA shrink & expand:
                #   buffer = (x @ A.T)
                #   y += buffer @ B.T
                delta = (x_flat @ A.t()) @ B.t()
                y_flat[:, offset : offset + slice_size].add_(delta)
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
