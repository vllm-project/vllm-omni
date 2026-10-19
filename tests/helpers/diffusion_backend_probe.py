# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Worker-local observations for diffusion backend runtime acceptance tests."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, TypedDict, cast

import pytest
import torch
from torch import nn
from torch.utils.hooks import RemovableHandle
from vllm.model_executor.layers import linear
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker as _WorkerBase
else:
    # WorkerWrapperBase supplies DiffusionWorker after this mixin in the MRO.
    _WorkerBase = object


class BackendRuntimeProbeSnapshot(TypedDict):
    """JSON/RPC-safe construction and per-request execution observations."""

    pid: int
    rank: int
    device: str
    pipeline_type: str
    linear_backend: str
    diffusion_attention_backend: str | None
    executor_backend: str | None
    diffusion_load_format: str
    enforce_eager: bool
    dtype: str
    selector_inputs: list[str]
    linear_impls: dict[str, str]
    attention_impls: dict[str, str]
    linear_calls: dict[str, int]
    attention_calls: dict[str, int]
    transformer_calls: int
    transformer_output_shapes: list[list[int]]
    transformer_outputs_finite: list[bool]


LinearImplementation = Callable[[nn.Module, torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]
AttentionImplementation = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, AttentionMetadata | None], torch.Tensor]


class DiffusionBackendProbe(_WorkerBase):
    """Worker extension that observes, but does not replace, real computation.

    The selector wrapper delegates constructor-time dispatch. Per-layer wrappers
    delegate to the selected implementations, and a transformer hook checks the
    floating-point output before the pipeline converts it to an image.
    """

    def load_model(
        self, load_format: str = "default", custom_pipeline_name: str | None = None, **kwargs: object
    ) -> None:
        self._probe_selector_inputs: list[str] = []
        self._probe_linear_originals: list[tuple[UnquantizedLinearMethod, LinearImplementation]] = []
        self._probe_attention_originals: list[tuple[Attention, AttentionImplementation]] = []
        self._probe_transformer_hook: RemovableHandle | None = None
        self._probe_linear_impls: dict[str, str] = {}
        self._probe_attention_impls: dict[str, str] = {}
        self._probe_linear_calls: dict[str, int] = {}
        self._probe_attention_calls: dict[str, int] = {}
        self._probe_transformer_calls = 0
        self._probe_transformer_output_shapes: list[list[int]] = []
        self._probe_transformer_outputs_finite: list[bool] = []

        original_selector = linear.dispatch_unquantized_gemm

        def observed_selector(linear_backend: str = "auto") -> LinearImplementation:
            self._probe_selector_inputs.append(linear_backend)
            return original_selector(linear_backend)

        with pytest.MonkeyPatch.context() as patcher:
            patcher.setattr(linear, "dispatch_unquantized_gemm", observed_selector)
            super().load_model(
                load_format=load_format,
                custom_pipeline_name=custom_pipeline_name,
                **kwargs,
            )

        self._install_backend_runtime_probe()

    def _install_backend_runtime_probe(self) -> None:
        assert self.model_runner is not None
        pipeline = self.model_runner.pipeline
        assert pipeline is not None
        transformer = pipeline.transformer
        assert isinstance(transformer, nn.Module)
        pipeline_type = type(pipeline)
        self._probe_pipeline_type = f"{pipeline_type.__module__}.{pipeline_type.__qualname__}"

        for name, module in transformer.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if isinstance(quant_method, UnquantizedLinearMethod):
                original = cast(LinearImplementation, quant_method._gemm_impl)
                self._probe_linear_impls[name] = f"{original.__module__}.{original.__qualname__}"
                self._probe_linear_calls[name] = 0
                self._probe_linear_originals.append((quant_method, original))
                quant_method._gemm_impl = self._count_linear(name, original)

            if isinstance(module, Attention):
                original_attention = cast(AttentionImplementation, module.attention.forward)
                attention_type = type(module.attention)
                self._probe_attention_impls[name] = f"{attention_type.__module__}.{attention_type.__qualname__}"
                self._probe_attention_calls[name] = 0
                self._probe_attention_originals.append((module, original_attention))
                module.attention.forward = self._count_attention(name, original_attention)

        assert self._probe_linear_originals, "No unquantized transformer linear layers were observed"
        assert self._probe_attention_originals, "No transformer attention layers were observed"
        self._probe_transformer_hook = transformer.register_forward_hook(self._record_transformer_output)

    def _count_linear(self, name: str, original: LinearImplementation) -> LinearImplementation:
        def counted_gemm(
            layer: nn.Module,
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            self._probe_linear_calls[name] += 1
            return original(layer, x, weight, bias)

        return counted_gemm

    def _count_attention(self, name: str, original: AttentionImplementation) -> AttentionImplementation:
        def counted_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_metadata: AttentionMetadata | None = None,
        ) -> torch.Tensor:
            self._probe_attention_calls[name] += 1
            return original(query, key, value, attn_metadata)

        return counted_attention

    def _record_transformer_output(self, _module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
        # Native FLUX invokes its transformer with return_dict=False.
        assert isinstance(output, tuple) and len(output) == 1
        tensor = output[0]
        assert isinstance(tensor, torch.Tensor) and tensor.is_floating_point()
        self._probe_transformer_calls += 1
        self._probe_transformer_output_shapes.append(list(tensor.shape))
        self._probe_transformer_outputs_finite.append(bool(torch.isfinite(tensor).all().item()))

    def reset_backend_runtime_probe(self) -> bool:
        """Clear request counters while preserving constructor-time evidence."""
        for name in self._probe_linear_calls:
            self._probe_linear_calls[name] = 0
        for name in self._probe_attention_calls:
            self._probe_attention_calls[name] = 0
        self._probe_transformer_calls = 0
        self._probe_transformer_output_shapes.clear()
        self._probe_transformer_outputs_finite.clear()
        return True

    def get_backend_runtime_probe(self) -> BackendRuntimeProbeSnapshot:
        """Return observations without tensors, modules, or bound callables."""
        assert self.vllm_config is not None
        attention = self.od_config.diffusion_attention_config.default
        return {
            "pid": os.getpid(),
            "rank": self.rank,
            "device": str(self.device),
            "pipeline_type": self._probe_pipeline_type,
            "linear_backend": self.vllm_config.kernel_config.linear_backend,
            "diffusion_attention_backend": None if attention is None else attention.backend,
            "executor_backend": self.od_config.distributed_executor_backend,
            "diffusion_load_format": self.od_config.diffusion_load_format,
            "enforce_eager": self.od_config.enforce_eager,
            "dtype": str(self.od_config.dtype),
            "selector_inputs": list(self._probe_selector_inputs),
            "linear_impls": dict(self._probe_linear_impls),
            "attention_impls": dict(self._probe_attention_impls),
            "linear_calls": dict(self._probe_linear_calls),
            "attention_calls": dict(self._probe_attention_calls),
            "transformer_calls": self._probe_transformer_calls,
            "transformer_output_shapes": [list(shape) for shape in self._probe_transformer_output_shapes],
            "transformer_outputs_finite": list(self._probe_transformer_outputs_finite),
        }

    def stop_backend_runtime_probe(self) -> bool:
        """Restore callables and release every probe-owned model reference."""
        for quant_method, original in self._probe_linear_originals:
            quant_method._gemm_impl = original
        for module, original_attention in self._probe_attention_originals:
            module.attention.forward = original_attention
        self._probe_linear_originals.clear()
        self._probe_attention_originals.clear()
        if self._probe_transformer_hook is not None:
            self._probe_transformer_hook.remove()
            self._probe_transformer_hook = None
        return True

    def shutdown(self) -> None:
        # Construction can fail before load_model initializes the probe.
        if hasattr(self, "_probe_linear_originals"):
            self.stop_backend_runtime_probe()
        super().shutdown()
