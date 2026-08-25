# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Format-agnostic schedule state, linear discovery, and dispatch."""

from __future__ import annotations

from typing import Literal

import torch
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase

from .config import Cosmos3MixedPrecisionConfig
from .strategy import (
    Cosmos3PrecisionStrategy,
    Fp8W8A8W8A16Strategy,
    Nvfp4W4A4W4A16Strategy,
)

PrecisionPath = Literal["reasoner", "generation"]
_STRATEGIES: tuple[Cosmos3PrecisionStrategy, ...] = (
    Fp8W8A8W8A16Strategy(),
    Nvfp4W4A4W4A16Strategy(),
)


def _strategy_for(method: object | None) -> Cosmos3PrecisionStrategy | None:
    return next((strategy for strategy in _STRATEGIES if strategy.accepts(method)), None)


class Cosmos3MixedPrecisionLinearMethod(LinearMethodBase):
    """Dispatch between a checkpoint-native method and dense A16."""

    def __init__(
        self,
        base_method: LinearMethodBase,
        strategy: Cosmos3PrecisionStrategy,
        runtime: Cosmos3MixedPrecisionRuntime,
        module_name: str,
        path: PrecisionPath,
    ) -> None:
        self.base_method = base_method
        self.strategy = strategy
        self.runtime = runtime
        self.module_name = module_name
        self.path = path

    def create_weights(self, *args, **kwargs) -> None:
        self.base_method.create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.strategy.validate_before_processing(
            self.base_method,
            layer,
            self.module_name,
        )
        self.base_method.process_weights_after_loading(layer)
        self.strategy.validate_after_processing(layer, self.module_name)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.runtime.use_high_precision(self.path):
            return self.base_method.apply(layer, x, bias)
        return self.strategy.apply_high(layer, x, bias)


class Cosmos3MixedPrecisionRuntime:
    """Own one transformer's schedule state and wrapped linear inventory."""

    def __init__(self, config: Cosmos3MixedPrecisionConfig) -> None:
        self.config = config
        # TODO: Make this request-local before Cosmos3 supports interleaved
        # denoising requests on one transformer instance.
        self._generation_high_precision = False

    def install(self, transformer: torch.nn.Module) -> None:
        components: dict[PrecisionPath, torch.nn.Module] = {
            "generation": transformer.gen_layers,
        }
        if self.config.reasoner == "a16":
            components["reasoner"] = transformer.language_model.layers

        wrapped_count = 0
        for path, component in components.items():
            for local_name, layer in component.named_modules():
                if not isinstance(layer, LinearBase):
                    continue
                base_method = getattr(layer, "quant_method", None)
                strategy = _strategy_for(base_method)
                if strategy is None:
                    continue
                module_name = getattr(layer, "prefix", None) or f"{path}.{local_name}"
                layer.quant_method = Cosmos3MixedPrecisionLinearMethod(
                    base_method,
                    strategy,
                    self,
                    module_name,
                    path,
                )
                wrapped_count += 1

        if not wrapped_count:
            raise ValueError("Cosmos3 mixed precision found no compatible FP8 or NVFP4 ModelOpt linears")

    def use_high_precision(self, path: PrecisionPath) -> bool:
        return self.config.reasoner == "a16" if path == "reasoner" else self._generation_high_precision

    def set_step(self, step_index: int, num_steps: int) -> None:
        self._generation_high_precision = self.config.use_high_precision(step_index, num_steps)

    def reset(self) -> None:
        self._generation_high_precision = False
