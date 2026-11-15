# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Opt-in TeaCache at Qwen-Image 2.1's decode block boundary.

Prefill must populate every layer's prefix KV. The weightless adapter owns only
approximation state; the transformer still owns all parameters and KV routing.
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any, Protocol

import torch
from torch import nn

from vllm_omni.diffusion.hooks import HookRegistry, ModelHook

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import DiffusionCacheConfig, OmniDiffusionConfig
    from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import QwenImage21Transformer2DModel


class _Pipeline(Protocol):
    transformer: QwenImage21Transformer2DModel
    denoising_transformer: QwenImage21DenoisingAdapter
    od_config: OmniDiffusionConfig


@dataclass(frozen=True)
class QwenImage21SkipPolicy:
    """Per-step skip rules applied on top of the calibrated TeaCache decision."""

    rel_l1_thresh: float
    cfg_rel_l1_thresh: float
    # Exclusive (low, high) bounds on the transformer timestep in [0, 1]. Steps
    # outside it always compute: early and late output changes are largest.
    skip_timestep_range: tuple[float, float] | None = None

    def threshold(self, do_true_cfg: bool) -> float:
        return self.cfg_rel_l1_thresh if do_true_cfg else self.rel_l1_thresh

    def allows_skip(self, timestep: torch.Tensor) -> bool:
        if self.skip_timestep_range is None:
            return True
        low, high = self.skip_timestep_range
        # Reading the scalar synchronizes with the device; only do it when a window is set.
        return low < float(timestep) < high


class QwenImage21DenoisingAdapter(nn.Module):
    """Hook target for the block stack, without registering the weights again."""

    def __init__(self, transformer: QwenImage21Transformer2DModel, policy: QwenImage21SkipPolicy | None = None):
        super().__init__()
        self._transformer_ref = weakref.ref(transformer)
        self.policy = policy

    @property
    def transformer(self) -> QwenImage21Transformer2DModel:
        transformer = self._transformer_ref()
        if transformer is None:
            raise RuntimeError("The Qwen-Image 2.1 transformer has been released")
        return transformer

    def reset_cache(self) -> None:
        registry = getattr(self, "_hook_registry", None)
        if registry is not None:
            registry.reset_hook("teacache")
            registry.reset_hook("teacache_collector")

    def run(
        self,
        hidden_states: torch.Tensor,
        modulation: torch.Tensor,
        *,
        is_decode: bool,
        timestep: torch.Tensor,
        **block_kwargs: Any,
    ) -> torch.Tensor:
        if torch.is_grad_enabled():
            raise ValueError("Qwen-Image 2.1 TeaCache requires inference mode")
        if hidden_states.shape[0] != 1:
            raise ValueError("Qwen-Image 2.1 TeaCache currently requires batch size 1")
        if block_kwargs["cache_branch"] not in {"cond", "uncond"}:
            raise ValueError("Qwen-Image 2.1 TeaCache requires a cond or uncond cache branch")
        if not is_decode:
            # Request-time LoRA can be loaded after startup. Reject it before
            # any prefix KV write, without scanning modules on every decode.
            _validate_no_dynamic_lora(self.transformer)
            # Both CFG prefills precede decode in request mode. Neither may
            # seed or reuse a target-only residual, even at identical shapes.
            self.reset_cache()
            return self.transformer._run_transformer_blocks(hidden_states, modulation, **block_kwargs)
        if self.policy is None:
            return self(hidden_states, modulation, **block_kwargs)
        return self(
            hidden_states,
            modulation,
            skip_allowed=self.policy.allows_skip(timestep),
            # The pipeline sets this flag on the transformer for each request.
            rel_l1_thresh=self.policy.threshold(getattr(self.transformer, "do_true_cfg", False)),
            **block_kwargs,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        modulation: torch.Tensor,
        *,
        skip_allowed: bool = True,
        rel_l1_thresh: float | None = None,
        **block_kwargs: Any,
    ) -> torch.Tensor:
        # Policy arguments only steer the TeaCache hook; dense execution ignores them.
        del skip_allowed, rel_l1_thresh
        return self.transformer._run_transformer_blocks(hidden_states, modulation, **block_kwargs)


def enable_qwen_image21_teacache(pipeline: _Pipeline, config: DiffusionCacheConfig) -> None:
    """Install only after validating the explicitly calibrated configuration."""
    from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
    from vllm_omni.diffusion.cache.teacache.hook import apply_teacache_hook

    coefficients = config.coefficients
    if (
        not isinstance(coefficients, (list, tuple))
        or len(coefficients) != 5
        or any(
            isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) for value in coefficients
        )
    ):
        raise ValueError(
            "Qwen-Image 2.1 TeaCache requires five explicit finite numeric coefficients; calibrate this model"
        )
    threshold = _positive_threshold(config.rel_l1_thresh, "rel_l1_thresh")
    cfg_threshold = getattr(config, "cfg_rel_l1_thresh", None)
    if cfg_threshold is not None:
        cfg_threshold = _positive_threshold(cfg_threshold, "cfg_rel_l1_thresh")
    policy = QwenImage21SkipPolicy(
        rel_l1_thresh=threshold,
        cfg_rel_l1_thresh=threshold if cfg_threshold is None else cfg_threshold,
        skip_timestep_range=_skip_timestep_range(getattr(config, "skip_timestep_range", None)),
    )
    _validate_pipeline(pipeline)

    tea_config = TeaCacheConfig(
        transformer_type="QwenImage21DenoisingAdapter",
        rel_l1_thresh=threshold,
        coefficients=[float(value) for value in coefficients],
    )
    adapter = QwenImage21DenoisingAdapter(pipeline.transformer, policy)
    apply_teacache_hook(adapter, tea_config)
    _attach_adapter(pipeline, adapter)


def _positive_threshold(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be an explicit finite positive number")
    return float(value)


def _skip_timestep_range(value: object) -> tuple[float, float] | None:
    if value is None:
        return None
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, Real) or not math.isfinite(item) for item in value)
        or not 0 <= value[0] < value[1] <= 1
    ):
        raise ValueError("skip_timestep_range must be [low, high] with 0 <= low < high <= 1")
    return float(value[0]), float(value[1])


def _validate_pipeline(pipeline: _Pipeline) -> None:
    from vllm_omni.diffusion.offloader.config import offload_enabled

    if pipeline.od_config.step_execution:
        raise ValueError("Qwen-Image 2.1 TeaCache currently requires request mode")
    parallel = pipeline.transformer.parallel_config
    if parallel is not None and (
        getattr(parallel, "use_hsdp", False)
        or any(
            getattr(parallel, name, 1) not in (None, 1)
            for name in (
                "tensor_parallel_size",
                "sequence_parallel_size",
                "cfg_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
                "world_size",
            )
        )
    ):
        raise ValueError("Qwen-Image 2.1 TeaCache currently requires a single GPU")
    if not pipeline.transformer.causal_condition:
        raise ValueError("Qwen-Image 2.1 TeaCache requires causal_condition=True")
    if pipeline.od_config.quantization_config is not None or pipeline.od_config.lora_path is not None:
        raise ValueError("Qwen-Image 2.1 TeaCache calibration currently requires unquantized weights without LoRA")
    if offload_enabled(pipeline.od_config):
        raise ValueError("Qwen-Image 2.1 TeaCache and calibration currently require offload disabled")
    if pipeline.transformer.prefix_kv_cache_dtype is not None:
        raise ValueError("Qwen-Image 2.1 TeaCache and calibration currently require native-precision prefix KV storage")
    _validate_no_dynamic_lora(pipeline.transformer)


def _validate_no_dynamic_lora(transformer: QwenImage21Transformer2DModel) -> None:
    from vllm_omni.diffusion.models.qwen_image_21.decode_graph import _dynamic_lora_wrappers_present

    if _dynamic_lora_wrappers_present(transformer):
        raise ValueError("Qwen-Image 2.1 TeaCache and calibration currently require dynamic LoRA disabled")


def _attach_adapter(pipeline: _Pipeline, adapter: QwenImage21DenoisingAdapter) -> None:
    # Whole-decode replay bypasses the Python cache decision. Clear captures
    # before disabling it; regional block compilation can still be evaluated.
    manager = pipeline.transformer._decode_graph_manager
    if manager is not None:
        manager.clear()
    pipeline.transformer._decode_graph_manager = None
    pipeline.transformer.enable_cuda_graph_decode = False
    pipeline.transformer._teacache_adapter = adapter
    pipeline.denoising_transformer = adapter


@dataclass(frozen=True)
class QwenImage21Observation:
    """Scalar-only calibration record; no prompt or activation content."""

    branch: str
    input_rel_l1: float
    output_rel_l1: float


class QwenImage21CalibrationHook(ModelHook):
    """Dense collection with O(one step) tensor storage per CFG branch."""

    _HOOK_NAME = "teacache_collector"

    def __init__(self, max_observations: int = 10000):
        if type(max_observations) is not int or not 1 <= max_observations <= 100000:
            raise ValueError("max_observations must be an integer between 1 and 100000")
        self.max_observations = max_observations
        self.observations: list[QwenImage21Observation] = []
        self._previous: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def initialize_hook(self, module: nn.Module) -> nn.Module:
        if module._hook_registry.get_hook("teacache") is not None:
            raise ValueError("Calibration requires dense execution; remove the TeaCache hook first")
        return module

    def new_forward(
        self,
        module: QwenImage21DenoisingAdapter,
        hidden_states: torch.Tensor,
        modulation: torch.Tensor,
        **block_kwargs: Any,
    ) -> torch.Tensor:
        from vllm_omni.diffusion.cache.teacache.extractors import extract_qwen_image21_context

        if len(self.observations) >= self.max_observations:
            self._previous.clear()
            return module.transformer._run_transformer_blocks(hidden_states, modulation, **block_kwargs)
        context = extract_qwen_image21_context(module, hidden_states, modulation, **block_kwargs)
        feature = context.modulated_input.detach()
        output = context.run_transformer_blocks()[0]
        branch = context.extra_states["teacache_branch"]
        if branch in self._previous:
            previous_feature, previous_output = self._previous[branch]
            distances = (
                torch.stack(
                    [
                        (feature.float() - previous_feature.float()).abs().mean()
                        / (previous_feature.float().abs().mean() + 1e-8),
                        (output.float() - previous_output.float()).abs().mean()
                        / (previous_output.float().abs().mean() + 1e-8),
                    ]
                )
                .cpu()
                .tolist()
            )
            if not all(math.isfinite(value) for value in distances):
                self._previous.clear()
                raise ValueError("Non-finite TeaCache calibration distance")
            self.observations.append(QwenImage21Observation(branch, *distances))
        # The feature is a fresh norm*scale allocation. The output needs its
        # own snapshot: downstream layers or callers may reuse it in-place.
        self._previous[branch] = (feature, output.detach().clone())
        return output

    def reset_state(self, module: nn.Module) -> nn.Module:
        # Keep bounded scalar observations across requests, but never pair
        # the last step of one request with the first of the next request.
        self._previous.clear()
        return module


def enable_qwen_image21_calibration(pipeline: _Pipeline, max_observations: int = 10000) -> QwenImage21CalibrationHook:
    """Attach the same decode boundary with a dense collector, without coefficients."""
    _validate_pipeline(pipeline)
    collector = QwenImage21CalibrationHook(max_observations)
    adapter = QwenImage21DenoisingAdapter(pipeline.transformer)
    HookRegistry.get_or_create(adapter).register_hook(collector._HOOK_NAME, collector)
    _attach_adapter(pipeline, adapter)
    return collector


def fit_qwen_image21_coefficients(observations: list[QwenImage21Observation]) -> list[float]:
    """Fit a quartic from dense scalar pairs; the caller must validate on held-out cases."""
    import numpy as np

    if not 6 <= len(observations) <= 100000:
        raise ValueError("Fit requires between 6 and 100000 observations")
    if any(
        item.branch not in {"positive", "negative"}
        or any(
            isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0
            for value in (item.input_rel_l1, item.output_rel_l1)
        )
        for item in observations
    ):
        raise ValueError("Fit requires finite nonnegative distances and explicit CFG branches")
    x = np.array([item.input_rel_l1 for item in observations], dtype=np.float64)
    y = np.array([item.output_rel_l1 for item in observations], dtype=np.float64)
    if len(np.unique(x)) < 5:
        raise ValueError("Quartic fit requires at least five distinct input distances")
    coefficients, _, rank, _, _ = np.polyfit(x, y, 4, full=True)
    if rank != 5 or not np.isfinite(coefficients).all():
        raise ValueError("Calibration fit is non-finite or ill-conditioned")
    return coefficients.tolist()
