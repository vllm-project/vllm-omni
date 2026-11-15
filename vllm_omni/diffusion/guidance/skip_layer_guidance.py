# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Skip Layer Guidance shared by DiT pipelines.

Unconditional skipping replaces CFG's negative branch. SLG adds
``scale * (positive - skipped_positive)`` to CFG, or uses the skipped positive
as its base when ordinary CFG is disabled.
Optional magnitude rescaling is performed in denoised/x0 space, as in
ComfyUI's SkipLayerGuidanceDiT. Statistics are computed independently for
each output image so batching does not couple unrelated requests.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Protocol

import torch

from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.models.interface import SupportsSkipLayerGuidance


def _parse_layers(value: Any, name: str = "slg_layers") -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        if not value.strip():
            return ()
        try:
            value = [int(part.strip()) for part in value.split(",")]
        except ValueError as exc:
            raise ValueError(f"{name} must contain comma-separated nonnegative integers.") from exc
    if not isinstance(value, (list, tuple)) or any(
        isinstance(layer, bool) or not isinstance(layer, Integral) or layer < 0 for layer in value
    ):
        raise ValueError(f"{name} must be a list of nonnegative integers or a comma-separated string.")
    return tuple(sorted(set(int(layer) for layer in value)))


def _finite_number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    return result


def _aliased_value(
    extra: Mapping[str, Any], name: str, alias: str, default: Any, normalize: Callable[[Any], Any]
) -> Any:
    value = normalize(extra.get(name, extra.get(alias, default)))
    if name in extra and alias in extra and value != normalize(extra[alias]):
        raise ValueError(f"Conflicting {name} and {alias} values.")
    return value


@dataclass(frozen=True)
class SkipLayerGuidanceConfig:
    skip_layers: tuple[int, ...] = ()
    scale: float = 0.0
    start: float = 0.0
    stop: float = 1.0
    rescaling_scale: float = 0.0
    uncond_skip_layers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "skip_layers", _parse_layers(self.skip_layers))
        object.__setattr__(self, "uncond_skip_layers", _parse_layers(self.uncond_skip_layers, "uncond_skip_layers"))
        for name in ("scale", "start", "stop", "rescaling_scale"):
            object.__setattr__(self, name, _finite_number(getattr(self, name), f"slg_{name}"))
        if self.scale < 0 or self.rescaling_scale < 0:
            raise ValueError("slg_scale and slg_rescaling_scale must be nonnegative.")
        if not 0 <= self.start <= self.stop <= 1:
            raise ValueError("SLG requires 0 <= slg_start <= slg_stop <= 1.")

    @classmethod
    def from_extra_args(cls, extra_args: Mapping[str, Any] | None) -> SkipLayerGuidanceConfig:
        extra = extra_args or {}
        return cls(
            skip_layers=_aliased_value(extra, "slg_layers", "slg_skip_layers", (), _parse_layers),
            scale=extra.get("slg_scale", 0.0),
            start=_aliased_value(
                extra, "slg_start", "slg_start_percent", 0.0, lambda value: _finite_number(value, "slg_start")
            ),
            stop=_aliased_value(
                extra, "slg_stop", "slg_end_percent", 1.0, lambda value: _finite_number(value, "slg_stop")
            ),
            rescaling_scale=extra.get("slg_rescaling_scale", 0.0),
            uncond_skip_layers=extra.get("uncond_skip_layers", ()),
        )

    @property
    def cond_enabled(self) -> bool:
        return self.scale > 0 and bool(self.skip_layers)

    @property
    def enabled(self) -> bool:
        return self.cond_enabled or bool(self.uncond_skip_layers)

    @property
    def batch_key(self) -> tuple[Any, ...] | None:
        if not self.enabled:
            return None
        return (
            self.skip_layers if self.cond_enabled else (),
            self.scale if self.cond_enabled else 0.0,
            self.uncond_skip_layers,
            self.start,
            self.stop,
            self.rescaling_scale if self.cond_enabled else 0.0,
        )


@dataclass(frozen=True)
class SkipLayerGuidanceState:
    """Request-local schedule; host-side flags avoid a GPU sync each step."""

    config: SkipLayerGuidanceConfig
    sigmas: tuple[float, ...]
    active_steps: tuple[bool, ...]


def prepare_skip_layer_guidance(
    config: SkipLayerGuidanceConfig,
    *,
    transformer: SupportsSkipLayerGuidance,
    sigmas: torch.Tensor,
) -> SkipLayerGuidanceState | None:
    """Prepare step-fraction windows using the actual evaluation step count."""
    if not config.skip_layers and not config.uncond_skip_layers:
        return None
    if not isinstance(transformer, SupportsSkipLayerGuidance) or not transformer.supports_skip_layer_guidance:
        raise ValueError("This transformer does not support skip layer guidance.")
    transformer.validate_skip_layers(config.skip_layers)
    transformer.validate_skip_layers(config.uncond_skip_layers)
    if not config.enabled:
        return None
    values = tuple(float(value) for value in sigmas.detach().cpu().reshape(-1).tolist())
    if any(not math.isfinite(sigma) or sigma < 0 for sigma in values):
        raise ValueError("SLG evaluation sigmas must be finite and nonnegative.")
    return SkipLayerGuidanceState(
        config=config,
        sigmas=values,
        active_steps=tuple(config.start <= index / len(values) < config.stop for index in range(len(values))),
    )


class DenoisedPredictionAdapter(Protocol):
    """Model-specific affine conversion between a prediction and denoised x0."""

    def to_denoised(self, prediction: torch.Tensor, latents: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor: ...

    def from_denoised(self, denoised: torch.Tensor, latents: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor: ...


class FlowMatchingPredictionAdapter:
    """Flow velocity uses x0 = x_t - sigma * velocity."""

    def to_denoised(self, prediction: torch.Tensor, latents: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return latents - sigma * prediction

    def from_denoised(self, denoised: torch.Tensor, latents: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return (latents - denoised) / sigma


def combine_skip_layer_guidance(
    predictions: list[torch.Tensor],
    *,
    do_true_cfg: bool,
    cfg_scale: float,
    config: SkipLayerGuidanceConfig,
    active: Sequence[bool],
    sigmas: Sequence[float],
    latents: torch.Tensor,
    prediction_adapter: DenoisedPredictionAdapter,
    skipped_negative_index: int | None = None,
) -> torch.Tensor:
    positive = predictions[0].float()
    if do_true_cfg:
        negative = predictions[1].float()
        base = negative + cfg_scale * (positive - negative)
    else:
        base = positive
    guided = base
    if skipped_negative_index is not None:
        skipped_negative = predictions[skipped_negative_index].float()
        guided = skipped_negative + cfg_scale * (positive - skipped_negative)
    if config.cond_enabled:
        skipped = predictions[-1].float()
        if not do_true_cfg:
            guided = skipped
        guided = guided + config.scale * (positive - skipped)
    row_shape = (-1,) + (1,) * (positive.ndim - 1)
    mask = torch.tensor(active, device=positive.device, dtype=torch.bool).reshape(row_shape)
    if config.cond_enabled and config.rescaling_scale != 0:
        # Inactive rows may have sigma zero. Use a safe value there and select
        # their unmodified base prediction after rescaling.
        sigma = torch.tensor(sigmas, device=positive.device, dtype=torch.float32).reshape(row_shape)
        sigma = torch.where(mask, sigma, torch.ones_like(sigma))
        latent_fp32 = latents.float()
        positive_x0 = prediction_adapter.to_denoised(positive, latent_fp32, sigma)
        guided_x0 = prediction_adapter.to_denoised(guided, latent_fp32, sigma)
        flat_positive = positive_x0.flatten(1)
        flat_guided = guided_x0.flatten(1)
        correction = int(flat_positive.shape[1] > 1)
        positive_std = flat_positive.std(dim=1, correction=correction).reshape(row_shape)
        guided_std = flat_guided.std(dim=1, correction=correction).reshape(row_shape)
        ratio = positive_std / guided_std.clamp_min(torch.finfo(torch.float32).tiny)
        ratio = torch.where(guided_std > 0, ratio, torch.ones_like(ratio))
        factor = 1 + config.rescaling_scale * (ratio - 1)
        guided = prediction_adapter.from_denoised(guided_x0 * factor, latent_fp32, sigma)
    # Preserve inactive rows exactly, including in a mixed-step batch.
    return torch.where(mask, guided, base)


class SkipLayerGuidanceMixin(CFGParallelMixin):
    """Shared SLG for single-tensor predictions using multi-branch CFG dispatch.

    Pipelines declare ``slg_prediction_adapter`` and forward ``skip_layers``
    to their transformer. Override ``slg_skip_kwargs`` if its argument names
    differ. Keep schedule/config state on requests rather than the pipeline.
    """

    slg_prediction_adapter: DenoisedPredictionAdapter

    def slg_skip_kwargs(self, skip_layers: tuple[int, ...]) -> dict[str, Any]:
        return {"skip_layers": skip_layers}

    def predict_noise_with_slg(
        self,
        do_true_cfg: bool,
        true_cfg_scale: float,
        positive_kwargs: dict[str, Any],
        negative_kwargs: dict[str, Any] | None,
        *,
        slg_config: SkipLayerGuidanceConfig | None,
        active: Sequence[bool] = (),
        sigmas: Sequence[float] = (),
        latents: torch.Tensor,
    ) -> torch.Tensor:
        if slg_config is None or not slg_config.enabled or (not do_true_cfg and not slg_config.cond_enabled):
            return self.predict_noise_maybe_with_cfg(
                do_true_cfg, true_cfg_scale, positive_kwargs, negative_kwargs, cfg_normalize=False
            )
        if len(active) != latents.shape[0] or len(sigmas) != latents.shape[0]:
            raise ValueError("SLG activity and sigma must have one value per output sample.")
        if not any(active):
            return self.predict_noise_maybe_with_cfg(
                do_true_cfg, true_cfg_scale, positive_kwargs, negative_kwargs, cfg_normalize=False
            )
        if (
            slg_config.cond_enabled
            and slg_config.rescaling_scale
            and any(enabled and sigma <= 0 for enabled, sigma in zip(active, sigmas))
        ):
            raise ValueError("SLG rescaling requires positive sigma for active samples.")
        branches = [positive_kwargs]
        skipped_negative_index = None
        if do_true_cfg:
            if negative_kwargs is None:
                raise ValueError("CFG requires negative prediction arguments.")
            if slg_config.uncond_skip_layers:
                skipped_negative_kwargs = {**negative_kwargs, **self.slg_skip_kwargs(slg_config.uncond_skip_layers)}
                if all(active):
                    # Replace the negative branch, retaining two forwards for
                    # uncond-only guidance when the entire batch is active.
                    branches.append(skipped_negative_kwargs)
                else:
                    # Mixed sampling phases also need the normal negative for
                    # inactive rows. Keep every branch's collective shape.
                    branches.append(negative_kwargs)
                    skipped_negative_index = len(branches)
                    branches.append(skipped_negative_kwargs)
            else:
                branches.append(negative_kwargs)
        if slg_config.cond_enabled:
            branches.append({**positive_kwargs, **self.slg_skip_kwargs(slg_config.skip_layers)})

        def combine(predictions):
            return combine_skip_layer_guidance(
                predictions,
                do_true_cfg=do_true_cfg,
                cfg_scale=true_cfg_scale,
                config=slg_config,
                active=active,
                sigmas=sigmas,
                latents=latents,
                prediction_adapter=self.slg_prediction_adapter,
                skipped_negative_index=skipped_negative_index,
            )

        # SLG is itself a second guidance branch even when ordinary CFG is
        # disabled. All branches retain the full batch shape on every rank.
        return self.predict_noise_with_multi_branch_cfg(
            True, true_cfg_scale, branches, cfg_normalize=False, combine_fn=combine
        )
