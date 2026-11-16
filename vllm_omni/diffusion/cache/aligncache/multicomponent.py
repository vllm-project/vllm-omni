# SPDX-License-Identifier: Apache-2.0
"""Per-transformer Wan controllers with one request owner and capacity limit.

Wan2.1 T2V and Wan2.2 TI2V-5B use one controller; Wan2.2 A14B uses one per expert.
"""

import json
import logging
import math
from contextlib import ExitStack, contextmanager
from dataclasses import replace

import torch

from .config import WAN22_EXPERT_PROFILES
from .runtime import AlignCacheRuntime
from .state import GenerationSignature, SchedulePlan, StepContext

logger = logging.getLogger(__name__)


class WanRuntime:
    def __init__(self, config):
        self.config, self.mode = config, config.mode
        self.active = False
        self.last_summary = {}
        keys = ("high", "low") if config.profile in WAN22_EXPERT_PROFILES else ("transformer",)
        # Each component has a disjoint share of the request-wide admission bound.
        child_config = replace(config, max_cache_bytes=config.max_cache_bytes // len(keys))
        self.components = {key: AlignCacheRuntime(child_config) for key in keys}
        for child in self.components.values():
            child.emit_logs = False
        self.reset()

    def reset(self):
        if self.active:
            raise RuntimeError("align_cache: request already active")
        for child in self.components.values():
            child.reset()
        self.routes = None
        self.completed_pairs = 0
        self.scheduler_identity = None
        self.concat_condition = False

    @contextmanager
    def request_context(self):
        self.reset()
        self.active = True
        for child in self.components.values():
            child.last_summary = {}
        status = "failed"
        try:
            with ExitStack() as stack:
                for child in self.components.values():
                    stack.enter_context(child.request_context())
                yield self
                status = "completed"
        finally:
            self.active = False
            self.routes = self.scheduler_identity = None
            self.concat_condition = False
            self.last_summary = {
                "backend": "align_cache",
                "policy": self.config.policy,
                "status": status,
                "configured": True,
                "installed": True,
                "mode": self.mode,
                "config_hash": self.config.config_hash,
                "components": {key: dict(child.last_summary) for key, child in self.components.items()},
            }
            for key in ("completed_pairs", "exact_pairs", "fast_pairs", "model_forward_calls", "feedback_updates"):
                self.last_summary[key] = sum(child.last_summary.get(key, 0) for child in self.components.values())
            if self.config.diagnostics != "off":
                try:
                    logger.info("AlignCache Wan request: %s", self.last_summary)
                    if self.config.diagnostics == "trace":
                        logger.info(
                            "AlignCache component traces: %s",
                            {key: list(child.trace) for key, child in self.components.items()},
                        )
                except Exception:
                    pass

    def bind_schedule(
        self,
        pipeline,
        timesteps,
        x,
        cond,
        uncond,
        guidance_low,
        guidance_high,
        boundary,
        dtype,
        latent_condition=None,
        first_frame_mask=None,
    ):
        if not self.active or self.routes is not None:
            raise RuntimeError("align_cache: schedule requires a fresh request")
        profile = self.config.profile
        if profile == "wan22_i2v_reference_v1":
            if latent_condition is None or pipeline.expand_timesteps:
                raise ValueError("align_cache: Wan2.2 I2V-A14B requires its concatenated image condition")
        elif profile == "wan22_ti2v_reference_v1":
            if not pipeline.expand_timesteps or (latent_condition is None) != (first_frame_mask is None):
                raise ValueError("align_cache: Wan2.2 TI2V-5B requires expand-timesteps first-frame conditioning")
        elif latent_condition is not None or first_frame_mask is not None or pipeline.expand_timesteps:
            raise ValueError("align_cache: Wan image-conditioned paths are not admitted")
        if x.ndim != 5 or x.shape[0] != 1 or x.device.type != "cuda" or dtype != torch.bfloat16:
            raise ValueError("align_cache: requires B=1 CUDA BF16 input")
        if uncond is None or cond.shape[0] != 1 or uncond.shape[0] != 1:
            raise ValueError("align_cache: serial CFG conditions required")
        single = len(self.components) == 1
        if single and (pipeline.transformer_2 is not None or guidance_low != guidance_high):
            raise ValueError("align_cache: single-transformer Wan requires one transformer and one guidance scale")
        if not single and (boundary is None or not math.isfinite(boundary)):
            raise ValueError("align_cache: finite native boundary required")
        for guidance in (guidance_low, guidance_high):
            if not math.isfinite(guidance) or guidance <= 1:
                raise ValueError("align_cache: both component guidance scales must exceed 1")
        if timesteps is not pipeline.scheduler.timesteps or len(timesteps) != pipeline.scheduler.num_inference_steps:
            raise ValueError("align_cache: requires native scheduler time grid")
        values = tuple(float(t) for t in timesteps)
        if not values or any(not math.isfinite(t) for t in values) or any(a <= b for a, b in zip(values, values[1:])):
            raise ValueError("align_cache: strictly descending finite schedule required")
        config = json.dumps(dict(pipeline.scheduler.config), sort_keys=True, default=str)
        self.scheduler_identity = (id(pipeline.scheduler), id(timesteps), config)
        self.concat_condition = profile == "wan22_i2v_reference_v1"
        if single:
            self.routes = ("transformer",) * len(values)
            bindings = (("transformer", pipeline.transformer, guidance_low),)
        else:
            self.routes = tuple("low" if t < boundary else "high" for t in values)
            bindings = (
                ("high", pipeline.transformer, guidance_low),
                ("low", pipeline.transformer_2, guidance_high),
            )
        for key, model, guidance in bindings:
            selected = tuple(t for t, route in zip(values, self.routes) if route == key)
            if not selected:
                continue
            child = self.components[key]
            child.bind_schedule(
                SchedulePlan(selected, type(pipeline.scheduler).__qualname__, config),
                GenerationSignature(
                    child.epoch, id(model), (id(cond), id(uncond)), guidance, tuple(x.shape), dtype, x.device
                ),
            )

    def predict_noise(self, pipeline, index, t, guidance, positive, negative, latents=None):
        if not self.active or self.routes is None or index != self.completed_pairs or index >= len(self.routes):
            raise ValueError("align_cache: missing schedule or out-of-order evaluation")
        identity = (
            id(pipeline.scheduler),
            id(pipeline.scheduler.timesteps),
            json.dumps(dict(pipeline.scheduler.config), sort_keys=True, default=str),
        )
        if identity != self.scheduler_identity:
            raise ValueError("align_cache: scheduler changed within request")
        child = self.components[self.routes[index]]
        if negative is None or negative["current_model"] is not positive["current_model"]:
            raise ValueError("align_cache: matching native CFG branches required")
        x = positive["hidden_states"]
        if negative["hidden_states"] is not x:
            raise ValueError("align_cache: CFG branches must share input")
        if (latents is not None) != self.concat_condition:
            raise ValueError("align_cache: noisy latent channels are required exactly for concatenated I2V input")
        if latents is not None:
            # I2V appends a fixed image condition; predict on the noisy channels the model outputs.
            if latents.shape[1] >= x.shape[1] or latents.shape[2:] != x.shape[2:]:
                raise ValueError("align_cache: noisy latents must lead the concatenated I2V input")
            x = latents
        step = StepContext(child.counts["completed_pairs"], float(t), id(positive["current_model"]), guidance)
        conditions = (id(positive["encoder_hidden_states"]), id(negative["encoder_hidden_states"]))
        child._validate_step(step, x, conditions)
        if self.mode == "passthrough":
            output = pipeline.predict_noise_maybe_with_cfg(True, guidance, positive, negative, cfg_normalize=False)
            child.record_passthrough()
        else:
            cond, uncond = child.predict_pair(
                step,
                x,
                conditions,
                lambda branch: pipeline.predict_noise(**(positive if branch == "cond" else negative)),
            )
            output = pipeline.combine_cfg_noise(cond, uncond, guidance, False)
        self.completed_pairs += 1
        return output

    def record_forward(self, model):
        if self.mode == "passthrough" and self.active:
            for child in self.components.values():
                if child.signature is not None and child.signature.component == id(model):
                    child.counts["model_forward_calls"] += 1
                    return
