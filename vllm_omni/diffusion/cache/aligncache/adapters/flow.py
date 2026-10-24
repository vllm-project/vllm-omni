# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native flow model admission and Qwen target-tail/CFG boundary."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch

from ..flow import FlowRuntime


def validate_pipeline(pipeline: Any, policy: str, quantized: bool = False) -> None:
    if any(name in vars(pipeline) for name in ("forward", "diffuse", "predict_noise", "combine_cfg_noise")):
        raise ValueError("align_cache: replaced native pipeline methods are unsupported")
    if policy == "qwen_flow_v1":
        from diffusers import FlowMatchEulerDiscreteScheduler

        from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import QwenImage21Pipeline
        from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import QwenImage21Transformer2DModel

        if type(pipeline) is not QwenImage21Pipeline or type(pipeline.transformer) is not QwenImage21Transformer2DModel:
            raise ValueError("align_cache: qwen_flow_v1 requires native Qwen-Image-2.1")
        if type(pipeline.scheduler) is not FlowMatchEulerDiscreteScheduler:
            raise ValueError("align_cache: Qwen requires native FlowMatchEulerDiscreteScheduler")
        if getattr(pipeline.transformer, "_decode_graph_manager", None) is not None:
            raise ValueError("align_cache: Qwen decode graphs are unsupported")
    else:
        from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel
        from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

        if type(pipeline) is not MiniMaxH3Pipeline or type(pipeline.transformer) is not MiniMaxH3DiTModel:
            raise ValueError("align_cache: H3 requires native cfg-distilled MiniMaxH3Pipeline")
        if any(getattr(pipeline, name, None) is not None for name in ("_fasth3", "_fasth3_checkpoint", "_vdn")):
            raise ValueError("align_cache: FastH3/VDNH3 combinations are unsupported")
    if quantized:
        # Online FP8 keeps BF16 activations/outputs; FP8 weights carry FP32 scales.
        parameters = list(pipeline.transformer.parameters())
        dtypes = {p.dtype for p in parameters}
        if torch.bfloat16 not in dtypes or not dtypes <= {torch.bfloat16, torch.float8_e4m3fn, torch.float32}:
            raise ValueError("align_cache: quantized flow model requires BF16 with online FP8 weights")
        if any(p.device.type != "cuda" for p in parameters):
            raise ValueError("align_cache: flow model requires CUDA weights")
    else:
        parameter = next(pipeline.transformer.parameters())
        if parameter.dtype != torch.bfloat16 or parameter.device.type != "cuda":
            raise ValueError("align_cache: flow model requires CUDA BF16 weights")
    for module in pipeline.transformer.modules():
        if "forward" in vars(module) or any(
            getattr(module, key, None)
            for key in ("_forward_hooks", "_forward_pre_hooks", "_hook_registry", "_hf_hook", "_diffusers_hook")
        ):
            raise ValueError("align_cache: flow model hooks/replacements are unsupported")


class QwenFlowRuntime(FlowRuntime):
    def reset(self) -> None:
        super().reset()
        self._prefix_signature: tuple | None = None

    @contextmanager
    def request_context(self) -> Iterator[QwenFlowRuntime]:
        try:
            with super().request_context():
                yield self
        finally:
            # A rejected reentrant scope must not clear the active owner's signature.
            if not self.active:
                self._prefix_signature = None

    def _validate_prefix(
        self,
        latents: torch.Tensor,
        positive: dict[str, Any],
        negative: dict[str, Any] | None,
        cfg: bool,
        index: int,
    ) -> None:
        branches = [("cond", positive)]
        if cfg:
            if negative is None:
                raise ValueError("align_cache: missing unconditional branch")
            branches.append(("uncond", negative))
        cache = positive.get("kv_cache")
        signatures = []
        for name, kwargs in branches:
            if kwargs.get("cache_branch") != name:
                raise ValueError("align_cache: CFG cache branch identity changed")
            if kwargs.get("hidden_states") is not latents:
                raise ValueError("align_cache: branches must use the native target input")
            if kwargs.get("kv_cache") is not cache:
                raise ValueError("align_cache: branches must share the native prefix KV cache")
            signatures.append(
                tuple(
                    id(kwargs.get(key))
                    for key in ("kv_cache", "encoder_hidden_states_mask", "img_mask", "img_shapes", "attention_kwargs")
                )
            )
        signature = tuple(signatures)
        if self._prefix_signature is not None and signature != self._prefix_signature:
            raise ValueError("align_cache: prefix KV or condition metadata identity changed")
        if cache is not None:
            if not isinstance(cache, list) or not cache:
                raise ValueError("align_cache: invalid native prefix KV cache")
            if index > 0 and any("key" not in block.get(name, {}) for block in cache for name, _ in branches):
                raise ValueError("align_cache: native prefix KV lost its prefill state")
        self._prefix_signature = signature

    def bind_qwen(self, pipeline, latents, timesteps, cond, uncond, cfg, scale, image_latents, extras):
        # The native T2I call always passes return_dict=False; any other kwarg is an extra condition.
        if image_latents is not None or dict(extras) not in ({}, {"return_dict": False}):
            raise ValueError("align_cache: Qwen editing/additional conditions are not admitted")
        if latents.shape[0] != 1 or latents.device.type != "cuda":
            raise ValueError("align_cache: Qwen requires single-sample CUDA input")
        if timesteps is not pipeline.scheduler.timesteps:
            raise ValueError("align_cache: Qwen requires actual scheduler timesteps")
        sigmas = tuple(float(s) for s in pipeline.scheduler.sigmas)
        if len(sigmas) != len(timesteps) + 1:
            raise ValueError("align_cache: Qwen sigma schedule must include terminal sigma")
        if not math.isfinite(scale) or (cfg and scale <= 1):
            raise ValueError("align_cache: finite native CFG scale required")
        if cfg and uncond is None:
            raise ValueError("align_cache: missing unconditional condition")
        identity = (
            id(pipeline.transformer),
            id(pipeline.scheduler),
            id(cond),
            id(uncond) if cfg else id(None),
            bool(cfg),
            float(scale),
            tuple(latents.shape),
        )
        self.bind([sigmas] * (2 if cfg else 1), identity)

    def predict_qwen(
        self,
        pipeline: Any,
        index: int,
        latents: torch.Tensor,
        positive: dict[str, Any],
        negative: dict[str, Any] | None,
        cfg: bool,
        scale: float,
    ) -> torch.Tensor:
        if not self.active or self.schedule is None or index != self.index:
            raise ValueError("align_cache: missing plan or out-of-order evaluation")
        self._validate_prefix(latents, positive, negative, cfg, index)
        identity = (
            id(pipeline.transformer),
            id(pipeline.scheduler),
            id(positive["encoder_hidden_states"]),
            id(negative["encoder_hidden_states"]) if negative is not None and cfg else id(None),
            bool(cfg),
            float(scale),
            tuple(latents.shape),
        )

        def combine(outputs):
            if len(outputs) == 1:
                return outputs[0]
            return pipeline.combine_cfg_noise(outputs[0], outputs[1], scale, False)

        def exact(count):
            out = []
            for kwargs in [positive, negative] if cfg else [positive]:
                count()
                out.append(pipeline.predict_noise(**kwargs)[:, -latents.size(1) :])
            return out

        # Step zero must run the native prefix prefill regardless of warmup overrides.
        outputs = self.evaluate(index, [latents] * (2 if cfg else 1), identity, exact, combine, force_exact=index == 0)
        return combine(outputs)
