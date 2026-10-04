# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""π0.5 VLA pipeline for vllm-omni.

Entry point for ``DiffusionEngine.step() → pipeline.forward(req)``. Mirrors the
DreamZero contract and the π0 pipeline: the pipeline owns ALL preprocessing. It
reads the raw robot observation from
``req.sampling_params.extra_args["robot_obs"]`` (delivered by the OpenPI
realtime serving layer), builds model inputs, runs flow-matching denoising, and
returns ``DiffusionOutput(output={"actions": ndarray})``.

π0.5 is stateless across calls (no KV reuse, first-order Markov), so
``session_id`` / ``reset`` from the OpenPI protocol are accepted but ignored.

The post-processing order is load-bearing and matches LeRobot::

    unnormalize → absolute actions → to_cpu

``AbsoluteActionsProcessorStep`` must run *after* unnormalization, because a
relative-action checkpoint's ``norm_stats`` are computed in relative space.
"""

from __future__ import annotations

import json
import os
from dataclasses import fields as dataclass_fields
from functools import partial

import numpy as np
import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.pi05.config import SUPPORTED_DTYPE_NAMES, Pi05Config
from vllm_omni.diffusion.models.pi05.modeling_pi05 import Pi05ForActionPrediction
from vllm_omni.diffusion.models.pi05.processor_pi05 import Pi05Processor
from vllm_omni.diffusion.models.pi05_pipeline_config import PI05_PIPELINE as PI05_PIPELINE
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.utils import StepRequestState

logger = init_logger(__name__)

# π0.5 pins the PaliGemma tokenizer (LeRobot hardcodes it too).
DEFAULT_PI05_TOKENIZER = "google/paligemma-3b-pt-224"

# float16 is absent deliberately, not by oversight: nothing here has been
# validated against it. The float32/bfloat16 trade-off is in the deploy config
# and recipes/lerobot/Pi05.md.
SUPPORTED_DTYPES = (torch.float32, torch.bfloat16)

# The two lists guard different entry points — what the checkpoint declares
# versus the dtype actually cast to — so they must not drift apart.
assert {str(dtype).split(".")[-1] for dtype in SUPPORTED_DTYPES} == set(SUPPORTED_DTYPE_NAMES)


def _checkpoint_declared_keys(model_dir: str) -> set[str]:
    """The keys the checkpoint's config.json actually carries."""
    path = os.path.join(model_dir, "config.json")
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return set(json.load(f))


def _comparable(value):
    """Compare yaml and checkpoint values without tripping on list/tuple."""
    return list(value) if isinstance(value, (list, tuple)) else value


def _build_pi05_config(model_dir: str | None, model_config: dict | None) -> Pi05Config:
    checkpoint = Pi05Config.from_pretrained(model_dir) if model_dir else None
    if checkpoint is None:
        return Pi05Config.from_model_config(model_config)
    if not model_config:
        return checkpoint
    resolved = {item.name: getattr(checkpoint, item.name) for item in dataclass_fields(Pi05Config) if item.init}
    declared_keys = _checkpoint_declared_keys(model_dir)
    for key, value in model_config.items():
        if key in resolved and key in declared_keys and _comparable(value) != _comparable(resolved[key]):
            logger.warning(
                "Pi05Pipeline: the deploy config sets %s=%r, overriding %r from the checkpoint.",
                key,
                value,
                resolved[key],
            )
    resolved.update(model_config)
    return Pi05Config.from_model_config(resolved)


def _pi05_post_process(x):
    """Module-level identity post-process (picklable across the orchestrator's
    multiprocess boundary — a local closure is not)."""
    return x


def get_pi05_post_process_func(od_config: OmniDiffusionConfig):
    """π0.5 returns actions directly; post-processing is identity."""
    del od_config
    return _pi05_post_process


def _resolve_steps(value, default: int) -> int:
    value = default if value is None else value
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError(f"num_inference_steps must be a positive integer, got {value!r}.")
    return int(value)


def _is_dummy_request(prompt, sampling) -> bool:
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else ""
    prompt = prompt if isinstance(prompt, str) else (prompt.get("prompt") or "")
    return prompt == "dummy run" or sampling.num_inference_steps == 1


def _pi05_pre_process(req: OmniDiffusionRequest, *, default_steps: int, step_execution: bool):
    sampling = req.sampling_params
    if step_execution:
        # Warmup's zero-output forward has no conditioning and cannot share a
        # denoise batch. Keep its existing full-forward behavior.
        if not (sampling.extra_args or {}).get("robot_obs") and _is_dummy_request(req.prompt, sampling):
            req.use_step_execution = False
        sampling.num_inference_steps = _resolve_steps(sampling.num_inference_steps, default_steps)
        if req.use_step_execution:
            if sampling.timesteps is not None or sampling.sigmas is not None:
                raise ValueError(
                    "Pi0.5 step execution uses its fixed Euler schedule; timesteps/sigmas are unsupported."
                )
    return req


def get_pi05_pre_process_func(od_config: OmniDiffusionConfig):
    default_steps = 10
    if od_config.step_execution:
        model_dir = od_config.model
        if model_dir and not os.path.isdir(model_dir):
            from vllm_omni.transformers_utils.repo_utils import hf_api

            model_dir = hf_api().snapshot_download(repo_id=model_dir, allow_patterns=["config.json"])
        # Use precisely the same checkpoint/deploy resolution as the worker,
        # without constructing the model or downloading weights in the engine.
        config = _build_pi05_config(model_dir, od_config.model_config)
        default_steps = config.num_inference_steps
    return partial(_pi05_pre_process, default_steps=default_steps, step_execution=od_config.step_execution)


_LEROBOT_FLOAT32_IN_BFLOAT16 = (
    "vision_tower",
    "multi_modal_projector",
    "input_layernorm",
    "post_attention_layernorm",
    "model.norm",
)


def _to_bfloat16_for_inference(model: Pi05ForActionPrediction) -> None:
    """Apply LeRobot's mixed-precision inference layout explicitly."""
    inner_model = model.paligemma_with_expert
    for name, param in inner_model.named_parameters():
        target_dtype = (
            torch.float32 if any(selector in name for selector in _LEROBOT_FLOAT32_IN_BFLOAT16) else torch.bfloat16
        )
        param.data = param.data.to(dtype=target_dtype)
    for buffer in inner_model.buffers():
        if buffer.is_floating_point():
            buffer.data = buffer.data.to(dtype=torch.bfloat16)

    for name, param in model.named_parameters():
        if not name.startswith("paligemma_with_expert."):
            param.data = param.data.to(dtype=torch.float32)
    for name, buffer in model.named_buffers():
        if not name.startswith("paligemma_with_expert.") and buffer.is_floating_point():
            buffer.data = buffer.data.to(dtype=torch.float32)


def _set_inference_dtype(model: Pi05ForActionPrediction, dtype: torch.dtype) -> None:
    if dtype == torch.float32:
        model.to(dtype=torch.float32)
    elif dtype == torch.bfloat16:
        _to_bfloat16_for_inference(model)
    else:
        raise ValueError(f"Unsupported π0.5 inference dtype: {dtype!r}.")


class Pi05Pipeline(nn.Module):
    """π0.5 VLA pipeline: raw robot obs → continuous action chunk.

    Registered as ``"Pi05Pipeline"`` in the diffusion registry.
    """

    supports_step_execution = True

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.prefix = prefix
        self.model_dir = self._resolve_model_dir(od_config.model)
        self.config = self._build_config(od_config)

        custom_args = od_config.custom_pipeline_args or {}
        self.tokenizer_source = str(custom_args.get("tokenizer", self._resolve_tokenizer_source()))

        self._torch_dtype = self._resolve_dtype(od_config)
        self._device = self._resolve_device(od_config)

        self.tokenizer = self._load_tokenizer()
        self.model = self._initialize_model()

        self.processor = Pi05Processor(self.config, self.tokenizer, self._device)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_model_dir(model: str | None) -> str | None:
        """Return a local directory for ``model``; download an HF repo id if needed."""
        if not model:
            return None
        if os.path.isdir(model):
            return model
        # Via repo_utils' shared HfApi rather than huggingface_hub directly, so
        # the download carries vLLM-Omni's user agent like every other repo access.
        from vllm_omni.transformers_utils.repo_utils import hf_api

        return hf_api().snapshot_download(
            repo_id=model,
            allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
        )

    def _build_config(self, od_config: OmniDiffusionConfig) -> Pi05Config:
        """Read the config from the checkpoint, then let the deploy yaml override it."""
        return _build_pi05_config(self.model_dir, od_config.model_config)

    def _resolve_tokenizer_source(self) -> str:
        """Prefer the checkpoint dir if it ships tokenizer files; else PaliGemma."""
        if self.model_dir and os.path.isdir(self.model_dir):
            if os.path.exists(os.path.join(self.model_dir, "tokenizer_config.json")):
                return self.model_dir
        return DEFAULT_PI05_TOKENIZER

    @staticmethod
    def _resolve_dtype(od_config: OmniDiffusionConfig) -> torch.dtype:
        """Resolve the dtype the weights are actually cast to.

        This is the load-bearing check, not ``Pi05Config.dtype``: the cast in
        :meth:`_initialize_model` reads the *top-level* ``OmniDiffusionConfig``
        field, so a guard on the model config alone would let an unsupported
        dtype through. See :data:`SUPPORTED_DTYPES` for why float16 is excluded.
        """
        dt = od_config.dtype
        resolved = dt if isinstance(dt, torch.dtype) else getattr(torch, str(dt).split(".")[-1], None)
        if resolved not in SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported π0.5 dtype: {dt!r}. Supported: "
                f"{', '.join(sorted(str(d).split('.')[-1] for d in SUPPORTED_DTYPES))}."
            )
        return resolved

    @staticmethod
    def _resolve_device(od_config: OmniDiffusionConfig) -> torch.device:
        from vllm_omni.diffusion.distributed.utils import get_local_device

        try:
            return get_local_device()
        except Exception:  # noqa: BLE001
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        # padding_side="right" is part of the π0.5 spec: the prefix is a fixed
        # 200-token block whose live tokens must start at index 0.
        return AutoTokenizer.from_pretrained(self.tokenizer_source, padding_side="right")

    def has_real_checkpoint(self) -> bool:
        return bool(self.model_dir) and os.path.exists(os.path.join(self.model_dir, "model.safetensors"))

    def _initialize_model(self) -> Pi05ForActionPrediction:
        if not self.has_real_checkpoint():
            expected = os.path.join(self.model_dir or "<missing-model-dir>", "model.safetensors")
            raise FileNotFoundError(f"π0.5 serving requires checkpoint weights at {expected}.")
        model = Pi05ForActionPrediction(self.config)
        _set_inference_dtype(model, self._torch_dtype)
        model.to(device=self._device)
        self._load_checkpoint(model)
        model.eval()
        return model

    def _load_checkpoint(self, model: Pi05ForActionPrediction) -> None:
        import safetensors.torch

        path = os.path.join(self.model_dir, "model.safetensors")
        logger.info("Pi05Pipeline: loading π0.5 weights from %s", path)
        try:
            state = safetensors.torch.load_file(path)
            model.load_weights(state.items())
        except Exception as exc:
            raise RuntimeError(f"Failed to load complete π0.5 checkpoint {path}: {exc}") from exc

    # ------------------------------------------------------------------
    # Framework weight-loading hook
    # ------------------------------------------------------------------
    def load_weights(self, weights=()):  # noqa: D401
        """No-op for the diffusion loader: π0.5 self-loads its checkpoint."""
        for _ in weights:
            pass
        return None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def prepare_encode(self, state: StepRequestState, **kwargs) -> StepRequestState:
        obs = (state.sampling.extra_args or {}).get("robot_obs")
        if obs is None:
            raise ValueError("Pi0.5 step execution requires robot_obs.")
        steps = _resolve_steps(state.sampling.num_inference_steps, self.config.num_inference_steps)
        images, masks, tokens, token_masks = self.processor.build_model_inputs(obs)
        state.latents = self.model.initialize_noise(tokens, state.sampling.generator)
        prefix_mask, kv = self.model.encode_prefix(images, masks, tokens, token_masks)
        dt = -1.0 / steps
        # Compute in Python double precision before casting, exactly as the
        # monolithic loop does. A float32 arange changes rounded timesteps.
        state.timesteps = torch.tensor([1.0 + i * dt for i in range(steps)], device=tokens.device, dtype=torch.float32)
        state.step_index = 0
        state.extra.update(pi05_prefix_mask=prefix_mask, pi05_kv=kv, pi05_dt=dt, pi05_obs=obs)
        return state

    @torch.inference_mode()
    def denoise_step(self, input_batch: InputBatch, *, states=None, **kwargs) -> torch.Tensor:
        # InputBatch order is authoritative: newly admitted requests may precede
        # previously running ones. Never infer identity from a row's old slot.
        states = input_batch.states
        masks = torch.cat([s.extra["pi05_prefix_mask"] for s in states], dim=0)
        caches = [s.extra["pi05_kv"] for s in states]
        kv = [
            (torch.cat([c[i][0] for c in caches], dim=0), torch.cat([c[i][1] for c in caches], dim=0))
            for i in range(len(caches[0]))
        ]
        return self.model.denoise_step(masks, kv, input_batch.latents, input_batch.timesteps)

    def step_scheduler(self, state: StepRequestState, noise_pred: torch.Tensor, **kwargs) -> None:
        state.latents = state.latents + state.extra["pi05_dt"] * noise_pred
        state.step_index += 1

    def post_decode(self, state: StepRequestState, **kwargs) -> DiffusionOutput:
        return DiffusionOutput(
            output={"actions": self.processor.build_model_outputs(state.latents, state.extra["pi05_obs"])}
        )

    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest, **kwargs) -> DiffusionOutput:
        extra_args = getattr(req.sampling_params, "extra_args", None) or {}
        robot_obs = extra_args.get("robot_obs")

        if robot_obs is None:
            # Dummy warmup path (no obs): return zeros so engine warmup/capture
            # doesn't crash. Mirrors DreamZero's dummy-run handling.
            prompt = getattr(req, "prompt", getattr(req, "prompts", []))
            if _is_dummy_request(prompt, req.sampling_params):
                logger.info("Pi05Pipeline: dummy warmup request without robot_obs — returning zeros.")
                return DiffusionOutput(
                    output={
                        "actions": np.zeros(
                            (self.config.chunk_size, self.config.action_dim),
                            dtype=np.float32,
                        )
                    },
                )
            return DiffusionOutput(
                error="Pi05Pipeline.forward requires sampling_params.extra_args['robot_obs'].",
            )

        images, image_masks, lang_tokens, lang_masks = self.processor.build_model_inputs(robot_obs)

        num_steps = getattr(req.sampling_params, "num_inference_steps", None)
        if num_steps is not None:
            num_steps = _resolve_steps(num_steps, self.config.num_inference_steps)

        actions = self.model.sample_actions(
            images=images,
            image_masks=image_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            num_steps=num_steps,
            generator=req.sampling_params.generator,
        )

        return DiffusionOutput(output={"actions": self.processor.build_model_outputs(actions, robot_obs)})
