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

import os
from dataclasses import fields as dataclass_fields

import numpy as np
import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.pi05.config import SUPPORTED_DTYPE_NAMES, Pi05Config
from vllm_omni.diffusion.models.pi05.modeling_pi05 import Pi05ForActionPrediction
from vllm_omni.diffusion.models.pi05.processor_pi05 import (
    Pi05RelativeActions,
    build_model_inputs,
)
from vllm_omni.diffusion.models.pi05_pipeline_config import PI05_PIPELINE as PI05_PIPELINE
from vllm_omni.diffusion.request import OmniDiffusionRequest

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


def _pi05_post_process(x):
    """Module-level identity post-process (picklable across the orchestrator's
    multiprocess boundary — a local closure is not)."""
    return x


def get_pi05_post_process_func(od_config: OmniDiffusionConfig):
    """π0.5 returns actions directly; post-processing is identity."""
    del od_config
    return _pi05_post_process


_LEROBOT_FLOAT32_IN_BFLOAT16 = (
    "vision_tower",
    "multi_modal_projector",
    "input_layernorm",
    "post_attention_layernorm",
    "model.norm",
)


def _keep_float32_like_lerobot(model: nn.Module) -> None:
    """Restore the parameters LeRobot leaves in float32 under bfloat16."""
    for name, param in model.named_parameters():
        if any(selector in name for selector in _LEROBOT_FLOAT32_IN_BFLOAT16):
            param.data = param.data.to(dtype=torch.float32)


class Pi05Pipeline(nn.Module):
    """π0.5 VLA pipeline: raw robot obs → continuous action chunk.

    Registered as ``"Pi05Pipeline"`` in the diffusion registry.
    """

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

        # One object serving both pipeline directions, matching LeRobot's
        # "same instance" pairing of Relative/AbsoluteActionsProcessorStep.
        self.relative_actions = Pi05RelativeActions(
            enabled=self.config.use_relative_actions,
            exclude_joints=self.config.relative_exclude_joints,
            action_names=self.config.action_feature_names,
            max_action_dim=self.config.max_action_dim,
        )
        if self.relative_actions.enabled:
            logger.info(
                "Pi05Pipeline: relative actions enabled — %d of %d action dims are "
                "state-relative (excluded joints: %s).",
                self.relative_actions.num_relative_dims,
                self.config.max_action_dim,
                self.config.relative_exclude_joints,
            )

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
        # the download carries vLLM's user agent like every other repo access.
        from vllm.transformers_utils.repo_utils import hf_api

        return hf_api().snapshot_download(
            repo_id=model,
            allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
        )

    def _build_config(self, od_config: OmniDiffusionConfig) -> Pi05Config:
        """Build Pi05Config from deploy-yaml model_config, falling back to the
        checkpoint's config.json (raw LeRobot format).

        Explicit deploy values override checkpoint defaults, while omitted
        fields retain the checkpoint's complete runtime schema. Relative-action
        semantics remain checkpoint-owned because changing them would reinterpret
        the trained output distribution.
        """
        checkpoint_config = Pi05Config.from_pretrained(self.model_dir) if self.model_dir else None
        if od_config.model_config:
            deploy_raw = dict(od_config.model_config)
            if checkpoint_config is None:
                return Pi05Config.from_model_config(deploy_raw)

            merged = {
                item.name: getattr(checkpoint_config, item.name) for item in dataclass_fields(Pi05Config) if item.init
            }
            merged.update(deploy_raw)
            for schema_key in ("input_features", "output_features", "norm_stats", "state_norm_stats"):
                if not deploy_raw.get(schema_key):
                    merged[schema_key] = getattr(checkpoint_config, schema_key)

            if checkpoint_config.use_relative_actions and not deploy_raw.get("use_relative_actions"):
                logger.warning(
                    "Pi05Pipeline: checkpoint declares use_relative_actions=True; honouring the "
                    "checkpoint because its normalization statistics are in relative action space."
                )
                merged["use_relative_actions"] = True
                merged["relative_exclude_joints"] = checkpoint_config.relative_exclude_joints
                merged["action_feature_names"] = checkpoint_config.action_feature_names
            return Pi05Config.from_model_config(merged)
        if checkpoint_config is not None:
            return checkpoint_config
        return Pi05Config()

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
        self._load_checkpoint(model)
        model.to(device=self._device, dtype=self._torch_dtype)
        if self._torch_dtype is torch.bfloat16:
            _keep_float32_like_lerobot(model)
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
    def forward(self, req: OmniDiffusionRequest, **kwargs) -> DiffusionOutput:
        extra_args = getattr(req.sampling_params, "extra_args", None) or {}
        robot_obs = extra_args.get("robot_obs")

        if robot_obs is None:
            # Dummy warmup path (no obs): return zeros so engine warmup/capture
            # doesn't crash. Mirrors DreamZero's dummy-run handling.
            first_prompt = req.prompts[0] if req.prompts else ""
            prompt = first_prompt if isinstance(first_prompt, str) else (first_prompt.get("prompt") or "")
            num_steps = getattr(req.sampling_params, "num_inference_steps", None)
            if prompt == "dummy run" or num_steps == 1:
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

        # Serving input steps. Note: no state tensor comes back — π0.5 serializes
        # the (normalized, discretized) state into lang_tokens.
        images, image_masks, lang_tokens, lang_masks = build_model_inputs(
            robot_obs, self.config, self.tokenizer, self._device
        )

        num_steps = getattr(req.sampling_params, "num_inference_steps", None)
        if num_steps is not None and (
            isinstance(num_steps, bool) or not isinstance(num_steps, (int, np.integer)) or int(num_steps) < 1
        ):
            raise ValueError(f"num_inference_steps must be a positive integer, got {num_steps!r}.")
        if num_steps is not None:
            num_steps = int(num_steps)

        actions = self.model.sample_actions(
            images=images,
            image_masks=image_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            num_steps=num_steps,
        )

        # Output step 1: unnormalize.
        actions = self.model._unnormalize_actions(actions)
        # Output step 2: relative → absolute, against the RAW state (the same
        # state the prompt encoded, before normalization).
        if self.relative_actions.enabled:
            actions = self.relative_actions.to_absolute(actions, robot_obs.get("state"))

        # Output step 3: to_cpu. (B=1, horizon, action_dim) → (horizon, action_dim).
        actions_np = actions.squeeze(0)[..., : self.config.action_dim].float().cpu().numpy()

        return DiffusionOutput(output={"actions": actions_np})
