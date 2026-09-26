# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""π0 (Pi-Zero) VLA pipeline for vllm-omni.

Entry point for ``DiffusionEngine.step() → pipeline.forward(req)``. Mirrors the
DreamZero contract: the pipeline owns ALL preprocessing. It reads the raw robot
observation from ``req.sampling_params.extra_args["robot_obs"]`` (delivered by
the OpenPI realtime serving layer), builds model inputs, runs flow-matching
denoising, and returns ``DiffusionOutput(output={"actions": ndarray})`` — which
``diffusion_engine`` promotes to ``multimodal_output["actions"]`` for the client.

π0 is stateless across calls (no KV reuse), so ``session_id`` / ``reset`` from
the OpenPI protocol are accepted but ignored.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.pi.common import pipeline as pipeline_helpers
from vllm_omni.diffusion.models.pi.pi0.config import SUPPORTED_DTYPE_NAMES, Pi0Config
from vllm_omni.diffusion.models.pi.pi0.modeling_pi0 import Pi0ForActionPrediction
from vllm_omni.diffusion.models.pi.pi0.processor_pi0 import build_model_inputs
from vllm_omni.diffusion.models.pi0_pipeline_config import PI0_PIPELINE as PI0_PIPELINE
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)

# Default tokenizer for the PaliGemma prefix (matches LeRobot Pi0).
DEFAULT_PI0_TOKENIZER = "google/paligemma-3b-pt-224"

# Pi0 uses a homogeneous model dtype. The denoising state remains float32 and
# model-bound inputs are cast at their projection/vision boundaries.
SUPPORTED_DTYPES = (torch.float32, torch.bfloat16)
assert {str(dtype).split(".")[-1] for dtype in SUPPORTED_DTYPES} == set(SUPPORTED_DTYPE_NAMES)


# The registry imports this public name, and the returned module-level function
# must remain picklable across the orchestrator's multiprocess boundary.
_pi0_post_process = pipeline_helpers.identity_post_process
get_pi0_post_process_func = pipeline_helpers.get_identity_post_process_func


def _set_inference_dtype(model: Pi0ForActionPrediction, dtype: torch.dtype) -> None:
    """Apply Pi0's homogeneous FP32 or BF16 inference layout."""
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported π0 inference dtype: {dtype!r}.")
    model.to(dtype=dtype)


class Pi0Pipeline(nn.Module):
    """π0 VLA pipeline: raw robot obs → continuous action chunk.

    Registered as ``"Pi0Pipeline"`` in the diffusion registry. Weights are
    self-loaded in ``__init__`` from the checkpoint's ``model.safetensors`` via
    the kernel's ``load_weights`` (which handles the LeRobot key remaps).
    """

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.prefix = prefix
        # Resolve od_config.model to a LOCAL directory. A bare HF repo id (e.g.
        # the documented ``lerobot/pi0_base``) must be snapshot-downloaded first;
        # otherwise config/tokenizer/weights would silently fall back to random
        # init (this pipeline self-loads from a local dir and has no
        # ``weights_sources`` for the framework loader to download from).
        self.model_dir = pipeline_helpers.resolve_model_dir(od_config.model)
        self.config = self._build_config(od_config)

        custom_args = od_config.custom_pipeline_args or {}
        default_tokenizer = pipeline_helpers.resolve_tokenizer_source(self.model_dir, DEFAULT_PI0_TOKENIZER)
        self.tokenizer_source = str(custom_args.get("tokenizer", default_tokenizer))

        # Torch dtype/device from od_config.
        self._torch_dtype = self._resolve_dtype(od_config)
        self._device = pipeline_helpers.resolve_device()

        self.tokenizer = self._load_tokenizer()
        self.model = self._initialize_model()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _build_config(self, od_config: OmniDiffusionConfig) -> Pi0Config:
        """Build Pi0Config from deploy-yaml model_config, falling back to the
        checkpoint's config.json (raw LeRobot format)."""
        if od_config.model_config:
            config = Pi0Config.from_model_config(dict(od_config.model_config))
            # The deploy yaml is authoritative, but must not silently drop
            # checkpoint-derived fields it omits. Backfill the camera order and
            # any normalization stats the yaml didn't specify.
            if self.model_dir and (not config.image_feature_keys or config.norm_stats is None):
                ckpt = Pi0Config.from_pretrained(self.model_dir)
                if not config.image_feature_keys:
                    config.image_feature_keys = ckpt.image_feature_keys
                    if not config.input_features:
                        config.input_features = ckpt.input_features
                if config.norm_stats is None:
                    config.norm_stats = ckpt.norm_stats
            return config
        if self.model_dir:
            return Pi0Config.from_pretrained(self.model_dir)
        return Pi0Config()

    @staticmethod
    def _resolve_dtype(od_config: OmniDiffusionConfig) -> torch.dtype:
        dt = od_config.dtype
        resolved = dt if isinstance(dt, torch.dtype) else getattr(torch, str(dt).split(".")[-1], None)
        if resolved not in SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported π0 dtype: {dt!r}. Supported: "
                f"{', '.join(sorted(str(dtype).split('.')[-1] for dtype in SUPPORTED_DTYPES))}."
            )
        return resolved

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.tokenizer_source)

    def _initialize_model(self) -> Pi0ForActionPrediction:
        model = Pi0ForActionPrediction(self.config)
        _set_inference_dtype(model, self._torch_dtype)
        if pipeline_helpers.has_safetensors_checkpoint(self.model_dir):
            self._load_checkpoint(model)
        else:
            logger.info("Pi0Pipeline: no model.safetensors under %s; using random init.", self.model_dir)
        model.to(device=self._device)
        model.eval()
        return model

    def _load_checkpoint(self, model: Pi0ForActionPrediction) -> None:
        import safetensors.torch

        path = os.path.join(self.model_dir, "model.safetensors")
        logger.info("Pi0Pipeline: loading π0 weights from %s", path)
        state = safetensors.torch.load_file(path)
        model.load_weights(list(state.items()))

    # ------------------------------------------------------------------
    # Framework weight-loading hook
    # ------------------------------------------------------------------
    def load_weights(self, weights=()):  # noqa: D401
        """No-op for the diffusion loader: π0 self-loads its checkpoint in
        ``__init__`` (the kernel's ``load_weights`` handles the LeRobot remaps).
        We expose no ``weights_sources``, so the loader passes an empty iterator
        here; returning ``None`` skips its strict unloaded-weights check.
        """
        for _ in weights:
            pass
        return None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest, **kwargs) -> DiffusionOutput:
        num_steps = pipeline_helpers.resolve_num_inference_steps(req.sampling_params)
        extra_args = getattr(req.sampling_params, "extra_args", None) or {}
        robot_obs = extra_args.get("robot_obs")

        if robot_obs is None:
            # Dummy warmup path (no obs): return zeros so engine warmup/capture
            # doesn't crash. Mirrors DreamZero's dummy-run handling.
            first_prompt = req.prompts[0] if req.prompts else ""
            prompt = first_prompt if isinstance(first_prompt, str) else (first_prompt.get("prompt") or "")
            if prompt == "dummy run" or num_steps == 1:
                logger.info("Pi0Pipeline: dummy warmup request without robot_obs — returning zeros.")
                return DiffusionOutput(
                    output={
                        "actions": np.zeros(
                            (self.config.chunk_size, self.config.max_action_dim),
                            dtype=np.float32,
                        )
                    },
                )
            return DiffusionOutput(
                error="Pi0Pipeline.forward requires sampling_params.extra_args['robot_obs'].",
            )

        images, image_masks, lang_tokens, lang_masks, state = build_model_inputs(
            robot_obs, self.config, self.tokenizer, self._device
        )

        # State normalization (identity for pi0_base) inside the model.
        state = self.model._normalize_state(state)

        noise = extra_args.get("noise")
        if noise is not None and not isinstance(noise, torch.Tensor):
            noise = torch.as_tensor(noise, dtype=torch.float32, device=self._device)
        elif isinstance(noise, torch.Tensor):
            noise = noise.to(device=self._device, dtype=torch.float32)

        actions = self.model.sample_actions(
            images=images,
            image_masks=image_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
            noise=noise,
            num_steps=num_steps,
        )
        actions = self.model._unnormalize_actions(actions)

        # (B=1, horizon, action_dim) → (horizon, action_dim) numpy for the wire.
        actions_np = actions.squeeze(0).float().cpu().numpy()

        # Note: post-processing is applied engine-side via the registry
        # (_DIFFUSION_POST_PROCESS_FUNCS), so we don't attach it here (a local
        # closure wouldn't survive the orchestrator's multiprocess pickling).
        return DiffusionOutput(output={"actions": actions_np})
