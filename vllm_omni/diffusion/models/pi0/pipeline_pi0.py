# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
from vllm_omni.diffusion.models.pi0.config import Pi0Config
from vllm_omni.diffusion.models.pi0.modeling_pi0 import Pi0ForActionPrediction
from vllm_omni.diffusion.models.pi0.processor_pi0 import build_model_inputs
from vllm_omni.diffusion.models.pi0_pipeline_config import PI0_PIPELINE as PI0_PIPELINE
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)

# Default tokenizer for the PaliGemma prefix (matches LeRobot Pi0).
DEFAULT_PI0_TOKENIZER = "google/paligemma-3b-pt-224"


def _pi0_post_process(x):
    """Module-level identity post-process (picklable across the orchestrator's
    multiprocess boundary — a local closure is not)."""
    return x


def get_pi0_post_process_func(od_config: OmniDiffusionConfig):
    """π0 returns actions directly; post-processing is identity. The diffusion
    engine resolves this via the registry (``_DIFFUSION_POST_PROCESS_FUNCS``) and
    applies it engine-side, so the pipeline does NOT attach it to DiffusionOutput.
    """
    del od_config
    return _pi0_post_process


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
        self.model_dir = self._resolve_model_dir(od_config.model)
        self.config = self._build_config(od_config)

        custom_args = od_config.custom_pipeline_args or {}
        self.tokenizer_source = str(custom_args.get("tokenizer", self._resolve_tokenizer_source()))

        # Torch dtype/device from od_config.
        self._torch_dtype = self._resolve_dtype(od_config)
        self._device = self._resolve_device(od_config)

        self.tokenizer = self._load_tokenizer()
        self.model = self._initialize_model()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_model_dir(model: str | None) -> str | None:
        """Return a local directory for ``model``; download an HF repo id if needed.

        Limits the snapshot to the files π0 actually reads (config, weights,
        tokenizer) so a repo id doesn't pull unrelated artifacts.
        """
        if not model:
            return None
        if os.path.isdir(model):
            return model
        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id=model,
            allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
        )

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

    def _resolve_tokenizer_source(self) -> str:
        """Prefer the checkpoint dir if it ships tokenizer files; else PaliGemma."""
        if self.model_dir and os.path.isdir(self.model_dir):
            if os.path.exists(os.path.join(self.model_dir, "tokenizer_config.json")):
                return self.model_dir
        return DEFAULT_PI0_TOKENIZER

    @staticmethod
    def _resolve_dtype(od_config: OmniDiffusionConfig) -> torch.dtype:
        dt = od_config.dtype
        if isinstance(dt, torch.dtype):
            return dt
        return getattr(torch, str(dt).split(".")[-1], torch.float32)

    @staticmethod
    def _resolve_device(od_config: OmniDiffusionConfig) -> torch.device:
        from vllm_omni.diffusion.distributed.utils import get_local_device

        try:
            return get_local_device()
        except Exception:  # noqa: BLE001
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.tokenizer_source)

    def has_real_checkpoint(self) -> bool:
        return bool(self.model_dir) and os.path.exists(os.path.join(self.model_dir, "model.safetensors"))

    def _initialize_model(self) -> Pi0ForActionPrediction:
        model = Pi0ForActionPrediction(self.config)
        if self.has_real_checkpoint():
            self._load_checkpoint(model)
        else:
            logger.info("Pi0Pipeline: no model.safetensors under %s; using random init.", self.model_dir)
        model.to(device=self._device, dtype=self._torch_dtype)
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
        extra_args = getattr(req.sampling_params, "extra_args", None) or {}
        robot_obs = extra_args.get("robot_obs")

        if robot_obs is None:
            # Dummy warmup path (no obs): return zeros so engine warmup/capture
            # doesn't crash. Mirrors DreamZero's dummy-run handling.
            first_prompt = req.prompts[0] if req.prompts else ""
            prompt = first_prompt if isinstance(first_prompt, str) else (first_prompt.get("prompt") or "")
            num_steps = getattr(req.sampling_params, "num_inference_steps", None)
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

        num_steps = extra_args.get("num_inference_steps")

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
