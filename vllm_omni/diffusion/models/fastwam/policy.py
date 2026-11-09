# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FastWAM-family policy: checkpoint loading plus observation -> action chunk.

Assets:

* the FastWAM checkpoint directory (``amd/CAE-FastWAM`` or a local copy) holds the
  fine-tuned experts (``policy.pt``: ``{"mot": ..., "proprio_encoder": ...}``), the
  action codec (``action_vae.pt``, CAE only) and the normalisation statistics;
* the Wan2.2 base (``Wan-AI/Wan2.2-TI2V-5B-Diffusers``) provides the VAE encoder,
  the umT5 text encoder and its tokenizer (the fine-tuned checkpoints do not
  include them).

The ``.pt`` files are read with ``torch.load(weights_only=True)``: they contain
only tensors, strings and numbers. Every component is loaded strictly; the only
checkpoint keys skipped are the Wan2.2 video output head (``VIDEO_HEAD_PREFIX``),
which action-only inference never uses.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.models.fastwam.config import REPO_CONFIG, ActionCodecConfig, FastWAMConfig
from vllm_omni.diffusion.models.fastwam.modeling.action_codec import ActionLatentDecoder
from vllm_omni.diffusion.models.fastwam.modeling.fastwam import FastWAMModel
from vllm_omni.diffusion.models.fastwam.modeling.umt5_encoder import (
    WanUMT5Encoder,
    clean_prompt,
    hf_umt5_key_to_wan,
)
from vllm_omni.diffusion.models.fastwam.modeling.wan_vae_encoder import WanVAEFirstFrameEncoder
from vllm_omni.diffusion.models.fastwam.processing_fastwam import (
    FastWAMObservation,
    FastWAMStats,
    build_image,
    load_stats,
    postprocess_actions,
)

logger = init_logger(__name__)

DEFAULT_BASE_MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
_BASE_MODEL_PATTERNS = ["vae/*", "text_encoder/*", "tokenizer/*"]
_PROMPT_CACHE_SIZE = 256
# Wan2.2 video output head: present in every FastWAM policy checkpoint, never used by
# action-only inference, so it is not built and its keys are dropped as expected-unused.
VIDEO_HEAD_PREFIX = "mixtures.video.head."


def resolve_local_dir(model: str, allow_patterns: list[str], revision: str | None = None) -> str:
    """Return a local directory for ``model``, downloading only ``allow_patterns`` from the Hub."""
    if os.path.isdir(model):
        return model
    # Via repo_utils' shared HfApi so the download carries vLLM-Omni's user agent.
    from vllm_omni.transformers_utils.repo_utils import hf_api

    return hf_api().snapshot_download(repo_id=model, allow_patterns=allow_patterns, revision=revision)


def checkpoint_patterns(cfg: FastWAMConfig) -> list[str]:
    patterns = [REPO_CONFIG, cfg.policy_checkpoint, cfg.stats_path.replace("{suite}", "*")]
    if cfg.action_codec_checkpoint:
        patterns.append(cfg.action_codec_checkpoint)
    return patterns


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_pt(path: str, *, mmap: bool) -> dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"FastWAM checkpoint file not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a dict payload, got {type(payload).__name__}.")
    return payload


def _load_strict(module: torch.nn.Module, state: Mapping[str, torch.Tensor], what: str) -> None:
    result = module.load_state_dict(state, strict=False, assign=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            f"{what}: checkpoint does not match the model "
            f"(missing={result.missing_keys[:8]}, unexpected={result.unexpected_keys[:8]})."
        )


def load_text_encoder_state(text_encoder_dir: str) -> dict[str, torch.Tensor]:
    """Read a Diffusers/HF ``UMT5EncoderModel`` checkpoint and rename it for :class:`WanUMT5Encoder`."""
    from safetensors import safe_open

    index_path = os.path.join(text_encoder_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            files = sorted(set(json.load(f)["weight_map"].values()))
    else:
        files = ["model.safetensors"]
    state: dict[str, torch.Tensor] = {}
    for name in files:
        with safe_open(os.path.join(text_encoder_dir, name), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                target = hf_umt5_key_to_wan(key)
                if target is not None:
                    state[target] = handle.get_tensor(key)
    return state


def load_vae_encoder_state(vae_dir: str) -> dict[str, torch.Tensor]:
    """Read the encoder half of a Diffusers ``AutoencoderKLWan`` checkpoint."""
    from safetensors import safe_open

    state: dict[str, torch.Tensor] = {}
    with safe_open(os.path.join(vae_dir, "diffusion_pytorch_model.safetensors"), framework="pt", device="cpu") as f:
        for key in f.keys():
            if WanVAEFirstFrameEncoder.is_encoder_key(key):
                state[key] = f.get_tensor(key)
    return state


class FastWAMPolicy:
    """Owns the FastWAM model, tokenizer, prompt-context cache and per-suite statistics."""

    def __init__(
        self,
        *,
        cfg: FastWAMConfig,
        model_dir: str,
        base_model_dir: str,
        device: torch.device,
        dtype: torch.dtype,
        action_postprocess: str = "libero",
        default_suite: str | None = None,
    ) -> None:
        from transformers import AutoTokenizer

        self.model_dir = model_dir
        self.device = torch.device(device)
        self.action_postprocess = action_postprocess

        if cfg.action_codec_checkpoint:
            codec_payload = _load_pt(os.path.join(model_dir, cfg.action_codec_checkpoint), mmap=False)
            cfg = cfg.with_codec(ActionCodecConfig.from_checkpoint_config(codec_payload["config"]))
        else:
            codec_payload = None
        self.cfg = cfg
        # No silent fallback: a suite-conditioned checkpoint run on the wrong suite's
        # statistics and codec embedding gives wrong actions without any error.
        self.default_suite = default_suite or None
        if self.default_suite is not None:
            self.check_suite(self.default_suite)

        policy_payload = _load_pt(os.path.join(model_dir, cfg.policy_checkpoint), mmap=True)
        cfg.check_policy_state_dict(policy_payload["mot"])

        # Build on the meta device and adopt the checkpoint tensors (no random init
        # of ~11B parameters). The VAE encoder is built for real: it carries
        # non-persistent latent statistics buffers.
        vae_dir = os.path.join(base_model_dir, "vae")
        text_encoder_dir = os.path.join(base_model_dir, "text_encoder")
        vae_encoder = WanVAEFirstFrameEncoder.from_diffusers_config(_read_json(os.path.join(vae_dir, "config.json")))
        _load_strict(vae_encoder, load_vae_encoder_state(vae_dir), "Wan2.2 VAE encoder")
        if vae_encoder.z_dim != cfg.video_in_dim:
            raise ValueError(f"VAE z_dim={vae_encoder.z_dim} != video expert in_dim={cfg.video_in_dim}.")
        with torch.device("meta"):
            text_encoder = WanUMT5Encoder.from_hf_config(_read_json(os.path.join(text_encoder_dir, "config.json")))
            model = FastWAMModel(cfg, vae_encoder=vae_encoder, text_encoder=text_encoder)
        if text_encoder.dim != cfg.text_dim:
            raise ValueError(f"Text encoder width {text_encoder.dim} != FastWAM text_dim {cfg.text_dim}.")
        _load_strict(model.text_encoder, load_text_encoder_state(text_encoder_dir), "umT5 text encoder")
        mot_state = {k: v for k, v in policy_payload["mot"].items() if not k.startswith(VIDEO_HEAD_PREFIX)}
        _load_strict(model.mot, mot_state, f"{cfg.policy_checkpoint} mot")
        _load_strict(model.proprio_encoder, policy_payload["proprio_encoder"], f"{cfg.policy_checkpoint} proprio")
        if model.action_codec is not None and codec_payload is not None:
            _load_strict(
                model.action_codec,
                ActionLatentDecoder.decoder_state_dict(codec_payload["model"]),
                f"{cfg.action_codec_checkpoint} decoder",
            )
        meta = [name for name, p in model.named_parameters() if p.is_meta]
        if meta:
            raise RuntimeError(f"FastWAM parameters left uninitialised: {meta[:8]}")

        # The reference keeps the action codec in float32: cast everything else to
        # the run dtype, then move the codec without a round trip through it.
        codec, model.action_codec = model.action_codec, None
        model.to(device=self.device, dtype=dtype)
        if codec is not None:
            model.action_codec = codec.to(device=self.device, dtype=torch.float32)
        model.eval().requires_grad_(False)
        self.model = model
        del policy_payload

        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(base_model_dir, "tokenizer"))
        self._prompt_cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._stats: dict[str, FastWAMStats] = {}
        logger.info(
            "Loaded FastWAM variant %s (action expert %d/%d, %d %s tokens, shift %.1f%s) on %s in %s.",
            cfg.variant,
            cfg.action_hidden_dim,
            cfg.action_ffn_dim,
            cfg.num_action_tokens,
            "latent" if cfg.action_codec is not None else "action",
            cfg.action_shift,
            "" if cfg.validated else "; NOT validated",
            self.device,
            dtype,
        )

    def check_suite(self, suite: str | None) -> None:
        """Raise ``ValueError`` unless ``suite`` is usable with this checkpoint."""
        if not self.cfg.suite_conditioned_stats:
            return
        if suite is None:
            raise ValueError(
                f"task_suite is required: {self.cfg.variant} normalises and conditions its actions per task suite. "
                f"Send robot_obs['task_suite'] (one of {sorted(self.cfg.suite_ids)}) or pin one suite with "
                "model_config.task_suite in the deploy config."
            )
        if suite not in self.cfg.suite_ids:
            raise ValueError(f"Unknown task_suite {suite!r}; this checkpoint knows {sorted(self.cfg.suite_ids)}.")

    def stats(self, suite: str | None) -> FastWAMStats:
        self.check_suite(suite)
        # Pooled-statistics checkpoints share one file whatever the suite.
        key = (suite or "") if self.cfg.suite_conditioned_stats else ""
        if key not in self._stats:
            path = os.path.join(self.model_dir, self.cfg.stats_file(key))
            self._stats[key] = load_stats(path, action_dim=self.cfg.action_dim, proprio_dim=self.cfg.proprio_dim)
        return self._stats[key]

    def encode_prompt(self, task: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Prompt context, cached per unique instruction (it is a pure function of the text)."""
        prompt = self.cfg.prompt_template.format(task=task)
        cached = self._prompt_cache.get(prompt)
        if cached is not None:
            self._prompt_cache.move_to_end(prompt)
            return cached
        tokens = self.tokenizer(
            [clean_prompt(prompt)],
            padding="max_length",
            truncation=True,
            max_length=self.cfg.text_len,
            add_special_tokens=True,
            return_tensors="pt",
        )
        context = self.model.encode_prompt(tokens.input_ids, tokens.attention_mask)
        self._prompt_cache[prompt] = context
        if len(self._prompt_cache) > _PROMPT_CACHE_SIZE:
            self._prompt_cache.popitem(last=False)
        return context

    def infer(self, obs: FastWAMObservation, *, num_steps: int, generator: torch.Generator | None) -> np.ndarray:
        """One action chunk [action_horizon, action_dim] (float32) for one observation."""
        cfg = self.cfg
        self.check_suite(obs.suite)
        stats = self.stats(obs.suite)
        image = build_image(obs.images, cfg.camera_size)
        proprio = stats.state.normalize(torch.as_tensor(obs.state, dtype=torch.float32).unsqueeze(0))
        context, context_mask = self.encode_prompt(obs.task)
        suite_id = None
        if cfg.action_codec is not None and cfg.action_codec.num_suites:
            if obs.suite is None:
                raise ValueError("A suite-conditioned action codec needs a task_suite.")
            suite_id = cfg.suite_ids[obs.suite]
        actions = self.model.infer_action(
            image,
            context,
            context_mask,
            proprio,
            num_steps=num_steps,
            generator=generator,
            suite_id=suite_id,
        )
        denormalized = stats.action.denormalize(actions.unsqueeze(0)).numpy()[0]
        return postprocess_actions(denormalized, self.action_postprocess)
