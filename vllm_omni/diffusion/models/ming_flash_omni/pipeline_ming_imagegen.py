# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright 2025 The vLLM-Omni team.

"""Ming-flash-omni-2.0 imagegen (text-to-image / img2img) diffusion pipeline.

Cross-stage data flow:

    Stage 0 (thinker, llm)           Stage 1 (imagegen, diffusion)
    ────────────────────             ──────────────────────────────
    forward returns                  thinker2imagegen hook slices
    multimodal_output[               final_hidden_states at
      "final_hidden_states"]         <imagePatch> positions,
       ↓                             returns list[dict] with
    shared_memory_connector          {"extra": {"thinker_hidden_states"}}
       ↓                             ──── via OmniMsgpackEncoder ────>
                                     MingImagePipeline:
                                       prepare each request's conditions
                                       run the ZImage denoise algorithm
                                       decode and return request-local outputs
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl import DistributedAutoencoderKL
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.forward_context import (
    get_forward_context,
    is_forward_context_available,
    set_forward_context_ref_latent,
)
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import prefetch_subfolders
from vllm_omni.diffusion.models.ming_flash_omni.byte5_encoder import (
    MingByT5Encoder,
)
from vllm_omni.diffusion.models.ming_flash_omni.condition_encoder import (
    MingConditionEncoder,
)
from vllm_omni.diffusion.models.ming_flash_omni.ming_zimage_transformer import (
    MingZImageTransformer2DModel,
)
from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline, calculate_shift, retrieve_timesteps
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import (
    DiffusionRequestBatch,
    split_diffusion_output_by_request,
)
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)
from vllm_omni.transformers_utils.configs.ming_flash_omni import MingImageGenConfig

logger = logging.getLogger(__name__)


def _ming_image_config(od_config: OmniDiffusionConfig, model_path: str | None = None) -> MingImageGenConfig:
    """Read stage defaults without downloading model weights in preprocessing."""
    # Diffusion configs normally expose the stage-aware HF config through
    # ``tf_model_config``.  Keep the legacy attributes as fallbacks because
    # lightweight callers and older checkpoints may still provide them.
    for config_source in (
        getattr(od_config, "tf_model_config", None),
        getattr(od_config, "hf_config", None),
        getattr(od_config, "model_config", None),
    ):
        if config_source is None:
            continue
        if isinstance(config_source, MingImageGenConfig):
            return config_source
        if callable(getattr(config_source, "to_dict", None)):
            config_source = config_source.to_dict()
        image_gen_config = (
            config_source.get("image_gen_config")
            if isinstance(config_source, Mapping)
            else getattr(config_source, "image_gen_config", None)
        )
        if isinstance(image_gen_config, Mapping):
            image_gen_config = MingImageGenConfig(**dict(image_gen_config))
        if isinstance(image_gen_config, MingImageGenConfig):
            return image_gen_config
        if image_gen_config is not None:
            raise ValueError("Ming image_gen_config must be a configuration object or mapping")
        # The stage may receive the already-selected subconfig wrapped in
        # TransformerConfig instead of the composite Hugging Face config.
        if isinstance(config_source, Mapping) and (
            config_source.get("model_type") == MingImageGenConfig.model_type or "diffusion_c_input_dim" in config_source
        ):
            return MingImageGenConfig(**dict(config_source))

    model_path = model_path or getattr(od_config, "model", None)
    if not model_path or not os.path.isdir(model_path):
        return MingImageGenConfig()
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        try:
            with config_path.open(encoding="utf-8") as config_file:
                checkpoint_config = json.load(config_file)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to read Ming model config at {config_path}: {exc}") from exc
        if not isinstance(checkpoint_config, dict):
            raise ValueError(f"Ming model config at {config_path} must contain a JSON object")
        image_gen_config = checkpoint_config.get("image_gen_config")
        if image_gen_config is not None:
            if not isinstance(image_gen_config, dict):
                raise ValueError(f"Ming image_gen_config in {config_path} must be an object")
            return MingImageGenConfig(**image_gen_config)
    else:
        logger.warning("Ming model config %s is missing; using released-checkpoint defaults", config_path)
    return MingImageGenConfig()


def _resolve_ming_model_config(od_config: OmniDiffusionConfig) -> tuple[str, MingImageGenConfig]:
    """Resolve the same checkpoint root for the pipeline and postprocessor."""
    model_path = od_config.model
    if not model_path:
        raise ValueError("MingImagePipeline requires od_config.model")
    if not os.path.isdir(model_path):
        model_path = download_weights_from_hf_specific(model_path, getattr(od_config, "revision", None), ["*"])
    return model_path, _ming_image_config(od_config, model_path)


def _ming_sampling_values(sampling: OmniDiffusionSamplingParams, cfg: MingImageGenConfig) -> dict[str, Any]:
    """Resolve extra_args, explicit API fields, then checkpoint defaults."""
    extra = sampling.extra_args or {}
    values: dict[str, Any] = {}
    for key, attr, default in (
        ("height", "height", cfg.default_height),
        ("width", "width", cfg.default_width),
        ("steps", "num_inference_steps", cfg.num_inference_steps),
        ("cfg", "guidance_scale", cfg.guidance_scale),
        ("seed", "seed", None),
    ):
        api_value = getattr(sampling, attr, None)
        # OmniDiffusionRequest substitutes 1.0 for an omitted guidance_scale.
        if key == "cfg" and not sampling.guidance_scale_provided:
            api_value = None
        values[key] = next((value for value in (extra.get(key), api_value, default) if value is not None), None)
    for key in ("height", "width", "steps"):
        raw = values[key]
        try:
            value = int(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Ming {key} must be a positive integer, got {raw!r}") from exc
        if isinstance(raw, bool) or value <= 0 or (not isinstance(raw, str) and value != raw):
            raise ValueError(f"Ming {key} must be a positive integer, got {raw!r}")
        values[key] = value
    try:
        values["cfg"] = float(values["cfg"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Ming cfg must be a finite non-negative number") from exc
    if not math.isfinite(values["cfg"]) or values["cfg"] < 0:
        raise ValueError("Ming cfg must be a finite non-negative number")
    return values


class MingImagePipeline(ZImagePipeline):
    """Ming-flash-omni-2.0 text-to-image diffusion pipeline.

    Ming-specific components added on top of the inherited contract:
      * ``condition_encoder`` — Qwen2 connector + proj_in/out + F.normalize×1000
      * ``byte5``             — Optional ByT5 glyph encoder (loaded if checkpoint
                                ships ``byt5/``)
    """

    supports_request_batch = True
    supports_step_execution = True

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",  # noqa: ARG002
    ) -> None:
        # Skip ZImagePipeline.__init__ (it would eagerly load the Z-Image text
        # encoder/tokenizer that Ming replaces with its own condition_encoder).
        nn.Module.__init__(self)

        model_path, image_gen_config = _resolve_ming_model_config(od_config)

        dtype = getattr(od_config, "dtype", torch.bfloat16)
        local_files_only = os.path.exists(model_path)

        self.od_config = od_config
        self._execution_device = get_local_device()
        self.device = self._execution_device  # Ming convention alias
        self._dtype = dtype
        self._interrupt = False

        # Preprocessing and both execution modes use the image-stage defaults.
        self.image_gen_config = image_gen_config
        logger.info(
            "[MingImagePipeline] init: model=%s dtype=%s image_gen_config=%s",
            model_path,
            dtype,
            self.image_gen_config,
        )

        # ----- weights_sources: DiT transformer + VAE.
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model_path,
                subfolder=self.image_gen_config.transformer_subfolder,
                revision=od_config.revision,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model_path,
                subfolder=self.image_gen_config.vae_subfolder,
                revision=od_config.revision,
                prefix="vae.",
            ),
        ]

        prefetch_subfolders(
            model_path,
            [self.image_gen_config.scheduler_subfolder, self.image_gen_config.vae_subfolder],
            local_files_only=local_files_only,
        )

        # ----- Scheduler: load config-only from disk + Ming-specific override.
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path,
            subfolder=self.image_gen_config.scheduler_subfolder,
            local_files_only=local_files_only,
        )
        # Ming forces use_dynamic_shifting=True at runtime regardless of what
        # the checkpoint scheduler_config.json ships.
        self.scheduler.config["use_dynamic_shifting"] = True
        logger.info(
            "[MingImagePipeline] scheduler: %s (use_dynamic_shifting=True)",
            type(self.scheduler).__name__,
        )

        # ----- VAE: DistributedAutoencoderKL.
        vae_config = DistributedAutoencoderKL.load_config(
            model_path, subfolder=self.image_gen_config.vae_subfolder, local_files_only=local_files_only
        )
        self.vae = DistributedAutoencoderKL.from_config(vae_config).to(self._execution_device, dtype=dtype)
        self.vae.eval()

        # ----- DiT transformer.
        self.transformer = MingZImageTransformer2DModel(quant_config=None)

        # Ming brings its own conditioning path — no Z-Image text_encoder /
        # tokenizer.
        self.text_encoder = None
        self.tokenizer = None

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2, do_convert_rgb=True)
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=getattr(od_config, "enable_diffusion_pipeline_profiler", False)
        )

        # ----- Condition encoder (Qwen2 connector + proj_in/out + norm×1000).
        self.condition_encoder = MingConditionEncoder(
            self.image_gen_config,
            thinker_hidden_size=self.image_gen_config.thinker_hidden_size,
            device=self.device,
            dtype=dtype,
        )
        self.condition_encoder.load_from_checkpoint(model_path)
        self.condition_encoder.eval()

        # Optional ByT5 glyph/text encoder. Only loaded when the checkpoint
        # ships a byt5/ subfolder; otherwise byte5_text requests are ignored
        byte5_dir = Path(model_path) / "byt5"
        if byte5_dir.exists():
            self.byte5 = MingByT5Encoder.from_checkpoint(byte5_dir, device=self.device, dtype=dtype)
        else:
            self.byte5 = None
            logger.info("[MingImagePipeline] no byt5/ subfolder at %s; ByT5 enhancement disabled", byte5_dir)

        logger.info("[MingImagePipeline] ready — vae_scale_factor=%d", self.vae_scale_factor)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_byte5_texts(extra: dict, sampling_params) -> list[str]:
        """Resolve byte5 glyph texts.

        Two sources, in order of priority:
        1. `extra["byte5_text"]`: auto-extracted from the user prompt's quoted spans
            by thinker2imagegen (already wrapped as `'Text "<glyph>". '`
            by Ming's get_text_from_prompt).
        2. `sampling_params.extra_args["byte5_text"]`:
            raw strings without the `Text "..."` wrapper are auto-wrapped here
            to match the distribution ByT5 was trained on.
        """
        # Source 1: auto-extracted, already wrapped. Return as-is if non-empty.
        raw = extra.get("byte5_text")
        if isinstance(raw, str):
            raw = [raw]
        if isinstance(raw, list):
            cleaned = [t for t in raw if isinstance(t, str) and t.strip()]
            if cleaned:
                return cleaned

        # Source 2: explicit override — wrap raw strings so the byte5 encoder
        # sees the same ``Text "<glyph>". `` format Ming used during training.
        raw = (getattr(sampling_params, "extra_args", None) or {}).get("byte5_text")
        if isinstance(raw, str):
            raw = [raw]
        if isinstance(raw, list):
            out: list[str] = []
            for t in raw:
                if not isinstance(t, str):
                    continue
                s = t.strip()
                if not s:
                    continue
                # Don't double-wrap if the caller already supplied ``Text "...". ``.
                out.append(s if s.startswith('Text "') else f'Text "{s}". ')
            if out:
                return out
        return []

    @torch.inference_mode()
    def _encode_reference_image(self, ref, height: int, width: int) -> torch.Tensor | None:
        """Turn a PIL/tensor reference image into a VAE latent for ``ref_x``.

        Applies the same shift/scale Ming uses (``(z - shift_factor) * scaling_factor``)
        so the concatenated frame lives in the DiT's latent space.
        """
        if ref is None:
            return None
        if not isinstance(ref, torch.Tensor):
            ref = self.image_processor.preprocess(ref, height, width)
        ref = ref.to(device=self.device, dtype=self.vae.dtype)
        latent = self.vae.encode(ref).latent_dist.mode()
        return (latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor

    @staticmethod
    def _step_prompt_extra_from_prompt(prompt: Any) -> dict[str, Any]:
        if isinstance(prompt, dict):
            return prompt.get("extra") or {}
        if prompt is not None and hasattr(prompt, "_asdict"):
            return MingImagePipeline._step_prompt_extra_from_prompt(prompt._asdict())
        if prompt is not None and hasattr(prompt, "__dict__"):
            return MingImagePipeline._step_prompt_extra_from_prompt(vars(prompt))
        return {}

    def _step_prompt_extra(self, state: StepRequestState) -> dict[str, Any]:
        return self._step_prompt_extra_from_prompt(state.prompt)

    def _step_conditioning(self, state: StepRequestState) -> tuple[torch.Tensor, torch.Tensor]:
        extra = self._step_prompt_extra(state)
        hidden = extra.get("thinker_hidden_states")
        if hidden is None:
            hidden = (state.sampling.extra_args or {}).get("thinker_hidden_states")
        if hidden is None:
            hidden = torch.zeros(
                (self.image_gen_config.num_query_tokens, self.image_gen_config.thinker_hidden_size),
                dtype=self._dtype,
                device=self.device,
            )
            logger.warning(
                "[MingImagePipeline.step] request %s has no thinker hidden states; using zeros",
                state.request_id,
            )
        if not isinstance(hidden, torch.Tensor):
            raise TypeError(f"Ming request {state.request_id!r} thinker_hidden_states must be a Tensor")
        hidden = hidden.to(device=self.device, dtype=self._dtype)
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(0)
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError(f"Ming request {state.request_id!r} hidden states must have shape [1,N,H]")
        if hidden.shape[1] == 0 or hidden.shape[2] != self.image_gen_config.thinker_hidden_size:
            raise ValueError(
                f"Ming request {state.request_id!r} hidden states have invalid shape {tuple(hidden.shape)}"
            )
        positive = self.condition_encoder(hidden)[0]

        negative = extra.get("negative_thinker_hidden_states")
        if negative is None:
            negative_features = self.condition_encoder.zero_negative(positive)
        else:
            if not isinstance(negative, torch.Tensor):
                raise TypeError(f"Ming request {state.request_id!r} negative hidden states must be a Tensor")
            negative = negative.to(device=self.device, dtype=self._dtype)
            if negative.ndim == 2:
                negative = negative.unsqueeze(0)
            if tuple(negative.shape) != tuple(hidden.shape):
                raise ValueError(f"Ming request {state.request_id!r} negative hidden shape does not match positive")
            negative_features = self.condition_encoder(negative)[0]

        byte5_texts = self._resolve_byte5_texts(extra, state.sampling)
        if byte5_texts and self.byte5 is not None:
            byte5 = self.byte5(byte5_texts).to(device=self.device, dtype=self._dtype)
            byte5 = byte5.reshape(1, -1, byte5.shape[-1])[0]
            positive = torch.cat((positive, byte5), dim=0)
            negative_features = torch.cat((negative_features, torch.zeros_like(byte5)), dim=0)
        elif byte5_texts:
            logger.warning(
                "Ming request %s supplies byte5_text but this checkpoint has no ByT5 encoder", state.request_id
            )
        return positive, negative_features

    def _step_sampling_values(self, sampling: OmniDiffusionSamplingParams) -> dict[str, Any]:
        return _ming_sampling_values(sampling, self.image_gen_config)

    @staticmethod
    def _normalize_step_output_count(value: Any, request_id: str | None = None, *, step_execution: bool = True) -> int:
        if value is None:
            return 1
        try:
            count = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Ming num_outputs_per_prompt must be an integer, got {value!r}"
                + (f" for request {request_id!r}" if request_id is not None else "")
            ) from exc
        if isinstance(value, bool) or (not isinstance(value, str) and value != count):
            raise ValueError(
                f"Ming num_outputs_per_prompt must be an integer, got {value!r}"
                + (f" for request {request_id!r}" if request_id is not None else "")
            )
        if count <= 0:
            raise ValueError(
                f"Ming num_outputs_per_prompt must be positive, got {count}"
                + (f" for request {request_id!r}" if request_id is not None else "")
            )
        if step_execution and count != 1:
            raise ValueError(
                "Ming STEP_BATCH currently requires num_outputs_per_prompt=1"
                + (f" for request {request_id!r}" if request_id is not None else "")
            )
        return count

    @staticmethod
    def _normalize_cfg_truncation(value: Any, request_id: str | None = None) -> float | None:
        if value is None:
            return None
        try:
            normalized = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Ming cfg_truncation must be a finite number, got {value!r}"
                + (f" for request {request_id!r}" if request_id is not None else "")
            ) from exc
        if not math.isfinite(normalized):
            raise ValueError(
                f"Ming cfg_truncation must be finite, got {value!r}"
                + (f" for request {request_id!r}" if request_id is not None else "")
            )
        return normalized

    @staticmethod
    def _normalize_cfg_normalize(value: Any, request_id: str | None = None) -> float:
        if value is None:
            return 0.0
        if isinstance(value, bool):
            return float(value)
        try:
            normalized = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Ming cfg_normalize must be a finite non-negative number, got {value!r}"
                + (f" for request {request_id!r}" if request_id is not None else "")
            ) from exc
        if not math.isfinite(normalized) or normalized < 0:
            raise ValueError(
                f"Ming cfg_normalize must be a finite non-negative number, got {value!r}"
                + (f" for request {request_id!r}" if request_id is not None else "")
            )
        return normalized

    @staticmethod
    def _required_state_extra(state: StepRequestState, key: str) -> Any:
        if key not in state.extra:
            raise ValueError(f"Ming request {state.request_id!r} is missing required state.extra[{key!r}]")
        return state.extra[key]

    @staticmethod
    def _set_scheduler_begin_index(scheduler: Any, begin_index: int, request_id: str) -> None:
        set_begin_index = getattr(scheduler, "set_begin_index", None)
        if not callable(set_begin_index):
            raise RuntimeError(
                f"Ming request {request_id!r} requires a scheduler with set_begin_index() "
                "for timestep state synchronization"
            )
        set_begin_index(begin_index)

    def _prepare_generation_context(self, state: StepRequestState, *, step_execution: bool) -> dict[str, Any]:
        """Shared B1/B2 preparation, following Qwen Image's generation context.

        Ming references are a second DiT frame, not a noisy initialization
        image. Both execution modes start from random/explicit float32 latents
        and run the complete schedule, as required by the vendor reference path.
        """
        values = self._step_sampling_values(state.sampling)
        height, width, steps = values["height"], values["width"], values["steps"]
        count = self._normalize_step_output_count(
            state.sampling.num_outputs_per_prompt, state.request_id, step_execution=step_execution
        )
        if step_execution and (state.sampling.sigmas is not None or state.sampling.timesteps is not None):
            raise ValueError("Ming STEP_BATCH does not support custom sigmas or timesteps yet")
        vae_scale = self.vae_scale_factor * 2
        if height % vae_scale or width % vae_scale:
            raise ValueError(f"Ming request {state.request_id!r} dimensions must be divisible by {vae_scale}")
        output_type = state.sampling.output_type or "pil"
        if output_type not in {"pil", "pt", "np", "latent"}:
            raise ValueError(f"Ming request {state.request_id!r} has unsupported output_type={output_type!r}")
        cfg_normalize = self._normalize_cfg_normalize(state.sampling.cfg_normalize, state.request_id)
        cfg_truncation = self._normalize_cfg_truncation(
            (state.sampling.extra_args or {}).get("cfg_truncation", 1.0), state.request_id
        )
        explicit_seed = (state.sampling.extra_args or {}).get("seed")
        generator = state.sampling.generator
        if explicit_seed is not None or (generator is None and values["seed"] is not None):
            generator = torch.Generator(device=self.device).manual_seed(int(values["seed"]))
        prompt_embeds, negative_prompt_embeds = self._step_conditioning(state)
        scheduler = deepcopy(self.scheduler)
        latent_h, latent_w = height // self.vae_scale_factor, width // self.vae_scale_factor
        mu = calculate_shift(
            (latent_h // 2) * (latent_w // 2),
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )
        scheduler.sigma_min = 0.0
        timesteps, _ = retrieve_timesteps(
            scheduler,
            steps,
            self.device,
            timesteps=state.sampling.timesteps,
            sigmas=state.sampling.sigmas,
            mu=mu,
        )
        self._set_scheduler_begin_index(scheduler, 0, state.request_id)
        if len(timesteps) == 0:
            raise ValueError(f"Ming request {state.request_id!r} has an empty denoise schedule")
        reference_image = self._step_prompt_extra(state).get("reference_image")
        reference_latent = self._encode_reference_image(reference_image, height, width)
        if state.sampling.strength is not None:
            logger.warning(
                "Ming request %s ignores strength=%s: reference_image is DiT frame conditioning; "
                "the full denoise schedule is used",
                state.request_id,
                state.sampling.strength,
            )
        latents = self.prepare_latents(
            count,
            self.transformer.in_channels,
            height,
            width,
            torch.float32,
            self.device,
            generator,
            state.sampling.latents,
        ).to(dtype=torch.float32)
        expected_shape = (count, self.transformer.in_channels, latent_h, latent_w)
        if tuple(latents.shape) != expected_shape:
            raise ValueError(f"Ming request {state.request_id!r} latents shape must be {expected_shape}")
        if reference_latent is not None:
            if tuple(reference_latent.shape) != (1, *expected_shape[1:]):
                raise ValueError(f"Ming request {state.request_id!r} reference latent geometry must match latents")
            reference_latent = reference_latent.repeat_interleave(count, dim=0)
        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "latents": latents,
            "timesteps": timesteps,
            "scheduler": scheduler,
            "reference_latent": reference_latent,
            "height": height,
            "width": width,
            "guidance_scale": values["cfg"],
            "cfg_normalize": cfg_normalize,
            "cfg_truncation": cfg_truncation,
            "output_type": output_type,
            "output_count": count,
        }

    def prepare_encode(self, state: StepRequestState, **kwargs: Any) -> StepRequestState:
        """Encode once and populate the runner-owned per-request state."""
        del kwargs
        ctx = self._prepare_generation_context(state, step_execution=True)
        state.prompt_embeds = ctx["prompt_embeds"].unsqueeze(0)
        state.negative_prompt_embeds = ctx["negative_prompt_embeds"].unsqueeze(0)
        state.latents = ctx["latents"]
        state.timesteps = ctx["timesteps"]
        state.scheduler = ctx["scheduler"]
        state.step_index = 0
        state.do_true_cfg = ctx["guidance_scale"] > 0
        state.sampling.cfg_normalize = ctx["cfg_normalize"]
        state.extra.update(
            {
                "ming_guidance_scale": ctx["guidance_scale"],
                "ming_cfg_normalize": ctx["cfg_normalize"],
                "ming_cfg_truncation": ctx["cfg_truncation"],
                "ming_output_type": ctx["output_type"],
                "ming_reference_latent": ctx["reference_latent"],
                "ming_height": ctx["height"],
                "ming_width": ctx["width"],
            }
        )
        return state

    def _build_denoise_kwargs(self, x, timestep, positive, negative, do_true_cfg):
        """Keep the Z-Image list/frame convention at one model boundary."""
        positive_kwargs = {"x": x, "t": timestep, "cap_feats": positive}
        negative_kwargs = {"x": x, "t": timestep, "cap_feats": negative} if do_true_cfg else None
        return positive_kwargs, negative_kwargs

    def denoise_step(
        self,
        input_batch: InputBatch,
        *,
        states: list[StepRequestState] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run exactly one DIT denoise step for the active request batch."""
        del kwargs
        active = list(states if states is not None else input_batch.states)
        if not active:
            raise ValueError("Ming denoise_step received an empty batch")
        if len(active) != len(input_batch.request_ids):
            raise ValueError(
                "Ming denoise_step state count does not match input batch: "
                f"states={len(active)}, request_ids={len(input_batch.request_ids)}"
            )
        active_request_ids = [state.request_id for state in active]
        if active_request_ids != input_batch.request_ids:
            raise ValueError(
                "Ming denoise_step states must match input batch request order: "
                f"states={active_request_ids}, input_batch={input_batch.request_ids}"
            )
        if not isinstance(input_batch.latents, torch.Tensor):
            raise ValueError("Ming denoise_step input_batch.latents must be a Tensor")
        if input_batch.latents.ndim != 4 or input_batch.latents.shape[0] != len(active):
            raise ValueError("Ming denoise_step latents must have shape [requests,C,H,W]")
        if input_batch.timesteps is None or tuple(input_batch.timesteps.shape) != (len(active),):
            raise ValueError("Ming denoise_step requires one timestep per request")
        latent_model_input = input_batch.latents.to(dtype=self._dtype).unsqueeze(2)
        x = list(latent_model_input.unbind(dim=0))
        t = (1000.0 - input_batch.timesteps.to(dtype=torch.float32)) / 1000.0
        if not isinstance(input_batch.prompt_embeds, torch.Tensor):
            raise ValueError(f"Ming denoise_step prompt embeddings are missing for requests {input_batch.request_ids}")
        if input_batch.prompt_embeds.ndim != 3 or input_batch.prompt_embeds.shape[0] != len(active):
            raise ValueError("Ming denoise_step requires one positive embedding row per request")
        positive = list(input_batch.prompt_embeds.unbind(dim=0))
        cfg_scales = {float(self._required_state_extra(state, "ming_guidance_scale")) for state in active}
        if len(cfg_scales) != 1:
            raise ValueError("Ming STEP_BATCH requires one guidance scale per compatible batch")
        cfg_scale = cfg_scales.pop()
        cfg_normalizations = {self._required_state_extra(state, "ming_cfg_normalize") for state in active}
        if len(cfg_normalizations) != 1:
            raise ValueError("Ming STEP_BATCH requires one cfg_normalize value per compatible batch")
        references = [state.extra.get("ming_reference_latent") for state in active]
        if any(reference is not None for reference in references):
            if any(reference is None for reference in references):
                raise ValueError("Ming STEP_BATCH cannot mix reference-image and text-to-image requests")

        truncations = [state.extra.get("ming_cfg_truncation", 1.0) for state in active]
        normalized_truncations = [self._normalize_cfg_truncation(value) for value in truncations]
        if len(set(normalized_truncations)) != 1:
            raise ValueError("Ming STEP_BATCH requires one cfg_truncation per compatible batch")
        cfg_truncation = normalized_truncations[0]
        apply_cfg_rows = [
            bool(state.do_true_cfg and cfg_scale > 0 and (cfg_truncation is None or t_norm <= cfg_truncation))
            for state, t_norm in zip(active, t.tolist(), strict=True)
        ]
        negative = None
        if any(apply_cfg_rows):
            if not isinstance(input_batch.negative_prompt_embeds, torch.Tensor):
                raise ValueError(f"Ming negative prompt embeddings are missing for {input_batch.request_ids}")
            if input_batch.negative_prompt_embeds.shape != input_batch.prompt_embeds.shape:
                raise ValueError("Ming negative prompt embeddings must match positive embedding shape")
            negative = list(input_batch.negative_prompt_embeds.unbind(dim=0))

        row_predictions: dict[int, torch.Tensor] = {}
        for use_cfg in (False, True):
            indices = [index for index, value in enumerate(apply_cfg_rows) if value is use_cfg]
            if not indices:
                continue
            subset_refs = [references[index] for index in indices]
            reference = None
            if subset_refs[0] is not None:
                reference = torch.cat(subset_refs, dim=0)
            subset_x = [x[index] for index in indices]
            subset_t = t[indices]
            subset_positive = [positive[index] for index in indices]
            subset_negative = [negative[index] for index in indices] if use_cfg else None
            positive_kwargs, negative_kwargs = self._build_denoise_kwargs(
                subset_x, subset_t, subset_positive, subset_negative, use_cfg
            )
            previous_ref_latent = get_forward_context().ref_latent if is_forward_context_available() else None
            set_forward_context_ref_latent(reference)
            try:
                subset_prediction = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=use_cfg,
                    true_cfg_scale=cfg_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    cfg_normalize=self._required_state_extra(active[indices[0]], "ming_cfg_normalize"),
                )
            finally:
                set_forward_context_ref_latent(previous_ref_latent)
            if not isinstance(subset_prediction, torch.Tensor):
                raise TypeError("Ming denoise_step expected one tensor prediction")
            expected_shape = (len(indices), *latent_model_input.shape[1:])
            if tuple(subset_prediction.shape) != expected_shape:
                raise ValueError(
                    "Ming denoise_step prediction must have shape [requests,C,1,H,W]: "
                    f"prediction={tuple(subset_prediction.shape)}, expected={expected_shape}"
                )
            for offset, index in enumerate(indices):
                row_predictions[index] = subset_prediction[offset : offset + 1]
        prediction = torch.cat([row_predictions[index] for index in range(len(active))], dim=0)
        return -prediction.squeeze(2)

    def step_scheduler(self, state: StepRequestState, noise_pred: torch.Tensor, **kwargs: Any) -> None:
        """Advance only this request's FlowMatch scheduler and latent state."""
        del kwargs
        timestep = state.current_timestep
        if timestep is None:
            raise ValueError(f"Ming request {state.request_id!r} has no current timestep in step_scheduler")
        if state.scheduler is None or state.latents is None:
            raise ValueError(f"Ming request {state.request_id!r} has not been prepared")
        if noise_pred.shape != state.latents.shape:
            raise ValueError(f"Ming request {state.request_id!r} prediction shape does not match latents")
        state.latents = self.scheduler_step_maybe_with_cfg(
            noise_pred.to(torch.float32),
            timestep,
            state.latents,
            state.do_true_cfg,
            per_request_scheduler=state.scheduler,
        ).to(dtype=torch.float32)
        state.step_index += 1

    @torch.inference_mode()
    def post_decode(self, state: StepRequestState, **kwargs: Any) -> DiffusionOutput:
        """Decode once, after this request's final denoise step."""
        del kwargs
        return self._decode_latents(state.latents, self._required_state_extra(state, "ming_output_type"))

    def _decode_latents(self, latents: torch.Tensor, output_type: str) -> DiffusionOutput:
        """Return raw decoded pixels; engine postprocessing selects PIL/PT/NP per request."""
        if output_type == "latent":
            return DiffusionOutput(output=latents)
        latents = latents.to(self.vae.dtype)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        return DiffusionOutput(output=self.vae.decode(latents, return_dict=False)[0])

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch) -> list[DiffusionOutput]:
        """Run B1 through the same preparation/decode contracts as B2.

        The inherited ZImage forward has a single-request sampling accessor;
        use its denoise algorithm directly, with request-local inputs collated
        here, instead of handing a multi-request batch to that accessor.
        """
        if req.num_reqs == 0:
            return []
        contexts = [
            self._prepare_generation_context(
                StepRequestState(request_id=r.request_id, sampling=r.sampling_params, prompt=r.prompt),
                step_execution=False,
            )
            for r in req.requests
        ]
        first = contexts[0]
        for ctx in contexts[1:]:
            for field in ("height", "width", "guidance_scale", "cfg_normalize", "cfg_truncation", "output_count"):
                if ctx[field] != first[field]:
                    raise ValueError(f"Ming request batch has incompatible {field}: {req.request_ids}")
            if not torch.equal(ctx["timesteps"], first["timesteps"]):
                raise ValueError(f"Ming request batch has incompatible timesteps: {req.request_ids}")
            if (ctx["output_type"] == "latent") != (first["output_type"] == "latent"):
                raise ValueError(f"Ming request batch mixes latent and decoded output: {req.request_ids}")
        references = [ctx["reference_latent"] for ctx in contexts]
        if any(r is not None for r in references) and any(r is None for r in references):
            raise ValueError(f"Ming request batch mixes reference and non-reference requests: {req.request_ids}")
        reference = torch.cat(references, dim=0) if references[0] is not None else None
        count = first["output_count"]
        positive = [ctx["prompt_embeds"] for ctx in contexts for _ in range(count)]
        negative = [ctx["negative_prompt_embeds"] for ctx in contexts for _ in range(count)]
        previous_scheduler = self.scheduler
        previous_ref = get_forward_context().ref_latent if is_forward_context_available() else None
        self.scheduler = first["scheduler"]
        self._guidance_scale = first["guidance_scale"]
        self._cfg_normalization = first["cfg_normalize"]
        self._cfg_truncation = first["cfg_truncation"]
        self._joint_attention_kwargs = None
        self._interrupt = False
        set_forward_context_ref_latent(reference)
        try:
            latents = self.diffuse(
                prompt_embeds=positive,
                negative_prompt_embeds=negative,
                latents=torch.cat([ctx["latents"] for ctx in contexts], dim=0),
                timesteps=first["timesteps"],
                do_true_cfg=first["guidance_scale"] > 0,
                true_cfg_scale=first["guidance_scale"],
                cfg_normalize=first["cfg_normalize"],
                cfg_truncation=first["cfg_truncation"],
            )
            result = self._decode_latents(latents, first["output_type"])
        finally:
            self.scheduler = previous_scheduler
            set_forward_context_ref_latent(previous_ref)
        return split_diffusion_output_by_request(result, req, num_outputs_per_prompt=count)


def get_ming_image_post_process_func(od_config: OmniDiffusionConfig):
    """Convert raw VAE pixels to the requested PIL/PT/NP format, or preserve latents.

    The registered engine hook receives each request's sampling parameters.
    Decoded tensors use [B,3,H,W] in [-1,1]; latent tensors bypass image conversion
    and use the canonical payload envelope for the API's latents field.
    """
    model_path, image_gen_config = _resolve_ming_model_config(od_config)
    vae_config_path = Path(model_path) / image_gen_config.vae_subfolder / "config.json"
    if not vae_config_path.exists():
        logger.warning("Ming VAE config %s is missing; using released-checkpoint scale factor 8", vae_config_path)
        vae_scale_factor = 8
    else:
        try:
            with vae_config_path.open(encoding="utf-8") as config_file:
                vae_cfg = json.load(config_file)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to read Ming VAE config at {vae_config_path}: {exc}") from exc
        if not isinstance(vae_cfg, dict):
            raise ValueError(f"Ming VAE config {vae_config_path} must contain a JSON object")
        block_out_channels = vae_cfg.get("block_out_channels")
        if not isinstance(block_out_channels, list) or not block_out_channels:
            raise ValueError(f"Ming VAE config {vae_config_path} must contain non-empty block_out_channels")
        vae_scale_factor = 2 ** (len(block_out_channels) - 1)

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2, do_convert_rgb=True)

    def post_process_func(images: torch.Tensor, sampling_params=None):
        if sampling_params is not None and getattr(sampling_params, "output_type", None) == "latent":
            return {"payload": {"latents": images}, "metadata": {}}
        output_type = getattr(sampling_params, "output_type", None) or "pil"
        if output_type not in {"pil", "pt", "np"}:
            raise ValueError(f"Ming has unsupported output_type={output_type!r}")
        return {
            "payload": {"image": image_processor.postprocess(images.float(), output_type=output_type)},
            "metadata": {},
        }

    return post_process_func


def get_ming_image_pre_process_func(od_config: OmniDiffusionConfig):
    """Annotate Ming requests with the fields that must be homogeneous in a wave."""

    defaults = _ming_image_config(od_config)
    step_execution = bool(getattr(od_config, "step_execution", False))
    model_path = getattr(od_config, "model", None)
    # Only discard the glyph count when a local checkpoint proves that ByT5
    # is absent. A remote identifier gives preprocessing no such evidence.
    byt5_absent = bool(model_path and os.path.isdir(model_path) and not (Path(model_path) / "byt5").exists())

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        extra = MingImagePipeline._step_prompt_extra_from_prompt(request.prompt)
        sampling = request.sampling_params
        sampling_extra = sampling.extra_args or {}

        hidden = extra.get("thinker_hidden_states")
        if hidden is None:
            hidden = sampling_extra.get("thinker_hidden_states")
        if isinstance(hidden, torch.Tensor):
            hidden_shape = tuple(hidden.shape[-2:]) if hidden.dim() >= 2 else tuple(hidden.shape)
        else:
            hidden_shape = (defaults.num_query_tokens, defaults.thinker_hidden_size)
        values = _ming_sampling_values(sampling, defaults)
        # StepScheduler needs the effective number of steps before admission;
        # normalizing only in prepare_encode leaves it with an omitted value.
        sampling.height, sampling.width = values["height"], values["width"]
        sampling.num_inference_steps = values["steps"]
        sampling.guidance_scale = values["cfg"]
        byte5_count = 0 if byt5_absent else len(MingImagePipeline._resolve_byte5_texts(extra, sampling))
        cfg_truncation = MingImagePipeline._normalize_cfg_truncation(
            sampling_extra.get("cfg_truncation", 1.0),
            request.request_id,
        )
        output_count = MingImagePipeline._normalize_step_output_count(
            sampling.num_outputs_per_prompt,
            request.request_id,
            step_execution=step_execution and request.use_step_execution,
        )
        sampling.num_outputs_per_prompt = output_count
        cfg_normalize = MingImagePipeline._normalize_cfg_normalize(sampling.cfg_normalize, request.request_id)
        sampling.cfg_normalize = cfg_normalize
        output_type = sampling.output_type or "pil"
        if output_type not in {"pil", "pt", "np", "latent"}:
            raise ValueError(f"Ming request {request.request_id!r} has unsupported output_type={output_type!r}")
        if (
            step_execution
            and request.use_step_execution
            and (sampling.sigmas is not None or sampling.timesteps is not None)
        ):
            raise ValueError("Ming STEP_BATCH does not support custom sigmas or timesteps yet")
        request.batch_compatibility_key = (
            "ming_image",
            # Keep reference-image requests in request-local groups. This
            # preserves the established compatibility contract while the
            # direct step path still validates per-row reference latents.
            ("reference", request.request_id) if extra.get("reference_image") is not None else ("t2i",),
            byte5_count,
            hidden_shape,
            values["height"],
            values["width"],
            values["steps"],
            values["cfg"],
            cfg_truncation,
            cfg_normalize,
            output_count,
            "latent" if output_type == "latent" else "decoded",
            tuple(float(t) for t in sampling.timesteps) if sampling.timesteps is not None else None,
            tuple(float(s) for s in sampling.sigmas) if sampling.sigmas is not None else None,
        )
        return request

    return pre_process_func


__all__ = [
    "MingImagePipeline",
    "get_ming_image_pre_process_func",
    "get_ming_image_post_process_func",
]
