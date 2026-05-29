# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stable Audio 3 text-to-audio pipeline for vLLM-Omni.

PORT_FROM: stable_audio_3/model.py StableAudioModel (top-level facade)
           stable_audio_3/factory.py (component assembly)
           stable_audio_3/interface/diffusion_cond.py generate_cond (inference loop)
           stable_audio_3/cli.py (user-facing flow)

This pipeline IS the bridge between vllm-omni's engine and SA3's internals.

Engine contract (from .claude/skills/add-diffusion-model/SKILL.md):
  - subclass nn.Module + SupportAudioOutput + (later) SupportsComponentDiscovery
  - implement forward(req: OmniDiffusionRequest) → DiffusionOutput
  - define self.weights_sources so DiffusersPipelineLoader knows where to load
  - implement load_weights() entry point

Component layout (mirrors upstream factory.py's assembly):
  self.diffusion = ConditionedDiffusionModelWrapper(...)
       ↳ .conditioner = MultiConditioner({"prompt": T5GemmaConditioner,
                                            "seconds_total": NumberConditioner, ...})
       ↳ .model       = DiTWrapper(DiffusionTransformer(...))
       ↳ .pretransform = AutoencoderPretransform(AudioAutoencoder(SAMEEncoder + SAMEDecoder
                                                  + SoftNormBottleneck + PatchedPretransform))
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import Any, ClassVar

import torch
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import (
    SupportAudioOutput,
    SupportsComponentDiscovery,
)
from vllm_omni.diffusion.models.stable_audio_3.conditioners import (
    MultiConditioner,
    NumberConditioner,
    T5GemmaConditioner,
)
from vllm_omni.diffusion.models.stable_audio_3.diffusion_wrapper import (
    ConditionedDiffusionModelWrapper,
    DiTWrapper,
)
from vllm_omni.diffusion.models.stable_audio_3.same_autoencoder import (
    AudioAutoencoder,
    AutoencoderPretransform,
    PatchedPretransform,
    SAMEDecoder,
    SAMEEncoder,
    SoftNormBottleneck,
)
from vllm_omni.diffusion.models.stable_audio_3.sampling import (
    build_schedule,
    sample_diffusion,
)
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import (
    DiffusionPipelineProfilerMixin,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)


def get_stable_audio_3_post_process_func(od_config: OmniDiffusionConfig):
    """Audio post-process factory — registered in registry.py."""

    def post_process_func(audio: torch.Tensor, output_type: str = "np"):
        if output_type in ("latent", "pt"):
            return audio
        return audio.cpu().float().numpy()

    return post_process_func


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class StableAudio3Pipeline(
    nn.Module,
    SupportAudioOutput,
    SupportsComponentDiscovery,
    DiffusionPipelineProfilerMixin,
):
    """Top-level text-to-audio pipeline for Stable Audio 3.

    Initial scope (v1): text-to-audio only. audio-to-audio editing and
    inpainting are deferred — slots reserved by ConditionedDiffusionModelWrapper
    but no user-facing entry points.
    """

    # Engine routing markers
    support_audio_output: ClassVar[bool] = True
    audio_sample_rate: ClassVar[int] = 44100
    max_audio_seconds: ClassVar[float] = 380.0  # SA3 Medium upper bound per issue #3787

    # Component discovery for CPU offload / layerwise offload
    # (per SupportsComponentDiscovery protocol; dotted paths supported)
    _dit_modules: ClassVar[list[str]] = ["diffusion_model.model"]  # DiffusionTransformer (1.4B DiT)
    _encoder_modules: ClassVar[list[str]] = ["conditioner"]  # MultiConditioner → T5Gemma + Number
    _vae_modules: ClassVar[list[str]] = ["pretransform.model"]  # AudioAutoencoder (SAME)
    _resident_modules: ClassVar[list[str]] = []  # nothing extra to pin

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        self.dtype = getattr(od_config, "dtype", torch.float16)

        # ----------------------------------------------------------------
        # Load model_config.json from the checkpoint root.
        # SA3 ships {repo}/model_config.json + {repo}/model.safetensors.
        # ----------------------------------------------------------------
        model_path = od_config.model
        local_files_only = os.path.exists(model_path)
        config_path = os.path.join(model_path, "model_config.json")
        with open(config_path) as f:
            sa3_config = json.load(f)
        self.sa3_config = sa3_config

        sample_rate = sa3_config.get("sample_rate", 44100)
        self.sample_rate = sample_rate

        # ----------------------------------------------------------------
        # Build the assembly tree (mirrors factory.create_diffusion_cond_from_config).
        # PORT_FROM: factory.py:13-90
        # ----------------------------------------------------------------
        model_cfg = sa3_config["model"]
        diffusion_cfg = model_cfg["diffusion"]
        diffusion_objective = diffusion_cfg.get("diffusion_objective", "v")
        diffusion_model_cfg = diffusion_cfg["config"]
        modular_local_cond_configs = diffusion_cfg.get("modular_local_cond_configs", [])

        # DiTWrapper(DiffusionTransformer(**config))
        # PORT_FROM: factory.py:22-26
        self.diffusion_model = DiTWrapper(
            diffusion_objective=diffusion_objective,
            modular_local_cond_configs=modular_local_cond_configs,
            **diffusion_model_cfg,
        )

        # Pretransform = AutoencoderPretransform(AudioAutoencoder(...))
        # PORT_FROM: factory.py:97-156 (create_pretransform_from_config)
        self.pretransform = self._build_pretransform(model_cfg, sample_rate)

        # MultiConditioner({id: T5Gemma | Number, ...})
        # PORT_FROM: factory.py:115-156
        self.conditioner = self._build_conditioner(model_cfg["conditioning"])

        # Top-level wrapper
        # PORT_FROM: factory.py:43-90
        min_input_length = self.pretransform.downsampling_ratio * self.diffusion_model.model.patch_size
        self.diffusion = ConditionedDiffusionModelWrapper(
            self.diffusion_model,
            self.conditioner,
            min_input_length=min_input_length,
            sample_rate=sample_rate,
            cross_attn_cond_ids=diffusion_cfg.get("cross_attention_cond_ids", []),
            global_cond_ids=diffusion_cfg.get("global_cond_ids", []),
            input_concat_ids=diffusion_cfg.get("input_concat_ids", []),
            local_add_cond_ids=diffusion_cfg.get("local_add_cond_ids", []),
            modular_local_cond_ids=[c["id"] for c in modular_local_cond_configs],
            prepend_cond_ids=diffusion_cfg.get("prepend_cond_ids", []),
            pretransform=self.pretransform,
            io_channels=model_cfg.get("io_channels"),
            distribution_shift_options=diffusion_cfg.get("distribution_shift_options"),
            sampling_distribution_shift_options=diffusion_cfg.get("sampling_distribution_shift_options"),
            mask_padding_attention=diffusion_cfg.get("mask_padding_attention", False),
            use_effective_length_for_schedule=diffusion_cfg.get("use_effective_length_for_schedule", False),
            diffusion_objective=diffusion_objective,
        )

        # ----------------------------------------------------------------
        # Weights sources — vllm-omni loader will populate self.diffusion after init.
        # SA3 ships weights at the model root (no `transformer/` subfolder).
        # Pattern 2 (BAGEL-style): subfolder=None, prefix="".
        # ----------------------------------------------------------------
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model_path,
                subfolder=None,
                revision=None,
                prefix="",
                fall_back_to_pt=False,
            ),
        ]

        # CFG / timestep tracking (for cache backends + profiler)
        self._guidance_scale: float | None = None
        self._num_timesteps: int | None = None
        self._current_timestep: torch.Tensor | None = None
        self._cache_backend = None

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler,
        )

    # ------------------------------------------------------------------ helpers

    def _build_pretransform(self, model_cfg: dict, sample_rate: int) -> AutoencoderPretransform:
        """Mirror factory.create_pretransform_from_config + create_autoencoder_from_config.

        PORT_FROM: stable-audio-3 factory.py:79-115 (verbatim adaptation).
        """
        pretransform_block = model_cfg["pretransform"]
        # AE-only configs (SAME-S/SAME-L) have encoder/decoder at top; full SA3
        # configs nest them inside pretransform.config.
        if "encoder" in pretransform_block:
            ae_cfg = pretransform_block
        else:
            ae_cfg = pretransform_block["config"]

        encoder = SAMEEncoder(**ae_cfg["encoder"]["config"])
        decoder = SAMEDecoder(**ae_cfg["decoder"]["config"])
        bottleneck = SoftNormBottleneck(**ae_cfg["bottleneck"]["config"])
        patch_pretransform = PatchedPretransform(**ae_cfg["pretransform"]["config"])

        autoencoder = AudioAutoencoder(
            encoder,
            decoder,
            io_channels=ae_cfg.get("io_channels"),
            latent_dim=ae_cfg.get("latent_dim"),
            downsampling_ratio=ae_cfg.get("downsampling_ratio"),
            sample_rate=sample_rate,
            bottleneck=bottleneck,
            pretransform=patch_pretransform,
            in_channels=ae_cfg.get("in_channels"),
            out_channels=ae_cfg.get("out_channels"),
            soft_clip=ae_cfg["decoder"].get("soft_clip", False),
        )

        return AutoencoderPretransform(
            autoencoder,
            chunked=pretransform_block.get("chunked", False),
            iterate_batch=pretransform_block.get("iterate_batch", False),
        )

    def _build_conditioner(self, conditioning_config: dict) -> MultiConditioner:
        """Build MultiConditioner from config.

        PORT_FROM: factory.py:115-156
        """
        cond_dim = conditioning_config["cond_dim"]
        default_keys = conditioning_config.get("default_keys", {})
        pre_encoded_keys = conditioning_config.get("pre_encoded_keys", [])

        conditioners: dict[str, nn.Module] = {}
        for cinfo in conditioning_config["configs"]:
            cid = cinfo["id"]
            ctype = cinfo["type"]
            ccfg = {"output_dim": cond_dim, **cinfo["config"]}
            if ctype == "t5gemma":
                # Local checkpoint: load T5Gemma from {model}/{subfolder} instead of
                # the HF repo_id (which would require network access).
                if os.path.exists(self.od_config.model):
                    ccfg.setdefault("model_path", self.od_config.model)
                conditioners[cid] = T5GemmaConditioner(**ccfg)
            elif ctype == "number":
                conditioners[cid] = NumberConditioner(**ccfg)
            else:
                raise ValueError(f"Unknown conditioner type: {ctype}")

        return MultiConditioner(conditioners, default_keys=default_keys, pre_encoded_keys=pre_encoded_keys)

    # ------------------------------------------------------------------ inference entry

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """vLLM-omni engine entry — text → audio.

        Mirrors stable-audio-3 model.py StableAudioModel.generate() for the
        V1 text-to-audio path. Inpainting / audio-to-audio / RF-inversion are
        deferred.

        Steps:
          1. Extract prompt + duration from OmniDiffusionRequest
          2. Build conditioning batch dicts (per-prompt {prompt, seconds_*})
          3. Run MultiConditioner → conditioning_tensors dict
          4. Compute latent_sample_size from requested duration
          5. Build initial noise [B, io_channels, latent_sample_size]
          6. Route conditioning through ConditionedDiffusionModelWrapper.get_conditioning_inputs
          7. Build sigma schedule (sigma_max=1.0, RF convention)
          8. Run sample_diffusion with a closure that calls DiTWrapper with bound conditioning
          9. Decode latents → audio via pretransform
         10. Truncate to requested duration → DiffusionOutput
        """
        # --- 1. Extract request params -----------------------------------
        sampling_params = req.sampling_params
        steps: int = sampling_params.num_inference_steps or 8
        cfg_scale: float = sampling_params.guidance_scale or 6.0
        seed = sampling_params.seed

        extra = getattr(sampling_params, "extra_args", None) or {}
        audio_start_s: float = extra.get("audio_start_in_s", 0.0)
        audio_end_s: float | None = extra.get("audio_end_in_s")
        duration: float = (audio_end_s - audio_start_s) if audio_end_s is not None else 30.0
        sampler_type: str = extra.get("sampler_type", "dpmpp-3m-sde")

        if duration > self.max_audio_seconds:
            raise ValueError(
                f"Requested duration {duration:.1f}s exceeds SA3 Medium max of "
                f"{self.max_audio_seconds:.0f}s",
            )

        # --- 2. Parse prompts into per-batch dicts -----------------------
        prompts = req.prompts or []
        prompt_list: list[str] = []
        negative_prompt_list: list[str | None] = []
        for p in prompts:
            if isinstance(p, str):
                prompt_list.append(p)
                negative_prompt_list.append(None)
            else:
                prompt_list.append(p.get("prompt") or "")
                negative_prompt_list.append(p.get("negative_prompt"))

        if not prompt_list:
            raise ValueError("At least one prompt is required")

        batch_size = len(prompt_list)
        device = self.device

        # Build the per-element conditioning dicts the MultiConditioner consumes.
        conditioning_batch: list[dict[str, Any]] = [
            {"prompt": p, "seconds_start": audio_start_s, "seconds_total": duration}
            for p in prompt_list
        ]
        has_negative = any(np_ is not None for np_ in negative_prompt_list)
        negative_conditioning_batch: list[dict[str, Any]] | None = (
            [
                {
                    "prompt": np_ or "",
                    "seconds_start": audio_start_s,
                    "seconds_total": duration,
                }
                for np_ in negative_prompt_list
            ]
            if has_negative
            else None
        )

        # --- 3. Run conditioner -------------------------------------------
        conditioning_tensors = self.conditioner(conditioning_batch, device=device)
        negative_conditioning_tensors: dict[str, Any] = (
            self.conditioner(negative_conditioning_batch, device=device)
            if negative_conditioning_batch is not None
            else {}
        )

        # --- 4. Compute latent shape -------------------------------------
        downsampling_ratio = self.pretransform.downsampling_ratio
        audio_sample_size = int(duration * self.sample_rate)
        # Round UP so audio_sample_size is a multiple of downsampling_ratio
        audio_sample_size = (
            (audio_sample_size + downsampling_ratio - 1) // downsampling_ratio
        ) * downsampling_ratio
        latent_sample_size = audio_sample_size // downsampling_ratio

        # --- 5. Initial noise -------------------------------------------
        if seed is not None:
            torch.manual_seed(seed)
        noise = torch.randn(
            [batch_size, self.diffusion.io_channels, latent_sample_size],
            device=device,
        )

        # --- 5b. Inject default inpaint conditioning ---------------------
        # SA3 always routes inpaint_mask + inpaint_masked_input through
        # local_add_cond. For pure text-to-audio: all-zero mask = generate
        # everything, zero masked input = no source audio.
        # PORT_FROM: model.py:280-297
        io_channels = self.diffusion.io_channels
        inpaint_mask = torch.zeros((batch_size, 1, latent_sample_size), device=device)
        inpaint_masked_input = torch.zeros(
            (batch_size, io_channels, latent_sample_size), device=device
        )
        conditioning_tensors["inpaint_mask"] = [inpaint_mask]
        conditioning_tensors["inpaint_masked_input"] = [inpaint_masked_input]
        if negative_conditioning_tensors:
            negative_conditioning_tensors["inpaint_mask"] = [inpaint_mask]
            negative_conditioning_tensors["inpaint_masked_input"] = [inpaint_masked_input]

        # --- 6. Route conditioning into the 6 ConditionedDiffusion buckets
        cond_inputs = self.diffusion.get_conditioning_inputs(conditioning_tensors)
        if negative_conditioning_tensors:
            neg_inputs = self.diffusion.get_conditioning_inputs(
                negative_conditioning_tensors, negative=True,
            )
            cond_inputs.update(neg_inputs)
        cond_inputs["cfg_scale"] = cfg_scale

        # Bind conditioning into a model closure the sampler can call as model(x, t)
        def model_fn(x: torch.Tensor, t: torch.Tensor, **extra_args) -> torch.Tensor:
            return self.diffusion.model(x, t, **cond_inputs, **extra_args)

        # --- 7. Sigma schedule -------------------------------------------
        # Optional sampling distribution shift (per checkpoint config).
        dist_shift = getattr(self.diffusion, "sampling_dist_shift", None)
        sigmas = build_schedule(
            steps=steps,
            sigma_max=1.0,
            dist_shift=dist_shift,
            fallback_seq_len=latent_sample_size,
            device=device,
        )

        # --- 8. Run sampler ---------------------------------------------
        self._guidance_scale = cfg_scale
        self._num_timesteps = steps
        latents = sample_diffusion(
            model_fn, noise, sigmas, sampler_type=sampler_type,
        )
        self._current_timestep = None

        # --- 9. Decode latents → audio ----------------------------------
        # The DiT runs in the engine compute dtype (e.g. bf16) but the SAME
        # autoencoder runs in fp32; cast latents to the VAE dtype before decode.
        vae_dtype = next(self.pretransform.parameters()).dtype
        audio = self.pretransform.decode(latents.to(vae_dtype))

        # --- 10. Trim + wrap ---------------------------------------------
        target_samples = int(duration * self.sample_rate)
        if audio.shape[-1] > target_samples:
            audio = audio[..., :target_samples]

        return DiffusionOutput(
            output=audio,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    # ------------------------------------------------------------------ weight loading

    @staticmethod
    def _remap_weight_name(name: str) -> str:
        """Upstream checkpoint name -> vllm-omni param name.

        The checkpoint is the state_dict of upstream ConditionedDiffusionModelWrapper,
        whose DiT lives under ``model.`` (wrapper.model = DiTWrapper). Here the same
        DiTWrapper is registered as ``diffusion_model``, so the only rename needed is the
        leading ``model.`` -> ``diffusion_model.``. ``pretransform.*`` and
        ``conditioner.*`` match verbatim (verified: 997/997 tensors, zero shape mismatch).
        """
        if name.startswith("model."):
            return "diffusion_model." + name[len("model."):]
        return name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the upstream SA3 checkpoint into the pipeline module tree.

        Direct remap + copy (upstream ``copy_state_dict`` style) rather than
        AutoWeightsLoader: the upstream key namespace (``model.model.*``,
        ``pretransform.model.*``, ``conditioner.*``) does not align with
        AutoWeightsLoader's attribute-based routing, and T5Gemma is loaded separately in
        ``__init__`` (hidden from ``named_parameters`` when frozen).
        """
        params = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        loaded: set[str] = set()
        for name, tensor in weights:
            mapped = self._remap_weight_name(name)
            target = params.get(mapped)
            if target is not None:
                default_weight_loader(target, tensor)
                loaded.add(mapped)
                continue
            buf = buffers.get(mapped)
            if buf is not None:
                buf.copy_(tensor.to(buf.dtype))
                loaded.add(mapped)
        return loaded
