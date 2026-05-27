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
from vllm.model_executor.models.utils import AutoWeightsLoader

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

        PORT_FROM: factory.py:93-113
        """
        # TODO(stable-audio-3): replicate factory.create_autoencoder_from_config logic.
        # Reads:
        #   ae_cfg = model_cfg["pretransform"]["config"]
        #   encoder = SAMEEncoder(**ae_cfg["encoder"]["config"])
        #   decoder = SAMEDecoder(**ae_cfg["decoder"]["config"])
        #   bottleneck = SoftNormBottleneck(**ae_cfg["bottleneck"]["config"])
        #   patch_pretransform = PatchedPretransform(**ae_cfg["pretransform"]["config"])
        #   ae = AudioAutoencoder(encoder, decoder, ..., pretransform=patch_pretransform,
        #                          bottleneck=bottleneck, sample_rate=sample_rate)
        #   return AutoencoderPretransform(ae, chunked=model_cfg["pretransform"].get("chunked", False))
        raise NotImplementedError

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
                conditioners[cid] = T5GemmaConditioner(**ccfg)
            elif ctype == "number":
                conditioners[cid] = NumberConditioner(**ccfg)
            else:
                raise ValueError(f"Unknown conditioner type: {ctype}")

        return MultiConditioner(conditioners, default_keys=default_keys, pre_encoded_keys=pre_encoded_keys)

    # ------------------------------------------------------------------ inference entry

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """vLLM-omni engine entry — text → audio.

        Maps OmniDiffusionRequest fields onto upstream's generate_cond inputs.
        PORT_FROM: stable_audio_3/interface/diffusion_cond.py generate_cond
                   stable_audio_3/cli.py main() (the user-facing flow)

        High-level:
          1. Extract prompt + duration from req
          2. Run MultiConditioner to get conditioning_tensors dict
          3. Build initial noise latent [B, latent_dim, T_latent] where
             T_latent = (duration * sample_rate) // downsampling_ratio
          4. Build sigma schedule (default 8 steps)
          5. Loop sampler — sample_diffusion(self.diffusion, x, sigmas)
          6. Decode via self.pretransform.decode(latents)
          7. Return DiffusionOutput(output=audio)
        """
        # TODO(stable-audio-3): full inference loop. See docstring above.
        raise NotImplementedError

    # ------------------------------------------------------------------ weight loading

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """AutoWeightsLoader entry — delegates to each component's load_weights."""
        return AutoWeightsLoader(self).load_weights(weights)
