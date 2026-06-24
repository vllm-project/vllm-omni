# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
ACE-Step 1.5 Pipeline for vLLM-Omni.

Text-to-music generation using ACE-Step 1.5. Ported from diffusers PR #13095
(``src/diffusers/pipelines/ace_step/pipeline_ace_step.py``), restricted to the
``text2music`` task — cover / repaint / extract / lego / complete tasks and
their helpers (``audio_tokenizer`` / ``audio_token_detokenizer``,
``_build_chunk_mask``, ``prepare_reference_audio_latents``,
``prepare_src_latents``, APG normalised guidance) are intentionally omitted
from this first PR.

The turbo checkpoint distills guidance into the weights, so the default config
runs without CFG (``guidance_scale=1.0``). Sliding-window self-attention
currently requires the SDPA backend (see ``ace_step_transformer.py`` for the
flash-backend caveat).
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable
from typing import ClassVar

import torch
from diffusers import AutoencoderOobleck
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoModel, AutoTokenizer
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.ace_step.ace_step_transformer import (
    AceStepTransformer1DModel,
)
from vllm_omni.diffusion.models.ace_step.modeling_ace_step import (
    AceStepConditionEncoder,
)
from vllm_omni.diffusion.models.interface import SupportAudioOutput
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs

logger = init_logger(__name__)


# Prompt-template constants copied verbatim from diffusers PR #13095 so the
# tokenized inputs match what the ACE-Step text encoder was trained on. Changing
# these strings will silently degrade audio quality.
_SFT_GEN_PROMPT = "# Instruction\n{}\n\n# Caption\n{}\n\n# Metas\n{}<|endoftext|>\n"
_DEFAULT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"


def get_ace_step_post_process_func(
    od_config: OmniDiffusionConfig,
):
    """
    Create post-processing function for ACE-Step audio output.

    Converts raw audio tensor to numpy array for saving.
    """

    def post_process_func(
        audio: torch.Tensor,
        output_type: str = "np",
    ):
        if output_type == "latent":
            return audio
        if output_type == "pt":
            return audio
        # Convert to numpy
        audio_np = audio.cpu().float().numpy()
        return audio_np

    return post_process_func


class AceStepPipeline(nn.Module, SupportAudioOutput, DiffusionPipelineProfilerMixin):
    """
    Pipeline for text-to-music generation using ACE-Step 1.5.

    This pipeline generates music from text prompts (and optional lyrics) using
    the ACE-Step 1.5 model, integrated with vLLM-Omni's diffusion framework.

    Args:
        od_config: OmniDiffusion configuration object
        prefix: Weight prefix for loading (default: "")
    """

    # Picked up by ``supports_audio_output`` in the diffusion engine so the
    # default stage metadata reports ``final_output_type="audio"`` and the
    # ``multimodal_output`` payload includes the sample rate (mirrors the
    # contract introduced for AudioX in #2077).
    support_audio_output: ClassVar[bool] = True
    audio_sample_rate: ClassVar[int] = 48000

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config

        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)

        model = od_config.model
        local_files_only = os.path.exists(model)

        # Set up weights sources. The DiT and condition encoder both come from the
        # diffusers-converted ACE-Step checkpoint (see PR #13095's conversion
        # script). The shared text encoder (Qwen3-Embedding) and VAE
        # (AutoencoderOobleck) load directly via from_pretrained below.
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="condition_encoder",
                revision=None,
                prefix="condition_encoder.",
                fall_back_to_pt=True,
            ),
        ]

        # Load tokenizer (Qwen3-Embedding's tokenizer).
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            subfolder="tokenizer",
            local_files_only=local_files_only,
        )

        # Load text encoder (Qwen3-Embedding-0.6B).
        self.text_encoder = AutoModel.from_pretrained(
            model,
            subfolder="text_encoder",
            torch_dtype=dtype,
            local_files_only=local_files_only,
        ).to(self.device)

        # Load VAE (AutoencoderOobleck — same class Stable Audio uses, kept in fp32).
        self.vae = AutoencoderOobleck.from_pretrained(
            model,
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=local_files_only,
        ).to(self.device)

        # Initialize condition encoder from HF config; weights load via
        # AutoWeightsLoader from the ``condition_encoder/`` subfolder.
        condition_encoder_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, AceStepConditionEncoder)
        self.condition_encoder = AceStepConditionEncoder(**condition_encoder_kwargs)

        # Initialize DiT from HF config; weights load via AutoWeightsLoader
        # from the ``transformer/`` subfolder.
        transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, AceStepTransformer1DModel)
        self.transformer = AceStepTransformer1DModel(od_config=od_config, **transformer_kwargs)

        # Load scheduler. ACE-Step trains with flow matching; the pipeline supplies
        # its own shifted / turbo sigma schedule via ``set_timesteps(sigmas=...)``
        # in ``forward``, so the scheduler is created with the trivial
        # ``num_train_timesteps=1, shift=1.0`` config from the checkpoint's
        # ``scheduler/`` subfolder.
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )

        # Latent frame rate is fixed by the VAE: each VAE latent frame spans
        # ``hop_length`` samples. AutoencoderOobleck exposes hop_length as an
        # attribute (not on its config), so cache it once here.
        self.latents_per_second = self.vae.config.sampling_rate / self.vae.hop_length

        # Variant flag — drives default CFG behaviour in forward.
        self.is_turbo = bool(getattr(self.transformer.config, "is_turbo", False))

        # Cache backend (set by worker if needed)
        self._cache_backend = None

        # Profiler / state used by ``forward``.
        self._guidance_scale = None
        self._num_timesteps = None
        self._current_timestep = None
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    # --------------------------------------------------------------------- #
    #                              prompt helpers                            #
    # --------------------------------------------------------------------- #

    @staticmethod
    def _build_metadata_string(audio_duration: float | None) -> str:
        """Minimal metadata block matching the original handler.

        First PR exposes only ``audio_duration``; bpm / keyscale / timesignature
        default to ``N/A`` (the model handles ``N/A`` gracefully — it's what the
        handler emits when the user does not provide explicit values).
        """
        if audio_duration is not None and audio_duration > 0:
            dur_str = f"{int(audio_duration)} seconds"
        else:
            dur_str = "30 seconds"
        return f"- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n- duration: {dur_str}\n"

    def _format_prompt(
        self,
        prompt: str,
        lyrics: str = "",
        vocal_language: str = "en",
        audio_duration: float = 60.0,
        instruction: str | None = None,
    ) -> tuple[str, str]:
        """Wrap prompt + lyrics in the ACE-Step SFT templates.

        Text gets the SFT generation template (Instruction / Caption / Metas);
        lyrics get a separate (Languages / Lyric) block. Both end with
        ``<|endoftext|>`` to match what the text encoder was trained on.
        """
        if instruction is None:
            instruction = _DEFAULT_INSTRUCTION
        if not instruction.endswith(":"):
            instruction = instruction + ":"

        metas_str = self._build_metadata_string(audio_duration)
        formatted_text = _SFT_GEN_PROMPT.format(instruction, prompt, metas_str)
        formatted_lyrics = f"# Languages\n{vocal_language}\n\n# Lyric\n{lyrics}<|endoftext|>"
        return formatted_text, formatted_lyrics

    def encode_prompt(
        self,
        prompt: str | list[str],
        lyrics: str | list[str],
        device: torch.device,
        vocal_language: str | list[str] = "en",
        audio_duration: float = 60.0,
        instruction: str | None = None,
        max_text_length: int = 256,
        max_lyric_length: int = 2048,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize + encode text prompts and lyrics for the condition encoder.

        Text prompts go through the full Qwen3 text encoder; lyrics use only the
        embedding layer (token lookup), because contextual encoding for lyrics
        happens inside ``AceStepLyricEncoder``.

        Returns:
            A 4-tuple of ``text_hidden_states``, ``text_attention_mask``,
            ``lyric_hidden_states``, ``lyric_attention_mask``.
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(lyrics, str):
            lyrics = [lyrics]
        if isinstance(vocal_language, str):
            vocal_language = [vocal_language] * len(prompt)

        batch_size = len(prompt)
        # Pad lyrics list to batch size (caller may pass one lyric for all).
        while len(lyrics) < batch_size:
            lyrics.append(lyrics[-1] if lyrics else "")

        all_text_strs: list[str] = []
        all_lyric_strs: list[str] = []
        for i in range(batch_size):
            text_str, lyric_str = self._format_prompt(
                prompt=prompt[i],
                lyrics=lyrics[i],
                vocal_language=vocal_language[i],
                audio_duration=audio_duration,
                instruction=instruction,
            )
            all_text_strs.append(text_str)
            all_lyric_strs.append(lyric_str)

        text_inputs = self.tokenizer(
            all_text_strs,
            padding="longest",
            truncation=True,
            max_length=max_text_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        text_attention_mask = text_inputs.attention_mask.to(device).bool()

        lyric_inputs = self.tokenizer(
            all_lyric_strs,
            padding="longest",
            truncation=True,
            max_length=max_lyric_length,
            return_tensors="pt",
        )
        lyric_input_ids = lyric_inputs.input_ids.to(device)
        lyric_attention_mask = lyric_inputs.attention_mask.to(device).bool()

        # Run text through the full encoder (contextual hidden states).
        text_hidden_states = self.text_encoder(input_ids=text_input_ids).last_hidden_state

        # Lyrics use only the embedding layer; the lyric encoder inside the
        # condition encoder handles the contextual transformer pass.
        embed_layer = self.text_encoder.get_input_embeddings()
        lyric_hidden_states = embed_layer(lyric_input_ids)

        return text_hidden_states, text_attention_mask, lyric_hidden_states, lyric_attention_mask

    # --------------------------------------------------------------------- #
    #                          latents / timestep helpers                    #
    # --------------------------------------------------------------------- #

    def prepare_latents(
        self,
        batch_size: int,
        audio_duration: float,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample initial noise latents at the DiT's frame rate.

        Returns a tensor of shape ``(batch_size, latent_length, acoustic_dim)``.
        """
        latent_length = math.ceil(audio_duration * self.latents_per_second)
        acoustic_dim = self.transformer.config.audio_acoustic_hidden_dim

        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (batch_size, latent_length, acoustic_dim)
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    def _get_timestep_schedule(
        self,
        num_inference_steps: int = 8,
        shift: float = 3.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        timesteps: list[float] | None = None,
    ) -> torch.Tensor:
        """ACE-Step's flow-matching schedule, computed from ``num_inference_steps`` and ``shift``.

        Linear in [1, 0] with N+1 points, drop the terminal t=0, then apply the
        flow-matching shift transform. Turbo checkpoints use 8 steps with
        ``shift ∈ {1, 2, 3}`` — those tables are recovered exactly by this
        formula, so no separate lookup is needed.
        """
        if timesteps is not None:
            return torch.tensor(timesteps, device=device, dtype=dtype)

        t = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=dtype)
        if shift != 1.0:
            t = shift * t / (1 + (shift - 1) * t)
        return t[:-1]

    # --------------------------------------------------------------------- #
    #                                  forward                               #
    # --------------------------------------------------------------------- #

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self._guidance_scale is not None and self._guidance_scale > 1.0

    @property
    def guidance_scale(self) -> float | None:
        return self._guidance_scale

    @property
    def num_timesteps(self) -> int | None:
        return self._num_timesteps

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] | None = None,
        lyrics: str | list[str] = "",
        audio_duration: float = 60.0,
        vocal_language: str = "en",
        num_inference_steps: int = 8,
        guidance_scale: float = 1.0,
        shift: float = 3.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        output_type: str = "pt",
    ) -> DiffusionOutput:
        """Generate music from text prompts (text2music task only).

        Args mirror the diffusers ``AceStepPipeline.__call__`` but the engine
        side feeds ``OmniDiffusionRequest`` instead of raw kwargs; the explicit
        kwargs are kept for offline / direct-call usage.
        """
        # 0. Extract from req.
        # TODO: mirror the stable_audio dict-vs-str dance once the API layer
        # settles; for now treat req.prompts as a list of strings.
        prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in req.prompts] or prompt
        if num_inference_steps is None or req.sampling_params.num_inference_steps is not None:
            num_inference_steps = req.sampling_params.num_inference_steps or num_inference_steps
        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale
        extra = req.sampling_params.extra_args
        lyrics = extra.get("lyrics", lyrics)
        audio_duration = extra.get("audio_duration", audio_duration)
        vocal_language = extra.get("vocal_language", vocal_language)
        shift = extra.get("shift", shift)
        if generator is None:
            generator = req.sampling_params.generator
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(req.sampling_params.seed)

        if prompt is None or (isinstance(prompt, list) and not prompt):
            raise ValueError("Must provide `prompt` as a string or list of strings.")

        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)

        device = self.device
        dtype = self.transformer.dtype if hasattr(self.transformer, "dtype") else self.od_config.dtype
        acoustic_dim = self.transformer.config.audio_acoustic_hidden_dim

        # Turbo checkpoints have guidance distilled into the weights; CFG produces
        # over-guided audio. Warn + coerce so users forwarding base/sft settings
        # to a turbo pipeline still get sensible output.
        if self.is_turbo and guidance_scale > 1.0:
            logger.warning(
                "Guidance scale %s is ignored for turbo (guidance-distilled) checkpoints.",
                guidance_scale,
            )
            guidance_scale = 1.0
        self._guidance_scale = guidance_scale
        self._num_timesteps = num_inference_steps

        # 1. Encode prompts + lyrics.
        text_hidden_states, text_attention_mask, lyric_hidden_states, lyric_attention_mask = self.encode_prompt(
            prompt=prompt,
            lyrics=lyrics,
            device=device,
            vocal_language=vocal_language,
            audio_duration=audio_duration,
        )

        # 2. Timbre conditioning. text2music has no reference audio: slice the
        #    learned silence_latent (trained-in fallback) at the canonical
        #    30-second frame count. Zeros are out-of-distribution for the timbre
        #    encoder and produce drone-like output.
        timbre_fix_frame = math.ceil(30 * self.latents_per_second)
        refer_audio_acoustic = (
            self.condition_encoder.silence_latent[:, :timbre_fix_frame, :]
            .to(device=device, dtype=dtype)
            .expand(batch_size, -1, -1)
            .contiguous()
        )
        refer_audio_order_mask = torch.arange(batch_size, device=device, dtype=torch.long)

        # 3. Run condition encoder once before the denoising loop.
        encoder_hidden_states, _encoder_attention_mask = self.condition_encoder(
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic,
            refer_audio_order_mask=refer_audio_order_mask,
        )

        # 4. Build context latents. text2music has no source audio and no
        #    repaint window. Diffusers' equivalent (pipeline_ace_step.py
        #    prepare_src_latents + _build_chunk_mask) for text2music uses:
        #      * ``src_latents`` = silence_latent tiled to latent_length
        #      * ``chunk_mask``  = all-ones (generate the entire span)
        #    Zero-filling here (as a previous revision did) leaves the DiT with
        #    no positional / acoustic prior and produces noise output. The
        #    DiT concatenates this with the noisy latents on the channel dim.
        latent_length = math.ceil(audio_duration * self.latents_per_second)
        silence_latent = self.condition_encoder.silence_latent.to(device=device, dtype=dtype)
        if silence_latent.shape[1] >= latent_length:
            src_latents = silence_latent[:, :latent_length, :]
        else:
            repeats = (latent_length + silence_latent.shape[1] - 1) // silence_latent.shape[1]
            src_latents = silence_latent.repeat(1, repeats, 1)[:, :latent_length, :]
        src_latents = src_latents.expand(batch_size, -1, -1).contiguous()
        chunk_mask = torch.ones((batch_size, latent_length, acoustic_dim), device=device, dtype=dtype)
        context_latents = torch.cat([src_latents, chunk_mask], dim=-1)

        # 5. Sample noise latents.
        latents = self.prepare_latents(
            batch_size=batch_size,
            audio_duration=latent_length / self.latents_per_second,
            dtype=dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        # 6. Configure scheduler with the ACE-Step shifted sigma schedule.
        t_schedule = self._get_timestep_schedule(
            num_inference_steps=num_inference_steps,
            shift=shift,
            device=device,
            dtype=torch.float32,
        )
        self.scheduler.set_timesteps(sigmas=t_schedule.tolist(), device=device)
        num_steps = len(self.scheduler.timesteps)

        # 7. Denoising loop. Turbo defaults to no CFG.
        xt = latents
        for step_idx, t_sched in enumerate(self.scheduler.timesteps):
            current_timestep = float(t_sched)
            self._current_timestep = current_timestep
            t_curr_tensor = current_timestep * torch.ones((batch_size,), device=device, dtype=dtype)

            model_output = self.transformer(
                hidden_states=xt,
                timestep=t_curr_tensor,
                timestep_r=t_curr_tensor,
                encoder_hidden_states=encoder_hidden_states,
                context_latents=context_latents,
                return_dict=False,
            )
            vt = model_output[0]

            xt = self.scheduler.step(vt, t_sched, xt, return_dict=False)[0]
        self._current_timestep = None
        _ = num_steps  # progress-bar hook lives here when we wire profiling

        # 8. Decode to audio waveform.
        if output_type == "latent":
            return DiffusionOutput(
                output=xt,
                stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
            )

        # VAE expects [B, C, T]; latents are [B, T, C].
        audio_latents = xt.to(dtype=self.vae.dtype).transpose(1, 2)
        audio = self.vae.decode(audio_latents).sample

        # Two-stage normalization matching the diffusers reference: anti-clip,
        # then rescale to -1 dBFS for consistent loudness.
        if audio.dtype != torch.float32:
            audio = audio.float()
        peak = audio.abs().amax(dim=[1, 2], keepdim=True)
        if torch.any(peak > 1.0):
            audio = audio / peak.clamp(min=1.0)
        target_amp = 10.0 ** (-1.0 / 20.0)  # -1 dBFS
        peak = audio.abs().amax(dim=[1, 2], keepdim=True).clamp(min=1e-6)
        audio = audio * (target_amp / peak)

        if output_type == "np":
            audio = audio.cpu().float().numpy()

        return DiffusionOutput(
            output=audio,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights using AutoWeightsLoader for vLLM integration."""
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
