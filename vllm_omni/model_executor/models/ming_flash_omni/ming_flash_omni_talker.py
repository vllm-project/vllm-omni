# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Copyright (c) Ant Group. All rights reserved.
# Adapted from:
# https://github.com/inclusionAI/Ming/blob/e58533db227031990c5a6864dcf5f08fb53ed0d2/modeling_bailing_talker.py
"""Ming-flash-omni-2.0 talker (TTS) stage model."""

from __future__ import annotations

import glob as glob_module
import os
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoTokenizer, Qwen2Config, Qwen2Model
from transformers.utils.hub import cached_file
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.transformers_utils.configs.ming_flash_omni import MingFlashOmniTalkerConfig

from .audio_vae import AudioVAE, AudioVAEConfig
from .prompt_utils import DEFAULT_PROMPT as MING_DEFAULT_PROMPT
from .talker_module import CFM, Aggregator, DiT, MingAudioGenerator, build_tts_input
from .text_processing import segment_and_normalize
from .voice_presets import VoicePresetRegistry

logger = init_logger(__name__)


@dataclass(slots=True)
class _GenerationParams:
    """Resolved sampling / decoding parameters for one forward call."""

    prompt: str
    instruction: str | None
    cfg: float
    sigma: float
    temperature: float
    max_steps: int
    use_zero_spk_emb: bool
    max_text_length: int
    use_static_cache: bool
    stream_decode: bool


@dataclass(slots=True)
class _VoiceContext:
    """Voice cloning inputs resolved from request info + presets."""

    spk_emb: Any  # list[Tensor] | Tensor | list[float] | None
    prompt_text: str | None
    prompt_wav_lat: torch.Tensor | None
    prompt_wav_emb: torch.Tensor | None
    already_projected: bool


class MingFlashOmniTalkerForConditionalGeneration(nn.Module, CustomProcessMixin):
    """Ming-flash-omni-2.0 talker stage: text -> audio waveform.

    Uses Qwen2 LLM + CFM (Conditional Flow Matching with DiT) + Aggregator
    in an autoregressive loop to produce continuous audio latents, then
    AudioVAE decodes latents to waveforms.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False

        self.vllm_config = vllm_config
        root_config = vllm_config.model_config.hf_config

        model_path = vllm_config.model_config.model
        self._model_path = model_path
        self.talker_dir = (
            os.path.join(model_path, "talker") if os.path.isdir(os.path.join(model_path, "talker")) else model_path
        )

        # When used standalone (model_arch=MingFlashOmniTalkerForConditionalGeneration),
        # the root hf_config may be BailingMM2Config (thinker-only) due to model file structure
        # Resolve talker config from talker/config.json in that case.
        config = (
            root_config
            if isinstance(root_config, MingFlashOmniTalkerConfig)
            else self._resolve_talker_config(root_config, self.talker_dir, model_path)
        )
        self.config = config

        self._standalone = prefix in ("", "talker")
        if self._standalone:
            self.allow_patterns_overrides = ["talker/model*.safetensors"]
            self.fall_back_to_pt_during_load = False

        # LLM
        llm_config = self._resolve_llm_config(config, self.talker_dir, model_path)
        llm_config._attn_implementation = "sdpa"
        self.llm_config = llm_config
        self.hidden_size = llm_config.hidden_size
        self.latent_dim = config.latent_dim
        self.patch_size = config.patch_size
        self.his_patch_size = config.history_patch_size
        self.cfg_strength = config.cfg_strength

        self.model = Qwen2Model(llm_config)
        self.cfm = CFM(
            DiT(llm_input_dim=self.hidden_size, **config.flowmodel),
            steps=config.steps,
        )
        self.aggregator = Aggregator(llm_input_dim=self.hidden_size, **config.aggregator)
        self.stop_head = nn.Linear(self.hidden_size, 2, bias=True)
        # CAMPPlus 192-dim -> hidden
        self.spk_head = nn.Linear(192, self.hidden_size, bias=True)

        # AudioVAE
        self.audio_vae, self._vae_weight_source = self._init_audio_vae(config, self.talker_dir, model_path)

        self._use_cuda_graphs = not vllm_config.model_config.enforce_eager

        self.audio_generator = MingAudioGenerator(
            config=self.config,
            llm_config=self.llm_config,
            model=self.model,
            cfm=self.cfm,
            aggregator=self.aggregator,
            stop_head=self.stop_head,
            audio_vae=self.audio_vae,
            patch_size=self.patch_size,
            his_patch_size=self.his_patch_size,
            latent_dim=self.latent_dim,
            cfg_strength=self.cfg_strength,
            use_cuda_graphs=self._use_cuda_graphs,
        )
        self.voice_presets = VoicePresetRegistry(
            talker_dir=self.talker_dir,
            model_path=self._model_path,
            download_dir=vllm_config.load_config.download_dir,
            audio_vae=self.audio_vae,
            aggregator=self.aggregator,
            spk_head=self.spk_head,
            patch_size=self.patch_size,
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @cached_property
    def tokenizer(self):
        # Lazy Qwen2 tokenizer resolution:
        #   1. Try local dirs first (talker/llm, talker, and then model root).
        #   2. HF repo-id fallback: talker/llm is the canonical tokenizer location.
        candidates = (os.path.join(self.talker_dir, "llm"), self.talker_dir, self._model_path)
        for path in candidates:
            if os.path.isdir(path):
                try:
                    logger.debug("Resolving talker tokenizer from local dir %s", path)
                    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
                except Exception:
                    continue
        for subfolder in ("talker/llm", "llm"):
            try:
                logger.debug("Resolving talker tokenizer from HF subfolder %s", subfolder)
                return AutoTokenizer.from_pretrained(self._model_path, subfolder=subfolder, trust_remote_code=True)
            except Exception:
                continue
        logger.debug("Falling back to raw model_path tokenizer resolution")
        return AutoTokenizer.from_pretrained(self._model_path, trust_remote_code=True)

    @staticmethod
    def _resolve_talker_config(config, talker_dir: str, model_path: str) -> MingFlashOmniTalkerConfig:
        """Resolve MingFlashOmniTalkerConfig when the root config is not one.

        This happens in standalone TTS mode where hf_config is BailingMM2Config.
        """
        # If the root config wraps a talker_config, use it
        talker_config = getattr(config, "talker_config", None)
        if isinstance(talker_config, MingFlashOmniTalkerConfig):
            return talker_config

        # Try loading from talker/config.json
        if os.path.isdir(talker_dir):
            try:
                resolved = MingFlashOmniTalkerConfig.from_pretrained(talker_dir)
                logger.info("Resolved talker config from %s", talker_dir)
                return resolved
            except Exception:
                pass

        try:
            resolved = MingFlashOmniTalkerConfig.from_pretrained(model_path, subfolder="talker", trust_remote_code=True)
            logger.info("Resolved talker config from %s/talker (HF hub)", model_path)
            return resolved
        except Exception as e:
            raise ValueError(
                f"Cannot resolve MingFlashOmniTalkerConfig. The root config "
                f"is {type(config).__name__}, and talker/config.json was not "
                f"found at {talker_dir} or via HF hub: {e}"
            ) from e

    @staticmethod
    def _resolve_llm_config(config: MingFlashOmniTalkerConfig, talker_dir: str, model_path: str) -> Qwen2Config:
        """Resolve the Qwen2 LLM config for the talker backbone."""

        if config.llm_config is not None:
            return Qwen2Config(**config.llm_config) if isinstance(config.llm_config, dict) else config.llm_config

        # Try local talker/llm directory
        llm_dir = os.path.join(talker_dir, "llm")
        if os.path.isdir(llm_dir):
            return Qwen2Config.from_pretrained(llm_dir)

        # HF hub fallback
        for subfolder in ("talker/llm", "llm"):
            try:
                return Qwen2Config.from_pretrained(model_path, subfolder=subfolder, trust_remote_code=True)
            except Exception:
                continue

        raise ValueError(
            f"Cannot find talker LLM config at {llm_dir}. "
            "Either provide llm_config in MingFlashOmniTalkerConfig or "
            "ensure the model path contains talker/llm/config.json."
        )

    @staticmethod
    def _init_audio_vae(
        config: MingFlashOmniTalkerConfig, talker_dir: str, model_path: str
    ) -> tuple[AudioVAE | None, str | tuple[str, str] | None]:
        """Initialize AudioVAE and return (vae, weight_source).

        weight_source is either a local directory path (str) or an
        (repo_id, subfolder) tuple for HF hub downloads, or None.
        """
        vae_path = config.audio_vae_path or os.path.join(talker_dir, "vae")

        # Try local directory first
        if os.path.isdir(vae_path):
            try:
                vae_config = AudioVAEConfig.from_pretrained(vae_path)
                vae = AudioVAE(vae_config)
                logger.info("Initialized AudioVAE from %s (sr=%d)", vae_path, vae_config.sample_rate)
                return vae, vae_path
            except Exception as e:
                logger.warning("Failed to initialize AudioVAE from %s: %s", vae_path, e)
                return None, None

        # HF hub fallback
        for subfolder in ("talker/vae", "vae"):
            try:
                vae_config = AudioVAEConfig.from_pretrained(model_path, subfolder=subfolder, trust_remote_code=True)
                vae = AudioVAE(vae_config)
                logger.info(f"Initialized AudioVAE from {model_path}/{subfolder}")
                return vae, (model_path, subfolder)
            except Exception:
                continue

        logger.info("AudioVAE not found at %s; waveform decoding unavailable", vae_path)
        return None, None

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata=None) -> torch.Tensor | None:
        return None

    def sample(self, logits: torch.Tensor, sampling_metadata):
        return None

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors | None:
        return None

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict[str, object]]:
        info: dict[str, object] = {"text": "dummy", "use_zero_spk_emb": True, "max_steps": 1}
        return [info for _ in range(num_reqs)]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict] | None = None,
        **kwargs,
    ) -> OmniOutput:
        """Run TTS generation and return audio output.

        The full autoregressive generation loop is executed inside this method.
        """
        additional_info = self._extract_additional_info(runtime_additional_information)
        params = self._resolve_generation_params(additional_info)
        voice = self._resolve_voice(additional_info)

        latents = self._generate_latents(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            text=additional_info.get("text", ""),
            params=params,
            voice=voice,
        )
        return self._decode_to_output(latents, stream_decode=params.stream_decode)

    @staticmethod
    def _extract_additional_info(
        runtime_additional_information: list[dict] | None,
    ) -> dict[str, Any]:
        if runtime_additional_information and len(runtime_additional_information) > 0:
            return runtime_additional_information[0] or {}
        return {}

    def _resolve_generation_params(self, additional_info: dict[str, Any]) -> _GenerationParams:
        # "omni"    : thinker -> talker hand-off with hardcoded defaults
        # "instruct": standalone TTS with caller-supplied sampling knobs
        ming_task = additional_info.get("ming_task", "instruct")

        if ming_task == "omni":
            prompt = MING_DEFAULT_PROMPT
            instruction = None
            use_zero_spk_emb = additional_info.get("spk_emb") is None
            cfg = 2.0
            sigma = 0.25
            temperature = 0.0
            max_steps = 200
        else:
            prompt = additional_info.get("prompt", MING_DEFAULT_PROMPT)
            instruction = additional_info.get("instruction", None)
            use_zero_spk_emb = additional_info.get("use_zero_spk_emb", False)
            cfg = additional_info.get("cfg", self.cfg_strength)
            sigma = additional_info.get("sigma", 0.25)
            temperature = additional_info.get("temperature", 0.0)
            max_steps = int(additional_info.get("max_steps", additional_info.get("max_decode_steps", 200)))

        return _GenerationParams(
            prompt=prompt,
            instruction=instruction,
            cfg=cfg,
            sigma=sigma,
            temperature=temperature,
            max_steps=max_steps,
            use_zero_spk_emb=use_zero_spk_emb,
            max_text_length=int(additional_info.get("max_text_length", 50)),
            use_static_cache=bool(additional_info.get("use_static_cache", True)),
            stream_decode=bool(additional_info.get("stream_decode", True)),
        )

    def _resolve_voice(self, additional_info: dict[str, Any]) -> _VoiceContext:
        spk_emb = additional_info.get("spk_emb", None)
        prompt_text = additional_info.get("prompt_text", None)
        prompt_wav_lat = additional_info.get("prompt_wav_lat", None)
        prompt_wav_emb = additional_info.get("prompt_wav_emb", None)
        already_projected = False

        voice_name = additional_info.get("voice_name", None)
        if voice_name and spk_emb is None and voice_name in self.voice_presets:
            preset = self.voice_presets.get(voice_name) or {}
            prompt_wav_lat = preset.get("prompt_wav_lat")
            prompt_wav_emb = preset.get("prompt_wav_emb")
            spk_emb = preset.get("spk_emb")
            already_projected = True
            if prompt_text is None:
                prompt_text = preset.get("prompt_text")

        return _VoiceContext(
            spk_emb=spk_emb,
            prompt_text=prompt_text,
            prompt_wav_lat=prompt_wav_lat,
            prompt_wav_emb=prompt_wav_emb,
            already_projected=already_projected,
        )

    def _project_spk_emb(
        self, spk_emb: Any, already_projected: bool, use_zero_spk_emb: bool
    ) -> list[torch.Tensor] | None:
        if spk_emb is None:
            if use_zero_spk_emb:
                return [torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)]
            return None

        if already_projected:
            return spk_emb if isinstance(spk_emb, list) else [spk_emb]

        if isinstance(spk_emb, torch.Tensor):
            tensors = [spk_emb]
        elif isinstance(spk_emb, list) and spk_emb and isinstance(spk_emb[0], (int, float)):
            tensors = [torch.tensor(spk_emb, dtype=self.dtype).unsqueeze(0)]
        elif isinstance(spk_emb, list):
            tensors = spk_emb
        else:
            tensors = [spk_emb]
        return [self.spk_head(t.to(device=self.device, dtype=self.dtype)) for t in tensors]

    def _generate_latents(
        self,
        *,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        text: str,
        params: _GenerationParams,
        voice: _VoiceContext,
    ) -> list[torch.Tensor]:
        generator = self.audio_generator

        if inputs_embeds is not None:
            # Caller pre-built embeddings — run a single AR pass.
            return generator.generate_latents(
                inputs_embeds=inputs_embeds,
                prompt_wav_lat=voice.prompt_wav_lat,
                max_steps=params.max_steps,
                cfg=params.cfg,
                sigma=params.sigma,
                temperature=params.temperature,
                use_static_cache=params.use_static_cache,
            )

        spk_emb = self._project_spk_emb(voice.spk_emb, voice.already_projected, params.use_zero_spk_emb)
        text_segments = segment_and_normalize(text, max_length=params.max_text_length) if text else []

        if not text_segments:
            # vLLM passes 1D input_ids; Qwen2Model expects (batch, seq).
            inputs_embeds = self.model.get_input_embeddings()(input_ids.to(self.device)).unsqueeze(0)
            return generator.generate_latents(
                inputs_embeds=inputs_embeds,
                prompt_wav_lat=voice.prompt_wav_lat,
                max_steps=params.max_steps,
                cfg=params.cfg,
                sigma=params.sigma,
                temperature=params.temperature,
                use_static_cache=params.use_static_cache,
            )

        all_latents: list[torch.Tensor] = []
        for segment in text_segments:
            seg_embeds, _ = build_tts_input(
                tokenizer=self.tokenizer,
                embed_tokens=self.model.get_input_embeddings(),
                device=self.device,
                dtype=torch.bfloat16,
                text=segment,
                prompt=params.prompt,
                spk_emb=spk_emb,
                instruction=params.instruction,
                prompt_text=voice.prompt_text,
                prompt_wav_emb=voice.prompt_wav_emb,
            )
            effective_max_steps = generator.duration_capped_steps(len(segment), params.max_steps)
            all_latents.extend(
                generator.generate_latents(
                    inputs_embeds=seg_embeds,
                    prompt_wav_lat=voice.prompt_wav_lat,
                    max_steps=effective_max_steps,
                    cfg=params.cfg,
                    sigma=params.sigma,
                    temperature=params.temperature,
                    use_static_cache=params.use_static_cache,
                )
            )
        return all_latents

    def _decode_to_output(self, latents: list[torch.Tensor], *, stream_decode: bool) -> OmniOutput:
        multimodal_outputs: dict[str, Any] = {}
        if latents and self.audio_vae is not None:
            waveform = self.audio_generator.decode_to_waveform(latents, stream_decode=stream_decode)
            if not stream_decode:
                waveform = self.audio_generator.trim_trailing_silence(waveform)
            multimodal_outputs["audio"] = waveform.detach().float().cpu()
            multimodal_outputs["sr"] = torch.tensor(self.audio_vae.config.sample_rate)
        elif latents:
            all_lat = torch.cat(latents, dim=1)
            multimodal_outputs["audio_latents"] = all_lat.detach().float().cpu()

        return OmniOutput(text_hidden_states=None, multimodal_outputs=multimodal_outputs)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for all talker components.

        The talker's HF checkpoint (talker/model.safetensors) stores
        weights with prefixes matching this module's submodule names directly.
        And AudioVAE weights live in a separate file under talker/vae/
        """
        # Standalone: bypass the default loader's iterator (torch.load on
        # .safetensors crashes) and read talker/model*.safetensors directly.
        if self._standalone:
            weights = self._iter_talker_safetensors()

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["audio_vae."],  # loaded separately
            skip_substrs=["rotary_embed.inv_freq"],  # non-persistent buffer
        )
        loaded = loader.load_weights(weights)
        logger.info("Loaded %d talker weights from checkpoint", len(loaded))

        if self.audio_vae is not None and self._vae_weight_source is not None:
            loaded.update(self._load_vae_weights())

        # Register voice presets after all weights (incl. VAE) are loaded.
        try:
            self.voice_presets.load_presets_from_manifest(device=self.device, dtype=self.dtype)
        except Exception as e:  # pragma: no cover — best-effort
            logger.warning("Voice preset loading failed (non-fatal): %s", e)

        return loaded

    def _iter_talker_safetensors(self) -> Iterable[tuple[str, torch.Tensor]]:
        """Yield (name, tensor) pairs from talker/model*.safetensors."""
        model_path = self._model_path
        # Try local path first
        for candidate in (os.path.join(model_path, "talker"), model_path):
            sf_files = sorted(glob_module.glob(os.path.join(candidate, "model*.safetensors")))
            if sf_files:
                for sf_path in sf_files:
                    yield from load_file(sf_path, device="cpu").items()
                return

        # HF hub fallback: download only the talker checkpoint files
        model_root = download_weights_from_hf_specific(
            model_path,
            self.vllm_config.load_config.download_dir,
            allow_patterns=["talker/model*.safetensors"],
        )
        talker_dir = os.path.join(model_root, "talker")
        sf_files = sorted(glob_module.glob(os.path.join(talker_dir, "model*.safetensors")))
        if not sf_files:
            raise RuntimeError(f"No talker safetensors found under {model_root}. Expected talker/model*.safetensors.")
        for sf_path in sf_files:
            yield from load_file(sf_path, device="cpu").items()

    def _load_vae_weights(self) -> set[str]:
        """Load AudioVAE weights from talker/vae/model.safetensors."""
        if self.audio_vae is None or self._vae_weight_source is None:
            return set()

        # Resolve safetensors file paths from the weight source
        safetensors_files: list[str] = []
        source = self._vae_weight_source
        if isinstance(source, str):
            # Local directory path
            safetensors_files = sorted(glob_module.glob(os.path.join(source, "*.safetensors")))
        elif isinstance(source, tuple):
            # (repo_id, subfolder) for HF hub
            repo_id, subfolder = source
            for filename in ("model.safetensors", "diffusion_pytorch_model.safetensors"):
                try:
                    cached = cached_file(repo_id, filename, subfolder=subfolder)
                except Exception:
                    cached = None
                if cached is not None:
                    safetensors_files.append(cached)
                    break

        if not safetensors_files:
            logger.warning("No AudioVAE safetensors files found for source=%s", source)
            return set()

        vae_state_keys = set(self.audio_vae.state_dict().keys())
        vae_loader = AutoWeightsLoader(self.audio_vae)
        loaded: set[str] = set()
        for sf_path in safetensors_files:
            file_weights = load_file(sf_path, device="cpu")
            matched = ((name, tensor) for name, tensor in file_weights.items() if name in vae_state_keys)
            loaded.update(f"audio_vae.{name}" for name in vae_loader.load_weights(matched))

        logger.info("Loaded %d AudioVAE weights from %s", len(loaded), source)
        return loaded
