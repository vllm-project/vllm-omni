from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal.audio import resample_audio_resampy
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.vibevoice_tts.configuration_vibevoice import (
    VibeVoiceAcousticTokenizerConfig,
    VibeVoiceConfig,
    VibeVoiceDiffusionHeadConfig,
    VibeVoiceSemanticTokenizerConfig,
)

logger = init_logger(__name__)

_SPEECH_START_ID = 151652
_SPEECH_END_ID = 151653
_SPEECH_DIFFUSION_ID = 151654
_TARGET_SAMPLE_RATE = 24000


try:
    from vibevoice.modular.modular_vibevoice_diffusion_head import VibeVoiceDiffusionHead
    from vibevoice.modular.modular_vibevoice_tokenizer import (
        VibeVoiceAcousticTokenizerModel,
        VibeVoiceSemanticTokenizerModel,
        VibeVoiceTokenizerStreamingCache,
    )
    from vibevoice.processor.audio_utils import AudioNormalizer
    from vibevoice.schedule.dpm_solver import DPMSolverMultistepScheduler
except ImportError as e:
    raise ImportError(
        "VibeVoice TTS support requires the optional `vibevoice` package. "
        "Install it with: pip install git+https://github.com/microsoft/VibeVoice.git"
    ) from e


class VibeVoiceSpeechConnector(nn.Module):
    """Projects speech latents into the decoder hidden size."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.norm = RMSNorm(output_dim, eps=1e-6)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x


class VibeVoiceTTSForConditionalGeneration(nn.Module, SupportsPP, CustomProcessMixin):
    """Single-stage VibeVoice TTS wrapper for vLLM-Omni.

    The model autoregressively emits VibeVoice diffusion placeholder tokens.
    For each generated placeholder token, `preprocess()` uses the hidden state
    from the previous step to:
    1. sample the next acoustic latent with the diffusion head,
    2. decode the corresponding audio chunk with the acoustic tokenizer, and
    3. replace the token embedding with the acoustic connector output.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: VibeVoiceConfig = vllm_config.model_config.hf_config  # type: ignore[assignment]
        self.config = config
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = True

        self._speech_start_id = _SPEECH_START_ID
        self._speech_end_id = _SPEECH_END_ID
        self._speech_diffusion_id = _SPEECH_DIFFUSION_ID
        self._sample_rate = int(getattr(config, "target_sample_rate", _TARGET_SAMPLE_RATE))
        self._speech_tok_compress_ratio = int(getattr(config, "speech_tok_compress_ratio", 3200))

        decoder_config = config.decoder_config
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=decoder_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors
        self.logits_processor = LogitsProcessor(decoder_config.vocab_size)

        self.acoustic_tokenizer = VibeVoiceAcousticTokenizerModel(
            self._build_acoustic_tokenizer_config(config.acoustic_tokenizer_config)
        )
        self.semantic_tokenizer = VibeVoiceSemanticTokenizerModel(
            self._build_semantic_tokenizer_config(config.semantic_tokenizer_config)
        )
        self.prediction_head = VibeVoiceDiffusionHead(self._build_diffusion_head_config(config.diffusion_head_config))
        hidden_size = int(decoder_config.hidden_size)
        self.acoustic_connector = VibeVoiceSpeechConnector(int(config.acoustic_vae_dim), hidden_size)
        self.semantic_connector = VibeVoiceSpeechConnector(int(config.semantic_vae_dim), hidden_size)

        self.noise_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=config.diffusion_head_config.ddpm_num_steps,
            beta_schedule=config.diffusion_head_config.ddpm_beta_schedule,
            prediction_type=config.diffusion_head_config.prediction_type,
        )

        self.register_buffer("speech_scaling_factor", torch.tensor(float("nan")))
        self.register_buffer("speech_bias_factor", torch.tensor(float("nan")))

        allowed = torch.zeros((decoder_config.vocab_size,), dtype=torch.bool)
        if self._speech_diffusion_id < decoder_config.vocab_size:
            allowed[self._speech_diffusion_id] = True
        if self._speech_end_id < decoder_config.vocab_size:
            allowed[self._speech_end_id] = True
        self.register_buffer("_allowed_generation_mask", allowed, persistent=False)

        self.gpu_resident_buffer_keys: set[str] = {"last_condition_hidden"}
        self._audio_normalizer = AudioNormalizer()
        self._audio_module_dtype = torch.float32
        self._lm_dtype = getattr(vllm_config.model_config, "dtype", None) or torch.bfloat16
        self._noise_scheduler_timesteps: torch.Tensor | None = None
        self._noise_scheduler_sigmas: torch.Tensor | None = None
        self._noise_scheduler_num_inference_steps: int | None = None
        self._ensure_audio_module_dtype()
        self._cache_noise_scheduler_sampling_state()

    @staticmethod
    def _build_acoustic_tokenizer_config(config_like: object) -> VibeVoiceAcousticTokenizerConfig:
        if isinstance(config_like, VibeVoiceAcousticTokenizerConfig):
            return config_like
        if isinstance(config_like, dict):
            return VibeVoiceAcousticTokenizerConfig(**config_like)
        if hasattr(config_like, "to_dict"):
            return VibeVoiceAcousticTokenizerConfig(**config_like.to_dict())
        raise TypeError(f"Unsupported VibeVoice acoustic tokenizer config: {type(config_like)}")

    @staticmethod
    def _build_semantic_tokenizer_config(config_like: object) -> VibeVoiceSemanticTokenizerConfig:
        if isinstance(config_like, VibeVoiceSemanticTokenizerConfig):
            return config_like
        if isinstance(config_like, dict):
            return VibeVoiceSemanticTokenizerConfig(**config_like)
        if hasattr(config_like, "to_dict"):
            return VibeVoiceSemanticTokenizerConfig(**config_like.to_dict())
        raise TypeError(f"Unsupported VibeVoice semantic tokenizer config: {type(config_like)}")

    @staticmethod
    def _build_diffusion_head_config(config_like: object) -> VibeVoiceDiffusionHeadConfig:
        if isinstance(config_like, VibeVoiceDiffusionHeadConfig):
            return config_like
        if isinstance(config_like, dict):
            return VibeVoiceDiffusionHeadConfig(**config_like)
        if hasattr(config_like, "to_dict"):
            return VibeVoiceDiffusionHeadConfig(**config_like.to_dict())
        raise TypeError(f"Unsupported VibeVoice diffusion head config: {type(config_like)}")

    def _ensure_audio_module_dtype(self) -> None:
        for module in (
            self.acoustic_tokenizer,
            self.semantic_tokenizer,
            self.prediction_head,
            self.acoustic_connector,
            self.semantic_connector,
        ):
            try:
                module.to(dtype=self._audio_module_dtype)
            except Exception:
                logger.debug("Skipping dtype normalization for %s", module.__class__.__name__, exc_info=True)

    def _ensure_noise_scheduler_cpu_state(self) -> None:
        for attr in (
            "betas",
            "alphas",
            "alphas_cumprod",
            "alpha_t",
            "sigma_t",
            "lambda_t",
            "sigmas",
            "timesteps",
        ):
            value = getattr(self.noise_scheduler, attr, None)
            if isinstance(value, torch.Tensor) and value.device.type != "cpu":
                setattr(self.noise_scheduler, attr, value.to("cpu"))

        model_outputs = getattr(self.noise_scheduler, "model_outputs", None)
        if isinstance(model_outputs, list):
            for idx, value in enumerate(model_outputs):
                if isinstance(value, torch.Tensor) and value.device.type != "cpu":
                    model_outputs[idx] = value.to("cpu")

    def _cache_noise_scheduler_sampling_state(self) -> None:
        self._ensure_noise_scheduler_cpu_state()
        self.noise_scheduler.set_timesteps(
            self.config.diffusion_head_config.ddpm_num_inference_steps,
            device="cpu",
        )
        self._ensure_noise_scheduler_cpu_state()
        self._noise_scheduler_timesteps = self.noise_scheduler.timesteps.detach().clone()
        self._noise_scheduler_sigmas = self.noise_scheduler.sigmas.detach().clone()
        self._noise_scheduler_num_inference_steps = int(self.noise_scheduler.num_inference_steps)
        self._reset_noise_scheduler_state()

    def _reset_noise_scheduler_state(self) -> None:
        if (
            self._noise_scheduler_timesteps is None
            or self._noise_scheduler_sigmas is None
            or self._noise_scheduler_num_inference_steps is None
        ):
            self._cache_noise_scheduler_sampling_state()
            return
        self.noise_scheduler.timesteps = self._noise_scheduler_timesteps
        self.noise_scheduler.sigmas = self._noise_scheduler_sigmas
        self.noise_scheduler.num_inference_steps = self._noise_scheduler_num_inference_steps
        self.noise_scheduler.model_outputs = [None] * int(self.noise_scheduler.config.solver_order)
        self.noise_scheduler.lower_order_nums = 0
        self.noise_scheduler._step_index = None
        self.noise_scheduler._begin_index = None

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.language_model.embed_input_ids(input_ids)

    def get_language_model(self) -> nn.Module:
        return self.language_model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor | IntermediateTensors:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        if intermediate_tensors is not None:
            inputs_embeds = None

        return self.language_model.model(
            None,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None
    ) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None

        try:
            logits = self.language_model.compute_logits(hidden_states, sampling_metadata=sampling_metadata)
        except TypeError:
            logits = self.language_model.compute_logits(hidden_states)
        if logits is None:
            return None
        return logits.masked_fill(~self._allowed_generation_mask, float("-inf"))

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        info_dicts = kwargs.get("model_intermediate_buffer")
        if info_dicts is None:
            info_dicts = kwargs.get("runtime_additional_information") or []

        audio_chunks: list[torch.Tensor | None] = []
        sample_rates: list[torch.Tensor | None] = []
        for info in info_dicts:
            if not isinstance(info, dict):
                audio_chunks.append(None)
                sample_rates.append(None)
                continue
            chunk = info.get("audio_chunk")
            audio_chunks.append(chunk if isinstance(chunk, torch.Tensor) else None)
            sr = info.get("audio_sr")
            sample_rates.append(sr if isinstance(sr, torch.Tensor) else None)

        multimodal_outputs: dict[str, list[torch.Tensor | None]] = {}
        if any(chunk is not None for chunk in audio_chunks):
            multimodal_outputs["audio"] = audio_chunks
        if any(sr is not None for sr in sample_rates):
            multimodal_outputs["sr"] = sample_rates
        return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs=multimodal_outputs)

    @staticmethod
    def _flatten_info_dict(info_dict: dict[str, Any]) -> dict[str, Any]:
        additional_information = info_dict.get("additional_information")
        if not isinstance(additional_information, dict):
            return info_dict
        merged = {k: v for k, v in info_dict.items() if k != "additional_information"}
        for key, value in additional_information.items():
            merged.setdefault(key, value)
        return merged

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        info_dict = self._flatten_info_dict(info_dict)
        span_len = int(input_ids.shape[0])
        if span_len <= 0:
            embeds = input_embeds if input_embeds is not None else self.embed_input_ids(input_ids)
            return input_ids, embeds, {}

        clear_update = {"audio_chunk": None, "audio_sr": None}
        if span_len > 1:
            embeds, prefill_update = self._build_prefill_embeds(input_ids, info_dict)
            return input_ids, embeds, clear_update | prefill_update

        token_id = int(input_ids.reshape(-1)[0].item())
        token_embed = self.embed_input_ids(input_ids.reshape(1, 1)).reshape(1, -1).to(dtype=self._lm_dtype)
        if token_id != self._speech_diffusion_id:
            return input_ids, token_embed, clear_update

        last_condition_hidden = info_dict.get("last_condition_hidden")
        if not isinstance(last_condition_hidden, torch.Tensor):
            logger.warning("Missing last_condition_hidden for VibeVoice decode token; using text embedding fallback")
            return input_ids, token_embed, clear_update

        speech_latent = self._sample_speech_latent(last_condition_hidden.reshape(1, -1))
        speech_latent_seq = speech_latent.unsqueeze(1)
        speech_embed = self.acoustic_connector(speech_latent_seq.to(self.acoustic_connector.fc1.weight.dtype))
        speech_embed = speech_embed.reshape(1, -1).to(device=token_embed.device, dtype=token_embed.dtype)

        cache = info_dict.get("acoustic_streaming_cache")
        if cache is None:
            cache = VibeVoiceTokenizerStreamingCache()

        unscaled_latent = self._unscale_acoustic_latent(speech_latent).unsqueeze(1)
        cache_device = next(self.acoustic_tokenizer.parameters()).device
        sample_indices = torch.tensor([0], dtype=torch.long, device=cache_device)
        audio_chunk = self.acoustic_tokenizer.decode(
            unscaled_latent.to(device=cache_device, dtype=self._audio_module_dtype),
            cache=cache,
            sample_indices=sample_indices,
            use_cache=True,
            debug=False,
        )
        if isinstance(audio_chunk, torch.Tensor):
            audio_chunk = audio_chunk[0].reshape(-1).detach().to(torch.float32)
        else:
            audio_chunk = torch.as_tensor(audio_chunk[0], dtype=torch.float32)

        update = {
            "acoustic_streaming_cache": cache,
            "audio_chunk": audio_chunk,
            "audio_sr": torch.tensor(self._sample_rate, dtype=torch.int32),
        }
        return input_ids, speech_embed, update

    def postprocess(self, hidden_states: torch.Tensor, **_: Any) -> dict[str, Any]:
        if hidden_states.numel() == 0:
            return {}
        return {"last_condition_hidden": hidden_states[-1, :].detach().contiguous()}

    def _build_prefill_embeds(
        self,
        input_ids: torch.Tensor,
        info_dict: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        base = self.embed_input_ids(input_ids.reshape(1, -1).to(torch.long)).squeeze(0).to(dtype=self._lm_dtype)
        speech_input_mask = info_dict.get("speech_input_mask")
        voice_samples = info_dict.get("voice_samples")
        if not isinstance(speech_input_mask, list):
            return base, {}

        total_prompt_len = len(speech_input_mask)
        offset = int(info_dict.get("vibevoice_prefill_offset", 0) or 0)
        next_offset = min(offset + int(base.shape[0]), total_prompt_len)
        update: dict[str, Any] = {"vibevoice_prefill_offset": next_offset}

        voice_embeddings = info_dict.get("vibevoice_voice_prompt_embeds")
        if not isinstance(voice_embeddings, torch.Tensor):
            if not isinstance(voice_samples, list) or not voice_samples:
                return base, update
            voice_embeddings = self._encode_voice_prompt_samples(voice_samples, device=base.device)
            if voice_embeddings is None or voice_embeddings.numel() == 0:
                return base, update
            voice_embeddings = voice_embeddings.detach().to("cpu").contiguous()
            update["vibevoice_voice_prompt_embeds"] = voice_embeddings

        local_mask_list = speech_input_mask[offset:next_offset]
        if not local_mask_list:
            if next_offset >= total_prompt_len:
                update["vibevoice_voice_prompt_embeds"] = None
            return base, update

        local_mask = torch.tensor(local_mask_list, dtype=torch.bool, device=base.device)
        local_positions = int(local_mask.sum().item())
        if local_positions <= 0:
            if next_offset >= total_prompt_len:
                update["vibevoice_voice_prompt_embeds"] = None
            return base, update

        prompt_embed_offset = sum(1 for flag in speech_input_mask[:offset] if flag)
        voice_embeddings_slice = voice_embeddings[prompt_embed_offset : prompt_embed_offset + local_positions]
        if int(voice_embeddings_slice.shape[0]) > 0:
            mask_indices = local_mask.nonzero(as_tuple=False).reshape(-1)[: int(voice_embeddings_slice.shape[0])]
            base[mask_indices] = voice_embeddings_slice.to(device=base.device, dtype=base.dtype)

        if next_offset >= total_prompt_len:
            update["vibevoice_voice_prompt_embeds"] = None
        return base, update

    def _encode_voice_prompt_samples(
        self, voice_samples: list[dict[str, Any]], device: torch.device
    ) -> torch.Tensor | None:
        processed_wavs: list[np.ndarray] = []
        token_lengths: list[int] = []
        for sample in voice_samples:
            if not isinstance(sample, dict):
                continue
            wav = np.asarray(sample.get("samples") or [], dtype=np.float32)
            sample_rate = int(sample.get("sample_rate") or self._sample_rate)
            if wav.size == 0:
                continue
            wav = self._audio_normalizer(wav)
            if sample_rate != self._sample_rate:
                wav = resample_audio_resampy(wav, orig_sr=sample_rate, target_sr=self._sample_rate)
            processed_wavs.append(wav.astype(np.float32))
            token_lengths.append(int(math.ceil(len(wav) / self._speech_tok_compress_ratio)))

        if not processed_wavs:
            return None

        max_audio_len = max(len(wav) for wav in processed_wavs)
        max_token_len = max(token_lengths)
        padded = np.zeros((len(processed_wavs), max_audio_len), dtype=np.float32)
        speech_masks = np.zeros((len(processed_wavs), max_token_len), dtype=np.bool_)
        for idx, (wav, token_len) in enumerate(zip(processed_wavs, token_lengths, strict=False)):
            padded[idx, : len(wav)] = wav
            speech_masks[idx, :token_len] = True

        speech_tensors = torch.from_numpy(padded).to(device=device, dtype=self._audio_module_dtype)
        speech_masks_t = torch.from_numpy(speech_masks).to(device=device, dtype=torch.bool)

        acoustic_out = self.acoustic_tokenizer.encode(speech_tensors.unsqueeze(1))
        acoustic_tokens = acoustic_out.sample(self.acoustic_tokenizer.std_dist_type)[0]
        acoustic_features = self._scale_acoustic_features(acoustic_tokens, speech_masks_t)
        acoustic_embeds = self.acoustic_connector(acoustic_features)

        semantic_out = self.semantic_tokenizer.encode(speech_tensors.unsqueeze(1))
        semantic_tokens = semantic_out.mean
        semantic_embeds = self.semantic_connector(semantic_tokens)

        return (acoustic_embeds[speech_masks_t] + semantic_embeds[speech_masks_t]).to(dtype=self._lm_dtype)

    def _scale_acoustic_features(self, acoustic_tokens: torch.Tensor, speech_masks: torch.Tensor) -> torch.Tensor:
        if torch.isnan(self.speech_scaling_factor) or torch.isnan(self.speech_bias_factor):
            valid = acoustic_tokens[speech_masks]
            if valid.numel() > 0:
                scaling = 1.0 / valid.flatten().std()
                bias = -valid.flatten().mean()
                self.speech_scaling_factor.copy_(scaling)
                self.speech_bias_factor.copy_(bias)
        return (acoustic_tokens + self.speech_bias_factor) * self.speech_scaling_factor

    def _unscale_acoustic_latent(self, acoustic_latent: torch.Tensor) -> torch.Tensor:
        return acoustic_latent / self.speech_scaling_factor.to(acoustic_latent.device) - self.speech_bias_factor.to(
            acoustic_latent.device
        )

    @torch.no_grad()
    def _sample_speech_latent(self, condition: torch.Tensor) -> torch.Tensor:
        self._reset_noise_scheduler_state()
        condition = condition.to(device=next(self.prediction_head.parameters()).device, dtype=self._audio_module_dtype)
        speech = torch.randn(
            condition.shape[0],
            int(self.config.acoustic_vae_dim),
            device=condition.device,
            dtype=condition.dtype,
        )
        for timestep in self.noise_scheduler.timesteps:
            model_output = self.prediction_head(
                speech,
                timestep.repeat(speech.shape[0]).to(device=speech.device, dtype=speech.dtype),
                condition=condition,
            )
            speech = self.noise_scheduler.step(model_output, timestep, speech).prev_sample
        return speech

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded: set[str] = set()
        remaining_weights: list[tuple[str, torch.Tensor]] = []

        for name, weight in weights:
            if name == "model.speech_scaling_factor":
                with torch.no_grad():
                    self.speech_scaling_factor.copy_(
                        weight.to(
                            device=self.speech_scaling_factor.device,
                            dtype=self.speech_scaling_factor.dtype,
                        ).reshape_as(self.speech_scaling_factor)
                    )
                loaded.add("speech_scaling_factor")
                continue

            if name == "model.speech_bias_factor":
                with torch.no_grad():
                    self.speech_bias_factor.copy_(
                        weight.to(
                            device=self.speech_bias_factor.device,
                            dtype=self.speech_bias_factor.dtype,
                        ).reshape_as(self.speech_bias_factor)
                    )
                loaded.add("speech_bias_factor")
                continue

            remaining_weights.append((name, weight))

        mapper = WeightsMapper(
            orig_to_new_prefix={
                "model.language_model.": "language_model.model.",
                "model.acoustic_tokenizer.": "acoustic_tokenizer.",
                "model.semantic_tokenizer.": "semantic_tokenizer.",
                "model.acoustic_connector.": "acoustic_connector.",
                "model.semantic_connector.": "semantic_connector.",
                "model.prediction_head.": "prediction_head.",
            }
        )
        loader = AutoWeightsLoader(self)
        loaded |= loader.load_weights(remaining_weights, mapper=mapper)
        return loaded
