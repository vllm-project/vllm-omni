# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
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
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoTokenizer, Qwen2Config, Qwen2Model
from transformers.utils.hub import cached_file
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.model_loader.weight_utils import resolve_model_to_local_path
from vllm_omni.model_executor.models.common.ming.aggregator import Aggregator
from vllm_omni.model_executor.models.common.ming.audio_vae import AudioVAE, AudioVAEConfig
from vllm_omni.model_executor.models.ming_tts.constants import SPEAKER_EMBEDDING_DIM
from vllm_omni.model_executor.models.model_local_kv import ModelLocalKVScope, RowDriver, spec_from_hf_config
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.transformers_utils.configs.ming_talker import MingFlashOmniTalkerConfig

from .native_text_frontend import normalize_native_fragment
from .reference_audio import has_reference_audio, unpack_reference_audio
from .streaming import MingBatchedStreamJob, MingStreamingJob
from .talker_batch import BatchedTalkerRequest, MingTalkerBatchEngine, TalkerSegment
from .talker_module import CFM, DiT, MingAudioGenerator, build_tts_input
from .text_processing import segment_and_normalize
from .tts_request import GenerationParams as _GenerationParams
from .tts_request import resolve_conditioning_modes, resolve_generation_params
from .voice_presets import VoicePresetRegistry

if TYPE_CHECKING:
    from .talker_tn.talker_tn import TalkerTN

logger = init_logger(__name__)


def _build_audio_vae(config: AudioVAEConfig) -> AudioVAE:
    vae = AudioVAE(config)
    # NOTE: Official Encoder.__init__ shares the Qwen2 config between encoder and aggregator;
    # setting aggregator depth to 4 also limits the encoder:
    # https://huggingface.co/inclusionAI/Ming-3.0-audio-tiny/blob/25b96e30be3739849bf76c0f48ac11db4a262fb3/talker/AudioVAE/vae_modules.py#L71-L86
    # Preserve that behavior and the complete checkpoint weight layout here;
    # the shared VAE default used by Ming3-TTS keeps its configured depth.
    if config.patch_size != -1:
        vae.encoder.encoder.config.num_hidden_layers = vae.encoder.aggregator.config.num_hidden_layers
    return vae


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

    requires_request_ids = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self._stream_jobs: dict[str, MingStreamingJob] = {}

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
        if config.normalize_text and config.tn_mode == "upstream":
            # Validate resources/dependencies before constructing GPU modules.
            _ = self._upstream_tn

        self._standalone = prefix in ("", "talker")
        if self._standalone:
            self.allow_patterns_overrides = ["model*.safetensors", "talker/model*.safetensors"]
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
            DiT(
                llm_cond_dim=self.hidden_size, **{"native_attention": config.strict_weight_loading, **config.flowmodel}
            ),
            steps=config.steps,
        )
        # config.aggregator still wins `in_channels` if the checkpoint states it.
        self.aggregator = Aggregator(
            llm_input_dim=self.hidden_size,
            **{"in_channels": self.latent_dim, "native_attention": config.strict_weight_loading, **config.aggregator},
        )
        self.stop_head = nn.Linear(self.hidden_size, 2, bias=True)
        self.spk_head = nn.Linear(SPEAKER_EMBEDDING_DIM, self.hidden_size, bias=True)

        # AudioVAE
        self.audio_vae, self._vae_weight_source = self._init_audio_vae(config, self.talker_dir, model_path)
        upsampling = getattr(getattr(self.audio_vae, "decoder", None), "upsampling", None)
        if upsampling is not None and getattr(config, "low_latency_streaming", False):
            # Low-latency streaming also emits a decoded chunk without waiting
            # for the next one: only its last latent is held as lookahead.
            upsampling.frame_holdback = True

        self._use_cuda_graphs = not vllm_config.model_config.enforce_eager
        # With more than one scheduled sequence, every engine request joins
        # one shared AR batch instead of the one-request streaming job.
        max_num_seqs = int(getattr(vllm_config.scheduler_config, "max_num_seqs", 1) or 1)
        self._batch_rows = max_num_seqs if max_num_seqs > 1 else 0
        self._batch_engine: MingTalkerBatchEngine | None = None

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
            enable_llm_cuda_graph=bool(
                getattr(root_config, "enable_llm_cuda_graph", getattr(config, "enable_llm_cuda_graph", False))
            ),
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

    @cached_property
    def _upstream_tn(self) -> TalkerTN:
        """Own the normalizer for this checkpoint, without executing Hub code."""
        from .talker_tn.talker_tn import TalkerTN
        from .text_processing import _get_inflect

        normalizer = TalkerTN(Path(self.talker_dir) / "talker_tn")
        if _get_inflect() is None:
            raise ModuleNotFoundError(
                "Ming upstream text normalization requires pip install 'vllm-omni[ming-tn]' (missing inflect)"
            )
        return normalizer

    def model_local_kv_specs(self):
        """Forward to the generator, which owns the talker's StaticCache.

        ``MingAudioGenerator`` is a plain object rather than a submodule, so
        the module-tree walk cannot reach it on its own.
        """
        generator = getattr(self, "audio_generator", None)
        if generator is None:
            return []
        if getattr(self, "_batch_rows", 0):
            return [
                spec_from_hf_config(
                    self.llm_config,
                    name="talker_batched_kv",
                    dtype=self.dtype,
                    physical_capacity_positions=generator._max_cache_len,
                    capacity_source=f"talker config max_cache_len={generator._max_cache_len}",
                    scope=ModelLocalKVScope.MODEL,
                    rows=RowDriver.MAX_NUM_SEQS,
                    rows_reason="one KV row per scheduled sequence in the shared talker batch",
                    allocation_note="allocated with the batch engine on the first request",
                )
            ]
        return generator.model_local_kv_specs()

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
                vae = _build_audio_vae(vae_config)
                logger.info("Initialized AudioVAE from %s (sr=%d)", vae_path, vae_config.sample_rate)
                return vae, vae_path
            except Exception as e:
                logger.warning("Failed to initialize AudioVAE from %s: %s", vae_path, e)
                return None, None

        # HF hub fallback
        for subfolder in ("talker/vae", "vae"):
            try:
                vae_config = AudioVAEConfig.from_pretrained(model_path, subfolder=subfolder, trust_remote_code=True)
                vae = _build_audio_vae(vae_config)
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
        request_ids: list[str] | None = None,
        **kwargs,
    ) -> OmniOutput:
        """Return the next audio chunk, or the full waveform for offline calls.

        Engine requests carry request_ids and keep background generation alive
        across steps. Direct calls and profiling run the full loop inline.
        """
        if request_ids and self._batch_rows:
            return self._batched_forward(
                request_ids, runtime_additional_information, input_ids, inputs_embeds, kwargs.get("seq_token_counts")
            )
        additional_info = self._extract_additional_info(runtime_additional_information)
        params = self._resolve_generation_params(additional_info)
        if request_ids and params.stream_decode and not params.trim_trailing_silence:
            if len(request_ids) != 1:
                raise ValueError("Ming streaming Talker requires max_num_seqs=1")
            return self._stream_forward(request_ids[0], input_ids, inputs_embeds, additional_info, params)
        devices = [self.device.index] if self.device.type == "cuda" else []
        # Isolate seeded inference from other requests. Reference VAE sampling
        # uses its own generator, so cache hits do not change CFM's RNG stream.
        with torch.random.fork_rng(devices=devices, enabled=params.seed is not None):
            if params.seed is not None:
                torch.manual_seed(params.seed)
            voice = self._resolve_voice(additional_info)
            latents = self._generate_latents(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                text=additional_info.get("text", ""),
                params=params,
                voice=voice,
                native_fragments=additional_info.get("ming_text_fragments"),
            )
            return self._decode_to_output(
                latents, stream_decode=params.stream_decode, trim_silence=params.trim_trailing_silence
            )

    def get_pending_generation_request_ids(self) -> set[str]:
        return set(self._stream_jobs)

    def on_requests_finished(self, request_ids) -> None:
        for request_id in request_ids:
            job = self._stream_jobs.pop(request_id, None)
            if isinstance(job, MingBatchedStreamJob):
                job.close(self._batch_engine)
            elif job is not None:
                job.close()

    def _batched_forward(self, request_ids, runtime_infos, input_ids, inputs_embeds, token_counts) -> OmniOutput:
        """Advance every scheduled request through the shared AR batch.

        New requests are resolved here and handed to the engine thread; the
        step then waits once for the batch (as ``MingStreamingJob`` does per
        request) and VAE-decodes each request's newly arrived latents.
        """
        if self.audio_vae is None:
            raise RuntimeError("Batched Talker requires AudioVAE")
        engine = self._ensure_batch_engine()
        offsets = [0]
        for count in token_counts or []:
            offsets.append(offsets[-1] + int(count))
        jobs: list[MingBatchedStreamJob] = []
        for index, request_id in enumerate(request_ids):
            job = self._stream_jobs.get(request_id)
            if job is None:
                info = self._extract_additional_info([runtime_infos[index]] if runtime_infos else None)
                span = slice(offsets[index], offsets[index + 1]) if len(offsets) > index + 1 else slice(None)
                job = self._start_batched_job(
                    info,
                    input_ids[span] if input_ids is not None else None,
                    inputs_embeds[span] if inputs_embeds is not None else None,
                )
                self._stream_jobs[request_id] = job
            jobs.append(job)

        poll_seconds = 0.1
        if self.config.low_latency_streaming:
            engine.wait_for_output(lambda: any(job.urgent() for job in jobs), poll_seconds)
        else:
            # Legacy collection waits one polling interval, then drains.
            engine.wait_for_output(
                lambda: all(job.final_published for job in jobs) or any(job.error is not None for job in jobs),
                poll_seconds,
            )
        if engine.error is not None:
            raise RuntimeError("Ming talker batch engine failed") from engine.error

        audios = []
        finished = []
        for request_id, job in zip(request_ids, jobs, strict=True):
            try:
                audio, last = job.next_audio()
            except Exception:
                # One bad request must not fail the whole batched step.
                logger.exception("Ming talker request %s failed", request_id)
                audio, last = torch.empty(1, 1, 0, device=self.device), True
            audios.append(audio.detach().float())
            if last:
                finished.append(request_id)
        self.on_requests_finished(finished)
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"audio": audios, "sr": torch.tensor(self.audio_vae.config.sample_rate)},
        )

    def _ensure_batch_engine(self) -> MingTalkerBatchEngine:
        if self._batch_engine is None:
            engine = MingTalkerBatchEngine(
                self.audio_generator,
                max_rows=self._batch_rows,
                device=self.device,
                use_cuda_graphs=self._use_cuda_graphs,
            )
            engine.start()
            self._batch_engine = engine
        return self._batch_engine

    def _start_batched_job(self, info, input_ids, inputs_embeds) -> MingBatchedStreamJob:
        params = self._resolve_generation_params(info)
        streaming = params.stream_decode and not params.trim_trailing_silence

        def decode_full(latents):
            output = self._decode_to_output(
                latents, stream_decode=params.stream_decode, trim_silence=params.trim_trailing_silence
            )
            audio = output.multimodal_outputs.get("audio")
            return audio if audio is not None else torch.empty(1, 1, 0, device=self.device)

        job = MingBatchedStreamJob(
            self.audio_vae,
            self.device,
            low_latency_streaming=self.config.low_latency_streaming,
            decode_full=None if streaming else decode_full,
        )
        if inputs_embeds is not None:
            # Runner inputs are reusable buffers; the engine must own them.
            inputs_embeds = inputs_embeds.clone()
            if inputs_embeds.ndim == 2:
                inputs_embeds = inputs_embeds.unsqueeze(0)
        try:
            voice = self._resolve_voice(info)
            segments = self._tts_segments(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                text=info.get("text", ""),
                params=params,
                voice=voice,
                native_fragments=info.get("ming_text_fragments"),
            )
        except Exception as exc:
            job.fail(exc)
            return job
        if not segments:
            job.final_published = True
            return job
        job.request = BatchedTalkerRequest(
            segments,
            publish=job.publish,
            fail=job.fail,
            prompt_wav_lat=voice.prompt_wav_lat,
            min_steps=params.min_steps,
            stop_threshold=params.stop_threshold,
            cfg=params.cfg,
            sigma=params.sigma,
            temperature=params.temperature,
            seed=params.seed,
        )
        self._batch_engine.submit(job.request)
        return job

    def _stream_forward(self, request_id, input_ids, inputs_embeds, info, params) -> OmniOutput:
        job = self._stream_jobs.get(request_id)
        if job is None:
            if self._stream_jobs:
                raise ValueError("Ming streaming Talker supports one active request per worker")
            if self.audio_vae is None:
                raise RuntimeError("Streaming Talker requires AudioVAE")
            # Runner inputs are reusable buffers; the producer must own them.
            input_ids = input_ids.clone() if input_ids is not None else None
            inputs_embeds = inputs_embeds.clone() if inputs_embeds is not None else None

            def produce(on_latent):
                devices = [self.device.index] if self.device.type == "cuda" else []
                with torch.random.fork_rng(devices=devices, enabled=params.seed is not None):
                    if params.seed is not None:
                        torch.manual_seed(params.seed)
                    voice = self._resolve_voice(info)
                    self._generate_latents(
                        input_ids=input_ids,
                        inputs_embeds=inputs_embeds,
                        text=info.get("text", ""),
                        params=params,
                        voice=voice,
                        on_latent=on_latent,
                        native_fragments=info.get("ming_text_fragments"),
                    )

            job = MingStreamingJob(
                produce, self.audio_vae, self.device, low_latency_streaming=self.config.low_latency_streaming
            )
            self._stream_jobs[request_id] = job
        try:
            audio, last = job.next_audio()
            output = OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "audio": audio.detach().float().cpu(),
                    "sr": torch.tensor(self.audio_vae.config.sample_rate),
                },
            )
        except BaseException:
            self.on_requests_finished([request_id])
            raise
        if last:
            self.on_requests_finished([request_id])
        return output

    @staticmethod
    def _extract_additional_info(
        runtime_additional_information: list[dict] | None,
    ) -> dict[str, Any]:
        if runtime_additional_information and len(runtime_additional_information) > 0:
            info = runtime_additional_information[0] or {}
            # Async chunk transport nests the per-chunk talker inputs in the
            # structured payload's generic additional_information field.
            return info.get("additional_information", info)
        return {}

    def _resolve_generation_params(self, additional_info: dict[str, Any]) -> _GenerationParams:
        return resolve_generation_params(additional_info, self.config)

    def _resolve_voice(self, additional_info: dict[str, Any]) -> _VoiceContext:
        spk_emb = additional_info.get("spk_emb", None)
        prompt_text = additional_info.get("prompt_text", None)
        prompt_wav_lat = additional_info.get("prompt_wav_lat", None)
        prompt_wav_emb = additional_info.get("prompt_wav_emb", None)
        already_projected = False

        if (
            self.config.strict_weight_loading
            or "speaker_conditioning" in additional_info
            or "prompt_audio_mode" in additional_info
            or "prompt_wav" in additional_info
            or "reference_audio" in additional_info
            or has_reference_audio(additional_info)
        ):
            return self._resolve_formal_voice(additional_info)

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

    def _resolve_formal_voice(self, info: dict[str, Any]) -> _VoiceContext:
        spk_emb = info.get("spk_emb")
        reference = info.get("prompt_wav")
        reference_audio = unpack_reference_audio(info)
        preset = self.voice_presets.get(info.get("voice_name")) or {}
        has_reference = spk_emb is not None or bool(reference) or bool(reference_audio) or bool(preset)
        mode, audio_mode = resolve_conditioning_modes(info, has_reference=has_reference)
        need_speaker = mode == "prompt" and spk_emb is None
        need_latents = audio_mode == "continuation"
        projected = False
        if (reference or reference_audio) and (need_speaker or need_latents):
            preset = self.voice_presets.resolve_reference(
                reference,
                reference_audio=reference_audio,
                device=self.device,
                dtype=self.dtype,
                need_speaker=need_speaker,
                need_latents=need_latents and info.get("prompt_wav_lat") is None,
            )
        if mode == "prompt":
            if spk_emb is None:
                spk_emb = preset.get("spk_emb")
                projected = True
            if spk_emb is None or isinstance(spk_emb, list) and not spk_emb:
                raise ValueError("Reference did not provide a speaker embedding")
        elif mode == "zero":
            spk_emb = [torch.zeros(1, self.spk_head.in_features, device=self.device, dtype=self.dtype)]
        elif mode == "mute":
            spk_emb = [torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)]
            projected = True
        else:
            spk_emb = None
        # Explicit off/instruct must not be replaced by the legacy zero slot.
        if spk_emb is None:
            projected = True
        latents = info.get("prompt_wav_lat")
        embeddings = info.get("prompt_wav_emb")
        if latents is None:
            latents = preset.get("prompt_wav_lat")
        if embeddings is None:
            embeddings = preset.get("prompt_wav_emb")
        if need_latents and latents is not None:
            latents = torch.as_tensor(latents, device=self.device, dtype=self.dtype)
            if latents.ndim != 3 or latents.shape[0] != 1 or latents.shape[-1] != self.latent_dim:
                raise ValueError("prompt_wav_lat must have shape [1, frames, latent_dim]")
            if latents.shape[1] % self.patch_size:
                raise ValueError("Reference latent frames must be divisible by talker patch_size")
            if embeddings is None:
                patches = latents.reshape(-1, self.patch_size, self.latent_dim)
                embeddings = self.aggregator(patches).reshape(1, -1, self.hidden_size)
        if need_latents and embeddings is not None:
            embeddings = torch.as_tensor(embeddings, device=self.device, dtype=self.dtype)
            if embeddings.ndim != 3 or embeddings.shape[0] != 1 or embeddings.shape[-1] != self.hidden_size:
                raise ValueError("prompt_wav_emb must have shape [1, patches, hidden_size]")
        return _VoiceContext(
            spk_emb=spk_emb,
            prompt_text=(info.get("prompt_text") if info.get("prompt_text") is not None else preset.get("prompt_text"))
            if need_latents
            else None,
            prompt_wav_lat=latents if need_latents else None,
            prompt_wav_emb=embeddings if need_latents else None,
            already_projected=projected,
        )

    def _project_spk_emb(
        self, spk_emb: Any, already_projected: bool, use_zero_spk_emb: bool
    ) -> list[torch.Tensor] | None:
        if spk_emb is None:
            if use_zero_spk_emb and not already_projected:
                return [torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)]
            return None

        if already_projected:
            return spk_emb if isinstance(spk_emb, list) else [spk_emb]

        if isinstance(spk_emb, torch.Tensor):
            tensors = [row.unsqueeze(0) for row in spk_emb.reshape(-1, spk_emb.shape[-1])]
        elif isinstance(spk_emb, list) and spk_emb and isinstance(spk_emb[0], int | float):
            tensors = [torch.as_tensor(spk_emb).unsqueeze(0)]
        elif isinstance(spk_emb, list):
            tensors = [torch.as_tensor(t).reshape(1, -1) for t in spk_emb]
        else:
            tensors = [spk_emb]
        if any(t.shape[-1] != self.spk_head.in_features for t in tensors):
            raise ValueError(f"Raw speaker embeddings must have {self.spk_head.in_features} features")
        return [self.spk_head(t.to(device=self.device, dtype=self.dtype)) for t in tensors]

    def _tts_segments(
        self,
        *,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        text: str,
        params: _GenerationParams,
        voice: _VoiceContext,
        native_fragments: list[str] | None = None,
    ) -> list[TalkerSegment]:
        """Build the prompt embeddings and step budget of each AR pass."""
        if inputs_embeds is not None:
            # Caller pre-built embeddings — run a single AR pass.
            return [TalkerSegment(inputs_embeds, params.max_steps)]

        spk_emb = self._project_spk_emb(voice.spk_emb, voice.already_projected, params.use_zero_spk_emb)
        normalizer = self._upstream_tn if params.normalize_text and params.tn_mode == "upstream" else None
        if native_fragments is not None:
            # Producer-side fragments were already scrubbed and segmented.
            # Linguistic normalization must run exactly once, without splitting
            # the same fragment again at receiver-dependent boundaries.
            text_segments = [
                normalize_native_fragment(segment, params.prompt, tn_mode=params.tn_mode, normalizer=normalizer)
                if params.normalize_text
                else segment
                for segment in native_fragments
            ]
            text_segments = [segment for segment in text_segments if segment.strip()]
            if not text_segments:
                return []
        else:
            if text and params.scrub_markup:
                from .front_toolkit import strip_web_markup

                text = strip_web_markup(text)
                if not any(char.isalnum() for char in text):
                    raise ValueError("Ming frontend emitted no speakable text after markup removal")
            text_segments = (
                segment_and_normalize(
                    text, max_length=params.max_text_length, tn_mode=params.tn_mode, normalizer=normalizer
                )
                if text and params.normalize_text
                else [text]
                if text
                else []
            )

        if not text_segments:
            # vLLM passes 1D input_ids; Qwen2Model expects (batch, seq).
            inputs_embeds = self.model.get_input_embeddings()(input_ids.to(self.device)).unsqueeze(0)
            return [TalkerSegment(inputs_embeds, params.max_steps)]

        segments = []
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
            max_steps = (
                self.audio_generator.duration_capped_steps(len(segment), params.max_steps)
                if params.duration_limit
                else params.max_steps
            )
            segments.append(TalkerSegment(seg_embeds, max_steps))
        return segments

    def _generate_latents(
        self,
        *,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        text: str,
        params: _GenerationParams,
        voice: _VoiceContext,
        on_latent=None,
        native_fragments: list[str] | None = None,
    ) -> list[torch.Tensor]:
        segments = self._tts_segments(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            text=text,
            params=params,
            voice=voice,
            native_fragments=native_fragments,
        )
        all_latents: list[torch.Tensor] = []
        for segment_index, segment in enumerate(segments):
            all_latents.extend(
                self.audio_generator.generate_latents(
                    inputs_embeds=segment.embeds,
                    prompt_wav_lat=voice.prompt_wav_lat,
                    max_steps=segment.max_steps,
                    min_new_token=params.min_steps,
                    stop_threshold=params.stop_threshold,
                    on_latent=(
                        lambda latent, last, *ready: on_latent(
                            latent, last and segment_index == len(segments) - 1, *ready
                        )
                    )
                    if on_latent is not None
                    else None,
                    cfg=params.cfg,
                    sigma=params.sigma,
                    temperature=params.temperature,
                    use_static_cache=params.use_static_cache,
                )
            )
        return all_latents

    def _decode_to_output(
        self, latents: list[torch.Tensor], *, stream_decode: bool, trim_silence: bool = True
    ) -> OmniOutput:
        multimodal_outputs: dict[str, Any] = {}
        if latents and self.audio_vae is not None:
            waveform = self.audio_generator.decode_to_waveform(latents, stream_decode=stream_decode)
            if trim_silence:
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

        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(
            weights,
            mapper=WeightsMapper(
                orig_to_new_prefix={"audio_vae.": None}, orig_to_new_substr={"rotary_embed.inv_freq": None}
            ),
        )
        logger.info("Loaded %d talker weights from checkpoint", len(loaded))

        if self.audio_vae is not None and self._vae_weight_source is not None:
            loaded.update(self._load_vae_weights())

        if self.config.strict_weight_loading:
            if self.audio_vae is None:
                raise RuntimeError("Formal TTS requires the checkpoint's AudioVAE")
            missing = set(dict(self.named_parameters())) - loaded
            if missing:
                raise RuntimeError(f"Incomplete formal talker weights ({len(missing)} missing): {sorted(missing)[:20]}")
        else:
            # Ming2 presets depend on the loaded VAE and speaker projection.
            # Converted Ming3 exports use per-request IP/reference conditioning.
            try:
                self.voice_presets.load_presets_from_manifest(device=self.device, dtype=self.dtype)
            except (OSError, ValueError) as exc:
                logger.warning("Voice preset loading failed (non-fatal): %s", exc)

        return loaded

    def _iter_talker_safetensors(self) -> Iterable[tuple[str, torch.Tensor]]:
        """Yield (name, tensor) pairs from talker/model*.safetensors."""
        model_root = resolve_model_to_local_path(
            self._model_path,
            allow_download=True,
            allow_patterns=["model*.safetensors", "talker/model*.safetensors"],
            cache_dir=self.vllm_config.load_config.download_dir,
        )

        # Nested talker/ layout first, then a flat checkpoint at the root.
        for candidate in (os.path.join(model_root, "talker"), model_root):
            sf_files = sorted(glob_module.glob(os.path.join(candidate, "model*.safetensors")))
            if sf_files:
                for sf_path in sf_files:
                    yield from load_file(sf_path, device="cpu").items()
                return

        raise RuntimeError(f"No talker safetensors found under {model_root}. Expected talker/model*.safetensors.")

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
