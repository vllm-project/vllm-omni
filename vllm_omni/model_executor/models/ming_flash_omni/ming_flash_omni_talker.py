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
from transformers import AutoTokenizer, Qwen2Config
from transformers.utils.hub import cached_file
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.model_loader.weight_utils import resolve_model_to_local_path
from vllm_omni.model_executor.models.common.ming.aggregator import Aggregator
from vllm_omni.model_executor.models.common.ming.audio_vae import AudioVAE
from vllm_omni.model_executor.models.ming_tts.constants import SPEAKER_EMBEDDING_DIM
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.utils import add_prefix_to_loaded_weights
from vllm_omni.transformers_utils.configs.ming_flash_omni import (
    MingFlashOmniTalkerConfig,
    resolve_ming_talker_config,
)

from .prompt_utils import DEFAULT_MAX_TEXT_LENGTH, resolve_ming_prompt_fields
from .talker_module import (
    CFM,
    DiT,
    MingAudioGenerator,
    build_tts_input,
    resolve_audio_vae_config,
)
from .talker_request_state import (
    MingTalkerRequestState,
    MingTalkerStateManager,
)
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
    min_steps: int
    max_steps: int
    seed: int | None
    use_zero_spk_emb: bool
    max_text_length: int
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

        self.vllm_config = vllm_config
        root_config = vllm_config.model_config.hf_config
        self.has_preprocess = True
        self.has_postprocess = False
        self.requires_full_prefix_cached_hidden_states = False

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
        self.llm_config = llm_config
        self.hidden_size = llm_config.hidden_size
        self.latent_dim = config.latent_dim
        self.patch_size = config.patch_size
        self.his_patch_size = config.history_patch_size
        self.cfg_strength = config.cfg_strength

        self.model = self._init_paged_backbone(vllm_config=vllm_config, prefix=prefix)
        self.cfm = CFM(
            DiT(llm_input_dim=self.hidden_size, **config.flowmodel),
            steps=config.steps,
        )
        # config.aggregator still wins `in_channels` if the checkpoint states it.
        self.aggregator = Aggregator(
            llm_input_dim=self.hidden_size,
            **{"in_channels": self.latent_dim, **config.aggregator},
        )
        self.stop_head = nn.Linear(self.hidden_size, 2, bias=True)
        self.spk_head = nn.Linear(SPEAKER_EMBEDDING_DIM, self.hidden_size, bias=True)

        # AudioVAE
        self.audio_vae, self._vae_weight_source = self._init_audio_vae(config, self.talker_dir, model_path)

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
        )
        self.state_manager = MingTalkerStateManager()
        self.voice_presets = VoicePresetRegistry(
            talker_dir=self.talker_dir,
            model_path=self._model_path,
            download_dir=vllm_config.load_config.download_dir,
            audio_vae=self.audio_vae,
            aggregator=self.aggregator,
            spk_head=self.spk_head,
            patch_size=self.patch_size,
        )
        self._pending_requests: list[tuple[str, bool, int]] = []
        self._pending_state_creations: set[str] = set()
        self._pending_prefill_done_updates: dict[str, bool] = {}
        self._results_queue: list[tuple[str, torch.Tensor | None]] = []
        self._audio_queue: list[tuple[str, dict[str, Any] | None]] = []

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
        Probing is shared with the prompt-wav geometry derivation
        (``resolve_ming_talker_config``) so both read the same config.
        """
        resolved = resolve_ming_talker_config(config, talker_dir, model_path)
        if resolved is None:
            raise ValueError(
                f"Cannot resolve MingFlashOmniTalkerConfig. The root config "
                f"is {type(config).__name__}, and talker/config.json was not "
                f"found at {talker_dir} or via HF hub"
            )
        return resolved

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
        (repo_id, subfolder) tuple for HF hub downloads, or None. Config
        probing is shared with the prompt-wav geometry derivation
        (``resolve_audio_vae_config``) so both agree on which VAE applies.
        """
        resolved = resolve_audio_vae_config(config.audio_vae_path, talker_dir, model_path)
        if resolved is None:
            logger.info("AudioVAE not found; waveform decoding unavailable")
            return None, None
        vae_config, weight_source = resolved
        try:
            vae = AudioVAE(vae_config)
        except Exception as e:
            logger.warning("Failed to initialize AudioVAE from %s: %s", weight_source, e)
            return None, None
        logger.info("Initialized AudioVAE from %s (sr=%d)", weight_source, vae_config.sample_rate)
        return vae, weight_source

    def sample(self, logits: torch.Tensor, sampling_metadata):
        return None

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        return self._input_embedder()(input_ids)

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors | None:
        return self.model.make_empty_intermediate_tensors(batch_size, dtype, device)

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict[str, object]]:
        info: dict[str, object] = {"text": "dummy", "use_zero_spk_emb": True, "max_steps": 1}
        return [info for _ in range(num_reqs)]

    # Scheduler-driven talker hooks: vLLM owns the Qwen2 paged KV cache, while
    # MingTalkerRequestState carries the non-KV audio-side state across steps.

    def _init_paged_backbone(self, vllm_config: VllmConfig, prefix: str) -> Qwen2Model:
        """Construct the talker LLM with vLLM's native Qwen2Model."""
        llm_vllm_config = _replace_hf_config(vllm_config, self.llm_config)
        return Qwen2Model(vllm_config=llm_vllm_config, prefix=maybe_prefix(prefix, "model"))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict] | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        """One scheduler step over a batch of requests.

        Flow per call:
          1. resolve which req_ids are prefill vs decode (from scheduler meta);
          2. Phase A: hidden = self.model(input_ids, positions, inputs_embeds);
          3. Phase B: per request, run _talker_audio_step(...) using its state;
          4. emit next-step inputs_embeds (aggregator output) + stop signal;
          5. finalize finished requests, returning audio via OmniOutput.
        """
        try:
            model_output = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
            )
        except Exception:
            self._abort_pending_preprocess_batch()
            raise
        if isinstance(model_output, IntermediateTensors):
            self._commit_pending_step()
            return model_output

        hidden_states = model_output[0] if isinstance(model_output, tuple) else model_output
        try:
            self._validate_native_hidden_states(hidden_states)
        except Exception:
            self._abort_pending_preprocess_batch()
            raise
        # Phase B (CFM audio stepping / stop detection / finalize) intentionally
        # does NOT run here. With CUDA graphs enabled the captured forward is
        # replayed as recorded GPU kernels only, so host-side python placed in
        # this method would be silently skipped on every replayed step. The
        # scheduler rows accumulated by ``preprocess`` stay in
        # ``_pending_requests`` until ``compute_logits`` — which vLLM always
        # executes eagerly per step — consumes them in
        # ``_run_pending_audio_steps``.
        return hidden_states

    def _run_pending_audio_steps(self, rows: torch.Tensor) -> None:
        """Run one CFM audio step per pending scheduler row.

        ``rows`` is the logits-index-gathered hidden state handed to
        ``compute_logits``: one row per scheduled request in batch order, each
        already the last position of its span. Kept out of ``forward`` so it
        runs host-side even under CUDA graph replay; the talker stage must
        therefore avoid FULL cudagraphs (PIECEWISE is fine) because FULL
        capture records ``compute_logits`` GPU ops as well.
        """
        if not self._pending_requests:
            return
        try:
            if len(self._pending_requests) > int(rows.shape[0]):
                raise RuntimeError(
                    "Ming native talker scheduler row mismatch: "
                    f"pending={len(self._pending_requests)}, hidden_rows={int(rows.shape[0])}"
                )
            self._validate_unique_pending_requests()

            result_req_ids: list[str] = []
            stop_by_req: dict[str, torch.Tensor | None] = {}
            finalized_outputs: list[tuple[str, dict[str, Any]]] = []
            for row_idx, (req_id, should_audio_step, _span_len) in enumerate(self._pending_requests):
                req_hidden = rows[row_idx : row_idx + 1]
                result_req_ids.append(req_id)
                if not should_audio_step:
                    stop_by_req[req_id] = None
                    continue
                state = self.state_manager.get(req_id)
                if state.finished:
                    stop_by_req[req_id] = self._make_stop_token_logits(req_hidden)
                    continue

                # One CFM audio step per request (no cross-request batching:
                # that is a separate perf track under RFC #4129).
                _gen_lat, _next_inputs, stop_out = self._talker_audio_step(state, req_hidden[-1])
                stop_hit = bool(_stop_decision_mask(stop_out)[0])
                finished = self._request_should_stop(stop_hit, state.step, state.min_steps, state.max_steps)
                # Delay the stop token by one step. Within a runner step the
                # multimodal output (``make_omni_output``) is assembled BEFORE
                # ``compute_logits`` runs, so audio finalized here can only
                # ship with the NEXT step's output. Emitting CONTINUE now and
                # STOP on the following step (via the ``state.finished`` branch
                # above) guarantees the finalized audio travels with the
                # stop-token step instead of being dropped alongside an
                # already-finished request.
                stop_logits = torch.full_like(stop_out[:1], float("-inf"))
                stop_logits[:, 0] = 0.0
                stop_by_req[req_id] = stop_logits.detach()
                if finished:
                    output = self._finalize_request(state)
                    finalized_outputs.append((req_id, output.multimodal_outputs))

            for req_id, multimodal_outputs in finalized_outputs:
                state = self.state_manager.get(req_id)
                state.finished = True
                self._audio_queue.append((req_id, multimodal_outputs))

            for req_id in result_req_ids:
                self._results_queue.append((req_id, stop_by_req.get(req_id)))
        except Exception:
            self._abort_pending_preprocess_batch()
            raise

        self._commit_pending_step()

    def _commit_pending_step(self) -> None:
        """Drop per-step scheduler-row tracking once the step outcome is settled."""
        self._pending_requests.clear()
        self._pending_state_creations.clear()
        self._pending_prefill_done_updates.clear()

    def _abort_pending_preprocess_batch(self) -> None:
        """Roll back request state prepared for a model step that failed."""
        for req_id, prefill_done in self._pending_prefill_done_updates.items():
            if req_id in self.state_manager:
                self.state_manager.get(req_id).prefill_done = prefill_done
        for req_id in self._pending_state_creations:
            self.state_manager.evict(req_id)
        self._commit_pending_step()

    def _record_pending_prefill_done_update(self, state: MingTalkerRequestState) -> None:
        self._pending_prefill_done_updates.setdefault(state.req_id, bool(state.prefill_done))

    def _validate_native_hidden_states(self, hidden_states: torch.Tensor) -> None:
        if not isinstance(hidden_states, torch.Tensor):
            raise RuntimeError(f"Ming native talker expected tensor hidden states, got {type(hidden_states).__name__}")
        if hidden_states.ndim != 2:
            raise RuntimeError(
                "Ming native talker hidden-state shape mismatch: "
                f"shape={tuple(hidden_states.shape)}, expected=(scheduler_rows, {self.hidden_size})"
            )
        if int(hidden_states.shape[-1]) != int(self.hidden_size):
            raise RuntimeError(
                "Ming native talker hidden-state size mismatch: "
                f"got {int(hidden_states.shape[-1])}, expected {int(self.hidden_size)}"
            )

    def _validate_unique_pending_requests(self) -> None:
        seen: set[str] = set()
        duplicates: list[str] = []
        for req_id, _should_audio_step, _span_len in self._pending_requests:
            if req_id in seen:
                duplicates.append(req_id)
            else:
                seen.add(req_id)
        if duplicates:
            raise RuntimeError(f"Ming native talker duplicate scheduler rows: request_ids={duplicates}")

    @staticmethod
    def _make_stop_token_logits(reference: torch.Tensor) -> torch.Tensor:
        logits = torch.full(
            (1, 2),
            float("-inf"),
            device=reference.device,
            dtype=reference.dtype,
        )
        logits[:, 1] = 0.0
        return logits

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Prepare one scheduled AR span for the native-paged talker path."""
        additional_information = info_dict.get("additional_information")
        if isinstance(additional_information, dict):
            merged: dict[str, Any] = {k: v for k, v in info_dict.items() if k != "additional_information"}
            for k, v in additional_information.items():
                merged.setdefault(k, v)
            info_dict = merged

        req_id = str(info_dict.get("request_id", "default"))
        span_len = int(input_ids.reshape(-1).shape[0])
        if span_len <= 0:
            empty = self.embed_input_ids(input_ids.reshape(-1))
            return input_ids, empty, {}

        is_prefill_raw = info_dict.get("_omni_is_prefill")
        if isinstance(is_prefill_raw, bool):
            is_prefill = is_prefill_raw
        else:
            try:
                is_prefill = int(info_dict["_omni_num_computed_tokens"]) < int(info_dict["_omni_prompt_len"])
            except Exception:
                is_prefill = span_len > 1

        num_computed_tokens = int(info_dict.get("_omni_num_computed_tokens", 0) or 0)
        prompt_len = int(info_dict.get("_omni_prompt_len", span_len) or span_len)
        is_final_prefill = is_prefill and num_computed_tokens + span_len >= prompt_len

        try:
            if is_prefill:
                params = self._resolve_generation_params(info_dict)
                voice = self._resolve_voice(info_dict)
                text, segments = self._resolve_prefill_segments(info_dict, params)
                max_steps = self._native_prefill_max_steps(text, segments, params)
                embeds = self._build_native_prefill_embeds(
                    input_ids=input_ids,
                    input_embeds=input_embeds,
                    info_dict=info_dict,
                    params=params,
                    voice=voice,
                    segments=segments,
                )
                if req_id in self.state_manager and num_computed_tokens > 0:
                    # Later chunk of a chunked prefill: keep the existing state.
                    state = self.state_manager.get(req_id)
                    self._record_pending_prefill_done_update(state)
                else:
                    # First prefill chunk (or a restarted request): start fresh.
                    if req_id in self.state_manager:
                        self.state_manager.evict(req_id)
                    state = self._prefill_request(
                        req_id,
                        embeds.reshape(1, span_len, -1),
                        params,
                        voice,
                        max_steps=max_steps,
                    )
                    self._pending_state_creations.add(req_id)
                state.prefill_done = is_final_prefill
            else:
                try:
                    state = self.state_manager.get(req_id)
                except KeyError as exc:
                    raise RuntimeError(
                        "Ming native talker decode step is missing request state: "
                        f"request_id={req_id!r}, span_len={span_len}. "
                        "A native decode row must follow a prefill row for the same request "
                        "so audio state stays aligned with paged KV."
                    ) from exc

                if state.next_inputs_embed is not None:
                    embeds = state.next_inputs_embed.reshape(1, -1).to(device=input_ids.device, dtype=self.dtype)
                else:
                    embeds = self._coerce_scheduled_embeds(
                        input_ids=input_ids,
                        input_embeds=input_embeds,
                        provided=None,
                    )
        except Exception:
            self._abort_pending_preprocess_batch()
            raise

        input_ids_out = input_ids.reshape(-1).clone()
        input_ids_out[:] = 0
        self._pending_requests.append((req_id, (not is_prefill) or is_final_prefill, span_len))
        return input_ids_out, embeds.reshape(span_len, -1), {}

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata=None) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None

        self._run_pending_audio_steps(hidden_states)

        vocab_size = int(self.llm_config.vocab_size)
        logits = torch.full(
            (hidden_states.shape[0], vocab_size),
            float("-inf"),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if not self._results_queue:
            logits[:, 0] = 1.0
            return logits

        queued = self._results_queue
        if len(queued) > logits.shape[0]:
            req_ids = [req_id for req_id, _ in queued]
            raise RuntimeError(
                "Ming native talker logits row mismatch: "
                f"queued_results={len(queued)}, logits_rows={logits.shape[0]}, request_ids={req_ids}"
            )
        for row, (req_id, stop_logits) in enumerate(queued):
            if row >= logits.shape[0]:
                break
            if stop_logits is None:
                logits[row, 0] = 1.0
                continue
            if stop_logits.ndim == 0 or int(stop_logits.shape[-1]) < 2:
                raise RuntimeError(
                    "Ming native talker stop logits shape mismatch: "
                    f"request_id={req_id!r}, shape={tuple(stop_logits.shape)}, expected=(*, >=2)"
                )
            stop_logits = stop_logits.reshape(-1, stop_logits.shape[-1])[:1, :2].to(
                device=logits.device,
                dtype=logits.dtype,
            )
            logits[row, :2] = stop_logits[0]
        if len(queued) < logits.shape[0]:
            logits[len(queued) :, 0] = 1.0
        self._results_queue.clear()
        return logits

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        multimodal_outputs: dict[str, Any] = {}
        if self._audio_queue:
            req_ids: list[str] = []
            audios: list[torch.Tensor] = []
            audio_latents: list[torch.Tensor] = []
            sr_values: list[torch.Tensor] = []
            for req_id, payload in self._audio_queue:
                if not isinstance(payload, dict):
                    continue
                audio = payload.get("audio")
                if isinstance(audio, torch.Tensor):
                    req_ids.append(req_id)
                    audios.append(audio.reshape(-1))
                    sr = payload.get("sr")
                    if isinstance(sr, torch.Tensor):
                        sr_values.append(sr.detach().cpu())
                    elif sr is not None:
                        sr_values.append(torch.tensor(int(sr), dtype=torch.int32))
                    continue
                lat = payload.get("audio_latents")
                if isinstance(lat, torch.Tensor):
                    req_ids.append(req_id)
                    audio_latents.append(lat)

            if audios:
                multimodal_outputs["audio"] = audios
                multimodal_outputs["model_outputs"] = audios
                if sr_values:
                    multimodal_outputs["sr"] = sr_values
                multimodal_outputs["meta"] = {"req_id": req_ids, "sparse_audio": ["1"]}
            elif audio_latents:
                multimodal_outputs["audio_latents"] = audio_latents
                multimodal_outputs["meta"] = {"req_id": req_ids, "sparse_audio": ["1"]}
            self._audio_queue.clear()
        else:
            # The audio engine output processor remaps hidden payloads to
            # "audio". Mark non-final native steps as sparse-without-audio so
            # the runner does not send scheduled hidden states as audio chunks.
            multimodal_outputs["meta"] = {"req_id": [], "sparse_audio": ["1"]}

        return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs=multimodal_outputs)

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        finished = {str(req_id) for req_id in finished_req_ids}
        if not finished:
            return
        # The runner reports finished requests before the next step's forward,
        # and a finished request's audio already shipped one step before its
        # stop token (see _run_pending_audio_steps), so state is safe to evict
        # immediately. _audio_queue may still hold a payload for a request
        # aborted between its finalize step and the next make_omni_output.
        for req_id in finished:
            self.state_manager.evict(req_id)
        self._pending_requests = [item for item in self._pending_requests if item[0] not in finished]
        self._pending_state_creations.difference_update(finished)
        for req_id in finished:
            self._pending_prefill_done_updates.pop(req_id, None)
        self._results_queue = [item for item in self._results_queue if item[0] not in finished]
        self._audio_queue = [item for item in self._audio_queue if item[0] not in finished]

    def _build_native_prefill_embeds(
        self,
        *,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        info_dict: dict[str, Any],
        params: _GenerationParams,
        voice: _VoiceContext,
        segments: list[str],
    ) -> torch.Tensor:
        provided = info_dict.get("prefill_embeds", info_dict.get("inputs_embeds"))
        if isinstance(provided, torch.Tensor):
            return self._coerce_scheduled_embeds(
                input_ids=input_ids,
                input_embeds=input_embeds,
                provided=provided,
                offset=int(info_dict.get("_omni_num_computed_tokens", 0) or 0),
                prompt_len=int(info_dict.get("_omni_prompt_len", input_ids.reshape(-1).shape[0]) or 0),
            )

        if segments:
            spk_emb = self._project_spk_emb(voice.spk_emb, voice.already_projected, params.use_zero_spk_emb)
            built, _ = build_tts_input(
                tokenizer=self.tokenizer,
                embed_tokens=self._input_embedder(),
                device=input_ids.device,
                dtype=torch.bfloat16,
                text=segments[0],
                prompt=params.prompt,
                spk_emb=spk_emb,
                instruction=params.instruction,
                prompt_text=voice.prompt_text,
                prompt_wav_emb=voice.prompt_wav_emb,
            )
            return self._coerce_scheduled_embeds(
                input_ids=input_ids,
                input_embeds=input_embeds,
                provided=built,
                offset=int(info_dict.get("_omni_num_computed_tokens", 0) or 0),
                prompt_len=int(info_dict.get("_omni_prompt_len", input_ids.reshape(-1).shape[0]) or 0),
            )

        return self._coerce_scheduled_embeds(input_ids=input_ids, input_embeds=input_embeds, provided=None)

    def _coerce_scheduled_embeds(
        self,
        *,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        provided: torch.Tensor | None,
        offset: int = 0,
        prompt_len: int | None = None,
    ) -> torch.Tensor:
        span_len = int(input_ids.reshape(-1).shape[0])
        if provided is not None:
            embeds = provided
            if embeds.ndim == 3:
                embeds = embeds.reshape(-1, embeds.shape[-1])
            elif embeds.ndim != 2:
                raise ValueError(f"prefill embeds must be 2D or 3D, got shape={tuple(embeds.shape)}")
            embeds = embeds.to(device=input_ids.device, dtype=self.dtype)
            if prompt_len is not None and prompt_len > 0 and embeds.shape[0] > prompt_len:
                raise ValueError(
                    "native Ming talker prompt slots are shorter than the generated TTS "
                    f"prefill embeddings: prompt_len={prompt_len}, embed_len={embeds.shape[0]}. "
                    "Build prompt_token_ids with build_tts_prompt_token_ids before scheduling."
                )
            if embeds.shape[0] > span_len:
                start = max(0, min(int(offset), int(embeds.shape[0])))
                embeds = embeds[start : start + span_len]
        else:
            embeds = self.embed_input_ids(input_ids.reshape(-1).to(torch.long)).to(dtype=self.dtype)

        if embeds.shape[0] == span_len:
            return embeds
        if embeds.shape[0] > span_len:
            return embeds[:span_len].contiguous()

        pad_rows = span_len - int(embeds.shape[0])
        if embeds.shape[0] > 0:
            pad = embeds[-1:].expand(pad_rows, -1)
        else:
            pad = torch.zeros(pad_rows, self.hidden_size, device=input_ids.device, dtype=self.dtype)
        return torch.cat([embeds, pad], dim=0).contiguous()

    def _resolve_prefill_segments(self, info_dict: dict[str, Any], params: _GenerationParams) -> tuple[str, list[str]]:
        """Segment a prefill row's text once, for both the embeds and the step cap.

        Segmenting separately in each consumer would let the prefill length and
        the duration cap drift apart on any change to the segmenter.
        """
        text = str(info_dict.get("text", "") or "")
        if not text:
            return "", []
        return text, segment_and_normalize(text, max_length=params.max_text_length)

    def _native_prefill_max_steps(self, text: str, segments: list[str], params: _GenerationParams) -> int:
        """Apply Ming's duration cap to the native-paged request state."""
        if not text:
            return params.max_steps

        segment = segments[0] if segments else text
        return int(self.audio_generator.duration_capped_steps(len(segment), params.max_steps))

    def _prefill_request(
        self,
        req_id: str,
        inputs_embeds: torch.Tensor,
        params: _GenerationParams,
        voice: _VoiceContext,
        *,
        max_steps: int | None = None,
    ) -> MingTalkerRequestState:
        """Initialize per-request audio state at the prefill step.

        Builds his_lat (from prompt_wav_lat or zeros), seeds the RNG, resolves
        cfg/sigma/temperature/max_steps, registers state in the manager.
        """
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype
        his_lat = self.audio_generator._init_his_lat(voice.prompt_wav_lat, device, dtype)
        return self.state_manager.create(
            req_id,
            his_lat=his_lat,
            min_steps=params.min_steps,
            max_steps=params.max_steps if max_steps is None else int(max_steps),
            seed=params.seed,
            prefill_done=True,
            cfg=params.cfg,
            sigma=params.sigma,
            temperature=params.temperature,
            stream_decode=params.stream_decode,
            generator_device=device,
        )

    def _talker_audio_step(
        self,
        state: MingTalkerRequestState,
        last_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Phase B for one request: CFM sample + aggregator feedback.

        Runs the CFM step eagerly via ``MingAudioGenerator.cfm_sample_step``.
        Returns (gen_lat, next_inputs_embed, stop_out); advances
        state.his_lat / step.
        """
        if state.his_lat is None:
            raise RuntimeError(f"Ming talker state {state.req_id!r} has no latent history")

        hidden = _normalize_last_hidden_for_step(last_hidden).to(device=state.his_lat.device, dtype=state.his_lat.dtype)
        randn_tensor, sde_rnd = _sample_request_noise(
            state,
            steps=int(self.config.steps),
            patch_size=int(self.patch_size),
            latent_dim=int(state.his_lat.shape[-1]),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        gen_lat, next_inputs_embed, stop_out = self.audio_generator.cfm_sample_step(
            hidden,
            state.his_lat,
            cfg=state.cfg,
            sigma=state.sigma,
            temperature=state.temperature,
            randn_tensor=randn_tensor,
            sde_rnd=sde_rnd,
        )
        state.his_lat = self.audio_generator._update_his_lat(state.his_lat, gen_lat)
        state.all_latents.append(gen_lat)
        state.next_inputs_embed = next_inputs_embed
        state.step += 1
        return gen_lat, next_inputs_embed, stop_out

    @staticmethod
    def _request_should_stop(stop_hit: bool, step: int, min_steps: int, max_steps: int) -> bool:
        """Per-request stop decision, matching the original zero-based boundary.

        A model-signalled stop only takes effect after ``min_steps`` new tokens
        (``(step - 1) > min_steps``); ``max_steps`` forces a stop regardless.
        """
        return (stop_hit and (step - 1) > min_steps) or (step >= max_steps)

    def _finalize_request(self, state: MingTalkerRequestState) -> OmniOutput:
        """Decode accumulated latents to waveform and build OmniOutput.

        Reuses self._decode_to_output / AudioVAE. Request lifecycle updates
        happen only after all finished rows in the current scheduler step
        finalize successfully.
        """
        return self._decode_to_output(state.all_latents, stream_decode=state.stream_decode)

    @staticmethod
    def _extract_additional_info(
        runtime_additional_information: list[dict] | None,
    ) -> dict[str, Any]:
        if runtime_additional_information and len(runtime_additional_information) > 0:
            return runtime_additional_information[0] or {}
        return {}

    def _resolve_generation_params(self, additional_info: dict[str, Any]) -> _GenerationParams:
        # prompt/instruction/spk-emb fields are shared with the stage input
        # processor (slot sizing); the sampling knobs below are talker-only.
        # "omni"    : thinker -> talker hand-off with hardcoded defaults
        # "instruct": standalone TTS with caller-supplied sampling knobs
        is_omni, prompt, instruction, use_zero_spk_emb = resolve_ming_prompt_fields(additional_info)

        if is_omni:
            cfg = 2.0
            sigma = 0.25
            temperature = 0.0
            min_steps = 10
            max_steps = 200
        else:
            cfg = additional_info.get("cfg", self.cfg_strength)
            sigma = additional_info.get("sigma", 0.25)
            temperature = additional_info.get("temperature", 0.0)
            min_steps = int(additional_info.get("min_new_token", additional_info.get("min_steps", 10)))
            max_steps = int(additional_info.get("max_steps", additional_info.get("max_decode_steps", 200)))

        return _GenerationParams(
            prompt=prompt,
            instruction=instruction,
            cfg=cfg,
            sigma=sigma,
            temperature=temperature,
            min_steps=min_steps,
            max_steps=max_steps,
            use_zero_spk_emb=use_zero_spk_emb,
            seed=_optional_int(
                additional_info.get(
                    "ming_talker_seed",
                    additional_info.get("request_seed", additional_info.get("seed")),
                )
            ),
            max_text_length=int(additional_info.get("max_text_length", DEFAULT_MAX_TEXT_LENGTH)),
            stream_decode=bool(additional_info.get("stream_decode", True)),
        )

    def _resolve_voice(self, additional_info: dict[str, Any]) -> _VoiceContext:
        spk_emb = additional_info.get("spk_emb", None)
        prompt_text = additional_info.get("prompt_text", None)
        prompt_wav_lat = additional_info.get("prompt_wav_lat", None)
        prompt_wav_emb = additional_info.get("prompt_wav_emb", None)
        already_projected = False

        voice_name = additional_info.get("voice_name", None)
        # Native-paged scheduling reserves prompt-KV slots before preprocess
        # runs, so the prefill embeds must not exceed them. Inject the preset
        # only when the input processor signalled that it sized the slots for
        # one (non-native always injects); otherwise fall back to no preset.
        native_slots_reserved = additional_info.get("native_talker_prompt_wav_len") is not None
        # Preset geometry is cross-checked against the processor-side derived
        # metadata once at load time (VoicePresetRegistry._verify_derived_meta),
        # so the stamped slot counts are trusted here.
        if voice_name and spk_emb is None and voice_name in self.voice_presets and native_slots_reserved:
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

    def _input_embedder(self):
        if hasattr(self.model, "get_input_embeddings"):
            return self.model.get_input_embeddings()
        return self.model.embed_input_ids

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

    def _decode_to_output(
        self,
        latents: list[torch.Tensor],
        *,
        stream_decode: bool,
        hidden_rows: int = 1,
    ) -> OmniOutput:
        multimodal_outputs: dict[str, Any] = {}
        if latents and self.audio_vae is not None:
            waveform = self.audio_generator.decode_to_waveform(latents, stream_decode=stream_decode)
            if not stream_decode:
                waveform = self.audio_generator.trim_trailing_silence(waveform)
            multimodal_outputs["audio"] = waveform.detach().float().cpu()
            multimodal_outputs["sr"] = torch.tensor(
                [int(self.audio_vae.config.sample_rate)],
                dtype=torch.int32,
            )
        elif latents:
            all_lat = torch.cat(latents, dim=1)
            multimodal_outputs["audio_latents"] = all_lat.detach().float().cpu()

        text_hidden_states = torch.zeros(
            hidden_rows,
            self.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )
        return OmniOutput(text_hidden_states=text_hidden_states, multimodal_outputs=multimodal_outputs)

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

        return self._load_native_paged_weights(weights)

    def _load_native_paged_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        model_weights, component_weights = _partition_native_paged_weights(weights)

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["audio_vae.", "model."],  # VAE and native Qwen2 are loaded separately.
            skip_substrs=["rotary_embed.inv_freq"],
        )
        loaded = loader.load_weights(component_weights)
        model_loaded = self.model.load_weights(_strip_model_prefix(model_weights))
        loaded.update(add_prefix_to_loaded_weights(model_loaded, "model"))
        logger.info("Loaded %d native-paged talker weights from checkpoint", len(loaded))

        if self.audio_vae is not None and self._vae_weight_source is not None:
            loaded.update(self._load_vae_weights())

        try:
            self.voice_presets.load_presets_from_manifest(device=self.device, dtype=self.dtype)
        except Exception as e:  # pragma: no cover - best-effort
            logger.warning("Voice preset loading failed (non-fatal): %s", e)

        return loaded

    def _iter_talker_safetensors(self) -> Iterable[tuple[str, torch.Tensor]]:
        """Yield (name, tensor) pairs from talker/model*.safetensors."""
        model_root = resolve_model_to_local_path(
            self._model_path,
            allow_download=True,
            allow_patterns=["talker/model*.safetensors"],
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


def _partition_native_paged_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]]]:
    model_weights: list[tuple[str, torch.Tensor]] = []
    component_weights: list[tuple[str, torch.Tensor]] = []
    for name, tensor in weights:
        if name.startswith("model."):
            model_weights.append((name, tensor))
        else:
            component_weights.append((name, tensor))
    return model_weights, component_weights


def _strip_model_prefix(weights: Iterable[tuple[str, torch.Tensor]]) -> Iterable[tuple[str, torch.Tensor]]:
    for name, tensor in weights:
        if name.startswith("model."):
            name = name[len("model.") :]
        yield name, tensor


def _replace_hf_config(vllm_config: VllmConfig, hf_config: Any) -> VllmConfig:
    """Return a vLLM config whose model_config points at the talker Qwen2 config."""
    cloned = vllm_config.with_hf_config(hf_config)
    model_config = getattr(cloned, "model_config", None)
    if hasattr(model_config, "hf_text_config"):
        model_config.hf_text_config = hf_config
    return cloned


def _normalize_last_hidden_for_step(last_hidden: torch.Tensor) -> torch.Tensor:
    """Normalize one scheduler row to the CFM conditioning shape."""
    if last_hidden.ndim == 1:
        return last_hidden.reshape(1, 1, -1)
    if last_hidden.ndim == 2:
        return last_hidden[-1:, :].unsqueeze(0)
    if last_hidden.ndim == 3:
        return last_hidden[:, -1:, :]
    raise ValueError(f"last_hidden must be 1D, 2D, or 3D, got shape={tuple(last_hidden.shape)}")


def _sample_request_noise(
    state: MingTalkerRequestState,
    *,
    steps: int,
    patch_size: int,
    latent_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    randn_tensor = _randn_for_state(
        state,
        (1, patch_size, latent_dim),
        device=device,
        dtype=dtype,
    )
    sde_rnd = _randn_for_state(
        state,
        (steps, 1, patch_size, latent_dim),
        device=device,
        dtype=dtype,
    )
    return randn_tensor, sde_rnd


def _randn_for_state(
    state: MingTalkerRequestState,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = state.generator
    if generator is None:
        return torch.randn(shape, device=device, dtype=dtype)
    try:
        return torch.randn(shape, device=device, dtype=dtype, generator=generator)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Ming talker RNG for request {state.req_id!r} is not compatible with sampling on device {device}"
        ) from exc


def _stop_decision_mask(stop_out: torch.Tensor) -> torch.Tensor:
    """Per-row stop decision: the stop column (1) beats the continue column (0).

    ``stop_out`` is the softmaxed stop-head output; comparing the two columns
    is equivalent to thresholding the stop probability at 0.5 and also handles
    raw logits, so no probability-vs-logits detection is needed.
    """
    rows = stop_out.reshape(-1, stop_out.shape[-1])
    return rows[:, 1] > rows[:, 0]


def _optional_int(value: Any) -> int | None:
    """Parse optional scalar/list/tensor-ish ints from request metadata."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    if hasattr(value, "item"):
        value = value.item()
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
