# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Iterable
from types import MethodType
from typing import Any

import torch
import torch.nn as nn
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.modeling_outputs import BaseModelOutput
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.multimodal.inputs import MultiModalFeatureSpec

from vllm_omni.model_executor.models.funaudiochat.common import (
    ensure_funaudiochat_importable,
    register_funaudiochat_processor,
)

try:
    from vllm.model_executor.models.funaudiochat import (
        FunAudioChatForConditionalGeneration as VllmNativeFunAudioChatForConditionalGeneration,
    )
except ImportError:  # pragma: no cover - environment-specific dependency
    VllmNativeFunAudioChatForConditionalGeneration = None

_NativeFunAudioChatBase = (
    VllmNativeFunAudioChatForConditionalGeneration
    if VllmNativeFunAudioChatForConditionalGeneration is not None
    else nn.Module
)

logger = init_logger(__name__)

DEFAULT_SP_GEN_KWARGS = {
    "text_greedy": True,
    "only_crq_sampling": True,
    "disable_speech": False,
    "force_text_abos": True,
}

_OFFICIAL_CRQ_SAMPLING_DEFAULTS = {
    "repetition_penalty": 1.2,
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 0,
}

_AUDIO_TOKEN_IDS_KEY = "funaudiochat_audio_token_ids"
_CRQ_AUDIO_EMBEDS_KEY = "funaudiochat_crq_audio_embeds"
_CRQ_PAST_KEY_VALUES_KEY = "funaudiochat_crq_past_key_values"
_CURRENT_INPUT_TOKEN_ID_KEY = "funaudiochat_current_input_token_id"
_FORCE_AUDIO_BOS_KEY = "funaudiochat_force_audio_bos_pending"
_FINISH_SPEECH_KEY = "funaudiochat_finish_speech"
_GENERATE_SPEECH_KEY = "funaudiochat_generate_speech"
_SPEECH_IDS_KEY = "funaudiochat_speech_ids"
_TEXT_INPUT_IDS_KEY = "funaudiochat_text_input_ids"
_TEXT_SEQ_LEN_KEY = "funaudiochat_text_seq_len"


@register_funaudiochat_processor
class FunAudioChatForConditionalGeneration(_NativeFunAudioChatBase, SupportsMultiModal):
    supports_multimodal_raw_input_only = True
    supports_multimodal = True
    requires_raw_input_tokens = False
    input_modalities = "audio"
    pooler_output_buffer_keys = ("audio_token_ids",)
    # Ask the omni runner to forward the request's mm_features into the
    # per-request preprocess info_dict, so the prefill span can read the
    # user-uploaded audio (see _gather_user_audio_embeds / preprocess).
    wants_mm_features_in_preprocess = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        if VllmNativeFunAudioChatForConditionalGeneration is None:
            raise ImportError(
                "Installed vLLM does not expose a native FunAudioChat model. "
                "Upgrade vLLM to a build that includes "
                "`vllm.model_executor.models.funaudiochat`."
            )

        super().__init__(vllm_config=vllm_config, prefix=prefix)
        ensure_funaudiochat_importable()
        from funaudiochat.modeling_funaudiochat import FunAudioChatDecoder  # type: ignore

        self.audio_invert_tower = FunAudioChatDecoder(self.config.audio_config)
        self._patch_audio_invert_tower_sampling_step()
        self.sp_gen_kwargs = DEFAULT_SP_GEN_KWARGS.copy()
        self.has_preprocess = True
        self.has_postprocess = True
        # Opt into the async Omni output path (PR #4476): defer Omni
        # ModelRunnerOutput construction off the AR decode critical path so the
        # next GPU decode step can start earlier. FunAudioChat's stage-0 is
        # structurally a Qwen3-Omni talker: single model, has_postprocess, and
        # its postprocess reads live GPU hidden_states directly — so it uses the
        # same eager-postprocess + skip-hidden-D2H variant the talker uses.
        self.use_async_omni_output = True
        # postprocess() consumes live GPU hidden_states (the CRQ prefill warmup
        # at _run_audio_sidecar_prefill_warmup) and mutates per-request instance
        # state (_postprocess_cursor / _speech_state / _crq_gpu_state) in request
        # order. It cannot be deferred to the background output builder thread,
        # so run it eagerly on the main thread before the async D2H copy —
        # exactly why Qwen3-Omni talker keeps hidden_states.last on GPU before
        # the next decode step.
        self.eager_omni_postprocess_before_async_output = True
        # code2wav only needs the codec codes (audio_token_ids) shipped
        # downstream; the latent hidden-state D2H is dead weight here, so drop
        # it from the async snapshot payload.
        self.omni_pooler_payload_include_hidden = False
        self.have_multimodal_outputs = False
        self._batch_preprocess_in_progress = False
        self._batch_req_infos: list[dict[str, Any]] = []
        self._batch_sidecar_results: list[dict[str, Any]] = []
        self._postprocess_cursor = 0
        self._logged_stage0_backend = False
        # Per-request speech-span state owned by this model instance, NOT by the
        # runner's model_intermediate_buffer. The buffer can be reset/replaced by
        # the runner between steps (e.g. _update_intermediate_buffer merging the
        # preprocess update_dict), which would clobber the values
        # postprocess_sampled_tokens writes — breaking the force_text_abos ->
        # generate_speech handoff. Keeping state here makes it survive any buffer
        # churn. Cleared per-request on finish.
        self._speech_state: dict[str, dict[str, Any]] = {}
        # CRQ speech-sidecar KV (`crq_audio_embeds` + `crq_past_key_values`)
        # kept resident on GPU, keyed by request id -- mirrors
        # ``self._speech_state``'s lifecycle so it survives runner buffer churn
        # between decode steps. Without this the sidecar would, every step and
        # every active request, dump the whole (length-growing) KV to CPU and
        # haul it back to GPU (5 synchronous D2H/H2D copies per step per req),
        # which is the dominant stage-0 cost once speech generation starts.
        # Freed on finish_speech (see postprocess_sampled_tokens).
        self._crq_gpu_state: dict[str, dict[str, Any]] = {}
        # Generated CRQ codec history stays on the model device as well.  The
        # runner's intermediate buffer is allowed to retain a GPU reference,
        # but it is no longer the source of truth: otherwise every decode step
        # copies the complete, ever-growing speech history to CPU and back.
        self._speech_ids_gpu_state: dict[str, torch.Tensor] = {}

    @staticmethod
    def _move_nested_to_device(value: Any, device: torch.device) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device=device)
        if isinstance(value, tuple):
            return tuple(FunAudioChatForConditionalGeneration._move_nested_to_device(v, device) for v in value)
        if isinstance(value, list):
            return [FunAudioChatForConditionalGeneration._move_nested_to_device(v, device) for v in value]
        return value

    def _crq_resolve_gpu_kv(
        self,
        req_id: str | None,
        cached_audio_embeds: Any,
        cached_past_key_values: Any,
        device: torch.device,
    ) -> tuple[Any, Any]:
        # Source of truth is the per-request GPU-resident state. On first use
        # (or after finish_speech freed it) seed from the incoming buffer cache
        # (moved to GPU once); steady state reuses the resident tensors with
        # no CPU round-trip. ``req_id is None`` can't happen for real requests
        # (``_speech_state`` relies on the same key) but degrades gracefully to
        # a plain per-call device move with no cross-step caching.
        state = self._crq_gpu_state.get(req_id) if req_id is not None else None
        if state is not None:
            return state["embeds"], state["pkv"]
        return (
            self._move_nested_to_device(cached_audio_embeds, device),
            self._move_nested_to_device(cached_past_key_values, device),
        )

    def _crq_persist_gpu_kv(self, req_id: str | None, embeds: Any, pkv: Any) -> None:
        if req_id is None:
            return
        self._crq_gpu_state[req_id] = {"embeds": embeds, "pkv": pkv}

    def _resolve_gpu_speech_ids(
        self,
        req_id: str | None,
        cached_speech_ids: Any,
        device: torch.device,
    ) -> torch.Tensor:
        if req_id is not None:
            resident = self._speech_ids_gpu_state.get(req_id)
            if resident is not None:
                return resident
        speech_ids = self._as_2d_long_tensor(cached_speech_ids, device)
        if req_id is not None:
            self._speech_ids_gpu_state[req_id] = speech_ids
        return speech_ids

    def _persist_gpu_speech_ids(self, req_id: str | None, speech_ids: torch.Tensor) -> None:
        if req_id is not None:
            self._speech_ids_gpu_state[req_id] = speech_ids

    @staticmethod
    def _as_long_token_tensor(value: Any, device: torch.device) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=torch.long).reshape(-1)[-1:]
        return torch.as_tensor([value], dtype=torch.long, device=device)

    @staticmethod
    def _as_2d_long_tensor(value: Any, device: torch.device) -> torch.Tensor:
        if value is None:
            return torch.empty((1, 0), dtype=torch.long, device=device)
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=device, dtype=torch.long)
        else:
            tensor = torch.as_tensor(value, dtype=torch.long, device=device)
        if tensor.ndim == 0:
            tensor = tensor.reshape(1, 1)
        elif tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _patch_audio_invert_tower_sampling_step(self) -> None:
        if getattr(self.audio_invert_tower, "_vllm_omni_crq_generator_patched", False):
            return

        def _sampling_step_with_generator(
            decoder_self: nn.Module,
            logits: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            next_token_logits = logits[:, -1, :].to(copy=True, dtype=torch.float32, device=logits.device)
            next_token_scores = decoder_self.crq_logits_processor(
                torch.cat([decoder_self.crq_speech_ids, *decoder_self.crq_generate_tokens], dim=-1),
                next_token_logits,
            )

            if decoder_self.crq_do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            return next_tokens, logits

        self.audio_invert_tower.sampling_step = MethodType(_sampling_step_with_generator, self.audio_invert_tower)
        self.audio_invert_tower._vllm_omni_crq_generator_patched = True

    def _empty_audio_token_ids(self, device: torch.device) -> torch.Tensor:
        return torch.full(
            (1, int(self.config.audio_config.group_size)),
            -1,
            dtype=torch.long,
            device=device,
        )

    @staticmethod
    def _sampling_value_at(
        value: torch.Tensor | None,
        req_index: int,
        default: float,
    ) -> float:
        if value is None:
            return float(default)
        if value.ndim == 0:
            return float(value.item())
        if req_index >= value.shape[0]:
            return float(default)
        return float(value[req_index].item())

    @staticmethod
    def _resolve_text_seq_len(
        prev_text_seq_len: Any,
        span_len: int,
    ) -> tuple[int, int]:
        prev = int(prev_text_seq_len or 0)
        if span_len > 1:
            current = prev + span_len
            return current, current
        current = prev if prev > 0 else 1
        return current, current + 1

    @staticmethod
    def _resolve_next_speech_state(
        *,
        sampled_token_id: int,
        generate_speech: bool,
        finish_speech: bool,
        force_audio_bos_pending: bool,
        audio_bos_id: int,
        audio_eos_id: int,
    ) -> tuple[int, bool, bool]:
        if finish_speech:
            return audio_eos_id, False, False

        final_token_id = audio_bos_id if force_audio_bos_pending else sampled_token_id
        next_speech_active = generate_speech or final_token_id == audio_bos_id
        if final_token_id == audio_eos_id:
            next_speech_active = False

        return final_token_id, next_speech_active, False

    def _build_crq_sampling_config(
        self,
        sampling_metadata: Any,
        req_index: int,
    ) -> tuple[LogitsProcessorList, bool]:
        repetition_penalty = self._sampling_value_at(
            getattr(sampling_metadata, "repetition_penalties", None) if sampling_metadata is not None else None,
            req_index,
            _OFFICIAL_CRQ_SAMPLING_DEFAULTS["repetition_penalty"],
        )
        default_temperature = 0.0
        default_top_p = 1.0
        default_top_k = -1.0
        if self.sp_gen_kwargs["text_greedy"]:
            default_temperature = _OFFICIAL_CRQ_SAMPLING_DEFAULTS["temperature"]
            default_top_p = _OFFICIAL_CRQ_SAMPLING_DEFAULTS["top_p"]
            default_top_k = float(_OFFICIAL_CRQ_SAMPLING_DEFAULTS["top_k"])

        temperature = self._sampling_value_at(
            getattr(sampling_metadata, "temperature", None) if sampling_metadata is not None else None,
            req_index,
            default_temperature,
        )
        top_p = self._sampling_value_at(
            getattr(sampling_metadata, "top_p", None) if sampling_metadata is not None else None,
            req_index,
            default_top_p,
        )
        top_k = int(
            round(
                self._sampling_value_at(
                    getattr(sampling_metadata, "top_k", None) if sampling_metadata is not None else None,
                    req_index,
                    default_top_k,
                )
            )
        )

        if self.sp_gen_kwargs["text_greedy"] and temperature <= 0.0:
            temperature = float(_OFFICIAL_CRQ_SAMPLING_DEFAULTS["temperature"])
            if top_p >= 1.0:
                top_p = float(_OFFICIAL_CRQ_SAMPLING_DEFAULTS["top_p"])
            if top_k < 0:
                top_k = int(_OFFICIAL_CRQ_SAMPLING_DEFAULTS["top_k"])

        processors: list[Any] = []
        if repetition_penalty > 0.0 and abs(repetition_penalty - 1.0) > 1e-6:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))

        do_sample = temperature > 0.0
        if do_sample:
            if abs(temperature - 1.0) > 1e-6:
                processors.append(TemperatureLogitsWarper(temperature))
            if top_k > 0:
                processors.append(TopKLogitsWarper(top_k=top_k))
            if 0.0 < top_p < 1.0:
                processors.append(TopPLogitsWarper(top_p=top_p))

        return LogitsProcessorList(processors), do_sample

    def _get_stage0_backend(self) -> str:
        try:
            backend_cls = self.get_language_model().model.layers[0].self_attn.attn.get_attn_backend()
            backend_name = str(backend_cls.get_name())
        except (AttributeError, IndexError, TypeError):
            backend_name = "UNKNOWN"
        if not self._logged_stage0_backend:
            logger.debug("FunAudioChat stage-0 native language backend: %s", backend_name)
            self._logged_stage0_backend = True
        return backend_name

    def _run_audio_sidecar_step(
        self,
        hidden_state: torch.Tensor,
        current_input_token_id: torch.Tensor | int,
        speech_ids: torch.Tensor,
        cached_audio_embeds: Any,
        cached_past_key_values: Any,
        logits_processor: LogitsProcessorList,
        do_sample: bool,
        current_text_seq_len: int,
        req_id: str | None = None,
    ) -> dict[str, Any]:
        device = hidden_state.device
        text_embed = (
            self.get_language_model()
            .embed_input_ids(self._as_long_token_tensor(current_input_token_id, device))
            .reshape(1, 1, -1)
        )
        speech_inputs_embeds = hidden_state.reshape(1, 1, -1) + text_embed.detach()
        attention_mask = torch.ones((1, max(current_text_seq_len, 1)), dtype=torch.long, device=device)
        position_ids = torch.tensor([[max(current_text_seq_len - 1, 0)]], dtype=torch.long, device=device)

        embeds_gpu, pkv_gpu = self._crq_resolve_gpu_kv(
            req_id, cached_audio_embeds, cached_past_key_values, device
        )
        self.audio_invert_tower.crq_audio_embeds = embeds_gpu
        self.audio_invert_tower.crq_past_key_values = pkv_gpu
        self.audio_invert_tower.crq_do_sample = do_sample
        self.audio_invert_tower.crq_logits_processor = logits_processor
        self.audio_invert_tower.crq_speech_ids = speech_ids
        self.audio_invert_tower.crq_generate_forward(
            inputs_embeds=speech_inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )

        next_audio_tokens = self.audio_invert_tower.crq_generate_tokens.reshape(1, -1).to(dtype=torch.long)
        eos_token_id = int(self.config.audio_config.eos_token_id)
        finish_speech = bool((next_audio_tokens == eos_token_id).any().item())
        if finish_speech:
            next_audio_tokens = torch.full_like(next_audio_tokens, eos_token_id)

        updated_speech_ids = torch.cat([speech_ids, next_audio_tokens], dim=-1)
        self._persist_gpu_speech_ids(req_id, updated_speech_ids)
        # Persist the post-forward KV on GPU (per request); the returned
        # tensors stay on device so the runner buffer holds GPU refs (no D2H).
        embeds_out = self.audio_invert_tower.crq_audio_embeds
        pkv_out = self.audio_invert_tower.crq_past_key_values
        self._crq_persist_gpu_kv(req_id, embeds_out, pkv_out)
        return {
            _AUDIO_TOKEN_IDS_KEY: next_audio_tokens.detach(),
            _CRQ_AUDIO_EMBEDS_KEY: embeds_out,
            _CRQ_PAST_KEY_VALUES_KEY: pkv_out,
            _FINISH_SPEECH_KEY: finish_speech,
            _SPEECH_IDS_KEY: updated_speech_ids,
        }

    def _run_audio_sidecar_decode_warmup(
        self,
        hidden_state: torch.Tensor,
        current_input_token_id: torch.Tensor | int,
        speech_ids: torch.Tensor,
        cached_audio_embeds: Any,
        cached_past_key_values: Any,
        logits_processor: LogitsProcessorList,
        do_sample: bool,
        req_id: str | None = None,
    ) -> dict[str, Any]:
        device = hidden_state.device
        text_embed = (
            self.get_language_model()
            .embed_input_ids(self._as_long_token_tensor(current_input_token_id, device))
            .reshape(1, 1, -1)
        )
        speech_inputs_embeds = hidden_state.reshape(1, 1, -1) + text_embed.detach()

        embeds_gpu, pkv_gpu = self._crq_resolve_gpu_kv(
            req_id, cached_audio_embeds, cached_past_key_values, device
        )
        self.audio_invert_tower.crq_audio_embeds = embeds_gpu
        self.audio_invert_tower.crq_past_key_values = pkv_gpu
        self.audio_invert_tower.crq_do_sample = do_sample
        self.audio_invert_tower.crq_logits_processor = logits_processor
        self.audio_invert_tower.crq_speech_ids = speech_ids
        self.audio_invert_tower.crq_generate_forward(
            inputs_embeds=speech_inputs_embeds,
            return_dict=True,
        )
        embeds_out = self.audio_invert_tower.crq_audio_embeds
        pkv_out = self.audio_invert_tower.crq_past_key_values
        self._crq_persist_gpu_kv(req_id, embeds_out, pkv_out)
        return {
            _CRQ_AUDIO_EMBEDS_KEY: embeds_out,
            _CRQ_PAST_KEY_VALUES_KEY: pkv_out,
        }

    def _run_audio_sidecar_prefill_warmup(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        speech_ids: torch.Tensor,
        cached_audio_embeds: Any,
        cached_past_key_values: Any,
        logits_processor: LogitsProcessorList,
        do_sample: bool,
        req_id: str | None = None,
    ) -> dict[str, Any]:
        device = hidden_states.device
        # Prefill starts a fresh speech span: drop any stale resident KV for
        # this request (e.g. after preemption / re-prefill across turns) before
        # re-seeding, so the warmup forward rebuilds the tower cache cleanly.
        if req_id is not None:
            self._crq_gpu_state.pop(req_id, None)
        input_ids = input_ids.to(device=device, dtype=torch.long).reshape(1, -1)
        text_embeds = (
            self.get_language_model()
            .embed_input_ids(input_ids.reshape(-1))
            .reshape(
                1,
                -1,
                hidden_states.shape[-1],
            )
        )
        speech_inputs_embeds = hidden_states.reshape(1, -1, hidden_states.shape[-1]) + text_embeds.detach()

        embeds_gpu, pkv_gpu = self._crq_resolve_gpu_kv(
            req_id, cached_audio_embeds, cached_past_key_values, device
        )
        self.audio_invert_tower.crq_audio_embeds = embeds_gpu
        self.audio_invert_tower.crq_past_key_values = pkv_gpu
        self.audio_invert_tower.crq_do_sample = do_sample
        self.audio_invert_tower.crq_logits_processor = logits_processor
        self.audio_invert_tower.crq_speech_ids = speech_ids
        self.audio_invert_tower.crq_generate_forward(
            inputs_embeds=speech_inputs_embeds,
            return_dict=True,
        )
        embeds_out = self.audio_invert_tower.crq_audio_embeds
        pkv_out = self.audio_invert_tower.crq_past_key_values
        self._crq_persist_gpu_kv(req_id, embeds_out, pkv_out)
        return {
            _CRQ_AUDIO_EMBEDS_KEY: embeds_out,
            _CRQ_PAST_KEY_VALUES_KEY: pkv_out,
        }

    def _gather_user_audio_embeds(
        self,
        mm_features: Any,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...] | None:
        """Produce per-item audio embeddings from the user-uploaded audio.

        ``mm_features`` is the request's ``list[MultiModalFeatureSpec]`` the
        runner forwards into the prefill span (see
        ``wants_mm_features_in_preprocess``). Each feature's processed data
        carries the keys produced by ``FunAudioChatMultiModalProcessor``:
        ``speech_ids``/``speech_attention_mask`` (discrete codec path) and
        ``input_features``/``feature_attention_mask``/``feature_exist_mask``
        (continuous Whisper-mel path). We hand all of them to the inherited
        native ``embed_multimodal``, which runs the discrete + continuous
        audio towers and returns one ``(num_features_i, output_dim)`` tensor
        per audio item, in prompt order.

        Returns ``None`` when there is no audio item in this request (a
        text-only prompt) so the caller keeps the plain text embeddings.
        """
        if not mm_features:
            return None

        # One MultiModalFeatureSpec per audio item; order them by their
        # placeholder offset so the returned tuple matches prompt order
        # (matters when limit_mm_per_prompt.audio > 1).
        audio_features = sorted(
            (f for f in mm_features if getattr(f, "modality", "") == "audio"),
            key=lambda f: getattr(getattr(f, "mm_position", None), "offset", 0),
        )
        if not audio_features:
            return None

        keys = {
            "speech_ids",
            "speech_attention_mask",
            "input_features",
            "feature_attention_mask",
            "feature_exist_mask",
        }
        # MultiModalFeatureSpec.gather_kwargs returns dict[key -> list[tensor]],
        # one per-item slice (MultiModalBatchedField splits the processor's
        # batched tensor along dim 0). The native embed_multimodal reads:
        #   - speech_ids / speech_attention_mask: accepts a list of 1D tensors
        #     (it pads them itself) -> pass the list through unchanged.
        #   - input_features / feature_attention_mask / feature_exist_mask: the
        #     native code asserts isinstance(..., torch.Tensor) and expects a
        #     batched tensor (N, ...). So we re-stack the per-item slices with
        #     torch.stack(dim=0), matching how the runner's mm branch would
        #     have batched them. Move to the compute device while stacking.
        gathered = MultiModalFeatureSpec.gather_kwargs(audio_features, keys)
        if "speech_ids" not in gathered:
            return None
        kwargs: dict[str, Any] = {
            "speech_ids": [t.to(device=device) for t in gathered["speech_ids"]],
            "speech_attention_mask": (
                [t.to(device=device) for t in gathered["speech_attention_mask"]]
                if "speech_attention_mask" in gathered
                else None
            ),
        }
        for tensor_key in ("input_features", "feature_attention_mask", "feature_exist_mask"):
            items = gathered.get(tensor_key)
            if not items:
                kwargs[tensor_key] = None
                continue
            kwargs[tensor_key] = torch.stack(
                [t.to(device=device) for t in items], dim=0
            )

        embeds = self.embed_multimodal(**kwargs)
        if embeds is None or len(embeds) == 0:
            return None
        return tuple(embeds)

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        del input_embeds
        if not self._batch_preprocess_in_progress:
            self._batch_req_infos = []
            self._batch_sidecar_results = []
            self._postprocess_cursor = 0
            self._batch_preprocess_in_progress = True

        span_len = int(input_ids.shape[0])
        device = input_ids.device
        _req_id = info_dict.get("request_id")
        text_embeds = self.get_language_model().embed_input_ids(input_ids.reshape(-1))
        req_embeds = text_embeds

        # Prefill: read the user-uploaded audio. The prefill span carries the
        # <|AUDIO|> placeholder tokens (token id == audio_token_index); we run
        # the inherited audio towers over the request's mm_features and scatter
        # the per-item audio embeddings into those positions, matching native
        # FunAudioChat semantics. Decode spans (span_len == 1) and text-only
        # prompts skip this and keep the plain text embeddings above.
        if span_len > 1:
            mm_embeds = self._gather_user_audio_embeds(info_dict.get("mm_features"), device)
            if mm_embeds is not None:
                audio_token_id = int(self.config.audio_token_index)
                flat_ids = input_ids.reshape(-1)
                is_multimodal = flat_ids == audio_token_id
                expected = int(is_multimodal.sum().item())
                got = int(sum(e.shape[0] for e in mm_embeds))
                if expected != got:
                    raise ValueError(
                        "FunAudioChat prefill audio placeholder count "
                        f"({expected}) does not match audio embedding rows "
                        f"({got}) for request {_req_id}."
                    )
                elif expected > 0:
                    req_embeds = self.embed_input_ids(
                        flat_ids,
                        multimodal_embeddings=mm_embeds,
                        is_multimodal=is_multimodal,
                    )

        # Speech-span state lives on the model instance (self._speech_state), not the
        # runner buffer, so postprocess_sampled_tokens' writes survive buffer
        # churn between steps. Fall back to buffer (legacy) then sp_gen_kwargs.
        _ss = self._speech_state.get(_req_id, {}) if _req_id is not None else {}
        generate_speech = bool(_ss.get(_GENERATE_SPEECH_KEY, info_dict.get(_GENERATE_SPEECH_KEY, False)))
        force_audio_bos_pending = bool(
            _ss.get(
                _FORCE_AUDIO_BOS_KEY,
                info_dict.get(_FORCE_AUDIO_BOS_KEY, self.sp_gen_kwargs["force_text_abos"]),
            )
        )
        speech_ids = self._resolve_gpu_speech_ids(
            _req_id,
            info_dict.get(_SPEECH_IDS_KEY),
            device,
        )
        current_text_seq_len, next_text_seq_len = self._resolve_text_seq_len(
            info_dict.get(_TEXT_SEQ_LEN_KEY),
            span_len,
        )

        if span_len == 1:
            current_text_embed = text_embeds.reshape(1, -1)
            if generate_speech and speech_ids.shape[-1] >= int(self.config.audio_config.group_size):
                last_group = speech_ids[:, -int(self.config.audio_config.group_size) :]
                audio_features = self.audio_tower(last_group.to(device=device, dtype=torch.long))
                if isinstance(audio_features, BaseModelOutput):
                    audio_features = audio_features.last_hidden_state
                elif isinstance(audio_features, (tuple, list)):
                    audio_features = audio_features[0]
                req_embeds = (current_text_embed + audio_features.reshape(1, -1)) / 2

        # Keep the current token on device.  Converting it to a Python int here
        # synchronizes the CUDA stream, only for the sidecar to recreate the
        # same one-element tensor on GPU immediately afterwards.
        current_input_token_id = input_ids.reshape(-1)[-1:]
        self._get_stage0_backend()
        update_dict = {
            _CURRENT_INPUT_TOKEN_ID_KEY: current_input_token_id,
            _FORCE_AUDIO_BOS_KEY: force_audio_bos_pending,
            _GENERATE_SPEECH_KEY: generate_speech,
            _SPEECH_IDS_KEY: speech_ids,
            _TEXT_SEQ_LEN_KEY: next_text_seq_len,
            "audio_token_ids": self._empty_audio_token_ids(device).to("cpu"),
        }

        self._batch_req_infos.append(
            {
                "req_id": _req_id,
                _CURRENT_INPUT_TOKEN_ID_KEY: current_input_token_id,
                _FORCE_AUDIO_BOS_KEY: force_audio_bos_pending,
                _GENERATE_SPEECH_KEY: generate_speech,
                _SPEECH_IDS_KEY: speech_ids,
                # Fallback seed for the sidecar's first step / re-prefill; from
                # the second step on the resident GPU state (self._crq_gpu_state)
                # is the source of truth and these are ignored.
                _CRQ_AUDIO_EMBEDS_KEY: info_dict.get(_CRQ_AUDIO_EMBEDS_KEY),
                _CRQ_PAST_KEY_VALUES_KEY: info_dict.get(_CRQ_PAST_KEY_VALUES_KEY),
                _TEXT_INPUT_IDS_KEY: input_ids.detach().to("cpu").contiguous(),
                _TEXT_SEQ_LEN_KEY: current_text_seq_len,
            }
        )
        return input_ids, req_embeds, update_dict

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        logits = super().compute_logits(hidden_states)
        if logits is None:
            self._batch_preprocess_in_progress = False
            return None

        self._batch_sidecar_results = []
        for idx, req_info in enumerate(self._batch_req_infos):
            force_audio_bos_pending = bool(req_info.get(_FORCE_AUDIO_BOS_KEY, False))
            speech_active = bool(req_info.get(_GENERATE_SPEECH_KEY, False))
            req_id = req_info.get("req_id")

            sidecar_result = {
                "req_id": req_id,
                _AUDIO_TOKEN_IDS_KEY: self._empty_audio_token_ids(hidden_states.device).to("cpu"),
                _CRQ_AUDIO_EMBEDS_KEY: req_info.get(_CRQ_AUDIO_EMBEDS_KEY),
                _CRQ_PAST_KEY_VALUES_KEY: req_info.get(_CRQ_PAST_KEY_VALUES_KEY),
                _FORCE_AUDIO_BOS_KEY: force_audio_bos_pending,
                _FINISH_SPEECH_KEY: False,
                _GENERATE_SPEECH_KEY: speech_active,
                _SPEECH_IDS_KEY: req_info.get(_SPEECH_IDS_KEY),
                "audio_token_ids": self._empty_audio_token_ids(hidden_states.device).to("cpu"),
            }

            req_input_ids = self._as_2d_long_tensor(req_info.get(_TEXT_INPUT_IDS_KEY), hidden_states.device).reshape(-1)
            crq_logits_processor, do_sample = self._build_crq_sampling_config(
                sampling_metadata=sampling_metadata,
                req_index=idx,
            )
            if speech_active and not self.sp_gen_kwargs["disable_speech"]:
                sidecar_step = self._run_audio_sidecar_step(
                    hidden_state=hidden_states[idx],
                    current_input_token_id=req_info[_CURRENT_INPUT_TOKEN_ID_KEY],
                    speech_ids=self._resolve_gpu_speech_ids(
                        req_id, req_info.get(_SPEECH_IDS_KEY), hidden_states.device
                    ),
                    cached_audio_embeds=req_info.get(_CRQ_AUDIO_EMBEDS_KEY),
                    cached_past_key_values=req_info.get(_CRQ_PAST_KEY_VALUES_KEY),
                    logits_processor=crq_logits_processor,
                    do_sample=do_sample,
                    current_text_seq_len=int(req_info.get(_TEXT_SEQ_LEN_KEY, 1)),
                    req_id=req_id,
                )
                sidecar_result.update(sidecar_step)
                sidecar_result["audio_token_ids"] = sidecar_step[_AUDIO_TOKEN_IDS_KEY]
            elif not self.sp_gen_kwargs["disable_speech"]:
                if req_input_ids.numel() > 1:
                    sidecar_result["_run_prefill_crq_warmup"] = True
                    sidecar_result["_prefill_input_ids"] = req_info.get(_TEXT_INPUT_IDS_KEY)
                    sidecar_result["_prefill_crq_logits_processor"] = crq_logits_processor
                    sidecar_result["_prefill_crq_do_sample"] = do_sample
                else:
                    warmup_state = self._run_audio_sidecar_decode_warmup(
                        hidden_state=hidden_states[idx],
                        current_input_token_id=req_info[_CURRENT_INPUT_TOKEN_ID_KEY],
                        speech_ids=self._resolve_gpu_speech_ids(
                            req_id, req_info.get(_SPEECH_IDS_KEY), hidden_states.device
                        ),
                        cached_audio_embeds=req_info.get(_CRQ_AUDIO_EMBEDS_KEY),
                        cached_past_key_values=req_info.get(_CRQ_PAST_KEY_VALUES_KEY),
                        logits_processor=crq_logits_processor,
                        do_sample=do_sample,
                        req_id=req_id,
                    )
                    sidecar_result.update(warmup_state)

            self._batch_sidecar_results.append(sidecar_result)
        self._postprocess_cursor = 0
        self._batch_preprocess_in_progress = False
        return logits

    def postprocess(self, hidden_states: torch.Tensor, **_: Any) -> dict[str, Any]:
        if self._postprocess_cursor >= len(self._batch_sidecar_results):
            return {}
        sidecar_result = self._batch_sidecar_results[self._postprocess_cursor]
        self._postprocess_cursor += 1
        if bool(sidecar_result.pop("_run_prefill_crq_warmup", False)):
            prefill_input_ids = sidecar_result.pop("_prefill_input_ids", None)
            if prefill_input_ids is not None:
                warmup_state = self._run_audio_sidecar_prefill_warmup(
                    hidden_states=hidden_states,
                    input_ids=self._as_2d_long_tensor(prefill_input_ids, hidden_states.device).reshape(-1),
                    speech_ids=self._resolve_gpu_speech_ids(
                        sidecar_result.get("req_id"),
                        sidecar_result.get(_SPEECH_IDS_KEY),
                        hidden_states.device,
                    ),
                    cached_audio_embeds=sidecar_result.get(_CRQ_AUDIO_EMBEDS_KEY),
                    cached_past_key_values=sidecar_result.get(_CRQ_PAST_KEY_VALUES_KEY),
                    logits_processor=sidecar_result.pop("_prefill_crq_logits_processor"),
                    do_sample=bool(sidecar_result.pop("_prefill_crq_do_sample", False)),
                    req_id=sidecar_result.get("req_id"),
                )
                sidecar_result.update(warmup_state)
        return {
            _AUDIO_TOKEN_IDS_KEY: sidecar_result[_AUDIO_TOKEN_IDS_KEY],
            _CRQ_AUDIO_EMBEDS_KEY: sidecar_result[_CRQ_AUDIO_EMBEDS_KEY],
            _CRQ_PAST_KEY_VALUES_KEY: sidecar_result[_CRQ_PAST_KEY_VALUES_KEY],
            _FORCE_AUDIO_BOS_KEY: sidecar_result[_FORCE_AUDIO_BOS_KEY],
            _FINISH_SPEECH_KEY: sidecar_result[_FINISH_SPEECH_KEY],
            _GENERATE_SPEECH_KEY: sidecar_result[_GENERATE_SPEECH_KEY],
            _SPEECH_IDS_KEY: sidecar_result[_SPEECH_IDS_KEY],
            "audio_token_ids": sidecar_result["audio_token_ids"],
        }

    def postprocess_sampled_tokens(
        self,
        sampled_token_ids: torch.Tensor,
        req_ids: list[str],
        req_id_to_index: dict[str, int],
        model_intermediate_buffer: dict[str, dict[str, Any]],
    ) -> torch.Tensor:
        if sampled_token_ids.numel() == 0:
            return sampled_token_ids

        if sampled_token_ids.ndim == 2 and sampled_token_ids.shape[-1] != 1:
            return sampled_token_ids

        # Preserve the runner-owned sampled tensor while rewriting the small
        # per-request token vector for the speech sidecar state machine.
        updated_token_ids = sampled_token_ids.clone()
        audio_bos_id = int(self.config.text_config.audio_bos_index)
        audio_eos_id = int(self.config.text_config.audio_eos_index)

        for rid in req_ids:
            req_buffer = model_intermediate_buffer.get(rid) or {}
            # Authoritative speech state lives on the model instance so it
            # survives runner buffer churn between steps.
            ss = self._speech_state.setdefault(rid, {})

            idx = req_id_to_index.get(rid)
            if idx is None:
                continue

            token_slot = updated_token_ids[idx] if updated_token_ids.ndim == 1 else updated_token_ids[idx, 0]
            original_token_id = int(token_slot.item())
            speech_active = bool(
                ss.get(_GENERATE_SPEECH_KEY, req_buffer.get(_GENERATE_SPEECH_KEY, False))
            )
            force_audio_bos_pending = bool(
                ss.get(
                    _FORCE_AUDIO_BOS_KEY,
                    req_buffer.get(_FORCE_AUDIO_BOS_KEY, self.sp_gen_kwargs["force_text_abos"]),
                )
            )
            finish_speech = bool(ss.pop(_FINISH_SPEECH_KEY, req_buffer.pop(_FINISH_SPEECH_KEY, False)))

            final_token_id, next_speech_active, next_force_audio_bos_pending = self._resolve_next_speech_state(
                sampled_token_id=original_token_id,
                generate_speech=speech_active,
                finish_speech=finish_speech,
                force_audio_bos_pending=force_audio_bos_pending,
                audio_bos_id=audio_bos_id,
                audio_eos_id=audio_eos_id,
            )

            if final_token_id != original_token_id:
                token_slot.fill_(final_token_id)

            ss[_GENERATE_SPEECH_KEY] = next_speech_active
            ss[_FORCE_AUDIO_BOS_KEY] = next_force_audio_bos_pending
            # Mirror to runner buffer too, in case other code paths read it.
            if isinstance(req_buffer, dict):
                req_buffer[_GENERATE_SPEECH_KEY] = next_speech_active
                req_buffer[_FORCE_AUDIO_BOS_KEY] = next_force_audio_bos_pending

        return updated_token_ids

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        # Drop resident per-request sidecar state for requests leaving the
        # batch. ``_crq_gpu_state`` holds GPU tower KV that the decode-warmup
        # path populates even during text phases, so without this every
        # finished request would leak its (length-growing) resident KV. Also
        # clears ``_speech_state`` (whose entries otherwise accumulate per
        # request in the baseline). The runner calls this before forward();
        # finished reqs are no longer in the scheduled batch, so freeing now
        # is safe -- the sidecar only touches req ids it sees in preprocess.
        for req_id in finished_req_ids:
            self._crq_gpu_state.pop(req_id, None)
            self._speech_ids_gpu_state.pop(req_id, None)
            self._speech_state.pop(req_id, None)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
