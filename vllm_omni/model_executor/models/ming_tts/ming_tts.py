# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections.abc import Iterable
from functools import cached_property
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.models import SupportsPP
from vllm.model_executor.models.utils import init_vllm_registered_model
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin

from .audio_prep import (
    _coerce_prompt_latents,
    _find_audio_placeholder_positions,
    _find_speaker_placeholder_positions,
    _initial_history,
    _take_scalar,
    coerce_speaker_embeddings,
)
from .config_ming_tts import (
    AUDIO_START_TOKEN_ID,
    KEY_CFG,
    KEY_DECODE_STEP,
    KEY_LAST_STOP_PROB,
    KEY_LATENT_HISTORY,
    KEY_MAX_DECODE_STEPS,
    KEY_MIN_DECODE_STEPS,
    KEY_NEXT_EMBEDS,
    KEY_PROMPT_LATENT_TAIL,
    KEY_REQUEST_ID,
    KEY_SIGMA,
    KEY_SPEAKER_EMBEDDING,
    KEY_TEMPERATURE,
    KEY_TEXT_MODE,
    MingTTSConfig,
)
from .patch_emission import MING_STOP_REASON_KEY
from .prompt_encoder import _resolve_prompt_latents


class _ModelSampleAdapter(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, logits, sampling_metadata):
        return self.model.sample(logits, sampling_metadata)


class MingTTSForConditionalGeneration(nn.Module, SupportsPP, CustomProcessMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        del prefix
        self.vllm_config = vllm_config
        self.ming_config = MingTTSConfig.from_hf_config(vllm_config.model_config.hf_config)
        self.ming_config.validate()
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.requires_raw_input_tokens = False
        self.model_stage = vllm_config.model_config.model_stage
        self._prompt_encoder = None

        if self.model_stage == "llm":
            self.model = init_vllm_registered_model(vllm_config=vllm_config, architectures=["MingLLMModel"])
            self.has_preprocess = True
            self.has_postprocess = True
            self.set_custom_preprocess(self.preprocess)
            self.set_custom_postprocess(self.postprocess)
        elif self.model_stage == "audio_vae":
            self.model = init_vllm_registered_model(vllm_config=vllm_config, architectures=["MingAudioVAEModel"])
            self.requires_raw_input_tokens = True
        else:
            raise ValueError(f"Invalid Ming model_stage={self.model_stage}")
        self.make_empty_intermediate_tensors = getattr(self.model, "make_empty_intermediate_tensors", lambda: None)

    @cached_property
    def sampler(self):
        if hasattr(self.model, "sample"):
            return _ModelSampleAdapter(self.model)
        if hasattr(self.model, "sampler"):
            return self.model.sampler
        return Sampler()

    def embed_input_ids(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids=input_ids, **kwargs)

    def forward(self, *args: Any, **kwargs: Any):
        return self.model(*args, **kwargs)

    def compute_logits(self, hidden_states, sampling_metadata=None):
        return self.model.compute_logits(hidden_states, sampling_metadata=sampling_metadata)

    def sample(self, logits, sampling_metadata):
        return self.model.sample(logits, sampling_metadata) if hasattr(self.model, "sample") else None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = list(weights)
        if self.model_stage == "llm":
            allowed = ("model.", "linear_proj_audio.", "flowloss.", "stop_head.", "spk_head.")
            llm_weights = [(k, v) for k, v in weights if k.startswith(allowed)]
            if not llm_weights:
                raise RuntimeError(
                    "Ming Stage-0 received no loadable checkpoint weights. "
                    "Expected prefixes: model.*, linear_proj_audio.*, flowloss.*, stop_head.*, spk_head.*"
                )
            loaded = self.model.load_weights(llm_weights)
            return {f"model.{name}" for name in loaded}

        audio_weights = [(k, v) for k, v in weights if k.startswith("audio.")]
        if not audio_weights:
            raise RuntimeError("Ming Stage-1 received no loadable checkpoint weights. Expected prefix: audio.*")
        loaded = self.model.load_weights(audio_weights)
        return {f"model.{name}" for name in loaded}

    def _resolve_prompt_latents(self, info_dict: dict[str, Any]):
        return _resolve_prompt_latents(self, info_dict)

    def preprocess(self, input_ids: torch.Tensor, input_embeds: torch.Tensor | None, **info_dict: Any):
        if self.model_stage != "llm":
            return input_ids, input_embeds, {}
        input_embeds = self.model.embed_input_ids(input_ids).clone()
        return (
            self._prefill_preprocess(input_ids, input_embeds, **info_dict)
            if int(input_ids.shape[0]) > 1
            else self._decode_preprocess(input_ids, input_embeds, **info_dict)
        )

    def postprocess(self, hidden_states: torch.Tensor, **info_dict: Any) -> dict[str, Any]:
        if self.model_stage != "llm" or hidden_states.numel() == 0:
            return {}
        req_id = info_dict.get(KEY_REQUEST_ID, info_dict.get("request_id"))
        pending = self.model.pop_postprocess_update(req_id)
        if not pending or not isinstance(pending.get("ming_latent_patch"), torch.Tensor):
            return {}

        update = {
            KEY_LATENT_HISTORY: pending[KEY_LATENT_HISTORY].detach().to("cpu").contiguous(),
            KEY_NEXT_EMBEDS: pending[KEY_NEXT_EMBEDS].detach().to("cpu").contiguous(),
            KEY_DECODE_STEP: int(info_dict.get(KEY_DECODE_STEP, 0)) + 1,
        }
        stop_prob = _take_scalar(pending.get("ming_stop_prob"), 0)
        if stop_prob is not None:
            update[KEY_LAST_STOP_PROB] = stop_prob
        stop_reason = pending.get(MING_STOP_REASON_KEY)
        if isinstance(stop_reason, str):
            update[MING_STOP_REASON_KEY] = stop_reason
        if isinstance(req_id, str):
            update[KEY_REQUEST_ID] = req_id
        return update

    def _prefill_preprocess(self, input_ids: torch.Tensor, input_embeds: torch.Tensor, **info_dict: Any):
        if bool(info_dict.get(KEY_TEXT_MODE, False)):
            update: dict[str, Any] = {KEY_TEXT_MODE: True}
            request_id = info_dict.get(KEY_REQUEST_ID, info_dict.get("request_id"))
            if request_id is not None:
                update[KEY_REQUEST_ID] = request_id
            if int(input_ids.shape[0]) > 1 and int(input_ids[-1].item()) == AUDIO_START_TOKEN_ID:
                return input_ids[:-1], input_embeds[:-1], update
            return input_ids, input_embeds, update

        update: dict[str, Any] = {KEY_DECODE_STEP: int(info_dict.get(KEY_DECODE_STEP, 0))}
        prompt_latents = _resolve_prompt_latents(self, info_dict)
        history = _initial_history(
            prompt_latents["frames"] if prompt_latents is not None else None,
            history_size=self.ming_config.history_patch_size,
            latent_dim=self.ming_config.latent_dim,
            device=input_embeds.device,
            dtype=torch.float32,
        )
        update[KEY_LATENT_HISTORY] = history.detach().to("cpu").contiguous()
        update[KEY_PROMPT_LATENT_TAIL] = update[KEY_LATENT_HISTORY]

        speaker_embedding = info_dict.get(KEY_SPEAKER_EMBEDDING, info_dict.get("speaker_embedding"))
        speaker_embeddings = None
        if speaker_embedding is not None:
            speaker_embeddings = coerce_speaker_embeddings(
                speaker_embedding,
                use_zero_spk_emb=bool(info_dict.get("use_zero_spk_emb", False)),
            )
        if speaker_embeddings is not None and len(speaker_embeddings) > 0:
            speaker_slots = _find_speaker_placeholder_positions(input_ids, self.vllm_config.model_config.hf_config)
            if len(speaker_slots) < len(speaker_embeddings):
                raise RuntimeError(
                    "Could not locate enough speaker placeholder slots: "
                    f"found {len(speaker_slots)}, need {len(speaker_embeddings)}"
                )
            for speaker_slot, spk in zip(speaker_slots, speaker_embeddings):
                spk_proj = self.model.project_speaker_embedding(
                    spk.to(device=input_embeds.device, dtype=input_embeds.dtype).unsqueeze(0)
                ).squeeze(0)
                input_embeds[speaker_slot] = spk_proj

        if prompt_latents is not None and prompt_latents["patches"] is not None:
            prompt_patches = prompt_latents["patches"].to(dtype=getattr(self.model, "fm_dtype", torch.float32))
            prompt_embeds = self.model.linear_proj_audio(prompt_patches).squeeze(1)
            placeholder_pos = _find_audio_placeholder_positions(input_ids, self.ming_config)
            take = min(int(placeholder_pos.numel()), int(prompt_embeds.shape[0]))
            if take > 0:
                input_embeds[placeholder_pos[:take]] = prompt_embeds[:take].to(dtype=input_embeds.dtype)

        request_id = info_dict.get(KEY_REQUEST_ID, info_dict.get("request_id"))
        if request_id is not None:
            update[KEY_REQUEST_ID] = request_id
        _copy_runtime_controls(update, info_dict)
        return input_ids, input_embeds, update

    def _decode_preprocess(self, input_ids: torch.Tensor, input_embeds: torch.Tensor, **info_dict: Any):
        if bool(info_dict.get(KEY_TEXT_MODE, False)):
            update: dict[str, Any] = {KEY_TEXT_MODE: True}
            request_id = info_dict.get(KEY_REQUEST_ID, info_dict.get("request_id"))
            if request_id is not None:
                update[KEY_REQUEST_ID] = request_id
            return input_ids, input_embeds, update

        update: dict[str, Any] = {KEY_DECODE_STEP: int(info_dict.get(KEY_DECODE_STEP, 0))}
        history = info_dict.get(KEY_LATENT_HISTORY)
        if isinstance(history, torch.Tensor):
            update[KEY_LATENT_HISTORY] = history.detach().to("cpu").contiguous()
        else:
            zero_history = torch.zeros(
                (self.ming_config.history_patch_size, self.ming_config.latent_dim),
                device=input_embeds.device,
                dtype=torch.float32,
            )
            update[KEY_LATENT_HISTORY] = zero_history.detach().to("cpu").contiguous()

        next_embeds = info_dict.get(KEY_NEXT_EMBEDS)
        if isinstance(next_embeds, torch.Tensor) and input_ids.numel() == 1:
            if not torch.isfinite(next_embeds).all():
                raise RuntimeError("Non-finite next_embeds before decode preprocess write.")
            input_embeds[0] = (
                next_embeds.detach()
                .reshape(-1, self.ming_config.llm_hidden_size)[0]
                .to(
                    device=input_embeds.device,
                    dtype=input_embeds.dtype,
                )
            )
            if not torch.isfinite(input_embeds[0]).all():
                raise RuntimeError("Non-finite backbone input_embeds after decode preprocess write.")

        request_id = info_dict.get(KEY_REQUEST_ID, info_dict.get("request_id"))
        if request_id is not None:
            update[KEY_REQUEST_ID] = request_id
        _copy_runtime_controls(update, info_dict)
        return input_ids, input_embeds, update


def _copy_runtime_controls(update: dict[str, Any], info_dict: dict[str, Any]) -> None:
    for key in (KEY_CFG, KEY_SIGMA, KEY_TEMPERATURE, KEY_MAX_DECODE_STEPS, KEY_MIN_DECODE_STEPS):
        if key in info_dict:
            update[key] = info_dict[key]


__all__ = [
    "MingTTSForConditionalGeneration",
    "_ModelSampleAdapter",
    "_coerce_prompt_latents",
    "_find_audio_placeholder_positions",
    "_find_speaker_placeholder_positions",
    "_initial_history",
]
