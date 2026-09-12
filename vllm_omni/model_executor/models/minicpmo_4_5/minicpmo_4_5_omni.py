# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from:
# https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/main/modeling_minicpmo.py
#
# Copyright 2025 The OpenBMB Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterable
from contextlib import suppress
from dataclasses import replace
from functools import cached_property
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMRoPE, SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_omni.model_executor.duplex_sampling import DuplexSamplingRow
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import MiniCPMO45DuplexPolicy
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMO45OmniLLMDummyInputsBuilder,
    MiniCPMO45OmniLLMMultiModalProcessor,
    MiniCPMO45OmniLLMProcessingInfo,
    MiniCPMOConfig,
)
from vllm_omni.model_executor.models.output_templates import ModelInputError, OmniOutput
from vllm_omni.model_executor.models.utils import add_prefix_to_loaded_weights
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


@MULTIMODAL_REGISTRY.register_processor(
    MiniCPMO45OmniLLMMultiModalProcessor,
    info=MiniCPMO45OmniLLMProcessingInfo,
    dummy_inputs=MiniCPMO45OmniLLMDummyInputsBuilder,
)
class MiniCPMO45OmniForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP, SupportsMRoPE):
    """MiniCPM-o 4.5 Omni model for conditional generation.

    Three-stage pipeline:
    - thinker (model_stage="llm"): image / video / audio encoders + 3D
      resampler + the omni LLM that emits text + hidden states.
    - talker  (model_stage="tts"): native continuous MiniCPMTTS AR that emits
      codec-token deltas for the separate Code2Wav stage.
    """

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "(<image>./</image>)"
        if modality.startswith("video"):
            return "(<video>./</video>)"
        if modality.startswith("audio"):
            return "(<audio>./</audio>)"
        raise ValueError("Only image, video or audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.have_multimodal_outputs = True
        config: MiniCPMOConfig = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        # keep vllm_config for later submodule init
        self.vllm_config = vllm_config

        # Store configs
        self.config = config
        self.multimodal_config = multimodal_config
        from vllm_omni.model_executor.models.minicpmo_4_5.duplex.compat import (
            patch_minicpmo_remote_config,
        )

        patch_minicpmo_remote_config(config)

        self.model_stage = vllm_config.model_config.model_stage

        if self.model_stage == "llm":
            # Initialize thinker model (image preprocessing + vision encoder + 3D resampler)
            self.thinker = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "thinker"),
                hf_config=config,
                # Use registry architecture key
                architectures=["MiniCPMO45OmniLLMForConditionalGeneration"],
            )
            self.model = self.thinker
            self.talker = None

        elif self.model_stage == "tts":
            self.thinker = None
            # The Talker is always the runner-owned continuous codec producer.
            self.talker = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "talker"),
                hf_config=config,
                # Use registry architecture key
                architectures=["MiniCPMO45OmniTTSForConditionalGeneration"],
            )
            # Initialize multimodal components if needed
            if hasattr(self.talker, "init_multi_modal"):
                self.talker.init_multi_modal(config)
            self.model = self.talker
        else:
            raise ValueError(f"Invalid model stage: {self.model_stage}. Must be one of: 'llm', 'tts'")

        # Set up intermediate tensors
        self.make_empty_intermediate_tensors = (
            (self.thinker.make_empty_intermediate_tensors)
            if self.model_stage == "llm" and self.thinker is not None
            else self.talker.make_empty_intermediate_tensors
            if self.talker is not None
            else lambda: None
        )

        self._language_model_names = ["model"]
        self.prefer_model_sampler = self.model_stage in {"llm", "tts"}
        # Both AR stages require model-specific embeddings.  The Thinker uses
        # preprocess for duplex audio, while the Talker converts the
        # tts_token_ids/tts_hidden_states handoff into its conditioning
        # embeddings and initializes request-local codec generation state.
        self.has_preprocess = self.model_stage in {"llm", "tts"}

    @cached_property
    def sampler(self):
        if hasattr(self.model, "sampler"):
            return self.model.sampler
        from vllm.v1.sample.sampler import Sampler

        return Sampler()

    @property
    def omni_pooler_payload_include_hidden(self) -> bool:
        # Thinker hidden states condition the Talker. Code2Wav consumes only
        # Talker codec IDs and metadata, so its handoff needs no hidden D2H.
        # This does not remove the device hidden states used to sample logits.
        return self.model_stage != "tts"

    def prepare_duplex_sampling(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        rows: tuple[DuplexSamplingRow, ...],
    ) -> None:
        """Apply MiniCPM duplex policy before the standard model sampler."""
        del sampling_metadata
        self._minicpmo45_active_duplex_rows = [row.row_idx for row in rows]
        closed_units = getattr(self, "_minicpmo45_closed_duplex_units", {})
        self._minicpmo45_duplex_row_units = {row.row_idx: (row.request_id, row.seq) for row in rows}
        self._minicpmo45_discarded_duplex_rows = {
            row.row_idx
            for row in rows
            if not row.sampling_enabled or (row.seq is not None and closed_units.get(row.request_id) == row.seq)
        }
        gander = bool(getattr(getattr(self, "config", None), "gander_unit8", False))
        self._minicpmo45_duplex_row_sessions = {
            row.row_idx: (row.request_id if gander else row.session_id, row.incarnation)
            for row in rows
            if row.session_id is not None
        }
        request_sessions = getattr(self, "_minicpmo45_duplex_request_sessions", None)
        if not isinstance(request_sessions, dict):
            request_sessions = {}
            self._minicpmo45_duplex_request_sessions = request_sessions
        request_sessions.update(
            {
                row.request_id: (row.request_id if gander else row.session_id, row.incarnation)
                for row in rows
                if row.session_id is not None
            }
        )
        self._minicpmo45_duplex_row_payloads = {row.row_idx: row.payload for row in rows if row.payload is not None}
        self._minicpmo45_duplex_row_max_tokens = {
            row.row_idx: row.max_tokens for row in rows if row.max_tokens is not None
        }
        self._minicpmo45_duplex_row_sampling = {row.row_idx: (row.temperature, row.top_k, row.top_p) for row in rows}
        if self.model_stage != "llm" or not rows or logits.ndim != 2:
            return

        token_ids = self._minicpmo45_native_duplex_token_ids()
        listen_id = int(token_ids.get("listen_token_id", -1))
        turn_eos_id = int(token_ids.get("turn_eos_token_id", -1))
        if listen_id < 0 or listen_id >= logits.shape[-1]:
            return

        force_listen_segments = getattr(
            self,
            "_minicpmo45_force_listen_applied_segments",
            None,
        )
        if not isinstance(force_listen_segments, set):
            force_listen_segments = set()
            self._minicpmo45_force_listen_applied_segments = force_listen_segments
        helper = getattr(self, "_minicpmo45_duplex_data_plane_helper", None)
        helper_sessions = getattr(helper, "sessions", None) if helper is not None else None
        for row in rows:
            if row.row_idx in self._minicpmo45_discarded_duplex_rows:
                continue
            row_idx = row.row_idx
            if row_idx < 0 or row_idx >= logits.shape[0]:
                continue
            payload = row.payload
            if not isinstance(payload, dict):
                continue
            force_listen = payload.get("force_listen") is True
            is_speech = payload.get("is_speech")
            segment_key = (row.request_id, row.seq if row.seq is not None else -1)
            session_key = (row.session_id, row.incarnation) if row.session_id is not None else None
            if (
                turn_eos_id >= 0
                and session_key is not None
                and not getattr(getattr(self, "config", None), "gander_unit8", False)
            ):
                state = helper_sessions.get(session_key) if isinstance(helper_sessions, dict) else None
                pending_speech_context = (
                    bool(getattr(state, "pending_speech_context", False)) if state is not None else False
                )
                if is_speech is True:
                    if state is not None:
                        with suppress(Exception):
                            state.last_terminator_token = None
                else:
                    turn_ended = bool(getattr(state, "current_turn_ended", True)) if state is not None else False
                    if turn_ended and not pending_speech_context:
                        force_listen = True

            if not force_listen:
                continue
            if force_listen and segment_key in force_listen_segments:
                continue
            if force_listen:
                logits[row_idx, :] = float("-inf")
                logits[row_idx, listen_id] = 0.0
                force_listen_segments.add(segment_key)

    # -------------------- Device utilities --------------------
    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration:
            # No parameters; fall back to buffers or cpu
            for _, buf in module.named_buffers(recurse=True):
                return buf.device
            return torch.device("cpu")

    def move_submodules_to_devices(
        self,
        *,
        thinker_device: str | torch.device | None = None,
        talker_device: str | torch.device | None = None,
    ) -> None:
        """Optionally move the thinker and talker to different devices.

        Example:
            model.move_submodules_to_devices(
                thinker_device='cuda:0',
                talker_device='cuda:1',
            )
        """
        if thinker_device is not None and self.thinker is not None:
            self.thinker.to(thinker_device)
        if talker_device is not None and self.talker is not None:
            self.talker.to(talker_device)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
    ) -> torch.Tensor:
        embed_fn = getattr(self.model, "get_input_embeddings", None)
        if callable(embed_fn):
            try:
                return embed_fn(input_ids, multimodal_embeddings)
            except TypeError:
                embeddings = embed_fn()
                if callable(embeddings):
                    return embeddings(input_ids)
            except AttributeError:
                pass

        embed_tokens = getattr(getattr(getattr(self.model, "llm", None), "model", None), "embed_tokens", None)
        if callable(embed_tokens):
            return embed_tokens(input_ids)

        raise AttributeError(f"{type(self.model).__name__} does not expose token embeddings")

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal=None,
    ) -> torch.Tensor:
        if self.model_stage == "tts":
            return self.get_input_embeddings(input_ids)
        return super().embed_input_ids(input_ids, multimodal_embeddings, is_multimodal=is_multimodal)

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        """Model-runner data-plane hook for MiniCPM-o 4.5 duplex audio.

        The scheduler owns the request, block table, attention metadata, KV,
        and sampler. This hook only turns the current duplex audio append into
        the prompt embeddings consumed by the normal runner forward.
        """
        if self.model_stage == "tts":
            return self.talker.preprocess(input_ids=input_ids, input_embeds=input_embeds, **kwargs)
        if self.model_stage != "llm":
            embeds = input_embeds if input_embeds is not None else self.get_input_embeddings(input_ids)
            return input_ids, embeds, {}

        duplex = kwargs.get("duplex")
        if not isinstance(duplex, dict) or duplex.get("data_plane") is not True:
            embeds = input_embeds if input_embeds is not None else self.get_input_embeddings(input_ids)
            return input_ids, embeds, {}

        prompt_len_meta = kwargs.get("duplex_prompt_len")
        token_offset_meta = kwargs.get("duplex_token_offset", 0)
        if (
            isinstance(prompt_len_meta, int)
            and isinstance(token_offset_meta, int)
            and token_offset_meta >= prompt_len_meta
        ):
            # Decode step of the resumable duplex request: input_ids are the
            # runner-sampled tokens and the normal embedding lookup is the
            # correct input. Slicing the (prompt-only) duplex embeddings here
            # would come up empty and pad-fill, feeding a </unit> embedding in
            # place of every sampled token and corrupting generation.
            embeds = input_embeds if input_embeds is not None else self.get_input_embeddings(input_ids)
            return input_ids, embeds, {}

        helper = self._duplex_data_plane_helper()
        session_id = str(duplex.get("session_id") or "")
        try:
            incarnation = int(duplex.get("incarnation", 0))
        except (TypeError, ValueError):
            incarnation = 0
        payload = duplex.get("payload")
        if not session_id or not isinstance(payload, dict):
            raise ModelInputError("native_duplex_prefill_failed: bad_duplex_payload")

        physical_owner = str(kwargs.get("request_id") or f"{session_id}:{incarnation}:{duplex.get('epoch', 0)}")
        gander = bool(getattr(getattr(self, "config", None), "gander_unit8", False))
        session_key = (physical_owner if gander else session_id, incarnation)
        state = helper.sessions.get(session_key)
        if state is None:
            from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import (
                _MiniCPMO45Stage0SessionState,
            )

            state = _MiniCPMO45Stage0SessionState(session_id=session_id)
            helper.sessions[session_key] = state
            session_config = duplex.get("session_config")
            session_config = dict(session_config) if isinstance(session_config, dict) else {}
            runtime_config = duplex.get("runtime_config")
            runtime_config = dict(runtime_config) if isinstance(runtime_config, dict) else {}
            if hasattr(helper.thinker, "audio_past_key_values"):
                helper.thinker.audio_past_key_values = None
            helper._configure_streaming_processor(state)
            helper._prepare_session_context(state, session_config, runtime_config=runtime_config)

        audio_waveform = None if payload.get("gander_control") is True else helper._decode_audio_payload(payload)
        try:
            video_frames = helper._decode_video_frames_payload(payload)
        except ValueError as exc:
            raise ModelInputError(f"native_duplex_prefill_failed: {exc}") from exc
        seq = duplex.get("seq")
        try:
            seq = int(seq) if seq is not None else None
        except (TypeError, ValueError):
            seq = None
        epoch = duplex.get("epoch")
        try:
            epoch = int(epoch) if epoch is not None else None
        except (TypeError, ValueError):
            epoch = None
        turn_id = duplex.get("turn_id")
        try:
            turn_id = int(turn_id) if turn_id is not None else None
        except (TypeError, ValueError):
            turn_id = None
        if payload.get("gander_control") is True:
            if not getattr(self.config, "gander_unit8", False):
                raise ValueError("Gander context payload on a non-Gander model")
            result = helper._stage_control_embeddings(state, payload, epoch=epoch, seq=seq)
        else:
            result = helper._stage_prefill_embeddings_only(
                state,
                audio_waveform,
                video_frames=video_frames,
                epoch=epoch,
                turn_id=turn_id,
                seq=seq,
                is_speech=bool(payload.get("is_speech", False)),
                final=bool(duplex.get("final")),
            )
        if payload.get("gander_replay") and result.get("success") and not result.get("gander_history_embedded"):
            history = payload.get("gander_replay_output_ids", [])
            if history:
                past_ids = list(history[:-1])
                if past_ids:
                    past_embeds = torch.cat([helper._as_2d_tensor(helper._embed_token(t)) for t in past_ids], dim=0)
                    result["inputs_embeds"] = torch.cat([result["inputs_embeds"], past_embeds], dim=0)
                    result["input_token_ids"] = [*result["input_token_ids"], *past_ids]
                    result["num_input_tokens"] = len(result["input_token_ids"])
                state.current_turn_ended = any(
                    t in history
                    for t in (
                        helper.turn_eos_token_id,
                        helper.listen_token_id,
                        helper._special_token_ids().get("tool_call_token_id"),
                    )
                )
            result["gander_history_embedded"] = True
            state.prepared_inputs_embeds = result["inputs_embeds"]
            state.prepared_input_token_ids = list(result["input_token_ids"])
            state.prepared_result = {k: v for k, v in result.items() if k not in {"inputs_embeds", "input_token_ids"}}
        if getattr(getattr(self, "config", None), "gander_unit8", False) and result.get("success"):
            result["special_token_ids"].update(
                gander_append_seq=seq or 0,
                gander_context_version=state.gander_context_version,
                gander_control_input=int(payload.get("gander_control") is True),
            )
        update_result = dict(result)
        update_result.pop("inputs_embeds", None)
        if result.get("success") is not True:
            raise ModelInputError(f"native_duplex_prefill_failed: {result.get('reason', 'no prepared model unit')}")

        target_dtype = (
            input_embeds.dtype if input_embeds is not None else self.get_input_embeddings(input_ids[:1]).dtype
        )
        full_req_embeds = result["inputs_embeds"].to(device=input_ids.device, dtype=target_dtype)
        full_input_token_ids = list(result.get("input_token_ids") or [])
        prompt_len = kwargs.get("duplex_prompt_len")
        try:
            prompt_len = int(prompt_len) if prompt_len is not None else int(full_req_embeds.shape[0])
        except (TypeError, ValueError):
            prompt_len = int(full_req_embeds.shape[0])
        token_offset = kwargs.get("duplex_token_offset", 0)
        try:
            token_offset = max(0, int(token_offset))
        except (TypeError, ValueError):
            token_offset = 0
        from vllm_omni.model_executor.models.minicpmo_4_5.duplex.input_history import DuplexPromptHistory

        request_id = str(kwargs.get("request_id") or f"{session_id}:{incarnation}:{epoch}")
        histories = getattr(self, "_minicpmo45_duplex_input_histories", None)
        if histories is None:
            histories = self._minicpmo45_duplex_input_histories = {}
        history = histories.get(request_id)
        if history is None:
            model_config = getattr(getattr(self, "vllm_config", None), "model_config", None)
            max_tokens = min(40960, int(getattr(model_config, "max_model_len", 40960)))
            history = histories[request_id] = DuplexPromptHistory(max_tokens=max_tokens)
        history.append(
            prompt_len=prompt_len,
            embeddings=full_req_embeds,
            token_ids=full_input_token_ids,
            identity=(epoch, seq, prompt_len),
            expected_start=duplex.get("kv_append_start"),
        )
        # Generated-token gaps use normal token lookup; actual multimodal
        # spans come from the owned recovery history. Keep the same path for
        # live append and historical preemption recomputation.
        req_input_ids = input_ids.clone()
        req_embeds = self.get_input_embeddings(req_input_ids).to(dtype=target_dtype).clone()
        history.overlay(offset=token_offset, input_ids=req_input_ids, embeddings=req_embeds)
        scheduler_ids = kwargs.get("duplex_scheduler_prompt_token_ids")
        if isinstance(scheduler_ids, list) and len(scheduler_ids) == prompt_len:
            update_result["duplex_prompt_token_ids"] = history.prompt_token_ids(scheduler_ids)
        elif token_offset == 0 and input_ids.shape[0] == prompt_len:
            update_result["duplex_prompt_token_ids"] = req_input_ids.tolist()
        return req_input_ids, req_embeds, {"duplex": update_result}

    def _duplex_data_plane_helper(self):
        helper = getattr(self, "_minicpmo45_duplex_data_plane_helper", None)
        if helper is not None:
            return helper
        from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import MiniCPMO45Stage0DuplexRuntime

        model_path = getattr(getattr(self.vllm_config, "model_config", None), "model", None)
        device = str(self._module_device(self.thinker if self.thinker is not None else self))
        helper = MiniCPMO45Stage0DuplexRuntime(self, model_path=model_path, device=device)
        self._minicpmo45_duplex_data_plane_helper = helper
        return helper

    def get_multimodal_embeddings(self, **kwargs):
        # Delegate to the active stage submodule when it implements MM encoding.
        mm_fn = getattr(self.model, "get_multimodal_embeddings", None)
        if mm_fn is not None:
            return mm_fn(**kwargs)
        return []

    def embed_multimodal(self, **kwargs: object):
        """vLLM V1 encoder profiling calls this; the inherited Protocol stub returns None."""
        return self.get_multimodal_embeddings(**kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        sampling_metadata: SamplingMetadata | None = None,
        logits_index: int | None = None,
        sampler=None,
        additional_information: dict[str, object] | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors | OmniOutput:
        """
        Forward pass for MiniCPM-o Omni model.

        Workflow:
        1) Thinker (model_stage="llm"): Image / video / audio encoders +
           3D resampler + omni LLM → text + hidden states.
        2) Talker (model_stage="tts"): native MiniCPMTTS AR → codec deltas.
        """
        if self.model_stage == "llm":
            # Normalize to batched inputs if caller provides 1D/2D unbatched tensors
            # TODO: Remove this hack when NPU supports batched inputs properly
            added_batch_dim = False
            if input_ids is not None and input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
                added_batch_dim = True
            if positions is not None and positions.ndim == 1:
                positions = positions.unsqueeze(0)
                added_batch_dim = True
            if inputs_embeds is not None and inputs_embeds.ndim == 2:
                inputs_embeds = inputs_embeds.unsqueeze(0)
                added_batch_dim = True
            thinker_dev = self._module_device(self.thinker)

            # if input_ids is None, set it to a zero tensor
            if input_ids is None:
                input_ids = torch.zeros(inputs_embeds.shape[1], dtype=torch.long, device=thinker_dev).unsqueeze(0)
                added_batch_dim = True

            # Ensure inputs on thinker's device
            if input_ids is not None and input_ids.device != thinker_dev:
                input_ids = input_ids.to(thinker_dev)
            if positions is not None and positions.device != thinker_dev:
                positions = positions.to(thinker_dev)
            if inputs_embeds is not None and inputs_embeds.device != thinker_dev:
                inputs_embeds = inputs_embeds.to(thinker_dev)

            if current_omni_platform.is_npu():
                # TODO: remove this hack when NPU supports batched inputs properly
                thinker_input_ids = input_ids[0] if input_ids is not None and added_batch_dim else input_ids
                thinker_positions = positions[0] if positions.ndim > 1 else positions
                thinker_inputs_embeds = (
                    inputs_embeds[0] if inputs_embeds is not None and added_batch_dim else inputs_embeds
                )
            else:
                thinker_input_ids = input_ids[0] if input_ids is not None and added_batch_dim else input_ids
                thinker_positions = positions[0] if positions is not None and added_batch_dim else positions
                thinker_inputs_embeds = (
                    inputs_embeds[0] if inputs_embeds is not None and added_batch_dim else inputs_embeds
                )

            # Run thinker
            thinker_output = self.thinker(
                input_ids=thinker_input_ids,
                positions=thinker_positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=thinker_inputs_embeds,
                **kwargs,
            )

            if isinstance(thinker_output, tuple):
                embeds, text_hidden_states = thinker_output
            else:
                text_hidden_states = thinker_output

            # Prepare hidden states for downstream stages
            # Ensure correct shape: (batch_size, seq_len, hidden_dim)
            if added_batch_dim:
                text_hidden_states = text_hidden_states.squeeze(0)

            # Return hidden states with latent in multimodal_outputs for stage_input_processors
            multimodal_outputs = {"latent": text_hidden_states}
            runtime_info = kwargs.get("runtime_additional_information")
            if runtime_info and isinstance(runtime_info, list) and len(runtime_info) > 0:
                duplex_rows = []
                for req_info in runtime_info:
                    duplex_info = req_info.get("duplex") if isinstance(req_info, dict) else None
                    duplex_rows.append(duplex_info if isinstance(duplex_info, dict) else {})

                prompt_rows = []
                for duplex_info in duplex_rows:
                    prompt_token_ids = duplex_info.get("duplex_prompt_token_ids")
                    # This is a complete per-handoff snapshot, not a generated
                    # tensor delta. Keep it as row-local metadata so output
                    # accumulation replaces the previous value instead of
                    # attempting to concatenate variable-length prompts.
                    prompt_rows.append(list(prompt_token_ids) if isinstance(prompt_token_ids, list) else None)
                if any(row is not None for row in prompt_rows):
                    multimodal_outputs["duplex_prompt_token_ids"] = prompt_rows

                recovery_rows = [bool(duplex_info.get("recovery_replay", False)) for duplex_info in duplex_rows]
                if any(duplex_rows):
                    # Explicit False clears a prior replay marker in this
                    # physical request's cumulative multimodal output state.
                    multimodal_outputs["duplex_recovery_replay"] = recovery_rows

                special_keys = {
                    key
                    for duplex_info in duplex_rows
                    for key, value in (
                        duplex_info.get("special_token_ids", {}).items()
                        if isinstance(duplex_info.get("special_token_ids"), dict)
                        else ()
                    )
                    if isinstance(key, str) and isinstance(value, int) and value >= 0
                }
                if special_keys:
                    multimodal_outputs["meta"] = {
                        key: [
                            torch.tensor(
                                [int(value)],
                                dtype=torch.long,
                                device=text_hidden_states.device,
                            )
                            if isinstance(value, int) and value >= 0
                            else None
                            for duplex_info in duplex_rows
                            for value in [
                                (
                                    duplex_info.get("special_token_ids", {}).get(key)
                                    if isinstance(duplex_info.get("special_token_ids"), dict)
                                    else None
                                )
                            ]
                        ]
                        for key in sorted(special_keys)
                    }
            return OmniOutput(
                text_hidden_states=text_hidden_states,
                multimodal_outputs=multimodal_outputs,
            )

        # Talker stage: runner-owned native AR only. Waveform generation belongs
        # to the separate Code2Wav stage.
        if self.model_stage == "tts":
            return self.talker(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        raise ValueError(f"Unsupported model stage: {self.model_stage}")

    def make_omni_output(self, model_outputs, **kwargs):
        if self.model_stage != "tts":
            return model_outputs
        return self.talker.make_omni_output(model_outputs, **kwargs)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput) -> torch.Tensor | None:
        # Handle OmniOutput type
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states

        # Use model for logits computation
        return self.model.compute_logits(hidden_states)

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        histories = getattr(self, "_minicpmo45_duplex_input_histories", {})
        closed_units = getattr(self, "_minicpmo45_closed_duplex_units", {})
        for request_id in finished_req_ids:
            histories.pop(request_id, None)
            closed_units.pop(request_id, None)
        request_sessions = getattr(self, "_minicpmo45_duplex_request_sessions", None)
        helper = getattr(self, "_minicpmo45_duplex_data_plane_helper", None)
        sessions = getattr(helper, "sessions", None) if helper is not None else None
        if isinstance(request_sessions, dict):
            for request_id in finished_req_ids:
                session_key = request_sessions.pop(request_id, None)
                if session_key is not None and isinstance(sessions, dict):
                    sessions.pop(session_key, None)
        forced_segments = getattr(self, "_minicpmo45_force_listen_applied_segments", None)
        if isinstance(forced_segments, set):
            finished = set(finished_req_ids)
            completed_segments = {segment for segment in forced_segments if segment[0] in finished}
            forced_segments.difference_update(completed_segments)
        if hasattr(self.model, "on_requests_finished"):
            self.model.on_requests_finished(finished_req_ids)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        native_duplex = self._sample_minicpmo45_native_duplex_stage0(
            logits,
            sampling_metadata,
            duplex_rows=getattr(self, "_minicpmo45_active_duplex_rows", None),
        )
        if native_duplex is not None:
            return native_duplex
        if self.model_stage == "tts":
            return self.model.sample(logits, sampling_metadata)
        return None

    def _sample_minicpmo45_native_duplex_stage0(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        *,
        duplex_rows: list[int] | None = None,
    ) -> SamplerOutput | None:
        if self.model_stage != "llm" or logits.ndim != 2 or logits.shape[0] == 0:
            return None
        token_ids = self._minicpmo45_native_duplex_token_ids()
        unit_id = token_ids.get("unit_token_id", -1)
        if unit_id < 0:
            return None
        native_rows = self._minicpmo45_native_duplex_prompt_rows(
            sampling_metadata,
            unit_id,
            logits.shape[0],
            duplex_rows=duplex_rows,
        )
        if not native_rows:
            return None
        native_rows = sorted(set(native_rows))

        standard_output = None
        if len(native_rows) != logits.shape[0]:
            # Chat and native requests share this Stage0 scheduler. Preserve
            # the standard sampler (including its logprobs for chat rows),
            # but never let a mixed batch bypass MiniCPM's native unit policy.
            # Its unused native samples must not advance request-owned RNG.
            native_set = set(native_rows)
            standard_metadata = replace(
                sampling_metadata,
                generators={
                    row: generator.clone_state() if row in native_set else generator
                    for row, generator in sampling_metadata.generators.items()
                },
            )
            standard_output = self.sampler(logits.clone(), standard_metadata)

        sampled_ids: list[int] = []
        for row_idx in native_rows:
            if row_idx in getattr(self, "_minicpmo45_discarded_duplex_rows", ()):
                # Partial prefills (including KV recomputation) have no
                # delivered sample. Do not advance session policy or RNG for
                # a token the runner will discard.
                sampled_ids.append(0)
                continue
            row_logits = logits[row_idx : row_idx + 1].clone()
            sampled = self._sample_minicpmo45_native_duplex_row(
                row_logits,
                sampling_metadata,
                row_idx=row_idx,
                token_ids=token_ids,
            )
            self._record_minicpmo45_duplex_terminator(row_idx, sampled, token_ids)
            # The async engine may execute a lookahead decode before the
            # scheduler consumes this unit's stop token. It must not sample
            # again and clear the terminator that the next append feeds into
            # KV. One closed sequence per physical request bounds this fence.
            # turn_eos is deliberately excluded: it still needs chunk_eos.
            unit = getattr(self, "_minicpmo45_duplex_row_units", {}).get(row_idx)
            if (
                unit is not None
                and unit[1] is not None
                and sampled
                in {
                    token_ids.get("listen_token_id", -1),
                    token_ids.get("interrupt_token_id", -1),
                    token_ids.get("chunk_eos_token_id", -1),
                    token_ids.get("chunk_tts_eos_token_id", -1),
                }
            ):
                if not hasattr(self, "_minicpmo45_closed_duplex_units"):
                    self._minicpmo45_closed_duplex_units = {}
                self._minicpmo45_closed_duplex_units[unit[0]] = unit[1]
            sampled_ids.append(sampled)
        native_ids = torch.tensor(sampled_ids, device=logits.device, dtype=torch.int32).unsqueeze(-1)
        if standard_output is not None:
            standard_output.sampled_token_ids[native_rows] = native_ids
            # Native requests do not request standard LM logprobs; chat rows
            # retain the untouched logprob tensors returned by vLLM.
            return standard_output
        return SamplerOutput(sampled_token_ids=native_ids, logprobs_tensors=None)

    def _sample_minicpmo45_native_duplex_row(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        *,
        row_idx: int,
        token_ids: dict[str, int],
    ) -> int:
        if getattr(getattr(self, "config", None), "gander_unit8", False):
            return self._sample_gander_dialogue_row(logits, sampling_metadata, row_idx=row_idx, token_ids=token_ids)
        chunk_eos_id = token_ids.get("chunk_eos_token_id", -1)
        generator = getattr(sampling_metadata, "generators", {}).get(row_idx)
        output_token_ids = getattr(sampling_metadata, "output_token_ids", None) or []
        raw_recent_tokens = output_token_ids[row_idx] if row_idx < len(output_token_ids) else []
        recent_tokens = [int(token_id) for token_id in raw_recent_tokens if isinstance(token_id, int) and token_id >= 0]
        temperature, top_k, top_p = getattr(self, "_minicpmo45_duplex_row_sampling", {}).get(
            row_idx, (None, None, None)
        )
        if temperature is None:
            temperature = (
                0.0
                if getattr(sampling_metadata, "all_greedy", False)
                else float(self._sampling_metadata_value(sampling_metadata, "temperature", row_idx, 0.7))
            )
        row_greedy = temperature <= 0
        if not row_greedy:
            if top_k is None:
                top_k = int(self._sampling_metadata_value(sampling_metadata, "top_k", row_idx, 100))
            if top_p is None:
                top_p = float(self._sampling_metadata_value(sampling_metadata, "top_p", row_idx, 0.8))
        state = self._minicpmo45_duplex_state_for_row(row_idx)
        turn_eos_id = token_ids.get("turn_eos_token_id", -1)
        prepared_append_identity = getattr(state, "prepared_append_identity", None)
        last_final_append_identity = getattr(state, "last_final_append_identity", None)
        is_post_final_continuation = (
            prepared_append_identity is not None
            and last_final_append_identity is not None
            and prepared_append_identity != last_final_append_identity
        )
        has_pending_turn_end = state is not None and getattr(state, "pending_turn_end_identity", None) is not None
        promote_turn_boundary = (
            has_pending_turn_end
            and (not recent_tokens or is_post_final_continuation)
            and 0 <= turn_eos_id < logits.shape[-1]
        )
        if (
            state is not None
            and getattr(state, "pending_post_turn_eos_chunk", False)
            and 0 <= chunk_eos_id < logits.shape[-1]
        ):
            # A real turn close is a two-token wire contract.  Do not rely on
            # the model distribution after <|turn_eos|>: the next token must
            # close the current unit so vLLM stops this segment with both
            # tokens committed to the same request/KV history.
            return int(chunk_eos_id)
        if chunk_eos_id >= 0 and chunk_eos_id < logits.shape[-1]:
            max_speak_tokens = int(
                getattr(
                    self,
                    "max_new_speak_tokens_per_chunk",
                    MiniCPMO45DuplexPolicy.DEFAULT_MAX_NEW_SPEAK_TOKENS_PER_CHUNK,
                )
                or MiniCPMO45DuplexPolicy.DEFAULT_MAX_NEW_SPEAK_TOKENS_PER_CHUNK
            )
            request_max_tokens = self._minicpmo45_duplex_row_request_max_tokens(row_idx)
            effective_max_speak_tokens = max_speak_tokens
            if request_max_tokens is not None:
                effective_max_speak_tokens = min(effective_max_speak_tokens, request_max_tokens)
            if len(recent_tokens) >= max(1, effective_max_speak_tokens - 1):
                if promote_turn_boundary:
                    # The length guard is another forced chunk boundary. It
                    # must obey the same final-append fence as a model-sampled
                    # chunk_eos or a busy continuation will never close its
                    # user turn.
                    return int(turn_eos_id)
                return int(chunk_eos_id)

            # Match the released StreamDecoder: first sample the original
            # distribution only to preserve the model's own chunk boundary.
            # If it does not choose chunk_eos, mask that token before the
            # normal text/listen/turn sampling pass below.
            if row_greedy:
                boundary_sample = int(torch.argmax(logits, dim=-1).item())
            else:
                boundary_probs = F.softmax(logits, dim=-1)
                boundary_sample = int(torch.multinomial(boundary_probs, num_samples=1, generator=generator).item())
            if boundary_sample == chunk_eos_id:
                if promote_turn_boundary:
                    # The released streaming loop commits a user turn by
                    # generating <|turn_eos|> before it closes the next unit.
                    # A scheduler continuation can contain filler/text tokens,
                    # so append identity -- not an empty output list -- is the
                    # authoritative signal that the final speech append has
                    # already completed. vLLM stops this segment on the
                    # promoted token and folds it into the next append, keeping
                    # worker KV and the downstream terminal envelope aligned.
                    return int(turn_eos_id)
                return int(chunk_eos_id)

        logits = logits.clone()
        forbidden = self._minicpmo45_native_forbidden_token_ids(token_ids)
        if forbidden:
            valid_forbidden = [token_id for token_id in forbidden if 0 <= token_id < logits.shape[-1]]
            if valid_forbidden:
                logits[:, valid_forbidden] = float("-inf")

        generated_tokens = getattr(state, "generated_tokens", None)
        repetition_tokens = generated_tokens if generated_tokens else recent_tokens
        repetition_penalty = 1.05
        if repetition_penalty != 1.0 and repetition_tokens:
            history_size = MiniCPMO45DuplexPolicy.REPETITION_HISTORY_SIZE
            for token_id in set(repetition_tokens[-history_size:]):
                if token_id < 0 or token_id >= logits.shape[-1]:
                    continue
                logits[0, token_id] /= repetition_penalty

        if row_greedy:
            sampled = int(torch.argmax(logits, dim=-1).item())
            self._record_minicpmo45_duplex_generation_token(row_idx, sampled)
            sampled = self._maybe_cut_minicpmo45_native_duplex_text_chunk(
                sampled,
                recent_tokens,
                token_ids,
            )
            if sampled == chunk_eos_id and promote_turn_boundary:
                sampled = int(turn_eos_id)
            return self._finalize_minicpmo45_native_duplex_sample(row_idx, sampled, token_ids)

        logits = logits / temperature
        logits = self._top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits, dim=-1)
        sampled = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
        self._record_minicpmo45_duplex_generation_token(row_idx, sampled)
        sampled = self._maybe_cut_minicpmo45_native_duplex_text_chunk(
            sampled,
            recent_tokens,
            token_ids,
        )
        if sampled == chunk_eos_id and promote_turn_boundary:
            sampled = int(turn_eos_id)
        return self._finalize_minicpmo45_native_duplex_sample(
            row_idx,
            sampled,
            token_ids,
        )

    def _sample_gander_dialogue_row(self, logits, sampling_metadata, *, row_idx, token_ids):
        payload = self._minicpmo45_duplex_payload_for_row(row_idx)
        if isinstance(payload, dict) and payload.get("gander_replay"):
            history = payload.get("gander_replay_output_ids", [])
            return int(history[-1]) if history else token_ids["listen_token_id"]
        from vllm_omni.model_executor.models.minicpmo_4_5.gander_tools import current_unit, tool_constraint

        history = getattr(sampling_metadata, "output_token_ids", None) or []
        recent = list(history[row_idx]) if row_idx < len(history) else []
        recent = current_unit(recent, token_ids)
        state = self._minicpmo45_duplex_state_for_row(row_idx)
        allow, constrained = tool_constraint(
            recent, token_ids, enabled=bool(getattr(state, "gander_tools_enabled", False))
        )
        logits = logits.clone()
        if allow:
            masked = torch.full_like(logits, float("-inf"))
            masked[:, list(constrained)] = logits[:, list(constrained)]
            logits = masked
        else:
            logits[:, list(constrained)] = float("-inf")
        eos_id = getattr(self._minicpmo45_tokenizer(), "eos_token_id", None)
        if isinstance(eos_id, int) and 0 <= eos_id < logits.shape[-1]:
            logits[:, eos_id] = float("-inf")
        temperature = float(self._sampling_metadata_value(sampling_metadata, "temperature", row_idx, 0.7))
        if getattr(sampling_metadata, "all_greedy", False) or temperature <= 0:
            return int(logits.argmax(dim=-1).item())
        logits = self._top_k_top_p_filter(
            logits / temperature,
            top_k=int(self._sampling_metadata_value(sampling_metadata, "top_k", row_idx, 20)),
            top_p=float(self._sampling_metadata_value(sampling_metadata, "top_p", row_idx, 0.8)),
        )
        generator = getattr(sampling_metadata, "generators", {}).get(row_idx)
        return int(torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator).item())

    def _maybe_cut_minicpmo45_native_duplex_text_chunk(
        self,
        sampled: int,
        recent_tokens: list[int],
        token_ids: dict[str, int],
    ) -> int:
        chunk_eos_id = token_ids.get("chunk_eos_token_id", -1)
        if chunk_eos_id < 0:
            return int(sampled)
        special_ids = self._minicpmo45_native_special_token_ids(token_ids)
        if sampled in special_ids:
            return int(sampled)
        max_chars = int(
            getattr(
                self,
                "max_speak_chars_per_chunk",
                MiniCPMO45DuplexPolicy.DEFAULT_MAX_SPEAK_CHARS_PER_CHUNK,
            )
            or MiniCPMO45DuplexPolicy.DEFAULT_MAX_SPEAK_CHARS_PER_CHUNK
        )
        if max_chars <= 0:
            return int(sampled)
        tokenizer = self._minicpmo45_tokenizer()
        decode = getattr(tokenizer, "decode", None)
        if not callable(decode):
            return int(sampled)
        candidate_tokens = self._minicpmo45_current_chunk_tokens(recent_tokens, token_ids)
        candidate_tokens.append(int(sampled))
        try:
            text = decode(candidate_tokens, skip_special_tokens=True)
        except TypeError:
            text = decode(candidate_tokens)
        except Exception:
            return int(sampled)
        return int(chunk_eos_id) if isinstance(text, str) and len(text) >= max_chars else int(sampled)

    @staticmethod
    def _minicpmo45_current_chunk_tokens(
        tokens: list[int],
        token_ids: dict[str, int],
    ) -> list[int]:
        boundaries = {
            token_ids.get("listen_token_id", -1),
            token_ids.get("chunk_eos_token_id", -1),
            token_ids.get("chunk_tts_eos_token_id", -1),
            token_ids.get("turn_eos_token_id", -1),
        }
        start = 0
        for idx, token_id in enumerate(tokens):
            if token_id in boundaries:
                start = idx + 1
        return list(tokens[start:])

    def _minicpmo45_duplex_state_for_row(self, row_idx: int):
        row_sessions = getattr(self, "_minicpmo45_duplex_row_sessions", None)
        session_key = row_sessions.get(row_idx) if isinstance(row_sessions, dict) else None
        if not session_key:
            return None
        helper = getattr(self, "_minicpmo45_duplex_data_plane_helper", None)
        sessions = getattr(helper, "sessions", None) if helper is not None else None
        return sessions.get(session_key) if isinstance(sessions, dict) else None

    def _minicpmo45_duplex_payload_for_row(self, row_idx: int) -> dict[str, Any] | None:
        row_payloads = getattr(self, "_minicpmo45_duplex_row_payloads", None)
        payload = row_payloads.get(row_idx) if isinstance(row_payloads, dict) else None
        return payload if isinstance(payload, dict) else None

    def _minicpmo45_duplex_row_request_max_tokens(self, row_idx: int) -> int | None:
        row_max_tokens = getattr(self, "_minicpmo45_duplex_row_max_tokens", None)
        value = row_max_tokens.get(row_idx) if isinstance(row_max_tokens, dict) else None
        try:
            max_tokens = int(value)
        except (TypeError, ValueError):
            return None
        return max_tokens if max_tokens > 0 else None

    def _finalize_minicpmo45_native_duplex_sample(
        self,
        row_idx: int,
        sampled: int,
        token_ids: dict[str, int],
    ) -> int:
        listen_id = token_ids.get("listen_token_id", -1)
        tts_bos_id = token_ids.get("tts_bos_token_id", -1)
        state = self._minicpmo45_duplex_state_for_row(row_idx)
        payload = self._minicpmo45_duplex_payload_for_row(row_idx)
        force_listen = isinstance(payload, dict) and payload.get("force_listen") is True
        if (
            sampled == listen_id
            and 0 <= tts_bos_id
            and state is not None
            and not getattr(state, "current_turn_ended", True)
            and not force_listen
        ):
            return int(tts_bos_id)
        return int(sampled)

    def _record_minicpmo45_duplex_generation_token(self, row_idx: int, sampled: int) -> None:
        """Track tokens returned by the model-policy decoder.

        The released StreamDecoder is constructed without special_token_ids, so
        every normally decoded token participates in repetition penalty. Forced
        listen bypasses that decoder, while chunk_eos boundary decisions return
        before this method is called.
        """
        state = self._minicpmo45_duplex_state_for_row(row_idx)
        if state is None:
            return
        payload = self._minicpmo45_duplex_payload_for_row(row_idx)
        if isinstance(payload, dict) and payload.get("force_listen") is True:
            return
        generated_tokens = getattr(state, "generated_tokens", None)
        if not isinstance(generated_tokens, list):
            generated_tokens = []
            state.generated_tokens = generated_tokens
        generated_tokens.append(int(sampled))
        history_size = MiniCPMO45DuplexPolicy.REPETITION_HISTORY_SIZE
        del generated_tokens[:-history_size]

    def _record_minicpmo45_duplex_terminator(self, row_idx: int, sampled: int, token_ids: dict[str, int]) -> None:
        """Remember sampled unit state for the next append.

        The scheduler session update discards the final sampled token of a
        segment before the next streaming update, but the official duplex
        format feeds it (terminator + </unit>) into the KV at every unit
        boundary, and the model's listen/speak policy depends on seeing its own
        past decisions. Non-terminators clear the turn-ended latch."""
        state = self._minicpmo45_duplex_state_for_row(row_idx)
        if state is None:
            return
        payload = self._minicpmo45_duplex_payload_for_row(row_idx)
        if getattr(getattr(self, "config", None), "gander_unit8", False):
            if isinstance(payload, dict) and payload.get("gander_replay"):
                state.pending_terminator_token = int(sampled)
                state.last_terminator_token = int(sampled)
                return
            if isinstance(payload, dict) and payload.get("gander_control") and payload.get("force_listen"):
                # A slate changes model context, not the active speech turn.
                state.pending_terminator_token = int(sampled)
                state.last_terminator_token = int(sampled)
                return
            if sampled == token_ids.get("tool_call_token_id"):
                state.gander_tool_active = True
            if getattr(state, "gander_tool_active", False):
                state.current_turn_ended = True
                if sampled == token_ids.get("chunk_eos_token_id"):
                    state.gander_tool_active = False
                    state.pending_terminator_token = sampled
                    state.last_terminator_token = sampled
                return
        force_listen = isinstance(payload, dict) and payload.get("force_listen") is True
        listen_id = token_ids.get("listen_token_id", -1)
        tts_bos_id = token_ids.get("tts_bos_token_id", -1)
        chunk_eos_id = token_ids.get("chunk_eos_token_id", -1)
        chunk_tts_eos_id = token_ids.get("chunk_tts_eos_token_id", -1)
        turn_eos_id = token_ids.get("turn_eos_token_id", -1)
        interrupt_id = (
            token_ids.get("interrupt_token_id", -1)
            if getattr(getattr(self, "config", None), "gander_unit8", False)
            else -1
        )
        terminators = {listen_id, chunk_eos_id, chunk_tts_eos_id, turn_eos_id, interrupt_id}
        if sampled in terminators:
            state.pending_terminator_token = int(sampled)
            state.last_terminator_token = int(sampled)
            if sampled in {turn_eos_id, interrupt_id}:
                state.current_turn_ended = True
                with suppress(Exception):
                    state.pending_speech_response_open = False
                    state.pending_turn_end_identity = None
                    state.pending_post_turn_eos_chunk = True
            elif sampled == chunk_eos_id and getattr(state, "pending_post_turn_eos_chunk", False):
                # Consume the second half of the deterministic
                # <|turn_eos|>, <|chunk_eos|> close sequence without reopening
                # the completed turn.
                state.pending_post_turn_eos_chunk = False
            elif sampled == listen_id and force_listen:
                state.current_turn_ended = True
                with suppress(Exception):
                    state.pending_speech_response_open = False
                    state.pending_turn_end_identity = None
            return
        if (
            sampled == tts_bos_id
            and getattr(state, "current_turn_ended", True)
            and getattr(state, "pending_speech_context", False)
        ):
            with suppress(Exception):
                state.pending_speech_response_open = True
        elif getattr(state, "pending_speech_response_open", False):
            with suppress(Exception):
                state.pending_speech_context = False
                state.pending_speech_response_open = False
        elif getattr(state, "current_turn_ended", True):
            with suppress(Exception):
                state.pending_speech_context = False
        state.pending_terminator_token = None
        state.last_terminator_token = None
        state.current_turn_ended = False

    def _minicpmo45_tokenizer(self):
        if hasattr(self, "_minicpmo45_tokenizer_cache"):
            return self._minicpmo45_tokenizer_cache
        tokenizer = None
        get_tokenizer = getattr(getattr(self, "thinker", None), "get_tokenizer", None)
        if callable(get_tokenizer):
            tokenizer = get_tokenizer()
        if tokenizer is None:
            try:
                from vllm.tokenizers import cached_tokenizer_from_config

                tokenizer = cached_tokenizer_from_config(self.vllm_config.model_config)
            except Exception:
                pass
        self._minicpmo45_tokenizer_cache = tokenizer
        return tokenizer

    def _minicpmo45_native_duplex_token_ids(self) -> dict[str, int]:
        cached = getattr(self, "_minicpmo45_native_duplex_token_ids_cache", None)
        if isinstance(cached, dict):
            return cached
        tokenizer = self._minicpmo45_tokenizer()
        cached = MiniCPMO45DuplexPolicy.token_ids_from_tokenizer(tokenizer)
        if getattr(getattr(self, "config", None), "gander_unit8", False):
            from vllm_omni.model_executor.models.minicpmo_4_5.gander import control_token_ids

            cached.update(control_token_ids(tokenizer))
        self._minicpmo45_native_duplex_token_ids_cache = cached
        return cached

    def _minicpmo45_native_duplex_prompt_rows(
        self,
        sampling_metadata: SamplingMetadata,
        unit_id: int,
        batch_size: int,
        *,
        duplex_rows: list[int] | None = None,
    ) -> list[int]:
        if duplex_rows is not None:
            rows: list[int] = []
            for row in duplex_rows:
                try:
                    row_idx = int(row)
                except (TypeError, ValueError):
                    continue
                if 0 <= row_idx < batch_size:
                    rows.append(row_idx)
            return rows

        prompt_token_ids = getattr(sampling_metadata, "prompt_token_ids", None)
        if prompt_token_ids is None:
            return []
        if prompt_token_ids.ndim == 1:
            prompt_token_ids = prompt_token_ids.unsqueeze(0)
        rows: list[int] = []
        for row_idx in range(min(batch_size, int(prompt_token_ids.shape[0]))):
            row = prompt_token_ids[row_idx]
            if torch.count_nonzero(row == unit_id).item() >= 2:
                rows.append(row_idx)
        return rows

    def _minicpmo45_native_forbidden_token_ids(self, token_ids: dict[str, int]) -> list[int]:
        tokenizer = self._minicpmo45_tokenizer()
        bad_token_ids = getattr(tokenizer, "bad_token_ids", []) if tokenizer is not None else []
        return MiniCPMO45DuplexPolicy.native_forbidden_token_ids(token_ids, bad_token_ids=bad_token_ids)

    def _minicpmo45_native_special_token_ids(self, token_ids: dict[str, int]) -> set[int]:
        tokenizer = self._minicpmo45_tokenizer()
        return MiniCPMO45DuplexPolicy.native_special_token_ids(
            token_ids,
            tokenizer_special_ids=getattr(tokenizer, "all_special_ids", []) if tokenizer is not None else [],
        )

    @staticmethod
    def _sampling_metadata_value(
        sampling_metadata: SamplingMetadata,
        name: str,
        row_idx: int,
        default: float,
    ) -> float:
        value = getattr(sampling_metadata, name, None)
        if value is None:
            return default
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return default
            if value.ndim == 0:
                return float(value.item())
            idx = min(row_idx, int(value.numel()) - 1)
            return float(value.reshape(-1)[idx].item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _top_k_top_p_filter(logits: torch.Tensor, *, top_k: int, top_p: float) -> torch.Tensor:
        if top_k > 0 and top_k < logits.shape[-1]:
            kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_remove = cumulative_probs > top_p
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            remove = torch.zeros_like(logits, dtype=torch.bool)
            remove.scatter_(dim=-1, index=sorted_indices, src=sorted_remove)
            logits = logits.masked_fill(remove, float("-inf"))
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for all components of the omni model."""
        loaded_weights = set()
        thinker_weights = []
        talker_weights = []

        # MiniCPM-o checkpoint prefixes → stage mapping:
        #   thinker: vpm, resampler, llm, apm, audio_projection_layer
        #   talker:  tts (native MiniCPMTTS AR codec producer)
        for k, v in weights:
            if k.startswith(("vpm.", "resampler.", "llm.", "apm.", "audio_projection_layer.")):
                thinker_weights.append((k, v))
            elif k.startswith("tts."):
                talker_weights.append((k, v))
            else:
                logger.warning("Unknown weight prefix: %s, skipping", k)

        # Load thinker weights
        if self.thinker is not None and thinker_weights:
            thinker_loaded = self.thinker.load_weights(thinker_weights)
            thinker_loaded = add_prefix_to_loaded_weights(thinker_loaded, "thinker")
            loaded_weights.update(thinker_loaded)

        # Load talker weights
        if self.talker is not None and talker_weights:
            talker_loaded = self.talker.load_weights(talker_weights)
            talker_loaded = add_prefix_to_loaded_weights(talker_loaded, "talker")
            loaded_weights.update(talker_loaded)

        return loaded_weights
