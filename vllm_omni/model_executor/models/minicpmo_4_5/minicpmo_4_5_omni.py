# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

from vllm_omni.experimental.fullduplex.engine.intermediate import get_stream_request_key
from vllm_omni.experimental.fullduplex.minicpmo45.policy import MiniCPMO45DuplexPolicy
from vllm_omni.experimental.fullduplex.model_executor import DuplexSamplingRow
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMO45OmniLLMDummyInputsBuilder,
    MiniCPMO45OmniLLMMultiModalProcessor,
    MiniCPMO45OmniLLMProcessingInfo,
    MiniCPMOConfig,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
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

    This model has two pipeline stages:
    - llm: multimodal thinker and text generation
    - tts: talker generation followed by its built-in Token2Wav vocoder
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
        from vllm_omni.experimental.fullduplex.minicpmo45.compat import (
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
            # Initialize talker model (LLM generation)
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
            else lambda: None
        )

        self._language_model_names = ["model"]
        self.prefer_model_sampler = self.model_stage in {"llm", "tts"}
        self.has_preprocess = self.model_stage == "llm"

    @cached_property
    def sampler(self):
        if hasattr(self.model, "sampler"):
            return self.model.sampler
        from vllm.v1.sample.sampler import Sampler

        return Sampler()

    def prepare_duplex_sampling(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        rows: tuple[DuplexSamplingRow, ...],
    ) -> None:
        """Apply MiniCPM duplex policy before the standard model sampler."""
        del sampling_metadata
        self._minicpmo45_active_duplex_rows = [row.row_idx for row in rows]
        self._minicpmo45_duplex_row_sessions = {
            row.row_idx: (row.session_id, row.incarnation) for row in rows if row.session_id is not None
        }
        request_sessions = getattr(self, "_minicpmo45_duplex_request_sessions", None)
        if not isinstance(request_sessions, dict):
            request_sessions = {}
            self._minicpmo45_duplex_request_sessions = request_sessions
        request_sessions.update(
            {row.request_id: (row.session_id, row.incarnation) for row in rows if row.session_id is not None}
        )
        self._minicpmo45_duplex_row_payloads = {row.row_idx: row.payload for row in rows if row.payload is not None}
        self._minicpmo45_duplex_row_max_tokens = {
            row.row_idx: row.max_tokens for row in rows if row.max_tokens is not None
        }
        if self.model_stage != "llm" or not rows or logits.ndim != 2:
            return

        token_ids = self._minicpmo45_native_duplex_token_ids()
        listen_id = int(token_ids.get("listen_token_id", -1))
        tts_bos_id = int(token_ids.get("tts_bos_token_id", -1))
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
            row_idx = row.row_idx
            if row_idx < 0 or row_idx >= logits.shape[0]:
                continue
            payload = row.payload
            if not isinstance(payload, dict):
                continue
            force_listen = payload.get("force_listen") is True
            is_speech = payload.get("is_speech")
            redirect_listen = False
            segment_key = (row.request_id, row.seq if row.seq is not None else -1)
            session_key = (row.session_id, row.incarnation) if row.session_id is not None else None
            if turn_eos_id >= 0 and session_key is not None:
                state = helper_sessions.get(session_key) if isinstance(helper_sessions, dict) else None
                pending_speech_context = (
                    bool(getattr(state, "pending_speech_context", False)) if state is not None else False
                )
                if is_speech is True:
                    if state is not None:
                        if not getattr(state, "current_turn_ended", True):
                            redirect_listen = True
                        with suppress(Exception):
                            state.last_terminator_token = None
                else:
                    turn_ended = bool(getattr(state, "current_turn_ended", True)) if state is not None else False
                    if turn_ended and not pending_speech_context:
                        force_listen = True

            if not force_listen and not redirect_listen:
                continue
            if force_listen and segment_key in force_listen_segments:
                continue
            if force_listen:
                logits[row_idx, :] = float("-inf")
                logits[row_idx, listen_id] = 0.0
                force_listen_segments.add(segment_key)
            elif 0 <= tts_bos_id < logits.shape[-1]:
                if logits[row_idx, listen_id] > logits[row_idx, tts_bos_id]:
                    logits[row_idx, tts_bos_id] = logits[row_idx, listen_id]
                logits[row_idx, listen_id] = float("-inf")

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
            embeds = input_embeds if input_embeds is not None else self.get_input_embeddings(input_ids)
            return input_ids, embeds, {"duplex": {"prefill_success": False, "reason": "bad_duplex_payload"}}

        session_key = (session_id, incarnation)
        state = helper.sessions.get(session_key)
        if state is None:
            from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
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

        audio_waveform = helper._decode_audio_payload(payload)
        try:
            video_frames = helper._decode_video_frames_payload(payload)
        except ValueError as exc:
            embeds = input_embeds if input_embeds is not None else self.get_input_embeddings(input_ids)
            return input_ids, embeds, {"duplex": {"prefill_success": False, "reason": str(exc)}}
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
        result = helper._stage_prefill_embeddings_only(
            state,
            audio_waveform,
            video_frames=video_frames,
            epoch=epoch,
            seq=seq,
            is_speech=bool(payload.get("is_speech", False)),
            final=bool(duplex.get("final")),
        )
        update_result = dict(result)
        update_result.pop("inputs_embeds", None)
        if result.get("success") is not True:
            embeds = input_embeds if input_embeds is not None else self.get_input_embeddings(input_ids)
            return input_ids, embeds, {"duplex": update_result}

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
        pad_token_id = helper.stage_padding_token_id()
        if prompt_len > int(full_req_embeds.shape[0]):
            pad_len = prompt_len - int(full_req_embeds.shape[0])
            pad_ids = torch.full(
                (pad_len,),
                pad_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            pad_embeds = self.get_input_embeddings(pad_ids).to(dtype=full_req_embeds.dtype)
            # The appended duplex tokens occupy the tail of the request prompt
            # and the runner schedules the span [num_computed_tokens, prompt_len).
            # Padding must therefore sit in front of the real chunk embeddings;
            # otherwise the audio lands outside the scheduled span, is never
            # forwarded, and generation runs on pad tokens only. Keeping the
            # embeddings last also places the decode position directly after the
            # final audio embedding, matching the official listen/speak decision
            # point.
            full_req_embeds = torch.cat([pad_embeds, full_req_embeds], dim=0)
            full_input_token_ids = [pad_token_id] * pad_len + full_input_token_ids
        elif prompt_len < int(full_req_embeds.shape[0]):
            logger.warning(
                "MiniCPM-o duplex append produced %d embeddings but the scheduler "
                "reserved only %d prompt slots; the tail will be truncated. "
                "Increase the duplex scheduler token budget.",
                int(full_req_embeds.shape[0]),
                prompt_len,
            )

        span_len = int(input_ids.shape[0])
        token_offset = kwargs.get("duplex_token_offset", 0)
        try:
            token_offset = max(0, int(token_offset))
        except (TypeError, ValueError):
            token_offset = 0
        req_embeds = full_req_embeds[token_offset : token_offset + span_len]
        if req_embeds.shape[0] < span_len:
            pad_ids = torch.full(
                (span_len - req_embeds.shape[0],),
                pad_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            pad_embeds = self.get_input_embeddings(pad_ids).to(dtype=req_embeds.dtype)
            req_embeds = torch.cat([req_embeds, pad_embeds], dim=0)
        elif req_embeds.shape[0] > span_len:
            req_embeds = req_embeds[:span_len]

        input_token_ids = full_input_token_ids[token_offset : token_offset + span_len]
        if len(input_token_ids) < span_len:
            input_token_ids.extend([pad_token_id] * (span_len - len(input_token_ids)))
        if input_token_ids:
            req_input_ids = torch.tensor(input_token_ids, dtype=input_ids.dtype, device=input_ids.device)
            update_result["duplex_prompt_token_ids"] = full_input_token_ids
        else:
            req_input_ids = torch.full_like(input_ids, helper._required_token_id("unit_token_id"))
        return req_input_ids, req_embeds, {"duplex": update_result}

    def _duplex_data_plane_helper(self):
        helper = getattr(self, "_minicpmo45_duplex_data_plane_helper", None)
        if helper is not None:
            return helper
        from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import MiniCPMO45Stage0DuplexRuntime

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

    def _run_tts_request(
        self,
        *,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        talker_info: dict[str, Any],
        device: torch.device,
    ) -> tuple[dict[str, Any] | None, bool, bool]:
        meta_info = talker_info.get("meta") if isinstance(talker_info.get("meta"), dict) else {}
        if talker_info.get("native_duplex") is True:
            # ids/hidden_states may be accumulated for the talker KV stream,
            # but displayed transcript belongs to the current thinker segment.
            tts_text = meta_info.get("native_duplex_segment_text", "")
        else:
            tts_text = talker_info.get("llm_output_text", "")
        if isinstance(tts_text, list):
            tts_text = (
                tts_text[-1]
                if talker_info.get("native_duplex") is True and tts_text
                else (tts_text[0] if tts_text else "")
            )
        if not isinstance(tts_text, str):
            tts_text = ""

        with torch.inference_mode():
            talker_result = self.talker(
                input_ids=input_ids,
                positions=positions,
                inputs_embeds=inputs_embeds,
                additional_information=talker_info,
            )
        chunk_flags = getattr(self.talker, "_ar_last_chunk_flags", [True])
        chunk_is_last = bool(chunk_flags[-1]) if chunk_flags else True
        turn_end_flags = getattr(self.talker, "_ar_turn_end_flags", [False])
        turn_ended = bool(turn_end_flags[-1]) if turn_end_flags else False
        if talker_info.get("native_duplex") is True:
            emitted_text = getattr(self.talker, "_ar_last_emitted_text", "")
            tts_text = emitted_text if isinstance(emitted_text, str) else ""

        if not (isinstance(talker_result, tuple) and len(talker_result) == 2):
            return None, chunk_is_last, turn_ended

        mel_spec, waveform = talker_result
        mm_out: dict[str, Any] = {}
        duplex_info = talker_info.get("duplex") if isinstance(talker_info.get("duplex"), dict) else {}
        if talker_info.get("native_duplex") is True:
            turn_id = duplex_info.get("turn_id")
            if isinstance(turn_id, int):
                mm_out["meta.duplex_turn_id"] = torch.tensor([turn_id], dtype=torch.int32, device=device)
            epoch = duplex_info.get("epoch")
            if isinstance(epoch, int):
                mm_out["meta.duplex_epoch"] = torch.tensor([epoch], dtype=torch.int32, device=device)
        mm_out["meta.tts_is_last_chunk"] = torch.tensor(
            [int(chunk_is_last)],
            dtype=torch.int32,
            device=device,
        )
        if talker_info.get("native_duplex") is True:
            mm_out["meta.turn_end"] = torch.tensor(
                [int(turn_ended)],
                dtype=torch.int32,
                device=device,
            )
        if tts_text or talker_info.get("native_duplex") is True:
            mm_out["meta.llm_output_text_utf8"] = torch.tensor(
                list(tts_text.encode("utf-8")),
                dtype=torch.uint8,
                device=device,
            )
            mm_out["meta.audio_text_total_chars"] = torch.tensor(
                [len(tts_text)],
                dtype=torch.int32,
                device=device,
            )
        if mel_spec is not None:
            mm_out["mel_spec"] = [mel_spec]
        if waveform is not None:
            mm_out["model_outputs"] = [waveform]
        elif mel_spec is not None:
            mm_out["model_outputs"] = [mel_spec]
        return mm_out, chunk_is_last, turn_ended

    @staticmethod
    def _slice_tts_request_rows(value: torch.Tensor | None, start: int, end: int) -> torch.Tensor | None:
        return value[start:end] if isinstance(value, torch.Tensor) else value

    def _run_batched_tts_requests(
        self,
        *,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        runtime_info: list[object],
        request_token_spans: object,
        num_tokens: int,
        hidden_dim: int,
        device: torch.device,
    ) -> OmniOutput:
        if not isinstance(request_token_spans, list) or len(request_token_spans) != len(runtime_info):
            raise RuntimeError(
                "MiniCPM-o 4.5 batched TTS requires one request_token_spans entry "
                f"per request; got {request_token_spans!r} for {len(runtime_info)} requests"
            )

        request_ids: list[str] = []
        request_outputs: list[dict[str, Any]] = []
        row_chunk_flags = [True] * num_tokens
        row_turn_end_flags = [False] * num_tokens
        previous_end = 0
        for index, raw_info in enumerate(runtime_info):
            if not isinstance(raw_info, dict):
                raise RuntimeError(f"MiniCPM-o 4.5 TTS request metadata at index {index} is not a dict")
            span = request_token_spans[index]
            if not (
                isinstance(span, (list, tuple)) and len(span) == 2 and all(isinstance(value, int) for value in span)
            ):
                raise RuntimeError(f"Invalid MiniCPM-o 4.5 TTS token span at index {index}: {span!r}")
            start, end = span
            if start != previous_end or start < 0 or end < start or end > num_tokens:
                raise RuntimeError(
                    f"Invalid MiniCPM-o 4.5 TTS token span at index {index}: "
                    f"expected start {previous_end} and 0 <= start <= end <= {num_tokens}, got {span!r}"
                )
            previous_end = end
            if start == end:
                continue
            mm_out, chunk_is_last, turn_ended = self._run_tts_request(
                input_ids=self._slice_tts_request_rows(input_ids, start, end),
                positions=self._slice_tts_request_rows(positions, start, end),
                inputs_embeds=self._slice_tts_request_rows(inputs_embeds, start, end),
                talker_info=raw_info,
                device=device,
            )
            row_chunk_flags[start:end] = [chunk_is_last] * (end - start)
            row_turn_end_flags[start:end] = [turn_ended] * (end - start)
            if mm_out is not None:
                request_ids.append(get_stream_request_key(raw_info))
                request_outputs.append(mm_out)
        # vLLM pads the flattened token tensor to a CUDA Graph capture size.
        # The spans describe only scheduled request rows, so an uncovered tail
        # is graph padding rather than a missing request. Contiguity checks
        # above still reject gaps between real requests.

        self.talker._ar_last_chunk_flags = row_chunk_flags
        self.talker._ar_turn_end_flags = row_turn_end_flags
        dummy_hidden = torch.zeros(num_tokens, hidden_dim, device=device)
        if not request_outputs:
            return OmniOutput(text_hidden_states=dummy_hidden, multimodal_outputs=None)

        sparse_output: dict[str, Any] = {
            "meta.req_id": request_ids,
            "meta.sparse_audio": ["1"],
        }
        output_keys = set().union(*(output.keys() for output in request_outputs))
        for key in output_keys:
            values: list[Any] = []
            for output in request_outputs:
                value = output.get(key)
                if isinstance(value, list) and len(value) == 1:
                    value = value[0]
                if value is None:
                    dtype = torch.uint8 if key == "meta.llm_output_text_utf8" else torch.float32
                    value = torch.empty(0, dtype=dtype, device=device)
                values.append(value)
            sparse_output[key] = values
        return OmniOutput(text_hidden_states=dummy_hidden, multimodal_outputs=sparse_output)

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
        1) LLM: multimodal thinker → hidden states and text tokens
        2) TTS: talker + Token2Wav → speech waveform
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

        # Talker stage: runs TTS generation and its built-in Token2Wav vocoder.
        if self.model_stage == "tts":
            if input_ids is not None:
                num_tokens = input_ids.shape[0]
                device = input_ids.device
            elif inputs_embeds is not None:
                num_tokens = inputs_embeds.shape[0]
                device = inputs_embeds.device
            else:
                num_tokens = 1
                device = current_omni_platform.get_torch_device()
            hidden_dim = self.config.hidden_size if hasattr(self.config, "hidden_size") else 2560

            # Profile/dummy run: both input_ids and inputs_embeds are None.
            # Note: SupportsMultiModal preprocessing converts input_ids to
            # inputs_embeds, so input_ids=None alone does NOT indicate a dummy run.
            if input_ids is None and inputs_embeds is None:
                dummy_hidden = torch.zeros(num_tokens, hidden_dim, device=device)
                return OmniOutput(text_hidden_states=dummy_hidden, multimodal_outputs=None)

            runtime_info = kwargs.get("runtime_additional_information")
            request_token_spans = kwargs.get("request_token_spans")
            if isinstance(runtime_info, list) and (len(runtime_info) > 1 or request_token_spans is not None):
                return self._run_batched_tts_requests(
                    input_ids=input_ids,
                    positions=positions,
                    inputs_embeds=inputs_embeds,
                    runtime_info=runtime_info,
                    request_token_spans=request_token_spans,
                    num_tokens=num_tokens,
                    hidden_dim=hidden_dim,
                    device=device,
                )
            talker_info = {}
            if runtime_info and isinstance(runtime_info, list) and len(runtime_info) > 0:
                talker_info = runtime_info[0] if isinstance(runtime_info[0], dict) else {}
            dummy_hidden = torch.zeros(num_tokens, hidden_dim, device=device)
            mm_out, _, _ = self._run_tts_request(
                input_ids=input_ids,
                positions=positions,
                inputs_embeds=inputs_embeds,
                talker_info=talker_info,
                device=device,
            )
            if mm_out is not None:
                return OmniOutput(text_hidden_states=dummy_hidden, multimodal_outputs=mm_out)

            return OmniOutput(text_hidden_states=dummy_hidden, multimodal_outputs=None)

        raise ValueError(f"Unsupported model stage: {self.model_stage}")

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput) -> torch.Tensor | None:
        # Handle OmniOutput type
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states

        # Use model for logits computation
        return self.model.compute_logits(hidden_states)

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
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
        if not native_rows or len(native_rows) != logits.shape[0]:
            return None

        sampled_ids: list[int] = []
        for row_idx in range(logits.shape[0]):
            row_logits = logits[row_idx : row_idx + 1].clone()
            sampled = self._sample_minicpmo45_native_duplex_row(
                row_logits,
                sampling_metadata,
                row_idx=row_idx,
                token_ids=token_ids,
            )
            self._record_minicpmo45_duplex_terminator(row_idx, sampled, token_ids)
            sampled_ids.append(sampled)
        return SamplerOutput(
            sampled_token_ids=torch.tensor(sampled_ids, device=logits.device, dtype=torch.int32).unsqueeze(-1),
            logprobs_tensors=None,
        )

    def _sample_minicpmo45_native_duplex_row(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        *,
        row_idx: int,
        token_ids: dict[str, int],
    ) -> int:
        chunk_eos_id = token_ids.get("chunk_eos_token_id", -1)
        generator = getattr(sampling_metadata, "generators", {}).get(row_idx)
        output_token_ids = getattr(sampling_metadata, "output_token_ids", None) or []
        raw_recent_tokens = output_token_ids[row_idx] if row_idx < len(output_token_ids) else []
        recent_tokens = [int(token_id) for token_id in raw_recent_tokens if isinstance(token_id, int) and token_id >= 0]
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
                return int(chunk_eos_id)

            # Match the released StreamDecoder: first sample the original
            # distribution only to preserve the model's own chunk boundary.
            # If it does not choose chunk_eos, mask that token before the
            # normal text/listen/turn sampling pass below.
            if getattr(sampling_metadata, "all_greedy", False):
                boundary_sample = int(torch.argmax(logits, dim=-1).item())
            else:
                boundary_probs = F.softmax(logits, dim=-1)
                boundary_sample = int(torch.multinomial(boundary_probs, num_samples=1, generator=generator).item())
            if boundary_sample == chunk_eos_id:
                return int(chunk_eos_id)

        logits = logits.clone()
        forbidden = self._minicpmo45_native_forbidden_token_ids(token_ids)
        if forbidden:
            valid_forbidden = [token_id for token_id in forbidden if 0 <= token_id < logits.shape[-1]]
            if valid_forbidden:
                logits[:, valid_forbidden] = float("-inf")

        special_ids = self._minicpmo45_native_special_token_ids(token_ids)
        state = self._minicpmo45_duplex_state_for_row(row_idx)
        generated_text_tokens = getattr(state, "generated_text_tokens", None)
        repetition_tokens = generated_text_tokens if generated_text_tokens else recent_tokens
        repetition_penalty = 1.05
        if repetition_penalty != 1.0 and repetition_tokens:
            history_size = MiniCPMO45DuplexPolicy.REPETITION_HISTORY_SIZE
            for token_id in set(repetition_tokens[-history_size:]):
                if token_id in special_ids or token_id < 0 or token_id >= logits.shape[-1]:
                    continue
                logits[0, token_id] /= repetition_penalty

        temperature = float(self._sampling_metadata_value(sampling_metadata, "temperature", row_idx, 0.7))
        top_k = int(self._sampling_metadata_value(sampling_metadata, "top_k", row_idx, 100))
        top_p = float(self._sampling_metadata_value(sampling_metadata, "top_p", row_idx, 0.8))
        if getattr(sampling_metadata, "all_greedy", False) or temperature <= 0:
            sampled = int(torch.argmax(logits, dim=-1).item())
            sampled = self._maybe_cut_minicpmo45_native_duplex_text_chunk(
                sampled,
                recent_tokens,
                token_ids,
            )
            return self._finalize_minicpmo45_native_duplex_sample(row_idx, sampled, token_ids)

        logits = logits / temperature
        logits = self._top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits, dim=-1)
        sampled = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
        sampled = self._maybe_cut_minicpmo45_native_duplex_text_chunk(
            sampled,
            recent_tokens,
            token_ids,
        )
        return self._finalize_minicpmo45_native_duplex_sample(
            row_idx,
            sampled,
            token_ids,
        )

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
        force_listen = isinstance(payload, dict) and payload.get("force_listen") is True
        listen_id = token_ids.get("listen_token_id", -1)
        tts_bos_id = token_ids.get("tts_bos_token_id", -1)
        chunk_eos_id = token_ids.get("chunk_eos_token_id", -1)
        chunk_tts_eos_id = token_ids.get("chunk_tts_eos_token_id", -1)
        turn_eos_id = token_ids.get("turn_eos_token_id", -1)
        terminators = {listen_id, chunk_eos_id, chunk_tts_eos_id, turn_eos_id}
        if sampled in terminators:
            state.pending_terminator_token = int(sampled)
            state.last_terminator_token = int(sampled)
            if sampled == turn_eos_id or (sampled == listen_id and force_listen):
                state.current_turn_ended = True
                with suppress(Exception):
                    state.pending_speech_response_open = False
            return
        special_ids = self._minicpmo45_native_special_token_ids(token_ids)
        if sampled not in special_ids:
            generated_text_tokens = getattr(state, "generated_text_tokens", None)
            if not isinstance(generated_text_tokens, list):
                generated_text_tokens = []
                state.generated_text_tokens = generated_text_tokens
            generated_text_tokens.append(int(sampled))
            history_size = MiniCPMO45DuplexPolicy.REPETITION_HISTORY_SIZE
            del generated_text_tokens[:-history_size]
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
        #   talker:  tts (ConditionalChatTTS)
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
