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
from dataclasses import dataclass
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
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.utils import add_prefix_to_loaded_weights
from vllm_omni.platforms import current_omni_platform
from vllm_omni.utils.device_copy import index_to_device, to_device_nonblocking

logger = init_logger(__name__)


@dataclass(slots=True)
class _MiniCPMO45PendingSamples:
    """A deferred Stage-0 duplex step awaiting its host commit.

    ``host`` receives (final token, raw stage-2 sample, boundary hit) per
    row once ``event`` completes; ``rewinds`` maps a row to the generator
    offset before its stage-2 draw, restored if the boundary resolved it.
    The row maps are the step's own, since the next step replaces them.
    """

    host: torch.Tensor
    event: Any
    row_idxs: list[int]
    stage2: list[bool]
    rewinds: dict[int, tuple[torch.Generator, Any]]
    token_ids: dict[str, int]
    row_sessions: dict[int, str] | None
    row_payloads: dict[int, dict[str, Any]] | None


def _generator_mark(generator: torch.Generator) -> Any:
    """Where ``generator`` stands: its Philox offset (a host value), else its full state."""
    try:
        return ("offset", generator.get_offset())
    except RuntimeError:  # not a Philox generator (CPU)
        return ("state", generator.get_state())


def _generator_rewind(generator: torch.Generator, mark: Any) -> None:
    """Put ``generator`` back where ``_generator_mark`` found it, undoing later draws."""
    kind, value = mark
    if kind == "offset":
        generator.set_offset(value)
    else:
        generator.set_state(value)


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

    requires_raw_input_tokens = True

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
        self._use_v2_model_runner = bool(getattr(vllm_config.model_config, "use_v2_model_runner", False))
        if (
            self.model_stage == "llm"
            and self._use_v2_model_runner
            and getattr(vllm_config.model_config, "session_mode", "turn") != "turn"
        ):
            raise NotImplementedError("MiniCPM-o duplex Thinker requires model_runner: v1")
        if (
            self.model_stage == "llm"
            and self._use_v2_model_runner
            and getattr(vllm_config.model_config, "async_chunk", False)
        ):
            raise ValueError("MiniCPM-o MRv2 Thinker requires async_chunk: false for its full llm2tts payload")
        # The Thinker's row ledger needs real token identities even when
        # embeddings are supplied, including during CUDA graph capture/replay.
        self.requires_raw_input_tokens = self.model_stage == "llm"

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

            if getattr(getattr(vllm_config, "model_config", None), "session_mode", None) == "duplex":
                from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_kv import (
                    DUPLEX_WINDOW_BLOCK_SIZE,
                    duplex_window_geometry,
                    install_duplex_window_layers,
                    validate_duplex_window_install,
                )

                cache_config = getattr(vllm_config, "cache_config", None)
                model_config = getattr(vllm_config, "model_config", None)
                block_size = int(
                    getattr(cache_config, "block_size", DUPLEX_WINDOW_BLOCK_SIZE) or DUPLEX_WINDOW_BLOCK_SIZE
                )
                max_model_len = getattr(model_config, "max_model_len", None) if model_config is not None else None
                if max_model_len is None:
                    max_model_len = 8192

                geometry = duplex_window_geometry(
                    prefix_tokens=96,
                    window_tokens=6000,
                    block_size=block_size,
                    max_model_len=max_model_len,
                    high_watermark_tokens=8000,
                )
                install_duplex_window_layers(self.thinker, geometry=geometry)
                validate_duplex_window_install(
                    cache_config,
                    model_config,
                    geometry,
                )

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
            # The runner looks this hook up on the wrapper. Without it every
            # decode row runs scalar preprocess, which reads its codec id with
            # a blocking ``.item()``.
            batch_decode = getattr(self.talker, "preprocess_decode_batch", None)
            if callable(batch_decode):
                self.preprocess_decode_batch = batch_decode
            # Model Runner V2 hooks: device-side codec output, EOS control and
            # codec penalty (see MiniCPMO45OmniTTSForConditionalGeneration).
            for hook_name in ("preprocess_decode_batch_mrv2", "make_omni_output_mrv2", "mrv2_custom_sampler"):
                hook = getattr(self.talker, hook_name, None)
                if callable(hook):
                    setattr(self, hook_name, hook)
            self.mrv2_decode_preprocess_is_identity = bool(
                getattr(self.talker, "mrv2_decode_preprocess_is_identity", False)
            )
            self.logits_vocab_size = int(self.talker.logits_vocab_size)
            # Opt-in (hf_overrides talker_kstep_graph_sampling): the CUDA K-step
            # loop replays captured per-frame sampling tails (talker_kstep_graph).
            if getattr(config, "talker_kstep_graph_sampling", False) is True:
                from vllm_omni.model_executor.models.minicpmo_4_5.talker_kstep_graph import (
                    TalkerKStepFrameGraphs,
                )

                self._kstep_frame_graphs = TalkerKStepFrameGraphs()
                logger.info("[minicpmo] Talker K-step frame graphs on (talker_kstep_graph_sampling)")
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
        # Turn-mode Thinker inputs use the native multimodal encoder/cache on
        # MRv2. Marking it as a custom-preprocess model disables that encoder
        # path in the runner. Duplex still uses the V1 preprocess hook.
        self.has_preprocess = self.model_stage == "tts" or not self._use_v2_model_runner
        # Neither AR stage has a postprocess, so step outputs can use the
        # runner's async snapshot instead of a blocking per-step D2H.
        self.use_async_omni_output = self.model_stage in {"llm", "tts"}

        if self.model_stage == "llm" and getattr(vllm_config.model_config, "session_mode", "turn") == "duplex":
            # Build the Stage-0 duplex runtime (remote-code processor and
            # tokenizer) with the model. Built lazily, it costs several seconds
            # inside the first session's first audio unit, and the session then
            # runs that far behind the real-time input stream. The loader
            # constructs the model under the target-device context; the
            # processor is CPU preprocessing, so keep its tensors on the CPU.
            with torch.device("cpu"):
                self._duplex_data_plane_helper()

    @cached_property
    def sampler(self):
        if hasattr(self.model, "sampler"):
            return self.model.sampler
        from vllm.v1.sample.sampler import Sampler

        return Sampler()

    def apply_duplex_kv_reanchor(self, runner: Any, scheduler_output: Any = None) -> None:
        """Apply in-place Stage-0 KV reanchor and rotation on worker before model forward."""
        from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_kv import (
            MiniCPMO45DuplexWorkerHelper,
        )

        MiniCPMO45DuplexWorkerHelper.maybe_apply_reanchor(runner, scheduler_output=scheduler_output)

    def prepare_duplex_sampling(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        rows: tuple[DuplexSamplingRow, ...],
    ) -> None:
        """Apply MiniCPM duplex policy before the standard model sampler."""
        del sampling_metadata
        # The previous step's deferred decisions update the session state read below.
        self._commit_minicpmo45_duplex_pending_samples()
        self._minicpmo45_active_duplex_rows = [row.row_idx for row in rows]
        self._minicpmo45_duplex_row_sampling_host = {
            row.row_idx: (row.temperature, row.top_k, row.top_p)
            for row in rows
            if row.temperature is not None and row.top_k is not None and row.top_p is not None
        }
        self._minicpmo45_duplex_row_sessions = {
            row.row_idx: row.session_id for row in rows if row.session_id is not None
        }
        request_sessions = getattr(self, "_minicpmo45_duplex_request_sessions", None)
        if not isinstance(request_sessions, dict):
            request_sessions = {}
            self._minicpmo45_duplex_request_sessions = request_sessions
        request_sessions.update({row.request_id: row.session_id for row in rows if row.session_id is not None})
        self._minicpmo45_duplex_row_payloads = {row.row_idx: row.payload for row in rows if row.payload is not None}
        self._minicpmo45_duplex_row_max_tokens = {
            row.row_idx: row.max_tokens for row in rows if row.max_tokens is not None
        }
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
            row_idx = row.row_idx
            if row_idx < 0 or row_idx >= logits.shape[0]:
                continue
            payload = row.payload
            if not isinstance(payload, dict):
                continue
            force_listen = payload.get("force_listen") is True
            is_speech = payload.get("is_speech")
            segment_key = (row.request_id, row.seq if row.seq is not None else -1)
            session_key = row.session_id
            if turn_eos_id >= 0 and session_key is not None:
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
                # fill_ keeps the scalars on the host; assigning one element
                # copies a host scalar tensor and waits for the device.
                logits[row_idx].fill_(float("-inf"))
                logits[row_idx].narrow(0, listen_id, 1).fill_(0.0)
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
        payload = duplex.get("payload")
        if not session_id or not isinstance(payload, dict):
            embeds = input_embeds if input_embeds is not None else self.get_input_embeddings(input_ids)
            return input_ids, embeds, {"duplex": {"prefill_success": False, "reason": "bad_duplex_payload"}}

        # A deferred sample of this session updates the latches its append reads.
        self._commit_minicpmo45_duplex_pending_samples(session_ids={session_id})
        state = self._minicpmo45_duplex_session_state(helper, session_id, duplex)
        prefill_kwargs = self._minicpmo45_duplex_prefill_kwargs(duplex, payload)
        seq = prefill_kwargs["seq"]
        result = helper.take_staged_prefill(state, prefill_kwargs["epoch"], seq)
        if result is None:  # not built by this step's preprocess_batch
            audio_waveform = helper._decode_audio_payload(payload)
            try:
                prefill_kwargs.update(self._minicpmo45_duplex_append_frames(helper, duplex, payload))
            except ValueError as exc:
                embeds = input_embeds if input_embeds is not None else self.get_input_embeddings(input_ids)
                return input_ids, embeds, {"duplex": {"prefill_success": False, "reason": str(exc)}}
            result = helper._stage_prefill_embeddings_only(state, audio_waveform, **prefill_kwargs)
        update_result = dict(result)
        if result.get("stage0_window_replaced") is True:
            window = duplex.get("stage0_window", {})
            logger.info(
                "MiniCPM-o Stage-0 window replaced: mode=%s drop_units=%s tokens=%s seq=%s",
                window.get("mode"),
                window.get("drop_units"),
                result.get("num_input_tokens"),
                seq,
            )
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
            # Pinned + non-blocking: a pageable copy would wait for all queued GPU work.
            req_input_ids = index_to_device(input_token_ids, input_ids.device, input_ids.dtype)
            update_result["duplex_prompt_token_ids"] = full_input_token_ids
        else:
            req_input_ids = torch.full_like(input_ids, helper._required_token_id("unit_token_id"))
        return req_input_ids, req_embeds, {"duplex": update_result}

    def preprocess_batch(
        self,
        *,
        req_ids: list[str],
        model_intermediate_buffer: dict[str, dict[str, Any]],
        device: torch.device,
    ) -> None:
        """Runner hook: build this step's new duplex appends across sessions at once.

        Called before the per-request ``preprocess`` loop; both batches only
        pay off with two or more appends. The camera frames of every append go
        through one vision-tower call (``prefetch_vision``), then every append
        not built yet is prefilled with its streaming-encoder units batched
        across sessions (``stage_prefill_batch``). ``preprocess`` takes the
        staged results; anything skipped here is built or reported there.
        """
        del device
        if self.model_stage != "llm":
            return
        appends: list[dict[str, Any]] = []
        for req_id in req_ids:
            info = model_intermediate_buffer.get(req_id)
            duplex = info.get("duplex") if isinstance(info, dict) else None
            if isinstance(duplex, dict) and duplex.get("data_plane") is True:
                appends.append(duplex)
        frame_appends = [
            duplex
            for duplex in appends
            if isinstance(duplex.get("payload"), dict) and duplex["payload"].get("video_frames")
        ]
        helper = getattr(self, "_minicpmo45_duplex_data_plane_helper", None)
        if len(frame_appends) < 2:
            if helper is not None:
                helper.prefetch_vision([])  # drop entries the previous step did not consume
        else:
            try:
                helper = self._duplex_data_plane_helper()
                helper.prefetch_vision(frame_appends)
            except Exception:  # noqa: BLE001 - preprocess encodes each append itself instead
                logger.warning("MiniCPM-o duplex frame prefetch failed; encoding per request", exc_info=True)
        if helper is None or len(appends) < 2 or not helper.batches_audio_encoder():
            return
        candidates: list[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        seen: set[str] = set()
        for duplex in appends:
            session_id = str(duplex.get("session_id") or "")
            payload = duplex.get("payload")
            if not session_id or not isinstance(payload, dict) or session_id in seen:
                # A second append of one session keeps its order: preprocess.
                seen.add(session_id)
                continue
            seen.add(session_id)
            prefill_kwargs = self._minicpmo45_duplex_prefill_kwargs(duplex, payload)
            seq = prefill_kwargs["seq"]
            if seq is not None and helper.needs_prefill(helper.sessions.get(session_id), prefill_kwargs["epoch"], seq):
                candidates.append((session_id, duplex, payload, prefill_kwargs))
        if len(candidates) < 2:
            return
        self._commit_minicpmo45_duplex_pending_samples(session_ids={candidate[0] for candidate in candidates})
        batch = []
        for session_id, duplex, payload, prefill_kwargs in candidates:
            try:
                audio_waveform = helper._decode_audio_payload(payload)
                prefill_kwargs.update(self._minicpmo45_duplex_append_frames(helper, duplex, payload))
            except ValueError:
                continue
            state = self._minicpmo45_duplex_session_state(helper, session_id, duplex)
            batch.append((state, audio_waveform, prefill_kwargs))
        if len(batch) >= 2:
            helper.stage_prefill_batch(batch)

    @staticmethod
    def _minicpmo45_duplex_append_frames(helper, duplex: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        """The append's prefetched frame embeddings, else its decoded frames (may raise ValueError)."""
        encoded_frames = helper.take_prefetched_vision(duplex)
        if encoded_frames is not None:
            return {"encoded_frames": encoded_frames}
        return {"video_frames": helper._decode_video_frames_payload(payload)}

    def _minicpmo45_duplex_session_state(self, helper, session_id: str, duplex: dict[str, Any]):
        """The Stage-0 state of ``session_id``, created with its session context on first use."""
        state = helper.sessions.get(session_id)
        if state is None:
            from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import (
                _MiniCPMO45Stage0SessionState,
            )

            state = _MiniCPMO45Stage0SessionState(session_id=session_id)
            helper.sessions[session_id] = state
            session_config = duplex.get("session_config")
            session_config = dict(session_config) if isinstance(session_config, dict) else {}
            runtime_config = duplex.get("runtime_config")
            runtime_config = dict(runtime_config) if isinstance(runtime_config, dict) else {}
            if hasattr(helper.thinker, "audio_past_key_values"):
                helper.thinker.audio_past_key_values = None
            helper._configure_streaming_processor(state)
            helper._prepare_session_context(state, session_config, runtime_config=runtime_config)
        return state

    @staticmethod
    def _minicpmo45_duplex_prefill_kwargs(duplex: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        """Keyword arguments of ``_stage_prefill_embeddings_only`` for one append, frames aside."""
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
        return {
            "epoch": epoch,
            "seq": seq,
            "is_speech": bool(payload.get("is_speech", False)),
            "final": bool(duplex.get("final")),
            "stage0_window": duplex.get("stage0_window") if isinstance(duplex.get("stage0_window"), dict) else None,
            "stage0_reanchor": (
                duplex.get("stage0_reanchor") if isinstance(duplex.get("stage0_reanchor"), dict) else None
            ),
        }

    def omni_post_load(self) -> None:
        """Runner hook once the weights are loaded and the model is in eval mode.

        The runner calls it once, at the start of its profile run (outside the
        worker's load-time allocator scope). Captures the duplex Stage-0 streaming audio encoder's CUDA graphs on the
        thinker's device (``duplex_audio_encoder_cuda_graph``, default on). The
        runtime that owns them is built in ``__init__``, where the encoder is not
        loaded yet.
        """
        helper = getattr(self, "_minicpmo45_duplex_data_plane_helper", None)
        build = getattr(helper, "build_audio_cuda_graph", None)
        if not callable(build):
            return
        device = self._module_device(self.thinker if self.thinker is not None else self)
        if device.type == "cuda":
            with torch.cuda.device(device):
                build()
        else:
            build()

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
                _, text_hidden_states = thinker_output
            else:
                text_hidden_states = thinker_output

            # Prepare hidden states for downstream stages
            # Ensure correct shape: (seq_len, hidden_dim)
            if text_hidden_states.ndim == 3 and text_hidden_states.shape[0] == 1:
                text_hidden_states = text_hidden_states.squeeze(0)

            if getattr(self, "_use_v2_model_runner", False):
                # FULL graphs return tensors. The row ledger is reconstructed
                # from this step's live input batch after replay, never from
                # Python values or token buffers captured during warmup.
                return text_hidden_states

            # Return hidden states with latent in multimodal_outputs for stage_input_processors
            multimodal_outputs = {"latent": text_hidden_states}
            # Keep per-forward row identities alongside the latent payload.
            if thinker_input_ids is not None and thinker_positions is not None:
                multimodal_outputs["latent_input_ids"] = thinker_input_ids.reshape(-1, 1)
                multimodal_outputs["latent_positions"] = thinker_positions.reshape(-1, 1)

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
                    # Host tensors: every consumer reads these back on the host, and a
                    # pageable host->device copy per row and key would wait for the
                    # whole forward, serializing the step.
                    multimodal_outputs["meta"] = {
                        key: [
                            torch.tensor([int(value)], dtype=torch.long)
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

    def make_omni_output_mrv2(self, model_outputs, *, input_batch, req_states, model_intermediate_buffer):
        """Attach the Thinker row identities used by the llm2tts bridge."""
        if self.model_stage != "llm":
            return self.talker.make_omni_output_mrv2(
                model_outputs,
                input_batch=input_batch,
                req_states=req_states,
                model_intermediate_buffer=model_intermediate_buffer,
            )
        if any(isinstance(info, dict) and info.get("duplex") for info in model_intermediate_buffer):
            raise NotImplementedError("MiniCPM-o duplex Thinker requires model_runner: v1")
        num_tokens = model_outputs.shape[0]
        return OmniOutput(
            text_hidden_states=model_outputs,
            multimodal_outputs={
                "latent": model_outputs,
                "latent_input_ids": input_batch.input_ids[:num_tokens].reshape(-1, 1),
                "latent_positions": input_batch.positions[:num_tokens].reshape(-1, 1),
            },
        )

    def make_omni_output(self, model_outputs, **kwargs):
        if self.model_stage != "tts":
            return model_outputs
        return self.talker.make_omni_output(model_outputs, **kwargs)

    @property
    def requires_request_sample_eligibility(self) -> bool:
        """Forward the Talker's flag: the runner only sees this wrapper.

        Without ``request_sample_eligible`` an incomplete prefill chunk would
        advance the Talker's codec history and RNG state.
        """
        talker = getattr(self, "talker", None)
        return talker is not None and bool(getattr(talker, "requires_request_sample_eligibility", False))

    # Multi-frame Talker decode: the runners resolve these through getattr on
    # this wrapper, never the inner Talker. Without them the multi-frame loop
    # silently stays off, and on NPU the rejection sampler consumes width-0
    # rows and stage 1 dies with empty audio.
    @property
    def supports_multi_frame_decode(self) -> bool:
        # Implemented by the Talker stage; stage 1's speculative_config decides
        # whether a step runs it.
        return self.model_stage == "tts"

    @property
    def _batch_stop_logits(self):
        if self.model_stage != "tts":
            return None
        return self.talker._batch_stop_logits

    def take_batch_stop_logits(self):
        if self.model_stage != "tts":
            return None
        return self.talker.take_batch_stop_logits()

    def set_batch_stop_logits(self, logits) -> None:
        if self.model_stage != "tts":
            return
        self.talker.set_batch_stop_logits(logits)

    def merge_frame_outputs(self, frame_outputs, frame_stop_logits):
        if self.model_stage != "tts":
            return frame_outputs
        return self.talker.merge_frame_outputs(frame_outputs, frame_stop_logits)

    # CUDA multi-frame decode (talker_kstep.py, talker_frame_plan.py).
    @property
    def multi_frame_decode_hook(self) -> Any:
        """The K-step loop the GPU runners drive for the Talker; None for the
        other stages and on NPU (whose runner owns its own K-step)."""
        if self.model_stage != "tts" or current_omni_platform.is_npu():
            return None
        from vllm_omni.model_executor.models.minicpmo_4_5 import talker_kstep

        return talker_kstep

    @property
    def codec_eos_token_id(self) -> int:
        return int(self.talker._codec_eos_id)

    @property
    def codec_vocab_size(self) -> int | None:
        # Stage 1's config has no vocab_size; the CUDA runner reads the codec
        # head width from here (talker_kstep.ensure_codec_vocab).
        if self.model_stage != "tts" or self.talker is None:
            return None
        width = getattr(self.talker, "_num_audio_tokens", None)
        return int(width) if width is not None else None

    def plan_codec_frames(self, infos, frames: int):
        from vllm_omni.model_executor.models.minicpmo_4_5.talker_frame_plan import plan_codec_frames

        return plan_codec_frames(self.talker, infos, frames)

    def commit_codec_frames(self, plan, forwarded: list[list[int]]) -> list[bool]:
        from vllm_omni.model_executor.models.minicpmo_4_5.talker_frame_plan import commit_codec_frames

        return commit_codec_frames(self.talker, plan, forwarded)

    def kstep_decode_frames(self, **kwargs):
        """``gpu_talker_multiframe.maybe_run``'s frames-1..K-1 hook: ``(sampled, emitted)``
        from captured per-frame tails, or None for the eager loop (always, unless
        ``talker_kstep_graph_sampling`` is on)."""
        graphs = getattr(self, "_kstep_frame_graphs", None)
        return None if graphs is None else graphs(**kwargs)

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
        self._commit_minicpmo45_duplex_pending_samples()
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

        chunk_terminators = self._minicpmo45_chunk_terminator_token_ids(token_ids)
        output_token_ids = getattr(sampling_metadata, "output_token_ids", None) or []
        num_rows = logits.shape[0]
        row_params = self._minicpmo45_duplex_row_params(sampling_metadata, num_rows)
        sampled_ids: list[int] = [-1] * num_rows
        pending_rows: list[int] = []
        for row_idx in range(num_rows):
            accepted = output_token_ids[row_idx] if row_idx < len(output_token_ids) else []
            last_accepted = next((int(t) for t in reversed(accepted) if isinstance(t, int) and t >= 0), None)
            if last_accepted in chunk_terminators:
                # Async scheduling runs one lookahead frame after the chunk
                # terminator was sampled but before the scheduler observes
                # the segment stop. The scheduler discards this frame's
                # token, so decide nothing here: re-emit the terminator and
                # leave the model-owned policy state exactly as the accepted
                # history left it. The next append re-injects that terminator.
                sampled_ids[row_idx] = last_accepted
            else:
                pending_rows.append(row_idx)

        if pending_rows:
            deferred = self._sample_minicpmo45_native_duplex_rows_deferred(
                logits,
                sampling_metadata,
                row_idxs=pending_rows,
                token_ids=token_ids,
                row_params=row_params,
            )
            if deferred is not None:
                # Lookahead rows re-emit their host-known terminator; the rest
                # take their device-decided token.
                sampled = index_to_device(sampled_ids, logits.device, torch.int32)
                sampled[index_to_device(pending_rows, logits.device)] = deferred.to(torch.int32)
                return SamplerOutput(sampled_token_ids=sampled.unsqueeze(-1), logprobs_tensors=None)
            batched = self._sample_minicpmo45_native_duplex_rows(
                logits,
                sampling_metadata,
                row_idxs=pending_rows,
                token_ids=token_ids,
                row_params=row_params,
            )
            for row_idx, sampled in zip(pending_rows, batched, strict=True):
                sampled_ids[row_idx] = sampled
                self._record_minicpmo45_duplex_terminator(row_idx, sampled, token_ids)

        return SamplerOutput(
            sampled_token_ids=index_to_device(sampled_ids, logits.device, torch.int32).unsqueeze(-1),
            logprobs_tensors=None,
        )

    def _sample_minicpmo45_native_duplex_rows(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        *,
        row_idxs: list[int],
        token_ids: dict[str, int],
        row_params: list[tuple[float, float, float]],
    ) -> list[int]:
        """``_sample_minicpmo45_native_duplex_row`` for many rows with at most two host reads.

        A row with its own generator draws exactly as in the per-row loop. Rows
        sharing the process-wide generator still draw once each, but all
        boundary draws come before all stage-2 draws.
        """
        chunk_eos_id = token_ids.get("chunk_eos_token_id", -1)
        all_greedy = bool(getattr(sampling_metadata, "all_greedy", False))
        generators = getattr(sampling_metadata, "generators", {}) or {}
        recent_tokens_by_row = self._minicpmo45_duplex_recent_tokens(sampling_metadata, row_idxs)

        resolved: dict[int, int] = {}
        if 0 <= chunk_eos_id < logits.shape[-1]:
            capped = self._minicpmo45_duplex_capped_rows(row_idxs, recent_tokens_by_row)
            resolved = dict.fromkeys(capped, int(chunk_eos_id))
            boundary_rows = [row_idx for row_idx in row_idxs if row_idx not in capped]
            if boundary_rows:
                hits = self._duplex_boundary_hits(
                    logits[index_to_device(boundary_rows, logits.device)],
                    boundary_rows,
                    generators,
                    all_greedy,
                    chunk_eos_id,
                )
                for row_idx, hit in zip(boundary_rows, hits.tolist(), strict=True):
                    if hit:
                        resolved[row_idx] = int(chunk_eos_id)

        stage2_rows = [row_idx for row_idx in row_idxs if row_idx not in resolved]
        if stage2_rows:
            stage2_logits = self._minicpmo45_duplex_stage2_logits(
                logits,
                index_to_device(stage2_rows, logits.device),
                stage2_rows,
                token_ids,
                recent_tokens_by_row,
            )
            samples = self._minicpmo45_duplex_stage2_draws(
                stage2_logits, stage2_rows, row_params, all_greedy, generators
            )
            for row_idx, sampled in zip(stage2_rows, samples.tolist(), strict=True):
                sampled = int(sampled)
                self._record_minicpmo45_duplex_generation_token(row_idx, sampled)
                sampled = self._maybe_cut_minicpmo45_native_duplex_text_chunk(
                    sampled,
                    recent_tokens_by_row[row_idx],
                    token_ids,
                )
                resolved[row_idx] = self._finalize_minicpmo45_native_duplex_sample(row_idx, sampled, token_ids)

        return [resolved[row_idx] for row_idx in row_idxs]

    @staticmethod
    def _minicpmo45_duplex_recent_tokens(
        sampling_metadata: SamplingMetadata,
        row_idxs: list[int],
    ) -> dict[int, list[int]]:
        """Each row's accepted output tokens."""
        output_token_ids = getattr(sampling_metadata, "output_token_ids", None) or []
        recent_tokens_by_row: dict[int, list[int]] = {}
        for row_idx in row_idxs:
            raw_recent_tokens = output_token_ids[row_idx] if row_idx < len(output_token_ids) else []
            recent_tokens_by_row[row_idx] = [
                int(token_id) for token_id in raw_recent_tokens if isinstance(token_id, int) and token_id >= 0
            ]
        return recent_tokens_by_row

    def _minicpmo45_policy_limit(self, name: str, default: int) -> int:
        return int(getattr(self, name, default) or default)

    def _minicpmo45_duplex_capped_rows(
        self,
        row_idxs: list[int],
        recent_tokens_by_row: dict[int, list[int]],
    ) -> set[int]:
        """Rows at their per-chunk speak-length cap: forced to chunk_eos without sampling."""
        max_speak_tokens = self._minicpmo45_policy_limit(
            "max_new_speak_tokens_per_chunk",
            MiniCPMO45DuplexPolicy.DEFAULT_MAX_NEW_SPEAK_TOKENS_PER_CHUNK,
        )
        capped: set[int] = set()
        for row_idx in row_idxs:
            effective_max_speak_tokens = max_speak_tokens
            request_max_tokens = self._minicpmo45_duplex_row_request_max_tokens(row_idx)
            if request_max_tokens is not None:
                effective_max_speak_tokens = min(effective_max_speak_tokens, request_max_tokens)
            if len(recent_tokens_by_row[row_idx]) >= max(1, effective_max_speak_tokens - 1):
                capped.add(row_idx)
        return capped

    def _minicpmo45_duplex_stage2_logits(
        self,
        logits: torch.Tensor,
        index: torch.Tensor,
        row_idxs: list[int],
        token_ids: dict[str, int],
        recent_tokens_by_row: dict[int, list[int]],
    ) -> torch.Tensor:
        """Rows ``index`` of ``logits`` with the forbidden ids masked and the repetition penalty applied."""
        vocab = logits.shape[-1]
        stage2_logits = logits[index]  # a gather: already a copy
        forbidden_index = self._minicpmo45_duplex_forbidden_index(token_ids, vocab, logits.device)
        if forbidden_index is not None:
            stage2_logits.index_fill_(1, forbidden_index, float("-inf"))
        repetition_penalty = 1.05
        history_size = MiniCPMO45DuplexPolicy.REPETITION_HISTORY_SIZE
        penalty_rows: list[int] = []
        penalty_cols: list[int] = []
        for local_idx, row_idx in enumerate(row_idxs):
            generated_tokens = getattr(self._minicpmo45_duplex_state_for_row(row_idx), "generated_tokens", None)
            repetition_tokens = generated_tokens if generated_tokens else recent_tokens_by_row[row_idx]
            penalized = [token_id for token_id in set(repetition_tokens[-history_size:]) if 0 <= token_id < vocab]
            penalty_rows.extend([local_idx] * len(penalized))
            penalty_cols.extend(penalized)
        if penalty_rows:
            # Distinct (row, col) pairs, so one indexed division equals dividing each once.
            rows_t = index_to_device(penalty_rows, logits.device)
            cols_t = index_to_device(penalty_cols, logits.device)
            stage2_logits[rows_t, cols_t] = stage2_logits[rows_t, cols_t] / repetition_penalty
        return stage2_logits

    def _minicpmo45_duplex_stage2_draws(
        self,
        stage2_logits: torch.Tensor,
        row_idxs: list[int],
        row_params: list[tuple[float, float, float]],
        all_greedy: bool,
        generators: dict[int, torch.Generator],
        marks: dict[int, tuple[torch.Generator, Any]] | None = None,
    ) -> torch.Tensor:
        """Each row's stage-2 token, drawn in row order with the row's own generator.

        Greedy rows (``all_greedy`` or temperature <= 0) take the argmax; the
        rest draw in top-k/top-p candidate space. ``marks`` receives, per local
        row, the generator position just before its draw.
        """
        device = stage2_logits.device
        samples: list[torch.Tensor | None] = [None] * len(row_idxs)
        greedy_local: list[int] = []
        sampled_local: list[int] = []
        for local_idx, row_idx in enumerate(row_idxs):
            if all_greedy or float(row_params[row_idx][0]) <= 0:
                greedy_local.append(local_idx)
            else:
                sampled_local.append(local_idx)
        if greedy_local:
            greedy_sample = torch.argmax(stage2_logits[index_to_device(greedy_local, device)], dim=-1)
            for pos, local_idx in enumerate(greedy_local):
                samples[local_idx] = greedy_sample[pos : pos + 1]
        if sampled_local:
            cand_probs, cand_indices = self._duplex_stage2_candidate_probs(
                stage2_logits[index_to_device(sampled_local, device)],
                [row_params[row_idxs[local_idx]] for local_idx in sampled_local],
            )
            for pos, local_idx in enumerate(sampled_local):
                generator = generators.get(row_idxs[local_idx])
                if marks is not None:
                    marks[local_idx] = (generator, _generator_mark(generator))
                draw = torch.multinomial(cand_probs[pos], num_samples=1, generator=generator)
                samples[local_idx] = cand_indices[pos].gather(0, draw)
        return torch.cat(samples, dim=0)  # type: ignore[arg-type]

    def _minicpmo45_duplex_row_params(
        self,
        sampling_metadata: SamplingMetadata,
        num_rows: int,
    ) -> list[tuple[float, float, float]]:
        """Every row's (temperature, top_k, top_p) as ``_sampling_metadata_rows`` reads them.

        The runner's host copies (``DuplexSamplingRow``) hold the values the
        device tensors were copied from, so reading them skips a device read.
        A parameter the metadata carries no tensor for keeps its default, as in
        the device read; without host copies for every row that read is used.
        """
        host = getattr(self, "_minicpmo45_duplex_row_sampling_host", None) or {}
        columns = []
        for column, (name, default) in enumerate((("temperature", 0.7), ("top_k", 100), ("top_p", 0.8))):
            value = getattr(sampling_metadata, name, None)
            if (
                isinstance(value, torch.Tensor)
                and value.numel() == num_rows
                and all(row in host for row in range(num_rows))
            ):
                columns.append([float(host[row][column]) for row in range(num_rows)])
            else:
                columns.append(self._sampling_metadata_rows(sampling_metadata, name, num_rows, default))
        return list(zip(*columns, strict=True))

    def _minicpmo45_duplex_forbidden_index(self, token_ids: dict[str, int], vocab: int, device) -> torch.Tensor | None:
        """``_minicpmo45_native_forbidden_token_ids`` inside the vocab, as a cached device index."""
        key = (vocab, str(device))
        cache = getattr(self, "_minicpmo45_duplex_forbidden_index_cache", None)
        if cache is None or cache[0] != key:
            forbidden = self._minicpmo45_native_forbidden_token_ids(token_ids)
            valid = sorted({token_id for token_id in forbidden if 0 <= token_id < vocab})
            cache = (key, torch.tensor(valid, dtype=torch.long, device=device) if valid else None)
            self._minicpmo45_duplex_forbidden_index_cache = cache
        return cache[1]

    def _minicpmo45_duplex_cut_tables(self, token_ids: dict[str, int], vocab: int, device):
        """Per-token decoded length and "never cut" mask for ``_maybe_cut_minicpmo45_native_duplex_text_chunk``.

        Byte-level decoding is additive after a prefix that ends on a
        character boundary, so ``len(decode(chunk + [t]))`` is
        ``len(decode(chunk)) + length[t]``. Returns None (no device cut) when
        the tokenizer can not guarantee that: no batch decode, or the
        whitespace clean-up that rewrites across token boundaries.
        """
        key = (vocab, str(device))
        cache = getattr(self, "_minicpmo45_duplex_cut_tables_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        tables = None
        tokenizer = self._minicpmo45_tokenizer()
        batch_decode = getattr(tokenizer, "batch_decode", None)
        if callable(batch_decode) and not getattr(tokenizer, "clean_up_tokenization_spaces", False):
            try:
                num_tokens = min(vocab, len(tokenizer))
                texts = batch_decode([[token_id] for token_id in range(num_tokens)], skip_special_tokens=True)
            except Exception:
                texts = None
            if texts is not None and len(texts) == num_tokens:
                lengths = torch.zeros(vocab, dtype=torch.int32)
                lengths[:num_tokens] = torch.tensor([len(text) for text in texts], dtype=torch.int32)
                never_cut = torch.ones(vocab, dtype=torch.bool)
                never_cut[:num_tokens] = False
                special = [t for t in self._minicpmo45_native_special_token_ids(token_ids) if 0 <= t < vocab]
                if special:
                    never_cut[torch.tensor(special, dtype=torch.long)] = True
                tables = (lengths.to(device), never_cut.to(device))
        self._minicpmo45_duplex_cut_tables_cache = (key, tables)
        return tables

    def _minicpmo45_duplex_cut_bases(
        self,
        row_idxs: list[int],
        recent_tokens_by_row: dict[int, list[int]],
        token_ids: dict[str, int],
    ) -> list[int] | None:
        """Each row's decoded current-chunk length (-1: never cut), or None when a row needs the host decode.

        Mirrors the early returns of ``_maybe_cut_minicpmo45_native_duplex_text_chunk``.
        A chunk whose text ends in U+FFFD may end inside a character, where
        the next token's bytes would merge with it, so it is not additive.
        """
        max_chars = self._minicpmo45_policy_limit(
            "max_speak_chars_per_chunk",
            MiniCPMO45DuplexPolicy.DEFAULT_MAX_SPEAK_CHARS_PER_CHUNK,
        )
        tokenizer = self._minicpmo45_tokenizer()
        decode = getattr(tokenizer, "decode", None)
        if max_chars <= 0 or not callable(decode):
            return [-1] * len(row_idxs)
        bases = []
        for row_idx in row_idxs:
            chunk = self._minicpmo45_current_chunk_tokens(recent_tokens_by_row[row_idx], token_ids)
            try:
                text = decode(chunk, skip_special_tokens=True)
            except Exception:
                return None
            if not isinstance(text, str) or text.endswith("\ufffd"):
                return None
            bases.append(len(text))
        return bases

    def _sample_minicpmo45_native_duplex_rows_deferred(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        *,
        row_idxs: list[int],
        token_ids: dict[str, int],
        row_params: list[tuple[float, float, float]],
    ) -> torch.Tensor | None:
        """``_sample_minicpmo45_native_duplex_rows`` decided on the device, without a host sync.

        Returns each row's token as a device tensor, or None when this step
        needs the synchronous path. The cap, boundary decision, stage-2 draw,
        chunk cut and listen->tts_bos rewrite are combined with ``torch.where``.
        The host state they update is committed by
        ``_commit_minicpmo45_duplex_pending_samples`` one step later, before
        anything reads it.

        Stage 2 is drawn for every uncapped row, because the boundary outcome
        is only known on the device; a row whose boundary hits has its
        generator rewound at commit, so every generator advances exactly as on
        the synchronous path. Rows on the process-wide generator can not be
        rewound one at a time and keep the step synchronous.
        """
        device = logits.device
        vocab = logits.shape[-1]
        chunk_eos_id = token_ids.get("chunk_eos_token_id", -1)
        has_chunk_eos = 0 <= chunk_eos_id < vocab
        all_greedy = bool(getattr(sampling_metadata, "all_greedy", False))
        generators = getattr(sampling_metadata, "generators", {}) or {}
        recent_tokens_by_row = self._minicpmo45_duplex_recent_tokens(sampling_metadata, row_idxs)
        capped = self._minicpmo45_duplex_capped_rows(row_idxs, recent_tokens_by_row) if has_chunk_eos else set()
        stage2_rows = [row_idx for row_idx in row_idxs if row_idx not in capped]
        if any(
            generators.get(row_idx) is None
            for row_idx in stage2_rows
            if not (all_greedy or float(row_params[row_idx][0]) <= 0)
        ):
            return None
        cut = None
        if has_chunk_eos and stage2_rows:
            bases = self._minicpmo45_duplex_cut_bases(stage2_rows, recent_tokens_by_row, token_ids)
            if bases is None:
                return None
            if any(base >= 0 for base in bases):
                tables = self._minicpmo45_duplex_cut_tables(token_ids, vocab, device)
                if tables is None:
                    return None
                cut = (tables, bases)

        listen_id = token_ids.get("listen_token_id", -1)
        tts_bos_id = token_ids.get("tts_bos_token_id", -1)
        rewrite_listen = []
        for row_idx in stage2_rows:
            state = self._minicpmo45_duplex_state_for_row(row_idx)
            payload = self._minicpmo45_duplex_payload_for_row(row_idx)
            rewrite_listen.append(
                0 <= tts_bos_id
                and state is not None
                and not getattr(state, "current_turn_ended", True)
                and not (isinstance(payload, dict) and payload.get("force_listen") is True)
            )

        final = torch.full((len(row_idxs),), max(chunk_eos_id, 0), dtype=torch.long, device=device)
        raw = torch.full((len(stage2_rows),), -1, dtype=torch.long, device=device)
        hit = torch.zeros(len(stage2_rows), dtype=torch.bool, device=device)
        rewinds: dict[int, tuple[torch.Generator, Any]] = {}
        position = {row_idx: pos for pos, row_idx in enumerate(row_idxs)}
        if stage2_rows:
            stage2_index = index_to_device(stage2_rows, device)
            positions_t = index_to_device([position[row_idx] for row_idx in stage2_rows], device)
            if has_chunk_eos:
                hit = self._duplex_boundary_hits(
                    logits[stage2_index], stage2_rows, generators, all_greedy, chunk_eos_id
                )
            stage2_logits = self._minicpmo45_duplex_stage2_logits(
                logits, stage2_index, stage2_rows, token_ids, recent_tokens_by_row
            )
            raw = self._minicpmo45_duplex_stage2_draws(
                stage2_logits,
                stage2_rows,
                row_params,
                all_greedy,
                generators,
                marks=rewinds if has_chunk_eos else None,
            )

            token = raw
            if cut is not None:
                (lengths, never_cut), bases = cut
                max_chars = self._minicpmo45_policy_limit(
                    "max_speak_chars_per_chunk",
                    MiniCPMO45DuplexPolicy.DEFAULT_MAX_SPEAK_CHARS_PER_CHUNK,
                )
                base_t = index_to_device(bases, device)
                cut_t = (base_t >= 0) & ~never_cut[raw] & (base_t + lengths[raw].long() >= max_chars)
                token = torch.where(cut_t, torch.full_like(raw, chunk_eos_id), raw)
            if 0 <= listen_id and any(rewrite_listen):
                rewrite_t = index_to_device([int(flag) for flag in rewrite_listen], device).bool()
                token = torch.where((token == listen_id) & rewrite_t, torch.full_like(token, tts_bos_id), token)
            if has_chunk_eos:
                token = torch.where(hit, torch.full_like(token, chunk_eos_id), token)
            final[positions_t] = token

        # One pinned copy of (final, raw stage-2 sample, boundary hit) per row
        # for the commit; the host reads it once the runner has waited for it.
        packed = torch.stack([final, torch.full_like(final, -1), torch.zeros_like(final)], dim=1)
        if stage2_rows:
            packed[positions_t, 1] = raw
            packed[positions_t, 2] = hit.long()
        host = torch.empty(packed.shape, dtype=packed.dtype, pin_memory=device.type == "cuda")
        host.copy_(packed, non_blocking=True)
        event = None
        if device.type == "cuda":
            event = torch.cuda.Event()
            event.record()
        self._minicpmo45_duplex_pending_samples = _MiniCPMO45PendingSamples(
            host=host,
            event=event,
            row_idxs=list(row_idxs),
            stage2=[row_idx not in capped for row_idx in row_idxs],
            rewinds={position[stage2_rows[local_idx]]: rewind for local_idx, rewind in rewinds.items()},
            token_ids=token_ids,
            row_sessions=getattr(self, "_minicpmo45_duplex_row_sessions", None),
            row_payloads=getattr(self, "_minicpmo45_duplex_row_payloads", None),
        )
        return final

    def _commit_minicpmo45_duplex_pending_samples(self, *, session_ids: set[str] | None = None) -> None:
        """Apply the host state updates of the last deferred step, in the synchronous path's order.

        With ``session_ids``, only when that step sampled one of them: a
        caller building those sessions' next appends needs their latches, and
        must not wait for an unrelated step otherwise.
        """
        pending = getattr(self, "_minicpmo45_duplex_pending_samples", None)
        if pending is None:
            return
        if session_ids is not None:
            sessions = pending.row_sessions or {}
            if not any(sessions.get(row_idx) in session_ids for row_idx in pending.row_idxs):
                return
        self._minicpmo45_duplex_pending_samples = None
        if pending.event is not None:
            pending.event.synchronize()
        values = pending.host.tolist()
        saved = (
            getattr(self, "_minicpmo45_duplex_row_sessions", None),
            getattr(self, "_minicpmo45_duplex_row_payloads", None),
        )
        self._minicpmo45_duplex_row_sessions = pending.row_sessions
        self._minicpmo45_duplex_row_payloads = pending.row_payloads
        try:
            for pos, row_idx in enumerate(pending.row_idxs):
                _final, raw, hit = values[pos]
                if not pending.stage2[pos]:
                    continue
                if hit:
                    rewind = pending.rewinds.get(pos)
                    if rewind is not None:
                        _generator_rewind(*rewind)
                else:
                    self._record_minicpmo45_duplex_generation_token(row_idx, int(raw))
            for pos, row_idx in enumerate(pending.row_idxs):
                self._record_minicpmo45_duplex_terminator(row_idx, int(values[pos][0]), pending.token_ids)
        finally:
            self._minicpmo45_duplex_row_sessions, self._minicpmo45_duplex_row_payloads = saved

    def _sample_minicpmo45_native_duplex_row(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        *,
        row_idx: int,
        token_ids: dict[str, int],
        params: tuple[float, float, float] | None = None,
    ) -> int:
        """Sample one duplex row; ``params`` is the row's (temperature, top_k, top_p) if already read."""
        chunk_eos_id = token_ids.get("chunk_eos_token_id", -1)
        generator = getattr(sampling_metadata, "generators", {}).get(row_idx)
        output_token_ids = getattr(sampling_metadata, "output_token_ids", None) or []
        raw_recent_tokens = output_token_ids[row_idx] if row_idx < len(output_token_ids) else []
        recent_tokens = [int(token_id) for token_id in raw_recent_tokens if isinstance(token_id, int) and token_id >= 0]
        if params is None:
            params = (
                self._sampling_metadata_value(sampling_metadata, "temperature", row_idx, 0.7),
                self._sampling_metadata_value(sampling_metadata, "top_k", row_idx, 100),
                self._sampling_metadata_value(sampling_metadata, "top_p", row_idx, 0.8),
            )
        temperature, top_k, top_p = float(params[0]), int(params[1]), float(params[2])
        state = self._minicpmo45_duplex_state_for_row(row_idx)
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
            all_greedy = bool(getattr(sampling_metadata, "all_greedy", False))
            if self._duplex_boundary_hits(logits, [row_idx], {row_idx: generator}, all_greedy, chunk_eos_id).item():
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
            penalized = [t for t in set(repetition_tokens[-history_size:]) if 0 <= t < logits.shape[-1]]
            if penalized:
                # Distinct ids, so one indexed division equals dividing each once.
                columns = index_to_device(penalized, logits.device)
                logits[0, columns] = logits[0, columns] / repetition_penalty

        if getattr(sampling_metadata, "all_greedy", False) or temperature <= 0:
            sampled = int(torch.argmax(logits, dim=-1).item())
            self._record_minicpmo45_duplex_generation_token(row_idx, sampled)
            sampled = self._maybe_cut_minicpmo45_native_duplex_text_chunk(
                sampled,
                recent_tokens,
                token_ids,
            )
            return self._finalize_minicpmo45_native_duplex_sample(row_idx, sampled, token_ids)

        cand_probs, cand_indices = self._duplex_top_k_top_p_candidates(logits / temperature, top_k=top_k, top_p=top_p)
        draw = torch.multinomial(cand_probs[0], num_samples=1, generator=generator)
        sampled = int(cand_indices[0].gather(0, draw).item())
        self._record_minicpmo45_duplex_generation_token(row_idx, sampled)
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
        past decisions. Text clears the turn-ended latch; <|turn_eos|> sets it
        without becoming pending, because it was forwarded in this unit."""
        state = self._minicpmo45_duplex_state_for_row(row_idx)
        if state is None:
            return
        payload = self._minicpmo45_duplex_payload_for_row(row_idx)
        force_listen = isinstance(payload, dict) and payload.get("force_listen") is True
        listen_id = token_ids.get("listen_token_id", -1)
        tts_bos_id = token_ids.get("tts_bos_token_id", -1)
        turn_eos_id = token_ids.get("turn_eos_token_id", -1)
        if sampled in self._minicpmo45_chunk_terminator_token_ids(token_ids):
            state.pending_terminator_token = int(sampled)
            state.last_terminator_token = int(sampled)
            if sampled == listen_id and force_listen:
                state.current_turn_ended = True
                with suppress(Exception):
                    state.pending_speech_response_open = False
            return
        if sampled == turn_eos_id:
            # Official streaming_generate feeds <|turn_eos|> like text (its
            # hidden state conditions the Talker) and keeps sampling until a
            # chunk terminator, so nothing is pending for the next append;
            # only the turn-ended latch flips.
            state.pending_terminator_token = None
            state.last_terminator_token = int(sampled)
            state.current_turn_ended = True
            with suppress(Exception):
                state.pending_speech_response_open = False
            return
        # A seeded prefix can open the response before tts_bos is sampled.
        # Keep its pending input until the first content token in either case.
        if (
            sampled == tts_bos_id
            and (getattr(state, "current_turn_ended", True) or getattr(state, "pending_speech_response_open", False))
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

    @staticmethod
    def _minicpmo45_chunk_terminator_token_ids(token_ids: dict[str, int]) -> set[int]:
        """Official ``chunk_terminator_token_ids``: the tokens that close a unit.

        <|turn_eos|> is deliberately absent. It ends the turn but not the
        unit: the model forwards it and keeps sampling until one of these.
        """
        return {
            int(token_id)
            for token_id in (
                token_ids.get("listen_token_id", -1),
                token_ids.get("chunk_eos_token_id", -1),
                token_ids.get("chunk_tts_eos_token_id", -1),
            )
            if token_id is not None and int(token_id) >= 0
        }

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

    @classmethod
    def _sampling_metadata_rows(
        cls,
        sampling_metadata: SamplingMetadata,
        name: str,
        num_rows: int,
        default: float,
    ) -> list[float]:
        """``_sampling_metadata_value`` for rows ``0..num_rows-1`` with a single device read."""
        value = getattr(sampling_metadata, name, None)
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return [default] * num_rows
            flat = [float(v) for v in value.reshape(-1).tolist()]
            return [flat[min(row, len(flat) - 1)] for row in range(num_rows)]
        return [cls._sampling_metadata_value(sampling_metadata, name, 0, default)] * num_rows

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

    @staticmethod
    def _duplex_top_k_top_p_candidates(
        logits: torch.Tensor, *, top_k: int, top_p: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``softmax(_top_k_top_p_filter(logits, ...))`` restricted to its support, as ``(probs, indices)``.

        The nucleus is decided on the k sorted candidates, O(V + k log k)
        instead of sorting the vocabulary. The only difference from the filter
        is an exact tie at the k-th logit: exactly k candidates are kept here.
        """
        vocab = logits.shape[-1]
        keep = vocab if not top_k or top_k <= 0 else min(int(top_k), vocab)
        values, indices = torch.topk(logits, keep, dim=-1)
        probs = torch.softmax(values, dim=-1)
        if 0.0 < float(top_p) < 1.0:
            cumulative = probs.cumsum(dim=-1)
            remove = cumulative > float(top_p)
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            probs = probs.masked_fill(remove, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs, indices

    def _duplex_stage2_candidate_probs(
        self,
        rows_logits: torch.Tensor,
        row_params: list[tuple[float, float, float]],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Per-row ``(candidate probs, candidate ids)``; rows sharing ``(top_k, top_p)`` share one top-k."""
        groups: dict[tuple[int, float], list[int]] = {}
        for pos, (_temperature, top_k, top_p) in enumerate(row_params):
            groups.setdefault((int(top_k), float(top_p)), []).append(pos)
        temps = to_device_nonblocking(
            torch.tensor([float(p[0]) for p in row_params], dtype=rows_logits.dtype),
            rows_logits.device,
        ).view(-1, 1)
        probs: list[torch.Tensor] = [torch.empty(0)] * len(row_params)
        indices: list[torch.Tensor] = [torch.empty(0, dtype=torch.long)] * len(row_params)
        for (top_k, top_p), members in groups.items():
            members_t = index_to_device(members, rows_logits.device)
            rows = rows_logits[members_t] / temps[members_t]
            cand_probs, cand_indices = self._duplex_top_k_top_p_candidates(rows, top_k=top_k, top_p=top_p)
            for row, pos in enumerate(members):
                probs[pos] = cand_probs[row]
                indices[pos] = cand_indices[row]
        return probs, indices

    @staticmethod
    def _duplex_boundary_chunk_eos_probs(logits: torch.Tensor, chunk_eos_id: int) -> torch.Tensor:
        """Per-row ``softmax(logits)[chunk_eos_id] = exp(l_eos - logsumexp(l))``."""
        eos_logits = logits[:, chunk_eos_id]
        return torch.exp(eos_logits - torch.logsumexp(logits, dim=-1))

    @classmethod
    def _duplex_boundary_hits(
        cls,
        rows_logits: torch.Tensor,
        row_idxs: list[int],
        generators: dict[int, torch.Generator | None],
        all_greedy: bool,
        chunk_eos_id: int,
    ) -> torch.Tensor:
        """Whether each row's boundary sample is chunk_eos, drawn in row order with the row's own generator.

        The boundary sample is discarded unless it is chunk_eos, so one
        Bernoulli at P(sample == chunk_eos) is the same decision as a
        full-vocabulary softmax + multinomial.
        """
        if all_greedy:
            return torch.argmax(rows_logits, dim=-1) == chunk_eos_id
        boundary_p = cls._duplex_boundary_chunk_eos_probs(rows_logits, chunk_eos_id)
        draws = [torch.rand((), generator=generators.get(row_idx), device=rows_logits.device) for row_idx in row_idxs]
        return torch.stack(draws) < boundary_p

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
