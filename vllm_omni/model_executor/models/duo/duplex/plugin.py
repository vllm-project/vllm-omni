# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Frame-v5 Realization through official duplex sessions and resident stage engines."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import io
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pybase64
import torch
from PIL import Image
from vllm.sampling_params import RequestOutputKind

from vllm_omni.engine.duplex.config import DuplexCapabilities
from vllm_omni.engine.duplex.contracts import DuplexAppendPlan
from vllm_omni.engine.duplex.plugin import DuplexModelPlugin, DuplexRuntimeConfigError, PartialStageForward
from vllm_omni.model_executor.models.duo.frame_v5 import NativeFrameV5Serializer
from vllm_omni.model_executor.models.duo.h50_connector import H50Connector
from vllm_omni.model_executor.models.duo.token_mask import assistant_bigram_pairs, forbidden_token_ids
from vllm_omni.model_executor.stage_input_processors.duo import multimodal_payload

from .data_plane import DuoDataPlane
from .scheduling import DuoPhaseSchedule
from .session import DuoChunkState, DuoInputBuffer, DuoSessionState
from .trace import trace


def _new_asr_text(previous: str, current: str) -> str:
    """Trim the recognized overlap while retaining punctuation inside new text."""
    old = [char.casefold() for char in previous if char.isalnum()]
    offsets = [index for index, char in enumerate(current) if char.isalnum()]
    new = [current[index].casefold() for index in offsets]
    for length in range(min(len(old), len(new)), 0, -1):
        if old[-length:] == new[:length]:
            return current[offsets[length] :] if length < len(offsets) else ""
    return current


class DuoDuplexPlugin(DuplexModelPlugin):
    plugin_id = "duo"
    pipelined_input_limit = 3
    private_runtime_config_keys = frozenset({"reference", "adapter_path", "model", "assistant_description"})

    def __init__(self, encode_audio):
        self.data_plane = DuoDataPlane()
        self.processor = self.tokenizer = self.reference = None
        path = os.environ.get("VLLM_OMNI_DUO_SESSION_CONFIG")
        self.options = json.loads(Path(path).read_text()) if path else {}
        self.data_plane.defer_clean = self.options.get("realization_defer_clean_until_codec", False)
        self.intent_stage_id, self.realization_stage_id, self.codec_stage_id = 0, 1, 2
        self.asr_stage_id = None
        self.duo_legacy = self.options.get("conditioning_profile") == "duo_legacy_training_prefix"
        self._processor_lock = asyncio.Lock()
        self._camera_prepare_slots = asyncio.Semaphore(3)
        self._loop_trace_task = None

    def configure_pipeline(self, pipeline_config):
        stages = {s.model_stage: s.stage_id for s in pipeline_config.stages}
        self.intent_stage_id, self.realization_stage_id, self.codec_stage_id = (
            stages[k] for k in ("intent", "realization", "codec")
        )
        self.asr_stage_id = stages.get("asr")
        self.data_plane.realization_stage_id, self.data_plane.codec_stage_id = (
            self.realization_stage_id,
            self.codec_stage_id,
        )

    def input_stage_id(self, payload):
        if isinstance(payload, Mapping) and payload.get("audio"):
            if self.asr_stage_id is None:
                raise DuplexRuntimeConfigError("Speech input requires the duo_asr profile")
            return self.asr_stage_id
        return self.intent_stage_id

    def stage_is_resumable(self, stage_id, capabilities):
        return stage_id == self.intent_stage_id

    def model_state_session_key(self, session_id, epoch):
        return f"{session_id}/epoch/{epoch}"

    def model_state_stage_ids(self):
        return frozenset((self.realization_stage_id, self.codec_stage_id))

    def split_transcription_input(self, payload):
        if not isinstance(payload, Mapping) or not payload.get("audio"):
            return payload, None
        self.input_stage_id(payload)
        first = {k: v for k, v in payload.items() if k != "video_frames"}
        remaining = {
            k: v
            for k, v in payload.items()
            if k not in ("audio", "format", "sample_rate_hz", "committed_audio_segments")
        }
        return first, remaining

    def start_transcription_preparation(self, payload):
        frames = tuple((payload.get("video_frames") or [])[-self._frames_per_block() :])
        if not frames:
            return None
        block_frames = self._frames_per_block()
        target_size = self.options.get("camera_frame_size", (320, 384))
        target_size = tuple(target_size) if target_size is not None else None

        async def prepare():
            await self._camera_prepare_slots.acquire()
            try:
                worker = asyncio.create_task(
                    asyncio.to_thread(self._prepare_camera_frames, frames, block_frames, None, target_size)
                )
            except BaseException:
                self._camera_prepare_slots.release()
                raise

            def finished(task):
                self._camera_prepare_slots.release()
                if not task.cancelled():
                    task.exception()

            worker.add_done_callback(finished)
            return await asyncio.shield(worker)

        return asyncio.create_task(prepare())

    def create_session_state(self):
        return DuoSessionState(audio_buffer=DuoInputBuffer())

    def capabilities(self, *, max_sessions):
        if self.options.get("realization_scheduling", "free") in ("alternating", "serial") and max_sessions != 1:
            raise DuplexRuntimeConfigError("Phased Realization scheduling requires a single-session replica")
        return DuplexCapabilities(
            supports_input_append=True,
            supports_replace_latest_chunk=False,
            supports_reencode_context=False,
            supports_turn_commit_only=True,
            supports_core_resumable_request=True,
            supports_stage_resumption=True,
            supports_scheduler_native_append=True,
            supports_model_internal_state=True,
            required_input_modalities=frozenset(),
            optional_input_modalities=frozenset({"video", "audio"} if self.asr_stage_id is not None else {"video"}),
            supports_independent_io_streams=True,
            supports_realtime_endpoint=True,
            supports_multi_session=max_sessions > 1,
            supports_multi_session_same_replica=max_sessions > 1,
            supports_session_lease=True,
            supports_session_resume=True,
            session_admission_mode="engine_managed",
            supports_output_video=True,
            supports_text_only_turn=False,
            adapter_patterns=["frame_v5_append"],
            signal_sources=["client_event", "server_policy"],
            stage_handoff_transport="inline_tensor_payload",
            chunk_period_ms=round(self._frames_per_block() * 1000 / 24),
            target_barge_in_latency_ms=None,
        )

    async def prepare_runtime_config(self, config, *, model_config):
        self.validate_client_extra_body(config.extra_body)
        async with self._processor_lock:
            if self.processor is None:
                trace_path = os.environ.get("DUO_TRACE_PATH")
                if trace_path and self._loop_trace_task is None:
                    from .trace import trace_loop_delays

                    self._loop_trace_task = asyncio.create_task(trace_loop_delays(trace_path))
                path = os.environ.get("VLLM_OMNI_DUO_SESSION_CONFIG")
                if not path:
                    raise DuplexRuntimeConfigError("Server must set VLLM_OMNI_DUO_SESSION_CONFIG")
                self.options = json.loads(Path(path).read_text())
                self._realization_inference_steps()
                penalty = self.options.get("intent_output_repeat_penalty", 0.0)
                if not 0 <= penalty <= 10:
                    raise ValueError("intent_output_repeat_penalty must be between 0 and 10")
                from transformers import AutoProcessor, AutoTokenizer

                self.tokenizer = await asyncio.to_thread(AutoTokenizer.from_pretrained, self.options["tokenizer"])
                if self.options.get("intent_protocol_manifest"):
                    from ..frame_v5 import bind_intent_protocol

                    await asyncio.to_thread(
                        bind_intent_protocol,
                        self.tokenizer,
                        self.options["tokenizer"],
                        self.options["intent_protocol_manifest"],
                    )
                self.processor = await asyncio.to_thread(AutoProcessor.from_pretrained, self.options["processor"])
                self.reference = torch.load(self.options["reference_latent"], map_location="cpu", weights_only=True)
                if (
                    not isinstance(self.reference, torch.Tensor)
                    or self.reference.ndim != 4
                    or self.reference.shape[:2] != (24, 1)
                    or not torch.isfinite(self.reference).all()
                ):
                    raise DuplexRuntimeConfigError("Server reference latent has invalid geometry")
                if model_config is None:
                    raise DuplexRuntimeConfigError("Realization requires its Intent model configuration")
                self.max_rows = model_config.max_model_len
                self.eos = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
                period_ids = self.tokenizer.encode("。", add_special_tokens=False)
                self._trailing_period_id = period_ids[0] if len(period_ids) == 1 else None
                self._repeat_special_ids = set(self.tokenizer.all_special_ids)
                self._repeat_special_ids.update(
                    self.tokenizer.convert_tokens_to_ids(token) for token in ("<wait>", "<speak>", "<|im_end|>")
                )
                self._repeat_punctuation_ids = {
                    tokens[0]
                    for mark in ("。", "，", "？", "！", "、", ".", ",", "?", "!", "\n")
                    if len(tokens := self.tokenizer.encode(mark, add_special_tokens=False)) == 1
                }
                self._period_bad_words = [
                    text
                    for text in self.tokenizer.batch_decode(
                        [[token] for token in range(len(self.tokenizer))], clean_up_tokenization_spaces=False
                    )
                    if "。" in text
                ]
                self.forbidden = forbidden_token_ids(self.tokenizer)
                native_mask = getattr(model_config.hf_config, "duo_forbidden_token_ids", None)
                if native_mask is not None and native_mask != self.forbidden:
                    raise DuplexRuntimeConfigError("Native Intent hard mask differs from the session tokenizer")
                self.native_token_mask = native_mask is not None
        return self.runtime_config_for_update(config, {})

    def runtime_config_for_update(self, config, current):
        self.validate_client_extra_body(config.extra_body)
        if (
            self.options.get("realization_defer_clean_until_codec", False)
            and (config.extra_body or {}).get("auto_response") is not True
        ):
            raise DuplexRuntimeConfigError("Deferred clean requires the continuous auto_response lifecycle")
        from vllm_omni.protocol.realtime.formats import realtime_output_format

        if realtime_output_format(config.response_format) != "pcm16" or config.speed not in (None, 1, 1.0):
            raise DuplexRuntimeConfigError("Realization synchronized AV requires pcm16 at native speed")
        audio = (config.extra_body or {}).get("realtime_audio", {})
        if audio:
            from vllm_omni.protocol.realtime.formats import parse_realtime_audio_format

            output = audio.get("output", {})
            fmt, rate = parse_realtime_audio_format(output.get("format", config.response_format))
            rate = rate or output.get("sample_rate_hz") or output.get("sample_rate")
            if fmt != "pcm16" or rate != 32000:
                raise DuplexRuntimeConfigError("Realization AV requires audio.output.format audio/pcm at rate 32000")
        realization_conditioning = (config.extra_body or {}).get(
            "duo_realization_conditioning", self.options.get("realization_conditioning", "training_prefix")
        )
        intent_backend = (config.extra_body or {}).get("duo_intent_backend", self.options.get("intent_backend", "duo"))
        if realization_conditioning != "native_ntp" or intent_backend != "vllm":
            raise DuplexRuntimeConfigError("Realization requires resident vLLM Intent with native_ntp conditioning")
        generation = (config.extra_body or {}).get(
            "realization_media_reset_generation", current.get("media_reset_generation", 0)
        )
        if type(generation) is not int or generation < 0:
            raise DuplexRuntimeConfigError("Media reset generation must be a nonnegative integer")
        seed = (config.extra_body or {}).get("realization_seed", current.get("generation_seed"))
        if seed is not None and (type(seed) is not int or not 0 <= seed <= 2**32 - 1):
            raise DuplexRuntimeConfigError("Realization seed must be an integer between 0 and 4294967295")
        return {
            "generation_seed": seed,
            "media_reset_generation": generation,
            "wait_until_block": (config.extra_body or {}).get(
                "realization_wait_until_block", current.get("wait_until_block", 0)
            ),
            "defer_clean": self.options.get("realization_defer_clean_until_codec", False)
            and (config.extra_body or {}).get("realization_defer_clean_until_codec", True),
            "native_intent": True,
            "realization_conditioning": realization_conditioning,
            "persona_id": (config.extra_body or {}).get("realization_persona_id"),
            "suppress_period": self.options.get("intent_suppress_period", False)
            or (config.extra_body or {}).get("realization_suppress_period", False),
            "prewarm_text_graphs": (config.extra_body or {}).get("realization_prewarm_text_graphs", False),
            "initial_user_text": config.initial_user_text,
            "temperature": 0.0,
            "max_tokens": min(config.max_tokens or 128, 128),
        }

    def _frames_per_block(self):
        mode = self.options.get("codec_mode", "stream_4frame")
        if mode not in ("stream_4frame", "old", "duo_5_17"):
            raise DuplexRuntimeConfigError("Unknown server Realization codec mode")
        return 17 if mode == "duo_5_17" else 4

    def _realization_inference_steps(self):
        steps = getattr(self, "options", {}).get("realization_num_inference_steps", 4)
        if type(steps) is not int or steps < 1:
            raise DuplexRuntimeConfigError("Server realization_num_inference_steps must be a positive integer")
        return steps

    def configure_sampling_params(self, *, runtime_config, defaults):
        params = []
        for stage_id, value in enumerate(defaults):
            value = value.clone()
            if stage_id == self.intent_stage_id:
                value.temperature = 0.0
                value.max_tokens = runtime_config.get("max_tokens", 128)
                value.detokenize = False
                if runtime_config.get("suppress_period"):
                    from vllm.tokenizers.hf import get_cached_tokenizer

                    value.bad_words = self._period_bad_words
                    value.update_from_tokenizer(get_cached_tokenizer(self.tokenizer))
                if self.tokenizer is not None:
                    value.stop_token_ids = [self.eos]
                    # The fixed model owns the identical -inf mask when declared
                    # in hf_config. Avoid cloning/serializing a 151k-ID allowlist
                    # on every resumable append. Keep fallback deployments exact.
                    value.allowed_token_ids = (
                        None
                        if getattr(self, "native_token_mask", False)
                        else sorted(set(range(len(self.tokenizer))) - set(self.forbidden))
                    )
                value.output_kind = RequestOutputKind.CUMULATIVE
                if self.duo_legacy:
                    value.allowed_token_ids = None
            if stage_id == self.realization_stage_id:
                value.num_inference_steps = self._realization_inference_steps()
                value.extra_args = {
                    **(value.extra_args or {}),
                    "realization_early_codec_dispatch": True,
                    "realization_defer_clean_until_codec": runtime_config.get("defer_clean", False),
                    "realization_prewarm_text_graphs": runtime_config.get("prewarm_text_graphs", False),
                }
            if stage_id == self.asr_stage_id:
                value.detokenize = True
                value.output_kind = RequestOutputKind.FINAL_ONLY
            params.append(value)
        return tuple(params)

    def prepare_prompt_config(self, config, *, state, payload):
        return {**config, "realization_session_state": state}

    def _initialize_epoch(self, state, request_id, fence, runtime_config):
        state.close()
        state.consumed_input_items = set()
        state.fence = fence.epoch
        state.session_id = self.model_state_session_key(fence.session_id, fence.epoch)
        state.request_id = request_id
        state.next_block = state.next_publish = state.sampled_rows = state.history_cursor = 0
        state.asr_next_block = 0
        state.first_utterance_seen = state.first_utterance_complete = False
        state.waiting_for_user = self.options.get("wait_for_user_speech", False)
        state.prefilled_wait = False
        state.media_reset_generation = runtime_config.get("media_reset_generation", 0)
        state.reset_media_cache = False
        state.generation_seed = runtime_config.get("generation_seed")
        state.connector = (
            H50Connector(request_id, fence.epoch, max_rows=self.max_rows) if request_id is not None else None
        )
        state.reference = self.reference
        state.description = self.options["assistant_description"]
        system_message = self.options.get("intent_system_message")
        persona_id = runtime_config.get("persona_id")
        if persona_id is not None:
            from uuid import UUID

            folder = Path(self.options["persona_directory"]) / UUID(persona_id).hex
            state.description = json.loads((folder / "description.json").read_text())
            state.reference = torch.load(folder / "reference.pt", map_location="cpu", weights_only=True)
            message_file = folder / "system_message.txt"
            if message_file.is_file():
                system_message = message_file.read_text()
            trace("persona_applied", state, reference_shape=list(state.reference.shape))
        state.native_turns.clear()
        state.native_rope_shift = state.native_evicted = 0
        state.native_intent = runtime_config.get("native_intent", False)
        if state.native_intent and runtime_config.get("realization_conditioning") != "native_ntp":
            raise DuplexRuntimeConfigError("Native Intent requires native_ntp conditioning")
        state.realization_conditioning = runtime_config.get("realization_conditioning", "training_prefix")
        state.serializer = NativeFrameV5Serializer(
            self.tokenizer,
            self.processor,
            assistant_description=state.description,
            system_prompt=self.options.get("system_prompt", ""),
            system_message=system_message,
            total_frames=self.max_rows * self._frames_per_block(),
            duo_legacy=self.duo_legacy,
            grid_only=self.options.get("frame_v5_grid_only", True),
        )
        state.audio_buffer.max_frames = self._frames_per_block()
        state.intent_ready = asyncio.Event()
        state.intent_ready.set()
        # Admit the sole successor on latent-ready for joint clean execution.
        state.realization_credit = asyncio.Semaphore(1)
        state.capacity = asyncio.Semaphore(3)
        state.schedule = DuoPhaseSchedule(
            state,
            self.options.get("realization_scheduling", "free"),
            defer_clean=runtime_config.get("defer_clean", False),
        )

    async def prepare_append_plan(self, **kwargs):
        state = kwargs["session_config"]["realization_session_state"]
        if kwargs["payload"].get("audio"):
            fence = kwargs["fence"]
            if state.fence != fence.epoch:
                # ASR owns a temporary input unit, not Intent's computed-row ledger.
                self._initialize_epoch(state, None, fence, kwargs["runtime_config"])
            block = state.asr_next_block
            state.asr_next_block += 1
            ready_ns = time.perf_counter_ns()
            trace("asr_admission_requested", state, block=block)
            mode = self.options.get("asr_scheduling", "overlap")
            await state.schedule.admit(f"asr_{mode}", block)
            plan = self.plan_append(**kwargs)
            plan.prompt["additional_information"]["realization_asr_timing"].update(
                block=block,
                input_ready_ns=ready_ns,
                admission_wait_ms=(time.perf_counter_ns() - ready_ns) / 1e6,
                scheduling=mode,
            )
            self.data_plane.bind_state(
                kwargs["request_id"],
                DuoChunkState(
                    kwargs["request_id"],
                    state.request_id or kwargs["request_id"],
                    state,
                    state.session_id,
                    state.fence,
                    block,
                    state.tenant,
                    "",
                ),
            )
            return plan
        fence = kwargs["fence"]
        if state.fence != fence.epoch:
            self._initialize_epoch(state, kwargs["request_id"], fence, kwargs["runtime_config"])
        elif state.request_id is None:
            state.request_id = kwargs["request_id"]
            state.connector = H50Connector(state.request_id, fence.epoch, max_rows=self.max_rows)
        trace("intent_admission_requested", state)
        ready, capacity, schedule = state.intent_ready, state.capacity, state.schedule
        await ready.wait()
        await capacity.acquire()
        try:
            if state.fence != fence.epoch or state.connector.closed:
                raise DuplexRuntimeConfigError("Pipeline epoch was cancelled before admission")
            ready.clear()
            await schedule.admit("cpu", state.next_block)
            prepared = time.perf_counter_ns()
            trace("intent_cpu_prepare_started", state)
            preparation = kwargs["session_config"].get("_duplex_input_preparation")
            if preparation is None:
                rgb = await asyncio.to_thread(
                    self._prepare_camera_frames,
                    kwargs["payload"].get("video_frames") or [],
                    self._frames_per_block(),
                    state.last_frames,
                    self.options.get("camera_frame_size", (320, 384)),
                )
            else:
                rgb = await preparation
            if state.fence != fence.epoch or state.connector.closed:
                raise DuplexRuntimeConfigError("Pipeline epoch was cancelled during CPU preparation")
            plan = self.plan_append(**kwargs, prepared_rgb=rgb)
            trace("intent_cpu_prepare_complete", state, prepare_ms=(time.perf_counter_ns() - prepared) / 1e6)
            await schedule.admit("intent", state.next_block)
            trace("intent_submitted", state)
            return plan
        except BaseException:
            capacity.release()
            ready.set()
            raise

    @staticmethod
    def _prepare_camera_frames(frames, block_frames, last_frames, target_size=(320, 384)):
        """Decode owned camera observations without mutating session state."""
        decoded = []
        for encoded in frames[-block_frames:]:
            if encoded.startswith("data:"):
                encoded = encoded.split(",", 1)[1]
            with Image.open(io.BytesIO(pybase64.b64decode(encoded, validate=True))) as image:
                image = image.convert("RGB")
                if target_size is not None:
                    image = image.resize(target_size)
                decoded.append(np.asarray(image).copy())
        if decoded:
            return np.stack([decoded[0]] * (block_frames - len(decoded)) + decoded)
        if last_frames is not None:
            return last_frames.copy()
        raise DuplexRuntimeConfigError("The first Realization observation requires camera frames")

    def plan_append(
        self,
        *,
        request_id,
        fence,
        session_config,
        runtime_config,
        seq,
        turn_seq,
        payload,
        final,
        sampling_params,
        prepared_rgb=None,
    ):
        if not final or not isinstance(payload, Mapping):
            raise DuplexRuntimeConfigError("Realization requires committed causal observations")
        self.data_plane.begin_request(request_id)
        if payload.get("audio"):
            if payload.get("format") != "pcm_f32le" or payload.get("sample_rate_hz") != 16000:
                raise DuplexRuntimeConfigError("Realization ASR requires normalized 16 kHz float32 PCM")
            audio = np.frombuffer(pybase64.b64decode(payload["audio"], validate=True), dtype="<f4")
            if not audio.size or not np.isfinite(audio).all():
                raise DuplexRuntimeConfigError("Invalid ASR PCM")
            state = session_config["realization_session_state"]
            item_id = payload.get("realtime_item_id")
            input_samples = audio.size
            overlap_item = None
            if self.options.get("asr_overlap_previous_chunk", False) and item_id:
                items = session_config.get("input_items") or ()
                previous = state.asr_previous_audio
                state.asr_previous_audio = (item_id, audio)
                if len(items) >= 2 and items[-1][0] == item_id and previous and previous[0] == items[-2][0]:
                    overlap_item, previous_audio = previous
                    audio = np.concatenate((previous_audio, audio))
            asr_timing = dict(
                submitted_ns=time.perf_counter_ns(),
                session_id=self.model_state_session_key(fence.session_id, fence.epoch),
                fence=fence.epoch,
                item_id=item_id,
                request_id=request_id,
                input_audio_ms=input_samples * 1000 / 16000,
                asr_window_audio_ms=audio.size * 1000 / 16000,
                input_seq=seq,
                turn_seq=turn_seq,
            )
            return DuplexAppendPlan(
                prompt={
                    "prompt": (
                        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
                        "Please transcribe this speech.<|im_end|>\n<|im_start|>assistant\n"
                    ),
                    "multi_modal_data": {"audio": (audio, 16000)},
                    "additional_information": {
                        "realtime_item_id": item_id,
                        "realization_asr_overlap_item": overlap_item,
                        "realization_asr_timing": asr_timing,
                    },
                }
            )
        state = session_config["realization_session_state"]
        if state.fence != fence.epoch:
            self._initialize_epoch(state, request_id, fence, runtime_config)
        if state.request_id is None:
            state.request_id = request_id
            state.connector = H50Connector(request_id, fence.epoch, max_rows=self.max_rows)
        if state.connector.closed:
            raise DuplexRuntimeConfigError("This Realization epoch was invalidated; cancel or reopen before appending")
        if state.busy:
            raise DuplexRuntimeConfigError("Previous Realization clean/codec block still owns session credit")
        block_frames = self._frames_per_block()
        rgb = prepared_rgb
        if rgb is None:
            rgb = self._prepare_camera_frames(
                payload.get("video_frames") or [],
                block_frames,
                state.last_frames,
                self.options.get("camera_frame_size", (320, 384)),
            )
        history = session_config.get("conversation", [])
        user_text = []
        if state.next_block == 0 and runtime_config.get("initial_user_text"):
            user_text.append(runtime_config["initial_user_text"])
        input_items = session_config.get("input_items")
        if input_items is not None:
            selected = [
                (item_id, message) for item_id, message in input_items if item_id not in state.consumed_input_items
            ]
            state.consumed_input_items.update(item_id for item_id, _ in selected)
            observations = [message for _, message in selected]
        else:
            observations = history[state.history_cursor :]
        for message in observations:
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                user_text.append(content)
            elif isinstance(content, list):
                user_text.extend(
                    p.get("text", p.get("transcript", ""))
                    for p in content
                    if p.get("type") in ("text", "input_text", "audio_url")
                )
        generation = runtime_config.get("media_reset_generation", 0)
        state.reset_media_cache = generation > state.media_reset_generation
        state.media_reset_generation = generation
        ids = self.tokenizer.encode("\n".join(user_text), add_special_tokens=False)
        has_text = any(text.strip() for text in user_text)
        if self.options.get("wait_until_first_utterance_complete", False) and not runtime_config.get(
            "wait_until_block"
        ):
            if not state.first_utterance_complete:
                # Energy VAD can stay active on background noise. After actual
                # recognized speech, a full blank ASR chunk also releases the gate.
                if has_text:
                    state.first_utterance_seen = True
                if state.first_utterance_seen and (not has_text or not payload.get("is_speech")):
                    state.first_utterance_complete = True
                    state.waiting_for_user = False
        elif has_text:
            state.waiting_for_user = False
        state.prefilled_wait = state.native_intent and (
            state.waiting_for_user or state.next_block < runtime_config.get("wait_until_block", 0)
        )
        if state.native_intent:
            sampling_params.allowed_token_ids = [self.eos] if state.prefilled_wait else None
            sampling_params.max_tokens = 1 if state.prefilled_wait else runtime_config.get("max_tokens", 128) + 1
        fragment = state.serializer.fragment(rgb, frame_start=state.next_block * block_frames, user_token_ids=ids)
        if state.prefilled_wait:
            wait_id = self.tokenizer.convert_tokens_to_ids("<wait>")
            fragment.prompt["prompt_token_ids"].append(wait_id)
            fragment.expected_token_ids.append(wait_id)
            trace("intent_opening_wait", state)
        if os.environ.get("DUO_SEGMENT_H50") == "1":
            state.connector.snapshot_rows = 0
        if state.native_intent:
            prefix = len(state.serializer.prefix) - len(state.serializer.separator)
            drop = 0
            while (
                len(state.ledger) - drop + len(fragment.expected_token_ids) + sampling_params.max_tokens > self.max_rows
                and state.native_turns
            ):
                drop += state.native_turns.pop(0)
                state.native_evicted += 1
            if drop:
                where = [prefix - 1, prefix + drop]
                pos = state.connector.select(where, [state.ledger[i] for i in where]).positions
                state.native_rope_shift = int(pos[1].max() - pos[0].max()) - 1
                del state.ledger[prefix : prefix + drop]
                state.connector.remove_window_rows(prefix, drop)
            if state.native_rope_shift:
                fragment.prompt["model_intermediate_buffer"] = {
                    "meta": {
                        "realization_prefix": prefix,
                        "realization_drop": drop,
                        "realization_rope_shift": state.native_rope_shift,
                    }
                }
        if len(state.ledger) + len(fragment.expected_token_ids) + sampling_params.max_tokens > self.max_rows:
            raise DuplexRuntimeConfigError("Realization Intent context exhausted; no silent history rebuild")
        if state.pending_tail_token is not None:
            meta = fragment.prompt.setdefault("model_intermediate_buffer", {}).setdefault("meta", {})
            meta["drop_computed_tail_token_id"] = state.pending_tail_token
            trace("intent_tail_prefill_correction", state, removed_token_id=state.pending_tail_token)
        if state.native_intent and not state.prefilled_wait:
            # One extra sampled EOS computes the final body token's KV/H50.
            sampling_params.max_tokens = runtime_config.get("max_tokens", 128) + 1
            sampling_params.extra_args = {
                **(sampling_params.extra_args or {}),
                "realization_action": {
                    "start": len(state.ledger) + len(fragment.expected_token_ids),
                    "limit": runtime_config.get("max_tokens", 128),
                    "wait": self.tokenizer.convert_tokens_to_ids("<wait>"),
                    "speak": self.tokenizer.convert_tokens_to_ids("<speak>"),
                    "eos": self.eos,
                },
            }
            if penalty := self.options.get("intent_output_repeat_penalty", 0.0):
                pairs = assistant_bigram_pairs(state.recent_assistant_tokens, self._repeat_punctuation_ids, set(ids))
                if pairs:
                    sampling_params.extra_args["realization_action"]["output_repeat"] = (penalty, pairs)
                    trace("intent_output_repeat_penalty", state, strength=penalty, bigrams=len(pairs))
        else:
            sampling_params.extra_args = {
                k: v for k, v in (sampling_params.extra_args or {}).items() if k != "realization_action"
            }
        state.fragment, state.expected = fragment, fragment.expected_token_ids
        prompt = fragment.prompt
        state.request_id = request_id
        state.busy, state.intent_done = True, False
        state.history_cursor, state.last_frames = len(history), rgb
        self.data_plane.bind_state(request_id, state)
        trace("intent_condition_prepared", state, user_text_token_count=len(ids))
        return DuplexAppendPlan(prompt=prompt)

    def commit_append_plan(self, request_id):
        state = self.data_plane.states.get(request_id)
        if state is not None:
            state.context_locked = True

    def decide_output(self, **kwargs):
        return None

    def record_stage_timing(self, *, stage_id, output, stage_metrics, prompt, finished):
        if stage_id == self.codec_stage_id and finished:
            chunk = self.data_plane.states.get(output.request_id)
            if chunk is not None:
                chunk.owner.schedule.complete("codec", chunk.next_block)
            return
        if stage_id == getattr(self, "intent_stage_id", None) and finished:
            state = self.data_plane.states.get(output.request_id)
            if state is not None:
                trace(
                    "intent_engine_complete",
                    state,
                    block=state.next_block - int(state.intent_done),
                    **{
                        name: getattr(stage_metrics, name, None)
                        for name in (
                            "vllm_queued_ms",
                            "vllm_prefill_ms",
                            "vllm_decode_ms",
                            "stage_gen_time_ms",
                        )
                    },
                    h50_computed_rows=state.connector.computed_rows,
                )
            return
        if stage_id != self.asr_stage_id or not finished or not isinstance(prompt, Mapping):
            return
        chunk = self.data_plane.states.get(output.request_id)
        if chunk is not None:
            chunk.owner.schedule.complete("asr", chunk.next_block)
        info = prompt.get("additional_information", {}).get("realization_asr_timing")
        path = os.environ.get("DUO_TRACE_PATH")
        if not info or not path:
            return
        now = time.perf_counter_ns()
        metrics = {
            name: getattr(stage_metrics, name, None)
            for name in (
                "vllm_queued_ms",
                "vllm_prefill_ms",
                "vllm_decode_ms",
                "stage_gen_time_ms",
                "rx_decode_time_ms",
                "rx_in_flight_time_ms",
            )
        }
        phases = [metrics[k] for k in ("vllm_queued_ms", "vllm_prefill_ms", "vllm_decode_ms")]
        service_ms = (now - info["submitted_ns"]) / 1e6
        event = dict(
            info,
            phase="asr_complete",
            monotonic_ns=now,
            observed_service_ms=service_ms,
            engine_inference_ms=None if any(x is None for x in phases[1:]) else sum(phases[1:]),
            outside_engine_ms=None if any(x is None for x in phases) else service_ms - sum(phases),
            **metrics,
        )
        with open(path, "a") as stream:
            stream.write(json.dumps(event) + "\n")

    def user_transcript(self, *, stage_id, output, prompt, finished):
        if stage_id != self.asr_stage_id or not finished or not getattr(output, "outputs", None):
            return None
        from vllm_omni.model_executor.stage_input_processors.joyai_vl_interaction import (
            _clean_qwen3_asr_transcript,
            _extract_completed_action_text,
        )

        # A fixed audio chunk is not an utterance boundary. Qwen3-ASR often
        # invents a final period on these short fragments; do not commit it.
        text = _clean_qwen3_asr_transcript(_extract_completed_action_text(output)).rstrip("。．.")
        if self.options.get("asr_overlap_previous_chunk", False):
            info = prompt["additional_information"]
            chunk = self.data_plane.states.get(output.request_id)
            if chunk is not None and chunk.fence == chunk.owner.fence:
                previous = chunk.owner.asr_previous_result
                chunk.owner.asr_previous_result = (info["realtime_item_id"], text)
                if previous and previous[0] == info["realization_asr_overlap_item"]:
                    text = _new_asr_text(previous[1], text)
        return text or None

    def project_intermediate_output(self, *, stage_id, output, context):
        return stage_id == self.realization_stage_id

    def plan_partial_stage_output(self, orchestrator, stage_id, replica_id, output, req_state):
        if stage_id not in (self.intent_stage_id, self.realization_stage_id):
            return None
        state = self.data_plane.states.get(output.request_id)
        if state is None or not state.busy or output.request_id != state.request_id:
            # Cancelled projection or an older epoch of a reused session state.
            return PartialStageForward(output, is_final_update=True, forward=False)
        if stage_id == self.intent_stage_id:
            try:
                mm = multimodal_payload(output)
            except ValueError:
                mm = None
            if mm and (not self.duo_legacy or state.native_intent):
                state.connector.append_snapshot(mm, request_id=state.request_id, fence=state.fence)
            segment = req_state.streaming.segment(stage_id)
            finished = bool(output.finished) or segment.finished
            if not finished:
                return PartialStageForward(output, is_final_update=False, forward=False)
            if state.intent_done:
                raise ValueError("Duplicate Intent segment completion")
            # CUMULATIVE includes prior segments; the raw segment boundary may
            # contain only the final sampled EOS. Never use it as the body.
            generated = list(output.outputs[0].token_ids)[state.sampled_rows :]
            if state.prefilled_wait:
                if generated != [self.eos]:
                    raise ValueError("Opening wait must end after the prefilled action")
                body_ids = [self.tokenizer.convert_tokens_to_ids("<wait>")]
                expected = state.ledger + state.expected
                start = len(expected) - 1
            else:
                if not generated or generated[-1] != self.eos or not generated[:-1]:
                    raise ValueError("Intent must produce a computed body followed by message end")
                body_ids = generated[:-1]
                start = len(state.ledger) + len(state.expected)
                expected = state.ledger + state.expected + body_ids
            if state.connector.computed_rows != len(expected):
                if os.environ.get("DUO_TRACE_PATH"):
                    trace(
                        "intent_h50_mismatch",
                        state,
                        computed_rows=state.connector.computed_rows,
                        expected_rows=len(expected),
                        ledger_rows=len(state.ledger),
                        expected_prompt_rows=len(state.expected),
                        generated=generated,
                        computed_ids=[t[1].tolist() for t in state.connector._chunks],
                        expected_ids=expected,
                        prompt_ids=list(getattr(output, "prompt_token_ids", ()) or ()),
                    )
                raise ValueError(f"Computed H50 ledger mismatch: {state.connector.computed_rows} vs {len(expected)}")
            state.connector.select(range(len(state.ledger), len(expected)), expected[len(state.ledger) :])
            fragment = state.fragment
            shared = state.connector.select_shared_once(
                fragment.shared_offsets or state.connector._shared.offsets,
                fragment.shared_token_ids or state.connector._shared.token_ids.tolist(),
            )
            state.pending_tail_token = None
            if (
                self.options.get("strip_trailing_period", False)
                and len(body_ids) > 1
                and body_ids[-1] == self._trailing_period_id
            ):
                # Sample the complete chunk normally, then commit its body
                # without the final Chinese period. H50 and next-prefill KV
                # must describe exactly the same corrected history.
                state.pending_tail_token = body_ids[-1]
                body_ids = body_ids[:-1]
                expected = expected[:-1]
                state.connector.drop_tail()
                trace("intent_trailing_period_removed", state, removed_token_id=state.pending_tail_token)
            if self.options.get("intent_output_repeat_penalty", 0.0):
                state.recent_assistant_tokens.append(
                    tuple(token for token in body_ids if token not in self._repeat_special_ids)
                )
            body = state.connector.select(range(start, len(expected)), body_ids)
            if state.serializer.bootstrap_offset is not None:
                # Reference v2 block 0 is the executed bootstrap <wait>. User
                # observation k conditions plan k+1; overlap its Intent execution.
                generated_hidden, generated_ids = body.hidden, body_ids
                if state.next_block == 0:
                    offset = state.serializer.bootstrap_offset
                    body = state.connector.select([offset], [self.tokenizer.convert_tokens_to_ids("<wait>")])
                    body_ids = [self.tokenizer.convert_tokens_to_ids("<wait>")]
                    body_hidden = body.hidden
                else:
                    body_hidden, body_ids = state.next_body_hidden, state.next_body_ids
                state.next_body_hidden, state.next_body_ids = generated_hidden, generated_ids
            else:
                body_hidden = body.hidden
            state.current_text = self.tokenizer.decode(body_ids, skip_special_tokens=False)
            if state.native_intent:
                pinned = (
                    len(state.serializer.prefix) - len(state.serializer.separator)
                    if not state.next_block
                    else len(state.ledger)
                )
                state.native_turns.append(len(expected) - pinned)
            state.ledger = expected
            state.sampled_rows += len(generated)
            from vllm_omni.diffusion.models.duo.assembly import assemble_condition
            from vllm_omni.diffusion.models.duo.condition import RealizationConditionTicket
            from vllm_omni.diffusion.models.duo.wire import condition_to_wire

            base_seed = state.generation_seed
            if base_seed is None:
                base_seed = self.options.get("seed", 42)
                seed_scope = self.options.get("noise_seed_session_id", state.session_id)
            else:
                seed_scope = "explicit"
            digest = hashlib.sha256(f"{base_seed}:{seed_scope}:{state.next_block}".encode())
            if self.duo_legacy:
                seed = int(digest.hexdigest()[:15], 16)
            else:
                seed = int.from_bytes(digest.digest()[:8], "big") % (2**63 - 1)
            final = self.options.get("finite_model_blocks") == state.next_block + 1
            condition = assemble_condition(
                RealizationConditionTicket(state.session_id, state.fence, state.next_block, state.tenant),
                reference=state.reference,
                shared=shared.hidden,
                body=body_hidden,
                shared_lengths=fragment.shared_lengths,
                seed=seed,
                final=final,
                codec_mode=self.options.get("codec_mode", "stream_4frame"),
                rope=self.options.get("realization_rope", "old"),
                reset_media_cache=state.reset_media_cache,
            )
            output.multimodal_output = {"realization_condition": condition_to_wire(condition)}
            state.intent_done = True
            trace(
                "intent_complete",
                state,
                body_token_ids=body_ids,
                noise_seed=seed,
                intent_backend="vllm",
                intent_input_tokens=len(state.ledger),
                evicted_chunks=state.native_evicted,
            )
            state.schedule.complete("intent", state.next_block)
            child_id = f"{state.request_id}-block{state.next_block}"
            child_state = DuoChunkState(
                child_id,
                state.request_id,
                state,
                state.session_id,
                state.fence,
                state.next_block,
                state.tenant,
                state.current_text,
                final=final,
                reset_media_cache=state.reset_media_cache,
            )
            state.inflight[state.next_block] = child_id
            self.data_plane.bind_state(child_id, child_state)
            child_request = orchestrator.fork_stage_request(state.request_id, child_id, self.realization_stage_id)
            child_output = copy.copy(output)
            child_output.request_id = child_id
            task = asyncio.create_task(
                self._dispatch_realization(orchestrator, child_request, child_output, child_state)
            )
            state.dispatch_tasks.add(task)
            task.add_done_callback(state.dispatch_tasks.discard)
            state.next_block += 1
            state.busy = False
            state.intent_ready.set()
            return PartialStageForward(output, is_final_update=True, forward=False)
        mm = multimodal_payload(output)
        meta = mm["metadata"]
        expected = (state.session_id, state.fence, state.next_block)
        if (meta.get("session_id"), meta.get("fence"), meta.get("block")) != expected or meta.get(
            "denoise_nfe"
        ) != self._realization_inference_steps():
            raise ValueError("Realization output identity/denoise budget mismatch")
        payload = mm.get("payload", mm)
        if meta.get("phase") == "latent_ready":
            if meta.get("defer_clean_until_codec", False) != state.owner.schedule.defer_clean:
                raise ValueError("Session and Realization deferred-clean settings differ")
            if not all(k in payload for k in ("video_latent", "audio_latent")):
                raise ValueError("Realization early output requires video/audio latents")
            if state.awaiting_clean or meta.get("commit_nfe") != 0:
                raise ValueError("Invalid/duplicate final Euler snapshot")
            if self.options.get("realization_joint_execution", False) and not state.clean_released:
                # Admission credit only. AV publication still requires clean ACK.
                state.clean_released = True
                state.owner.realization_credit.release()
            state.awaiting_clean = True
            mm["realization_codec_route"] = {
                "height": state.owner.reference.shape[2] * 16,
                "width": state.owner.reference.shape[3] * 16,
                "reference": state.owner.reference,
                "tenant": state.tenant,
                "dispatch_ns": time.perf_counter_ns(),
                "codec_mode": self.options.get("codec_mode", "stream_4frame"),
                "final": self.options.get("finite_model_blocks") == state.next_block + 1,
            }
            trace("latent_dispatched", state, realization_ready_ns=meta["realization_state_monotonic_ns"])
            if state.owner.schedule.defer_clean:
                state.owner.schedule.complete("latent", state.next_block)
                task = asyncio.create_task(self._resume_clean_after_codec(orchestrator, req_state, state))
                state.owner.dispatch_tasks.add(task)
                task.add_done_callback(state.owner.dispatch_tasks.discard)
            if state.owner.schedule.mode != "free":
                task = asyncio.create_task(self._dispatch_codec(orchestrator, req_state, output, state, replica_id))
                state.owner.dispatch_tasks.add(task)
                task.add_done_callback(state.owner.dispatch_tasks.discard)
                return PartialStageForward(output, is_final_update=True, forward=False)
            trace("codec_dispatched", state)
            return PartialStageForward(output, is_final_update=True)
        if (
            meta.get("phase") != "clean_committed"
            or not state.awaiting_clean
            or meta.get("clean_committed") is not True
            or meta.get("commit_nfe") != 1
        ):
            raise ValueError("Missing clean acknowledgement after final Euler")
        trace(
            "clean_result_received",
            state,
            clean_worker_finished_ns=meta["realization_state_monotonic_ns"],
            clean_handoff_ms=(time.perf_counter_ns() - meta["realization_state_monotonic_ns"]) / 1e6,
        )
        state.awaiting_clean = False
        state.owner.schedule.complete("clean", state.next_block)
        if not state.clean_released:
            state.clean_released = True
            state.owner.realization_credit.release()
        return PartialStageForward(output, is_final_update=True, forward=False)

    async def _dispatch_realization(self, orchestrator, request, output, chunk):
        try:
            if chunk.owner.connector.closed or orchestrator.request_states.get(request.request_id) is not request:
                return
            await chunk.owner.realization_credit.acquire()
            await chunk.owner.schedule.admit("realization", chunk.next_block)
            if chunk.owner.connector.closed or orchestrator.request_states.get(request.request_id) is not request:
                return
            if chunk.reset_media_cache:
                if chunk.owner.next_publish != chunk.next_block:
                    raise RuntimeError("Media reset must wait for preceding AV publication")
                trace("media_cache_reset_boundary", chunk, intent_rows=len(chunk.owner.ledger))
            trace("realization_dispatched", chunk)
            await orchestrator._forward_to_next_stage(
                request.request_id,
                self.intent_stage_id,
                output,
                request,
                is_streaming_session=True,
                is_final_update=True,
            )
            trace("stage_forward_complete", chunk, stage="realization")
            chunk.owner.schedule.complete("realization_sent", chunk.next_block)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await orchestrator._handle_forward_failure(request.request_id, self.realization_stage_id, request, exc)

    async def _resume_clean_after_codec(self, orchestrator, request, chunk):
        try:
            await chunk.owner.schedule.admit("clean", chunk.next_block)
            if chunk.owner.connector.closed or orchestrator.request_states.get(request.request_id) is not request:
                return
            trace("clean_resume_requested", chunk)
            await orchestrator.stage_pools[self.realization_stage_id].submit_interaction(
                request.request_id,
                {
                    "event_id": f"clean-{chunk.fence}-{chunk.next_block}",
                    "event": {"realization_resume_clean": [chunk.session_id, chunk.fence, chunk.next_block]},
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await orchestrator._handle_forward_failure(request.request_id, self.realization_stage_id, request, exc)

    async def _dispatch_codec(self, orchestrator, request, output, chunk, replica_id):
        try:
            await chunk.owner.schedule.admit("codec", chunk.next_block)
            if chunk.owner.connector.closed or orchestrator.request_states.get(request.request_id) is not request:
                return
            output.multimodal_output["realization_codec_route"]["dispatch_ns"] = time.perf_counter_ns()
            trace("codec_dispatched", chunk)
            await orchestrator._forward_to_next_stage(
                request.request_id,
                self.realization_stage_id,
                output,
                request,
                src_replica_id=replica_id,
                is_streaming_session=True,
                is_final_update=True,
            )
            trace("stage_forward_complete", chunk, stage="codec")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await orchestrator._handle_forward_failure(request.request_id, self.codec_stage_id, request, exc)
