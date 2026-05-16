# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extended RealtimeConnection with tool calling support for vLLM Omni."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import AsyncGenerator
from typing import cast
from uuid import uuid4

import numpy as np
from vllm.engine.protocol import StreamingInput
from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel, UsageInfo
from vllm.entrypoints.openai.realtime.connection import (
    RealtimeConnection as VllmRealtimeConnection,
)
from vllm.entrypoints.openai.realtime.protocol import (
    InputAudioBufferCommit,
    TranscriptionDelta,
    TranscriptionDone,
)
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.renderers.inputs.preprocess import parse_model_prompt
from vllm.tokenizers import cached_tokenizer_from_config

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.realtime_protocol import (
    RealtimeEventType,
    ResponseAudioDelta,
    ResponseAudioDone,
    ResponseFunctionCallArgumentsDelta,
    ResponseFunctionCallArgumentsDone,
)
from vllm_omni.entrypoints.utils import coerce_param_message_types

logger = init_logger(__name__)


class RealtimeConnection(VllmRealtimeConnection):
    """Extended RealtimeConnection with tool calling support for Qwen3 Omni.

    This class extends vLLM's base RealtimeConnection to add:
    - Tool configuration via session.update
    - XML tool call detection and streaming (Qwen3 format)
    - Tool result handling via conversation.item.create
    - Multi-turn conversation context management
    - Text streaming alongside audio
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = cast(AsyncOmni, self.serving.engine_client)
        # Model class supplies the chat-template / tool-call / special-token
        # surface (see ``Qwen3OmniMoeForConditionalGeneration``'s realtime
        # formatting helpers). Same plumbing as ``buffer_realtime_audio``.
        self.model_cls = self.serving.model_cls
        self._realtime_audio_ref: np.ndarray | None = None

        # Tool calling and conversation state
        self.tools: list[dict] | None = None
        self.instructions: str | None = None
        self.conversation_items: list[dict] = []
        self.conversation_context: str | None = None

        # Tool calls accumulated for the in-flight turn.
        self.current_tool_calls: list[dict] = []

        # Set when a tool result arrives while a generation pass is still
        # running; the pass's finally-block then dispatches the follow-up.
        self._pending_tool_context: bool = False

        # Cache user audio chunks for replay in the audio pass after tool results.
        # The talker needs audio-grounded hidden states from the thinker — a text-only
        # prompt produces garbled audio.
        self._cached_user_audio: list[np.ndarray] = []

        # Side-channel ASR: every audio commit fires a parallel transcription
        # request (text-only, stage 0) that batches with the chat thinker.
        # Result fills _current_user_item["content"] so the next turn's
        # history has a real transcript.
        self._pending_transcript_task: asyncio.Task | None = None
        self._current_user_item: dict | None = None
        # Set once audio_queue has been fully drained into _cached_user_audio.
        self._audio_cached_event: asyncio.Event = asyncio.Event()

        # Track engine.generate request_ids so cleanup() can abort any still
        # in flight — the engine is shared across connections.
        self._active_request_ids: set[str] = set()

        # First-turn audio kept as fallback acoustic reference for the talker's
        # hidden_projection bootstrap when the current turn's buffer is empty.
        self._turn_audio_cache: list[np.ndarray] | None = None

        # Tokenizer for decoding thinker text tokens
        self.tokenizer = None
        try:
            model_config = self.serving.model_config
            self.tokenizer = cached_tokenizer_from_config(model_config)
            logger.debug("Tokenizer loaded for tool call parsing")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}")

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    async def cleanup(self):
        """Cancel side-channel tasks and abort any in-flight engine requests."""
        self._is_connected = False
        if self._pending_transcript_task and not self._pending_transcript_task.done():
            self._pending_transcript_task.cancel()
        if self.generation_task and not self.generation_task.done():
            self.generation_task.cancel()

        for req_id in list(self._active_request_ids):
            try:
                await self.engine.abort(req_id)
            except Exception:
                pass
        self._active_request_ids.clear()

        # Unblock anything awaiting the audio-cache event so cancellations
        # propagate instead of hanging on wait_for timeouts.
        self._audio_cached_event.set()

        for task in (self._pending_transcript_task, self.generation_task):
            if task and not task.done():
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                    pass

        await super().cleanup()

    # -------------------------------------------------------------------------
    # Event handling
    # -------------------------------------------------------------------------

    async def handle_event(self, event: dict):
        """Override to handle tool-related events."""
        event_type = event.get("type")

        if event_type == RealtimeEventType.SESSION_UPDATE:
            session = event.get("session", {})
            self.tools = session.get("tools")
            self.instructions = session.get("instructions")
            logger.info(f"Session updated with {len(self.tools) if self.tools else 0} tools")
            await super().handle_event(event)

        elif event_type == RealtimeEventType.INPUT_AUDIO_BUFFER_COMMIT:
            # Override commit handling: start generation AND close the audio stream
            # so buffer_realtime_audio() can flush and finish, which in turn lets
            # _add_streaming_input_request send resumable=False to the engine.
            # Without the None sentinel, the thinker stage never gets finished=True
            # and never forwards to talker→code2wav, so audio never arrives.
            commit_event = InputAudioBufferCommit(**event)
            if not commit_event.final:
                await self.start_generation()
            self.audio_queue.put_nowait(None)

        elif event_type == RealtimeEventType.CONVERSATION_ITEM_CREATE:
            item = event.get("item", {})
            await self._handle_conversation_item(item)

        else:
            await super().handle_event(event)

    async def _handle_conversation_item(self, item: dict):
        """Handle conversation item creation (e.g., tool results)."""
        item_type = item.get("type")

        if item_type == "function_call_output":
            if not self.tools:
                # Plain/instructions modes don't run the tool-call detection
                # path, so a function_call_output here is from a misbehaving
                # client — ignore rather than dispatching the audio-pass.
                logger.warning("Received function_call_output but no tools are configured; ignoring.")
                return
            tool_result = {
                "role": "tool",
                "content": item.get("output", ""),
                "call_id": item.get("call_id"),
            }
            self.conversation_items.append(tool_result)
            logger.info(f"Received tool result for call_id: {tool_result['call_id']}")
            await self._generate_with_tool_context()

    async def _generate_with_tool_context(self):
        """Run audio-only generation after tool results have been received."""
        if self.generation_task is not None and not self.generation_task.done():
            # Prior pass still running — flag so its finally-block dispatches
            # the follow-up audio pass once it finishes.
            logger.info("Prior generation still running — deferring audio pass")
            self._pending_tool_context = True
            return

        self.generation_task = asyncio.create_task(self._run_audio_from_tool_context(append_response=True))

    # -------------------------------------------------------------------------
    # Generation entry point
    # -------------------------------------------------------------------------

    async def start_generation(self):
        """Start the transcription generation task with conversation context support."""
        if self.generation_task is not None and not self.generation_task.done():
            logger.warning("Generation already in progress, ignoring commit")
            return

        # Wait briefly for the prior turn's STT so this turn's history has a
        # real transcript instead of a placeholder.
        if self._pending_transcript_task is not None and not self._pending_transcript_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._pending_transcript_task), timeout=3.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "Prior-turn transcript still pending after 3s — history will use placeholder for that turn"
                )
        self._pending_transcript_task = None

        # Reset the audio cache and clear the event BEFORE firing the STT
        # task — otherwise STT sees a still-set event from the prior turn
        # and reads stale _cached_user_audio.
        self._cached_user_audio = []
        self._audio_cached_event.clear()

        audio_stream = self.audio_stream_generator()
        input_stream = asyncio.Queue[list[int]]()

        # Streaming text-pass route (plain + instructions modes) injects the
        # system body via ``conversation_context``. The audio-pass route
        # (tools mode) builds its own prompt and ignores this — passing
        # context=None there avoids buffer_realtime_audio splitting the
        # initial add_request and leaving the talker with prompt_token_ids
        # that have no <|im_start|>user marker.
        conversation_context = getattr(self, "conversation_context", None)
        if conversation_context is None and not self.tools:
            conversation_context = self._build_system_context()

        prior_blocks = self.model_cls.render_history(self.conversation_items) if self.conversation_items else None

        # Reserve a user slot whose content will be filled by the side-channel
        # STT. _build_conversation_context skips None-content items, so the slot is invisible
        # until STT completes (typically before the next turn).
        self._current_user_item = {"role": "user", "content": None}
        self.conversation_items.append(self._current_user_item)

        self._pending_transcript_task = asyncio.create_task(self._transcribe_user_audio(self._current_user_item))

        streaming_input_gen = self._transcribe_realtime(audio_stream, input_stream, conversation_context, prior_blocks)
        self.conversation_context = None

        self.generation_task = asyncio.create_task(self._run_generation(streaming_input_gen, input_stream))

    # -------------------------------------------------------------------------
    # Audio utilities (from main — delta deduplication + format conversion)
    # -------------------------------------------------------------------------

    @staticmethod
    def _tensor_to_numpy(value) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            arr = value
        elif hasattr(value, "detach"):
            arr = value.detach().float().cpu().numpy()
        else:
            try:
                arr = np.asarray(value)
            except Exception:
                return None
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _numpy_audio_prefix_match(prev: np.ndarray, curr: np.ndarray) -> bool:
        n = prev.shape[0]
        if n == 0:
            return True
        if curr.shape[0] < n:
            return False
        return bool(np.allclose(curr[:n], prev, rtol=1e-3, atol=2e-4))

    def _raw_waveform_to_deltas(self, arr: np.ndarray) -> list[np.ndarray]:
        """Convert one streaming PCM f32 chunk into incremental piece(s).

        Handles both cumulative-waveform and true-delta engine output modes
        without duplicating audio on the client.
        """
        if arr.size == 0:
            return []
        ref = self._realtime_audio_ref
        if ref is None:
            self._realtime_audio_ref = arr.copy()
            return [arr]
        if self._numpy_audio_prefix_match(ref, arr):
            delta = arr[ref.shape[0] :]
            self._realtime_audio_ref = arr.copy()
            return [delta] if delta.size > 0 else []
        self._realtime_audio_ref = np.concatenate([ref, arr])
        return [arr]

    def _extract_audio_chunks(self, output) -> tuple[list[np.ndarray], int]:
        mm = getattr(output, "multimodal_output", None)
        if not isinstance(mm, dict):
            return [], 24000

        sr = mm.get("sr") or mm.get("sample_rate") or mm.get("audio_sample_rate") or 24000
        key = "audio" if "audio" in mm else ("model_outputs" if "model_outputs" in mm else None)
        if key is None:
            return [], int(sr)

        raw_audio = mm.get(key)
        chunks: list[np.ndarray] = []
        if isinstance(raw_audio, (list, tuple)):
            if len(raw_audio) > 0:
                arr = self._tensor_to_numpy(raw_audio[-1])
                if arr is not None and arr.size > 0:
                    chunks.extend(self._raw_waveform_to_deltas(arr))
        else:
            arr = self._tensor_to_numpy(raw_audio)
            if arr is not None and arr.size > 0:
                chunks.extend(self._raw_waveform_to_deltas(arr))
        return chunks, int(sr)

    # Maximum raw PCM bytes per WebSocket message for response.audio.delta.
    # Base64 encoding inflates by ~4/3, so 200 KB raw → ~267 KB on the wire.
    _AUDIO_DELTA_CHUNK_BYTES: int = 200 * 1024

    async def _send_audio_delta(self, chunk_f32: np.ndarray, sample_rate: int) -> None:
        """Send a f32 PCM chunk as one or more response.audio.delta messages.

        Converts to int16 and splits into _AUDIO_DELTA_CHUNK_BYTES pieces so
        no single WebSocket frame exceeds client size limits.
        """
        raw = (np.clip(chunk_f32, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        size = self._AUDIO_DELTA_CHUNK_BYTES
        for i in range(0, max(len(raw), 1), size):
            piece = raw[i : i + size]
            if not piece:
                break
            await self.send(
                ResponseAudioDelta(
                    audio=base64.b64encode(piece).decode("utf-8"),
                    sample_rate_hz=sample_rate,
                )
            )

    # -------------------------------------------------------------------------
    # Generation loops
    # -------------------------------------------------------------------------

    async def _run_generation(
        self,
        streaming_input_gen: AsyncGenerator,
        input_stream: asyncio.Queue[list[int]],
    ):
        """Override generation to add text streaming and tool call detection."""
        request_id = f"rt-{self.connection_id}-{uuid4()}"
        sent_audio = False
        done_sent = False
        self._realtime_audio_ref = None

        self.current_tool_calls = []

        try:
            if self.tools or self.instructions:
                # Tools or instructions mode: skip the streaming text pass and
                # route through the audio pass. Tools mode needs the tool-call
                # detection / abort path; instructions mode needs the audio pass
                # because buffer_realtime_audio with a system-prefix yields a
                # text-only first chunk that the streaming engine path processes
                # as text-only, producing TranscriptionDelta but no audio.
                self._cached_user_audio = []
                self._audio_cached_event.clear()
                while True:
                    chunk = await self.audio_queue.get()
                    if chunk is None:
                        break
                    self._cached_user_audio.append(chunk)
                if self._turn_audio_cache is None and self._cached_user_audio:
                    self._turn_audio_cache = list(self._cached_user_audio)
                self._audio_cached_event.set()
                logger.info(
                    "Audio queue drained (%d chunks) — dispatching audio pass",
                    len(self._cached_user_audio),
                )
                self.generation_task = asyncio.create_task(self._run_audio_from_tool_context(append_response=True))
                return

            else:
                # No tools and no instructions — single audio pass (fast path)
                self._active_request_ids.add(request_id)
                result_gen = self.engine.generate(
                    prompt=streaming_input_gen,
                    request_id=request_id,
                    output_modalities=["audio"],
                    sampling_params_list=coerce_param_message_types(
                        list(self.engine.default_sampling_params_list), is_streaming=True
                    ),
                )
                full_text = ""
                prompt_token_ids_len = 0
                completion_tokens_len = 0
                last_prompt_token_ids_len = 0  # detect Stage-0 segment rollover
                async for output in result_gen:
                    if output.outputs and len(output.outputs) > 0:
                        first_output = output.outputs[0]
                        new_token_ids = list(first_output.token_ids)

                        # Stage-0 segment rollover: buffer_realtime_audio may
                        # yield multiple TokensPrompt segments and the second
                        # segment's prompt_token_ids include the first segment's
                        # decoded output. Clear accumulated full_text at the
                        # boundary so the carried-over prefix is not duplicated
                        # in the final TranscriptionDone.text.
                        cur_prompt_token_ids_len = len(output.prompt_token_ids or [])
                        stage_id = getattr(output, "stage_id", None)
                        if stage_id == 0 and cur_prompt_token_ids_len > last_prompt_token_ids_len > 0:
                            full_text = ""
                        last_prompt_token_ids_len = cur_prompt_token_ids_len

                        if not prompt_token_ids_len and output.prompt_token_ids:
                            prompt_token_ids_len = len(output.prompt_token_ids)
                        if new_token_ids:
                            input_stream.put_nowait(new_token_ids)
                        delta_text = first_output.text or ""
                        full_text += delta_text
                        if delta_text:
                            await self.send(TranscriptionDelta(delta=delta_text))
                        completion_tokens_len += len(new_token_ids)
                    audio_chunks, sample_rate = self._extract_audio_chunks(output)
                    for chunk in audio_chunks:
                        sent_audio = True
                        await self._send_audio_delta(chunk, sample_rate)
                    if not self._is_connected:
                        break

                self.conversation_items.append(
                    {
                        "role": "assistant",
                        "content": full_text or None,
                        "tool_calls": None,
                    }
                )
                usage = UsageInfo(
                    prompt_tokens=prompt_token_ids_len,
                    completion_tokens=completion_tokens_len,
                    total_tokens=prompt_token_ids_len + completion_tokens_len,
                )
                await self.send(TranscriptionDone(text=full_text, usage=usage))

            if sent_audio:
                await self.send(ResponseAudioDone())
                done_sent = True

        except Exception as e:
            logger.exception("Error in generation: %s", e)
            await self.send_error(str(e), "processing_error")
        finally:
            if self._is_connected and not done_sent and sent_audio:
                try:
                    await self.send(ResponseAudioDone())
                except Exception:
                    logger.exception("Failed to send response.audio.done")
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()

    async def _run_audio_from_tool_context(self, append_response: bool = False) -> None:
        """Generate speech after receiving tool results (or for direct responses).

        The actual prompt is assembled by ``model_cls.build_audio_pass_prompt``;
        the server passes only the system body (instructions + tool schema)
        and the rendered history. The audio user block stays first so the
        talker's hidden_projection sees purely acoustic states for bootstrap.
        """
        sent_audio = False
        done_sent = False
        self._realtime_audio_ref = None
        try:
            # Wait for the side-channel STT so the current user item has a
            # transcript before we render history. Without it, render_history
            # skips the None-content user item and leaves an
            # assistant->assistant adjacency that breaks alternation.
            if self._pending_transcript_task is not None and not self._pending_transcript_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(self._pending_transcript_task), timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Side-channel transcript not done before audio pass — "
                        "this turn's user line will be missing from history"
                    )

            system_body = self._build_system_context()
            history = self.model_cls.render_history(self.conversation_items)
            full_prompt = self.model_cls.build_audio_pass_prompt(system_body, history)

            # Wait for audio_stream_generator's finally-block to finish draining
            # audio_queue. On a tool-call abort it can be interrupted mid-stream
            # and we'd otherwise read a partial _cached_user_audio.
            try:
                await asyncio.wait_for(self._audio_cached_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("audio-pass timed out waiting for audio cache — proceeding with whatever is buffered")

            # Prefer current-turn audio for acoustic conditioning; fall back
            # to the first-turn cache if the buffer is empty.
            ref_audio = self._cached_user_audio if self._cached_user_audio else self._turn_audio_cache
            audio_array = np.concatenate(ref_audio) if ref_audio else np.zeros(8000, dtype=np.float32)

            # Estimate total tokens (text + ~640 samples per audio token at
            # 16 kHz) and warn when it eats into the smallest stage's budget.
            prompt_token_count = -1
            if self.tokenizer is not None:
                try:
                    prompt_token_count = len(self.tokenizer.encode(full_prompt))
                except Exception as exc:
                    logger.warning("Could not count prompt tokens: %s", exc)
            estimated_audio_tokens = audio_array.size // 640
            estimated_total = max(prompt_token_count, 0) + estimated_audio_tokens

            stage_limits: list[tuple[int, int]] = []
            try:
                for idx, vllm_cfg in enumerate(getattr(self.engine, "stage_vllm_configs", []) or []):
                    if vllm_cfg is None:
                        continue
                    model_config = getattr(vllm_cfg, "model_config", None)
                    if model_config is None:
                        continue
                    mml = getattr(model_config, "max_model_len", None)
                    if mml:
                        stage_limits.append((idx, int(mml)))
            except Exception as exc:
                logger.debug("Could not read per-stage max_model_len: %s", exc)

            if stage_limits:
                min_limit = min(m for _, m in stage_limits)
                # 50% threshold leaves headroom for the codec generation that
                # follows the prompt.
                if estimated_total > min_limit // 2:
                    logger.warning(
                        "audio-pass prompt (~%d tokens) is %d%% of the smallest "
                        "stage's max_model_len (%d). Codec generation needs the "
                        "remaining budget — hangs or truncated audio likely if "
                        "total runs past that limit. Consider trimming history "
                        "or shortening tool responses.",
                        estimated_total,
                        int(100 * estimated_total / min_limit),
                        min_limit,
                    )

            prompt = {
                "prompt": full_prompt,
                "multi_modal_data": {
                    "audio": (audio_array, 16000),
                },
            }

            request_id = f"rt-{self.connection_id}-{uuid4()}-aud"
            self._active_request_ids.add(request_id)
            result_gen = self.engine.generate(
                prompt=prompt,
                request_id=request_id,
                output_modalities=["audio"],
                sampling_params_list=coerce_param_message_types(
                    list(self.engine.default_sampling_params_list), is_streaming=True
                ),
            )

            spoken_text = ""
            audio_chunk_count = 0
            t_audio_start = time.monotonic()
            last_progress_len = 0
            last_progress_time = t_audio_start
            audio_pass_tool_call_detected = False
            # Buffer audio chunks until the thinker has generated enough text
            # that a tool call is no longer possible. With async_chunk the
            # pipeline runs in parallel so audio may arrive mid-thinker; we
            # hold it here and flush once unlocked, or discard on tool call.
            # Without async_chunk audio only arrives after the thinker
            # finishes anyway, so the buffer just flushes immediately.
            # The unlock threshold is model-defined (see model_cls).
            _AUDIO_UNLOCK_CHARS = self.model_cls.REALTIME_AUDIO_UNLOCK_CHARS
            pending_audio: list[tuple[np.ndarray, int]] = []
            audio_streaming_unlocked = False
            try:
                async for output in result_gen:
                    if output.outputs:
                        spoken_text += output.outputs[0].text or ""

                    # If the model emits a tool call here, the talker tries to
                    # vocalize the XML and spins until max_tokens. Detect,
                    # abort, and let _handle_conversation_item re-dispatch the
                    # audio pass once the function_call_output arrives.
                    tc_open = self.model_cls.TOOL_CALL_OPEN
                    tc_close = self.model_cls.TOOL_CALL_CLOSE
                    if not audio_pass_tool_call_detected and tc_open in spoken_text and tc_close in spoken_text:
                        tc_start = spoken_text.find(tc_open)
                        tc_end = spoken_text.find(tc_close) + len(tc_close)
                        tool_call_block = spoken_text[tc_start:tc_end]
                        parsed = self.model_cls.parse_tool_call(tool_call_block)
                        if parsed:
                            tool_call_id = f"call_{uuid4().hex[:24]}"
                            self.current_tool_calls.append(
                                {
                                    "id": tool_call_id,
                                    "name": parsed["name"],
                                    "arguments": parsed["arguments"],
                                }
                            )
                            visible = spoken_text[:tc_start]
                            visible_clean = self.model_cls.strip_special_tokens(visible) or None
                            self.conversation_items.append(
                                {
                                    "role": "assistant",
                                    "content": visible_clean,
                                    "tool_calls": self.current_tool_calls,
                                }
                            )
                            args_json = json.dumps(parsed["arguments"])
                            await self.send(
                                ResponseFunctionCallArgumentsDelta(
                                    call_id=tool_call_id,
                                    name=parsed["name"],
                                    delta=args_json,
                                )
                            )
                            await self.send(
                                ResponseFunctionCallArgumentsDone(
                                    call_id=tool_call_id,
                                    name=parsed["name"],
                                    arguments=args_json,
                                )
                            )
                            logger.info(
                                "audio-pass tool call detected at %.2fs — aborting; "
                                "discarding %d buffered audio chunk(s); emitted %d tool call(s)",
                                time.monotonic() - t_audio_start,
                                len(pending_audio),
                                len(self.current_tool_calls),
                            )
                            pending_audio.clear()
                            await self.engine.abort(request_id)
                            audio_pass_tool_call_detected = True
                            break

                    # Unlock streaming once enough text has been generated
                    # without a tool call open tag — safe to start sending audio.
                    if (not audio_streaming_unlocked
                            and tc_open not in spoken_text
                            and len(spoken_text) >= _AUDIO_UNLOCK_CHARS):
                        audio_streaming_unlocked = True
                        logger.info(
                            "audio streaming unlocked at %.2fs (%d chars, %d buffered chunk(s))",
                            time.monotonic() - t_audio_start,
                            len(spoken_text),
                            len(pending_audio),
                        )

                    audio_chunks, sample_rate = self._extract_audio_chunks(output)
                    for chunk in audio_chunks:
                        if audio_streaming_unlocked:
                            sent_audio = True
                            audio_chunk_count += 1
                            await self._send_audio_delta(chunk, sample_rate)
                        else:
                            pending_audio.append((chunk, sample_rate))

                    # Flush any buffered chunks once unlocked.
                    if audio_streaming_unlocked and pending_audio:
                        for chunk, sr in pending_audio:
                            sent_audio = True
                            audio_chunk_count += 1
                            await self._send_audio_delta(chunk, sr)
                        pending_audio.clear()

                    # Periodic progress: text accumulating without audio
                    # chunks ⇒ stage-1 hang; neither ticking ⇒ stage-0 hang.
                    now = time.monotonic()
                    if len(spoken_text) - last_progress_len > 100 or now - last_progress_time > 5.0:
                        last_progress_len = len(spoken_text)
                        last_progress_time = now

                    if not self._is_connected:
                        break
            finally:
                # Flush buffered audio only on a clean (no-tool-call) exit.
                if not audio_pass_tool_call_detected:
                    for chunk, sample_rate in pending_audio:
                        sent_audio = True
                        audio_chunk_count += 1
                        await self._send_audio_delta(chunk, sample_rate)
                pending_audio.clear()
                # Final state — visible even when stage-1/2 hangs.
                logger.info(
                    "audio-pass exit: %.2fs | spoken_len=%d | audio_chunks=%d | sent_audio=%s",
                    time.monotonic() - t_audio_start,
                    len(spoken_text),
                    audio_chunk_count,
                    sent_audio,
                )

            if sent_audio:
                await self.send(ResponseAudioDone())
                done_sent = True

            if append_response and not audio_pass_tool_call_detected:
                # Only append when no tool call was detected — the detection
                # branch already appended an assistant entry with tool_calls=[...].
                # Appending again would create an assistant→assistant adjacency
                # in history and duplicate the visible pre-tool text.
                clean = self.model_cls.strip_special_tokens(spoken_text)
                self.conversation_items.append(
                    {
                        "role": "assistant",
                        "content": clean or None,
                        "tool_calls": None,
                    }
                )

        except Exception as exc:
            logger.exception("Error in audio-from-tool-context pass: %s", exc)
            await self.send_error(str(exc), "processing_error")
        finally:
            if not done_sent and sent_audio:
                try:
                    await self.send(ResponseAudioDone())
                except Exception:
                    pass

            # If a tool result arrived while this pass was running,
            # _generate_with_tool_context flagged it and bailed; dispatch
            # the follow-up audio pass now so history includes the result.
            if self._pending_tool_context:
                self._pending_tool_context = False
                logger.info("Tool result pending — dispatching follow-up audio pass")
                self.generation_task = asyncio.create_task(self._run_audio_from_tool_context(append_response=True))

    # -------------------------------------------------------------------------
    # Conversation helpers
    # -------------------------------------------------------------------------

    async def _transcribe_user_audio(self, user_item: dict) -> None:
        """Run an ASR pass on the just-committed user audio.

        Fires as a side-channel request alongside the chat pipeline. Stage 0
        continuous-batches this with whatever the chat thinker is doing;
        stages 1+2 never see it because output_modalities is text-only.

        Result is written into ``user_item["content"]`` in place so the
        next turn's history rendering picks up the real transcript.
        """
        try:
            # Wait for audio_stream_generator to finish draining audio_queue
            # into _cached_user_audio. Bounded so a stalled audio stream
            # doesn't hang the task forever.
            try:
                await asyncio.wait_for(self._audio_cached_event.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("[STT] audio cache wait timed out — aborting transcription")
                return

            if not self._cached_user_audio:
                logger.warning("[STT] no cached audio to transcribe")
                return

            audio_array = np.concatenate(self._cached_user_audio)
            transcribe_prompt = self.model_cls.build_transcription_prompt()

            request_id = f"rt-{self.connection_id}-{uuid4()}-stt"
            self._active_request_ids.add(request_id)
            result_gen = self.engine.generate(
                prompt={
                    "prompt": transcribe_prompt,
                    "multi_modal_data": {"audio": (audio_array, 16000)},
                },
                request_id=request_id,
                output_modalities=["text"],
                sampling_params_list=coerce_param_message_types(
                    list(self.engine.default_sampling_params_list), is_streaming=True
                ),
            )

            full_text = ""
            async for output in result_gen:
                if output.outputs:
                    token_ids = list(output.outputs[0].token_ids)
                    if token_ids:
                        delta = self._decode_tokens(token_ids)
                        if delta:
                            full_text += delta
                # Stop reading when the assistant turn closes — saves a few
                # tokens of trailing junk on greedy decoding.
                turn_end = self.model_cls.ASSISTANT_TURN_END
                if turn_end in full_text:
                    await self.engine.abort(request_id)
                    break

            transcript = full_text.split(self.model_cls.ASSISTANT_TURN_END)[0]
            transcript = self.model_cls.strip_special_tokens(transcript)
            user_item["content"] = transcript or None
        except Exception as exc:
            logger.exception("[STT] transcription failed: %s", exc)

    def _build_system_context(self) -> str | None:
        """System message body: instructions, then the model's tool schema.

        Ordering (instructions first, then tools) matches Qwen3's chat
        template; the model class is responsible for the schema text.
        """
        parts: list[str] = []
        if self.instructions:
            parts.append(self.instructions)
        if self.tools:
            parts.append(self.model_cls.format_tools_schema(self.tools))
        return "\n\n".join(parts) if parts else None

    def audio_stream_generator(self):
        """Override to cache audio chunks for replay in tool-context audio pass."""

        async def _gen():
            self._cached_user_audio = []
            self._audio_cached_event.clear()
            try:
                while True:
                    audio_chunk = await self.audio_queue.get()
                    if audio_chunk is None:
                        break
                    self._cached_user_audio.append(audio_chunk)
                    yield audio_chunk
            finally:
                # The text/audio pass can abort early (e.g. on tool_call
                # detection) before this generator consumes the trailing
                # None sentinel; drain non-blockingly so STT and the audio
                # pass still see the full user audio.
                try:
                    while True:
                        chunk = self.audio_queue.get_nowait()
                        if chunk is None:
                            break
                        self._cached_user_audio.append(chunk)
                except asyncio.QueueEmpty:
                    pass
                if self._turn_audio_cache is None and self._cached_user_audio:
                    self._turn_audio_cache = list(self._cached_user_audio)
                # Always signal — STT must not hang on early abort.
                self._audio_cached_event.set()

        return _gen()

    # -------------------------------------------------------------------------
    # Streaming input helpers
    # -------------------------------------------------------------------------

    async def _transcribe_realtime(
        self,
        audio_stream: AsyncGenerator,
        input_stream: asyncio.Queue,
        conversation_context: str | None = None,
        prior_blocks: str | None = None,
    ) -> AsyncGenerator[StreamingInput, None]:
        """Wrap buffer_realtime_audio into StreamingInput for engine.generate().

        Falls back to the upstream two-arg transcribe_realtime when no
        conversation context or prior history is present, so plain sessions
        are unaffected by the multi-turn machinery.
        """
        if conversation_context is None and prior_blocks is None:
            async for item in self.serving.transcribe_realtime(audio_stream, input_stream):
                yield item
            return

        model_config = self.serving.model_config
        renderer = self.serving.renderer
        stream_input_iter = cast(
            AsyncGenerator[PromptType, None],
            self.model_cls.buffer_realtime_audio(
                audio_stream, input_stream, model_config, conversation_context, prior_blocks
            ),
        )
        async for prompt in stream_input_iter:
            parsed_prompt = parse_model_prompt(model_config, prompt)
            (engine_prompt,) = await renderer.render_cmpl_async([parsed_prompt])
            yield StreamingInput(prompt=engine_prompt)

    # -------------------------------------------------------------------------
    # Text processing helpers
    # -------------------------------------------------------------------------

    def _decode_tokens(self, token_ids: list[int]) -> str:
        if not self.tokenizer or not token_ids:
            return ""
        try:
            return self.tokenizer.decode(token_ids, skip_special_tokens=False)
        except Exception as e:
            logger.warning(f"Failed to decode tokens: {e}")
            return ""

    # -------------------------------------------------------------------------
    # WebSocket send
    # -------------------------------------------------------------------------

    async def send(self, event: OpenAIBaseModel) -> None:  # type: ignore[override]
        await self.websocket.send_text(event.model_dump_json())
