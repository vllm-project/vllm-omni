from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncGenerator, Mapping
from typing import Any, cast
from uuid import uuid4

import numpy as np
from vllm.engine.protocol import StreamingInput
from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.speech_to_text.realtime.connection import RealtimeConnection as VllmRealtimeConnection
from vllm.entrypoints.speech_to_text.realtime.protocol import TranscriptionDelta, TranscriptionDone
from vllm.inputs import TokensPrompt
from vllm.logger import init_logger
from vllm.renderers.hf import safe_apply_chat_template
from vllm.renderers.inputs.preprocess import parse_model_prompt
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.transformers_utils.processor import cached_processor_from_config

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.realtime_tool_calls import ToolCallStreamState, extract_deltas
from vllm_omni.entrypoints.utils import coerce_param_message_types

logger = init_logger(__name__)


class RealtimeConnection(VllmRealtimeConnection):
    """Omni realtime connection with audio-only server events, plus
    OpenAI-Realtime-shaped tool/function calling.

    Reuses upstream vLLM websocket/session lifecycle and customizes
    generation output handling to emit audio deltas and tool-call events.

    Tool-calling protocol (mirrors OpenAI's Realtime API event shapes):
      - client -> server: `session.update` gains an optional `tools` field
        (list of OpenAI-style tool/function definitions).
      - server -> client, once the model starts a tool call:
        `response.output_item.added` (item.type="function_call", name, call_id)
        `response.function_call_arguments.delta` (call_id, delta)
        `response.function_call_arguments.done` (call_id, arguments)
      - client -> server, once the tool has run:
        `conversation.item.create` with
        `item = {"type": "function_call_output", "call_id": ..., "output": "..."}`
      - generation then continues automatically with the tool result appended,
        streaming the model's actual spoken reply as normal.

    No audio is synthesized/forwarded for a turn that turns out to be a tool
    call - the underlying 3-stage pipeline (thinker->talker->code2wav) still
    runs end to end for it (there is no clean lower-level hook to skip talker/
    code2wav without changing the shared orchestrator - see PR description),
    but the resulting audio bytes are simply not sent to the client.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = cast(AsyncOmni, self.serving.engine_client)
        self._realtime_audio_ref: np.ndarray | None = None
        self._tools: list[dict[str, Any]] | None = None
        # index (parser-assigned, per generation) -> {"call_id", "name", "arguments"}
        self._pending_tool_calls: dict[int, dict[str, Any]] = {}
        self._tool_result_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def handle_event(self, event: dict):
        event_type = event.get("type")
        if event_type == "session.update":
            tools = event.get("tools")
            if tools is not None:
                self._tools = tools
            await super().handle_event(event)
        elif event_type == "conversation.item.create":
            item = event.get("item") or {}
            if item.get("type") == "function_call_output":
                self._tool_result_queue.put_nowait(item)
            else:
                await self.send_error(f"Unsupported conversation.item type: {item.get('type')!r}", "unsupported_item")
        else:
            await super().handle_event(event)

    async def start_generation(self):
        if self.generation_task is not None and not self.generation_task.done():
            logger.warning("Generation already in progress, ignoring commit")
            return

        audio_stream = self.audio_stream_generator()
        input_stream: asyncio.Queue[list[int]] = asyncio.Queue()
        streaming_input_gen = self._buffer_realtime_audio_with_tools(audio_stream, input_stream)
        self.generation_task = asyncio.create_task(self._run_generation(streaming_input_gen, input_stream))

    async def _buffer_realtime_audio_with_tools(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
        input_stream: asyncio.Queue[list[int]],
    ) -> AsyncGenerator[StreamingInput, None]:
        """Equivalent to `OpenAIServingRealtime.transcribe_realtime`, but
        threads `self._tools` through to the model's `buffer_realtime_audio`.
        The base class's `transcribe_realtime` has a fixed
        (audio_stream, input_stream, model_config) call signature with no
        seam for extra per-connection state like tools, so this reimplements
        its (short) body directly rather than patching upstream vLLM."""
        model_config = self.serving.model_config
        renderer = self.serving.renderer
        stream_input_iter = self.serving.model_cls.buffer_realtime_audio(
            audio_stream, input_stream, model_config, tools=self._tools
        )
        async for prompt in stream_input_iter:
            parsed_prompt = parse_model_prompt(model_config, prompt)
            (engine_input,) = await renderer.render_cmpl_async([parsed_prompt])
            yield StreamingInput(prompt=engine_input)

    async def _render_token_prompt(self, prompt_token_ids: list[int]) -> AsyncGenerator[StreamingInput, None]:
        model_config = self.serving.model_config
        renderer = self.serving.renderer
        parsed_prompt = parse_model_prompt(model_config, TokensPrompt(prompt_token_ids=prompt_token_ids))
        (engine_input,) = await renderer.render_cmpl_async([parsed_prompt])
        yield StreamingInput(prompt=engine_input)

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
        """Convert one streaming PCM f32 chunk into incremental piece(s) for the client.

        Some engine paths emit a growing cumulative waveform each step; others emit
        true per-step deltas. We support both without duplicating audio on the client.
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
        # True per-step delta (not a prefix extension of what we have seen).
        self._realtime_audio_ref = np.concatenate([ref, arr])
        return [arr]

    def _extract_audio_chunks(self, output) -> tuple[list[np.ndarray], int]:
        mm = getattr(output, "multimodal_output", None)
        if mm is None:
            return [], 24000
        # Support both MultimodalPayload and plain dict
        if not isinstance(mm, Mapping):
            return [], 24000

        sr = mm.get("sr") or mm.get("sample_rate") or mm.get("audio_sample_rate") or 24000
        if isinstance(sr, (list, tuple)) and sr:
            sr = sr[-1]
        if hasattr(sr, "item"):
            sr = sr.item()
        sample_rate_hz = int(sr)
        key = "audio" if "audio" in mm else ("model_outputs" if "model_outputs" in mm else None)
        if key is None:
            return [], sample_rate_hz

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
        return chunks, sample_rate_hz

    @staticmethod
    def _pcm16_b64(audio_f32: np.ndarray) -> str:
        clipped = np.clip(audio_f32, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        return base64.b64encode(pcm16.tobytes()).decode("utf-8")

    async def _emit_tool_call_deltas(self, tool_deltas: list) -> None:
        for delta in tool_deltas:
            if delta.name is not None:
                call_id = f"call_{uuid4().hex[:24]}"
                self._pending_tool_calls[delta.index] = {"call_id": call_id, "name": delta.name, "arguments": ""}
                await self.send_json(
                    {
                        "type": "response.output_item.added",
                        "item": {"type": "function_call", "name": delta.name, "call_id": call_id},
                    }
                )
            if delta.arguments_delta:
                info = self._pending_tool_calls.get(delta.index)
                if info is None:
                    continue  # shouldn't happen: name delta always precedes argument deltas for the same index
                info["arguments"] += delta.arguments_delta
                await self.send_json(
                    {
                        "type": "response.function_call_arguments.delta",
                        "call_id": info["call_id"],
                        "delta": delta.arguments_delta,
                    }
                )

    async def _run_generation(
        self,
        streaming_input_gen: AsyncGenerator,
        input_stream: asyncio.Queue[list[int]],
    ):
        request_id = f"rt-{self.connection_id}-{uuid4()}"
        sent_audio = False
        audio_done_sent = False
        full_text = ""
        prompt_token_ids_len = 0
        completion_tokens_len = 0
        self._realtime_audio_ref = None

        request_prompt_token_ids: list[int] = []
        assistant_token_ids: list[int] = []
        tool_state = ToolCallStreamState()
        self._pending_tool_calls = {}

        # Coerce cumulative outputs to delta outputs; this ensures
        # we don't emit redundant MM data & drain after emitting.
        sampling_params_list = list(self.engine.default_sampling_params_list)
        sampling_params_list = coerce_param_message_types(
            sampling_params_list,
            is_streaming=True,
        )

        result_gen = None
        try:
            result_gen = self.engine.generate(
                prompt=streaming_input_gen,
                request_id=request_id,
                sampling_params_list=sampling_params_list,
            )

            async for output in result_gen:
                stage_id = getattr(output, "stage_id", None)
                if stage_id == 0 and output.outputs:
                    first_output = output.outputs[0]
                    new_token_ids = list(first_output.token_ids)
                    if new_token_ids:
                        input_stream.put_nowait(new_token_ids)
                        assistant_token_ids.extend(new_token_ids)

                    if output.prompt_token_ids:
                        prompt_token_ids_len = max(
                            prompt_token_ids_len,
                            len(output.prompt_token_ids),
                        )
                        if not request_prompt_token_ids:
                            request_prompt_token_ids = list(output.prompt_token_ids)

                    delta_text = first_output.text or ""
                    full_text += delta_text
                    completion_tokens_len += len(new_token_ids)

                    if delta_text:
                        content_delta, tool_deltas = extract_deltas(full_text, tool_state)
                        if content_delta:
                            await self.send(TranscriptionDelta(delta=content_delta))
                        if tool_deltas:
                            await self._emit_tool_call_deltas(tool_deltas)

                audio_chunks, sample_rate = self._extract_audio_chunks(output)
                if audio_chunks and not tool_state.has_tool_calls():
                    for chunk in audio_chunks:
                        sent_audio = True
                        await self.send_json(
                            {
                                "type": "response.audio.delta",
                                "audio": self._pcm16_b64(chunk),
                                "format": "pcm16",
                                "sample_rate_hz": sample_rate,
                            }
                        )
                # else: a tool-call turn - the pipeline still synthesizes audio for the
                # raw <tool_call> text (no cheap hook to skip talker/code2wav for just
                # this turn), we just don't forward it to the client.

                if not self._is_connected:
                    break

            if tool_state.has_tool_calls():
                for idx, info in self._pending_tool_calls.items():
                    await self.send_json(
                        {
                            "type": "response.function_call_arguments.done",
                            "call_id": info["call_id"],
                            "arguments": info["arguments"],
                        }
                    )
                if self._is_connected:
                    await self._await_tool_results_and_continue(request_prompt_token_ids, assistant_token_ids)
                return

            usage = UsageInfo(
                prompt_tokens=prompt_token_ids_len,
                completion_tokens=completion_tokens_len,
                total_tokens=prompt_token_ids_len + completion_tokens_len,
            )
            await self.send(TranscriptionDone(text=full_text, usage=usage))

            if sent_audio:
                await self.send_json({"type": "response.audio.done", "has_audio": True})
                audio_done_sent = True
        except Exception as e:
            logger.exception("Error in generation: %s", e)
            await self.send_error(str(e), "processing_error")
        finally:
            # Close the generator explicitly so AsyncOmni.generate's cleanup
            # (input-pump cancellation and engine-side abort) runs now rather
            # than whenever the event loop garbage-collects the async
            # generator; the delay window is where a disconnected session
            # keeps cycling through the stages (issue #4271).
            if result_gen is not None:
                try:
                    await result_gen.aclose()
                except Exception:
                    logger.exception("Failed to close realtime result generator")
            # Always send terminal event so clients don't hang forever.
            if self._is_connected and not audio_done_sent and not tool_state.has_tool_calls():
                try:
                    await self.send_json({"type": "response.audio.done", "has_audio": sent_audio})
                except Exception:
                    logger.exception("Failed to send response.audio.done")
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()

    async def _await_tool_results_and_continue(
        self,
        prior_prompt_token_ids: list[int],
        assistant_token_ids: list[int],
    ) -> None:
        """Block until the client has submitted a `function_call_output` for
        every pending tool call from the turn that just finished, then splice
        the results onto the raw token sequence (prior prompt + the model's
        own generated tool-call tokens + a freshly-rendered tool-result turn)
        and continue generation - which streams the model's actual spoken
        reply as a normal `_run_generation` call, recursing again if the
        model chains into another tool call."""
        pending = dict(self._pending_tool_calls)
        call_id_to_index = {info["call_id"]: idx for idx, info in pending.items()}
        results_by_index: dict[int, str] = {}

        while len(results_by_index) < len(pending) and self._is_connected:
            item = await self._tool_result_queue.get()
            call_id = item.get("call_id")
            idx = call_id_to_index.get(call_id)
            if idx is None:
                logger.warning("received function_call_output for unknown call_id=%s", call_id)
                continue
            results_by_index[idx] = str(item.get("output", ""))

        if not self._is_connected:
            return

        model_config = self.serving.model_config
        tokenizer = cached_tokenizer_from_config(model_config)
        # Pass the processor's chat_template explicitly - see the matching
        # comment in Qwen3OmniMoeForConditionalGeneration.buffer_realtime_audio
        # for why relying on safe_apply_chat_template's own auto-resolution
        # is unsafe for this checkpoint.
        processor = cached_processor_from_config(model_config)
        # Multiple tool calls in one turn -> one combined tool-role message,
        # matching how Qwen's own chat template batches consecutive tool
        # results under a single <|im_start|>user block. No `tools=` here:
        # this is a continuation of a conversation that already has the
        # tools system preamble in its token history, not a fresh turn.
        combined_result_text = "\n".join(results_by_index[i] for i in sorted(results_by_index))
        suffix_text = safe_apply_chat_template(
            model_config,
            tokenizer,
            [{"role": "tool", "content": combined_result_text}],
            chat_template=processor.chat_template,
            add_generation_prompt=True,
            tokenize=False,
        )
        continuation_token_ids = (
            list(prior_prompt_token_ids) + list(assistant_token_ids) + tokenizer.encode(suffix_text)
        )

        input_stream: asyncio.Queue[list[int]] = asyncio.Queue()
        await self._run_generation(self._render_token_prompt(continuation_token_ids), input_stream)

    async def send_json(self, payload: dict):
        try:
            await self.websocket.send_text(json.dumps(payload))
        except Exception:
            # A failed send means the client is gone; flag it so the
            # generation loop stops instead of retrying into a dead socket.
            self._is_connected = False
            raise
