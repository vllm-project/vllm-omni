from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import numpy as np
from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.openai.realtime.connection import RealtimeConnection as VllmRealtimeConnection
from vllm.entrypoints.openai.realtime.protocol import (
    InputAudioBufferCommit,
    TranscriptionDelta,
    TranscriptionDone,
)
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.stage_params import clone_sampling_params
from vllm_omni.entrypoints.utils import coerce_param_message_types

if TYPE_CHECKING:
    from vllm_omni.entrypoints.async_omni import AsyncOmni

logger = init_logger(__name__)


class RealtimeConnection(VllmRealtimeConnection):
    """Omni realtime connection with audio-only server events.

    Reuses upstream vLLM websocket/session lifecycle and only customizes
    generation output handling to emit audio deltas.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = cast("AsyncOmni", self._get_serving_engine_client())
        self._realtime_audio_ref: np.ndarray | None = None

    def _get_serving_engine_client(self):
        engine_client = getattr(self.serving, "engine_client", None)
        if engine_client is None:
            engine_client = getattr(self.serving, "_engine_client", None)
        if engine_client is None:
            raise ValueError("Realtime serving object does not expose an engine client.")
        return engine_client

    async def start_generation(self):
        if self._uses_async_chunk_bridge:
            # In async_chunk mode the bridge must wait for final=True so stage 0
            # receives a normal multimodal request instead of streaming updates.
            logger.debug(
                "Ignoring non-final realtime commit for async_chunk bridge: %s",
                self.connection_id,
            )
            return
        await super().start_generation()

    @property
    def _uses_async_chunk_bridge(self) -> bool:
        return bool(getattr(self.engine, "async_chunk", False))

    async def handle_event(self, event: dict):
        if not self._uses_async_chunk_bridge:
            await super().handle_event(event)
            return

        if event.get("type") == "input_audio_buffer.append" and self._generation_in_progress():
            await self.send_error("Generation already in progress", "generation_in_progress")
            return

        if event.get("type") != "input_audio_buffer.commit":
            await super().handle_event(event)
            return

        if not self._is_model_validated:
            err_msg = (
                "Model not validated. Make sure to validate the"
                " model by sending a session.update event."
            )
            await self.send_error(err_msg, "model_not_validated")
            return

        commit_event = InputAudioBufferCommit(**event)
        if not commit_event.final:
            logger.debug(
                "Received non-final realtime commit in async_chunk bridge mode: %s",
                self.connection_id,
            )
            return

        await self._start_async_chunk_bridge_generation()

    async def _start_async_chunk_bridge_generation(self) -> None:
        if self._generation_in_progress():
            logger.warning(
                "Generation already in progress, ignoring final commit: %s",
                self.connection_id,
            )
            await self.send_error("Generation already in progress", "generation_in_progress")
            return

        self.audio_queue.put_nowait(None)
        input_stream = asyncio.Queue[list[int]]()
        self.generation_task = asyncio.create_task(
            self._run_async_chunk_bridge_generation(input_stream),
        )

    def _generation_in_progress(self) -> bool:
        return self.generation_task is not None and not self.generation_task.done()

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

    @staticmethod
    def _pcm16_b64(audio_f32: np.ndarray) -> str:
        clipped = np.clip(audio_f32, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        return base64.b64encode(pcm16.tobytes()).decode("utf-8")

    async def _collect_committed_audio(self) -> np.ndarray:
        chunks: list[np.ndarray] = []
        total_pcm16_nbytes = 0

        while True:
            audio_chunk = await self.audio_queue.get()
            if audio_chunk is None:
                break

            arr = self._tensor_to_numpy(audio_chunk)
            if arr is None or arr.size == 0:
                continue

            arr = np.ascontiguousarray(arr, dtype=np.float32)
            chunks.append(arr)
            total_pcm16_nbytes += arr.size * np.dtype(np.int16).itemsize

            max_mb = getattr(self, "_max_audio_filesize_mb", None)
            if max_mb is not None and total_pcm16_nbytes / 1024**2 > max_mb:
                raise ValueError("Maximum file size exceeded")

        if not chunks:
            raise ValueError("No audio data received before final commit.")

        return np.concatenate(chunks).astype(np.float32, copy=False)

    def _get_realtime_input_sample_rate(self) -> int:
        model_config = getattr(self.serving, "model_config", None)
        if model_config is None:
            return 16000

        try:
            from vllm.transformers_utils.processor import cached_processor_from_config

            processor = cached_processor_from_config(model_config)
            feature_extractor = getattr(processor, "feature_extractor", None)
            sample_rate = getattr(feature_extractor, "sampling_rate", None)
            if sample_rate:
                return int(sample_rate)
        except Exception:
            logger.debug(
                "Failed to resolve realtime input sample rate from processor; "
                "falling back to 16 kHz.",
                exc_info=True,
            )

        return 16000

    @staticmethod
    def _audio_placeholder_from_model_cls(model_cls: Any) -> str:
        get_placeholder_str = getattr(model_cls, "get_placeholder_str", None)
        if get_placeholder_str is not None:
            placeholder = get_placeholder_str("audio", 0)
            if placeholder:
                return str(placeholder)
        return "<|audio_start|><|audio_pad|><|audio_end|>"

    @classmethod
    def _build_realtime_audio_prompt(
        cls,
        audio: np.ndarray,
        sample_rate: int,
        model_cls: Any = None,
    ) -> dict[str, Any]:
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.reshape(-1)
        if audio.size == 0:
            raise ValueError("No audio data received before final commit.")

        audio_placeholder = cls._audio_placeholder_from_model_cls(model_cls)
        prompt = f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n<|im_start|>assistant\n"
        return {
            "prompt": prompt,
            "multi_modal_data": {"audio": (audio, int(sample_rate))},
        }

    def _realtime_sampling_params_list(self):
        sampling_params_list = [
            clone_sampling_params(params)
            for params in self.engine.default_sampling_params_list
        ]
        return coerce_param_message_types(
            sampling_params_list,
            is_streaming=True,
        )

    async def _abort_generation_request(self, request_id: str) -> None:
        abort = getattr(self.engine, "abort", None)
        if abort is None:
            return
        try:
            await abort(request_id)
        except Exception:
            logger.exception("Failed to abort realtime request: %s", request_id)

    async def _consume_generation_outputs(
        self,
        result_gen,
        input_stream: asyncio.Queue[list[int]],
    ) -> tuple[bool, bool]:
        sent_audio = False
        full_text = ""
        prompt_token_ids_len = 0
        completion_tokens_len = 0

        async for output in result_gen:
            if output.outputs and len(output.outputs) > 0:
                first_output = output.outputs[0]
                new_token_ids = list(first_output.token_ids)
                new_tokens_len = len(new_token_ids)

                if not prompt_token_ids_len and output.prompt_token_ids:
                    prompt_token_ids_len = len(output.prompt_token_ids)

                if new_tokens_len:
                    input_stream.put_nowait(new_token_ids)

                delta_text = first_output.text or ""
                full_text += delta_text

                if delta_text:
                    await self.send(TranscriptionDelta(delta=delta_text))

                completion_tokens_len += new_tokens_len

            audio_chunks, sample_rate = self._extract_audio_chunks(output)

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

            if not self._is_connected:
                return sent_audio, False

        usage = UsageInfo(
            prompt_tokens=prompt_token_ids_len,
            completion_tokens=completion_tokens_len,
            total_tokens=prompt_token_ids_len + completion_tokens_len,
        )
        await self.send(TranscriptionDone(text=full_text, usage=usage))
        return sent_audio, True

    def _drain_audio_queue(self) -> None:
        while not self.audio_queue.empty():
            self.audio_queue.get_nowait()

    async def _run_async_chunk_bridge_generation(
        self,
        input_stream: asyncio.Queue[list[int]],
    ) -> None:
        request_id = f"rt-{self.connection_id}-{uuid4()}"
        sent_audio = False
        audio_done_sent = False
        completed = False
        engine_request_started = False
        self._realtime_audio_ref = None

        try:
            audio = await self._collect_committed_audio()
            prompt = self._build_realtime_audio_prompt(
                audio,
                self._get_realtime_input_sample_rate(),
                getattr(self.serving, "model_cls", None),
            )

            result_gen = self.engine.generate(
                prompt=prompt,
                request_id=request_id,
                sampling_params_list=self._realtime_sampling_params_list(),
            )
            engine_request_started = True
            sent_audio, completed = await self._consume_generation_outputs(result_gen, input_stream)

            if completed and sent_audio:
                await self.send_json({"type": "response.audio.done", "has_audio": True})
                audio_done_sent = True
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Error in async_chunk bridge generation: %s", e)
            await self.send_error(str(e), "processing_error")
        finally:
            if engine_request_started and not completed:
                await self._abort_generation_request(request_id)
            if self._is_connected and not audio_done_sent:
                try:
                    await self.send_json({"type": "response.audio.done", "has_audio": sent_audio})
                except Exception:
                    logger.exception("Failed to send response.audio.done")
            self._drain_audio_queue()

    async def _run_generation(
        self,
        streaming_input_gen: AsyncGenerator,
        input_stream: asyncio.Queue[list[int]],
    ):
        request_id = f"rt-{self.connection_id}-{uuid4()}"
        sent_audio = False
        audio_done_sent = False
        completed = False
        engine_request_started = False
        self._realtime_audio_ref = None

        try:
            result_gen = self.engine.generate(
                prompt=streaming_input_gen,
                request_id=request_id,
                sampling_params_list=self._realtime_sampling_params_list(),
            )
            engine_request_started = True
            sent_audio, completed = await self._consume_generation_outputs(result_gen, input_stream)

            if completed and sent_audio:
                await self.send_json({"type": "response.audio.done", "has_audio": True})
                audio_done_sent = True
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Error in generation: %s", e)
            await self.send_error(str(e), "processing_error")
        finally:
            if engine_request_started and not completed:
                await self._abort_generation_request(request_id)
            if self._is_connected and not audio_done_sent:
                try:
                    await self.send_json({"type": "response.audio.done", "has_audio": sent_audio})
                except Exception:
                    logger.exception("Failed to send response.audio.done")
            self._drain_audio_queue()

    async def send_json(self, payload: dict):
        await self.websocket.send_text(json.dumps(payload))
