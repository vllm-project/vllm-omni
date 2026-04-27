# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved
"""Triton Python backend for Qwen3-TTS-NV driven by vllm-omni's AsyncOmni engine.

Wraps :class:`Qwen3TTSTalkerForConditionalGenerationNv` (the NV-flavoured AR
talker, same model used by ``benchmark_qwen3_tts_talker.py``) — the talker
emits codec tokens directly and we stream them out through the
``codec_decoder`` BLS model.

Pipeline:
  1. Build an ``additional_information`` dict from ``{task_type, text,
     language, speaker}`` and a placeholder ``prompt_token_ids`` of length
     ``prompt_len`` (estimated from the same talker class).
  2. Submit one request to ``AsyncOmni.generate()``; stream codec frames out
     as they arrive, chunk-decoding them through the ``codec_decoder`` BLS.
  3. Client receives a sequence of audio chunks @ 24 kHz, the last marked
     final.

Notes:
  * The NV talker's ``audio_codes`` multimodal output **contains the
    ``prompt_len`` prefill rows up front**; we slice them off before
    forwarding to the codec.
  * ``max_num_batched_tokens`` must be at least the longest expected
    ``prompt_len`` (otherwise prefill is chunked, hurting TTFT). It is
    plumbed straight through to the engine args.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import queue
import tempfile
import threading
import time
import uuid
from typing import Any

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
import yaml

logging.basicConfig(
    format="%(asctime)s [%(levelname)s]: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("qwen3_tts_triton")


def _require_param(parameters: dict, key: str) -> str:
    val = parameters.get(key)
    if val is None:
        raise KeyError(f"Missing required model parameter: {key!r}")
    if isinstance(val, dict):
        val = val.get("string_value")
    if val is None:
        raise KeyError(f"Missing required model parameter: {key!r}")
    return str(val)


class TritonPythonModel:

    def initialize(self, args):
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        self.model_config = json.loads(args["model_config"])
        params = self.model_config.get("parameters", {})

        self.vllm_model_path = _require_param(params, "vllm_model_path")
        self.default_speaker = _require_param(params, "default_speaker").lower()
        self.default_language = _require_param(params, "default_language")
        self.task_type = _require_param(params, "task_type")
        if self.task_type != "CustomVoice":
            raise ValueError(
                f"Qwen3-TTS NV talker only supports task_type='CustomVoice', "
                f"got {self.task_type!r}."
            )

        self.max_model_len = int(_require_param(params, "max_model_len"))
        self.max_num_seqs = int(_require_param(params, "max_num_seqs"))
        self.max_num_batched_tokens = int(_require_param(params, "max_num_batched_tokens"))
        self.max_new_tokens = int(_require_param(params, "max_new_tokens"))
        self.gpu_memory_utilization = float(_require_param(params, "gpu_memory_utilization"))

        self.codec_chunk_size = int(_require_param(params, "codec_chunk_size"))
        self.codec_left_context = int(_require_param(params, "codec_left_context"))
        self.first_chunk_frames = int(_require_param(params, "first_chunk_frames"))
        self.codec_codebook_size = int(_require_param(params, "codec_codebook_size"))

        self.sampling_temperature = float(_require_param(params, "sampling_temperature"))
        self.sampling_top_k = int(_require_param(params, "sampling_top_k"))
        self.sampling_repetition_penalty = float(_require_param(params, "sampling_repetition_penalty"))
        self.sampling_seed = int(_require_param(params, "sampling_seed"))
        self.sampling_stop_token_ids = [
            int(x) for x in _require_param(params, "sampling_stop_token_ids").split(",") if x.strip()
        ]

        self._samples_per_frame = int(24000 / 12.5)  # 12.5 fps codec
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()

        # Dedicated pool for per-request codec workers. Each in-flight request
        # holds one thread that serializes its own codec decode + response_sender.send
        # calls; default size matches max_num_seqs so every request can run a
        # codec call concurrently and let Triton dynamic batching kick in on the
        # codec_decoder model.
        self._codec_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, self.max_num_seqs),
            thread_name_prefix="qwen3_tts_codec",
        )

        self._load_prompt_builders()
        self._start_omni_engine()

        logger.info("Qwen3-TTS initialized (default_speaker=%s)", self.default_speaker)

    def _load_prompt_builders(self):
        from transformers import AutoTokenizer
        from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
            Qwen3TTSConfig,
        )
        from vllm_omni.model_executor.models.qwen3_tts_nv.qwen3_tts_talker_nv import (
            Qwen3TTSTalkerForConditionalGenerationNv,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.vllm_model_path, trust_remote_code=True, padding_side="left",
        )
        hf_cfg = Qwen3TTSConfig.from_pretrained(self.vllm_model_path, trust_remote_code=True)
        talker_cfg = getattr(hf_cfg, "talker_config", None)
        self._codec_language_id = getattr(talker_cfg, "codec_language_id", None)
        self._spk_is_dialect = getattr(talker_cfg, "spk_is_dialect", None)
        self._estimate_prompt_len = (
            Qwen3TTSTalkerForConditionalGenerationNv.estimate_prompt_len_from_additional_information
        )

    def _build_stage_config_file(self) -> str:
        stage_cfg = {
            "stage_args": [
                {
                    "stage_id": 0,
                    "stage_type": "llm",
                    "is_comprehension": True,
                    "final_output": True,
                    "final_output_type": "latent",
                    "runtime": {"devices": "0"},
                    "engine_args": {
                        "model_stage": "qwen3_tts",
                        "max_num_seqs": self.max_num_seqs,
                        "model_arch": "Qwen3TTSTalkerForConditionalGenerationNv",
                        "worker_type": "ar",
                        "scheduler_cls": "vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler",
                        "enforce_eager": False,
                        "trust_remote_code": True,
                        "async_scheduling": True,
                        "enable_prefix_caching": False,
                        "engine_output_type": "latent",
                        "gpu_memory_utilization": self.gpu_memory_utilization,
                        "distributed_executor_backend": "mp",
                        "max_num_batched_tokens": self.max_num_batched_tokens,
                        "max_model_len": self.max_model_len,
                    },
                    "default_sampling_params": {
                        "temperature": self.sampling_temperature,
                        "top_k": self.sampling_top_k,
                        "max_tokens": self.max_new_tokens,
                        "seed": self.sampling_seed,
                        "detokenize": False,
                        "repetition_penalty": self.sampling_repetition_penalty,
                        "stop_token_ids": self.sampling_stop_token_ids,
                    },
                }
            ],
        }
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix="qwen3_tts_triton_", delete=False,
        )
        yaml.dump(stage_cfg, tmp, sort_keys=False)
        tmp.close()
        return tmp.name

    def _start_omni_engine(self):
        from vllm_omni import AsyncOmni

        self._stage_cfg_path = self._build_stage_config_file()
        self.omni = AsyncOmni(
            model=self.vllm_model_path,
            stage_configs_path=self._stage_cfg_path,
            log_stats=False,
            stage_init_timeout=300,
        )

    def _build_prompt(self, text: str, language: str, speaker: str) -> tuple[dict, int]:
        """Return (engine_input, prompt_len).

        The NV talker only takes ``{task_type, text, language, speaker}``
        in ``additional_information``. ``prompt_len`` is the placeholder
        prefill length that will appear at the front of the streamed
        ``audio_codes`` tensor and must be sliced off before codec decode.
        """
        additional_information = {
            "task_type": [self.task_type],
            "text": [text],
            "language": [language],
            "speaker": [speaker],
        }
        prompt_len = self._estimate_prompt_len(
            additional_information=additional_information,
            task_type=self.task_type,
            tokenize_prompt=lambda t: self.tokenizer(t, padding=False)["input_ids"],
            codec_language_id=self._codec_language_id,
            spk_is_dialect=self._spk_is_dialect,
        )
        if prompt_len > self.max_num_batched_tokens:
            logger.warning(
                "prompt_len=%d exceeds max_num_batched_tokens=%d; "
                "prefill will be chunked which hurts TTFT.",
                prompt_len, self.max_num_batched_tokens,
            )
        prompt = {
            "prompt_token_ids": [0] * prompt_len,
            "additional_information": additional_information,
        }
        return prompt, prompt_len

    def _normalize_codes(self, mm_codes: Any, prompt_len: int) -> torch.Tensor | None:
        """Concatenate streamed ``audio_codes`` tensors, drop the prefill
        ``prompt_len`` rows the NV talker prepends, then mask out invalid
        rows (zeros / out-of-codebook stop tokens) before codec decode."""
        if mm_codes is None:
            return None
        if isinstance(mm_codes, list):
            if not mm_codes:
                return None
            codes = torch.cat(mm_codes, dim=0)
        else:
            codes = mm_codes
        codes = codes.to(torch.long).cpu()
        if prompt_len > 0:
            if codes.shape[0] <= prompt_len:
                return None
            codes = codes[prompt_len:]
        valid = codes.any(dim=1) & (codes.max(dim=1).values < self.codec_codebook_size)
        return codes[valid]

    def _decode_codec(self, codes: torch.Tensor, left_context_frames: int) -> np.ndarray:
        codes_np = codes.numpy().astype(np.int64)
        pad = self.codec_chunk_size - codes_np.shape[0]
        if pad > 0:
            codes_np = np.pad(codes_np, ((0, pad), (0, 0)))

        response = pb_utils.InferenceRequest(
            model_name="codec_decoder",
            requested_output_names=["audio_values"],
            inputs=[pb_utils.Tensor("audio_codes", codes_np[np.newaxis])],
        ).exec()
        if response.has_error():
            raise RuntimeError(f"Codec decode failed: {response.error().message()}")

        audio_tensor = pb_utils.get_output_tensor_by_name(response, "audio_values")
        audio = (audio_tensor.as_numpy() if audio_tensor.is_cpu()
                 else torch.from_dlpack(audio_tensor.to_dlpack()).cpu().numpy())
        if audio.ndim > 1:
            audio = audio[0]

        left = left_context_frames * self._samples_per_frame
        right = pad * self._samples_per_frame
        return audio[left:-right] if right > 0 else audio[left:]

    def _send_audio(self, response_sender, audio: np.ndarray, final: bool):
        response_sender.send(
            pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor("audio", audio.astype(np.float32))]),
            flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL if final else 0,
        )

    def _send_error(self, response_sender, err: Exception):
        try:
            response_sender.send(
                pb_utils.InferenceResponse(output_tensors=[], error=pb_utils.TritonError(str(err))),
                flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
            )
        except Exception:
            pass

    def _codec_worker(self, codec_q: "queue.Queue", response_sender, state: dict) -> None:
        """Per-request worker. Pops (chunk, ctx, is_final) tuples; ``None`` is a
        sentinel meaning "send empty final response and exit". Runs on a thread
        from ``self._codec_pool`` so codec decode + sender.send don't block the
        asyncio loop, and so it can overlap with vLLM token generation for the
        same request."""
        finalized = False
        try:
            while True:
                item = codec_q.get()
                if item is None:
                    self._send_audio(
                        response_sender, np.array([], dtype=np.float32), final=True,
                    )
                    finalized = True
                    return
                chunk, ctx, is_final = item
                audio = self._decode_codec(chunk, ctx)
                self._send_audio(response_sender, audio, final=is_final)
                if state["t_first_audio"] is None:
                    state["t_first_audio"] = time.perf_counter()
                if is_final:
                    finalized = True
                    return
        except Exception as e:
            state["error"] = e
            if not finalized:
                self._send_error(response_sender, e)

    async def _synthesize(self, text: str, language: str, speaker: str, response_sender):
        t_start = time.perf_counter()
        request_id = f"tts-{uuid.uuid4().hex[:8]}"
        prompt, prompt_len = self._build_prompt(text, language, speaker)

        sent_frames = 0
        all_codes: torch.Tensor | None = None

        codec_q: queue.Queue = queue.Queue()
        state: dict = {"t_first_audio": None, "error": None}
        codec_future = self._codec_pool.submit(
            self._codec_worker, codec_q, response_sender, state,
        )

        def dispatch_chunk(codes: torch.Tensor, new_frames: int, is_final: bool) -> None:
            nonlocal sent_frames
            ctx = min(sent_frames, self.codec_left_context)
            # clone() so the worker thread owns its data and the engine can
            # overwrite/mutate the underlying buffer freely.
            chunk = codes[sent_frames - ctx: sent_frames + new_frames].clone()
            codec_q.put((chunk, ctx, is_final))
            sent_frames += new_frames

        try:
            async for out in self.omni.generate(prompt, request_id=request_id):
                if state["error"] is not None:
                    break
                codes = self._normalize_codes(
                    out.multimodal_output.get("audio_codes"), prompt_len,
                )
                if codes is None:
                    continue
                all_codes = codes
                threshold = (self.first_chunk_frames if sent_frames == 0
                             else self.codec_chunk_size - self.codec_left_context)
                new_frames = codes.shape[0] - sent_frames
                while new_frames >= threshold:
                    dispatch_chunk(codes, threshold, is_final=False)
                    new_frames = codes.shape[0] - sent_frames
                    threshold = self.codec_chunk_size - self.codec_left_context

            # Final trailing chunk (or empty-final sentinel)
            if state["error"] is None:
                if all_codes is not None and all_codes.shape[0] > sent_frames:
                    remaining = all_codes.shape[0] - sent_frames
                    dispatch_chunk(all_codes, remaining, is_final=True)
                else:
                    codec_q.put(None)

            # Wait for the worker to drain and send the final response without
            # blocking the asyncio loop thread.
            await asyncio.wrap_future(codec_future)

            if state["error"] is not None:
                raise state["error"]

            total_frames = all_codes.shape[0] if all_codes is not None else 0
            t_end = time.perf_counter()
            ttfa_ms = ((state["t_first_audio"] or t_end) - t_start) * 1000
            logger.info(
                "rid=%s ttfa=%.1fms total=%.1fms frames=%d speaker=%s text=%r",
                request_id, ttfa_ms, (t_end - t_start) * 1000,
                total_frames, speaker, text[:120],
            )
        except Exception as e:
            logger.error("rid=%s failed: %s", request_id, e, exc_info=True)
            try:
                await self.omni.abort(request_id)
            except Exception:
                pass
            # Make sure the worker exits if it's still alive.
            if not codec_future.done():
                codec_q.put(None)
                try:
                    await asyncio.wrap_future(codec_future)
                except Exception:
                    pass
            self._send_error(response_sender, e)

    def execute(self, requests):
        for request in requests:
            response_sender = request.get_response_sender()
            try:
                text = pb_utils.get_input_tensor_by_name(request, "text").as_numpy().flatten()[0].decode("utf-8")
                lang_tensor = pb_utils.get_input_tensor_by_name(request, "language")
                language = (lang_tensor.as_numpy().flatten()[0].decode("utf-8")
                            if lang_tensor is not None else self.default_language)
                spk_tensor = pb_utils.get_input_tensor_by_name(request, "speaker")
                speaker = (spk_tensor.as_numpy().flatten()[0].decode("utf-8")
                           if spk_tensor is not None else self.default_speaker).lower()
                asyncio.run_coroutine_threadsafe(
                    self._synthesize(text, language, speaker, response_sender), self._loop,
                )
            except Exception as e:
                logger.error("Request parse failed: %s", e, exc_info=True)
                self._send_error(response_sender, e)
        return None

    def finalize(self):
        if hasattr(self, "omni"):
            try:
                self.omni.shutdown()
            except Exception:
                pass
        if hasattr(self, "_loop") and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if hasattr(self, "_loop_thread"):
            self._loop_thread.join(timeout=10)
        if hasattr(self, "_codec_pool"):
            self._codec_pool.shutdown(wait=False)
        if getattr(self, "_stage_cfg_path", None):
            try:
                os.unlink(self._stage_cfg_path)
            except OSError:
                pass
