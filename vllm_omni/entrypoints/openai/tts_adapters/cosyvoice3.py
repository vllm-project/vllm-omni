# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CosyVoice3 TTS serving adapter."""

import asyncio
import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest
from vllm_omni.model_executor.models.cosyvoice3.runtime import cosyvoice3_standard_sampling

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
    from vllm_omni.utils.shared_reference_encoder import ReferenceEncoderPool

logger = init_logger(__name__)

# Without this delimiter the talker re-speaks the reference transcript instead
# of emitting target-only speech (issue #4644). Matches the offline example and
# upstream demo.
_PROMPT_DELIMITER = "<|endofprompt|>"
_PROMPT_PREFIX = f"You are a helpful assistant.{_PROMPT_DELIMITER}"


@register_tts_adapter
class CosyVoice3Adapter(ARTTSAdapter):
    stage_keys = frozenset({"cosyvoice3_talker"})
    name = "cosyvoice3"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        # Cache the resolved Qwen tokenizer used for dynamic-token sizing.
        # Rebuilding it would repeat snapshot resolution and add ~100 ms to the
        # TTFP-critical sampling override on every request.
        self._tokenizer = None
        # Cold references are resampled and conditioned here, in parallel
        # workers, instead of serially inside the input processor.
        self._reference_executor: ThreadPoolExecutor | None = None
        self._reference_resampler = None
        self._shared_reference_pool: ReferenceEncoderPool | None = None
        self._closed = False
        stage = getattr(ctx.server, "_tts_stage", None)
        additional_config: Mapping[str, object] = {}
        for owner in (getattr(stage, "runtime_config", None), getattr(stage, "engine_args", None)):
            value = getattr(owner, "additional_config", None)
            if isinstance(value, Mapping):
                additional_config = value
                break
        self._shared_reference = additional_config.get("shared_reference_encoder", False)
        if not isinstance(self._shared_reference, bool):
            raise ValueError("Cosy shared_reference_encoder must be a boolean")

    async def _start_shared_reference_encoder(self) -> "ReferenceEncoderPool":
        from vllm_omni.utils.shared_reference_encoder import (
            ReferenceEncoderPool,
            ReferenceEncoderSpec,
            socket_path_fits,
        )

        if self._closed:
            raise RuntimeError("Cosy reference encoder adapter is closed")
        # Construction has no await; the pool owns concurrent readiness waits.
        if self._shared_reference_pool is None:
            server = self.ctx.server
            shared_dir = getattr(server, "api_server_shared_dir", None)
            if shared_dir and not socket_path_fits(os.path.join(shared_dir, "cosyvoice3-ref")):
                raise ValueError("Cosy shared reference directory is too long for a Unix socket")
            module = "vllm_omni.model_executor.models.cosyvoice3.reference_encoder"
            spec = ReferenceEncoderSpec(
                name="cosyvoice3-ref",
                factory=f"{module}:CosyVoice3ReferenceEncoder",
                kwargs={
                    "model_path": server.model_config.model,
                    "config": server.model_config.hf_config.to_dict(),
                    "cache_bytes": server.speech_cache_config.speaker_max_bytes,
                },
                wire=f"{module}:COSY_REFERENCE_WIRE",
                namespace=f"cosyvoice3:{server.model_config.model}",
                nbytes=lambda artifact: artifact.nbytes,
                workers=1,
                # Keep the authoritative artifact cache in the one encoder,
                # not a duplicate cache in every frontend process.
                cache_bytes=0,
            )
            self._shared_reference_pool = ReferenceEncoderPool(spec, shared_dir)
        await self._shared_reference_pool.start()
        return self._shared_reference_pool

    async def warmup(self) -> None:
        if self._shared_reference:
            await self._start_shared_reference_encoder()

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._shared_reference_pool is not None:
            self._shared_reference_pool.close()
        if self._reference_executor is not None:
            self._reference_executor.shutdown(wait=False, cancel_futures=True)

    async def _prefetch_reference(self, wav: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        """Resample to the processor rate and warm its content-addressed cache.

        Uses the processor's own resampler, so the processor receives exactly
        these samples (its resampling becomes a no-op) and its cache lookup for
        the same bytes hits instead of recomputing on the serving event loop.
        """
        from vllm.multimodal.parse import MultiModalDataParser

        from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import prefetch_reference_conditioning

        server = self.ctx.server
        hf_cfg = server.model_config.hf_config
        target_sr = getattr(hf_cfg, "target_sr", None)
        if target_sr is None:
            return wav, sr
        if self._reference_executor is None:
            workers = int(os.environ.get("COSYVOICE3_REFERENCE_WORKERS", "16"))
            self._reference_executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cosyvoice3-ref")
            self._reference_resampler = MultiModalDataParser(target_sr=target_sr).audio_resampler
        model_dir = server.model_config.model
        resampler = self._reference_resampler
        assert resampler is not None

        def prepare() -> np.ndarray:
            audio = resampler.resample(wav, orig_sr=sr)
            prefetch_reference_conditioning(model_dir, hf_cfg, (audio, target_sr))
            return audio

        audio = await asyncio.get_running_loop().run_in_executor(self._reference_executor, prepare)
        return audio, int(target_sr)

    async def _build_prompt(
        self, request: "OpenAICreateSpeechRequest", *, has_inline_ref_audio: bool = False
    ) -> dict[str, Any]:
        """Build the multimodal CosyVoice3 voice-cloning prompt."""
        server = self.ctx.server
        wav_samples, sr, _ = await server._resolve_ref_audio_array(request.ref_audio)
        ref_text = request.ref_text or ""
        if _PROMPT_DELIMITER not in ref_text:
            ref_text = f"{_PROMPT_PREFIX}{ref_text}"
        mm_kwargs: dict[str, Any] = {"prompt_text": ref_text, "sample_rate": sr}
        if request.voice:
            voice_lower = request.voice.lower()
            if voice_lower in server.uploaded_speakers and not has_inline_ref_audio:
                mm_kwargs["voice_name"] = voice_lower
                mm_kwargs["voice_created_at"] = server._voice_created_at(voice_lower)
        wav = np.array(wav_samples, dtype=np.float32, copy=True)
        if self._shared_reference:
            from vllm_omni.model_executor.models.cosyvoice3.reference_encoder import REFERENCE_CONDITIONING_KEY

            pool = await self._start_shared_reference_encoder()
            artifact = await pool.get(wav, sr)
            mm_kwargs[REFERENCE_CONDITIONING_KEY] = artifact.processor_kwargs()
        elif "voice_name" not in mm_kwargs and os.environ.get("COSYVOICE3_REFERENCE_PREFETCH", "1") != "0":
            wav, sr = await self._prefetch_reference(wav, sr)
            mm_kwargs["sample_rate"] = sr
        return {
            "prompt": request.input,
            "multi_modal_data": {"audio": (wav, sr)},
            "mm_processor_kwargs": mm_kwargs,
        }

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        """Validate CosyVoice3 request parameters. Returns error message or None."""
        server = self.ctx.server
        err = server._apply_uploaded_speaker(request)
        if err:
            return err
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        # CosyVoice3 requires reference audio for voice cloning
        if request.ref_audio is None:
            return "CosyVoice3 requires 'ref_audio' (reference audio for voice cloning)"

        fmt_err = server._validate_ref_audio_format(request.ref_audio)
        if fmt_err:
            return fmt_err

        if not request.ref_text or not request.ref_text.strip():
            return "CosyVoice3 requires 'ref_text' (transcript of the reference audio)"

        return self._validate_max_new_tokens(request)

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        prompt = await self._build_prompt(request, has_inline_ref_audio=has_inline_ref_audio)
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="cosyvoice3")

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: "OpenAICreateSpeechRequest",
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        """Set dynamic min/max tokens based on tokenized text length.

        The official model uses ``min_token_text_ratio`` to prevent early EOS
        and ``max_token_text_ratio`` to cap the generated token count.
        """
        import copy

        from vllm_omni.model_executor.models.cosyvoice3.tokenizer import get_qwen_tokenizer
        from vllm_omni.model_executor.models.cosyvoice3.utils import extract_text_token

        server = self.ctx.server
        sampling_params_list = copy.deepcopy(sampling_params_list)
        hf_cfg = server.model_config.hf_config
        if cosyvoice3_standard_sampling(hf_cfg):
            first_control = int(hf_cfg.llm["speech_token_size"])
            controls = list(range(first_control, first_control + 200))
            sampling_params_list[0].stop_token_ids = controls
            sampling_params_list[0].all_stop_token_ids.update(controls)
        # Build the Qwen tokenizer once per process (resolving the model dir via
        # snapshot_download at most once) and reuse it across requests.
        tokenizer = self._tokenizer
        if tokenizer is None:
            model_path = server.engine_client.model_config.model
            if not os.path.isdir(model_path):
                from vllm_omni.transformers_utils.repo_utils import hf_api

                model_path = hf_api().snapshot_download(model_path)
            tokenizer = get_qwen_tokenizer(
                token_path=os.path.join(model_path, hf_cfg.qwen_pretrain_path),
                skip_special_tokens=hf_cfg.skip_special_tokens,
                version=hf_cfg.version,
            )
            self._tokenizer = tokenizer
        _, text_token_len = extract_text_token(
            request.input,
            tokenizer,
            hf_cfg.allowed_special,
        )
        min_ratio = getattr(hf_cfg, "min_token_text_ratio", 2)
        max_ratio = getattr(hf_cfg, "max_token_text_ratio", 20)
        sampling_params_list[0].min_tokens = max(1, int(text_token_len * min_ratio))
        sampling_params_list[0].max_tokens = min(2048, int(text_token_len * max_ratio))

        if request.max_new_tokens is not None:
            sampling_params_list[0].max_tokens = request.max_new_tokens
            sampling_params_list[0].min_tokens = min(
                getattr(sampling_params_list[0], "min_tokens", 0),
                request.max_new_tokens,
            )
        logger.info(
            "CosyVoice3 dynamic tokens: text_tokens=%d, min_tokens=%d, max_tokens=%d",
            text_token_len,
            sampling_params_list[0].min_tokens,
            sampling_params_list[0].max_tokens,
        )
        return sampling_params_list
