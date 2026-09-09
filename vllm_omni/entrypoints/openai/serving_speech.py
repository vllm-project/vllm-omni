# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import base64
import hashlib
import io
import json
import math
import os
import re
import struct
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

import numpy as np
import soundfile as sf
import torch
from fastapi import HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from vllm.entrypoints.generate.base.serving import GenerateBaseServing as OpenAIServing
from vllm.entrypoints.launcher import terminate_if_errored
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    RequestResponseMetadata,
)
from vllm.logger import init_logger
from vllm.multimodal.media import MediaConnector
from vllm.utils import random_uuid
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError

from vllm_omni.entrypoints.openai.audio_utils_mixin import AudioMixin, StreamingAudioResampler
from vllm_omni.entrypoints.openai.protocol.audio import (
    AudioResponse,
    BatchSpeechRequest,
    BatchSpeechResponse,
    CreateAudio,
    OpenAICreateSpeechRequest,
    SpeechBatchItem,
    SpeechBatchItemResult,
    SpeechInputTokenDetails,
    SpeechTokenUsage,
)
from vllm_omni.entrypoints.openai.speech_usage import (
    SpeechOutputTokenCounter,
    build_speech_usage,
    qwen3_tts_input_token_details,
)
from vllm_omni.entrypoints.openai.tts_adapters import (
    SpeechServingContext,
    TTSGenerationError,
    all_tts_stage_keys,
    detect_tts_model_type,
    resolve_adapter,
    tts_entry_stage_archs,
)
from vllm_omni.entrypoints.utils import coerce_param_message_types
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.utils.speaker_cache import get_speaker_cache

logger = init_logger(__name__)


_SPEECH_USAGE_INPUT_TOKENS_HEADER = "X-VLLM-OMNI-INPUT-TOKENS"
_SPEECH_USAGE_OUTPUT_TOKENS_HEADER = "X-VLLM-OMNI-OUTPUT-TOKENS"
_SPEECH_USAGE_TOTAL_TOKENS_HEADER = "X-VLLM-OMNI-TOTAL-TOKENS"
_SPEECH_USAGE_INPUT_TEXT_TOKENS_HEADER = "X-VLLM-OMNI-INPUT-TEXT-TOKENS"
_SPEECH_USAGE_INPUT_AUDIO_TOKENS_HEADER = "X-VLLM-OMNI-INPUT-AUDIO-TOKENS"

# TTS Configuration
#
# The stage-key -> model-type mapping is NOT declared here: it is derived from
# the ``stage_keys`` / ``model_archs`` each adapter declares, via
# ``tts_adapters.detect_tts_model_type``. Adding a TTS model must not require an
# edit to this module.
#
# Audex contract: zero-codec / invalid generations arrive as empty terminal
# payloads and must fail the request, never serialize as a successful empty
# WAV. Covers both the TTS ("audex") and TTA ("audex_tta") pipelines.
_AUDEX_NO_AUDIO_GUARD_MODEL_TYPES = frozenset({"audex", "audex_tta"})
_REF_AUDIO_MIN_DURATION = 1.0  # seconds
_REF_AUDIO_MAX_DURATION = 30.0  # seconds
_REF_AUDIO_METADATA_FETCH_ATTEMPTS = 3
_REMOTE_REF_AUDIO_SCHEMES = frozenset({"http", "https", "data"})
_REF_AUDIO_RESOLVE_CACHE_MAX_ENTRIES = 256
_REF_AUDIO_RESOLVE_CACHE_MAX_BYTES = 256 * 1024 * 1024
_TTS_MAX_INSTRUCTIONS_LENGTH = 500
_DEFAULT_VOICE_NAME = "default"


def _is_default_voice(voice, supported_speakers):
    """Check if a lowercased voice name is the placeholder default and not
    an actual registered/built-in speaker."""
    return voice == _DEFAULT_VOICE_NAME and voice not in supported_speakers


def _create_wav_header(sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Create a WAV header with placeholder size values for streaming.

    Uses 0xFFFFFFFF as placeholder for data size fields, which is accepted
    by most audio clients and matches OpenAI's streaming WAV implementation.

    Args:
        sample_rate: Audio sample rate in Hz
        num_channels: Number of audio channels (1 for mono, 2 for stereo)
        bits_per_sample: Bits per sample (typically 16)

    Returns:
        44-byte WAV header as bytes
    """
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8

    # Use 0xFFFFFFFF as placeholder for unknown size (streaming)
    placeholder_size = 0xFFFFFFFF

    # ref https://docs.fileformat.com/audio/wav/
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",  # ChunkID
        placeholder_size,  # ChunkSize (placeholder)
        b"WAVE",  # Format
        b"fmt ",  # Subchunk1ID
        16,  # Subchunk1Size (16 for PCM)
        1,  # AudioFormat (1 for PCM)
        num_channels,  # NumChannels
        sample_rate,  # SampleRate
        byte_rate,  # ByteRate
        block_align,  # BlockAlign
        bits_per_sample,  # BitsPerSample
        b"data",  # Subchunk2ID
        placeholder_size,  # Subchunk2Size (placeholder)
    )

    return header


def _infer_audio_num_channels(audio: np.ndarray) -> int:
    """Infer channel count before streaming PCM bytes are wrapped as WAV."""
    if audio.ndim == 3 and audio.shape[0] == 1:
        audio = audio[0]
    if audio.ndim == 2:
        if audio.shape[0] in (1, 2):
            return int(audio.shape[0])
        if audio.shape[1] in (1, 2):
            return int(audio.shape[1])
    return 1


def _sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks.

    Only allows alphanumeric characters, underscores, hyphens, and dots.
    Replaces any other characters with underscores.
    """
    # Remove any path components
    filename = os.path.basename(filename)
    # Replace any non-alphanumeric, underscore, hyphen, or dot with underscore
    sanitized = re.sub(r"[^a-zA-Z0-9_.\-]", "_", filename)
    # Ensure filename is not empty
    if not sanitized:
        sanitized = "file"
    # Limit length to prevent potential issues
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    return sanitized


def _validate_speaker_name(name: str) -> str:
    """Trim and reject empty / path-separator / NUL / reserved voice names."""
    trimmed = (name or "").strip()
    if not trimmed or trimmed in (".", "..") or any(c in trimmed for c in "/\\\x00"):
        raise ValueError(f"Invalid voice name {name!r}: must be non-empty, no path separators or NUL")
    return trimmed


def _validate_path_within_directory(file_path: Path, directory: Path) -> bool:
    """Validate that file_path is within the specified directory.

    Prevents path traversal attacks by ensuring the resolved path
    is within the target directory.
    """
    try:
        # Resolve both paths to absolute paths
        file_path_resolved = file_path.resolve()
        directory_resolved = directory.resolve()
        # Check if file_path is within directory
        return directory_resolved in file_path_resolved.parents or directory_resolved == file_path_resolved
    except Exception:
        return False


class OmniOpenAIServingSpeech(OpenAIServing, AudioMixin):
    _diffusion_mode: bool = False
    _media_connector: MediaConnector | None = None
    _allowed_local_media_path: str = ""
    _tts_executor: ThreadPoolExecutor | None = None

    def _init_speaker_storage(self) -> None:
        """Initialize speaker storage + cache, restoring any persisted uploads."""
        speaker_samples_dir = os.environ.get("SPEAKER_SAMPLES_DIR", os.path.expanduser("~/.cache/vllm-omni/speakers"))
        self.uploaded_speakers_dir = Path(speaker_samples_dir).expanduser()
        self.uploaded_speakers_dir.mkdir(parents=True, exist_ok=True)
        _raw_cap = os.environ.get("SPEAKER_MAX_UPLOADED", "")
        try:
            self._max_uploaded_speakers = int(_raw_cap) if _raw_cap else 1000
        except ValueError:
            logger.warning("Invalid SPEAKER_MAX_UPLOADED=%r; using default 1000", _raw_cap)
            self._max_uploaded_speakers = 1000
        self.uploaded_speakers: dict[str, dict[str, Any]] = {}
        self._ref_audio_data_url_cache: dict[str, str] = {}
        self._ref_audio_resolve_cache: OrderedDict[str, tuple[list[float], int, int, str]] = OrderedDict()
        self._ref_audio_resolve_cache_bytes = 0
        self._ref_audio_resolve_cache_max_entries = _REF_AUDIO_RESOLVE_CACHE_MAX_ENTRIES
        self._ref_audio_resolve_cache_max_bytes = _REF_AUDIO_RESOLVE_CACHE_MAX_BYTES
        # Readiness is keyed by (artifact_key, x_vector_only). An x-vector-only
        # request caches a speaker embedding but no ref_code, so its artifact
        # must not satisfy a later ICL request that needs ref_code (#5049).
        self._ref_audio_model_artifact_ready: set[tuple[str, bool]] = set()
        self._request_ref_audio_artifact_keys: dict[str, tuple[str, bool]] = {}
        self._speaker_cache = get_speaker_cache()
        self._last_upload_ts = 0
        self._upload_lock = asyncio.Lock()
        self._restore_uploaded_speakers()
        logger.info(
            "Speaker storage: dir=%s, max_speakers=%d, restored=%d",
            self.uploaded_speakers_dir,
            self._max_uploaded_speakers,
            len(self.uploaded_speakers),
        )

    def _next_upload_timestamp(self) -> int:
        ts = max(int(time.time()), self._last_upload_ts + 1)
        self._last_upload_ts = ts
        return ts

    _META_SCALAR_INT_KEYS: tuple[str, ...] = (
        "created_at",
        "file_size",
        "sample_rate",
        "embedding_dim",
    )

    @classmethod
    def _speaker_metadata_to_header(cls, speaker_data: dict[str, Any]) -> dict[str, str]:
        """Serialize a speaker_data dict into safetensors' ``dict[str, str]`` header."""
        header: dict[str, str] = {}
        for k, v in speaker_data.items():
            if v is None:
                continue
            # file_path is re-derived from the path on load; don't persist it.
            if k == "file_path":
                continue
            header[k] = str(v)
        return header

    @classmethod
    def _speaker_metadata_from_header(cls, header: dict[str, str], file_path: str) -> dict[str, Any]:
        """Reverse of :meth:`_speaker_metadata_to_header`: coerce ints back and re-inject file_path."""
        data: dict[str, Any] = dict(header)
        for k in cls._META_SCALAR_INT_KEYS:
            if k in data:
                try:
                    data[k] = int(data[k])
                except ValueError:
                    logger.warning(
                        "Speaker metadata %r in %s is not a valid int (got %r); leaving as string",
                        k,
                        file_path,
                        data[k],
                    )
        data["file_path"] = file_path
        return data

    def _restore_uploaded_speakers(self) -> None:
        """Scan ``uploaded_speakers_dir`` for safetensors files and rebuild state."""
        try:
            from safetensors import safe_open
        except ImportError:
            logger.warning("safetensors unavailable; uploaded voices will not persist across restarts")
            return

        restored = 0
        for path in sorted(self.uploaded_speakers_dir.glob("*.safetensors")):
            try:
                with safe_open(str(path), framework="pt") as f:
                    header = dict(f.metadata() or {})
            except Exception as e:
                logger.warning("Could not read voice file %s: %s", path, e)
                continue
            voice_name_lower = header.get("voice_name_lower") or header.get("name", "").lower()
            if not voice_name_lower:
                logger.warning("Voice file %s has no voice name in metadata; skipping", path)
                continue
            speaker_data = self._speaker_metadata_from_header(header, str(path))
            speaker_data.setdefault("name", voice_name_lower)
            speaker_data.setdefault("file_size", int(path.stat().st_size))
            self.uploaded_speakers[voice_name_lower] = speaker_data
            self._last_upload_ts = max(self._last_upload_ts, int(speaker_data.get("created_at", 0)))
            restored += 1
        if restored:
            logger.info("Restored %d uploaded voice(s) from %s", restored, self.uploaded_speakers_dir)

    @classmethod
    def for_diffusion(
        cls,
        diffusion_engine: "Any",
        model_name: str,
        stage_configs: "list[Any] | None" = None,
        allowed_local_media_path: str = "",
        allowed_media_domains: list[str] | None = None,
    ) -> "OmniOpenAIServingSpeech":
        """Create a speech serving instance for pure diffusion TTS models.

        Bypasses OpenAIServing.__init__ which requires a fully configured
        engine client that pure diffusion engines don't provide.
        """
        instance = cls.__new__(cls)
        instance._diffusion_mode = True
        instance._diffusion_engine = diffusion_engine
        instance._diffusion_model_name = model_name
        instance._diffusion_stage_configs = stage_configs
        instance._allowed_local_media_path = allowed_local_media_path
        instance._media_connector = MediaConnector(
            allowed_local_media_path=allowed_local_media_path,
            allowed_media_domains=allowed_media_domains,
        )
        instance._tts_model_type = "omnivoice"
        instance._is_tts = False
        # Diffusion-only instances don't have a TTS stage; set None so any
        # ``_is_tts_model()`` / ``_tts_stage`` access doesn't raise AttributeError.
        instance._tts_stage = None
        instance._adapter = None
        instance._init_speaker_storage()
        return instance

    def __init__(self, *args, **kwargs):
        self._media_connector = None
        self._allowed_local_media_path = ""
        self.model_name = kwargs.pop("model_name", None)
        # True when the server was launched with --forced-aligner (a pooling
        # aligner stage is appended to the pipeline). Gates word_timestamps.
        self.forced_aligner_enabled: bool = bool(kwargs.pop("forced_aligner_enabled", False))
        super().__init__(*args, **kwargs)
        self._init_speaker_storage()

        # Find and cache the TTS stage (if any) during initialization
        self._tts_stage = self._find_tts_stage()
        self._is_tts = self._tts_stage is not None

        # Determine TTS model type or None
        self._tts_model_type = self._detect_tts_model_type()

        # Shared executor for blocking adapter preprocessing. It must exist
        # before adapter construction so adapters can create their make_async
        # wrappers during their own lifecycle initialization.
        self._tts_executor = ThreadPoolExecutor(max_workers=1)
        # Resolve the per-model serving adapter (RFC #4327), keyed on the
        # detected model-type. Every dedicated TTS model has an adapter; the
        # adapter owns request validation, prompt/param building, capability
        # metadata, and sampling overrides. The model-type label remains in the
        # orchestrator for compatibility during this incremental migration.
        self._adapter = None
        if self._tts_stage is not None:
            adapter_cls = resolve_adapter(self._tts_model_type)
            if adapter_cls is not None:
                ctx = SpeechServingContext(server=self, engine_client=self.engine_client)
                self._adapter = adapter_cls(ctx)
                logger.info("Resolved TTS serving adapter: %s", adapter_cls.__name__)

        adapter = self._adapter
        if adapter is not None:
            adapter.load_capabilities()
        available_speakers = self._get_available_speakers()
        logger.info("Loaded %d supported speakers: %s", len(available_speakers), sorted(available_speakers))

        # Cache TTS configuration values (computed once, reused per request)
        self._max_instructions_length = self._compute_max_instructions_length()

        self._tts_tokenizer = None

        # Batch configuration
        self._batch_max_items: int = getattr(self.engine_client, "tts_batch_max_items", 32)

    def _get_tts_adapter(self):
        """Return the per-model serving adapter for the current ``_tts_model_type``.

        Pure-diffusion speech uses its dedicated request path and does not
        resolve adapters for AR-stage TTS models.

        Resolved lazily (rebuilt if ``_tts_model_type`` changed since the cached
        instance was built) so callers that set ``_tts_model_type`` after
        construction still dispatch to the matching adapter. In production
        ``_tts_model_type`` is fixed at init, so the cached instance is reused.
        """
        if self._diffusion_mode:
            return None

        adapter_cls = resolve_adapter(self._tts_model_type)
        if adapter_cls is None:
            self._adapter = None
            return None
        if self._adapter is None or type(self._adapter) is not adapter_cls:
            ctx = SpeechServingContext(server=self, engine_client=self.engine_client)
            self._adapter = adapter_cls(ctx)
            self._adapter.load_capabilities()
        return self._adapter

    def _uses_native_speed_control(self) -> bool:
        adapter = self._get_tts_adapter()
        return bool(adapter is not None and adapter.native_speed_control)

    def _get_available_speakers(self) -> set[str]:
        """Return all built-in, precomputed, and runtime-uploaded speakers."""
        available_speakers = set(self.uploaded_speakers)
        if self._adapter is not None:
            available_speakers.update(self._adapter.capabilities.supported_speakers)
            available_speakers.update(self._adapter.capabilities.precomputed_speakers)
        return available_speakers

    def _audio_encode_speed(self, request: OpenAICreateSpeechRequest) -> float:
        if self._uses_native_speed_control():
            return 1.0
        return float(request.speed or 1.0)

    async def warmup(self) -> None:
        """Run model-specific startup warmup through the resolved adapter.

        Warmup requirements differ by model. For example, Qwen3-TTS can warm up
        its standalone tokenizer decoder during model initialization, while
        VoxCPM2 requires a real inference request with an active vLLM
        ``ForwardContext``. The adapter owns that model-specific lifecycle logic.
        """
        if self._adapter is None:
            return
        await self._adapter.warmup()

    def shutdown(self) -> None:
        """Shut down the TTS thread pool executor."""
        if self._tts_executor is not None:
            self._tts_executor.shutdown(wait=False, cancel_futures=True)
            self._tts_executor = None
        for name in list(self.uploaded_speakers):
            self._speaker_cache.clear(name)

    def _find_tts_stage(self):
        """Find and return the TTS stage config, or None if not found."""
        tts_stage_keys = all_tts_stage_keys()
        entry_stage_archs = tts_entry_stage_archs()
        all_stages = frozenset(
            getattr(stage.engine_args, "model_stage", None) for stage in self.engine_client.stage_configs
        )
        for stage in self.engine_client.stage_configs:
            engine_args = stage.engine_args
            model_stage = engine_args.model_stage
            model_arch = getattr(engine_args, "model_arch", None)
            worker_type = getattr(engine_args, "worker_type", None)
            if model_stage in tts_stage_keys:
                # Owning the stage key is not always enough: a model may be
                # speech-capable only in some deployment topologies. Ask the
                # adapter. Stages that resolve to no adapter keep the legacy
                # behaviour of being accepted here (and detected as ``None``).
                adapter_cls = resolve_adapter(detect_tts_model_type(model_stage, model_arch))
                if adapter_cls is not None and not adapter_cls.stage_serves_speech(model_stage, all_stages):
                    continue
                return stage
            # Models with no dedicated TTS model_stage value identify their AR
            # entry stage by architecture (Ming dense).
            if model_arch in entry_stage_archs and worker_type == "ar":
                return stage
        return None

    def _detect_tts_model_type(self) -> str | None:
        """Detect TTS model type from the resolved stage's deployment metadata.

        The mapping lives on the adapters (``stage_keys`` / ``model_archs``);
        this only supplies the stage under inspection.
        """
        if self._tts_stage is None:
            return None
        return detect_tts_model_type(
            getattr(self._tts_stage.engine_args, "model_stage", None),
            getattr(self._tts_stage.engine_args, "model_arch", None),
        )

    def _compute_max_instructions_length(self) -> int:
        """Compute max instructions length with precedence: CLI > stage config > default.

        Called once during initialization; result is cached in self._max_instructions_length.
        """
        # 1. CLI override takes highest priority (stored in engine_client)
        cli_override = getattr(self.engine_client, "tts_max_instructions_length", None)
        if cli_override is not None:
            return cli_override

        # 2. Try to get from TTS stage config
        if self._tts_stage is not None:
            tts_args = getattr(self._tts_stage, "tts_args", {})
            if "max_instructions_length" in tts_args:
                return tts_args["max_instructions_length"]

        # 3. Default fallback
        return _TTS_MAX_INSTRUCTIONS_LENGTH

    def _get_available_voices(self) -> set[str]:
        """Get all voice names accepted by the API, including the placeholder default."""
        return self._get_available_speakers() | {_DEFAULT_VOICE_NAME}

    def _get_usage_text_tokenizer(self):
        """Return a text tokenizer for counting input-text usage tokens.

        Prefer the per-model tokenizer already loaded for prompt-length
        estimation (`_tts_tokenizer`, which is the *correct* tokenizer for the
        active model). Fall back to a lazily-loaded, cached generic tokenizer
        for models that never populate `_tts_tokenizer`. Returns None if no
        tokenizer can be obtained (usage then reports text_tokens=0).
        """
        if self._tts_tokenizer is not None:
            return self._tts_tokenizer
        if getattr(self, "_usage_text_tokenizer", None) is None:
            try:
                from transformers import AutoTokenizer

                self._usage_text_tokenizer = AutoTokenizer.from_pretrained(
                    self.engine_client.model_config.model, trust_remote_code=True
                )
            except Exception as e:  # pragma: no cover - environment dependent
                logger.warning("Usage: could not load a text tokenizer (%s); text_tokens will be 0", e)
                self._usage_text_tokenizer = None
        return self._usage_text_tokenizer

    def _count_usage_text_tokens(self, text: str) -> int:
        """Token count of `text` using the model's text tokenizer (0 on failure)."""
        if not text:
            return 0
        tok = self._get_usage_text_tokenizer()
        if tok is None:
            return 0
        try:
            return len(tok(text, padding=False)["input_ids"])
        except Exception:
            return 0

    def _compute_speech_input_details(
        self, request: OpenAICreateSpeechRequest, tts_params: dict[str, Any]
    ) -> SpeechInputTokenDetails:
        """Input-token breakdown (text + reference-audio) for a speech request.

        Counts `input` (+ `instructions`) as text tokens, and reference-audio
        codec frames as audio tokens *only* when in-context voice cloning is
        active (see `qwen3_tts_input_token_details` / `gate_audio_tokens`). The
        audio gating reads Qwen3-TTS `tts_params` conventions; other TTS models
        do not set those keys, so they degrade cleanly to text-only counts.
        """
        return qwen3_tts_input_token_details(
            input_text=request.input,
            instructions=request.instructions,
            tts_params=tts_params or {},
            count_text_tokens=self._count_usage_text_tokens,
        )

    def _build_speech_usage(
        self,
        request: OpenAICreateSpeechRequest,
        tts_params: dict[str, Any],
        output_tokens: int,
    ) -> SpeechTokenUsage:
        """Assemble the full usage object (input breakdown + generated tokens)."""
        details = self._compute_speech_input_details(request, tts_params)
        return build_speech_usage(details, output_tokens)

    def _validate_tts_generation(
        self,
        tts_params: dict[str, Any],
        usage_acc: SpeechOutputTokenCounter,
    ) -> None:
        adapter = self._get_tts_adapter()
        if adapter is None:
            return
        adapter.validate_generation(
            tts_params,
            stage0_finish_reason=usage_acc.stage0_finish_reason,
            output_tokens=usage_acc.total(),
        )

    @staticmethod
    def _build_speech_usage_headers(usage: SpeechTokenUsage | None) -> dict[str, str]:
        """Map speech usage into non-streaming response headers.

        Returns an empty dict when usage is unavailable, allowing callers to
        merge these headers with other optional response headers.
        """
        if usage is None:
            return {}

        return {
            _SPEECH_USAGE_INPUT_TOKENS_HEADER: str(usage.input_tokens),
            _SPEECH_USAGE_OUTPUT_TOKENS_HEADER: str(usage.output_tokens),
            _SPEECH_USAGE_TOTAL_TOKENS_HEADER: str(usage.total_tokens),
            _SPEECH_USAGE_INPUT_TEXT_TOKENS_HEADER: str(usage.input_token_details.text_tokens),
            _SPEECH_USAGE_INPUT_AUDIO_TOKENS_HEADER: str(usage.input_token_details.audio_tokens),
        }

    def _voice_created_at(self, voice_lower: str) -> int:
        """Return the upload timestamp of an uploaded voice, or 0 for built-ins.

        Plumbed through to the model-side cache key so that delete + re-upload
        of the same name yields a fresh cache slot.
        """
        info = self.uploaded_speakers.get(voice_lower)
        return int(info.get("created_at", 0)) if info else 0

    def _load_uploaded_audio(self, voice_name: str) -> tuple[np.ndarray, int] | None:
        """Load decoded audio samples + sample rate from an uploaded voice's safetensors."""
        voice_name_lower = voice_name.lower()
        info = self.uploaded_speakers.get(voice_name_lower)
        if info is None or info.get("embedding_source") != "audio":
            return None
        file_path = Path(info["file_path"])
        if not file_path.exists():
            logger.warning("Voice file not found for %s: %s", voice_name, file_path)
            return None
        try:
            from safetensors import safe_open
        except ImportError:
            logger.error("The 'safetensors' package is required to load uploaded voices")
            return None
        try:
            with safe_open(str(file_path), framework="pt") as f:
                if "audio" not in f.keys():
                    return None
                samples = f.get_tensor("audio").numpy()
                sr = int((f.metadata() or {}).get("sample_rate", info.get("sample_rate", 0)))
        except Exception as e:
            logger.error("Could not load audio for voice %s: %s", voice_name, e)
            return None
        if sr <= 0:
            return None
        return samples, sr

    def _get_uploaded_audio_data(self, voice_name: str) -> str | None:
        """Return a base64-encoded WAV data URL for an uploaded voice.

        Memoized so the WAV re-encode runs once per voice per process.
        """
        voice_name_lower = voice_name.lower()
        cached = self._ref_audio_data_url_cache.get(voice_name_lower)
        if cached is not None:
            return cached

        data = self._load_uploaded_audio(voice_name)
        if data is None:
            return None
        samples, sr = data
        try:
            buf = io.BytesIO()
            sf.write(buf, samples, sr, format="WAV")
            audio_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            data_url = f"data:audio/wav;base64,{audio_b64}"
        except Exception as e:
            logger.error("Could not encode voice %s as WAV: %s", voice_name, e)
            return None
        self._ref_audio_data_url_cache[voice_name_lower] = data_url
        return data_url

    def _get_uploaded_speaker_embedding(self, voice_name: str) -> list[float] | None:
        """Load a pre-computed speaker embedding from an uploaded voice's safetensors.

        Returns ``None`` if the voice has audio (not a direct embedding)."""
        voice_name_lower = voice_name.lower()
        info = self.uploaded_speakers.get(voice_name_lower)
        if info is None or info.get("embedding_source") != "direct":
            return None
        file_path = Path(info["file_path"])
        if not file_path.exists():
            logger.warning("Embedding file not found for voice %s: %s", voice_name, file_path)
            return None
        if not _validate_path_within_directory(file_path, self.uploaded_speakers_dir):
            logger.error("File path traversal detected for voice %s: %s", voice_name, file_path)
            return None
        try:
            from safetensors.torch import load_file
        except ImportError:
            logger.error("The 'safetensors' package is required to load speaker embeddings")
            return None
        try:
            tensors = load_file(str(file_path))
            if "speaker_embedding" not in tensors:
                logger.warning("Key 'speaker_embedding' missing in %s", file_path)
                return None
            return tensors["speaker_embedding"].squeeze().tolist()
        except Exception as e:
            logger.error("Could not load embedding for voice %s: %s", voice_name, e)
            return None

    def _apply_uploaded_speaker(self, request: OpenAICreateSpeechRequest) -> str | None:
        """Resolve ``request.voice`` against uploaded speakers, mutating
        ``request.ref_audio`` / ``request.ref_text`` in place. Returns an
        error string if the voice is invalid, else ``None``.
        """
        if request.voice is None or request.ref_audio is not None:
            return None

        voice_lower = request.voice.lower()
        if voice_lower not in self.uploaded_speakers:
            if self._tts_model_type in (
                "cosyvoice3",
                "fish_tts",
                "omnivoice",
                "moss_tts_nano",
                "glm_tts",
                "higgs_audio_v2",
                "higgs_audio_v3",
            ):
                label = {
                    "cosyvoice3": "CosyVoice3",
                    "fish_tts": "Fish Speech",
                    "omnivoice": "OmniVoice",
                    "moss_tts_nano": "MOSS-TTS-Nano",
                    "higgs_audio_v2": "Higgs-Audio V2",
                    "higgs_audio_v3": "Higgs-Audio V3",
                    "glm_tts": "GLM-TTS",
                }.get(self._tts_model_type, self._tts_model_type)
                return (
                    f"Unknown voice '{request.voice}'. {label} has no "
                    f"built-in speakers. Upload a voice first via "
                    f"POST /v1/audio/voices, or use ref_audio + ref_text."
                )
            return None

        speaker_info = self.uploaded_speakers[voice_lower]
        if speaker_info.get("embedding_source") == "direct":
            return (
                f"Uploaded voice '{request.voice}' uses a speaker embedding "
                f"(Qwen3-only). Re-upload with an audio file for this model."
            )

        audio_data = self._get_uploaded_audio_data(request.voice)
        if not audio_data:
            return f"Audio file for uploaded voice '{request.voice}' is missing"

        request.ref_audio = audio_data
        if not request.ref_text or not request.ref_text.strip():
            stored_ref_text = speaker_info.get("ref_text")
            if stored_ref_text:
                request.ref_text = stored_ref_text

        logger.info("Resolved uploaded voice '%s' for %s", voice_lower, self._tts_model_type)
        return None

    def _check_upload_cap(self) -> None:
        if len(self.uploaded_speakers) >= self._max_uploaded_speakers:
            raise ValueError(
                f"Uploaded voice limit reached ({self._max_uploaded_speakers}). "
                f"Delete an existing voice before registering a new one, or raise "
                f"the cap via SPEAKER_MAX_UPLOADED."
            )

    def _evict_existing_upload(self, voice_name_lower: str, name: str) -> None:
        """Drop an existing upload with this name so the caller can re-register it."""
        if voice_name_lower not in self.uploaded_speakers:
            return
        old = self.uploaded_speakers.pop(voice_name_lower)
        self._ref_audio_data_url_cache.pop(voice_name_lower, None)
        old_path = old.get("file_path")
        if old_path:
            try:
                Path(old_path).unlink(missing_ok=True)
            except Exception as e:
                logger.warning("Failed to remove previous file for '%s': %s", name, e)
        self._speaker_cache.clear(voice_name_lower)
        logger.info("Speaker '%s' re-uploaded; previous cache and file overwritten", name)

    async def upload_voice(
        self,
        audio_file: UploadFile,
        consent: str,
        name: str,
        *,
        ref_text: str | None = None,
        speaker_description: str | None = None,
    ) -> dict:
        """Upload a new voice sample."""
        name = _validate_speaker_name(name)
        # Normalize optional strings: treat whitespace-only as absent
        if ref_text is not None:
            ref_text = ref_text.strip() or None
        if speaker_description is not None:
            speaker_description = speaker_description.strip() or None
        # Validate file size (max 10MB)
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        audio_file.file.seek(0, 2)  # Seek to end
        file_size = audio_file.file.tell()
        audio_file.file.seek(0)  # Reset to beginning

        if file_size > MAX_FILE_SIZE:
            raise ValueError(f"File size exceeds maximum limit of 10MB. Got {file_size} bytes.")

        # Detect MIME type from filename if content_type is generic
        mime_type = audio_file.content_type
        if mime_type == "application/octet-stream":
            # Simple MIME type detection based on file extension
            filename_lower = audio_file.filename.lower()
            if filename_lower.endswith(".wav"):
                mime_type = "audio/wav"
            elif filename_lower.endswith((".mp3", ".mpeg")):
                mime_type = "audio/mpeg"
            elif filename_lower.endswith(".flac"):
                mime_type = "audio/flac"
            elif filename_lower.endswith(".ogg"):
                mime_type = "audio/ogg"
            elif filename_lower.endswith(".aac"):
                mime_type = "audio/aac"
            elif filename_lower.endswith(".webm"):
                mime_type = "audio/webm"
            elif filename_lower.endswith(".mp4"):
                mime_type = "audio/mp4"
            else:
                mime_type = "audio/wav"  # Default

        # Validate MIME type
        allowed_mime_types = {
            "audio/mpeg",
            "audio/wav",
            "audio/x-wav",
            "audio/ogg",
            "audio/aac",
            "audio/flac",
            "audio/webm",
            "audio/mp4",
        }

        if mime_type not in allowed_mime_types:
            raise ValueError(f"Unsupported MIME type: {mime_type}. Allowed: {allowed_mime_types}")

        # Read content before acquiring the lock; decode happens inside.
        content = await audio_file.read()

        async with self._upload_lock:
            voice_name_lower = name.lower()
            self._evict_existing_upload(voice_name_lower, name)
            self._check_upload_cap()

            sanitized_name = _sanitize_filename(name)
            sanitized_consent = _sanitize_filename(consent)
            timestamp = self._next_upload_timestamp()
            file_suffix = Path(audio_file.filename).suffix
            file_ext = file_suffix[1:] if file_suffix and len(file_suffix) > 1 else "wav"
            sanitized_ext = _sanitize_filename(file_ext)
            if not sanitized_ext or sanitized_ext == "file":
                sanitized_ext = "wav"

            filename = f"{sanitized_name}_{sanitized_consent}_{timestamp}.safetensors"
            file_path = self.uploaded_speakers_dir / filename
            if not _validate_path_within_directory(file_path, self.uploaded_speakers_dir):
                raise ValueError("Invalid file path: potential path traversal attack detected")

            try:
                wav_np, sr = sf.read(io.BytesIO(content))
            except Exception as e:
                raise ValueError(f"Could not decode audio file: {e}")
            duration = len(wav_np) / sr if sr > 0 else 0.0
            if duration < _REF_AUDIO_MIN_DURATION:
                raise ValueError(
                    f"Reference audio too short ({duration:.1f}s). "
                    f"At least {_REF_AUDIO_MIN_DURATION:.0f}s of clear speech is required."
                )
            if duration > _REF_AUDIO_MAX_DURATION:
                raise ValueError(
                    f"Reference audio too long ({duration:.1f}s). "
                    f"Maximum {_REF_AUDIO_MAX_DURATION:.0f}s supported — use a shorter clip."
                )

            speaker_data: dict[str, Any] = {
                "name": name,
                "voice_name_lower": voice_name_lower,
                "consent": consent,
                "file_path": str(file_path),
                "created_at": timestamp,
                "mime_type": mime_type,
                "original_filename": audio_file.filename,
                "file_size": file_size,
                "sample_rate": int(sr),
                "ref_text": ref_text,
                "embedding_source": "audio",
            }
            if speaker_description:
                speaker_data["speaker_description"] = speaker_description

            try:
                from safetensors.torch import save_file
            except ImportError as exc:
                raise ValueError("safetensors is required for voice upload") from exc
            try:
                audio_tensor = torch.from_numpy(np.asarray(wav_np, dtype=np.float32)).contiguous()
                save_file(
                    {"audio": audio_tensor},
                    str(file_path),
                    metadata=self._speaker_metadata_to_header(speaker_data),
                )
            except Exception as e:
                raise ValueError(f"Failed to save voice file: {e}")

            self.uploaded_speakers[voice_name_lower] = speaker_data

        logger.info("Uploaded new voice '%s' with consent ID '%s'", name, consent)

        # Return voice information without exposing the server file path
        result = {
            "name": name,
            "consent": consent,
            "created_at": timestamp,
            "mime_type": mime_type,
            "file_size": file_size,
        }
        if speaker_data.get("ref_text"):
            result["ref_text"] = speaker_data["ref_text"]
        if speaker_data.get("speaker_description"):
            result["speaker_description"] = speaker_data["speaker_description"]
        return result

    async def upload_voice_embedding(self, embedding_json: str, consent: str, name: str) -> dict:
        """Upload a voice from a pre-computed speaker embedding.

        Stores the embedding as a safetensors file and marks it immediately
        ready (no audio processing needed).

        Args:
            embedding_json: JSON-encoded list of floats (1024 or 2048 dim).
            consent: Consent recording ID.
            name: Name for the new voice.

        Returns:
            dict with voice information.
        """
        name = _validate_speaker_name(name)
        try:
            embedding = json.loads(embedding_json)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"'speaker_embedding' must be valid JSON: {exc}") from exc

        if not isinstance(embedding, list) or not embedding:
            raise ValueError("'speaker_embedding' must be a non-empty list of numbers")

        if len(embedding) > 4096:
            raise ValueError("'speaker_embedding' exceeds maximum length (4096 elements)")

        if not all(isinstance(x, (int, float)) for x in embedding):
            raise ValueError("'speaker_embedding' must contain only numeric values")

        if not all(math.isfinite(x) for x in embedding):
            raise ValueError("'speaker_embedding' values must be finite (no NaN or Inf)")

        emb_dim = len(embedding)
        dim_err = self._adapter.validate_tts_embedding_dim(emb_dim) if self._adapter is not None else None
        if dim_err is not None:
            raise ValueError(dim_err)

        async with self._upload_lock:
            voice_name_lower = name.lower()
            self._evict_existing_upload(voice_name_lower, name)
            self._check_upload_cap()

            sanitized_name = _sanitize_filename(name)
            sanitized_consent = _sanitize_filename(consent)
            timestamp = self._next_upload_timestamp()

            tensor = torch.tensor(embedding, dtype=torch.float32)
            filename = f"{sanitized_name}_{sanitized_consent}_{timestamp}.safetensors"
            file_path = self.uploaded_speakers_dir / filename
            if not _validate_path_within_directory(file_path, self.uploaded_speakers_dir):
                raise ValueError("Invalid file path: potential path traversal attack detected")

            speaker_data: dict[str, Any] = {
                "name": name,
                "voice_name_lower": voice_name_lower,
                "consent": consent,
                "file_path": str(file_path),
                "created_at": timestamp,
                "mime_type": "application/x-safetensors",
                "original_filename": filename,
                "embedding_source": "direct",
                "embedding_dim": emb_dim,
            }
            try:
                from safetensors.torch import save_file
            except ImportError as exc:
                raise ValueError("safetensors is required for embedding upload") from exc
            save_file(
                {"speaker_embedding": tensor},
                str(file_path),
                metadata=self._speaker_metadata_to_header(speaker_data),
            )
            speaker_data["file_size"] = file_path.stat().st_size

            self.uploaded_speakers[voice_name_lower] = speaker_data

        logger.info("Uploaded voice '%s' from speaker embedding (%d-dim)", name, emb_dim)

        return {
            "name": name,
            "consent": consent,
            "created_at": timestamp,
            "embedding_source": "direct",
            "embedding_dim": emb_dim,
        }

    async def delete_voice(self, name: str) -> bool:
        """
        Delete an uploaded voice.

        Args:
            name: Voice name to delete

        Returns:
            bool: True if successful, False if voice doesn't exist
        """
        async with self._upload_lock:
            voice_name_lower = name.lower()

            if voice_name_lower not in self.uploaded_speakers:
                logger.warning("Voice '%s' not found", name)
                return False

            speaker_info = self.uploaded_speakers.pop(voice_name_lower)
            self._ref_audio_data_url_cache.pop(voice_name_lower, None)

            file_path = speaker_info.get("file_path")
            if file_path:
                try:
                    Path(file_path).unlink(missing_ok=True)
                except Exception as e:
                    logger.warning("Failed to delete audio file for '%s': %s", name, e)

            self._speaker_cache.clear(voice_name_lower)

        logger.info("Deleted voice '%s'", name)
        return True

    def _is_tts_model(self) -> bool:
        """Check if the current model is a supported TTS model."""
        return self._find_tts_stage() is not None

    def _validate_tts_request(self, request: OpenAICreateSpeechRequest) -> str | None:
        """Validate TTS request parameters. Returns error message or None."""
        sample_rate_error = self._validate_speech_sample_rate(request)
        if sample_rate_error is not None:
            return sample_rate_error

        adapter = self._get_tts_adapter()
        if adapter is not None:
            return adapter.validate(request)

        adapter_cls = resolve_adapter("qwen3_tts")
        if adapter_cls is None:
            raise ValueError("Qwen3-TTS adapter is not registered")

        ctx = SpeechServingContext(
            server=self,
            engine_client=self.engine_client,
        )
        return adapter_cls(ctx).validate(request)

    def _validate_speech_sample_rate(self, request: OpenAICreateSpeechRequest) -> str | None:
        if request.sample_rate is None:
            return None

        adapter_cls = resolve_adapter(self._tts_model_type)
        supported_rates = adapter_cls.supported_output_sample_rates if adapter_cls is not None else frozenset()
        if request.sample_rate not in supported_rates:
            if supported_rates:
                rates = ", ".join(str(rate) for rate in sorted(supported_rates))
                return (
                    f"sample_rate={request.sample_rate} is not supported by the current TTS model; "
                    f"supported rates: {rates}"
                )
            return "sample_rate is not supported by the current TTS model"
        return None

    def _validate_ref_audio_format(self, ref_audio: str) -> str | None:
        """Validate ref_audio is a supported URI format. Returns error or None."""
        if not isinstance(ref_audio, str):
            return "ref_audio must be a URL (http/https), base64 data URL (data:...), or file URI (file://...)"
        scheme = (urlparse(ref_audio).scheme or "").lower()
        if scheme not in {"http", "https", "data", "file"}:
            return "ref_audio must be a URL (http/https), base64 data URL (data:...), or file URI (file://...)"
        return None

    @staticmethod
    def _local_ref_audio_stat_path(ref_audio_str: str) -> str | None:
        """Filesystem path to stat for a local locator, else ``None``.

        Scheme comparison is case-insensitive (``urlparse`` lowercases it), so
        ``FILE:///...`` is treated as a local file the same way ``file://`` is.
        Remote ``http`` / ``https`` / ``data`` locators, including ``HTTP://``,
        return ``None`` and stay string-keyed.
        """
        parsed = urlparse(ref_audio_str)
        scheme = (parsed.scheme or "").lower()
        if scheme in _REMOTE_REF_AUDIO_SCHEMES:
            return None
        if scheme == "file":
            netloc = parsed.netloc or ""
            if netloc.lower() not in ("", "localhost"):
                raise OSError(f"file:// URI with non-local authority {netloc!r} cannot be stat'd")
            return url2pathname(parsed.path or "")
        if scheme:
            return None
        return ref_audio_str

    @staticmethod
    def _get_ref_audio_cache_key(
        ref_audio_str: str,
        allowed_local_media_path: str | None = None,
    ) -> str:
        """Compute a cache key hash for *ref_audio_str*.

        For local files (bare paths and ``file://`` URIs) the key folds in
        ``st_mtime_ns`` and ``st_size`` so that an on-disk edit automatically
        invalidates the cached waveform without a server restart.  Remote URLs
        and ``data:`` URIs are keyed on the raw string alone — the server does
        not re-fetch them to check for changes.

        ``os.stat`` runs only when *allowed_local_media_path* is a non-empty
        path. vLLM's default is ``""`` (not ``None``); both mean local media
        is refused, so a stat would be exposure for a request that cannot
        load the file.

        Note: when the key changes the *previous* entry stays in
        ``_ref_audio_resolve_cache`` until LRU eviction, so
        ``_discard_ref_audio_artifact_ready_if_unreferenced`` will not fire for
        the replaced file and its stale artifact key remains in
        ``_ref_audio_model_artifact_ready``.  This is functionally correct
        (the stale entry is never *used*) but doubles memory until eviction.
        """
        cache_key_source = ref_audio_str
        try:
            path = OmniOpenAIServingSpeech._local_ref_audio_stat_path(ref_audio_str)
        except OSError as exc:
            path = None
            logger.debug("Skipping local-file cache metadata for %s: %s", ref_audio_str[:80], exc)
        if path is not None:
            try:
                # Only stat paths the server operator has explicitly permitted
                # via --allowed-local-media-path.  The default is "" (and
                # diffusion mode passes None); MediaConnector refuses local
                # file loads in both cases, so a stat is exposure for zero
                # benefit.  ``if not`` covers "" and None; ``is None`` would
                # treat "" as an allowlist of the process cwd.
                if not allowed_local_media_path:
                    raise OSError("no allowed_local_media_path configured; skipping stat")
                resolved = os.path.realpath(path)
                allowed_base = os.path.realpath(allowed_local_media_path)
                if os.path.commonpath([resolved, allowed_base]) != allowed_base:
                    raise OSError("path outside allowed_local_media_path; skipping stat")
                st = os.stat(path)
                cache_key_source = f"{ref_audio_str}:{st.st_mtime_ns}:{st.st_size}"
            except (OSError, ValueError):
                # Truncate the value to avoid dumping huge base64 blobs into
                # the server log when a client omits the ``data:`` prefix.
                display = ref_audio_str[:80]
                if len(ref_audio_str) > 80:
                    display += f"... ({len(ref_audio_str)} chars)"
                logger.debug(
                    "Failed to stat ref_audio path %s; falling back to string-only cache key (stale cache possible)",
                    display,
                )
        return hashlib.sha1(cache_key_source.encode("utf-8")).hexdigest()

    async def _ref_audio_cache_key(self, ref_audio_str: str, allowed_local_media_path: str | None) -> str:
        """Stat local files off the event loop; remote locators stay a cheap hash."""
        return await asyncio.to_thread(
            self._get_ref_audio_cache_key,
            ref_audio_str,
            allowed_local_media_path,
        )

    def _finalize_fetched_ref_audio(self, wav_np: np.ndarray, sr: int) -> tuple[list[float], int, str, float]:
        wav_np = np.asarray(wav_np, dtype=np.float32)
        if wav_np.ndim > 1:
            wav_np = np.mean(wav_np, axis=-1)
        sr = int(sr)
        duration = len(wav_np) / sr if sr > 0 else 0.0
        if duration < _REF_AUDIO_MIN_DURATION:
            raise ValueError(
                f"Reference audio too short ({duration:.1f}s). "
                f"At least {_REF_AUDIO_MIN_DURATION:.0f}s of clear speech is required."
            )
        if duration > _REF_AUDIO_MAX_DURATION:
            raise ValueError(
                f"Reference audio too long ({duration:.1f}s). "
                f"Maximum {_REF_AUDIO_MAX_DURATION:.0f}s supported — use a shorter clip."
            )
        artifact_key = self._make_ref_audio_artifact_cache_key(wav_np, sr)
        return wav_np.tolist(), sr, artifact_key, duration

    async def _resolve_ref_audio(self, ref_audio_str: str) -> tuple[list[float], int, str]:
        """Resolve ref_audio to (wav_samples, sample_rate, cache_key).

        Delegates to upstream vLLM's MediaConnector which handles http(s)
        URLs, ``data:`` base64 URIs, and ``file:`` local paths (the latter
        gated by ``--allowed-local-media-path``).

        Local file references incorporate mtime and size into the cache key
        so that modified files are automatically reloaded without a server
        restart. Remote URLs remain cached by their original string locator.

        The returned *cache_key* should be passed to
        ``_get_resolved_ref_audio_artifact_key`` when the caller needs the
        artifact key, avoiding a redundant ``os.stat`` and the TOCTOU window.
        After a cache miss the file is re-stat'd; a metadata change retries
        the fetch so the stored waveform matches the key.
        """
        # Pass the allowed-local-media-path so the stat is restricted to
        # paths the server operator has explicitly permitted.
        allowed_path: str | None
        if self._diffusion_mode:
            allowed_path = self._allowed_local_media_path
        else:
            allowed_path = getattr(self.model_config, "allowed_local_media_path", None)

        wav_list: list[float] | None = None
        sr = 0
        artifact_key = ""
        post_key = ""
        for attempt in range(_REF_AUDIO_METADATA_FETCH_ATTEMPTS):
            cache_key = await self._ref_audio_cache_key(ref_audio_str, allowed_path)
            cached = self._ref_audio_resolve_cache.get(cache_key)
            if cached is not None:
                self._ref_audio_resolve_cache.move_to_end(cache_key)
                wav_list, sr, _, _ = cached
                logger.debug(
                    "Resolved ref_audio from cache: samples=%d sr=%d duration_s=%.3f",
                    len(wav_list),
                    sr,
                    len(wav_list) / sr if sr > 0 else 0.0,
                )
                return wav_list, sr, cache_key

            if self._media_connector is None:
                model_config = self.model_config
                self._media_connector = MediaConnector(
                    allowed_local_media_path=model_config.allowed_local_media_path,
                    allowed_media_domains=model_config.allowed_media_domains,
                )

            fetch_start_s = time.perf_counter()
            wav_np, fetched_sr = await self._media_connector.fetch_audio_async(ref_audio_str)
            fetch_decode_ms = (time.perf_counter() - fetch_start_s) * 1000.0
            tolist_start_s = time.perf_counter()
            wav_list, sr, artifact_key, duration = self._finalize_fetched_ref_audio(wav_np, fetched_sr)
            tolist_ms = (time.perf_counter() - tolist_start_s) * 1000.0
            logger.debug(
                "Resolved ref_audio: fetch_decode_ms=%.3f tolist_ms=%.3f samples=%d sr=%d duration_s=%.3f",
                fetch_decode_ms,
                tolist_ms,
                len(wav_list),
                sr,
                duration,
            )
            post_key = await self._ref_audio_cache_key(ref_audio_str, allowed_path)
            if post_key == cache_key:
                self._put_resolved_ref_audio(cache_key, wav_list, sr, artifact_key)
                return wav_list, sr, cache_key
            logger.debug(
                "ref_audio metadata changed during fetch (attempt %d/%d); retrying",
                attempt + 1,
                _REF_AUDIO_METADATA_FETCH_ATTEMPTS,
            )

        logger.warning(
            "ref_audio file changed during fetch after %d attempts; skipping resolve cache",
            _REF_AUDIO_METADATA_FETCH_ATTEMPTS,
        )
        assert wav_list is not None and post_key
        return wav_list, sr, post_key

    @staticmethod
    def _make_ref_audio_artifact_cache_key(wav: np.ndarray, sr: int) -> str:
        wav_f32 = wav.astype(np.float32, copy=False).reshape(-1)
        h = hashlib.sha1()
        h.update(int(sr).to_bytes(4, byteorder="little", signed=False))
        h.update(int(wav_f32.size).to_bytes(8, byteorder="little", signed=False))
        h.update(wav_f32.tobytes(order="C"))
        return h.hexdigest()

    def _get_resolved_ref_audio_artifact_key(self, cache_key: str) -> str | None:
        """Look up the artifact key for a previously resolved ref_audio.

        *cache_key* must be the exact key returned by the preceding
        ``_resolve_ref_audio`` call.  Passing the key explicitly avoids a
        second ``os.stat`` syscall and eliminates the TOCTOU window that
        would arise from recomputing the key independently.
        """
        cached = self._ref_audio_resolve_cache.get(cache_key)
        if cached is None:
            return None
        self._ref_audio_resolve_cache.move_to_end(cache_key)
        return cached[3]

    def _put_resolved_ref_audio(self, cache_key: str, wav_list: list[float], sr: int, artifact_key: str) -> None:
        if self._ref_audio_resolve_cache_max_entries <= 0 or self._ref_audio_resolve_cache_max_bytes <= 0:
            return
        # Approximate list[float] storage. CPython float objects add per-element
        # overhead, so max_entries remains the hard cache cap.
        size = len(wav_list) * 40
        if size > self._ref_audio_resolve_cache_max_bytes:
            return
        previous = self._ref_audio_resolve_cache.pop(cache_key, None)
        if previous is not None:
            self._ref_audio_resolve_cache_bytes -= previous[2]
            if previous[3] != artifact_key:
                self._discard_ref_audio_artifact_ready_if_unreferenced(previous[3])
        self._ref_audio_resolve_cache[cache_key] = (wav_list, int(sr), size, artifact_key)
        self._ref_audio_resolve_cache_bytes += size
        while len(self._ref_audio_resolve_cache) > self._ref_audio_resolve_cache_max_entries:
            _, (_, _, old_size, old_artifact_key) = self._ref_audio_resolve_cache.popitem(last=False)
            self._ref_audio_resolve_cache_bytes -= old_size
            self._discard_ref_audio_artifact_ready_if_unreferenced(old_artifact_key)
        while self._ref_audio_resolve_cache_bytes > self._ref_audio_resolve_cache_max_bytes:
            _, (_, _, old_size, old_artifact_key) = self._ref_audio_resolve_cache.popitem(last=False)
            self._ref_audio_resolve_cache_bytes -= old_size
            self._discard_ref_audio_artifact_ready_if_unreferenced(old_artifact_key)

    def _discard_ref_audio_artifact_ready_if_unreferenced(self, artifact_key: str) -> None:
        if artifact_key and all(entry[3] != artifact_key for entry in self._ref_audio_resolve_cache.values()):
            self._ref_audio_model_artifact_ready = {
                (key, mode) for (key, mode) in self._ref_audio_model_artifact_ready if key != artifact_key
            }

    @staticmethod
    def _tts_x_vector_only(tts_params: dict[str, Any]) -> bool:
        return bool((tts_params.get("x_vector_only_mode") or [False])[0])

    def _track_ref_audio_artifact_warmup(
        self, request_id: str, artifact_key: str | None, x_vector_only: bool = False
    ) -> None:
        if artifact_key:
            self._request_ref_audio_artifact_keys[request_id] = (artifact_key, bool(x_vector_only))

    def _mark_ref_audio_artifact_ready_for_request(self, request_id: str) -> None:
        tracked = self._request_ref_audio_artifact_keys.pop(request_id, None)
        if tracked is None:
            return
        artifact_key, x_vector_only = tracked
        if artifact_key and any(entry[3] == artifact_key for entry in self._ref_audio_resolve_cache.values()):
            self._ref_audio_model_artifact_ready.add((artifact_key, x_vector_only))

    def _discard_ref_audio_artifact_warmup(self, request_id: str) -> None:
        self._request_ref_audio_artifact_keys.pop(request_id, None)

    async def _resolve_ref_audio_many(self, ref_audio_list: list[str]) -> list[tuple[list[float], int]]:
        resolved = []
        for ref_audio in ref_audio_list:
            wav_list, sr, _ = await self._resolve_ref_audio(ref_audio)
            resolved.append((wav_list, sr))
        return resolved

    async def _generate_audio_chunks(
        self,
        generator,
        request_id: str,
        response_format: str = "pcm",
        raw_request: Request | None = None,
        request_start_s: float | None = None,
        include_sample_rate: bool = False,
        usage_acc: SpeechOutputTokenCounter | None = None,
        tts_params: dict[str, Any] | None = None,
        collect: dict | None = None,
        target_sample_rate: int | None = None,
    ):
        """Generate audio chunks for streaming response.

        Handles two audio output modes from the engine:
        - Cumulative mode (list): Engine returns growing list of chunks;
        we emit only the new tail on each iteration.
        - Per-step mode (tensor): Engine returns single tensor per iteration;
        we emit it directly.

        Args:
            generator: Async generator from the engine
            request_id: Request identifier for logging
            response_format: Audio format (pcm or wav)

        Yields:
            Raw audio bytes for each chunk (with WAV header for first chunk if wav format)
        """
        prev_count = 0
        sample_rate_val = 24000
        first_chunk = True
        first_audio_chunk_s: float | None = None
        stream_start_s = request_start_s if request_start_s is not None else time.perf_counter()
        artifact_ready = False
        source_sample_rate: int | None = None
        resampler: StreamingAudioResampler | None = None

        # SSE supplies an accumulator for usage output. Raw-audio and WebSocket
        # streams retain terminal metrics only when their model adapter needs
        # generation validation.
        adapter = self._get_tts_adapter()
        if tts_params is not None and usage_acc is None and adapter is not None and adapter.validates_generation:
            usage_acc = SpeechOutputTokenCounter()

        try:
            async for res in generator:
                # Tally generated codec tokens for usage (reads per-stage metrics
                # off the final output; a cheap early-return on every other res).
                if usage_acc is not None:
                    usage_acc.observe(res)
                audio_output, audio_key = self._extract_audio_output(res)
                if audio_key is None:
                    # Stash the aligner's timestamps output for streaming callers.
                    if collect is not None and self._is_timestamps_output(res):
                        collect["aligner_res"] = res
                    continue

                sr_raw = audio_output.get("sr")
                if sr_raw is not None:
                    sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
                    sample_rate_val = sr_val.item() if hasattr(sr_val, "item") else int(sr_val)
                    if source_sample_rate is not None and sample_rate_val != source_sample_rate:
                        raise ValueError(
                            "Audio sample rate changed during streaming: "
                            f"{source_sample_rate} Hz to {sample_rate_val} Hz"
                        )
                audio_val = audio_output[audio_key]
                if isinstance(audio_val, list):
                    # Cumulative mode: each update grows the list; emit only new tail.
                    new_chunks = audio_val[prev_count:]
                    prev_count = len(audio_val)
                else:
                    # Per-step mode: each update is a single tensor; emit directly.
                    if audio_val is not None:
                        new_chunks = [audio_val]
                        prev_count += 1
                    else:
                        new_chunks = []

                if new_chunks and source_sample_rate is None:
                    if response_format == "wav" and sr_raw is None:
                        raise ValueError("First audio output must include sample rate metadata for WAV streaming")
                    source_sample_rate = sample_rate_val
                    if target_sample_rate is not None and target_sample_rate != source_sample_rate:
                        resampler = StreamingAudioResampler(source_sample_rate, target_sample_rate)

                output_sample_rate = target_sample_rate or sample_rate_val

                for chunk_tensor in new_chunks:
                    chunk_np = (
                        chunk_tensor.float().detach().cpu().numpy() if hasattr(chunk_tensor, "float") else chunk_tensor
                    )
                    if chunk_np.ndim > 1:
                        chunk_np = chunk_np.squeeze()
                    if resampler is not None:
                        chunk_np = resampler.process(chunk_np)
                        if chunk_np.size == 0:
                            continue
                    if self._tts_model_type in _AUDEX_NO_AUDIO_GUARD_MODEL_TYPES and int(np.size(chunk_np)) == 0:
                        # Zero-size chunks must not emit a WAV header or count
                        # as first audio; the post-loop guard below needs to
                        # see an audio-less stream to fail the request.
                        continue
                    # For WAV format, emit header before first audio chunk
                    if response_format == "wav" and first_chunk:
                        num_channels = _infer_audio_num_channels(np.asarray(chunk_np))
                        wav_header = _create_wav_header(
                            sample_rate=output_sample_rate,
                            num_channels=num_channels,
                            bits_per_sample=16,
                        )
                        yield wav_header
                        first_chunk = False

                    # Convert audio to PCM bytes
                    audio_obj = CreateAudio(
                        audio_tensor=chunk_np,
                        sample_rate=output_sample_rate,
                        response_format="pcm",
                        speed=1.0,
                        base64_encode=False,
                    )
                    if first_audio_chunk_s is None:
                        first_audio_chunk_s = time.perf_counter()
                    audio_bytes = self.create_audio(audio_obj).audio_data
                    if include_sample_rate:
                        yield audio_bytes, output_sample_rate
                    else:
                        yield audio_bytes

            if resampler is not None:
                final_chunk = resampler.process(np.empty((0,), dtype=np.float32), final=True)
                if final_chunk.size:
                    output_sample_rate = target_sample_rate or sample_rate_val
                    if response_format == "wav" and first_chunk:
                        yield _create_wav_header(
                            sample_rate=output_sample_rate,
                            num_channels=1,
                            bits_per_sample=16,
                        )
                        first_chunk = False
                    audio_obj = CreateAudio(
                        audio_tensor=final_chunk,
                        sample_rate=output_sample_rate,
                        response_format="pcm",
                        speed=1.0,
                        base64_encode=False,
                    )
                    if first_audio_chunk_s is None:
                        first_audio_chunk_s = time.perf_counter()
                    audio_bytes = self.create_audio(audio_obj).audio_data
                    if include_sample_rate:
                        yield audio_bytes, output_sample_rate
                    else:
                        yield audio_bytes
            if self._tts_model_type in _AUDEX_NO_AUDIO_GUARD_MODEL_TYPES and first_audio_chunk_s is None:
                # Audex contract: zero codec tokens must abort the stream, not
                # complete it cleanly with zero audio bytes.
                raise ValueError("Audex produced no audio (the thinker emitted zero or invalid codec tokens)")
            # Check before committing the reference-audio artifact or logging
            # success. Streaming protocols may already have emitted partial
            # bytes, but they must terminate as an error rather than cleanly.
            if tts_params is not None and usage_acc is not None:
                self._validate_tts_generation(tts_params, usage_acc)
            self._mark_ref_audio_artifact_ready_for_request(request_id)
            artifact_ready = True
            total_ms = (time.perf_counter() - stream_start_s) * 1000.0
            if first_audio_chunk_s is not None:
                first_chunk_ms = (first_audio_chunk_s - stream_start_s) * 1000.0
                logger.info(
                    "[SpeechE2E] request_id=%s stream=true status=ok total_ms=%.2f first_chunk_ms=%.2f",
                    request_id,
                    total_ms,
                    first_chunk_ms,
                )
            else:
                logger.info(
                    "[SpeechE2E] request_id=%s stream=true status=ok total_ms=%.2f first_chunk_ms=NA",
                    request_id,
                    total_ms,
                )
        except asyncio.CancelledError:
            total_ms = (time.perf_counter() - stream_start_s) * 1000.0
            logger.info(
                "[SpeechE2E] request_id=%s stream=true status=cancelled total_ms=%.2f",
                request_id,
                total_ms,
            )
            logger.info("Streaming request %s cancelled by client", request_id)
            raise
        except EngineDeadError as e:
            total_ms = (time.perf_counter() - stream_start_s) * 1000.0
            logger.error(
                "[SpeechE2E] request_id=%s stream=true status=engine_dead total_ms=%.2f",
                request_id,
                total_ms,
            )
            logger.error(
                "EngineDeadError during streaming speech for %s: %s",
                request_id,
                e,
            )
            # Actively signal shutdown rather than relying on the watchdog.
            if raw_request is not None:
                terminate_if_errored(
                    server=raw_request.app.state.server,
                    engine=self.engine_client,
                )
            raise
        except Exception as e:
            total_ms = (time.perf_counter() - stream_start_s) * 1000.0
            logger.exception(
                "[SpeechE2E] request_id=%s stream=true status=error total_ms=%.2f error=%s",
                request_id,
                total_ms,
                e,
            )
            logger.exception("Streaming speech generation failed for %s: %s", request_id, e)
            raise
        finally:
            if not artifact_ready:
                self._discard_ref_audio_artifact_warmup(request_id)

    async def _generate_audio_sse_events(
        self,
        generator,
        request_id: str,
        response_format: str = "pcm",
        raw_request: Request | None = None,
        request_start_s: float | None = None,
        request: OpenAICreateSpeechRequest | None = None,
        tts_params: dict[str, Any] | None = None,
    ):
        """Generate OpenAI-style SSE events with base64 audio deltas.

        Field naming follows the OpenAI ``speech.audio.delta`` schema, which
        carries the base64 chunk in ``audio`` (not ``delta`` — that is the
        Realtime API ``response.audio.delta`` convention, a different event).
        See https://platform.openai.com/docs/api-reference/audio-streaming.

        The terminal ``speech.audio.done`` event carries a ``usage`` object
        (``input_tokens``/``output_tokens``/``total_tokens`` + a per-modality
        ``input_token_details`` breakdown), matching OpenAI's documented
        ``speech.audio.done`` schema. ``output_tokens`` is accumulated from the
        stage-0 deltas as they stream (see ``SpeechOutputTokenCounter``);
        ``input_tokens`` is computed from the request text + reference audio.
        """
        usage_acc = SpeechOutputTokenCounter()
        emitted_audio = False
        try:
            async for chunk in self._generate_audio_chunks(
                generator,
                request_id,
                response_format,
                raw_request=raw_request,
                request_start_s=request_start_s,
                usage_acc=usage_acc,
                tts_params=tts_params,
                target_sample_rate=request.sample_rate if request is not None else None,
            ):
                payload = {
                    "type": "speech.audio.delta",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                    "response_format": response_format,
                }
                data = json.dumps(payload, separators=(",", ":"))
                emitted_audio = True
                yield f"event: speech.audio.delta\ndata: {data}\n\n"
            done_payload: dict[str, Any] = {"type": "speech.audio.done"}
            if request is not None:
                # Streaming path: output_tokens = sum of stage-0 deltas.
                usage = self._build_speech_usage(request, tts_params or {}, usage_acc.total())
                done_payload["usage"] = usage.model_dump()
            done = json.dumps(done_payload, separators=(",", ":"))
            yield f"event: speech.audio.done\ndata: {done}\n\n"
        except asyncio.CancelledError:
            raise
        except Exception as e:
            error: dict[str, Any] = {
                "message": str(e),
                "type": "server_error",
                "param": None,
                "code": HTTPStatus.INTERNAL_SERVER_ERROR.value,
            }
            if emitted_audio:
                error.update(partial_audio=True, action="discard")
            error_payload: dict[str, Any] = {
                "type": "speech.audio.error",
                "error": error,
            }
            data = json.dumps(error_payload, separators=(",", ":"))
            yield f"event: speech.audio.error\ndata: {data}\n\n"

    @staticmethod
    def _is_timestamps_output(res) -> bool:
        """True when ``res`` is the forced-aligner stage's terminal timestamps output."""
        from vllm_omni.model_executor.stage_input_processors.forced_aligner import TIMESTAMPS_MODALITY

        return getattr(res, "final_output_type", None) == TIMESTAMPS_MODALITY

    @staticmethod
    def _extract_audio_output(res) -> tuple[dict | None, str | None]:
        """Return (audio_output dict, audio key) or (None, None).

        Returns the raw dict so callers can apply their own extraction strategy:
        streaming needs per-chunk delta slicing; non-streaming needs full concatenation.
        """
        mm = getattr(res, "multimodal_output", None)
        ro = None
        if not mm:
            ro = res
            mm = getattr(ro, "multimodal_output", None) if ro else None
        if not mm:
            # MultimodalOutputProcessor attaches mm_accumulated on per-completion outputs.
            container = res if hasattr(res, "outputs") else ro
            outputs = getattr(container, "outputs", None) if container is not None else None
            if outputs:
                for completion_output in outputs:
                    completion_mm = getattr(completion_output, "multimodal_output", None)
                    if completion_mm:
                        mm = completion_mm
                        break
        if not mm:
            return None, None
        key = "audio" if "audio" in mm else ("model_outputs" if "model_outputs" in mm else None)
        return mm, key

    async def _prepare_speech_generation(
        self,
        request: OpenAICreateSpeechRequest,
        request_id: str | None = None,
        has_inline_ref_audio: bool | None = None,
    ) -> tuple[str, Any, dict[str, Any]]:
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        sample_rate_error = self._validate_speech_sample_rate(request)
        if sample_rate_error is not None:
            raise ValueError(sample_rate_error)

        request_id = request_id or f"speech-{random_uuid()}"
        qwen3_ref_audio_warmup_artifact_key: str | None = None

        # If this is a streaming request with real async chunks, we need to
        # coerce cumulative outputs to delta outputs; this ensures we don't
        # emit redundant MM data & drain after emitting. Qwen3-TTS full-payload
        # (async_chunk=False) has no incremental audio chunks, so keep
        # FINAL_ONLY semantics and let the streaming response send the final
        # waveform once. Scoped to qwen3_tts: other async_chunk=False models
        # keep the DELTA coercion they stream with today.
        # list() makes a copy to avoid mutating the params.
        sampling_params_list = list(self.engine_client.default_sampling_params_list)
        async_chunk = getattr(self.model_config, "async_chunk", True)
        qwen3_full_payload = self._tts_model_type == "qwen3_tts" and not bool(async_chunk)
        is_streaming_request = request.is_streaming() and not qwen3_full_payload
        sampling_params_list = coerce_param_message_types(sampling_params_list, is_streaming_request)

        # Build prompt + tts_params via the per-model adapter (RFC #4327). Every
        # dedicated TTS model resolves to an adapter that owns its validation,
        # uploaded-speaker handling, prompt/param building, and sampling
        # overrides. The model-type label remains available for compatibility.
        # Non-TTS deployments (no adapter) fall through to the rejection below.
        # Capture inline-ref-audio status BEFORE validate(): several adapters
        # apply uploaded speakers inside validate(), which sets request.ref_audio
        # in place. The builders need to know whether the caller supplied audio
        # inline vs. via an uploaded voice.
        model_type: str | None = None
        has_inline_ref_audio = (request.ref_audio is not None) if has_inline_ref_audio is None else has_inline_ref_audio
        if (adapter := self._get_tts_adapter()) is not None:
            validation_error = adapter.validate(request)
            if validation_error:
                raise ValueError(validation_error)
            prepared = await adapter.build(request, sampling_params_list, has_inline_ref_audio)
            prompt = prepared.prompt
            tts_params = prepared.tts_params
            model_type = prepared.model_type
            qwen3_ref_audio_warmup_artifact_key = prepared.warmup_artifact_key
        else:
            # Qwen omni models (Qwen3-Omni, Qwen2.5-Omni) use a "talker"
            # stage whose preprocess requires chat-templated tokens.  The
            # async-chunk orchestrator prewarms the talker via
            # compute_talker_prompt_ids_length(), which scans for Qwen
            # chat-template markers (im_start_token_id 151644).  A raw-text
            # prompt produces a 1-token placeholder that crashes the talker's
            # prefill/decode handoff.  Reject early with an actionable message.
            stage_names = {
                getattr(getattr(s, "engine_args", None), "model_stage", None) for s in self.engine_client.stage_configs
            }
            if "talker" in stage_names:
                raise ValueError(
                    "The /v1/audio/speech endpoint is only supported for "
                    "dedicated TTS models (e.g., Qwen3-TTS, Voxtral, Fish "
                    "Speech, CosyVoice3, OmniVoice, VoxCPM2). For omni "
                    "models like Qwen3-Omni, use /v1/chat/completions with "
                    '\'"modalities": ["audio"]\' instead.'
                )
            tts_params = {}
            prompt = {"prompt": request.input}

        if model_type is None:
            if self._is_tts:
                model_type = tts_params.get("task_type", ["unknown"])[0]
            else:
                model_type = "generic"
        logger.info("TTS speech request %s: model=%s", request_id, model_type)
        _rl = getattr(self, "request_logger", None)
        if _rl:
            base_len = len(f"TTS speech request {request_id}: text=")
            raw_max = getattr(_rl, "max_log_len", None)
            cap = raw_max if isinstance(raw_max, int) else 200
            text = request.input[: max(cap - base_len, 0)]
            logger.debug("TTS speech request %s: text=%r", request_id, text)

        # Apply model-specific extra parameters
        if request.extra_params is not None and sampling_params_list:
            if not isinstance(request.extra_params, dict):
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST.value,
                    detail="extra_params must be a JSON object/dict.",
                )
            import copy

            sampling_params_list = copy.deepcopy(sampling_params_list)
            for name in ("temperature", "top_p", "top_k"):
                if (value := request.extra_params.get(name)) is not None:
                    setattr(sampling_params_list[0], name, value)
            if sampling_params_list[0].extra_args is None:
                sampling_params_list[0].extra_args = {}
            sampling_params_list[0].extra_args.update(request.extra_params)
            logger.info("Applied extra_params: %s", request.extra_params)

        # Apply adapter-owned sampling overrides, including request-level token
        # limits and model-specific dynamic token or stop-token configuration.
        if sampling_params_list and (adapter := self._get_tts_adapter()) is not None:
            sampling_params_list = adapter.apply_sampling_overrides(sampling_params_list, request, prompt, request_id)

        if request.seed is not None and sampling_params_list:
            import copy

            sampling_params_list = copy.deepcopy(sampling_params_list)
            stage0_params = sampling_params_list[0]
            stage0_params.seed = request.seed
            if stage0_params.extra_args is None:
                stage0_params.extra_args = {}
            stage0_params.extra_args["tts_local_seed"] = request.seed

        # When word_timestamps is requested, also ask for the aligner stage's
        # output so the orchestrator drives the request through the forced-aligner
        # stage (final_stage_id extends to it). Harmless if no aligner stage exists.
        output_modalities = ["audio"]
        if getattr(request, "word_timestamps", False):
            from vllm_omni.model_executor.stage_input_processors.forced_aligner import TIMESTAMPS_MODALITY

            output_modalities.append(TIMESTAMPS_MODALITY)

        generator = self.engine_client.generate(
            prompt=prompt,
            request_id=request_id,
            sampling_params_list=sampling_params_list,
            output_modalities=output_modalities,
        )
        self._track_ref_audio_artifact_warmup(
            request_id,
            qwen3_ref_audio_warmup_artifact_key,
            x_vector_only=self._tts_x_vector_only(tts_params),
        )
        return request_id, generator, tts_params

    async def _generate_pcm_chunks(
        self,
        generator,
        request_id: str,
        *,
        include_sample_rate: bool = False,
        tts_params: dict[str, Any] | None = None,
        collect: dict | None = None,
        target_sample_rate: int | None = None,
    ):
        """Yield raw PCM byte chunks from the engine generator.

        Delegates to ``_generate_audio_chunks`` with ``response_format="pcm"``.
        Used by the WebSocket streaming handler and ``_iter_pcm_audio_bytes``.
        ``collect`` (when given) receives the forced-aligner stage's pooling
        output under ``"aligner_res"`` for downstream word-timestamp extraction.
        """
        async for chunk in self._generate_audio_chunks(
            generator,
            request_id,
            response_format="pcm",
            include_sample_rate=include_sample_rate,
            tts_params=tts_params,
            collect=collect,
            target_sample_rate=target_sample_rate,
        ):
            yield chunk

    async def _iter_pcm_audio_bytes(self, request: OpenAICreateSpeechRequest):
        """Yield raw PCM bytes for a speech request as soon as chunks are decoded."""
        request_id, generator, tts_params = await self._prepare_speech_generation(request)
        try:
            async for chunk in self._generate_pcm_chunks(
                generator,
                request_id,
                tts_params=tts_params,
                target_sample_rate=request.sample_rate,
            ):
                yield chunk
        finally:
            self._discard_ref_audio_artifact_warmup(request_id)

    async def _generate_audio_bytes(
        self,
        request: OpenAICreateSpeechRequest,
        base64_encode: bool = False,
        request_id: str | None = None,
        usage_out: list[SpeechTokenUsage] | None = None,
        has_inline_ref_audio: bool | None = None,
        collect: dict | None = None,
    ) -> tuple[bytes | str, str]:
        # ``usage_out`` is an opt-in output channel: when a list is passed, the
        # computed SpeechTokenUsage is appended to it. The return stays a
        # 2-tuple so existing callers (and their test mocks) are unaffected;
        # batch and non-streaming response-header paths opt in when surfacing
        # usage outside the raw audio body.
        request_id, generator, bytes_tts_params = await self._prepare_speech_generation(
            request, request_id=request_id, has_inline_ref_audio=has_inline_ref_audio
        )
        artifact_ready = False

        try:
            # MOSS-TTS-Nano emits delta chunks per yield (single-stage,
            # async_chunk=false). The engine surfaces each yield as its own
            # RequestOutput, so we need to accumulate across the async-for loop —
            # final_output alone only carries the last (often empty) sentinel.
            is_moss = self._tts_model_type == "moss_tts_nano"
            moss_chunks: list[Any] = []
            moss_sample_rate: int | None = None

            final_output: OmniRequestOutput | None = None
            # Non-streaming is FINAL_ONLY, so the stage-0 output carries the full
            # token sequence; the counter records its length for output_tokens.
            usage_acc = SpeechOutputTokenCounter()
            audio_res: OmniRequestOutput | None = None
            aligner_res: OmniRequestOutput | None = None
            async for res in generator:
                final_output = res
                usage_acc.observe(res)
                # The generator yields both the audio output (Code2Wav) and, with
                # a forced-aligner stage, a timestamps output. Keep the audio res
                # for the WAV and the aligner res for word timestamps.
                if self._is_timestamps_output(res):
                    aligner_res = res
                else:
                    _, audio_key = self._extract_audio_output(res)
                    if audio_key is not None:
                        audio_res = res
                if not is_moss:
                    continue
                try:
                    step_audio, step_key = self._extract_audio_output(res)
                except Exception:
                    continue
                if step_key is None:
                    continue
                chunk = step_audio[step_key]
                candidates = chunk if isinstance(chunk, list) else [chunk]
                for cand in candidates:
                    if hasattr(cand, "numel") and cand.numel() > 0:
                        moss_chunks.append(cand)
                sr_step = step_audio.get("sr")
                if sr_step is not None:
                    sr_val_step = sr_step[-1] if isinstance(sr_step, list) and sr_step else sr_step
                    moss_sample_rate = int(sr_val_step.item()) if hasattr(sr_val_step, "item") else int(sr_val_step)

            if final_output is None:
                raise ValueError("No output generated from the model.")

            self._validate_tts_generation(bytes_tts_params or {}, usage_acc)

            # Extract audio from the audio-bearing res (not necessarily the last
            # yielded one, which may be the aligner's timestamps output).
            audio_source = audio_res if audio_res is not None else final_output
            audio_output, audio_key = self._extract_audio_output(audio_source)
            if audio_key is None:
                raise ValueError("TTS model did not produce audio output.")

            # Surface forced-aligner word timestamps to the caller (set as a
            # response header) when requested and an aligner stage produced them.
            if collect is not None and getattr(request, "word_timestamps", False):
                from vllm_omni.utils.forced_aligner import extract_word_timestamps

                ts = (
                    extract_word_timestamps(aligner_res, request.input, getattr(request, "language", None))
                    if aligner_res is not None
                    else None
                )
                if ts is not None:
                    collect["word_timestamps"] = ts

            audio_tensor = audio_output[audio_key]
            sr_raw = audio_output.get("sr", 24000)
            sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
            sample_rate = sr_val.item() if hasattr(sr_val, "item") else int(sr_val)

            if is_moss:
                # Prefer the engine's own consolidated audio when present. After the
                # vllm 0.20 rebase non-stream requests resolve to FINAL_ONLY, so
                # final_output already carries the full concatenated waveform; the
                # delta-accumulator below is kept as a fallback for DELTA-style
                # engines that surface chunks one yield at a time.
                if isinstance(audio_tensor, list):
                    non_empty_final = [c for c in audio_tensor if hasattr(c, "numel") and c.numel() > 0]
                    final_audio = torch.cat(non_empty_final, dim=-1) if non_empty_final else None
                elif hasattr(audio_tensor, "numel") and audio_tensor.numel() > 0:
                    final_audio = audio_tensor
                else:
                    final_audio = None

                if final_audio is not None:
                    audio_tensor = final_audio
                elif moss_chunks:
                    audio_tensor = torch.cat(moss_chunks, dim=-1)
                else:
                    audio_tensor = np.zeros((0,), dtype=np.float32)
                if moss_sample_rate is not None:
                    sample_rate = moss_sample_rate
            elif isinstance(audio_tensor, list):
                async_chunk = bool(getattr(self.engine_client.model_config, "async_chunk", False))
                if async_chunk:
                    non_empty_chunks = [candidate for candidate in audio_tensor if candidate.numel() > 0]
                    audio_tensor = (
                        torch.cat(non_empty_chunks, dim=-1) if non_empty_chunks else np.zeros((0,), dtype=np.float32)
                    )
                else:
                    audio_history = audio_tensor
                    audio_tensor = np.zeros((0,), dtype=np.float32)
                    # Non-async Qwen3-TTS returns cumulative history snapshots, so keep the latest non-empty tensor.
                    for candidate in reversed(audio_history):
                        if candidate.numel() > 0:
                            audio_tensor = candidate
                            break
            if hasattr(audio_tensor, "float"):
                audio_tensor = audio_tensor.float().detach().cpu().numpy()

            if audio_tensor.ndim > 1:
                audio_tensor = audio_tensor.squeeze()

            if self._tts_model_type in _AUDEX_NO_AUDIO_GUARD_MODEL_TYPES and int(np.size(audio_tensor)) == 0:
                # Audex contract: zero codec tokens must fail the request, not
                # serialize as an empty-but-successful WAV.
                raise ValueError("Audex produced no audio (the thinker emitted zero or invalid codec tokens)")

            audio_obj = CreateAudio(
                audio_tensor=audio_tensor,
                sample_rate=sample_rate,
                output_sample_rate=request.sample_rate,
                response_format=request.response_format or "wav",
                speed=self._audio_encode_speed(request),
                base64_encode=base64_encode,
            )
            audio_response: AudioResponse = self.create_audio(audio_obj)
            self._mark_ref_audio_artifact_ready_for_request(request_id)
            artifact_ready = True
            if usage_out is not None:
                usage_out.append(self._build_speech_usage(request, bytes_tts_params or {}, usage_acc.total()))
            return audio_response.audio_data, audio_response.media_type
        finally:
            if not artifact_ready:
                self._discard_ref_audio_artifact_warmup(request_id)

    def _get_normalized_voice(self, voice: str | None) -> str | None:
        """Get the normalized voice to be used; currently this means that
        the voice is a:
            - lowercase str if it's a valid supported/uploaded speaker
            - None if the voice is the placeholder default or not provided
        """
        if voice is not None:
            voice = voice.lower()
            available_speakers = self._get_available_speakers()
            if voice not in available_speakers:
                raise ValueError(
                    f"Invalid voice '{voice}'. Supported: {', '.join(sorted(self._get_available_voices()))}"
                )
        return voice

    async def _create_diffusion_speech(
        self,
        request: OpenAICreateSpeechRequest,
    ) -> Response:
        """Handle speech generation for pure diffusion TTS models (e.g. OmniVoice)."""
        from vllm_omni.outputs import OmniRequestOutput

        try:
            if not request.input or not request.input.strip():
                raise ValueError("Input text cannot be empty")

            if request.ref_audio is not None:
                fmt_err = self._validate_ref_audio_format(request.ref_audio)
                if fmt_err:
                    return self._diffusion_error_response(fmt_err, status_code=400)

            request.voice = self._get_normalized_voice(request.voice)

            has_inline_ref_audio = request.ref_audio is not None
            err = self._apply_uploaded_speaker(request)
            if err:
                raise ValueError(err)

            request_id = f"speech-{random_uuid()}"
            prompt: dict[str, Any] = {"input": request.input}
            if request.ref_audio:
                wav, sr, _ = await self._resolve_ref_audio(request.ref_audio)
                prompt["ref_audio"] = (np.asarray(wav, dtype=np.float32), sr)
            if request.ref_text:
                prompt["ref_text"] = request.ref_text
            if request.voice:
                if request.voice in self.uploaded_speakers and not has_inline_ref_audio:
                    prompt["voice_name"] = request.voice
                    prompt["voice_created_at"] = self._voice_created_at(request.voice)
            if request.language:
                prompt["lang"] = request.language
            if request.instructions:
                prompt["instruct"] = request.instructions

            logger.info(
                "Diffusion TTS speech request %s: voice_clone=%s",
                request_id,
                "ref_audio" in prompt,
            )
            _rl = getattr(self, "request_logger", None)
            if _rl:
                base_len = len(f"Diffusion TTS speech request {request_id}: text=")
                raw_max = getattr(_rl, "max_log_len", None)
                cap = raw_max if isinstance(raw_max, int) else 200
                text = request.input[: max(cap - base_len, 0)]
                logger.debug("Diffusion TTS speech request %s: text=%r", request_id, text)
            if request.extra_params is not None and not isinstance(request.extra_params, dict):
                raise ValueError("extra_params must be a JSON object/dict.")
            extra = dict(request.extra_params or {})
            if request.seed is not None:
                extra["seed"] = request.seed
            # Apply extra_params from the request to sampling params
            sampling_params_list = self._diffusion_engine.default_sampling_params_list
            if extra:
                import copy

                sampling_params_list = copy.deepcopy(sampling_params_list)
                if sampling_params_list[0].extra_args is None:
                    sampling_params_list[0].extra_args = {}
                sampling_params_list[0].extra_args.update(extra)
                logger.info("Applied extra_params to diffusion: %s", extra)

            generator = self._diffusion_engine.generate(
                prompt=prompt,
                request_id=request_id,
                sampling_params_list=sampling_params_list,
                output_modalities=["audio"],
            )

            final_output: OmniRequestOutput | None = None
            async for res in generator:
                final_output = res

            if final_output is None:
                raise ValueError("No output generated from the model.")

            audio_output, audio_key = self._extract_audio_output(final_output)
            if audio_key is None:
                raise ValueError("TTS model did not produce audio output.")

            audio_tensor = audio_output[audio_key]
            sr_raw = audio_output.get("sr", 24000)
            sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
            sample_rate = sr_val.item() if hasattr(sr_val, "item") else int(sr_val)

            if isinstance(audio_tensor, list):
                non_empty = [c for c in audio_tensor if c.numel() > 0]
                audio_tensor = torch.cat(non_empty, dim=-1) if non_empty else np.zeros((0,), dtype=np.float32)
            if hasattr(audio_tensor, "float"):
                audio_tensor = audio_tensor.float().detach().cpu().numpy()
            if audio_tensor.ndim > 1:
                audio_tensor = audio_tensor.squeeze()

            audio_obj = CreateAudio(
                audio_tensor=audio_tensor,
                sample_rate=sample_rate,
                output_sample_rate=request.sample_rate,
                response_format=request.response_format or "wav",
                speed=self._audio_encode_speed(request),
                base64_encode=False,
            )
            audio_response: AudioResponse = self.create_audio(audio_obj)
            return Response(content=audio_response.audio_data, media_type=audio_response.media_type)

        except asyncio.CancelledError:
            return self._diffusion_error_response("Client disconnected")
        except (EngineGenerateError, EngineDeadError):
            raise  # Propagate to the global Omni exception handler
        except ValueError as e:
            return self._diffusion_error_response(str(e), status_code=400)
        except Exception as e:
            logger.exception("Diffusion speech generation failed: %s", e)
            return self._diffusion_error_response(f"Speech generation failed: {e}")

    @staticmethod
    def _diffusion_error_response(message: str, status_code: int = 500) -> Response:
        """Create a JSON error response without depending on OpenAIServing.

        Args:
            message: Error message to surface to the client.
            status_code: HTTP status code; defaults to 500. Pass a 4xx code for
                client-input validation failures so the response semantics match
                the OpenAI-compatible behavior used by ``create_speech``.
        """
        err_type = "BadRequestError" if 400 <= status_code < 500 else "server_error"
        error_body = json.dumps({"error": {"message": message, "type": err_type, "param": None, "code": status_code}})
        return Response(content=error_body, media_type="application/json", status_code=status_code)

    def _validate_speech_streaming_request(
        self,
        request: OpenAICreateSpeechRequest,
        *,
        mode_label: str,
    ) -> tuple[str, Response | None]:
        """Validate pcm/wav + speed constraints for streaming speech responses."""
        response_format = (request.response_format or "wav").lower()
        if response_format not in ("pcm", "wav"):
            return response_format, self.create_error_response(
                f"{mode_label} is only supported for 'pcm' and 'wav' formats. Got '{response_format}'."
            )
        if request.speed is not None and request.speed != 1.0 and not self._uses_native_speed_control():
            return response_format, self.create_error_response(
                f"{mode_label} is not supported with speed adjustment. "
                "Use a non-streaming request or remove the speed parameter."
            )
        return response_format, None

    async def create_speech(
        self,
        request: OpenAICreateSpeechRequest,
        raw_request: Request | None = None,
    ):
        """
        Create Speech API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/audio/createSpeech
        for the API specification. This API mimics the OpenAI
        Create Speech API.

        For Qwen3-TTS models, additional parameters are supported:
        - task_type: "CustomVoice", "VoiceDesign", or "Base"
        - language: Language code (e.g., "Chinese", "English", "Auto")
        - voice: Speaker name (e.g., "Vivian", "Ryan") for CustomVoice
        - instructions: Voice style/emotion instructions
        - ref_audio: Reference audio for voice cloning (Base task)
        - ref_text: Transcript of reference audio (Base task)
        - x_vector_only_mode: Use speaker embedding only (Base task)
        - sample_rate: Target output sample rate (8000 or 24000 Hz)

        Streaming is supported via the ``stream=True`` switch or ``stream_format='sse'``,
        which return OpenAI ``speech.audio.*`` SSE events. ``stream_format='audio'``
        opts into raw audio streaming with ``response_format='pcm'`` or ``'wav'``.
        Raw audio streaming yields each Code2Wav chunk as raw bytes as soon as it is
        decoded. Raw WAV streaming emits a header with placeholder size values first.
        """
        if request.voice is not None:
            if _is_default_voice(request.voice.lower(), self._get_available_speakers()):
                request.voice = None

        sample_rate_error = self._validate_speech_sample_rate(request)
        if sample_rate_error is not None:
            return self.create_error_response(sample_rate_error)

        if self._diffusion_mode:
            return await self._create_diffusion_speech(request)

        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        request_id = f"speech-{random_uuid()}"
        request_start_s = time.perf_counter()
        if raw_request:
            raw_request.state.request_metadata = RequestResponseMetadata(
                request_id=request_id,
            )

        try:
            if request.is_streaming() and request.word_timestamps:
                return self.create_error_response(
                    "word_timestamps=true is currently supported by the WebSocket "
                    "/v1/audio/speech/stream path. Use session.config with "
                    "stream_audio=true and response_format='pcm'."
                )
            if request.word_timestamps and not self.forced_aligner_enabled:
                # Fail loud instead of silently returning 200 with no timestamps header.
                return self.create_error_response(
                    "word_timestamps=true requires the server to be launched with --forced-aligner."
                )

            if request.is_raw_audio_stream():
                response_format, error = self._validate_speech_streaming_request(
                    request,
                    mode_label="Streaming",
                )
                if error is not None:
                    return error

                media_type = "audio/wav" if response_format == "wav" else "audio/pcm"
                _, generator, raw_tts_params = await self._prepare_speech_generation(request, request_id=request_id)
                return StreamingResponse(
                    self._generate_audio_chunks(
                        generator,
                        request_id,
                        response_format,
                        raw_request=raw_request,
                        request_start_s=request_start_s,
                        tts_params=raw_tts_params,
                        target_sample_rate=request.sample_rate,
                    ),
                    media_type=media_type,
                )

            if request.is_sse_stream():
                response_format, error = self._validate_speech_streaming_request(
                    request,
                    mode_label="SSE streaming",
                )
                if error is not None:
                    return error

                _, generator, sse_tts_params = await self._prepare_speech_generation(request, request_id=request_id)
                return StreamingResponse(
                    self._generate_audio_sse_events(
                        generator,
                        request_id,
                        response_format,
                        raw_request=raw_request,
                        request_start_s=request_start_s,
                        request=request,
                        tts_params=sse_tts_params,
                    ),
                    media_type="text/event-stream",
                )

            collect: dict = {}
            usage_box: list[SpeechTokenUsage] = []
            try:
                audio_bytes, media_type = await self._generate_audio_bytes(
                    request, request_id=request_id, usage_out=usage_box, collect=collect
                )
            except TTSGenerationError as error:
                # An adapter can reject otherwise completed audio. Retry only
                # retryable, stochastic, server-budgeted non-streaming
                # requests: changing an explicit seed would violate
                # reproducibility, and changing a caller-provided token limit
                # would violate the requested budget.
                if not error.retryable or request.seed is not None or request.max_new_tokens is not None:
                    raise
                retry_seed = int(random_uuid()[:8], 16)
                retry_request = request.model_copy(update={"seed": retry_seed})
                retry_request_id = f"speech-{random_uuid()}"
                collect.clear()
                usage_box.clear()
                logger.warning(
                    "TTS request %s failed generation validation; retrying once as %s with seed=%d",
                    request_id,
                    retry_request_id,
                    retry_seed,
                )
                audio_bytes, media_type = await self._generate_audio_bytes(
                    retry_request,
                    request_id=retry_request_id,
                    usage_out=usage_box,
                    collect=collect,
                )
            total_ms = (time.perf_counter() - request_start_s) * 1000.0
            logger.info(
                "[SpeechE2E] request_id=%s stream=false status=ok total_ms=%.2f response_bytes=%d",
                request_id,
                total_ms,
                len(audio_bytes) if isinstance(audio_bytes, (bytes, bytearray)) else len(str(audio_bytes)),
            )
            headers = self._build_speech_usage_headers(usage_box[0] if usage_box else None)
            if collect.get("word_timestamps") is not None:
                # Default ensure_ascii keeps the header latin-1 encodable (non-ASCII words \uXXXX-escaped).
                ts_json = json.dumps(collect["word_timestamps"])
                # Cap at 4 KB: oversized headers turn into opaque 502s at common reverse-proxy defaults.
                if len(ts_json) <= 4096:
                    headers["X-Word-Timestamps"] = ts_json
                else:
                    # Marker header so clients can tell an oversized alignment from no alignment.
                    headers["X-Word-Timestamps-Omitted"] = f"oversize; bytes={len(ts_json)}; limit=4096"
                    logger.warning(
                        "X-Word-Timestamps header omitted: %d bytes exceeds the 4 KB budget "
                        "(use the WebSocket streaming path for long transcripts)",
                        len(ts_json),
                    )
            return Response(content=audio_bytes, media_type=media_type, headers=headers)

        except asyncio.CancelledError:
            total_ms = (time.perf_counter() - request_start_s) * 1000.0
            logger.info(
                "[SpeechE2E] request_id=%s stream=%s status=cancelled total_ms=%.2f",
                request_id,
                bool(request.stream),
                total_ms,
            )
            return self.create_error_response("Client disconnected")
        except (EngineGenerateError, EngineDeadError):
            total_ms = (time.perf_counter() - request_start_s) * 1000.0
            logger.error(
                "[SpeechE2E] request_id=%s stream=%s status=engine_error total_ms=%.2f",
                request_id,
                bool(request.stream),
                total_ms,
            )
            raise  # Propagate to the global Omni exception handler
        except ValueError as e:
            total_ms = (time.perf_counter() - request_start_s) * 1000.0
            logger.warning(
                "[SpeechE2E] request_id=%s stream=%s status=bad_request total_ms=%.2f error=%s",
                request_id,
                bool(request.stream),
                total_ms,
                e,
            )
            return self.create_error_response(e)
        except TTSGenerationError as e:
            total_ms = (time.perf_counter() - request_start_s) * 1000.0
            logger.error(
                "[SpeechE2E] request_id=%s stream=%s status=invalid_generation total_ms=%.2f error=%s",
                request_id,
                bool(request.stream),
                total_ms,
                e,
            )
            return self._diffusion_error_response(str(e), status_code=500)
        except Exception as e:
            total_ms = (time.perf_counter() - request_start_s) * 1000.0
            logger.exception(
                "[SpeechE2E] request_id=%s stream=%s status=error total_ms=%.2f error=%s",
                request_id,
                bool(request.stream),
                total_ms,
                e,
            )
            logger.exception("Speech generation failed: %s", e)
            return self.create_error_response(f"Speech generation failed: {e}")

    @staticmethod
    def _merge_batch_item(
        batch: BatchSpeechRequest,
        item: SpeechBatchItem,
    ) -> OpenAICreateSpeechRequest:
        """Merge batch-level defaults with per-item overrides into a full request."""

        def _pick(field: str):
            """Return item-level value if set, else batch-level value."""
            item_val = getattr(item, field, None)
            return item_val if item_val is not None else getattr(batch, field, None)

        picked_speed = _pick("speed")
        return OpenAICreateSpeechRequest(
            input=item.input,
            model=batch.model,
            voice=_pick("voice"),
            instructions=_pick("instructions"),
            response_format=_pick("response_format") or "wav",
            sample_rate=_pick("sample_rate"),
            speed=picked_speed if picked_speed is not None else 1.0,
            stream=False,
            task_type=_pick("task_type"),
            language=_pick("language"),
            ref_audio=_pick("ref_audio"),
            ref_text=_pick("ref_text"),
            x_vector_only_mode=_pick("x_vector_only_mode"),
            max_new_tokens=_pick("max_new_tokens"),
            initial_codec_chunk_frames=_pick("initial_codec_chunk_frames"),
            non_streaming_mode=_pick("non_streaming_mode"),
        )

    async def create_speech_batch(
        self,
        batch_request: BatchSpeechRequest,
    ) -> BatchSpeechResponse | ErrorResponse:
        """Generate speech for multiple items concurrently."""
        if self._diffusion_mode:
            raise ValueError("Batch speech is not supported in diffusion mode")
        if len(batch_request.items) > self._batch_max_items:
            raise ValueError(
                f"Batch contains {len(batch_request.items)} items, exceeding the maximum of {self._batch_max_items}."
            )

        error_check_ret = await self._check_model(batch_request)
        if error_check_ret is not None:
            return error_check_ret

        if self.engine_client.errored:
            raise self.engine_client.dead_error

        batch_id = f"speech-batch-{random_uuid()}"

        merged_requests = [self._merge_batch_item(batch_request, item) for item in batch_request.items]

        async def _run_item(idx: int, req: OpenAICreateSpeechRequest) -> SpeechBatchItemResult:
            has_inline_ref_audio = req.ref_audio is not None
            validation_error = self._validate_tts_request(req)
            if validation_error is not None:
                return SpeechBatchItemResult(index=idx, status="error", error=validation_error)
            usage_box: list[SpeechTokenUsage] = []
            try:
                audio_data, media_type = await self._generate_audio_bytes(
                    req, base64_encode=True, usage_out=usage_box, has_inline_ref_audio=has_inline_ref_audio
                )
            except Exception as e:
                logger.exception("Batch item %d failed: %s", idx, e)
                return SpeechBatchItemResult(index=idx, status="error", error=str(e))
            return SpeechBatchItemResult(
                index=idx,
                status="success",
                audio_data=audio_data,
                media_type=media_type,
                usage=usage_box[0] if usage_box else None,
            )

        results = await asyncio.gather(
            *[_run_item(i, req) for i, req in enumerate(merged_requests)],
            return_exceptions=True,
        )

        final_results: list[SpeechBatchItemResult] = []
        for i, r in enumerate(results):
            if isinstance(r, BaseException):
                logger.exception("Batch item %d raised unexpected exception: %s", i, r)
                final_results.append(SpeechBatchItemResult(index=i, status="error", error=str(r)))
            else:
                final_results.append(r)

        succeeded = sum(1 for r in final_results if r.status == "success")
        return BatchSpeechResponse(
            id=batch_id,
            results=final_results,
            total=len(final_results),
            succeeded=succeeded,
            failed=len(final_results) - succeeded,
        )


ServingSpeech = OmniOpenAIServingSpeech
