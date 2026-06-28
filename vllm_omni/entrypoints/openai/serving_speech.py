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
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from fastapi import HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from transformers.utils.hub import cached_file
from vllm.entrypoints.launcher import terminate_if_errored
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    RequestResponseMetadata,
)
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.inputs import tokens_input
from vllm.logger import init_logger
from vllm.multimodal.media import MediaConnector
from vllm.utils import random_uuid
from vllm.utils.async_utils import make_async
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError

from vllm_omni.entrypoints.openai.audio_utils_mixin import AudioMixin
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
    resolve_adapter,
)
from vllm_omni.entrypoints.utils import coerce_param_message_types
from vllm_omni.model_executor.models.fish_speech.prompt_utils import (
    build_fish_text_only_prompt_ids,
    estimate_fish_voice_clone_prompt_len_from_normalized,
    normalize_fish_voice_clone_texts,
)
from vllm_omni.model_executor.models.ming_flash_omni.prompt_utils import (
    DEFAULT_PROMPT as MING_DEFAULT_PROMPT,
)
from vllm_omni.model_executor.models.ming_flash_omni.prompt_utils import (
    create_instruction as ming_create_instruction,
)
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.utils.speaker_cache import (
    get_speaker_cache,
    iter_custom_voice_profiles,
    load_validated_profile_tensors,
)

logger = init_logger(__name__)

# TTS Configuration
_MING_TTS_MODEL_ARCHS = {"MingTTSForConditionalGeneration"}
_VOXTRAL_TTS_MODEL_STAGES = {"audio_generation"}
_QWEN3_TTS_MODEL_STAGES = {"qwen3_tts"}
_FISH_TTS_MODEL_STAGES = {"fish_speech_slow_ar"}
_COSYVOICE3_TTS_MODEL_STAGES = {"cosyvoice3_talker"}
# CosyVoice3 talker expects its reference transcript wrapped in the model
# instruction template; without the delimiter the talker re-speaks the
# reference (issue #4644). Matches the offline example/test and upstream demo.
_COSYVOICE3_PROMPT_DELIMITER = "<|endofprompt|>"
_COSYVOICE3_PROMPT_PREFIX = f"You are a helpful assistant.{_COSYVOICE3_PROMPT_DELIMITER}"
_OMNIVOICE_TTS_MODEL_STAGES = {"omnivoice_generator"}
_COVO_AUDIO_MODEL_STAGES = {"fused_thinker_talker"}
_VOXCPM2_TTS_MODEL_STAGES = {"latent_generator"}
_MING_TTS_MODEL_STAGES = {"ming_tts"}
_MOSS_TTS_MODEL_STAGES = {"moss_tts_nano"}
_MOSS_TTS_FULL_MODEL_STAGES = {"moss_tts", "moss_tts_codec"}
_MOSS_TTS_LOCAL_MODEL_STAGES = {"moss_tts_local", "moss_tts_local_codec"}
_HIGGS_AUDIO_V2_TTS_MODEL_STAGES = {"higgs_audio_v2"}
_HIGGS_V3_TTS_MODEL_STAGES = {"higgs_audio_v3"}
_GLM_TTS_MODEL_STAGES = {"glm_tts"}
_STEP_AUDIO2_TTS_MODEL_STAGES = {"step_audio2_thinker"}
_INDEXTTS2_TTS_MODEL_STAGES = {"indextts2_talker"}
_TTS_MODEL_STAGES: set[str] = (
    _VOXTRAL_TTS_MODEL_STAGES
    | _QWEN3_TTS_MODEL_STAGES
    | _FISH_TTS_MODEL_STAGES
    | _COSYVOICE3_TTS_MODEL_STAGES
    | _OMNIVOICE_TTS_MODEL_STAGES
    | _HIGGS_AUDIO_V2_TTS_MODEL_STAGES
    | _HIGGS_V3_TTS_MODEL_STAGES
    | _COVO_AUDIO_MODEL_STAGES
    | _VOXCPM2_TTS_MODEL_STAGES
    | _MING_TTS_MODEL_STAGES
    | _MOSS_TTS_MODEL_STAGES
    | _MOSS_TTS_FULL_MODEL_STAGES
    | _MOSS_TTS_LOCAL_MODEL_STAGES
    | _GLM_TTS_MODEL_STAGES
    | _STEP_AUDIO2_TTS_MODEL_STAGES
    | _INDEXTTS2_TTS_MODEL_STAGES
)
_SAMPLING_MAX_TOKENS_TTS_MODEL_TYPES = {
    "fish_tts",
    "qwen3_tts",
    "voxtral_tts",
    "cosyvoice3",
    "voxcpm2",
    "higgs_audio_v2",
    "higgs_audio_v3",
    "indextts2",
}
_TTS_LANGUAGES = frozenset(
    {
        "Auto",
        "Chinese",
        "English",
        "Japanese",
        "Korean",
        "German",
        "French",
        "Russian",
        "Portuguese",
        "Spanish",
        "Italian",
    }
)
_REF_AUDIO_MIN_DURATION = 1.0  # seconds
_REF_AUDIO_MAX_DURATION = 30.0  # seconds
_REF_AUDIO_RESOLVE_CACHE_MAX_ENTRIES = 256
_REF_AUDIO_RESOLVE_CACHE_MAX_BYTES = 256 * 1024 * 1024
_HIGGS_V3_REF_CODE_CACHE_MAX_ENTRIES = 256
_HIGGS_V3_REF_CODE_CACHE_MAX_BYTES = 64 * 1024 * 1024
_QWEN3_TTS_REF_AUDIO_CACHE_KEY = "_qwen3_tts_ref_audio_cache_key"
_TTS_MAX_INSTRUCTIONS_LENGTH = 500
_TTS_MAX_NEW_TOKENS_MIN = 1
_TTS_MAX_NEW_TOKENS_MAX = 4096
_MING_DEFAULT_PROMPT = MING_DEFAULT_PROMPT


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


def _conditioning_cache_salt(request, tts_params: dict | None = None) -> str:
    """Stable hash of the real Stage 0 conditioning for the prefix cache.

    The talker's vLLM prompt is placeholder token ids; the real inputs are
    rebuilt from text / ref_audio / ref_text into inputs_embeds. vLLM hashes
    token ids (folded with cache_salt) for prefix caching, so without a salt
    every request collides and a hit could reuse KV from a semantically
    different input. Tying the salt to the conditioning keeps a hit safe:
    identical conditioning may share the prefix, any difference never does.

    Raw request fields alone are not enough for uploaded voices: the request
    only carries the voice *name* (ref_audio/ref_text/task_type are resolved
    from stored voice data into ``tts_params``). Delete + re-upload under the
    same name leaves every raw field identical, so the resolved conditioning
    must also be folded in. ``voice_created_at`` bumps on every (re-)upload,
    which uniquely identifies the resolved reference artifact together with
    the voice name; the decoded ref_audio array itself need not be hashed.
    """
    h = hashlib.sha256()
    for part in (
        request.input,
        request.task_type,
        request.language,
        request.voice,
        request.ref_text,
        request.ref_audio,
        request.instructions,
        request.x_vector_only_mode,
        request.speaker_embedding,
    ):
        h.update(b"\x00")
        if part is not None:
            h.update(repr(part).encode("utf-8"))
    # Fold resolved conditioning that is auto-derived for uploaded voices and
    # absent from the raw request.
    for key in (
        "voice_created_at",
        "task_type",
        "speaker",
        "ref_text",
        "x_vector_only_mode",
    ):
        h.update(b"\x00")
        value = tts_params.get(key) if tts_params is not None else None
        if value is not None:
            h.update(repr(value).encode("utf-8"))
    return h.hexdigest()[:32]


class OmniOpenAIServingSpeech(OpenAIServing, AudioMixin):
    _diffusion_mode: bool = False
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
        self.uploaded_speakers: dict[str, dict] = {}
        self.precomputed_speakers: dict[str, dict[str, Any]] = {}
        self.supported_speakers: set[str] = set()
        self._ref_audio_data_url_cache: dict[str, str] = {}
        self._ref_audio_resolve_cache: OrderedDict[str, tuple[list[float], int, int, str]] = OrderedDict()
        self._ref_audio_resolve_cache_bytes = 0
        self._ref_audio_resolve_cache_max_entries = _REF_AUDIO_RESOLVE_CACHE_MAX_ENTRIES
        self._ref_audio_resolve_cache_max_bytes = _REF_AUDIO_RESOLVE_CACHE_MAX_BYTES
        self._ref_audio_model_artifact_ready: set[str] = set()
        self._request_ref_audio_artifact_keys: dict[str, str] = {}
        self._higgs_audio_v3_ref_code_cache: OrderedDict[str, tuple[torch.Tensor, int]] = OrderedDict()
        self._higgs_audio_v3_ref_code_cache_bytes = 0
        self._higgs_audio_v3_ref_code_inflight: dict[str, asyncio.Task[torch.Tensor]] = {}
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
            self.supported_speakers.add(voice_name_lower)
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
        instance._tts_model_type = "omnivoice"
        instance._is_tts = False
        instance._is_fish_speech = False
        # Diffusion-only instances don't have a TTS stage; set None so any
        # ``_is_tts_model()`` / ``_tts_stage`` access doesn't raise AttributeError.
        instance._tts_stage = None
        instance._init_speaker_storage()
        return instance

    def __init__(self, *args, **kwargs):
        self.model_name = kwargs.pop("model_name", None)
        self.forced_aligner_config: Any | None = kwargs.pop("forced_aligner_config", None)
        super().__init__(*args, **kwargs)
        self._init_speaker_storage()

        # Find and cache the TTS stage (if any) during initialization
        self._tts_stage = self._find_tts_stage()
        self._is_tts = self._tts_stage is not None
        self._is_fish_speech = (
            self._tts_stage is not None
            and getattr(getattr(self._tts_stage, "engine_args", None), "model_stage", None) == "fish_speech_slow_ar"
        )
        self._fish_speech_tokenizer = None
        self._covo_audio_tokenizer = None
        # Cached per process: the CosyVoice3 Qwen tokenizer + resolved model
        # path used for dynamic-token sizing. Without this, every request
        # re-ran snapshot_download + reloaded the tokenizer (~100 ms on the
        # TTFP critical path) in _apply_cosyvoice3_dynamic_tokens.
        self._cosyvoice3_tokenizer = None

        self._is_cosyvoice3 = (
            self._tts_stage is not None
            and getattr(getattr(self._tts_stage, "engine_args", None), "model_stage", None)
            in _COSYVOICE3_TTS_MODEL_STAGES
        )
        # Determine TTS model type or None
        self._tts_model_type = self._detect_tts_model_type()
        self.precomputed_speakers = self._load_precomputed_speakers()

        # Sub-variant inside the full MOSS-TTS family. We collapse all five
        # variants onto the same _tts_model_type="moss_tts" because they share
        # the same stage layout, but request validation + param building
        # differ per HF repo (voice-clone vs dialogue vs ambient-sound vs
        # instruction vs streaming voice-clone).
        self._moss_variant = self._detect_moss_variant() if self._tts_model_type == "moss_tts" else None

        # GLM-TTS lazy-cached resources (populated on first GLM-TTS request)
        self._glm_tts_text_tokenizer: object | None = None
        self._glm_tts_text_frontend: object | None = None

        # Cache TTS configuration values (computed once, reused per request)
        self._max_instructions_length = self._compute_max_instructions_length()

        # Merge built-in speakers into the set initialized by _init_speaker_storage.
        self.supported_speakers |= self._load_supported_speakers()
        self.supported_speakers |= set(self.precomputed_speakers)

        self.supported_languages = self._load_supported_languages()

        self._tts_tokenizer = None
        self._voxcpm2_tokenizer = None
        self._voxcpm2_split_map: dict[int, list[int]] = {}

        logger.info("Loaded %d supported speakers: %s", len(self.supported_speakers), sorted(self.supported_speakers))

        # Batch configuration
        self._batch_max_items: int = getattr(self.engine_client, "tts_batch_max_items", 32)

        # Load speech tokenizer codec parameters for prompt length estimation
        self._codec_frame_rate: float | None = self._load_codec_frame_rate()

        # Shared thread pool executor for blocking TTS preprocessing
        # operations. max_workers=1 serializes tokenizer access to avoid
        # Rust RefCell "Already borrowed" errors from concurrent use.
        self._tts_executor = ThreadPoolExecutor(max_workers=1)
        self._build_voxtral_prompt_async = make_async(self._build_voxtral_prompt, executor=self._tts_executor)
        self._build_fish_speech_prompt_async = make_async(self._build_fish_speech_prompt, executor=self._tts_executor)
        self._estimate_prompt_len_async = make_async(self._estimate_prompt_len, executor=self._tts_executor)

        # Resolve the per-model serving adapter (RFC #4327), keyed on the
        # detected model-type. Every dedicated TTS model has an adapter; the
        # adapter owns request validation + prompt/param building. Sampling
        # overrides and the model-type label remain in the orchestrator tail
        # (keyed on ``_tts_model_type``) during this incremental migration.
        self._adapter = None
        if self._tts_stage is not None:
            adapter_cls = resolve_adapter(self._tts_model_type)
            if adapter_cls is not None:
                ctx = SpeechServingContext(server=self, engine_client=self.engine_client)
                self._adapter = adapter_cls(ctx)
                logger.info("Resolved TTS serving adapter: %s", adapter_cls.__name__)

    def _get_tts_adapter(self):
        """Return the per-model serving adapter for the current ``_tts_model_type``.

        Resolved lazily (rebuilt if ``_tts_model_type`` changed since the cached
        instance was built) so callers that set ``_tts_model_type`` after
        construction still dispatch to the matching adapter. In production
        ``_tts_model_type`` is fixed at init, so the cached instance is reused.
        """
        adapter_cls = resolve_adapter(self._tts_model_type)
        if adapter_cls is None:
            self._adapter = None
            return None
        if self._adapter is None or type(self._adapter) is not adapter_cls:
            ctx = SpeechServingContext(server=self, engine_client=self.engine_client)
            self._adapter = adapter_cls(ctx)
        return self._adapter

    async def warmup(self) -> None:
        """Run a synthetic speech request to trigger all first-request warmup.

        Unlike qwen3-tts, whose CUDA Graph warmup targets a standalone tokenizer
        decoder (no vLLM dependencies) and can complete entirely at model-init
        time, VoxCPM2 needs to warm up PagedAttention scaffold/residual LLMs.
        Their CUDA Graph capture requires a vLLM ``ForwardContext``
        (attn_metadata, slot_mapping, etc.) that only exists during real
        inference steps.  The same request also pays the one-time torch.compile
        JIT tax for the LocDiT estimator, feat_encoder, AudioVAE decoder, and
        projection helpers.

        For VoxCPM2 this shifts ~15s of torch.compile + CUDA Graph capture from
        the first user request to server startup.
        """
        if self._tts_model_type != "voxcpm2":
            return

        t0 = time.time()
        logger.info("Running warmup speech request for model_type=%s", self._tts_model_type)
        # VoxCPM2 has no predefined speaker presets — "default" means zero-shot
        # mode (no voice cloning).  The voice field is required by the OpenAI
        # API schema but semantically ignored by the model.
        warmup_req = OpenAICreateSpeechRequest(
            input="Warmup.",
            voice="default",
            response_format="wav",
            speed=1.0,
            stream=False,
            model=self.model_name,
        )
        try:
            _audio_bytes, _media_type = await self._generate_audio_bytes(warmup_req, request_id="speech-warmup")
        except Exception as exc:
            logger.warning("Speech warmup failed (non-fatal): %s", exc)
            return

        elapsed = time.time() - t0
        logger.info("Speech warmup complete in %.1fs", elapsed)

    def _get_qwen_tts_expected_speaker_embedding_dim(self) -> int | None:
        """Return the loaded Qwen3-TTS speaker embedding dim, if known.

        The user-provided speaker embedding is concatenated directly with
        talker codec embeddings, so the real compatibility requirement is the
        talker hidden size.
        """
        if self._tts_model_type != "qwen3_tts":
            return None
        hf_config = self.engine_client.model_config.hf_config
        talker_config = hf_config.talker_config
        return int(talker_config.hidden_size)

    def _validate_qwen_tts_speaker_embedding_dim(self, emb_dim: int) -> str | None:
        expected_dim = self._get_qwen_tts_expected_speaker_embedding_dim()
        if expected_dim is None:
            return None
        if emb_dim != expected_dim:
            return f"speaker_embedding has {emb_dim} dimensions; expected {expected_dim} for the loaded Qwen3-TTS model"
        return None

    def _load_codec_frame_rate(self) -> float | None:
        """Load codec frame rate from speech tokenizer config for prompt length estimation."""
        if self._tts_model_type == "ming_tts":
            try:
                from vllm_omni.model_executor.models.ming_tts.config_ming_tts import MingTTSConfig

                hf_config = self.engine_client.model_config.hf_config
                ming_cfg = MingTTSConfig.from_hf_config(hf_config)
                patch_size = int(ming_cfg.patch_size)
                audio_frame_hop = int(ming_cfg.audio_frame_hop)
                sample_rate = int(ming_cfg.sample_rate)
                if patch_size <= 0 or audio_frame_hop <= 0 or sample_rate <= 0:
                    raise ValueError(
                        "Ming config has invalid tokenizer timing values: "
                        f"patch_size={patch_size}, audio_frame_hop={audio_frame_hop}, sample_rate={sample_rate}"
                    )
                rate = float(sample_rate) / float(audio_frame_hop * patch_size)
                logger.info(
                    "Derived Ming codec frame rate: %.1f Hz (sample_rate=%s, audio_frame_hop=%s, patch_size=%s)",
                    rate,
                    sample_rate,
                    audio_frame_hop,
                    patch_size,
                )
                return rate
            except Exception as e:
                logger.warning(f"Failed to derive Ming codec frame rate from hf_config: {e}")

        try:
            model_path = self.engine_client.model_config.model
            st_config_path = os.path.join(model_path, "speech_tokenizer", "config.json")
            if not os.path.exists(st_config_path):
                st_config_path = cached_file(model_path, "speech_tokenizer/config.json")
            if st_config_path is not None and os.path.exists(st_config_path):
                with open(st_config_path) as f:
                    st_config = json.load(f)
                output_sr = st_config.get("output_sample_rate")
                downsample = st_config.get("encode_downsample_rate")
                if output_sr and downsample and downsample > 0:
                    rate = float(output_sr) / float(downsample)
                    logger.info(
                        "Loaded codec frame rate: %.1f Hz (output_sample_rate=%s, encode_downsample_rate=%s)",
                        rate,
                        output_sr,
                        downsample,
                    )
                    return rate
        except Exception as e:
            logger.warning("Failed to load codec frame rate from speech tokenizer config: %s", e)

        # Fallback: try codec_frame_rate_hz from hf_config
        try:
            hf_config = self.engine_client.model_config.hf_config
            rate = getattr(hf_config, "codec_frame_rate_hz", None)
            if rate is not None:
                logger.info("Using codec frame rate from hf_config: %s Hz", rate)
                return float(rate)
        except Exception:
            pass
        return None

    def shutdown(self) -> None:
        """Shut down the TTS thread pool executor."""
        if self._tts_executor is not None:
            self._tts_executor.shutdown(wait=False, cancel_futures=True)
            self._tts_executor = None
        for name in list(self.uploaded_speakers.keys()):
            self._speaker_cache.clear(name)

    def _find_tts_stage(self):
        """Find and return the TTS stage config, or None if not found."""
        for stage in self.engine_client.stage_configs:
            engine_args = stage.engine_args
            model_stage = engine_args.model_stage
            model_arch = getattr(engine_args, "model_arch", None)
            worker_type = getattr(engine_args, "worker_type", None)
            if model_stage in _TTS_MODEL_STAGES:
                return stage
            # Ming dense identifies its AR entry stage by architecture because
            # it does not use a dedicated TTS model_stage value.
            if model_arch in _MING_TTS_MODEL_ARCHS and worker_type == "ar":
                return stage
        return None

    def _detect_tts_model_type(self) -> str | None:
        """Detect TTS model type from the stage's model_stage attribute."""
        if self._tts_stage is None:
            return None
        model_stage = getattr(self._tts_stage.engine_args, "model_stage", None)
        model_arch = getattr(self._tts_stage.engine_args, "model_arch", None)
        if model_arch == "VoxCPM2TalkerForConditionalGeneration":
            return "voxcpm2"
        if model_stage in _QWEN3_TTS_MODEL_STAGES:
            return "qwen3_tts"
        if model_stage in _VOXTRAL_TTS_MODEL_STAGES:
            return "voxtral_tts"
        if model_stage in _FISH_TTS_MODEL_STAGES:
            return "fish_tts"
        if model_stage in _COSYVOICE3_TTS_MODEL_STAGES:
            return "cosyvoice3"
        if model_stage in _OMNIVOICE_TTS_MODEL_STAGES:
            return "omnivoice"
        if model_stage in _COVO_AUDIO_MODEL_STAGES:
            if model_arch and "CovoAudio" in model_arch:
                return "covo_audio"
        if model_stage in _VOXCPM2_TTS_MODEL_STAGES:
            return "voxcpm2"
        if model_stage in _MING_TTS_MODEL_STAGES:
            return "ming_flash_omni_tts"
        if model_arch in _MING_TTS_MODEL_ARCHS:
            return "ming_tts"
        if model_stage in _MOSS_TTS_MODEL_STAGES:
            return "moss_tts_nano"
        if model_stage in _MOSS_TTS_FULL_MODEL_STAGES:
            return "moss_tts"
        if model_stage in _MOSS_TTS_LOCAL_MODEL_STAGES:
            return "moss_tts"
        if model_stage in _HIGGS_AUDIO_V2_TTS_MODEL_STAGES:
            return "higgs_audio_v2"
        if model_stage in _HIGGS_V3_TTS_MODEL_STAGES:
            return "higgs_audio_v3"
        if model_stage in _GLM_TTS_MODEL_STAGES:
            return "glm_tts"
        if model_stage in _STEP_AUDIO2_TTS_MODEL_STAGES:
            return "step_audio2"
        if model_stage in _INDEXTTS2_TTS_MODEL_STAGES:
            return "indextts2"
        return None

    def _get_custom_voice_dir(self) -> str | None:
        try:
            value = getattr(self.engine_client.model_config.hf_config, "custom_voice_dir", None)
        except AttributeError:
            return None
        if isinstance(value, os.PathLike):
            return os.fspath(value)
        if isinstance(value, str) and value:
            return value
        return None

    def _load_precomputed_speakers(self) -> dict[str, dict[str, Any]]:
        """Load precomputed voice names from ``custom_voice_dir`` for API validation."""
        if self._tts_model_type not in ("qwen3_tts", "voxcpm2"):
            return {}
        custom_voice_dir = self._get_custom_voice_dir()
        if not custom_voice_dir:
            return {}

        profiles: dict[str, dict[str, Any]] = {}
        qwen3_embedding_dim = self._get_qwen_tts_expected_speaker_embedding_dim()
        for profile in iter_custom_voice_profiles(custom_voice_dir, expected_model_type=self._tts_model_type):
            tensors = load_validated_profile_tensors(
                profile,
                expected_model_type=self._tts_model_type,
                qwen3_embedding_dim=qwen3_embedding_dim,
            )
            if tensors is None:
                continue
            profiles[profile["voice_name_lower"]] = profile
        if profiles:
            logger.info(
                "Loaded %d precomputed %s voice profile(s) from %s",
                len(profiles),
                self._tts_model_type,
                custom_voice_dir,
            )
        return profiles

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

    def _load_supported_speakers(self) -> set[str]:
        """Load supported speakers (case-insensitive) from the model configuration."""
        if self._tts_model_type == "ming_flash_omni_tts":
            # Ming-flash-omni drives speaker selection via the caption JSON
            # (audio_sequence[0]["说话人"]) rather than a spk_id table, so there
            # is no static speaker list to surface here.
            return set()
        try:
            if self._tts_model_type == "glm_tts":
                return set()
            if self._tts_model_type == "ming_tts":
                return set()
            if self._tts_model_type == "voxcpm2":
                return {"default"}
            if self._tts_model_type == "voxtral_tts":
                config = self.engine_client.model_config.hf_config.audio_config
            else:
                # Default is qwen3_tts path
                config = self.engine_client.model_config.hf_config.talker_config

            # Check for speakers in either spk_id or speaker_id
            for attr_name in ["spk_id", "speaker_id"]:
                if isinstance(config, dict):
                    speakers_dict = config.get(attr_name)
                else:
                    speakers_dict = getattr(config, attr_name, None)
                if speakers_dict and isinstance(speakers_dict, dict):
                    return {speaker.lower() for speaker in speakers_dict.keys()}

            logger.warning("No speakers found in config (checked spk_id and speaker_id)")
        except Exception as e:
            logger.warning("Could not load speakers from model config: %s", e)

        return set()

    def _load_supported_languages(self) -> frozenset[str]:
        """Load supported languages (title-cased) from the model configuration"""
        if self._tts_model_type != "qwen3_tts":
            return _TTS_LANGUAGES
        try:
            config = self.engine_client.model_config.hf_config.talker_config

            if isinstance(config, dict):
                codec_language_id = config.get("codec_language_id")
            else:
                codec_language_id = getattr(config, "codec_language_id", None)

            if codec_language_id and isinstance(codec_language_id, Mapping):
                return frozenset(str(language).title() for language in codec_language_id) | {"Auto"}

            logger.warning("No codec_language_id found in talker_config; falling back to default languages")
        except Exception as e:
            logger.warning("Could not load languages from model config: %s", e)
        return _TTS_LANGUAGES

    def _estimate_ref_code_len(self, ref_audio: object) -> int | None:
        """Estimate ref_code length from ref_audio waveform without running the codec.

        The codec produces one frame per (output_sample_rate / encode_downsample_rate)
        audio samples, so ref_code_len = ceil(duration_seconds * codec_frame_rate).
        """
        if self._codec_frame_rate is None:
            return None
        try:
            # ref_audio comes from tts_params as [[wav_array, sr]] or similar nested structure
            item = ref_audio
            while isinstance(item, list) and item:
                if len(item) == 2 and isinstance(item[1], (int, float)):
                    break
                item = item[0]
            if isinstance(item, list) and len(item) == 2:
                wav, sr = item
            elif isinstance(item, tuple) and len(item) == 2:
                wav, sr = item
            else:
                return None
            sr = int(sr)
            if hasattr(wav, "__len__"):
                n_samples = len(wav)
            elif hasattr(wav, "shape"):
                n_samples = wav.shape[-1] if wav.ndim > 1 else wav.shape[0]
            else:
                return None
            if sr <= 0 or n_samples <= 0:
                return None
            duration = n_samples / sr
            return math.ceil(duration * self._codec_frame_rate)
        except Exception:
            return None

    def _estimate_prompt_len(self, tts_params: dict[str, Any]) -> int:
        """Estimate prompt length so the placeholder matches model-side embeddings."""
        try:
            from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
                Qwen3TTSPromptEmbedsBuilder,
            )

            if self._tts_tokenizer is None:
                from transformers import AutoTokenizer

                model_name = self.engine_client.model_config.model
                self._tts_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    padding_side="left",
                )
            hf_config = self.engine_client.model_config.hf_config
            talker_config = hf_config.talker_config
            task_type = (tts_params.get("task_type") or ["CustomVoice"])[0]
            return Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
                additional_information=tts_params,
                task_type=task_type,
                tokenize_prompt=lambda t: self._tts_tokenizer(t, padding=False)["input_ids"],
                codec_language_id=getattr(talker_config, "codec_language_id", None),
                spk_is_dialect=getattr(talker_config, "spk_is_dialect", None),
                estimate_ref_code_len=self._estimate_ref_code_len,
            )
        except Exception as e:
            logger.warning("Failed to estimate TTS prompt length, using fallback 2048: %s", e)
            return 2048

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

    def _estimate_fish_ref_code_len(self, ref_audio: object) -> int | None:
        """Estimate Fish Speech semantic token length from raw reference audio."""
        from vllm_omni.model_executor.models.fish_speech.dac_utils import (
            DAC_HOP_LENGTH,
            DAC_SAMPLE_RATE,
        )

        if not isinstance(ref_audio, (list, tuple)) or len(ref_audio) != 2:
            return None
        wav, sr = ref_audio
        sr = int(sr)
        n_samples = len(wav)
        if sr <= 0 or n_samples <= 0:
            return None
        resampled_len = max(1, math.ceil(n_samples * DAC_SAMPLE_RATE / sr))
        return max(1, math.ceil(resampled_len / DAC_HOP_LENGTH))

    def _estimate_fish_prompt_len(self, text: str, ref_text: str, ref_audio: object) -> int:
        """Estimate Fish Speech clone prompt length without encoding reference audio."""
        try:
            from transformers import AutoTokenizer

            if self._fish_speech_tokenizer is None:
                model_name = self.engine_client.model_config.model
                self._fish_speech_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

            tokenizer = self._fish_speech_tokenizer
            semantic_len = self._estimate_fish_ref_code_len(ref_audio)
            if semantic_len is None:
                raise ValueError("Failed to estimate Fish Speech semantic token length")
            return estimate_fish_voice_clone_prompt_len_from_normalized(tokenizer, text, ref_text, semantic_len)
        except Exception as e:
            logger.warning("Failed to estimate Fish Speech prompt length, using fallback 2048: %s", e)
            return 2048

    def _voice_created_at(self, voice_lower: str) -> int:
        """Return the upload timestamp of an uploaded voice, or 0 for built-ins.

        Plumbed through to the model-side cache key so that delete + re-upload
        of the same name yields a fresh cache slot.
        """
        info = self.uploaded_speakers.get(voice_lower)
        return int(info.get("created_at", 0)) if info else 0

    async def _build_voxcpm2_prompt(
        self,
        request: OpenAICreateSpeechRequest,
        *,
        uploaded_ref: tuple[np.ndarray, int] | None = None,
    ) -> dict[str, Any]:
        """Build prefill prompt for VoxCPM2 TTS (`prompt_token_ids` padded to full prefill length).

        ``uploaded_ref`` supplies the audio for uploaded voices (no explicit
        ``ref_audio`` in the request) so prefill length includes it.
        """
        from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import build_voxcpm2_prompt

        self._voxcpm2_encode("")  # lazy-init tokenizer + split_map
        ref_audio = None
        ref_sr = None
        voice_profile = None
        if request.ref_audio is not None:
            ref_audio, ref_sr = await self._resolve_ref_audio(request.ref_audio)
        elif uploaded_ref is not None:
            wav_np, ref_sr = uploaded_ref
            ref_audio = wav_np.tolist()
        elif request.voice is not None:
            voice_profile = self.precomputed_speakers.get(request.voice.lower())
        return build_voxcpm2_prompt(
            hf_config=self.engine_client.model_config.hf_config,
            tokenizer=self._voxcpm2_tokenizer,
            split_map=self._voxcpm2_split_map,
            text=request.input,
            ref_audio=ref_audio,
            ref_sr=ref_sr,
            ref_text=request.ref_text,
            voice_profile=voice_profile,
        )

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
        self.supported_speakers.discard(voice_name_lower)
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
            self.supported_speakers.add(voice_name_lower)

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
        if self._tts_model_type == "ming_tts":
            if emb_dim != 192:
                logger.warning("speaker_embedding has %d dimensions; Ming dense expects 192", emb_dim)
        else:
            dim_err = self._validate_qwen_tts_speaker_embedding_dim(emb_dim)
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
            self.supported_speakers.add(voice_name_lower)

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
            self.supported_speakers.discard(voice_name_lower)
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
        if self._tts_model_type == "step_audio2":
            # StepAudio2 only requires non-empty input text
            if not request.input or not request.input.strip():
                return "Input text cannot be empty"
            return None
        if self._tts_model_type == "voxtral_tts":
            return self._validate_voxtral_tts_request(request)
        if self._tts_model_type == "fish_tts":
            return self._validate_fish_tts_request(request)
        if self._tts_model_type == "cosyvoice3":
            return self._validate_cosyvoice3_request(request)
        if self._tts_model_type == "voxcpm2":
            return self._validate_voxcpm2_request(request)
        if self._tts_model_type == "ming_flash_omni_tts":
            return self._validate_ming_flash_omni_tts_request(request)
        if self._tts_model_type == "ming_tts":
            return self._validate_ming_tts_request(request)
        if self._tts_model_type in ("moss_tts_nano", "moss_tts"):
            return self._validate_moss_tts_request(request)
        if self._tts_model_type == "glm_tts":
            return self._validate_glm_tts_request(request)
        return self._validate_qwen_tts_request(request)

    def _validate_voxcpm2_request(self, request: OpenAICreateSpeechRequest) -> str | None:
        """Validate VoxCPM2 request parameters. Returns error message or None."""
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        if request.voice is not None:
            request.voice = request.voice.lower()
            available_voices = set(self.uploaded_speakers) | set(self.precomputed_speakers) | {"default"}
            if request.voice not in available_voices:
                supported = ", ".join(sorted(available_voices)) or "none"
                return f"Invalid voice '{request.voice}'. Supported: {supported}"

        if request.max_new_tokens is not None:
            if request.max_new_tokens < _TTS_MAX_NEW_TOKENS_MIN:
                return f"max_new_tokens must be at least {_TTS_MAX_NEW_TOKENS_MIN}"
            if request.max_new_tokens > _TTS_MAX_NEW_TOKENS_MAX:
                return f"max_new_tokens cannot exceed {_TTS_MAX_NEW_TOKENS_MAX}"

        return None

    def _voxcpm2_encode(self, text: str) -> list[int]:
        """Tokenize text for VoxCPM2, splitting multichar Chinese tokens."""
        from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import (
            build_cjk_split_map,
            split_multichar_chinese,
        )

        if self._voxcpm2_tokenizer is None:
            from transformers import AutoTokenizer

            model_name = self.engine_client.model_config.model
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self._voxcpm2_split_map = build_cjk_split_map(tokenizer)
            self._voxcpm2_tokenizer = tokenizer
            logger.info("VoxCPM2 serving: built multichar split map (%d entries)", len(self._voxcpm2_split_map))

        ids = self._voxcpm2_tokenizer.encode(text, add_special_tokens=True)
        return split_multichar_chinese(ids, self._voxcpm2_split_map)

    def _validate_ming_flash_omni_tts_request(self, request: OpenAICreateSpeechRequest) -> str | None:
        """Validate Ming-flash-omni standalone-talker request parameters."""
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"
        if request.instructions is not None:
            if not isinstance(request.instructions, str):
                return "instructions must be a string"
            if len(request.instructions) > self._max_instructions_length:
                return f"instructions exceeds max length {self._max_instructions_length}"

        if request.task_type is not None:
            return "'task_type' is not supported for Ming-flash-omni TTS"
        if request.language is not None:
            return "'language' is not supported for Ming-flash-omni TTS (language is inferred from input text)"
        if request.x_vector_only_mode is not None:
            return "'x_vector_only_mode' is not supported for Ming-flash-omni TTS"
        if request.initial_codec_chunk_frames is not None:
            return "'initial_codec_chunk_frames' is not supported for Ming-flash-omni TTS"

        # Per-request voice cloning from raw audio is not yet wired up: Ming
        # extracts spk_emb / prompt_wav_lat / prompt_wav_emb model-side via
        # register_prompt_wav() at engine init. For ad-hoc cloning, callers
        # should pre-compute speaker_embedding and pass it directly.
        if request.ref_audio is not None:
            return (
                "'ref_audio' is not yet supported for Ming-flash-omni TTS; "
                "use a preset 'voice' or 'speaker_embedding' instead"
            )
        if request.ref_text is not None:
            return "'ref_text' is not yet supported for Ming-flash-omni TTS"

        if request.max_new_tokens is not None and request.max_new_tokens <= 0:
            return "'max_new_tokens' must be a positive integer"
        return None

    def _validate_ref_audio_format(self, ref_audio: str) -> str | None:
        """Validate ref_audio is a supported URI format. Returns error or None."""
        if not isinstance(ref_audio, str):
            return "ref_audio must be a URL (http/https), base64 data URL (data:...), or file URI (file://...)"
        if not (
            ref_audio.startswith(("http://", "https://"))
            or ref_audio.startswith("data:")
            or ref_audio.startswith("file://")
        ):
            return "ref_audio must be a URL (http/https), base64 data URL (data:...), or file URI (file://...)"
        return None

    def _validate_voxtral_tts_request(self, request: OpenAICreateSpeechRequest) -> str | None:
        """Validate Voxtral TTS request parameters. Returns error message or None."""
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        # Voxtral TTS requires either a preset voice or ref_audio for voice cloning.
        if request.voice is None and request.ref_audio is None:
            return "Either 'voice' (preset speaker) or 'ref_audio' (voice cloning) must be provided"

        if request.ref_audio is not None:
            fmt_err = self._validate_ref_audio_format(request.ref_audio)
            if fmt_err:
                return fmt_err

        if request.voice is not None:
            request.voice = request.voice.lower()
            if self.supported_speakers and request.voice not in self.supported_speakers:
                return f"Invalid speaker '{request.voice}'. Supported: {', '.join(sorted(self.supported_speakers))}"

        if request.max_new_tokens is not None:
            if request.max_new_tokens < _TTS_MAX_NEW_TOKENS_MIN:
                return f"max_new_tokens must be at least {_TTS_MAX_NEW_TOKENS_MIN}"
            if request.max_new_tokens > _TTS_MAX_NEW_TOKENS_MAX:
                return f"max_new_tokens cannot exceed {_TTS_MAX_NEW_TOKENS_MAX}"

        return None

    def _validate_qwen_tts_request(self, request: OpenAICreateSpeechRequest) -> str | None:
        """Validate Qwen TTS request parameters. Returns error message or None."""
        # Infer Base task when ref_audio or ref_text is provided without explicit task_type.
        if request.task_type is None and (request.ref_audio is not None or request.ref_text is not None):
            request.task_type = "Base"

        # Normalize voice to lowercase for case-insensitive matching
        if request.voice is not None:
            request.voice = request.voice.lower()
            if request.task_type is None and request.voice in self.precomputed_speakers:
                request.task_type = "Base"
        task_type = request.task_type or "CustomVoice"

        # Validate input is not empty
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        # Validate language (case-insensitive; normalized to the title-cased config form)
        if request.language is not None:
            request.language = request.language.title()
            if request.language not in self.supported_languages:
                return (
                    f"Invalid language '{request.language}'. Supported: {', '.join(sorted(self.supported_languages))}"
                )

        # Validate speaker for CustomVoice task
        if task_type == "CustomVoice":
            if not self.supported_speakers:
                return (
                    "This model does not support CustomVoice task (no speakers configured). "
                    "Use task_type='Base' with ref_audio/ref_text for voice cloning, "
                    "or use a CustomVoice model."
                )
            if request.voice is not None and request.voice not in self.supported_speakers:
                return f"Invalid voice '{request.voice}'. Supported: {', '.join(sorted(self.supported_speakers))}"

        # Validate speaker_embedding constraints
        if request.speaker_embedding is not None:
            if task_type != "Base":
                return "'speaker_embedding' is only valid for Base task"
            if not request.speaker_embedding:
                return "'speaker_embedding' must be a non-empty list of floats"
            # speaker_embedding implies x_vector_only_mode — set it before
            # Base task validation so callers don't need to pass it explicitly.
            request.x_vector_only_mode = True
            emb_len = len(request.speaker_embedding)
            dim_err = self._validate_qwen_tts_speaker_embedding_dim(emb_len)
            if dim_err is not None:
                return dim_err
        # Validate Base task requirements
        if task_type == "Base":
            if request.voice is None:
                # 1. Ensure a voice source is provided
                if request.ref_audio is None and getattr(request, "speaker_embedding", None) is None:
                    return "Base task requires 'ref_audio' or 'speaker_embedding' for voice cloning"
                # 2. Validate ref_audio format if it exists (using the helper from main)
                if request.ref_audio is not None:
                    fmt_err = self._validate_ref_audio_format(request.ref_audio)
                    if fmt_err:
                        return fmt_err
                # 3. Validate text requirements based on the mode
                if not getattr(request, "x_vector_only_mode", False):
                    if not request.ref_text or not request.ref_text.strip():
                        return (
                            "Base task requires non-empty 'ref_text' (transcript of "
                            "the reference audio) unless 'x_vector_only_mode' is enabled"
                        )
            else:
                voice_lower = request.voice.lower()
                if voice_lower in self.uploaded_speakers:
                    # Check if data file exists for uploaded speaker
                    speaker_info = self.uploaded_speakers[voice_lower]
                    file_path = Path(speaker_info["file_path"])
                    if not file_path.exists():
                        return f"Data file for uploaded speaker '{request.voice}' not found on disk"
                elif voice_lower in self.precomputed_speakers:
                    profile = self.precomputed_speakers[voice_lower]
                    mode = str(profile.get("mode") or "xvec").lower()
                    ref_text = request.ref_text or profile.get("ref_text")
                    if mode == "icl" and (not isinstance(ref_text, str) or not ref_text.strip()):
                        return (
                            f"Precomputed voice '{request.voice}' uses ICL mode but has no ref_text in "
                            "the request or custom voice manifest"
                        )
                else:
                    # need ref_audio for built-in speaker
                    if request.ref_audio is None:
                        return (
                            f"Base task with built-in speaker '{request.voice}' requires 'ref_audio' for voice cloning"
                        )
                    fmt_err = self._validate_ref_audio_format(request.ref_audio)
                    if fmt_err:
                        return fmt_err
                    if not getattr(request, "x_vector_only_mode", False) and (
                        not request.ref_text or not request.ref_text.strip()
                    ):
                        return (
                            "Base task requires non-empty 'ref_text' (transcript of "
                            "the reference audio) unless 'x_vector_only_mode' is enabled"
                        )

        # Validate cross-parameter dependencies
        if task_type != "Base":
            if request.ref_text is not None:
                return "'ref_text' is only valid for Base task"
            if request.x_vector_only_mode is not None:
                return "'x_vector_only_mode' is only valid for Base task"

        # Validate VoiceDesign task requirements
        if task_type == "VoiceDesign" and not request.instructions:
            return "VoiceDesign task requires 'instructions' to describe the voice"

        # Validate instructions length (using cached value from initialization)
        if request.instructions and len(request.instructions) > self._max_instructions_length:
            return f"Instructions too long (max {self._max_instructions_length} characters)"

        # Validate max_new_tokens range
        if request.max_new_tokens is not None:
            if request.max_new_tokens < _TTS_MAX_NEW_TOKENS_MIN:
                return f"max_new_tokens must be at least {_TTS_MAX_NEW_TOKENS_MIN}"
            if request.max_new_tokens > _TTS_MAX_NEW_TOKENS_MAX:
                return f"max_new_tokens cannot exceed {_TTS_MAX_NEW_TOKENS_MAX}"

        return None

    def _detect_moss_variant(self) -> str:
        """Sub-classify a ``moss_tts``-stage server into the actual MOSS-TTS
        variant family (tts, ttsd, sound_effect, voice_generator, realtime).

        Detection key is the HF repo path / model_name; matches
        ``_try_resolve_omni_model_type`` in entrypoints/utils.py so users get
        consistent behaviour no matter how they launched the server (--model
        OpenMOSS-Team/MOSS-TTSD-v1.0 vs --deploy-config moss_ttsd.yaml).
        """
        try:
            name = (self.engine_client.model_config.model or "").lower().replace("-", "").replace("_", "")
        except Exception:
            name = ""
        if "realtime" in name:
            return "realtime"
        if "local" in name:
            return "local"
        if "ttsd" in name:
            return "ttsd"
        if "soundeffect" in name:
            return "sound_effect"
        if "voicegenerator" in name:
            return "voice_generator"
        return "tts"

    def _validate_moss_tts_request(self, request: OpenAICreateSpeechRequest) -> str | None:
        """Validate any MOSS-TTS-family request (nano + 5 full variants).

        Dispatches by ``self._moss_variant``:
          - ``tts``/``realtime``: require ``ref_audio`` (voice cloning).
          - ``ttsd``: require ``ref_audio`` (speaker 1); ``ref_audio_2``
            optional (defaults to the same ref for both speakers).
          - ``sound_effect``: require ``ambient_sound`` (no ref_audio).
          - ``voice_generator``: require ``instructions`` (no ref_audio).
          - For the legacy moss_tts_nano model_type the variant is None and
            we fall through to the original nano contract (ref_audio only).
        """
        if not request.input or not request.input.strip():
            # SoundEffect can legitimately have empty input (just ambient_sound).
            if self._moss_variant != "sound_effect":
                return "Input text cannot be empty"

        v = self._moss_variant
        if v in (None, "tts", "realtime", "local"):
            if request.ref_audio is None:
                label = (
                    "MOSS-TTS-Nano"
                    if v is None
                    else (
                        "MOSS-TTS-Realtime"
                        if v == "realtime"
                        else ("MOSS-TTS-Local-Transformer" if v == "local" else "MOSS-TTS")
                    )
                )
                return f"{label} requires 'ref_audio' (reference audio for voice cloning)."
            return self._validate_ref_audio_format(request.ref_audio)

        if v == "ttsd":
            if request.ref_audio is None:
                return "MOSS-TTSD requires 'ref_audio' (speaker 1 reference)."
            fmt_err = self._validate_ref_audio_format(request.ref_audio)
            if fmt_err:
                return fmt_err
            if request.ref_audio_2 is not None:
                return self._validate_ref_audio_format(request.ref_audio_2)
            return None

        if v == "sound_effect":
            if not request.ambient_sound or not request.ambient_sound.strip():
                return (
                    "MOSS-SoundEffect requires 'ambient_sound' (natural language "
                    "description of the sound effect to synthesise)."
                )
            return None

        if v == "voice_generator":
            if not request.instructions or not request.instructions.strip():
                return (
                    "MOSS-VoiceGenerator requires 'instructions' (natural language "
                    "voice description, e.g. 'a warm female voice with an American accent')."
                )
            return None

        return None  # unreachable

    def _get_moss_processor(self):
        """Lazily load the upstream MOSS-TTS processor once per server.

        Cached on ``self._moss_processor_cache``. The processor owns its own
        audio_tokenizer (~1.6 B params); we keep it on CPU so it doesn't
        compete with the talker (~8 GiB) and codec (~7 GiB) for our 96 GiB
        GPU — per-request ref-audio encoding is fast enough on CPU.
        """
        cached = getattr(self, "_moss_processor_cache", None)
        if cached is not None:
            return cached
        from transformers import AutoProcessor

        model_id = self.engine_client.model_config.model
        proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        if hasattr(proc, "audio_tokenizer"):
            proc.audio_tokenizer = proc.audio_tokenizer.to("cpu").eval()
        self._moss_processor_cache = proc
        return proc

    async def _build_moss_tts_params(self, request: OpenAICreateSpeechRequest) -> dict[str, Any]:
        """Build the talker prompt + ``additional_information`` payload for any
        MOSS-TTS-family request (nano + 5 full variants).

        For the legacy ``moss_tts_nano`` model_type, keeps the original nano
        contract (``{text, mode=voice_clone, prompt_audio_array}``); the
        caller still uses a ``[1]`` placeholder prompt.

        For the full MOSS-TTS family (``MossTTSDelay*`` / ``MossTTSRealtime*``)
        we **call the upstream processor server-side** to produce the unified
        ``(text_ids, audio_codes)`` shape the talker actually consumes — same
        flow as ``examples/.../moss_tts/end2end.py:_build_unified_codes``.
        Returns ``{prompt_token_ids: list[int], codes.ref: torch.LongTensor,
        max_new_frames, ...}``. The caller treats ``prompt_token_ids`` as the
        prompt and forwards the rest as ``additional_information``.
        """
        import torch  # local to avoid pulling torch at module import time

        v = self._moss_variant

        # ---- Legacy nano path (unchanged) ----
        if v is None:  # moss_tts_nano
            params: dict[str, Any] = {
                "text": [request.input or ""],
                "mode": ["voice_clone"],
            }
            if request.max_new_tokens is not None:
                params["max_new_frames"] = [request.max_new_tokens]
            wav_list, sr = await self._resolve_ref_audio(request.ref_audio)
            params["prompt_audio_array"] = [[wav_list, sr]]
            return params

        # ---- MOSS-TTS-Realtime: keep the old prompt_audio_array path ----
        # ``AutoProcessor.from_pretrained`` doesn't auto-discover
        # ``MossTTSRealtimeProcessor`` (no ``processor_config.json`` in the
        # snapshot), and Realtime's prompt format diverges from MossTTSDelay
        # (16-channel grid, separate per-step text feed). The
        # ``prompt_audio_array`` shape lines up well enough with what the
        # talker reads for short prompts; full Realtime support needs a
        # separate processor.from_module path which we don't wire here.
        if v == "realtime":
            params: dict[str, Any] = {
                "text": [request.input or ""],
                "mode": ["voice_clone"],
            }
            if request.max_new_tokens is not None:
                params["max_new_frames"] = [request.max_new_tokens]
            wav_list, sr = await self._resolve_ref_audio(request.ref_audio)
            params["prompt_audio_array"] = [[wav_list, sr]]
            return params

        # ---- MossTTSDelay family (tts/ttsd/sound_effect/voice_generator)
        # and MOSS-TTS-Local-Transformer-v1.5: call the upstream processor
        # server-side to produce unified codes. Local-v1.5 ships its own
        # AutoProcessor (processor_config.json + processing_moss_tts.py) and
        # reuses this exact build_user_message/encode_audios_from_wav path in
        # the offline example (examples/.../moss_tts/end2end.py:
        # _build_unified_codes) -- it is NOT in the same boat as Realtime
        # (no processor_config.json there), so it must not fall back to the
        # prompt_audio_array path above (which the talker's preprocess()
        # never reads -- info_dict["codes"]["ref"] is the only thing it
        # consumes, so skipping this path silently drops all voice-clone
        # conditioning and produces unconditioned/garbage audio online). ----
        proc = self._get_moss_processor()
        n_vq = int(getattr(proc.model_config, "n_vq", 32))
        # Local-v1.5 encodes reference audio at a fixed 24 kHz working rate
        # regardless of its 48 kHz stereo *output* codec -- mirrors the
        # offline example's hardcoded encode_audios_from_wav(sampling_rate=24000)
        # for this variant; proc.model_config.sampling_rate there is the
        # output rate (48000), the wrong value to resample the reference into.
        sr_target = 24000 if v == "local" else int(getattr(proc.model_config, "sampling_rate", 24000))

        # Reference-audio encoding + speaker caching lives in the model package
        # (moss_tts.reference_encoder), mirroring Fish Speech / CosyVoice3 /
        # Qwen3-TTS which keep reference handling with the model rather than in
        # this shared serving file. Imported lazily so the API-server process
        # only pulls it on the delay-family path (alongside the upstream proc).
        from vllm_omni.model_executor.models.moss_tts.reference_encoder import encode_reference_codes

        _voice = getattr(request, "voice", None)
        _voice = _voice.strip() if isinstance(_voice, str) else ""
        _voice_created = self._voice_created_at(_voice.lower()) if _voice else 0

        async def _encode_ref(ref_str: str) -> torch.Tensor:
            return await encode_reference_codes(
                ref_str,
                processor=proc,
                resolve_ref_audio=self._resolve_ref_audio,
                speaker_cache=self._speaker_cache,
                variant=v,
                n_vq=n_vq,
                sr_target=sr_target,
                voice_name=_voice or None,
                voice_created_at=_voice_created,
            )

        user_kwargs: dict[str, Any] = {"text": request.input or ""}
        if v in ("tts", "local"):
            user_kwargs["reference"] = [await _encode_ref(request.ref_audio)]
        elif v == "ttsd":
            refs = [await _encode_ref(request.ref_audio)]
            if request.ref_audio_2:
                refs.append(await _encode_ref(request.ref_audio_2))
            user_kwargs["reference"] = refs
        elif v == "sound_effect":
            user_kwargs["text"] = request.input or ""  # may be empty
            user_kwargs["ambient_sound"] = request.ambient_sound or ""
            if request.duration_seconds is not None:
                user_kwargs["tokens"] = max(1, int(float(request.duration_seconds) * 12.5))
            elif request.max_new_tokens is not None:
                user_kwargs["tokens"] = int(request.max_new_tokens)
        elif v == "voice_generator":
            user_kwargs["instruction"] = request.instructions or ""

        # Optional language tag for the spoken-text variants. MOSS-TTS-v1.5's
        # headline improvement is multilingual synthesis when the language is
        # given (build_user_message(..., language=...)); 1.0 ignores it
        # gracefully. Sound-effect output is non-verbal, so skip it there.
        if v in ("tts", "ttsd", "voice_generator") and getattr(request, "language", None):
            user_kwargs["language"] = request.language

        # Build the unified-codes prompt: (L, 1+n_vq) where col 0 is text/special
        # tokens and cols 1..n_vq are the delay-pattern audio code grid (mostly
        # audio_pad_code outside the reference block).
        user_msg = proc.build_user_message(**user_kwargs)
        batch = proc(conversations=[[user_msg]], mode="generation")
        unified = batch["input_ids"][0]  # torch.LongTensor (L, 1+n_vq)
        text_ids: list[int] = unified[:, 0].tolist()
        audio_codes: torch.Tensor = unified[:, 1:].contiguous().to(torch.int64)

        params: dict[str, Any] = {
            "prompt_token_ids": text_ids,
            "codes": {"ref": audio_codes},
        }
        if request.max_new_tokens is not None:
            params["max_new_frames"] = [request.max_new_tokens]
        return params

    def _validate_higgs_audio_v2_request(self, request: OpenAICreateSpeechRequest) -> str | None:
        """Validate higgs_audio_v2 request parameters. Returns error message or None.

        Accepted: plain text -> speech, or shallow voice clone via ``ref_audio``
        + ``ref_text`` (both required together). Still out of scope: preset
        ``voice``/``speaker`` selection, ``x_vector_only_mode`` /
        ``speaker_embedding`` helpers, ``task_type``/``language``/
        ``instructions``/``speed`` overrides, and multi-speaker ``[SPEAKERn]``
        tags inside the input body.
        """
        from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_tokenizer import (
            MULTI_SPEAKER_TAG_PATTERN,
        )

        if not request.input or not request.input.strip():
            return "higgs_audio_v2: input text cannot be empty"

        # Voice clone: ref_audio and ref_text must come together.
        if request.ref_audio is not None and not request.ref_text:
            return (
                "higgs_audio_v2 voice clone requires both 'ref_audio' and "
                "'ref_text'; received ref_audio without ref_text"
            )
        if request.ref_text and request.ref_audio is None:
            return (
                "higgs_audio_v2 voice clone requires both 'ref_audio' and "
                "'ref_text'; received ref_text without ref_audio"
            )

        if request.x_vector_only_mode is not None:
            return "higgs_audio_v2 v1 does not support 'x_vector_only_mode' (voice-cloning helper field)"
        if request.speaker_embedding is not None:
            return "higgs_audio_v2 v1 does not support 'speaker_embedding' (voice-cloning helper field)"
        if request.voice and request.ref_audio is None:
            # _apply_uploaded_speaker runs before this validator; if voice was
            # an uploaded speaker, ref_audio is now populated and ref_text is
            # backfilled from the speaker entry. A bare voice= with no
            # ref_audio means the name didn't resolve to an uploaded speaker
            # (and higgs has no built-in preset voices).
            return (
                "higgs_audio_v2 v1 does not support 'voice'/'speaker' selection for built-in voices; "
                f"upload a voice first via POST /v1/audio/voices, or use ref_audio + ref_text. "
                f"Got voice={request.voice!r}"
            )
        if request.instructions:
            return (
                "higgs_audio_v2 v1 does not support 'instructions' (voice "
                "style/emotion control); supply plain text instead"
            )
        if request.task_type is not None:
            return "higgs_audio_v2 v1 does not support 'task_type'; the model is single-mode plain text -> speech"
        if request.language is not None:
            return (
                "higgs_audio_v2 v1 does not accept 'language' overrides; the model infers language from the input text"
            )
        if request.speed is not None and request.speed != 1.0:
            return (
                "higgs_audio_v2 v1 does not support 'speed' adjustments; the audio is rendered at native rate (24 kHz)"
            )

        if MULTI_SPEAKER_TAG_PATTERN.search(request.input):
            return "higgs_audio_v2 v1 does not support multi-speaker [SPEAKERn] tags; remove the tag from the input"

        if request.max_new_tokens is not None:
            if request.max_new_tokens < _TTS_MAX_NEW_TOKENS_MIN:
                return f"max_new_tokens must be at least {_TTS_MAX_NEW_TOKENS_MIN}"

    def _validate_glm_tts_request(self, request: OpenAICreateSpeechRequest) -> str | None:
        """Validate GLM-TTS request — requires ref_audio for voice cloning."""
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        if request.ref_audio is None:
            return "GLM-TTS requires 'ref_audio' for zero-shot voice cloning"
        fmt_err = self._validate_ref_audio_format(request.ref_audio)
        if fmt_err:
            return fmt_err
        if not request.ref_text or not request.ref_text.strip():
            return "GLM-TTS voice cloning requires 'ref_text' (transcript of the reference audio)"

        if request.max_new_tokens is not None:
            if request.max_new_tokens < _TTS_MAX_NEW_TOKENS_MIN:
                return f"max_new_tokens must be >= {_TTS_MAX_NEW_TOKENS_MIN}"
            if request.max_new_tokens > _TTS_MAX_NEW_TOKENS_MAX:
                return f"max_new_tokens cannot exceed {_TTS_MAX_NEW_TOKENS_MAX}"
        return None

    async def _build_higgs_audio_v2_params(self, request: OpenAICreateSpeechRequest):
        """Build prompt_token_ids for higgs_audio_v2 via the upstream processor.

        Plain-text path: runs ``build_plain_text_prompt`` and returns the
        token-only prompt. Voice-clone path (``ref_audio`` + ``ref_text``):
        resolves the reference clip via ``_resolve_ref_audio``, runs
        ``build_voice_clone_prompt`` (which encodes the clip through HF's
        ``HiggsAudioV2TokenizerModel`` loaded from the k2-fsa/OmniVoice
        ``audio_tokenizer/`` subdirectory), and attaches the encoded
        ``audio_input_ids`` + ``audio_input_ids_mask`` tensors via
        ``additional_information`` so the talker substitutes them at the
        prompt-side audio placeholders.
        """
        from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_tokenizer import (
            build_plain_text_prompt,
            build_voice_clone_prompt,
            input_ids_to_python_list,
        )

        processor = await self._resolve_higgs_audio_v2_processor()

        if request.ref_audio is None:
            inputs = await asyncio.to_thread(build_plain_text_prompt, processor, request.input)
            prompt_token_ids = input_ids_to_python_list(inputs)
            return tokens_input(prompt_token_ids=prompt_token_ids)

        wav_list, sr = await self._resolve_ref_audio(request.ref_audio)
        wav = np.asarray(wav_list, dtype=np.float32)
        out = await asyncio.to_thread(
            build_voice_clone_prompt,
            processor,
            request.input,
            wav,
            int(sr),
            request.ref_text or "",
        )
        prompt = tokens_input(prompt_token_ids=out["prompt_token_ids"])
        # Pass tensors at the top level of additional_information (NOT list-
        # wrapped). ``vllm_omni.data_entry_keys.serialize_payload`` routes
        # bare ``torch.Tensor`` values through ``_serialize_tensor``; a list
        # containing tensors would fall into the ``list_data`` field which
        # msgspec cannot serialize and the tensors would be dropped over the
        # process boundary (silent voice-clone failure).
        prompt["additional_information"] = {
            "audio_input_ids": out["audio_input_ids"],
            "audio_input_ids_mask": out["audio_input_ids_mask"],
        }
        return prompt

    async def _resolve_higgs_audio_v2_processor(self):
        """Lazy-load the AutoProcessor for higgs_audio_v2 (once per serving instance)."""
        cached = getattr(self, "_higgs_audio_v2_processor", None)
        if cached is not None:
            return cached

        from transformers import AutoProcessor

        model_path = None
        for stage in self.engine_client.stage_configs:
            model_path = getattr(getattr(stage, "engine_args", None), "model", None)
            if model_path:
                break
        if model_path is None:
            # Fallback: the orchestrator stores the served model id on the engine
            # itself (set by AsyncOmniEngine.__init__). Stage-level engine_args
            # may not surface ``model`` when the deploy yaml doesn't set it per
            # stage (the CLI-passed model id is the single source of truth).
            model_path = getattr(self.engine_client, "model", None)
        if model_path is None:
            raise RuntimeError("higgs_audio_v2 serving could not resolve the model path from the engine stage configs")
        processor = AutoProcessor.from_pretrained(model_path)
        self._higgs_audio_v2_processor = processor
        return processor

    # ---- higgs-audio v3 ----

    def _validate_higgs_audio_v3_request(self, request: OpenAICreateSpeechRequest) -> str | None:
        """Validate higgs_audio_v3 request parameters."""
        if not request.input or not request.input.strip():
            return "higgs_audio_v3: input text cannot be empty"
        if request.ref_audio is not None and not request.ref_text:
            # Voice clone ref_text is optional for v3 (improves fidelity but not required)
            pass
        if request.max_new_tokens is not None:
            if request.max_new_tokens < _TTS_MAX_NEW_TOKENS_MIN:
                return f"max_new_tokens must be at least {_TTS_MAX_NEW_TOKENS_MIN}"
        return None

    async def _build_higgs_audio_v3_params(self, request: OpenAICreateSpeechRequest):
        """Build prompt_token_ids for higgs_audio_v3.

        Plain-text path: builds ``[tts, text, tokens, audio]``.
        Voice-clone path: encodes reference audio, applies delay pattern,
        builds ``[tts, (ref_text, tokens,) ref_audio, -100xN, text, tokens, audio]``.
        """
        adapter = await self._resolve_higgs_audio_v3_adapter()

        if request.ref_audio is None:
            prompt_ids = adapter.build_prompt(request.input)
            return tokens_input(prompt_token_ids=prompt_ids)

        # Voice clone
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_tokenizer import (
            apply_delay_pattern,
            encode_reference_audio,
        )

        wav_list, sr = await self._resolve_ref_audio(request.ref_audio)
        artifact_key = self._get_resolved_ref_audio_artifact_key(request.ref_audio)
        wav = np.asarray(wav_list, dtype=np.float32)
        ref_codes_delayed, cache_hit, inflight_wait = await self._resolve_higgs_audio_v3_ref_codes(
            artifact_key,
            wav,
            int(sr),
            encode_reference_audio,
            apply_delay_pattern,
        )
        del cache_hit, inflight_wait

        prompt_ids = adapter.build_prompt(
            request.input,
            num_ref_tokens=int(ref_codes_delayed.shape[0]),
            reference_text=request.ref_text or None,
        )
        prompt = tokens_input(prompt_token_ids=prompt_ids)
        import torch

        prompt["additional_information"] = {
            "audio_input_ids": ref_codes_delayed.to(torch.long),
            "audio_input_ids_mask": torch.ones(ref_codes_delayed.shape[0], dtype=torch.bool),
        }
        return prompt

    async def _resolve_higgs_audio_v3_ref_codes(
        self,
        artifact_key: str | None,
        wav: np.ndarray,
        sr: int,
        encode_reference_audio,
        apply_delay_pattern,
    ) -> tuple[torch.Tensor, bool, bool]:
        ref_codes_delayed = self._get_higgs_audio_v3_ref_codes(artifact_key)
        if ref_codes_delayed is not None:
            return ref_codes_delayed, True, False
        if not artifact_key:
            ref_codes_raw = await asyncio.to_thread(encode_reference_audio, wav, sr)
            return apply_delay_pattern(ref_codes_raw), False, False

        task = self._higgs_audio_v3_ref_code_inflight.get(artifact_key)
        if task is not None:
            return (await task).clone(), False, True

        async def _encode_and_cache() -> torch.Tensor:
            ref_codes_raw = await asyncio.to_thread(encode_reference_audio, wav, sr)
            delayed = apply_delay_pattern(ref_codes_raw)
            self._put_higgs_audio_v3_ref_codes(artifact_key, delayed)
            cached = self._get_higgs_audio_v3_ref_codes(artifact_key)
            return cached if cached is not None else delayed.detach().to("cpu", dtype=torch.long).contiguous()

        task = asyncio.create_task(_encode_and_cache())
        self._higgs_audio_v3_ref_code_inflight[artifact_key] = task
        try:
            return (await task).clone(), False, False
        finally:
            if self._higgs_audio_v3_ref_code_inflight.get(artifact_key) is task:
                self._higgs_audio_v3_ref_code_inflight.pop(artifact_key, None)

    def _get_higgs_audio_v3_ref_codes(self, artifact_key: str | None) -> torch.Tensor | None:
        if not artifact_key:
            return None
        cached = self._higgs_audio_v3_ref_code_cache.get(artifact_key)
        if cached is None:
            return None
        self._higgs_audio_v3_ref_code_cache.move_to_end(artifact_key)
        return cached[0].clone()

    def _put_higgs_audio_v3_ref_codes(self, artifact_key: str, codes: torch.Tensor) -> None:
        if _HIGGS_V3_REF_CODE_CACHE_MAX_ENTRIES <= 0 or _HIGGS_V3_REF_CODE_CACHE_MAX_BYTES <= 0 or not artifact_key:
            return
        cached_codes = codes.detach().to("cpu", dtype=torch.long).contiguous()
        size = int(cached_codes.numel() * cached_codes.element_size())
        if size > _HIGGS_V3_REF_CODE_CACHE_MAX_BYTES:
            return
        previous = self._higgs_audio_v3_ref_code_cache.pop(artifact_key, None)
        if previous is not None:
            self._higgs_audio_v3_ref_code_cache_bytes -= previous[1]
        self._higgs_audio_v3_ref_code_cache[artifact_key] = (cached_codes, size)
        self._higgs_audio_v3_ref_code_cache_bytes += size
        while len(self._higgs_audio_v3_ref_code_cache) > _HIGGS_V3_REF_CODE_CACHE_MAX_ENTRIES:
            _, (_, old_size) = self._higgs_audio_v3_ref_code_cache.popitem(last=False)
            self._higgs_audio_v3_ref_code_cache_bytes -= old_size
        while self._higgs_audio_v3_ref_code_cache_bytes > _HIGGS_V3_REF_CODE_CACHE_MAX_BYTES:
            _, (_, old_size) = self._higgs_audio_v3_ref_code_cache.popitem(last=False)
            self._higgs_audio_v3_ref_code_cache_bytes -= old_size

    async def _resolve_higgs_audio_v3_adapter(self):
        """Lazy-load the tokenizer adapter for higgs_audio_v3."""
        cached = getattr(self, "_higgs_audio_v3_adapter", None)
        if cached is not None:
            return cached

        from transformers import AutoTokenizer

        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_tokenizer import (
            HiggsAudioV3TokenizerAdapter,
        )

        model_path = None
        for stage in self.engine_client.stage_configs:
            model_path = getattr(getattr(stage, "engine_args", None), "model", None)
            if model_path:
                break
        if model_path is None:
            model_path = getattr(self.engine_client, "model", None)
        if model_path is None:
            raise RuntimeError("higgs_audio_v3 serving could not resolve model path")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        adapter = HiggsAudioV3TokenizerAdapter(tokenizer)
        self._higgs_audio_v3_adapter = adapter
        return adapter

    def _validate_fish_tts_request(self, request: OpenAICreateSpeechRequest) -> str | None:
        """Validate Fish Speech request parameters. Returns error message or None."""
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        if request.ref_audio is not None:
            fmt_err = self._validate_ref_audio_format(request.ref_audio)
            if fmt_err:
                return fmt_err
            if not request.ref_text or not request.ref_text.strip():
                return "Voice cloning requires 'ref_text' (transcript of the reference audio)"

        if request.max_new_tokens is not None:
            if request.max_new_tokens < _TTS_MAX_NEW_TOKENS_MIN:
                return f"max_new_tokens must be at least {_TTS_MAX_NEW_TOKENS_MIN}"
            if request.max_new_tokens > _TTS_MAX_NEW_TOKENS_MAX:
                return f"max_new_tokens cannot exceed {_TTS_MAX_NEW_TOKENS_MAX}"

        return None

    def _validate_cosyvoice3_request(self, request: OpenAICreateSpeechRequest) -> str | None:
        """Validate CosyVoice3 request parameters. Returns error message or None."""
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        # CosyVoice3 requires reference audio for voice cloning
        if request.ref_audio is None:
            return "CosyVoice3 requires 'ref_audio' (reference audio for voice cloning)"

        fmt_err = self._validate_ref_audio_format(request.ref_audio)
        if fmt_err:
            return fmt_err

        if not request.ref_text or not request.ref_text.strip():
            return "CosyVoice3 requires 'ref_text' (transcript of the reference audio)"

        if request.max_new_tokens is not None:
            if request.max_new_tokens < _TTS_MAX_NEW_TOKENS_MIN:
                return f"max_new_tokens must be at least {_TTS_MAX_NEW_TOKENS_MIN}"
            if request.max_new_tokens > _TTS_MAX_NEW_TOKENS_MAX:
                return f"max_new_tokens cannot exceed {_TTS_MAX_NEW_TOKENS_MAX}"

        return None

    def _validate_ming_tts_request(self, request: OpenAICreateSpeechRequest) -> str | None:
        """Validate Ming TTS request parameters. Returns error message or None."""
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        if isinstance(request.ref_audio, list):
            return self._validate_ming_tts_podcast_request(request)
        return self._validate_ming_tts_single_speaker_request(request)

    def _validate_ming_tts_single_speaker_request(self, request: OpenAICreateSpeechRequest) -> str | None:
        if request.ref_audio is not None:
            fmt_err = self._validate_ref_audio_format(request.ref_audio)
            if fmt_err:
                return fmt_err

        if request.speaker_embedding is not None:
            if not request.speaker_embedding:
                return "'speaker_embedding' must be a non-empty list of floats"
            emb_len = len(request.speaker_embedding)
            if emb_len != 192:
                logger.warning(
                    "speaker_embedding has %d dimensions; Ming dense expects 192. "
                    "Wrong dimensions will likely fail or degrade output.",
                    emb_len,
                )

        voice_lower = request.voice.lower() if isinstance(request.voice, str) else None
        uploaded_voice = bool(voice_lower and voice_lower in self.uploaded_speakers)
        clone_source_present = request.ref_audio is not None or request.speaker_embedding is not None or uploaded_voice

        if request.task_type == "Base" and not clone_source_present:
            return "Base task requires 'ref_audio', 'speaker_embedding', or an uploaded voice sample"

        if request.ref_audio is not None and request.ref_text is not None and not request.ref_text.strip():
            return "'ref_text' must be non-empty when provided with 'ref_audio'"

        # Ming offline ref-audio cases use prompt_waveform without prompt_text;
        # keep the transcript requirement for other TTS models.
        if request.ref_audio is not None and request.speaker_embedding is None and not self._is_ming_tts_model():
            uploaded_ref_text = self.uploaded_speakers[voice_lower].get("ref_text") if uploaded_voice else None
            if not (request.ref_text and request.ref_text.strip()) and not uploaded_ref_text:
                return "Reference-audio cloning requires non-empty 'ref_text'"

        if request.ref_text is not None and request.ref_audio is None and not uploaded_voice:
            return "'ref_text' requires 'ref_audio' or an uploaded voice sample"

        if request.instructions and len(request.instructions) > self._max_instructions_length:
            return f"Instructions too long (max {self._max_instructions_length} characters)"

        if request.max_new_tokens is not None:
            if request.max_new_tokens < _TTS_MAX_NEW_TOKENS_MIN:
                return f"max_new_tokens must be at least {_TTS_MAX_NEW_TOKENS_MIN}"
            if request.max_new_tokens > _TTS_MAX_NEW_TOKENS_MAX:
                return f"max_new_tokens cannot exceed {_TTS_MAX_NEW_TOKENS_MAX}"

        return None

    def _validate_ming_tts_podcast_request(self, request: OpenAICreateSpeechRequest) -> str | None:
        if len(request.ref_audio) < 2:
            return "Podcast-style Ming requests require at least two 'ref_audio' clips"

        for ref_audio in request.ref_audio:
            fmt_err = self._validate_ref_audio_format(ref_audio)
            if fmt_err:
                return fmt_err

        if not request.ref_text or not request.ref_text.strip():
            return "Podcast-style Ming requests require non-empty 'ref_text'"

        if request.speaker_embedding is not None:
            embeddings = request.speaker_embedding
            embedding_count = len(embeddings) if embeddings and isinstance(embeddings[0], list) else 1
            if embedding_count != len(request.ref_audio):
                return (
                    "Podcast-style Ming requests require one speaker embedding per ref_audio clip; "
                    f"got {embedding_count} embeddings for {len(request.ref_audio)} clips"
                )
            if embeddings and isinstance(embeddings[0], list):
                for item in embeddings:
                    if len(item) != 192:
                        return "Podcast-style Ming speaker embeddings must each have 192 dimensions"

        if request.instructions and len(request.instructions) > self._max_instructions_length:
            return f"Instructions too long (max {self._max_instructions_length} characters)"

        if request.max_new_tokens is not None:
            if request.max_new_tokens < _TTS_MAX_NEW_TOKENS_MIN:
                return f"max_new_tokens must be at least {_TTS_MAX_NEW_TOKENS_MIN}"
            if request.max_new_tokens > _TTS_MAX_NEW_TOKENS_MAX:
                return f"max_new_tokens cannot exceed {_TTS_MAX_NEW_TOKENS_MAX}"

        return None

    async def _resolve_ref_audio(self, ref_audio_str: str) -> tuple[list[float], int]:
        """Resolve ref_audio to (wav_samples, sample_rate).

        Delegates to upstream vLLM's MediaConnector which handles http(s)
        URLs, ``data:`` base64 URIs, and ``file:`` local paths (the latter
        gated by ``--allowed-local-media-path``).
        """
        cache_key = hashlib.sha1(ref_audio_str.encode("utf-8")).hexdigest()
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
            return wav_list, sr

        # In diffusion mode, model_config may not be available
        if self._diffusion_mode:
            connector = MediaConnector()
        else:
            model_config = self.model_config
            connector = MediaConnector(
                allowed_local_media_path=model_config.allowed_local_media_path,
                allowed_media_domains=model_config.allowed_media_domains,
            )
        fetch_start_s = time.perf_counter()
        wav_np, sr = await connector.fetch_audio_async(ref_audio_str)
        fetch_decode_ms = (time.perf_counter() - fetch_start_s) * 1000.0
        wav_np = np.asarray(wav_np, dtype=np.float32)
        if wav_np.ndim > 1:
            wav_np = np.mean(wav_np, axis=-1)
        sr = int(sr)
        artifact_key = self._make_ref_audio_artifact_cache_key(wav_np, sr)
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
        tolist_start_s = time.perf_counter()
        wav_list = wav_np.tolist()
        tolist_ms = (time.perf_counter() - tolist_start_s) * 1000.0
        logger.debug(
            "Resolved ref_audio: fetch_decode_ms=%.3f tolist_ms=%.3f samples=%d sr=%d duration_s=%.3f",
            fetch_decode_ms,
            tolist_ms,
            len(wav_np),
            sr,
            duration,
        )
        self._put_resolved_ref_audio(cache_key, wav_list, sr, artifact_key)
        return wav_list, sr

    @staticmethod
    def _make_ref_audio_artifact_cache_key(wav: np.ndarray, sr: int) -> str:
        wav_f32 = wav.astype(np.float32, copy=False).reshape(-1)
        h = hashlib.sha1()
        h.update(int(sr).to_bytes(4, byteorder="little", signed=False))
        h.update(int(wav_f32.size).to_bytes(8, byteorder="little", signed=False))
        h.update(wav_f32.tobytes(order="C"))
        return h.hexdigest()

    def _get_resolved_ref_audio_artifact_key(self, ref_audio_str: str) -> str | None:
        source_key = hashlib.sha1(ref_audio_str.encode("utf-8")).hexdigest()
        cached = self._ref_audio_resolve_cache.get(source_key)
        if cached is None:
            return None
        self._ref_audio_resolve_cache.move_to_end(source_key)
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
            self._ref_audio_model_artifact_ready.discard(artifact_key)

    def _qwen3_tts_can_use_ref_audio_artifact_only(self, tts_params: dict[str, Any], artifact_key: str | None) -> bool:
        if self._tts_model_type != "qwen3_tts":
            return False
        if not artifact_key or artifact_key not in self._ref_audio_model_artifact_ready:
            return False
        return (tts_params.get("task_type") or ["CustomVoice"])[0] == "Base"

    def _track_ref_audio_artifact_warmup(self, request_id: str, artifact_key: str | None) -> None:
        if artifact_key:
            self._request_ref_audio_artifact_keys[request_id] = artifact_key

    def _mark_ref_audio_artifact_ready_for_request(self, request_id: str) -> None:
        artifact_key = self._request_ref_audio_artifact_keys.pop(request_id, None)
        if artifact_key and any(entry[3] == artifact_key for entry in self._ref_audio_resolve_cache.values()):
            self._ref_audio_model_artifact_ready.add(artifact_key)

    def _discard_ref_audio_artifact_warmup(self, request_id: str) -> None:
        self._request_ref_audio_artifact_keys.pop(request_id, None)

    async def _resolve_ref_audio_many(self, ref_audio_list: list[str]) -> list[tuple[list[float], int]]:
        resolved = []
        for ref_audio in ref_audio_list:
            resolved.append(await self._resolve_ref_audio(ref_audio))
        return resolved

    # ---- Ming TTS helpers ----

    def _is_ming_tts_model(self) -> bool:
        return self._tts_model_type == "ming_tts"

    def _coerce_ming_prompt_waveform(self, wav_samples, sample_rate):
        from torchaudio.functional import resample as resample_audio

        from vllm_omni.model_executor.models.ming_tts.config_ming_tts import SAMPLE_RATE

        waveform = torch.as_tensor(wav_samples, dtype=torch.float32).reshape(1, -1)
        if int(sample_rate) != SAMPLE_RATE:
            waveform = resample_audio(waveform, int(sample_rate), SAMPLE_RATE)
        return waveform

    def _build_ming_prompt_waveform(
        self,
        ref_audio_data: tuple[list[float], int] | list[tuple[list[float], int]] | None,
    ):
        if isinstance(ref_audio_data, list):
            return torch.cat(
                [self._coerce_ming_prompt_waveform(item[0], item[1]) for item in ref_audio_data],
                dim=-1,
            )
        if ref_audio_data is not None:
            return self._coerce_ming_prompt_waveform(ref_audio_data[0], ref_audio_data[1])
        return None

    def _extract_ming_speaker_embeddings_from_ref_audio(
        self,
        ref_audio_data_list: list[tuple[list[float], int]],
    ) -> list[list[float]]:
        from vllm_omni.model_executor.models.ming_tts.speaker_extractor import MingSpeakerEmbeddingExtractor

        extractor = MingSpeakerEmbeddingExtractor(self.engine_client.model_config.model, target_sr=16000)
        embeddings = []
        for wav_samples, sr in ref_audio_data_list:
            waveform = torch.as_tensor(wav_samples, dtype=torch.float32).reshape(1, -1)
            embedding = extractor.extract_from_waveform(waveform, int(sr))
            flat = embedding.detach().reshape(-1).to(torch.float32).cpu()
            if int(flat.numel()) != 192:
                raise ValueError(f"Ming speaker extractor returned {int(flat.numel())} dims; expected 192")
            embeddings.append(flat.tolist())
        return embeddings

    def _parse_ming_instruction_fields(
        self,
        request,
        *,
        include_language=False,
        include_voice=False,
        plain_text_passthrough=False,
    ):
        instruction_text = request.instructions.strip() if isinstance(request.instructions, str) else None
        instruction_dict: dict[str, Any] = {}

        if include_language and request.language not in (None, "", "Auto"):
            instruction_dict["方言"] = request.language

        voice_lower = request.voice.lower() if isinstance(request.voice, str) else None
        if include_voice and request.voice and not (voice_lower and voice_lower in self.uploaded_speakers):
            instruction_dict["IP"] = request.voice

        if instruction_text:
            try:
                parsed = json.loads(instruction_text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                instruction_dict.update(parsed)
            elif instruction_dict or not plain_text_passthrough:
                instruction_dict["风格"] = instruction_text
            else:
                return instruction_text

        return instruction_dict or None

    def _parse_ming_instruction(self, request: OpenAICreateSpeechRequest) -> Any:
        """Build a Ming instruction payload from OpenAI speech fields."""
        return self._parse_ming_instruction_fields(
            request,
            include_language=True,
            include_voice=True,
            plain_text_passthrough=True,
        )

    def _build_ming_dense_prompt(
        self,
        request: OpenAICreateSpeechRequest,
        *,
        ref_audio_data: tuple[list[float], int] | list[tuple[list[float], int]] | None = None,
    ) -> dict[str, Any]:
        """Build a Ming dense prompt directly from the OpenAI speech request."""
        from transformers import AutoTokenizer

        from vllm_omni.model_executor.models.ming_tts.config_ming_tts import KEY_MAX_DECODE_STEPS
        from vllm_omni.model_executor.models.ming_tts.prompt_assembly import build_ming_dense_prompt

        if self._tts_tokenizer is None:
            model_name = self.engine_client.model_config.model
            trust_remote_code = bool(getattr(self.engine_client.model_config, "trust_remote_code", False))
            self._tts_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)

        ref_text = request.ref_text
        prompt_waveform = self._build_ming_prompt_waveform(ref_audio_data) if ref_text is not None else None
        speaker_embedding = request.speaker_embedding
        use_zero_spk_emb = prompt_waveform is None and speaker_embedding is None

        runtime_controls = {}
        if request.max_new_tokens is not None:
            runtime_controls[KEY_MAX_DECODE_STEPS] = request.max_new_tokens

        prompt_dict = build_ming_dense_prompt(
            self._tts_tokenizer,
            # bgm / music-prompt mode not supported online;
            # requires prompt_mode API extension (deferred).
            prompt=_MING_DEFAULT_PROMPT,
            text=request.input,
            runtime_controls=runtime_controls or None,
            instruction=self._parse_ming_instruction(request),
            prompt_text=ref_text,
            prompt_waveform=prompt_waveform,
            speaker_embedding=speaker_embedding,
            use_zero_spk_emb=use_zero_spk_emb,
        )
        prompt = tokens_input(prompt_token_ids=prompt_dict["prompt_token_ids"])
        prompt["prompt"] = prompt_dict["prompt"]
        prompt["text"] = prompt_dict["text"]
        prompt["additional_information"] = prompt_dict["additional_information"]
        return prompt

    async def _generate_audio_chunks(
        self,
        generator,
        request_id: str,
        response_format: str = "pcm",
        raw_request: Request | None = None,
        request_start_s: float | None = None,
        include_sample_rate: bool = False,
        usage_acc: SpeechOutputTokenCounter | None = None,
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

        try:
            async for res in generator:
                # Tally generated codec tokens for usage (reads per-stage metrics
                # off the final output; a cheap early-return on every other res).
                if usage_acc is not None:
                    usage_acc.observe(res)
                audio_output, audio_key = self._extract_audio_output(res)
                if audio_key is None:
                    continue

                sr_raw = audio_output.get("sr")
                if sr_raw is not None:
                    sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
                    sample_rate_val = sr_val.item() if hasattr(sr_val, "item") else int(sr_val)

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

                for chunk_tensor in new_chunks:
                    chunk_np = (
                        chunk_tensor.float().detach().cpu().numpy() if hasattr(chunk_tensor, "float") else chunk_tensor
                    )
                    if chunk_np.ndim > 1:
                        chunk_np = chunk_np.squeeze()
                    # For WAV format, emit header before first audio chunk
                    if response_format == "wav" and first_chunk:
                        # Assert that sample rate has been set from chunk metadata (not just default)
                        # This ensures the WAV header contains the correct sample rate
                        assert sr_raw is not None, (
                            "First audio chunk must include sample rate metadata for WAV streaming"
                        )
                        num_channels = _infer_audio_num_channels(np.asarray(chunk_np))
                        wav_header = _create_wav_header(
                            sample_rate=sample_rate_val,
                            num_channels=num_channels,
                            bits_per_sample=16,
                        )
                        yield wav_header
                        first_chunk = False

                    # Convert audio to PCM bytes
                    audio_obj = CreateAudio(
                        audio_tensor=chunk_np,
                        sample_rate=sample_rate_val,
                        response_format="pcm",
                        speed=1.0,
                        base64_encode=False,
                    )
                    if first_audio_chunk_s is None:
                        first_audio_chunk_s = time.perf_counter()
                    audio_bytes = self.create_audio(audio_obj).audio_data
                    if include_sample_rate:
                        yield audio_bytes, sample_rate_val
                    else:
                        yield audio_bytes
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
        try:
            async for chunk in self._generate_audio_chunks(
                generator,
                request_id,
                response_format,
                raw_request=raw_request,
                request_start_s=request_start_s,
                usage_acc=usage_acc,
            ):
                payload = {
                    "type": "speech.audio.delta",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                    "response_format": response_format,
                }
                data = json.dumps(payload, separators=(",", ":"))
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
            payload = {
                "type": "speech.audio.error",
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "param": None,
                    "code": HTTPStatus.INTERNAL_SERVER_ERROR.value,
                },
            }
            data = json.dumps(payload, separators=(",", ":"))
            yield f"event: speech.audio.error\ndata: {data}\n\n"

    @staticmethod
    def _extract_audio_output(res) -> tuple[dict | None, str | None]:
        """Return (audio_output dict, audio key) or (None, None).

        Returns the raw dict so callers can apply their own extraction strategy:
        streaming needs per-chunk delta slicing; non-streaming needs full concatenation.
        """
        mm = getattr(res, "multimodal_output", None)
        ro = None
        if not mm:
            ro = getattr(res, "request_output", None)
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

    def _build_tts_params(self, request: OpenAICreateSpeechRequest) -> dict[str, Any]:
        """Build TTS parameters from request.

        Processes each parameter if present, skips if not.
        Values are wrapped in lists as required by the model.
        """
        params: dict[str, Any] = {}

        # Text content (always required)
        params["text"] = [request.input]

        # Task type
        if request.task_type is not None:
            params["task_type"] = [request.task_type]
        else:
            params["task_type"] = ["CustomVoice"]

        # Language
        if request.language is not None:
            params["language"] = [request.language]
        else:
            params["language"] = ["Auto"]

        # Speaker (voice)
        if request.voice is not None:
            voice_lower = request.voice.lower()
            params["speaker"] = [request.voice]
            params["voice_created_at"] = [self._voice_created_at(voice_lower)]

            # Uploaded voices use task_type="Base" (CustomVoice requires built-in spk_id).
            # If ref_text was provided at upload time, use in-context cloning; otherwise x_vector only.
            if voice_lower in self.uploaded_speakers and request.ref_audio is None:
                speaker_info = self.uploaded_speakers[voice_lower]

                # Check if this voice was uploaded with a pre-computed embedding.
                # Populate request.speaker_embedding so the existing code path
                # (below) handles voice_clone_prompt and x_vector_only_mode.
                embedding = self._get_uploaded_speaker_embedding(request.voice)
                if embedding is not None:
                    request.speaker_embedding = embedding
                    params["speaker"] = [voice_lower]
                    params["task_type"] = ["Base"]
                    logger.info("Auto-set speaker_embedding for uploaded voice: %s", request.voice)
                else:
                    audio_data = self._get_uploaded_audio_data(request.voice)
                    if not audio_data:
                        raise ValueError(f"Audio file for uploaded voice '{request.voice}' is missing or corrupted")
                    stored_ref_text = speaker_info.get("ref_text")
                    params["speaker"] = [voice_lower]
                    params["ref_audio"] = [audio_data]
                    params["task_type"] = ["Base"]
                    if stored_ref_text:
                        params["ref_text"] = [stored_ref_text]
                        params["x_vector_only_mode"] = [False]
                    else:
                        params["x_vector_only_mode"] = [True]
                    logger.info(
                        "Auto-set ref_audio for uploaded voice: %s (icl=%s)", request.voice, bool(stored_ref_text)
                    )
            elif voice_lower in self.precomputed_speakers and request.ref_audio is None:
                profile = self.precomputed_speakers[voice_lower]
                mode = str(profile.get("mode") or "xvec").lower()
                params["speaker"] = [voice_lower]
                params["task_type"] = ["Base"]
                params["x_vector_only_mode"] = [mode != "icl"]
                ref_text = request.ref_text or profile.get("ref_text")
                if isinstance(ref_text, str) and ref_text.strip():
                    params["ref_text"] = [ref_text]
                ref_code_length = profile.get("ref_code_length")
                if mode == "icl" and ref_code_length:
                    params["ref_code_length"] = [int(ref_code_length)]
                logger.info("Using precomputed Qwen3-TTS custom voice profile: %s (mode=%s)", voice_lower, mode)

        elif params["task_type"][0] == "CustomVoice":
            params["speaker"] = ["Vivian"]  # Default for CustomVoice

        # Instructions for style/emotion control
        if request.instructions is not None:
            params["instruct"] = [request.instructions]
        else:
            params["instruct"] = [""]

        # Voice clone: ref_audio resolved in create_speech(), not here.
        if request.ref_text is not None:
            params["ref_text"] = [request.ref_text]
        if request.speaker_embedding is not None:
            # Store as plain float list (not tensor) so it survives msgspec
            # serialization through the EngineCore IPC boundary.  The talker's
            # _build_prompt_embeds converts it back to a tensor on the GPU.
            params["voice_clone_prompt"] = [
                {
                    "ref_spk_embedding": list(request.speaker_embedding),
                }
            ]
            # speaker_embedding implies x_vector_only_mode
            params["x_vector_only_mode"] = [True]
        elif request.x_vector_only_mode is not None:
            params["x_vector_only_mode"] = [request.x_vector_only_mode]

        # Generation parameters
        if request.max_new_tokens is not None:
            params["max_new_tokens"] = [request.max_new_tokens]
        else:
            params["max_new_tokens"] = [2048]

        if request.initial_codec_chunk_frames is not None:
            params["initial_codec_chunk_frames"] = [request.initial_codec_chunk_frames]

        if request.non_streaming_mode is not None:
            params["non_streaming_mode"] = [request.non_streaming_mode]
        # Preserve the legacy VoiceDesign fallback when the request omits an
        # explicit override. CustomVoice and Base rely on model defaults
        # (True and False respectively).
        elif params["task_type"][0] == "VoiceDesign":
            params["non_streaming_mode"] = [True]

        return params

    # ---- Voxtral TTS helpers ----

    def _build_voxtral_prompt(self, request: OpenAICreateSpeechRequest) -> dict[str, Any]:
        """Build Voxtral TTS engine prompt, supporting both preset voices and inline
        ``ref_audio`` (base64 or data URI)."""
        from mistral_common.protocol.speech.request import SpeechRequest

        text = request.input
        voice = request.voice
        ref_audio = request.ref_audio
        if not voice and not ref_audio:
            raise ValueError("Voxtral requires either a voice name or ref_audio.")
        # mistral_common expects raw base64 (no data: prefix)
        if ref_audio is not None and isinstance(ref_audio, str) and ref_audio.startswith("data:"):
            _, _, ref_audio = ref_audio.partition(",")
        if self._tts_tokenizer is None:
            from vllm.tokenizers import cached_tokenizer_from_config

            mistral_tokenizer = cached_tokenizer_from_config(self.engine_client.model_config)
            self._tts_tokenizer = mistral_tokenizer.instruct
        if voice is not None:
            tokens = self._tts_tokenizer.encode_speech_request(SpeechRequest(input=text, voice=voice)).tokens
            prompt = tokens_input(prompt_token_ids=tokens)
            prompt["additional_information"] = {"voice": [voice]}
            return prompt
        else:
            tokenized = self._tts_tokenizer.encode_speech_request(SpeechRequest(input=text, ref_audio=ref_audio))
            audio = tokenized.audios[0]
            return {
                "prompt_token_ids": tokenized.tokens,
                "multi_modal_data": {"audio": [(audio.audio_array, audio.sampling_rate)]},
            }

    # ---- Step-Audio2 helpers ----

    def _build_step_audio2_prompt(
        self,
        request: OpenAICreateSpeechRequest,
    ) -> dict[str, Any]:
        """Build prompt for Step-Audio2 TTS.

        Constructs the chat prompt with ``<tts_start>`` as the last token
        of the assistant turn (without ``<|im_end|>``), so the thinker
        continues generating audio tokens.

        Prompt format::
            <|im_start|>system\\n{system_prompt}<|im_end|>\\n
            <|im_start|>user\\n{input_text}<|im_end|>\\n
            <|im_start|>assistant\\n<tts_start>
        """
        system_prompt = getattr(request, "instructions", None) or "You are a voice assistant. Read the text aloud."
        text = request.input

        raw_prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{text}<|im_end|>\n"
            f"<|im_start|>assistant\n<tts_start>"
        )
        return {"prompt": raw_prompt}

    # ---- Fish Speech helpers ----

    def _build_fish_speech_prompt(
        self,
        request: OpenAICreateSpeechRequest,
        ref_audio_data: tuple[list[float], int] | None = None,
        *,
        has_inline_ref_audio: bool = False,
    ) -> dict[str, Any]:
        """Build prompt for Fish Speech S2 Pro.

        Without voice cloning:
          <|im_start|>system\\nconvert the provided text to speech<|im_end|>
          <|im_start|>user\\n{text}<|im_end|>\\n<|im_start|>assistant\\n<|voice|>

        With voice cloning (ref_audio + ref_text):
          <|im_start|>system\\nconvert the provided text to speech reference to the following...
          <|im_end|>\\n<|im_start|>user\\n{text}<|im_end|>\\n<|im_start|>assistant\\n<|voice|>
        """
        from transformers import AutoTokenizer

        if self._fish_speech_tokenizer is None:
            model_name = self.engine_client.model_config.model
            self._fish_speech_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        tokenizer = self._fish_speech_tokenizer

        if ref_audio_data is None or not request.ref_text:
            prompt_ids, normalized_text = build_fish_text_only_prompt_ids(tokenizer, request.input)

            # Keep the prompt-dict metadata shape aligned with the existing text-only
            # TTS entrypoints: scalar values are wrapped in single-item lists before
            # EngineCore serialization. Structured clone below is different because
            # model-side preprocess consumes concrete per-request scalar fields.
            additional_information: dict[str, Any] = {
                "text": [normalized_text],
            }
            if request.max_new_tokens is not None:
                additional_information["max_new_tokens"] = [request.max_new_tokens]
            prompt = tokens_input(prompt_token_ids=prompt_ids)
            prompt["additional_information"] = additional_information
            return prompt

        wav_samples, sr = ref_audio_data
        normalized_text, normalized_ref_text = normalize_fish_voice_clone_texts(request.input, request.ref_text)
        ph_len = self._estimate_fish_prompt_len(normalized_text, normalized_ref_text, ref_audio_data)

        # Structured clone: scalars (not list-wrapped) because model-side
        # preprocess() consumes per-request fields directly.
        additional_information: dict[str, Any] = {
            "text": normalized_text,
            "ref_text": normalized_ref_text,
            "ref_audio_wav": torch.from_numpy(np.asarray(wav_samples, dtype=np.float32)),
            "ref_audio_sr": int(sr),
            "fish_structured_voice_clone": True,
        }
        # Pass voice identity for model-side DAC code caching.
        if request.voice is not None:
            voice_lower = request.voice.lower()
            if voice_lower in self.uploaded_speakers and not has_inline_ref_audio:
                additional_information["voice_name"] = voice_lower
                additional_information["voice_created_at"] = self._voice_created_at(voice_lower)
        if request.max_new_tokens is not None:
            additional_information["max_new_tokens"] = request.max_new_tokens
        prompt = tokens_input(prompt_token_ids=[1] * ph_len)
        prompt["additional_information"] = additional_information
        return prompt

    # ---- CosyVoice3 helpers ----

    async def _build_cosyvoice3_prompt(
        self,
        request: OpenAICreateSpeechRequest,
        *,
        has_inline_ref_audio: bool = False,
    ) -> dict[str, Any]:
        """Build prompt for CosyVoice3.

        CosyVoice3 uses multimodal input with reference audio for voice cloning.
        The prompt format matches the offline example: text prompt + audio data
        + mm_processor_kwargs with prompt_text.
        """
        # Resolve reference audio
        wav_samples, sr = await self._resolve_ref_audio(request.ref_audio)
        audio_data = (np.asarray(wav_samples, dtype=np.float32), sr)

        # Wrap the reference transcript in the CosyVoice3 instruction template
        # so the talker emits target-only speech (see _COSYVOICE3_PROMPT_PREFIX).
        # Skip if the caller already supplied a formatted prompt_text.
        ref_text = request.ref_text or ""
        if _COSYVOICE3_PROMPT_DELIMITER not in ref_text:
            ref_text = f"{_COSYVOICE3_PROMPT_PREFIX}{ref_text}"
        mm_kwargs: dict[str, Any] = {
            "prompt_text": ref_text,
            "sample_rate": sr,
        }
        # Pass voice metadata for caching in the processor
        if request.voice:
            voice_lower = request.voice.lower()
            if voice_lower in self.uploaded_speakers and not has_inline_ref_audio:
                mm_kwargs["voice_name"] = voice_lower
                mm_kwargs["voice_created_at"] = self._voice_created_at(voice_lower)

        return {
            "prompt": request.input,
            "multi_modal_data": {
                "audio": audio_data,
            },
            "mm_processor_kwargs": mm_kwargs,
        }

    # ---- Covo-Audio helpers ----

    def _build_covo_audio_prompt(
        self,
        request: OpenAICreateSpeechRequest,
    ) -> dict[str, Any]:
        """Build a chat-style prompt for Covo-Audio-Chat.

        Covo-Audio requires a specific system prompt that instructs the model
        to interleave text and audio tokens in its output.  We render the
        messages through the chat template and pass prompt_token_ids so that
        the engine does not need to re-tokenize.
        """
        from transformers import AutoTokenizer

        from vllm_omni.model_executor.models.covo_audio.prompt_utils import (
            build_covo_audio_prompt_token_ids,
        )

        if self._covo_audio_tokenizer is None:
            model_name = self.engine_client.model_config.model
            try:
                self._covo_audio_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to load Covo-Audio tokenizer from '{model_name}': {exc}") from exc

        prompt_ids = build_covo_audio_prompt_token_ids(
            self._covo_audio_tokenizer,
            request.input,
        )
        return {"prompt_token_ids": prompt_ids}

    def _apply_cosyvoice3_dynamic_tokens(
        self,
        sampling_params_list: list,
        request: OpenAICreateSpeechRequest,
    ) -> list:
        """Set min/max tokens from tokenized text length (ratios target tokens, not chars)."""
        import copy

        from vllm_omni.model_executor.models.cosyvoice3.tokenizer import get_qwen_tokenizer
        from vllm_omni.model_executor.models.cosyvoice3.utils import extract_text_token

        sampling_params_list = copy.deepcopy(sampling_params_list)
        hf_cfg = self.model_config.hf_config
        # Build the Qwen tokenizer once per process (resolving the model dir via
        # snapshot_download at most once) and reuse it across requests.
        tokenizer = self._cosyvoice3_tokenizer
        if tokenizer is None:
            model_path = self.engine_client.model_config.model
            if not os.path.isdir(model_path):
                from huggingface_hub import snapshot_download

                model_path = snapshot_download(model_path)
            tokenizer = get_qwen_tokenizer(
                token_path=os.path.join(model_path, hf_cfg.qwen_pretrain_path),
                skip_special_tokens=hf_cfg.skip_special_tokens,
                version=hf_cfg.version,
            )
            self._cosyvoice3_tokenizer = tokenizer
        _, text_token_len = extract_text_token(
            request.input,
            tokenizer,
            hf_cfg.allowed_special,
        )
        min_ratio = getattr(hf_cfg, "min_token_text_ratio", 2)
        max_ratio = getattr(hf_cfg, "max_token_text_ratio", 20)
        sampling_params_list[0].min_tokens = max(1, int(text_token_len * min_ratio))
        sampling_params_list[0].max_tokens = min(2048, int(text_token_len * max_ratio))
        logger.info(
            "CosyVoice3 dynamic tokens: text_tokens=%d, min_tokens=%d, max_tokens=%d",
            text_token_len,
            sampling_params_list[0].min_tokens,
            sampling_params_list[0].max_tokens,
        )
        return sampling_params_list

    # ---- GLM-TTS helpers ----

    async def _build_glm_tts_prompt(
        self,
        request: OpenAICreateSpeechRequest,
        *,
        has_inline_ref_audio: bool = False,
    ) -> dict[str, Any]:
        """Build prompt for GLM-TTS.

        Uses the multimodal processor path (same as CosyVoice3):
        - prompt: synthesis text
        - multi_modal_data["audio"]: (wav_samples, sr) reference audio
        - mm_processor_kwargs["prompt_text"]: reference text transcript

        AR preprocess() builds [PromptText | Text | BOA | PromptSpeechTokens + ATS].
        DiT receives prompt_token, prompt_feat, embedding for conditioning.
        """
        # Voice cloning requires ref_audio + ref_text
        if request.ref_audio is not None and request.ref_text:
            wav_samples, sr = await self._resolve_ref_audio(request.ref_audio)
            audio_data = (np.asarray(wav_samples, dtype=np.float32), int(sr))

            mm_kwargs: dict[str, Any] = {
                "prompt_text": request.ref_text,
            }
            if request.voice:
                voice_lower = request.voice.lower()
                if voice_lower in self.uploaded_speakers and not has_inline_ref_audio:
                    mm_kwargs["voice_name"] = voice_lower
                    mm_kwargs["voice_created_at"] = self._voice_created_at(voice_lower)

            return {
                "prompt": request.input,
                "multi_modal_data": {
                    "audio": audio_data,
                },
                "mm_processor_kwargs": mm_kwargs,
                "additional_information": self._build_glm_tts_prefill_metadata(
                    request.input,
                    request.ref_text,
                ),
            }

        raise ValueError("GLM-TTS requires ref_audio and ref_text for voice cloning.")

    def _glm_tts_text_tokenizer_and_frontend(self):
        from vllm_omni.model_executor.models.glm_tts.glm_tts import (
            load_glm_tts_tokenizer,
            resolve_glm_tts_tokenizer_path,
        )
        from vllm_omni.model_executor.models.glm_tts.text_frontend import GLMTTSTextFrontend

        cached = self._glm_tts_text_tokenizer
        if cached is None:
            model_name_or_path = self.engine_client.model_config.model
            tokenizer_path = getattr(self.engine_client.model_config, "tokenizer", None)
            if tokenizer_path is None:
                tokenizer_path = resolve_glm_tts_tokenizer_path(model_name_or_path)
            cached = load_glm_tts_tokenizer(
                tokenizer_path,
                model_name_or_path=model_name_or_path,
                trust_remote_code=bool(getattr(self.engine_client.model_config, "trust_remote_code", False)),
            )
            self._glm_tts_text_tokenizer = cached

        frontend = self._glm_tts_text_frontend
        if frontend is None:
            frontend = GLMTTSTextFrontend()
            self._glm_tts_text_frontend = frontend
        return cached, frontend

    def _estimate_glm_tts_text_token_len(self, text: str | None, *, add_trailing_space: bool = False) -> int:
        """Estimate GLM-TTS normalized text length with the model tokenizer."""
        cached, frontend = self._glm_tts_text_tokenizer_and_frontend()
        text = text or ""
        normalized = frontend.text_normalize(text) or text
        normalized = normalized.strip()
        if add_trailing_space and normalized:
            normalized = f"{normalized} "
        return max(1, len(cached.encode(normalized)))

    def _build_glm_tts_prefill_metadata(self, text: str, prompt_text: str | None) -> dict[str, Any]:
        """Build GLM-TTS processor length metadata for additional_information.

        The model preprocess hook runs before postprocess can mirror MM kwargs
        into additional_information, so these scalar fields must originate from
        the request payload rather than runner-level mm_features.
        """
        text_len = self._estimate_glm_tts_text_token_len(text)
        prompt_text_len = (
            self._estimate_glm_tts_text_token_len(prompt_text, add_trailing_space=True) if prompt_text else 0
        )
        return {
            "glm_tts_text_token_len": [text_len],
            "glm_tts_prompt_text_token_len": [prompt_text_len],
            "input_len": [prompt_text_len + text_len + 1],
        }

    # ---- Ming-flash-omni standalone-talker (TTS) helpers ----

    def _build_ming_flash_omni_prompt(self, request: OpenAICreateSpeechRequest) -> dict[str, Any]:
        # request.instructions accepts two forms:
        # 1. Plain text: mapped to the caption's 风格 (style) field
        # 2. JSON object: parsed and splatted into the caption. Unlocks
        #       Unknown keys are dropped by `ming_create_instruction`.
        caption_fields = self._parse_ming_instruction_fields(request) or {}

        has_spk_emb = request.speaker_embedding is not None

        # TTS path applies ming task type `instruct`.
        # voice_name enables talker-side voice preset resolution (e.g. "DB30").
        additional_information: dict[str, Any] = {
            "ming_task": "instruct",
            "prompt": MING_DEFAULT_PROMPT,
            "text": request.input,
            "instruction": ming_create_instruction(caption_fields),
            "voice_name": request.voice or None,
            "use_zero_spk_emb": not has_spk_emb,
            "max_decode_steps": request.max_new_tokens or _TTS_MAX_NEW_TOKENS_MAX,
            "cfg": 2.0,
            "sigma": 0.25,
            "temperature": 0.0,
        }
        if has_spk_emb:
            # Passed as plain float list
            additional_information["spk_emb"] = list(request.speaker_embedding)
        prompt = tokens_input(prompt_token_ids=[0])
        prompt["additional_information"] = additional_information
        return prompt

    # ---- Common speech generation helpers ----

    async def _build_qwen3_tts_request(
        self,
        request: OpenAICreateSpeechRequest,
    ) -> tuple[dict[str, Any], dict[str, Any], str | None]:
        """Build prompt + tts_params for Qwen3-TTS.

        Called from ``Qwen3TTSAdapter.build``. Returns
        ``(prompt, tts_params, warmup_artifact_key)`` where the warmup key is the
        Qwen3-TTS ref-audio artifact tracked after ``generate()``.
        """
        qwen3_ref_audio_warmup_artifact_key: str | None = None
        tts_params = self._build_tts_params(request)
        # Resolve ref_audio (explicit or auto-set for uploaded voices)
        # to [[wav_list, sr]] so the model doesn't re-decode base64.
        ref_audio_source = request.ref_audio
        if ref_audio_source is None and isinstance(tts_params.get("ref_audio"), list):
            # Uploaded voice: ref_audio was auto-set as [base64_data_url]
            ref_audio_source = tts_params["ref_audio"][0]
        if ref_audio_source is not None and isinstance(ref_audio_source, str):
            wav_list, sr = await self._resolve_ref_audio(ref_audio_source)
            artifact_key = self._get_resolved_ref_audio_artifact_key(ref_audio_source)
            if artifact_key:
                tts_params[_QWEN3_TTS_REF_AUDIO_CACHE_KEY] = [artifact_key]
            ref_code_length = self._estimate_ref_code_len([wav_list, sr])
            if ref_code_length is not None:
                tts_params["ref_code_length"] = [int(ref_code_length)]
            if self._qwen3_tts_can_use_ref_audio_artifact_only(tts_params, artifact_key):
                logger.debug("Using Qwen3-TTS ref_audio artifact-only path: %s", artifact_key)
            else:
                tts_params["ref_audio"] = [[wav_list, sr]]
                qwen3_ref_audio_warmup_artifact_key = artifact_key

        ph_len = await self._estimate_prompt_len_async(tts_params)
        prompt = tokens_input(prompt_token_ids=[1] * ph_len)
        prompt["additional_information"] = tts_params
        prompt["cache_salt"] = _conditioning_cache_salt(request, tts_params)
        return prompt, tts_params, qwen3_ref_audio_warmup_artifact_key

    async def _prepare_speech_generation(
        self,
        request: OpenAICreateSpeechRequest,
        request_id: str | None = None,
    ) -> tuple[str, Any, dict[str, Any]]:
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        request_id = request_id or f"speech-{random_uuid()}"
        qwen3_ref_audio_warmup_artifact_key: str | None = None

        # If this is a streaming request, we need to coerce
        # cumulative outputs to delta outputs; this ensures
        # we don't emit redundant MM data & drain after emitting.
        # list() makes a copy to avoid mutating the params.
        sampling_params_list = list(self.engine_client.default_sampling_params_list)
        is_streaming_request = request.is_streaming()
        sampling_params_list = coerce_param_message_types(sampling_params_list, is_streaming_request)

        # Build prompt + tts_params via the per-model adapter (RFC #4327). Every
        # dedicated TTS model resolves to an adapter that owns its validation,
        # uploaded-speaker handling, and prompt/param building. Sampling
        # overrides and the model-type label remain in the orchestrator tail
        # below (keyed on ``_tts_model_type``) during this incremental migration.
        # Non-TTS deployments (no adapter) fall through to the rejection below.
        # Capture inline-ref-audio status BEFORE validate(): several adapters
        # apply uploaded speakers inside validate(), which sets request.ref_audio
        # in place. The builders need to know whether the caller supplied audio
        # inline vs. via an uploaded voice.
        has_inline_ref_audio = request.ref_audio is not None
        if self._tts_model_type == "ming_flash_omni_tts":
            # ming_flash_omni is intentionally NOT migrated onto the adapter
            # framework in this PR (it has no registered adapter); keep it on the
            # legacy inline dispatch so serving still works.
            validation_error = self._validate_ming_flash_omni_tts_request(request)
            if validation_error:
                raise ValueError(validation_error)
            prompt = self._build_ming_flash_omni_prompt(request)
            tts_params = {}
            qwen3_ref_audio_warmup_artifact_key = None
        elif (adapter := self._get_tts_adapter()) is not None:
            validation_error = adapter.validate(request)
            if validation_error:
                raise ValueError(validation_error)
            prepared = await adapter.build(request, sampling_params_list, has_inline_ref_audio)
            prompt = prepared.prompt
            tts_params = prepared.tts_params
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

        if self._tts_model_type == "step_audio2":
            model_type = "step_audio2"
        elif self._is_fish_speech:
            model_type = "fish_speech"
        elif self._tts_model_type == "covo_audio":
            model_type = "covo_audio"
        elif self._tts_model_type == "voxtral_tts":
            model_type = "voxtral_tts"
        elif self._tts_model_type == "cosyvoice3":
            model_type = "cosyvoice3"
        elif self._tts_model_type == "voxcpm2":
            model_type = "voxcpm2"
        elif self._tts_model_type == "ming_flash_omni_tts":
            model_type = "ming_flash_omni_tts"
        elif self._tts_model_type == "ming_tts":
            model_type = "ming_tts"
        elif self._tts_model_type == "moss_tts_nano":
            model_type = "moss_tts_nano"
        elif self._tts_model_type == "moss_tts":
            model_type = "moss_tts"
        elif self._tts_model_type == "higgs_audio_v2":
            model_type = "higgs_audio_v2"
        elif self._tts_model_type == "glm_tts":
            model_type = "glm_tts"
        elif self._tts_model_type == "indextts2":
            model_type = "indextts2"
        elif self._is_tts:
            model_type = tts_params.get("task_type", ["unknown"])[0]
        else:
            model_type = "generic"
        logger.info(
            "TTS speech request %s: text=%r, model=%s",
            request_id,
            request.input[:50] + "..." if len(request.input) > 50 else request.input,
            model_type,
        )

        # CosyVoice3: set dynamic min/max tokens based on text length.
        # The official model requires min_token_text_ratio to prevent early
        # EOS and max_token_text_ratio to cap generation length.
        if self._tts_model_type == "cosyvoice3" and sampling_params_list:
            sampling_params_list = self._apply_cosyvoice3_dynamic_tokens(sampling_params_list, request)

        # GLM-TTS: set dynamic min/max tokens based on text length.
        if self._tts_model_type == "glm_tts" and sampling_params_list:
            import copy

            sampling_params_list = copy.deepcopy(sampling_params_list)
            glm_metadata = prompt.get("additional_information") if isinstance(prompt, dict) else None
            text_len_value = None
            if isinstance(glm_metadata, dict):
                text_len_value = glm_metadata.get("glm_tts_text_token_len")
                if isinstance(text_len_value, list) and text_len_value:
                    text_len_value = text_len_value[0]
            text_token_len = (
                int(text_len_value)
                if text_len_value is not None
                else self._estimate_glm_tts_text_token_len(request.input)
            )
            hf_cfg = self.model_config.hf_config
            min_ratio = getattr(hf_cfg, "min_token_text_ratio", 2)
            max_ratio = getattr(hf_cfg, "max_token_text_ratio", 20)
            stage_min_tokens = getattr(sampling_params_list[0], "min_tokens", None)
            stage_max_tokens = getattr(sampling_params_list[0], "max_tokens", None)
            cap_candidates = [int(cap) for cap in (stage_max_tokens, request.max_new_tokens) if cap is not None]
            hard_cap = min(cap_candidates) if cap_candidates else None

            min_tokens = max(1, int(text_token_len * min_ratio))
            if stage_min_tokens is not None:
                min_tokens = max(min_tokens, int(stage_min_tokens))
            if hard_cap is not None:
                min_tokens = min(min_tokens, hard_cap)

            max_tokens = max(min_tokens, int(text_token_len * max_ratio))
            if hard_cap is not None:
                max_tokens = min(max_tokens, hard_cap)
            sampling_params_list[0].min_tokens = min_tokens
            sampling_params_list[0].max_tokens = max_tokens
            seed = getattr(request, "seed", None)
            if seed is not None:
                sampling_params_list[0].seed = seed
            logger.info(
                "GLM-TTS dynamic tokens: text_tokens=%d, min_ratio=%s, max_ratio=%s, "
                "stage_min=%s, stage_max=%s, request_max=%s, min_tokens=%d, max_tokens=%d",
                text_token_len,
                min_ratio,
                max_ratio,
                stage_min_tokens,
                stage_max_tokens,
                request.max_new_tokens,
                min_tokens,
                max_tokens,
            )

        # Apply model-specific extra parameters
        if request.extra_params is not None and sampling_params_list:
            if not isinstance(request.extra_params, dict):
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST.value,
                    detail="extra_params must be a JSON object/dict.",
                )
            import copy

            sampling_params_list = copy.deepcopy(sampling_params_list)
            if sampling_params_list[0].extra_args is None:
                sampling_params_list[0].extra_args = {}
            sampling_params_list[0].extra_args.update(request.extra_params)
            logger.info("Applied extra_params: %s", request.extra_params)

        # Some TTS model defaults come from deploy YAML. Their AR
        # generation length is controlled by SamplingParams.max_tokens, so only
        # override it when the caller explicitly requests max_new_tokens.
        if (
            self._tts_model_type in _SAMPLING_MAX_TOKENS_TTS_MODEL_TYPES
            and request.max_new_tokens is not None
            and sampling_params_list
        ):
            import copy

            sampling_params_list = copy.deepcopy(sampling_params_list)
            sampling_params_list[0].max_tokens = request.max_new_tokens
            if self._tts_model_type == "cosyvoice3":
                sampling_params_list[0].min_tokens = min(
                    getattr(sampling_params_list[0], "min_tokens", 0),
                    request.max_new_tokens,
                )
        elif self._tts_model_type == "ming_tts" and sampling_params_list:
            import copy

            from vllm_omni.model_executor.models.ming_tts.config_ming_tts import (
                MOE_TEXT_EOS_TOKEN_ID,
                TEXT_EOS_TOKEN_ID,
            )

            hf_config = self.engine_client.model_config.hf_config
            is_moe = getattr(hf_config, "model_type", "") == "bailingmm"
            stop_token_id = MOE_TEXT_EOS_TOKEN_ID if is_moe else TEXT_EOS_TOKEN_ID

            sampling_params_list = copy.deepcopy(sampling_params_list)
            sampling_params_list[0].stop_token_ids = [int(stop_token_id)]
            if request.max_new_tokens is not None:
                # Ming emits TEXT_EOS after the latent decode budget is exhausted, so
                # Stage-0 needs one extra token beyond ming_max_decode_steps.
                sampling_params_list[0].max_tokens = int(request.max_new_tokens) + 1

        if request.seed is not None and sampling_params_list:
            if sampling_params_list is self.engine_client.default_sampling_params_list:
                import copy

                sampling_params_list = copy.deepcopy(sampling_params_list)
            sampling_params_list[0].seed = request.seed
            if self._tts_model_type == "qwen3_tts":
                if sampling_params_list[0].extra_args is None:
                    sampling_params_list[0].extra_args = {}
                sampling_params_list[0].extra_args["qwen3_tts_request_seed"] = request.seed

        generator = self.engine_client.generate(
            prompt=prompt,
            request_id=request_id,
            sampling_params_list=sampling_params_list,
            output_modalities=["audio"],
        )
        self._track_ref_audio_artifact_warmup(request_id, qwen3_ref_audio_warmup_artifact_key)
        return request_id, generator, tts_params

    async def _generate_pcm_chunks(self, generator, request_id: str, *, include_sample_rate: bool = False):
        """Yield raw PCM byte chunks from the engine generator.

        Delegates to ``_generate_audio_chunks`` with ``response_format="pcm"``.
        Used by the WebSocket streaming handler and ``_iter_pcm_audio_bytes``.
        """
        async for chunk in self._generate_audio_chunks(
            generator,
            request_id,
            response_format="pcm",
            include_sample_rate=include_sample_rate,
        ):
            yield chunk

    async def _iter_pcm_audio_bytes(self, request: OpenAICreateSpeechRequest):
        """Yield raw PCM bytes for a speech request as soon as chunks are decoded."""
        request_id, generator, _ = await self._prepare_speech_generation(request)
        try:
            async for chunk in self._generate_pcm_chunks(generator, request_id):
                yield chunk
        finally:
            self._discard_ref_audio_artifact_warmup(request_id)

    async def _generate_audio_bytes(
        self,
        request: OpenAICreateSpeechRequest,
        base64_encode: bool = False,
        request_id: str | None = None,
        usage_out: list[SpeechTokenUsage] | None = None,
    ) -> tuple[bytes | str, str]:
        # ``usage_out`` is an opt-in output channel: when a list is passed, the
        # computed SpeechTokenUsage is appended to it. The return stays a
        # 2-tuple so existing callers (and their test mocks) are unaffected;
        # only the batch path, which surfaces per-item usage, opts in.
        request_id, generator, bytes_tts_params = await self._prepare_speech_generation(request, request_id=request_id)
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
            async for res in generator:
                final_output = res
                usage_acc.observe(res)
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

            audio_output, audio_key = self._extract_audio_output(final_output)
            if audio_key is None:
                raise ValueError("TTS model did not produce audio output.")

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

            audio_obj = CreateAudio(
                audio_tensor=audio_tensor,
                sample_rate=sample_rate,
                response_format=request.response_format or "wav",
                speed=request.speed or 1.0,
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

            if request.voice:
                voice_lower = request.voice.lower()
                if voice_lower not in self.uploaded_speakers and voice_lower not in self.supported_speakers:
                    all_voices = sorted(self.uploaded_speakers.keys() | self.supported_speakers)
                    raise ValueError(f"Invalid voice '{request.voice}'. Supported: {', '.join(all_voices) or 'none'}")

            has_inline_ref_audio = request.ref_audio is not None
            err = self._apply_uploaded_speaker(request)
            if err:
                raise ValueError(err)

            request_id = f"speech-{random_uuid()}"
            prompt: dict[str, Any] = {"input": request.input}
            if request.ref_audio:
                wav, sr = await self._resolve_ref_audio(request.ref_audio)
                prompt["ref_audio"] = (np.asarray(wav, dtype=np.float32), sr)
            if request.ref_text:
                prompt["ref_text"] = request.ref_text
            if request.voice:
                voice_lower = request.voice.lower()
                if voice_lower in self.uploaded_speakers and not has_inline_ref_audio:
                    prompt["voice_name"] = voice_lower
                    prompt["voice_created_at"] = self._voice_created_at(voice_lower)
            if request.language:
                prompt["lang"] = request.language
            if request.instructions:
                prompt["instruct"] = request.instructions

            logger.info(
                "Diffusion TTS speech request %s: text=%r, voice_clone=%s",
                request_id,
                request.input[:50] + "..." if len(request.input) > 50 else request.input,
                "ref_audio" in prompt,
            )
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
                response_format=request.response_format or "wav",
                speed=request.speed or 1.0,
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
        if request.speed is not None and request.speed != 1.0:
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

        Streaming is supported via ``stream_format='audio'`` or the legacy
        ``stream=True`` switch, with ``response_format='pcm'`` or ``'wav'``.
        ``stream_format='sse'`` returns OpenAI ``speech.audio.*`` SSE events instead.
        Each Code2Wav chunk is yielded as raw audio bytes as soon as it is decoded.
        For WAV format, a header with placeholder size values is emitted first.
        """
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
            if request.is_raw_audio_stream():
                if request.word_timestamps:
                    return self.create_error_response(
                        "word_timestamps=true is currently supported by the WebSocket "
                        "/v1/audio/speech/stream path. Use session.config with "
                        "stream_audio=true and response_format='pcm'."
                    )

                response_format, error = self._validate_speech_streaming_request(
                    request,
                    mode_label="Streaming",
                )
                if error is not None:
                    return error

                media_type = "audio/wav" if response_format == "wav" else "audio/pcm"
                _, generator, _ = await self._prepare_speech_generation(request, request_id=request_id)
                return StreamingResponse(
                    self._generate_audio_chunks(
                        generator,
                        request_id,
                        response_format,
                        raw_request=raw_request,
                        request_start_s=request_start_s,
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

            audio_bytes, media_type = await self._generate_audio_bytes(request, request_id=request_id)
            total_ms = (time.perf_counter() - request_start_s) * 1000.0
            logger.info(
                "[SpeechE2E] request_id=%s stream=false status=ok total_ms=%.2f response_bytes=%d",
                request_id,
                total_ms,
                len(audio_bytes) if isinstance(audio_bytes, (bytes, bytearray)) else len(str(audio_bytes)),
            )
            return Response(content=audio_bytes, media_type=media_type)

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
            # Batch validation still goes through _validate_tts_request directly
            # (not the adapter). The single-request path validates via the
            # adapter; both ultimately call the same per-model validators, so the
            # surfaces stay in sync. Routing batch through the adapter (and its
            # uploaded-speaker handling) is a follow-up (RFC #4327).
            validation_error = self._validate_tts_request(req)
            if validation_error is not None:
                return SpeechBatchItemResult(index=idx, status="error", error=validation_error)
            usage_box: list[SpeechTokenUsage] = []
            try:
                audio_data, media_type = await self._generate_audio_bytes(req, base64_encode=True, usage_out=usage_box)
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
