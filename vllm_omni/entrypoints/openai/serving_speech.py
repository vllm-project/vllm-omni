import asyncio
import base64
import io
import json
import math
import os
import re
import struct
import time
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
)
from vllm_omni.model_executor.models.fish_speech.prompt_utils import (
    build_fish_text_only_prompt_ids,
    estimate_fish_voice_clone_prompt_len_from_normalized,
    normalize_fish_voice_clone_texts,
)
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)

# TTS Configuration
_VOXTRAL_TTS_MODEL_STAGES = {"audio_generation"}
_QWEN3_TTS_MODEL_STAGES = {"qwen3_tts"}
_FISH_TTS_MODEL_STAGES = {"fish_speech_slow_ar"}
_COSYVOICE3_TTS_MODEL_STAGES = {"cosyvoice3_talker"}
_OMNIVOICE_TTS_MODEL_STAGES = {"omnivoice_generator"}
_VOXCPM_TTS_MODEL_STAGES = {"latent_generator", "vae"}
_VOXCPM2_TTS_MODEL_STAGES = {"latent_generator"}
_TTS_MODEL_STAGES: set[str] = (
    _VOXTRAL_TTS_MODEL_STAGES
    | _QWEN3_TTS_MODEL_STAGES
    | _FISH_TTS_MODEL_STAGES
    | _COSYVOICE3_TTS_MODEL_STAGES
    | _OMNIVOICE_TTS_MODEL_STAGES
    | _VOXCPM_TTS_MODEL_STAGES
    | _VOXCPM2_TTS_MODEL_STAGES
)
_TTS_LANGUAGES: set[str] = {
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
_REF_AUDIO_MIN_DURATION = 1.0  # seconds
_REF_AUDIO_MAX_DURATION = 30.0  # seconds
_TTS_MAX_INSTRUCTIONS_LENGTH = 500
_TTS_MAX_NEW_TOKENS_MIN = 1
_TTS_MAX_NEW_TOKENS_MAX = 4096


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
    _tts_executor: ThreadPoolExecutor | None = None

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
        return instance

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize uploaded speakers storage (ephemeral — cleared on restart)
        speech_voice_samples_dir = os.environ.get("SPEECH_VOICE_SAMPLES", "/tmp/voice_samples")
        self.uploaded_speakers_dir = Path(speech_voice_samples_dir)
        self.uploaded_speakers_dir.mkdir(parents=True, exist_ok=True)

        # Find and cache the TTS stage (if any) during initialization
        self._tts_stage = self._find_tts_stage()
        self._is_tts = self._tts_stage is not None
        self._is_fish_speech = (
            self._tts_stage is not None
            and getattr(getattr(self._tts_stage, "engine_args", None), "model_stage", None) == "fish_speech_slow_ar"
        )
        self._fish_speech_tokenizer = None

        self._is_cosyvoice3 = (
            self._tts_stage is not None
            and getattr(getattr(self._tts_stage, "engine_args", None), "model_stage", None)
            in _COSYVOICE3_TTS_MODEL_STAGES
        )
        self._cosyvoice3_tokenizer = None

        # Determine TTS model type or None
        self._tts_model_type = self._detect_tts_model_type()

        # Cache TTS configuration values (computed once, reused per request)
        self._max_instructions_length = self._compute_max_instructions_length()

        # Load supported speakers (built-in only; uploaded voices start empty)
        self.supported_speakers = self._load_supported_speakers()
        self.uploaded_speakers: dict[str, dict] = {}
        logger.warning(
            "Uploaded voices are ephemeral and will be lost on server restart. "
            "Re-upload voices after each restart if needed."
        )
        self._tts_tokenizer = None
        self._voxcpm2_tokenizer = None
        self._voxcpm2_split_map: dict[int, list[int]] = {}

        logger.info(f"Loaded {len(self.supported_speakers)} supported speakers: {sorted(self.supported_speakers)}")

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

    def _load_codec_frame_rate(self) -> float | None:
        """Load codec frame rate from speech tokenizer config for prompt length estimation."""
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
                        f"Loaded codec frame rate: {rate:.1f} Hz "
                        f"(output_sample_rate={output_sr}, encode_downsample_rate={downsample})"
                    )
                    return rate
        except Exception as e:
            logger.warning(f"Failed to load codec frame rate from speech tokenizer config: {e}")

        # Fallback: try codec_frame_rate_hz from hf_config
        try:
            hf_config = self.engine_client.model_config.hf_config
            rate = getattr(hf_config, "codec_frame_rate_hz", None)
            if rate is not None:
                logger.info(f"Using codec frame rate from hf_config: {rate} Hz")
                return float(rate)
        except Exception:
            pass
        return None

    def shutdown(self) -> None:
        """Shut down the TTS thread pool executor."""
        if self._tts_executor is not None:
            self._tts_executor.shutdown(wait=False, cancel_futures=True)
            self._tts_executor = None

    def _find_tts_stage(self):
        """Find and return the TTS stage config, or None if not found."""
        for stage in self.engine_client.stage_configs:
            if stage.engine_args.model_stage in _TTS_MODEL_STAGES:
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
        if model_arch == "VoxCPMForConditionalGeneration":
            return "voxcpm"
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
        if model_stage in (_VOXCPM_TTS_MODEL_STAGES | _VOXCPM2_TTS_MODEL_STAGES):
            has_vae_stage = any(
                getattr(getattr(stage, "engine_args", None), "model_stage", None) == "vae"
                for stage in self.engine_client.stage_configs
            )
            return "voxcpm" if has_vae_stage or model_stage == "vae" else "voxcpm2"
        return None

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
        try:
            if self._tts_model_type == "voxcpm":
                return set()
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
            logger.warning(f"Could not load speakers from model config: {e}")

        return set()

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
            if self._tts_model_type == "voxcpm":
                return 1
            from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
                Qwen3TTSTalkerForConditionalGeneration,
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
            return Qwen3TTSTalkerForConditionalGeneration.estimate_prompt_len_from_additional_information(
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

    async def _build_voxcpm2_prompt(self, request: OpenAICreateSpeechRequest) -> dict[str, Any]:
        """Build prefill prompt for VoxCPM2 TTS (`prompt_token_ids` padded to full prefill length)."""
        from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import build_voxcpm2_prompt

        self._voxcpm2_encode("")  # lazy-init tokenizer + split_map
        ref_audio = None
        ref_sr = None
        if request.ref_audio is not None:
            ref_audio, ref_sr = await self._resolve_ref_audio(request.ref_audio)
        return build_voxcpm2_prompt(
            hf_config=self.engine_client.model_config.hf_config,
            tokenizer=self._voxcpm2_tokenizer,
            split_map=self._voxcpm2_split_map,
            text=request.input,
            ref_audio=ref_audio,
            ref_sr=ref_sr,
            ref_text=request.ref_text,
        )

    def _get_uploaded_audio_data(self, voice_name: str) -> str | None:
        """Get base64 encoded audio data for uploaded voice."""
        voice_name_lower = voice_name.lower()
        if voice_name_lower not in self.uploaded_speakers:
            return None

        speaker_info = self.uploaded_speakers[voice_name_lower]
        file_path = Path(speaker_info["file_path"])

        if not file_path.exists():
            logger.warning(f"Audio file not found for voice {voice_name}: {file_path}")
            return None

        try:
            # Read audio file
            with open(file_path, "rb") as f:
                audio_bytes = f.read()

            # Encode to base64
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

            # Get MIME type from file extension
            mime_type = speaker_info.get("mime_type", "audio/wav")

            # Return as data URL
            return f"data:{mime_type};base64,{audio_b64}"
        except Exception as e:
            logger.error(f"Could not read audio file for voice {voice_name}: {e}")
            return None

    def _get_uploaded_speaker_embedding(self, voice_name: str) -> list[float] | None:
        """Load pre-computed speaker embedding for an uploaded voice.

        Returns the embedding as a list of floats, or None if the voice
        was not uploaded with an embedding (i.e. it has audio instead).
        """
        voice_name_lower = voice_name.lower()
        if voice_name_lower not in self.uploaded_speakers:
            return None

        speaker_info = self.uploaded_speakers[voice_name_lower]
        if speaker_info.get("embedding_source") != "direct":
            return None

        cache_file = speaker_info.get("cache_file")
        if not cache_file or not Path(cache_file).exists():
            logger.warning("Embedding file not found for voice %s: %s", voice_name, cache_file)
            return None

        if not _validate_path_within_directory(Path(cache_file), self.uploaded_speakers_dir):
            logger.error("Cache file path traversal detected for voice %s: %s", voice_name, cache_file)
            return None

        try:
            from safetensors.torch import load_file
        except ImportError:
            logger.error(
                "The 'safetensors' package is required to load speaker embeddings. "
                "Install it with: pip install safetensors"
            )
            return None

        try:
            tensors = load_file(cache_file)
            if "speaker_embedding" not in tensors:
                logger.warning("Key 'speaker_embedding' not found in %s for voice %s", cache_file, voice_name)
                return None
            return tensors["speaker_embedding"].squeeze().tolist()
        except Exception as e:
            logger.error("Could not load embedding for voice %s: %s", voice_name, e)
            return None

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

        # Normalize voice name
        voice_name_lower = name.lower()

        # Check if voice already exists
        if voice_name_lower in self.uploaded_speakers:
            raise ValueError(
                f"Voice '{name}' already exists. To re-register this voice, delete it first and then upload it again."
            )

        # Sanitize name and consent to prevent path traversal
        sanitized_name = _sanitize_filename(name)
        sanitized_consent = _sanitize_filename(consent)

        # Generate filename with sanitized inputs
        timestamp = int(time.time())
        file_suffix = Path(audio_file.filename).suffix
        file_ext = file_suffix[1:] if file_suffix and len(file_suffix) > 1 else "wav"
        # Sanitize file extension as well
        sanitized_ext = _sanitize_filename(file_ext)
        if not sanitized_ext or sanitized_ext == "file":
            sanitized_ext = "wav"

        filename = f"{sanitized_name}_{sanitized_consent}_{timestamp}.{sanitized_ext}"
        file_path = self.uploaded_speakers_dir / filename

        # Double-check that the path is within the upload directory
        if not _validate_path_within_directory(file_path, self.uploaded_speakers_dir):
            raise ValueError("Invalid file path: potential path traversal attack detected")

        # Read content and validate duration before saving
        content = await audio_file.read()
        try:
            wav_np, sr = sf.read(io.BytesIO(content))
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
        except ValueError:
            raise
        except Exception as e:
            logger.warning("Could not validate audio duration: %s", e)

        # Save audio file
        try:
            with open(file_path, "wb") as f:
                f.write(content)
        except Exception as e:
            raise ValueError(f"Failed to save audio file: {e}")

        # Create speaker data
        speaker_data: dict[str, Any] = {
            "name": name,
            "consent": consent,
            "file_path": str(file_path),
            "created_at": timestamp,
            "mime_type": mime_type,
            "original_filename": audio_file.filename,
            "file_size": file_size,
            "ref_text": ref_text,
            "embedding_source": "audio",
        }

        # Store voice description if provided.
        if speaker_description:
            speaker_data["speaker_description"] = speaker_description

        self.uploaded_speakers[voice_name_lower] = speaker_data
        self.supported_speakers.add(voice_name_lower)

        logger.info(f"Uploaded new voice '{name}' with consent ID '{consent}'")

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
        try:
            embedding = json.loads(embedding_json)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"'speaker_embedding' must be valid JSON: {exc}") from exc

        if not isinstance(embedding, list) or not embedding:
            raise ValueError("'speaker_embedding' must be a non-empty list of numbers")

        if not all(isinstance(x, (int, float)) for x in embedding):
            raise ValueError("'speaker_embedding' must contain only numeric values")

        if not all(math.isfinite(x) for x in embedding):
            raise ValueError("'speaker_embedding' values must be finite (no NaN or Inf)")

        emb_dim = len(embedding)
        if emb_dim not in {1024, 2048}:
            logger.warning(
                "speaker_embedding has %d dimensions; expected 1024 (0.6B) or 2048 (1.7B)",
                emb_dim,
            )

        voice_name_lower = name.lower()
        if voice_name_lower in self.uploaded_speakers:
            raise ValueError(f"Voice '{name}' already exists")

        sanitized_name = _sanitize_filename(name)
        sanitized_consent = _sanitize_filename(consent)
        timestamp = int(time.time())

        # Store as safetensors for efficient loading
        try:
            import torch
            from safetensors.torch import save_file

            tensor = torch.tensor(embedding, dtype=torch.float32)
            filename = f"{sanitized_name}_{sanitized_consent}_{timestamp}.safetensors"
            file_path = self.uploaded_speakers_dir / filename

            if not _validate_path_within_directory(file_path, self.uploaded_speakers_dir):
                raise ValueError("Invalid file path: potential path traversal attack detected")

            save_file({"speaker_embedding": tensor}, str(file_path))
        except ImportError:
            raise ValueError("safetensors and torch are required for embedding upload")

        speaker_data = {
            "name": name,
            "consent": consent,
            "file_path": str(file_path),
            "created_at": timestamp,
            "mime_type": "application/x-safetensors",
            "original_filename": filename,
            "file_size": file_path.stat().st_size,
            "embedding_source": "direct",
            "embedding_dim": emb_dim,
        }

        self.uploaded_speakers[voice_name_lower] = speaker_data
        self.supported_speakers.add(voice_name_lower)

        logger.info(f"Uploaded voice '{name}' from speaker embedding ({emb_dim}-dim)")

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
        voice_name_lower = name.lower()

        if voice_name_lower not in self.uploaded_speakers:
            logger.warning(f"Voice '{name}' not found")
            return False

        speaker_info = self.uploaded_speakers.pop(voice_name_lower)
        self.supported_speakers.discard(voice_name_lower)

        # Clean up audio file on disk
        file_path = speaker_info.get("file_path")
        if file_path:
            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to delete audio file for '{name}': {e}")

        logger.info(f"Deleted voice '{name}'")
        return True

    def _is_tts_model(self) -> bool:
        """Check if the current model is a supported TTS model."""
        return any(stage.engine_args.model_stage in _TTS_MODEL_STAGES for stage in self.engine_client.stage_configs)

    def _validate_tts_request(self, request: OpenAICreateSpeechRequest) -> str | None:
        """Validate TTS request parameters. Returns error message or None."""
        if self._tts_model_type == "voxtral_tts":
            return self._validate_voxtral_tts_request(request)
        if self._tts_model_type == "fish_tts":
            return self._validate_fish_tts_request(request)
        if self._tts_model_type == "cosyvoice3":
            return self._validate_cosyvoice3_request(request)
        if self._tts_model_type == "voxcpm":
            return self._validate_voxcpm_request(request)
        if self._tts_model_type == "voxcpm2":
            return None  # VoxCPM2 accepts any text input
        return self._validate_qwen_tts_request(request)

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

    def _validate_ref_audio_format(self, ref_audio: str) -> str | None:
        """Validate ref_audio is a supported URI format. Returns error or None."""
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

    def _validate_voxcpm_request(self, request: OpenAICreateSpeechRequest) -> str | None:
        """Validate VoxCPM request parameters. Returns error message or None."""
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        if request.voice is not None:
            return "'voice' is not supported for VoxCPM"
        if request.instructions is not None:
            return "'instructions' is not supported for VoxCPM"
        if request.language is not None:
            return "'language' is not supported for VoxCPM"
        if request.task_type not in (None, "Base"):
            return "VoxCPM only supports plain TTS or voice cloning with ref_audio/ref_text"
        if request.x_vector_only_mode is not None:
            return "'x_vector_only_mode' is not supported for VoxCPM"
        if request.speaker_embedding is not None:
            return "'speaker_embedding' is not supported for VoxCPM"
        if request.initial_codec_chunk_frames is not None:
            return "'initial_codec_chunk_frames' is not supported for VoxCPM"

        if request.ref_audio is not None:
            fmt_err = self._validate_ref_audio_format(request.ref_audio)
            if fmt_err:
                return fmt_err
            if not request.ref_text or not request.ref_text.strip():
                return "Voice cloning requires 'ref_text' (transcript of the reference audio)"
        elif request.ref_text is not None:
            return "'ref_text' requires 'ref_audio' for VoxCPM voice cloning"

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
        task_type = request.task_type or "CustomVoice"

        # Normalize voice to lowercase for case-insensitive matching
        if request.voice is not None:
            request.voice = request.voice.lower()

        # Validate input is not empty
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        # Validate language
        if request.language is not None and request.language not in _TTS_LANGUAGES:
            return f"Invalid language '{request.language}'. Supported: {', '.join(sorted(_TTS_LANGUAGES))}"

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
            # ECAPA-TDNN produces 1024-dim (0.6B) or 2048-dim (1.7B)
            expected_dims = {1024, 2048}
            if emb_len not in expected_dims:
                logger.warning(
                    "speaker_embedding has %d dimensions; expected 1024 "
                    "(0.6B model) or 2048 (1.7B model). Wrong dimensions "
                    "will likely result in errors or degraded quality.",
                    emb_len,
                )
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
                # Handle the case where request.voice is NOT None
                pass
                # voice is not None
                voice_lower = request.voice.lower()
                if voice_lower in self.uploaded_speakers:
                    # Check if data file exists for uploaded speaker
                    speaker_info = self.uploaded_speakers[voice_lower]
                    file_path = Path(speaker_info["file_path"])
                    if not file_path.exists():
                        return f"Data file for uploaded speaker '{request.voice}' not found on disk"
                    # For embedding-uploaded voices, verify the cache is ready
                    if speaker_info.get("embedding_source") == "direct":
                        cache_file = speaker_info.get("cache_file")
                        if not cache_file or not Path(cache_file).exists():
                            status = speaker_info.get("cache_status", "unknown")
                            return f"Speaker embedding for '{request.voice}' is not yet ready (cache_status='{status}')"
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

    def _validate_fish_tts_request(self, request: OpenAICreateSpeechRequest) -> str | None:
        """Validate Fish Speech request parameters. Returns error message or None.

        Side effect: if request.voice references an uploaded speaker, resolves
        it to request.ref_audio and request.ref_text for voice cloning.
        """
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        # Support uploaded voices: auto-resolve voice → ref_audio + ref_text.
        if request.voice is not None and request.ref_audio is None:
            voice_lower = request.voice.lower()
            if voice_lower in self.uploaded_speakers:
                speaker_info = self.uploaded_speakers[voice_lower]
                file_path = Path(speaker_info["file_path"])
                if not file_path.exists():
                    return f"Audio file for uploaded voice '{request.voice}' not found on disk"
                audio_data_url = self._get_uploaded_audio_data(voice_lower)
                if audio_data_url is None:
                    return f"Could not load audio for uploaded voice '{request.voice}'"
                request.ref_audio = audio_data_url
                # Use ref_text from upload metadata if not provided in request.
                if not request.ref_text or not request.ref_text.strip():
                    upload_ref_text = speaker_info.get("ref_text")
                    if upload_ref_text and upload_ref_text.strip():
                        request.ref_text = upload_ref_text

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

    async def _resolve_ref_audio(self, ref_audio_str: str) -> tuple[list[float], int]:
        """Resolve ref_audio to (wav_samples, sample_rate).

        Delegates to upstream vLLM's MediaConnector which handles http(s)
        URLs, ``data:`` base64 URIs, and ``file:`` local paths (the latter
        gated by ``--allowed-local-media-path``).
        """
        # In diffusion mode, model_config may not be available
        if self._diffusion_mode:
            connector = MediaConnector()
        else:
            model_config = self.model_config
            connector = MediaConnector(
                allowed_local_media_path=model_config.allowed_local_media_path,
                allowed_media_domains=model_config.allowed_media_domains,
            )
        wav_np, sr = await connector.fetch_audio_async(ref_audio_str)
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
        return wav_np.tolist(), sr

    async def _generate_audio_chunks(
        self,
        generator,
        request_id: str,
        response_format: str = "pcm",
        raw_request: Request | None = None,
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

        try:
            async for res in generator:
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
                        wav_header = _create_wav_header(sample_rate=sample_rate_val, num_channels=1, bits_per_sample=16)
                        yield wav_header
                        first_chunk = False

                    # Convert audio to PCM bytes
                    audio_obj = CreateAudio(
                        audio_tensor=chunk_np,
                        sample_rate=sample_rate_val,
                        response_format="pcm",
                        speed=1.0,
                        stream_format="audio",
                        base64_encode=False,
                    )
                    yield self.create_audio(audio_obj).audio_data
        except asyncio.CancelledError:
            logger.info("Streaming request %s cancelled by client", request_id)
            raise
        except EngineDeadError as e:
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
            logger.exception("Streaming speech generation failed for %s: %s", request_id, e)
            raise

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
            if ro is None:
                ro = getattr(res, "request_output", None)
            outputs = getattr(ro, "outputs", None) if ro else None
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
        if self._tts_model_type == "voxcpm":
            params: dict[str, Any] = {
                "text": [request.input],
                "cfg_value": [2.0],
                "inference_timesteps": [10],
                "min_len": [2],
                "max_new_tokens": [request.max_new_tokens or 4096],
            }
            if request.ref_text is not None:
                params["ref_text"] = [request.ref_text]
            return params

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
            params["speaker"] = [request.voice]

            # Uploaded voices use task_type="Base" (CustomVoice requires built-in spk_id).
            # If ref_text was provided at upload time, use in-context cloning; otherwise x_vector only.
            if request.voice.lower() in self.uploaded_speakers and request.ref_audio is None:
                speaker_info = self.uploaded_speakers[request.voice.lower()]

                # Check if this voice was uploaded with a pre-computed embedding.
                # Populate request.speaker_embedding so the existing code path
                # (below) handles voice_clone_prompt and x_vector_only_mode.
                embedding = self._get_uploaded_speaker_embedding(request.voice)
                if embedding is not None:
                    request.speaker_embedding = embedding
                    params["task_type"] = ["Base"]
                    logger.info("Auto-set speaker_embedding for uploaded voice: %s", request.voice)
                else:
                    audio_data = self._get_uploaded_audio_data(request.voice)
                    if not audio_data:
                        raise ValueError(f"Audio file for uploaded voice '{request.voice}' is missing or corrupted")
                    stored_ref_text = speaker_info.get("ref_text")
                    params["ref_audio"] = [audio_data]
                    params["task_type"] = ["Base"]
                    params["voice_created_at"] = [speaker_info.get("created_at", 0)]
                    if stored_ref_text:
                        params["ref_text"] = [stored_ref_text]
                        params["x_vector_only_mode"] = [False]
                    else:
                        params["x_vector_only_mode"] = [True]
                    logger.info(
                        "Auto-set ref_audio for uploaded voice: %s (icl=%s)", request.voice, bool(stored_ref_text)
                    )

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

        # VoiceDesign requires non_streaming_mode (match offline script behaviour).
        # CustomVoice and Base rely on the model default (True and False respectively).
        if params["task_type"][0] == "VoiceDesign":
            params["non_streaming_mode"] = [True]

        return params

    # ---- Voxtral TTS helpers ----

    def _build_voxtral_prompt(self, request: OpenAICreateSpeechRequest) -> dict[str, Any]:
        """Build Voxtral TTS engine prompt from shared TTS parameters."""
        from mistral_common.protocol.speech.request import SpeechRequest

        text = request.input
        voice = request.voice
        ref_audio = request.ref_audio
        assert voice or ref_audio, "Either voice or ref_audio must be provided"
        # Strip data URI prefix — mistral_common expects raw base64
        if ref_audio is not None and isinstance(ref_audio, str) and ref_audio.startswith("data:"):
            _, _, ref_audio = ref_audio.partition(",")
        if self._tts_tokenizer is None:
            from vllm.tokenizers import cached_tokenizer_from_config

            mistral_tokenizer = cached_tokenizer_from_config(self.engine_client.model_config)
            self._tts_tokenizer = mistral_tokenizer.instruct
        if voice is not None:
            tokens = self._tts_tokenizer.encode_speech_request(SpeechRequest(input=text, voice=voice)).tokens
            return {
                "prompt_token_ids": tokens,
                "additional_information": {"voice": [voice]},
            }
        else:
            tokenized = self._tts_tokenizer.encode_speech_request(SpeechRequest(input=text, ref_audio=ref_audio))
            audio = tokenized.audios[0]
            return {
                "prompt_token_ids": tokenized.tokens,
                "multi_modal_data": {"audio": [(audio.audio_array, audio.sampling_rate)]},
            }

    # ---- Fish Speech helpers ----

    def _build_fish_speech_prompt(
        self,
        request: OpenAICreateSpeechRequest,
        ref_audio_data: tuple[list[float], int] | None = None,
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
            return {
                "prompt_token_ids": prompt_ids,
                "additional_information": additional_information,
            }

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
            if voice_lower in self.uploaded_speakers:
                additional_information["voice_name"] = voice_lower
                additional_information["voice_created_at"] = self.uploaded_speakers[voice_lower].get("created_at", 0)
        if request.max_new_tokens is not None:
            additional_information["max_new_tokens"] = request.max_new_tokens
        return {
            "prompt_token_ids": [1] * ph_len,
            "additional_information": additional_information,
        }

    # ---- CosyVoice3 helpers ----

    async def _build_cosyvoice3_prompt(
        self,
        request: OpenAICreateSpeechRequest,
    ) -> dict[str, Any]:
        """Build prompt for CosyVoice3.

        CosyVoice3 uses multimodal input with reference audio for voice cloning.
        The prompt format matches the offline example: text prompt + audio data
        + mm_processor_kwargs with prompt_text.
        """
        # Resolve reference audio
        wav_samples, sr = await self._resolve_ref_audio(request.ref_audio)
        audio_data = (np.asarray(wav_samples, dtype=np.float32), sr)

        return {
            "prompt": request.input,
            "multi_modal_data": {
                "audio": audio_data,
            },
            "mm_processor_kwargs": {
                "prompt_text": request.ref_text,
                "sample_rate": sr,
            },
        }

    # ---- Common speech generation helpers ----

    async def _prepare_speech_generation(
        self,
        request: OpenAICreateSpeechRequest,
        request_id: str | None = None,
    ) -> tuple[str, Any, dict[str, Any]]:
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        if self._is_fish_speech:
            validation_error = self._validate_fish_tts_request(request)
            if validation_error:
                raise ValueError(validation_error)
            ref_audio_data = None
            if request.ref_audio is not None:
                wav_list, sr = await self._resolve_ref_audio(request.ref_audio)
                ref_audio_data = (wav_list, sr)
            prompt = await self._build_fish_speech_prompt_async(request, ref_audio_data=ref_audio_data)
            tts_params = {}
        elif self._tts_model_type == "omnivoice":
            if not request.input or not request.input.strip():
                raise ValueError("Input text cannot be empty")
            tts_params = {}
            prompt: dict[str, Any] = {"input": request.input}
            # Resolve ref_audio: explicit request param or uploaded voice
            ref_src = request.ref_audio
            if not ref_src and request.voice:
                vl = request.voice.lower()
                if vl in self.uploaded_speakers:
                    sp = self.uploaded_speakers[vl]
                    if sp.get("embedding_source") == "audio":
                        ref_src = self._get_uploaded_audio_data(request.voice)
                        if not ref_src:
                            raise ValueError(f"Audio for voice '{request.voice}' missing")
                        prompt["ref_text"] = sp.get("ref_text")
            if ref_src:
                fmt_err = self._validate_ref_audio_format(ref_src)
                if fmt_err:
                    raise ValueError(fmt_err)
                wav, sr = await self._resolve_ref_audio(ref_src)
                prompt["ref_audio"] = (np.asarray(wav, dtype=np.float32), sr)
            if request.ref_text:
                prompt["ref_text"] = request.ref_text
            if request.language:
                prompt["lang"] = request.language
            if request.instructions:
                prompt["instruct"] = request.instructions
        elif self._tts_model_type == "voxcpm2":
            prompt = await self._build_voxcpm2_prompt(request)
            tts_params = {}
        elif self._is_tts:
            validation_error = self._validate_tts_request(request)
            if validation_error:
                raise ValueError(validation_error)

            if self._tts_model_type == "voxtral_tts":
                prompt = await self._build_voxtral_prompt_async(request)
                tts_params = {}
            elif self._tts_model_type == "cosyvoice3":
                prompt = await self._build_cosyvoice3_prompt(request)
                tts_params = {}
            else:
                tts_params = self._build_tts_params(request)
                # Resolve ref_audio (explicit or auto-set for uploaded voices)
                # to [[wav_list, sr]] so the model doesn't re-decode base64.
                ref_audio_source = request.ref_audio
                if ref_audio_source is None and isinstance(tts_params.get("ref_audio"), list):
                    # Uploaded voice: ref_audio was auto-set as [base64_data_url]
                    ref_audio_source = tts_params["ref_audio"][0]
                if ref_audio_source is not None and isinstance(ref_audio_source, str):
                    wav_list, sr = await self._resolve_ref_audio(ref_audio_source)
                    tts_params["ref_audio"] = [[wav_list, sr]]

                ph_len = await self._estimate_prompt_len_async(tts_params)
                prompt = {"prompt_token_ids": [1] * ph_len, "additional_information": tts_params}
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

        request_id = request_id or f"speech-{random_uuid()}"
        if self._is_fish_speech:
            model_type = "fish_speech"
        elif self._tts_model_type == "voxtral_tts":
            model_type = "voxtral_tts"
        elif self._tts_model_type == "cosyvoice3":
            model_type = "cosyvoice3"
        elif self._tts_model_type == "voxcpm":
            model_type = "voxcpm"
        elif self._tts_model_type == "voxcpm2":
            model_type = "voxcpm2"
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

        sampling_params_list = self.engine_client.default_sampling_params_list

        # CosyVoice3: set dynamic min/max tokens based on text length.
        # The official model requires min_token_text_ratio to prevent early
        # EOS and max_token_text_ratio to cap generation length.
        if self._tts_model_type == "cosyvoice3" and sampling_params_list:
            import copy

            sampling_params_list = copy.deepcopy(sampling_params_list)
            text_len = len(request.input)  # rough char-level estimate
            # Use the model's configured ratios (defaults: min=2, max=20)
            hf_cfg = self.model_config.hf_config
            min_ratio = getattr(hf_cfg, "min_token_text_ratio", 2)
            max_ratio = getattr(hf_cfg, "max_token_text_ratio", 20)
            min_tokens = max(1, int(text_len * min_ratio))
            max_tokens = min(2048, int(text_len * max_ratio))
            sampling_params_list[0].min_tokens = min_tokens
            sampling_params_list[0].max_tokens = max_tokens
            logger.info(
                "CosyVoice3 dynamic tokens: text_len=%d, min_tokens=%d, max_tokens=%d",
                text_len,
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

        # Fish defaults come from stage_configs YAML. Only override when the caller
        # explicitly requests a different generation length.
        if self._is_fish_speech and request.max_new_tokens is not None and sampling_params_list:
            import copy

            sampling_params_list = copy.deepcopy(sampling_params_list)
            sampling_params_list[0].max_tokens = request.max_new_tokens

        generator = self.engine_client.generate(
            prompt=prompt,
            request_id=request_id,
            sampling_params_list=sampling_params_list,
            output_modalities=["audio"],
        )
        return request_id, generator, tts_params

    async def _generate_pcm_chunks(self, generator, request_id: str):
        """Yield raw PCM byte chunks from the engine generator.

        Delegates to ``_generate_audio_chunks`` with ``response_format="pcm"``.
        Used by the WebSocket streaming handler and ``_iter_pcm_audio_bytes``.
        """
        async for chunk in self._generate_audio_chunks(generator, request_id, response_format="pcm"):
            yield chunk

    async def _iter_pcm_audio_bytes(self, request: OpenAICreateSpeechRequest):
        """Yield raw PCM bytes for a speech request as soon as chunks are decoded."""
        request_id, generator, _ = await self._prepare_speech_generation(request)
        async for chunk in self._generate_pcm_chunks(generator, request_id):
            yield chunk

    async def _generate_audio_bytes(
        self,
        request: OpenAICreateSpeechRequest,
        base64_encode: bool = False,
        request_id: str | None = None,
    ) -> tuple[bytes | str, str]:
        request_id, generator, _ = await self._prepare_speech_generation(request, request_id=request_id)

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
            stream_format=request.stream_format,
            base64_encode=base64_encode,
        )
        audio_response: AudioResponse = self.create_audio(audio_obj)
        return audio_response.audio_data, audio_response.media_type

    async def _create_diffusion_speech(
        self,
        request: OpenAICreateSpeechRequest,
    ) -> Response:
        """Handle speech generation for pure diffusion TTS models (e.g. OmniVoice)."""
        from vllm_omni.outputs import OmniRequestOutput

        try:
            if not request.input or not request.input.strip():
                raise ValueError("Input text cannot be empty")

            request_id = f"speech-{random_uuid()}"
            prompt: dict[str, Any] = {"input": request.input}
            if request.ref_audio:
                wav, sr = await self._resolve_ref_audio(request.ref_audio)
                prompt["ref_audio"] = (np.asarray(wav, dtype=np.float32), sr)
            if request.ref_text:
                prompt["ref_text"] = request.ref_text
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

            generator = self._diffusion_engine.generate(
                prompt=prompt,
                request_id=request_id,
                sampling_params_list=self._diffusion_engine.default_sampling_params_list,
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
                stream_format=request.stream_format,
                base64_encode=False,
            )
            audio_response: AudioResponse = self.create_audio(audio_obj)
            return Response(content=audio_response.audio_data, media_type=audio_response.media_type)

        except asyncio.CancelledError:
            return self._diffusion_error_response("Client disconnected")
        except (EngineGenerateError, EngineDeadError):
            raise  # Propagate to the global Omni exception handler
        except ValueError as e:
            return self._diffusion_error_response(str(e))
        except Exception as e:
            logger.exception("Diffusion speech generation failed: %s", e)
            return self._diffusion_error_response(f"Speech generation failed: {e}")

    @staticmethod
    def _diffusion_error_response(message: str) -> Response:
        """Create a JSON error response without depending on OpenAIServing."""
        error_body = json.dumps({"error": {"message": message, "type": "server_error", "param": None, "code": 500}})
        return Response(content=error_body, media_type="application/json", status_code=500)

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

        Streaming is supported via stream=True with response_format='pcm' or 'wav'.
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
        if raw_request:
            raw_request.state.request_metadata = RequestResponseMetadata(
                request_id=request_id,
            )

        try:
            if request.stream:
                # Determine response format and media type for streaming
                response_format = (request.response_format or "wav").lower()

                # Only pcm and wav support streaming without post-processing
                if response_format not in ["pcm", "wav"]:
                    return self.create_error_response(
                        f"Streaming is only supported for 'pcm' and 'wav' formats. "
                        f"Got '{response_format}'. For other formats, use stream=False."
                    )

                # Check if speed adjustment is requested (not compatible with streaming)
                if request.speed is not None and request.speed != 1.0:
                    return self.create_error_response(
                        "Streaming is not supported with speed adjustment. "
                        "Use stream=False or remove the speed parameter."
                    )

                media_type = "audio/wav" if response_format == "wav" else "audio/pcm"
                _, generator, _ = await self._prepare_speech_generation(request, request_id=request_id)
                return StreamingResponse(
                    self._generate_audio_chunks(
                        generator,
                        request_id,
                        response_format,
                        raw_request=raw_request,
                    ),
                    media_type=media_type,
                )

            audio_bytes, media_type = await self._generate_audio_bytes(request, request_id=request_id)
            return Response(content=audio_bytes, media_type=media_type)

        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except (EngineGenerateError, EngineDeadError):
            raise  # Propagate to the global Omni exception handler
        except ValueError as e:
            return self.create_error_response(e)
        except Exception as e:
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
            validation_error = self._validate_tts_request(req)
            if validation_error is not None:
                return SpeechBatchItemResult(index=idx, status="error", error=validation_error)
            try:
                audio_data, media_type = await self._generate_audio_bytes(req, base64_encode=True)
            except Exception as e:
                logger.exception("Batch item %d failed: %s", idx, e)
                return SpeechBatchItemResult(index=idx, status="error", error=str(e))
            return SpeechBatchItemResult(
                index=idx,
                status="success",
                audio_data=audio_data,
                media_type=media_type,
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
