import asyncio
import base64
import json
import math
import os
import re
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from transformers.utils.hub import cached_file
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.logger import init_logger
from vllm.multimodal.media import MediaConnector
from vllm.utils import random_uuid

from vllm_omni.entrypoints.openai.audio_utils_mixin import AudioMixin
from vllm_omni.entrypoints.openai.metadata_manager import MetadataManager
from vllm_omni.entrypoints.openai.protocol.audio import (
    AudioResponse,
    CreateAudio,
    OpenAICreateSpeechRequest,
)
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)

# TTS Configuration
_VOXTRAL_TTS_MODEL_STAGES = {"audio_generation"}
_QWEN3_TTS_MODEL_STAGES = {"qwen3_tts"}
_FISH_TTS_MODEL_STAGES = {"fish_speech_slow_ar"}
_TTS_MODEL_STAGES: set[str] = _VOXTRAL_TTS_MODEL_STAGES | _QWEN3_TTS_MODEL_STAGES | _FISH_TTS_MODEL_STAGES
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize uploaded speakers storage
        speech_voice_samples_dir = os.environ.get("SPEECH_VOICE_SAMPLES", "/tmp/voice_samples")
        self.uploaded_speakers_dir = Path(speech_voice_samples_dir)
        self.uploaded_speakers_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.uploaded_speakers_dir / "metadata.json"

        # Initialize metadata manager
        self.metadata_manager = MetadataManager(self.metadata_file)

        # Find and cache the TTS stage (if any) during initialization
        self._tts_stage = self._find_tts_stage()
        self._is_tts = self._tts_stage is not None
        self._is_fish_speech = (
            self._tts_stage is not None and getattr(self._tts_stage, "model_stage", None) == "fish_speech_slow_ar"
        )
        self._fish_speech_tokenizer = None

        # Determine TTS model type or None
        self._tts_model_type = self._detect_tts_model_type()

        # Cache TTS configuration values (computed once, reused per request)
        self._max_instructions_length = self._compute_max_instructions_length()

        # Load supported speakers
        self.supported_speakers = self._load_supported_speakers()
        # Load uploaded speakers
        self.uploaded_speakers = self.metadata_manager.get_uploaded_speakers()

        # Merge supported speakers with uploaded speakers
        self.supported_speakers.update(self.uploaded_speakers.keys())
        self._tts_tokenizer = None

        logger.info(f"Loaded {len(self.supported_speakers)} supported speakers: {sorted(self.supported_speakers)}")
        logger.info(f"Loaded {len(self.uploaded_speakers)} uploaded speakers")

        # Load speech tokenizer codec parameters for prompt length estimation
        self._codec_frame_rate: float | None = self._load_codec_frame_rate()

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
        if model_stage in _QWEN3_TTS_MODEL_STAGES:
            return "qwen3_tts"
        if model_stage in _VOXTRAL_TTS_MODEL_STAGES:
            return "voxtral_tts"
        if model_stage in _FISH_TTS_MODEL_STAGES:
            return "fish_tts"
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

    async def upload_voice(self, audio_file: UploadFile, consent: str, name: str) -> dict:
        """Upload a new voice sample."""
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
            raise ValueError(f"Voice '{name}' already exists")

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

        # Save audio file
        try:
            with open(file_path, "wb") as f:
                content = await audio_file.read()
                f.write(content)
        except Exception as e:
            raise ValueError(f"Failed to save audio file: {e}")

        # Create speaker data
        speaker_data = {
            "name": name,
            "consent": consent,
            "file_path": str(file_path),
            "created_at": timestamp,
            "mime_type": mime_type,
            "original_filename": audio_file.filename,
            "file_size": file_size,
            "cache_status": "pending",  # The initial cache state is pending.
            "cache_file": None,  # The initial cache file is empty.
            "cache_generated_at": None,  # The initial cache generation time is empty.
        }

        # Save metadata using metadata manager (concurrency safe)
        success = self.metadata_manager.create_speaker(voice_name_lower, speaker_data)
        if not success:
            # Clean up the saved file if metadata creation failed
            try:
                file_path.unlink()
            except Exception:
                pass
            raise ValueError(f"Failed to create metadata for voice '{name}' (possibly already exists)")

        # Update in-memory cache
        self.uploaded_speakers[voice_name_lower] = speaker_data
        self.supported_speakers.add(voice_name_lower)

        logger.info(f"Uploaded new voice '{name}' with consent ID '{consent}'")

        # Return voice information without exposing the server file path
        return {
            "name": name,
            "consent": consent,
            "created_at": timestamp,
            "mime_type": mime_type,
            "file_size": file_size,
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

        # Check if voice exists in memory cache
        if voice_name_lower not in self.uploaded_speakers:
            logger.warning(f"Voice '{name}' not found in memory cache")
            return False

        # Delete from metadata manager with file cleanup
        # Pass base_dir for path validation
        deleted_info = self.metadata_manager.delete_speaker(voice_name_lower)
        if not deleted_info:
            logger.error(f"Failed to delete voice '{name}' from metadata")
            return False

        # Update in-memory cache
        if voice_name_lower in self.uploaded_speakers:
            del self.uploaded_speakers[voice_name_lower]
        if voice_name_lower in self.supported_speakers:
            self.supported_speakers.remove(voice_name_lower)

        logger.info(f"Deleted voice '{name}' and associated files")
        return True

    def _is_tts_model(self) -> bool:
        """Check if the current model is a supported TTS model."""
        return any(stage.engine_args.model_stage in _TTS_MODEL_STAGES for stage in self.engine_client.stage_configs)

    def _validate_tts_request(self, request: OpenAICreateSpeechRequest) -> str | None:
        """Validate TTS request parameters. Returns error message or None."""
        if self._tts_model_type == "voxtral_tts":
            return self._validate_voxtral_tts_request(request)
        return self._validate_qwen_tts_request(request)

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
                return f"Invalid speaker '{request.voice}'. Supported: {', '.join(sorted(self.supported_speakers))}"

        # Validate Base task requirements
        if task_type == "Base":
            if request.voice is None:
                if request.ref_audio is None:
                    return "Base task requires 'ref_audio' for voice cloning"
                fmt_err = self._validate_ref_audio_format(request.ref_audio)
                if fmt_err:
                    return fmt_err
                # In-context voice cloning (default) requires non-empty ref_text.
                # x_vector_only_mode skips in-context and only uses speaker embedding.
                if not request.x_vector_only_mode:
                    if not request.ref_text or not request.ref_text.strip():
                        return (
                            "Base task requires non-empty 'ref_text' (transcript of "
                            "the reference audio) unless 'x_vector_only_mode' is enabled"
                        )
            else:
                # voice is not None
                voice_lower = request.voice.lower()
                if voice_lower in self.uploaded_speakers:
                    # Check if audio file exists for uploaded speaker
                    speaker_info = self.uploaded_speakers[voice_lower]
                    file_path = Path(speaker_info["file_path"])
                    if not file_path.exists():
                        return f"Audio file for uploaded speaker '{request.voice}' not found on disk"
                else:
                    # need ref_audio for built-in speaker
                    if request.ref_audio is None:
                        return (
                            f"Base task with built-in speaker '{request.voice}' requires 'ref_audio' for voice cloning"
                        )
                    fmt_err = self._validate_ref_audio_format(request.ref_audio)
                    if fmt_err:
                        return fmt_err

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

    async def _resolve_ref_audio(self, ref_audio_str: str) -> tuple[list[float], int]:
        """Resolve ref_audio to (wav_samples, sample_rate).

        Delegates to upstream vLLM's MediaConnector which handles http(s)
        URLs, ``data:`` base64 URIs, and ``file:`` local paths (the latter
        gated by ``--allowed-local-media-path``).
        """
        model_config = self.model_config
        connector = MediaConnector(
            allowed_local_media_path=model_config.allowed_local_media_path,
            allowed_media_domains=model_config.allowed_media_domains,
        )
        wav_np, sr = await connector.fetch_audio_async(ref_audio_str)
        wav_np = np.asarray(wav_np, dtype=np.float32)
        if wav_np.ndim > 1:
            wav_np = np.mean(wav_np, axis=-1)
        return wav_np.tolist(), int(sr)

    async def _generate_audio_chunks(self, generator, request_id: str, response_format: str = "pcm"):
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
        if not mm:
            ro = getattr(res, "request_output", None)
            mm = getattr(ro, "multimodal_output", None) if ro else None
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
            params["speaker"] = [request.voice]

            # If voice is an uploaded speaker and no ref_audio provided, auto-set it
            if request.voice.lower() in self.uploaded_speakers and request.ref_audio is None:
                audio_data = self._get_uploaded_audio_data(request.voice)
                if audio_data:
                    params["ref_audio"] = [audio_data]
                    params["x_vector_only_mode"] = [True]
                    logger.info(f"Auto-set ref_audio for uploaded voice: {request.voice}")
                else:
                    raise ValueError(f"Audio file for uploaded voice '{request.voice}' is missing or corrupted")

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
        if request.x_vector_only_mode is not None:
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

    async def _build_voxtral_prompt(self, request: OpenAICreateSpeechRequest) -> dict[str, Any]:
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
          <|im_start|>user\\n<|speaker:0|>{text}<|im_end|>\\n<|im_start|>assistant\\n<|voice|>

        With voice cloning (ref_audio + ref_text):
          <|im_start|>system\\n<|speaker:0|>{ref_text}<|audio_start|>{semantic_tokens}<|audio_end|><|im_end|>
          <|im_start|>user\\n<|speaker:0|>{text}<|im_end|>\\n<|im_start|>assistant\\n<|voice|>
        """
        from transformers import AutoTokenizer

        if self._fish_speech_tokenizer is None:
            model_name = self.engine_client.model_config.model
            self._fish_speech_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        tokenizer = self._fish_speech_tokenizer
        model_name = self.engine_client.model_config.model

        if ref_audio_data is not None and request.ref_text:
            # Voice cloning: encode reference audio and build system message.
            from vllm_omni.model_executor.models.fish_speech.dac_encoder import (
                encode_reference_audio,
            )

            wav_samples, sr = ref_audio_data
            semantic_token_ids = encode_reference_audio(model_name, wav_samples, sr)

            # Build system message with ref text + audio tokens.
            audio_start_id = tokenizer.encode("<|audio_start|>", add_special_tokens=False)
            audio_end_id = tokenizer.encode("<|audio_end|>", add_special_tokens=False)

            # System content: <|speaker:0|>{ref_text}<|audio_start|>{codes}<|audio_end|>
            prefix_text = f"<|speaker:0|>{request.ref_text}"
            prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
            system_content_ids = prefix_ids + audio_start_id + semantic_token_ids + audio_end_id

            # Manually build system turn: <|im_start|>system\n{content}<|im_end|>\n
            im_start = tokenizer.encode("<|im_start|>", add_special_tokens=False)
            im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
            system_tag = tokenizer.encode("system\n", add_special_tokens=False)
            newline = tokenizer.encode("\n", add_special_tokens=False)
            system_ids = im_start + system_tag + system_content_ids + im_end + newline

            # User turn via chat template.
            user_text = f"<|speaker:0|>{request.input}"
            user_messages = [{"role": "user", "content": user_text}]
            user_ids = tokenizer.apply_chat_template(user_messages, tokenize=True, add_generation_prompt=True)
            prompt_ids = system_ids + user_ids
        else:
            # No voice cloning: simple user message.
            user_text = f"<|speaker:0|>{request.input}"
            messages = [{"role": "user", "content": user_text}]
            prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

        # Append <|voice|> token to signal voice generation.
        voice_token_id = tokenizer.encode("<|voice|>", add_special_tokens=False)
        prompt_ids = prompt_ids + voice_token_id

        additional_information: dict[str, Any] = {
            "text": [request.input],
            "max_new_tokens": [request.max_new_tokens or 4096],
        }

        return {
            "prompt_token_ids": prompt_ids,
            "additional_information": additional_information,
        }

    # ---- Common speech generation helpers ----

    async def _prepare_speech_generation(
        self,
        request: OpenAICreateSpeechRequest,
    ) -> tuple[str, Any, dict[str, Any]]:
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        if self._is_fish_speech:
            if not request.input or not request.input.strip():
                raise ValueError("Input text cannot be empty")
            ref_audio_data = None
            if request.ref_audio is not None:
                if not request.ref_text or not request.ref_text.strip():
                    raise ValueError("Voice cloning requires 'ref_text' (transcript of the reference audio)")
                wav_list, sr = await self._resolve_ref_audio(request.ref_audio)
                ref_audio_data = (wav_list, sr)
            prompt = self._build_fish_speech_prompt(request, ref_audio_data=ref_audio_data)
            tts_params = {}
        elif self._is_tts:
            validation_error = self._validate_tts_request(request)
            if validation_error:
                raise ValueError(validation_error)

            if self._tts_model_type == "voxtral_tts":
                prompt = await self._build_voxtral_prompt(request)
                tts_params = {}
            else:
                tts_params = self._build_tts_params(request)
                if request.ref_audio is not None:
                    wav_list, sr = await self._resolve_ref_audio(request.ref_audio)
                    tts_params["ref_audio"] = [[wav_list, sr]]

                ph_len = self._estimate_prompt_len(tts_params)
                prompt = {"prompt_token_ids": [1] * ph_len, "additional_information": tts_params}
        else:
            tts_params = {}
            prompt = {"prompt": request.input}

        request_id = f"speech-{random_uuid()}"
        if self._is_fish_speech:
            model_type = "fish_speech"
        elif self._tts_model_type == "voxtral_tts":
            model_type = "voxtral_tts"
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

        # Override Stage-0 max_tokens if caller specified max_new_tokens (Fish Speech).
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

    async def _iter_pcm_audio_bytes(self, request: OpenAICreateSpeechRequest):
        """Yield raw PCM bytes for a speech request as soon as chunks are decoded."""
        request_id, generator, _ = await self._prepare_speech_generation(request)
        async for chunk in self._generate_pcm_chunks(generator, request_id):
            yield chunk

    async def _generate_audio_bytes(
        self,
        request: OpenAICreateSpeechRequest,
    ) -> tuple[bytes, str]:
        request_id, generator, _ = await self._prepare_speech_generation(request)

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
            import torch

            audio_tensor = torch.cat(audio_tensor, dim=-1)
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
        return audio_response.audio_data, audio_response.media_type

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
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

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
                request_id, generator, _ = await self._prepare_speech_generation(request)
                return StreamingResponse(
                    self._generate_audio_chunks(generator, request_id, response_format),
                    media_type=media_type,
                )

            audio_bytes, media_type = await self._generate_audio_bytes(request)
            return Response(content=audio_bytes, media_type=media_type)

        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            return self.create_error_response(e)
        except Exception as e:
            logger.exception("Speech generation failed: %s", e)
            return self.create_error_response(f"Speech generation failed: {e}")
