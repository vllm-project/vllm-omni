"""Stand-alone builder for Qwen3-TTS AR talker prompt embeddings.

Factors the prompt-construction logic out of
:class:`vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker.Qwen3TTSTalkerForConditionalGeneration`
so alternative AR talker implementations that share the same backbone
embedding tables and codec / speaker encoders can reuse it without
duplicating the (substantial) prompt layout logic.

The builder is intentionally *not* coupled to the talker class. The talker
constructs a builder, supplies its frozen embedding tables, projection
layers, codec embed lookups, speaker encoder, the precomputed pad-token
embedding buffer, and a callable that batch-encodes reference audio into
codec frames (so the builder doesn't have to know whether the talker
uses :class:`Qwen3TTSTokenizer` or :class:`Qwen3TTSTokenizerV2Encoder`
under the hood). The single public entry point
:meth:`Qwen3TTSPromptEmbedsBuilder.build_prompt_embeds` mirrors the
official ``Qwen3TTSForConditionalGeneration`` prompt layout for the three
supported task types (``CustomVoice``, ``VoiceDesign``, ``Base``) and
both streaming / non-streaming modes.
"""

from __future__ import annotations

import base64
import hashlib
import io
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from vllm.logger import init_logger
from vllm.multimodal.audio import AudioResampler

from vllm_omni.utils.audio import mel_filter_bank

if TYPE_CHECKING:
    from .configuration_qwen3_tts import Qwen3TTSConfig, Qwen3TTSTalkerConfig

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Constants shared with the talker's preprocess pipeline.
#
# These keys mark precomputed artifacts that the batched preprocess pass
# stashes into ``info_dict`` so the per-request prompt build can skip
# redundant tokenization / codec encoding work.
# ---------------------------------------------------------------------------

PRECOMPUTED_REF_CODE_KEY = "precomputed_ref"
NORMALIZED_REF_AUDIO_KEY = "_qwen3_tts_normalized_ref_audio"
REF_AUDIO_CACHE_KEY = "_qwen3_tts_ref_audio_cache_key"
PRECOMPUTED_TEXT_IDS_KEY = "_qwen3_tts_text_ids"
PRECOMPUTED_REF_IDS_KEY = "_qwen3_tts_ref_ids"


# ---------------------------------------------------------------------------
# Chat-template helpers (must match the official HuggingFace reference).
# ---------------------------------------------------------------------------


def build_assistant_text(text: str) -> str:
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


def build_ref_text(text: str) -> str:
    return f"<|im_start|>assistant\n{text}<|im_end|>\n"


def build_instruct_text(instruct: str) -> str:
    return f"<|im_start|>user\n{instruct}<|im_end|>\n"


# ---------------------------------------------------------------------------
# Type-coercion helpers. All accept the loose shapes that survive
# msgspec IPC ``additional_information`` round-trips (scalars, single-
# element lists, ndarrays, tensors).
# ---------------------------------------------------------------------------


def first_value(value: object, default: object = None) -> object:
    if isinstance(value, list):
        return value[0] if value else default
    return value if value is not None else default


def coerce_ref_code_tensor(value: object, *, device: torch.device) -> torch.Tensor | None:
    value = first_value(value)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        ref_code = value
    elif isinstance(value, np.ndarray):
        ref_code = torch.from_numpy(value)
    elif isinstance(value, list) and value:
        ref_code = torch.as_tensor(value, dtype=torch.long)
    else:
        return None
    if ref_code.ndim == 3:
        ref_code = ref_code[0]
    if ref_code.ndim != 2 or ref_code.numel() == 0:
        return None
    return ref_code.to(device=device, dtype=torch.long).contiguous()


def coerce_token_ids(value: object, *, device: torch.device) -> torch.Tensor | None:
    value = first_value(value)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        ids = value
    elif isinstance(value, np.ndarray):
        ids = torch.from_numpy(value)
    elif isinstance(value, list) and value and all(isinstance(v, (int, np.integer)) for v in value):
        ids = torch.tensor(value, dtype=torch.long)
    else:
        return None
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.ndim != 2 or ids.numel() == 0:
        return None
    return ids.to(device=device, dtype=torch.long).contiguous()


# ---------------------------------------------------------------------------
# Mel-spectrogram utilities used by the speaker encoder front-end.
# Module-level LRU caches for the (sr, n_fft, n_mels, ...) -> mel_basis and
# win_size -> hann_window tensors avoid rebuilding them on every request.
# ---------------------------------------------------------------------------


def _dynamic_range_compression(x: torch.Tensor, c: float = 1, clip_val: float = 1e-5) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=clip_val) * c)


@lru_cache(maxsize=8)
def _cached_mel_filter_bank(sampling_rate: int, n_fft: int, n_mels: int, fmin: int, fmax: int | None) -> torch.Tensor:
    return mel_filter_bank(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)


@lru_cache(maxsize=8)
def _cached_hann_window(win_size: int) -> torch.Tensor:
    return torch.hann_window(win_size)


def mel_spectrogram(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: int | None = None,
    center: bool = False,
    mel_basis: torch.Tensor | None = None,
    hann_window: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mel spectrogram via torch STFT and a (cached) mel filterbank."""
    if torch.min(y) < -1.0:
        logger.warning("Min value of input waveform signal is %s", torch.min(y))
    if torch.max(y) > 1.0:
        logger.warning("Max value of input waveform signal is %s", torch.max(y))
    device = y.device
    if mel_basis is None:
        mel_basis = _cached_mel_filter_bank(sampling_rate, n_fft, num_mels, fmin, fmax).to(device)
    elif mel_basis.device != device:
        mel_basis = mel_basis.to(device)
    if hann_window is None:
        hann_window = _cached_hann_window(win_size).to(device)
    elif hann_window.device != device:
        hann_window = hann_window.to(device)
    padding = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(y.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    mel_spec = torch.matmul(mel_basis, spec)
    return _dynamic_range_compression(mel_spec)


# ---------------------------------------------------------------------------
# Local audio loading helpers (URL / base64 / local path).
# ---------------------------------------------------------------------------


def _is_probably_base64(s: str) -> bool:
    if s.startswith("data:audio"):
        return True
    if ("/" not in s and "\\" not in s) and len(s) > 256:
        return True
    return False


def _is_url(s: str) -> bool:
    try:
        u = urlparse(s)
        if u.scheme in ("http", "https"):
            return bool(u.netloc)
        return u.scheme == "file"
    except Exception:
        return False


def _decode_base64_to_wav_bytes(b64: str) -> bytes:
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)


def load_audio_to_np(x: str) -> tuple[np.ndarray, int]:
    """Load mono float32 audio from a local path, URL, or base64 data URI.

    Uses upstream vLLM's MediaConnector for http(s) URLs and ``file:`` URIs
    with unrestricted local access (offline inference is trusted).
    """
    from vllm.multimodal.media.audio import load_audio

    if _is_url(x):
        from vllm.multimodal.media import MediaConnector

        connector = MediaConnector(allowed_local_media_path="/")
        audio, sr = connector.fetch_audio(x)
    elif _is_probably_base64(x):
        wav_bytes = _decode_base64_to_wav_bytes(x)
        with io.BytesIO(wav_bytes) as f:
            audio, sr = sf.read(f, dtype="float32", always_2d=False)
    else:
        audio, sr = load_audio(x, sr=None, mono=True)

    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = np.mean(audio, axis=-1)

    return np.asarray(audio, dtype=np.float32), int(sr)


# ---------------------------------------------------------------------------
# Prompt-embeds builder.
# ---------------------------------------------------------------------------


class Qwen3TTSPromptEmbedsBuilder:
    """Build prefill prompt embeddings for the Qwen3-TTS AR talker.

    Stand-alone wrapper around the embedding layers, encoders, and speaker
    cache required to assemble a talker prompt. Mirrors the layout of the
    official HuggingFace ``Qwen3TTSForConditionalGeneration`` pipeline for
    ``task_type in {CustomVoice, VoiceDesign, Base}`` and both streaming /
    non-streaming modes.

    The builder does **not** own the speech / codec tokenizer used to turn
    raw reference audio into codec frames. The caller passes in a
    ``encode_ref_audio_batch`` callable so the same builder works against
    either the legacy :class:`Qwen3TTSTokenizer` HF wrapper or the native
    :class:`Qwen3TTSTokenizerV2Encoder` exposed on the talker.

    Args:
        config: Top-level Qwen3-TTS config; supplies ``tts_*_token_id`` and
            ``speaker_encoder_config``.
        talker_config: Talker sub-config; supplies the codec control-token
            ids, ``num_code_groups``, ``spk_id``, ``spk_is_dialect`` and
            ``codec_language_id`` maps.
        model_path: HuggingFace repo or local path used to lazily load the
            text tokenizer.
        text_embedding: Text-side embedding table.
        text_projection: Callable projecting text-side embeddings into the
            talker hidden size.
        codec_embed: Callable performing codec-side embedding lookup
            (normally the talker backbone's ``embed_input_ids``).
        residual_code_embeddings: Callable returning the per-codebook
            embedding modules used by the code predictor; called lazily so
            the predictor doesn't need to be constructed before the builder.
        speaker_encoder: ECAPA-TDNN speaker encoder used by the Base task.
        tts_pad_embed: Pre-computed ``tts_pad`` embedding (shape
            ``[1, hidden]``) reused at every prefill to avoid recomputing.
        encode_ref_audio_batch: Callable ``(wavs, sr, *, device) -> list[Tensor]``
            that returns one ``[T, num_quantizers]`` long tensor per input
            waveform. Single-waveform encoding is implemented in terms of
            this callable.
        speaker_cache: Optional :class:`SpeakerEmbeddingCache`. When
            provided, Base ``CustomVoice``-style results are cached by
            ``(speaker_name, mode, created_at)``.
        ref_audio_artifact_cache_max_entries: Upper bound on the SHA1
            keyed LRU cache of per-ref-audio ``(ref_code, ref_spk_embedding)``
            artifacts. ``0`` disables the cache.
    """

    def __init__(
        self,
        *,
        config: Qwen3TTSConfig,
        talker_config: Qwen3TTSTalkerConfig,
        model_path: str,
        text_embedding: nn.Module,
        text_projection: Callable[[torch.Tensor], torch.Tensor],
        codec_embed: Callable[[torch.Tensor], torch.Tensor],
        residual_code_embeddings: Callable[[], Sequence[Callable[[torch.Tensor], torch.Tensor]]],
        speaker_encoder: nn.Module,
        tts_pad_embed: torch.Tensor,
        encode_ref_audio_batch: Callable[..., list[torch.Tensor]],
        speaker_cache: Any | None = None,
        ref_audio_artifact_cache_max_entries: int = 256,
    ):
        self._config = config
        self._talker_config = talker_config
        self._model_path = model_path
        self._text_embedding = text_embedding
        self._text_projection = text_projection
        self._codec_embed = codec_embed
        self._residual_code_embeddings = residual_code_embeddings
        self._speaker_encoder = speaker_encoder
        self._tts_pad_embed_buffer = tts_pad_embed
        self._encode_ref_audio_batch_fn = encode_ref_audio_batch
        self._speaker_cache = speaker_cache
        self._embedding_dtype = torch.bfloat16

        self._text_tokenizer: Any | None = None

        self._ref_audio_artifact_cache_max_entries = int(ref_audio_artifact_cache_max_entries)
        self._ref_audio_artifact_cache: OrderedDict[str, dict[str, torch.Tensor | bool]] = OrderedDict()

        # Bounded LRU; caller-supplied orig_sr can otherwise grow this without limit.
        self._resampler_cache: OrderedDict[tuple[int, int], AudioResampler] = OrderedDict()
        self._resampler_cache_max = 16

    # -------------------- device / buffer helpers --------------------

    def _device(self) -> torch.device:
        # text_embedding is always present and lives on the talker device.
        return next(self._text_embedding.parameters()).device

    def _pad_embed(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return the pad-token embedding on the requested device/dtype, shape ``[1, 1, H]``."""
        return self._tts_pad_embed_buffer.to(device=device, dtype=dtype).reshape(1, 1, -1)

    def _get_resampler(self, orig_sr: int, target_sr: int) -> AudioResampler:
        key = (int(orig_sr), int(target_sr))
        cache = self._resampler_cache
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        cache[key] = AudioResampler(target_sr=int(target_sr))
        if len(cache) > self._resampler_cache_max:
            cache.popitem(last=False)
        return cache[key]

    # -------------------- text tokenizer (lazy) --------------------

    def get_text_tokenizer(self):
        if self._text_tokenizer is None:
            import transformers

            kwargs: dict[str, Any] = dict(trust_remote_code=True, use_fast=True)
            if transformers.__version__ < "5":
                kwargs["fix_mistral_regex"] = True
            tok = AutoTokenizer.from_pretrained(self._model_path, **kwargs)
            tok.padding_side = "left"
            self._text_tokenizer = tok
        return self._text_tokenizer

    # -------------------- ref-audio handling --------------------

    def load_audio_to_np(self, x: str) -> tuple[np.ndarray, int]:
        return load_audio_to_np(x)

    def normalize_ref_audio(self, ref_audio: object) -> tuple[np.ndarray, int]:
        """Coerce a serialized ref_audio payload into ``(wav_np, sample_rate)``.

        ``additional_information`` may serialize ``(wav, sr)`` into (nested)
        lists across processes, so we accept and unwrap a variety of shapes.
        """
        if isinstance(ref_audio, str):
            return self.load_audio_to_np(ref_audio)

        def _is_sr(x: object) -> bool:
            try:
                v = int(x)  # type: ignore[arg-type]
            except Exception:
                return False
            return 1_000 <= v <= 200_000

        def _is_number_sequence(xs: list[object]) -> bool:
            if not xs:
                return False
            for v in xs[:8]:
                if not isinstance(v, (int, float, np.number)):
                    return False
            return True

        wav_candidates: list[object] = []
        sr_candidates: list[int] = []

        def _summarize(obj: object, depth: int = 0) -> str:
            if depth > 2:
                if isinstance(obj, (int, np.integer)):
                    return f"int({int(obj)})"
                return type(obj).__name__
            if obj is None:
                return "None"
            if isinstance(obj, str):
                if len(obj) <= 16:
                    return f"str({obj!r})"
                return f"str(len={len(obj)})"
            if isinstance(obj, (int, float, np.number)):
                return f"{type(obj).__name__}({obj})"
            if isinstance(obj, np.ndarray):
                return f"ndarray(shape={obj.shape}, dtype={obj.dtype})"
            if isinstance(obj, torch.Tensor):
                return f"Tensor(shape={tuple(obj.shape)}, dtype={obj.dtype}, device={obj.device})"
            if isinstance(obj, dict):
                keys = list(obj.keys())
                return f"dict(keys={keys[:8]})"
            if isinstance(obj, (tuple, list)):
                items = list(obj)
                head = ", ".join(_summarize(x, depth + 1) for x in items[:3])
                return f"{type(obj).__name__}(len={len(items)}; head=[{head}])"
            return f"{type(obj).__name__}"

        def _scan(obj: object, depth: int = 0) -> None:
            if depth > 4:
                return
            if obj is None:
                return
            if _is_sr(obj):
                sr_candidates.append(int(obj))  # type: ignore[arg-type]
                return
            if isinstance(obj, np.ndarray) and obj.size > 0:
                wav_candidates.append(obj)
                return
            if isinstance(obj, torch.Tensor) and obj.numel() > 0:
                wav_candidates.append(obj)
                return
            if isinstance(obj, dict):
                # Inlined ndarray/tensor payloads from the input processor.
                if obj.get("__ndarray__") and "data" in obj and "dtype" in obj and "shape" in obj:
                    try:
                        data = obj["data"]
                        dtype = obj["dtype"]
                        shape = obj["shape"]
                        if isinstance(data, (bytes, bytearray, memoryview)):
                            arr = np.frombuffer(data, dtype=dtype).reshape(shape)
                            if arr.size > 0:
                                wav_candidates.append(arr)
                                return
                    except Exception:
                        pass
                if obj.get("__tensor__") and "data" in obj and "dtype" in obj and "shape" in obj:
                    try:
                        data = obj["data"]
                        dtype = obj["dtype"]
                        shape = obj["shape"]
                        if isinstance(data, (bytes, bytearray, memoryview)):
                            # Stored as raw CPU bytes; interpret as numpy for audio.
                            np_dtype = np.dtype(dtype)
                            arr = np.frombuffer(data, dtype=np_dtype).reshape(shape)
                            if arr.size > 0:
                                wav_candidates.append(arr)
                                return
                    except Exception:
                        pass
                wav_obj = obj.get("array") or obj.get("wav") or obj.get("audio")
                sr_obj = obj.get("sampling_rate") or obj.get("sr") or obj.get("sample_rate")
                if wav_obj is not None:
                    _scan(wav_obj, depth + 1)
                if sr_obj is not None:
                    _scan(sr_obj, depth + 1)
                return
            if isinstance(obj, (tuple, list)):
                obj_list = list(obj)
                # Unwrap singleton nesting ([[wav, sr]]).
                while isinstance(obj_list, list) and len(obj_list) == 1:
                    inner = obj_list[0]
                    if isinstance(inner, np.ndarray) and inner.size > 0:
                        wav_candidates.append(inner)
                        return
                    if isinstance(inner, torch.Tensor) and inner.numel() > 0:
                        wav_candidates.append(inner)
                        return
                    if isinstance(inner, dict):
                        _scan(inner, depth + 1)
                        return
                    if isinstance(inner, (tuple, list)):
                        obj_list = list(inner)  # type: ignore[list-item]
                        continue
                    break

                # If the *unwrapped* list is a long list of numbers, treat it as waveform.
                if len(obj_list) >= 512 and _is_number_sequence(obj_list):
                    wav_candidates.append(obj_list)
                    return

                # Otherwise, recurse into elements (but avoid descending into huge numeric lists).
                for item in obj_list:
                    if isinstance(item, list) and len(item) >= 512 and _is_number_sequence(item):  # type: ignore[arg-type]
                        wav_candidates.append(item)
                        continue
                    _scan(item, depth + 1)
                return

        _scan(ref_audio)
        if not sr_candidates:
            raise TypeError(f"ref_audio missing sample_rate: {_summarize(ref_audio)}")
        sr = int(sr_candidates[0])

        def _wav_len(x: object) -> int:
            try:
                if isinstance(x, np.ndarray):
                    return int(x.size)
                if isinstance(x, torch.Tensor):
                    return int(x.numel())
                if isinstance(x, list):
                    return int(len(x))
            except Exception:
                pass
            return 0

        if not wav_candidates:
            raise TypeError(f"ref_audio missing waveform: {_summarize(ref_audio)}")
        wav_obj = max(wav_candidates, key=_wav_len)

        def _to_np(x: object) -> np.ndarray:
            if isinstance(x, np.ndarray):
                return x.astype(np.float32).reshape(-1)
            if isinstance(x, torch.Tensor):
                return x.detach().to("cpu").float().contiguous().numpy().reshape(-1)
            if isinstance(x, dict) and x.get("__ndarray__") and "data" in x and "dtype" in x and "shape" in x:
                data = x["data"]
                dtype = x["dtype"]
                shape = x["shape"]
                if isinstance(data, (bytes, bytearray, memoryview)):
                    return np.frombuffer(data, dtype=dtype).reshape(shape).astype(np.float32).reshape(-1)
            if isinstance(x, list):
                # list of numbers
                if len(x) >= 2 and _is_number_sequence(x):  # type: ignore[arg-type]
                    return np.asarray(x, dtype=np.float32).reshape(-1)
                # list of chunks
                parts: list[np.ndarray] = []
                for part in x:
                    if isinstance(part, (np.ndarray, torch.Tensor, list)):
                        parts.append(_to_np(part))
                if parts:
                    return np.concatenate(parts, axis=0)
            raise TypeError(f"Unsupported waveform type: {type(x)}")

        wav_np = _to_np(wav_obj)
        if wav_np.size < 1024:
            raise ValueError(f"ref_audio waveform too short: {wav_np.size} samples")
        return wav_np, sr

    # -------------------- ref-audio artifact cache --------------------

    @staticmethod
    def make_ref_audio_cache_key(wav: np.ndarray, sr: int) -> str:
        wav_f32 = wav.astype(np.float32, copy=False).reshape(-1)
        h = hashlib.sha1()
        h.update(int(sr).to_bytes(4, byteorder="little", signed=False))
        h.update(int(wav_f32.size).to_bytes(8, byteorder="little", signed=False))
        h.update(wav_f32.tobytes(order="C"))
        return h.hexdigest()

    def get_ref_audio_artifacts(self, cache_key: str) -> dict[str, torch.Tensor | bool] | None:
        if self._ref_audio_artifact_cache_max_entries <= 0:
            return None
        entry = self._ref_audio_artifact_cache.get(cache_key)
        if entry is None:
            return None
        self._ref_audio_artifact_cache.move_to_end(cache_key, last=True)
        return entry

    def put_ref_audio_artifacts(
        self,
        cache_key: str,
        *,
        ref_code: torch.Tensor | None = None,
        ref_spk_embedding: torch.Tensor | None = None,
    ) -> None:
        if self._ref_audio_artifact_cache_max_entries <= 0 or not cache_key:
            return
        entry = self._ref_audio_artifact_cache.get(cache_key)
        if entry is None:
            entry = {}
        if isinstance(ref_code, torch.Tensor):
            entry["ref_code"] = ref_code.detach().to("cpu", dtype=torch.long).contiguous()
        if isinstance(ref_spk_embedding, torch.Tensor):
            entry["ref_spk_embedding"] = ref_spk_embedding.detach().to("cpu", dtype=self._embedding_dtype).reshape(-1)
        if not entry:
            return
        self._ref_audio_artifact_cache[cache_key] = entry
        self._ref_audio_artifact_cache.move_to_end(cache_key, last=True)
        while len(self._ref_audio_artifact_cache) > self._ref_audio_artifact_cache_max_entries:
            self._ref_audio_artifact_cache.popitem(last=False)

    # -------------------- speaker encoder / codec encoder --------------------

    def extract_speaker_embedding(self, wav: np.ndarray, sr: int) -> torch.Tensor:
        # vLLM workers do not automatically move arbitrary torch.nn.Modules to
        # CUDA. Ensure the speaker encoder is on the same device/dtype as the
        # main model before running it.
        dev = self._device()
        dtype = self._embedding_dtype
        try:
            spk_param = next(self._speaker_encoder.parameters())
            if spk_param.device != dev or spk_param.dtype != dtype:
                self._speaker_encoder.to(device=dev, dtype=dtype)
        except StopIteration:
            pass

        # Resample to 24kHz for speaker encoder.
        target_sr = int(getattr(self._config.speaker_encoder_config, "sample_rate", 24000))
        if sr != target_sr:
            resampler = self._get_resampler(int(sr), target_sr)
            wav = resampler.resample(wav.astype(np.float32), orig_sr=int(sr))
            sr = target_sr

        # Follow official implementation: mel_spectrogram expects 24kHz. Move
        # the waveform first so STFT/mel computation stays on the model device
        # instead of materializing a CPU mel tensor and copying it per request.
        wav_tensor = torch.from_numpy(wav).to(device=dev, dtype=torch.float32).unsqueeze(0)
        mels = mel_spectrogram(
            wav_tensor,
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)
        spk = self._speaker_encoder(mels.to(dtype=dtype))[0]
        return spk.to(dtype=dtype)

    def encode_ref_audio_batch(
        self,
        wavs: list[np.ndarray],
        sr: int,
        *,
        device: torch.device,
    ) -> list[torch.Tensor]:
        """Delegate to the talker-provided batched ref-audio encoder."""
        return self._encode_ref_audio_batch_fn(wavs, int(sr), device=device)

    def encode_ref_audio_to_code(self, wav: np.ndarray, sr: int) -> torch.Tensor:
        """Single-waveform convenience wrapper around :meth:`encode_ref_audio_batch`."""
        codes = self.encode_ref_audio_batch([wav], int(sr), device=self._device())
        if not codes:
            raise ValueError("encode_ref_audio_batch returned no codes")
        return codes[0]

    # -------------------- batched preprocess (cross-request) --------------------

    @staticmethod
    def _voice_clone_prompt_dict(raw: object) -> dict[str, object] | None:
        raw = first_value(raw)
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            return raw[0]
        return None

    @staticmethod
    def _has_ref_code_like(value: object) -> bool:
        value = first_value(value)
        if isinstance(value, torch.Tensor):
            return value.numel() > 0 and value.ndim in (2, 3)
        if isinstance(value, np.ndarray):
            return value.size > 0 and value.ndim in (2, 3)
        if isinstance(value, list):
            return bool(value)
        return False

    @staticmethod
    def _needs_initial_prompt_preprocess(info_dict: dict[str, Any]) -> bool:
        embed = info_dict.get("embed")
        return not (isinstance(embed, dict) and "prefill" in embed)

    def _needs_batched_ref_code(self, info_dict: dict[str, Any]) -> bool:
        if not self._needs_initial_prompt_preprocess(info_dict):
            return False
        if first_value(info_dict.get("task_type"), "CustomVoice") != "Base":
            return False
        codes = info_dict.get("codes")
        if isinstance(codes, dict) and self._has_ref_code_like(codes.get(PRECOMPUTED_REF_CODE_KEY)):
            return False

        voice_clone_prompt = self._voice_clone_prompt_dict(info_dict.get("voice_clone_prompt"))
        if voice_clone_prompt is not None:
            if self._has_ref_code_like(voice_clone_prompt.get("ref_code")):
                return False
            icl_flag = first_value(voice_clone_prompt.get("icl_mode"))
            if isinstance(icl_flag, bool) and not icl_flag:
                return False

        xvec_only = first_value(info_dict.get("x_vector_only_mode"), False)
        if isinstance(xvec_only, bool) and xvec_only:
            return False

        ref_audio_list = info_dict.get("ref_audio")
        return isinstance(ref_audio_list, list) and bool(ref_audio_list)

    @torch.inference_mode()
    def preprocess_batch(
        self,
        *,
        req_ids: list[str],
        model_intermediate_buffer: dict[str, dict[str, Any]],
        device: torch.device,
    ) -> None:
        """Batch Base voice-clone ref-audio codec extraction for current prefill requests.

        Tokenizes assistant / ref text and runs the talker-supplied
        ``encode_ref_audio_batch`` callable once per sample-rate group,
        stashing the precomputed artifacts under
        :data:`PRECOMPUTED_TEXT_IDS_KEY`, :data:`PRECOMPUTED_REF_IDS_KEY`,
        :data:`PRECOMPUTED_REF_CODE_KEY`, and :data:`NORMALIZED_REF_AUDIO_KEY`
        so the per-request :meth:`build_prompt_embeds` call can skip the
        equivalent serial work.
        """
        pending_text: list[tuple[dict[str, Any], str]] = []
        pending_ref_text: list[tuple[dict[str, Any], str]] = []
        groups: dict[int, list[tuple[dict[str, Any], np.ndarray, int]]] = {}
        for req_id in req_ids:
            info_dict = model_intermediate_buffer.get(req_id)
            if not isinstance(info_dict, dict):
                continue

            if (
                self._needs_initial_prompt_preprocess(info_dict)
                and first_value(info_dict.get("task_type"), "CustomVoice") == "Base"
            ):
                if PRECOMPUTED_TEXT_IDS_KEY not in info_dict:
                    text = first_value(info_dict.get("text"), "")
                    if isinstance(text, str):
                        pending_text.append((info_dict, build_assistant_text(text)))

                if PRECOMPUTED_REF_IDS_KEY not in info_dict:
                    ref_text = first_value(info_dict.get("ref_text"))
                    if isinstance(ref_text, str) and ref_text.strip():
                        pending_ref_text.append((info_dict, build_ref_text(ref_text)))

            if not self._needs_batched_ref_code(info_dict):
                continue
            ref_audio_list = info_dict.get("ref_audio")
            if not isinstance(ref_audio_list, list) or not ref_audio_list:
                continue
            cache_key_from_serving = first_value(info_dict.get(REF_AUDIO_CACHE_KEY))
            if isinstance(cache_key_from_serving, str) and cache_key_from_serving:
                cached = self.get_ref_audio_artifacts(cache_key_from_serving)
                if cached is not None:
                    cached_ref_code = coerce_ref_code_tensor(cached.get("ref_code"), device=device)
                    if isinstance(cached_ref_code, torch.Tensor):
                        info_dict.setdefault("codes", {})[PRECOMPUTED_REF_CODE_KEY] = cached_ref_code
                        continue
            try:
                wav, sr = self.normalize_ref_audio(ref_audio_list[0])
            except Exception:
                # Keep the original per-request path responsible for surfacing
                # invalid ref_audio errors with its existing messages.
                continue
            info_dict[NORMALIZED_REF_AUDIO_KEY] = (wav, sr)
            cache_key = self.make_ref_audio_cache_key(wav, sr)
            info_dict[REF_AUDIO_CACHE_KEY] = cache_key
            cached = self.get_ref_audio_artifacts(cache_key)
            if cached is not None:
                cached_ref_code = coerce_ref_code_tensor(cached.get("ref_code"), device=device)
                if isinstance(cached_ref_code, torch.Tensor):
                    info_dict.setdefault("codes", {})[PRECOMPUTED_REF_CODE_KEY] = cached_ref_code
                    continue
            groups.setdefault(int(sr), []).append((info_dict, wav, int(sr)))

        if pending_text or pending_ref_text:
            tok_text = self.get_text_tokenizer()
            for items, key in (
                (pending_text, PRECOMPUTED_TEXT_IDS_KEY),
                (pending_ref_text, PRECOMPUTED_REF_IDS_KEY),
            ):
                if not items:
                    continue
                try:
                    tokenized = tok_text([text for _, text in items], padding=False)
                except Exception as exc:
                    logger.debug("Qwen3-TTS batched text tokenization failed; falling back to serial path: %s", exc)
                    continue
                input_ids = tokenized.get("input_ids") if isinstance(tokenized, dict) else None
                if not isinstance(input_ids, list) or len(input_ids) != len(items):
                    continue
                for (info_dict, _), ids in zip(items, input_ids, strict=True):
                    if isinstance(ids, list) and ids:
                        info_dict[key] = torch.tensor(ids, dtype=torch.long)

        groups = {sr: items for sr, items in groups.items() if len(items) >= 2}
        if not groups:
            return

        for sr, items in groups.items():
            wavs = [wav for _, wav, _ in items]
            try:
                ref_codes = self.encode_ref_audio_batch(wavs, int(sr), device=device)
            except Exception as exc:
                logger.debug("Qwen3-TTS batched ref_code encode failed; falling back to serial path: %s", exc)
                continue
            for (info_dict, wav, item_sr), ref_code in zip(items, ref_codes, strict=True):
                ref_code_t = coerce_ref_code_tensor(ref_code, device=device)
                if ref_code_t is None:
                    continue
                info_dict.setdefault("codes", {})[PRECOMPUTED_REF_CODE_KEY] = ref_code_t
                info_dict[NORMALIZED_REF_AUDIO_KEY] = (wav, item_sr)
                cache_key = info_dict.get(REF_AUDIO_CACHE_KEY)
                if isinstance(cache_key, str) and cache_key:
                    self.put_ref_audio_artifacts(cache_key, ref_code=ref_code_t)

    # -------------------- ICL prompt assembly --------------------

    def _generate_icl_prompt(
        self,
        *,
        text_id: torch.Tensor,
        ref_id: torch.Tensor,
        ref_code: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        tts_eos_embed: torch.Tensor,
        non_streaming_mode: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Port of the official ``Qwen3TTSForConditionalGeneration.generate_icl_prompt``."""
        text_embed = self._text_projection(self._text_embedding(torch.cat([ref_id, text_id], dim=-1)))
        text_embed = torch.cat([text_embed, tts_eos_embed], dim=1)

        # codec embed (codec bos + codec) 1 T2 D
        residual_embeds = self._residual_code_embeddings()
        codec_embed: list[torch.Tensor] = []
        for i in range(int(self._talker_config.num_code_groups)):
            if i == 0:
                codec_embed.append(self._codec_embed(ref_code[:, :1]))
            else:
                codec_embed.append(residual_embeds[i - 1](ref_code[:, i : i + 1]))
        codec_embed_sum = torch.cat(codec_embed, dim=1).sum(1).unsqueeze(0)  # [1,T,H]
        codec_embed_sum = torch.cat(
            [
                self._codec_embed(
                    torch.tensor([[self._talker_config.codec_bos_id]], device=codec_embed_sum.device, dtype=torch.long)
                ),
                codec_embed_sum,
            ],
            dim=1,
        )

        text_lens = int(text_embed.shape[1])
        codec_lens = int(codec_embed_sum.shape[1])
        if non_streaming_mode:
            # Official non-streaming mode: append the full text conditioning in
            # prefill, and use PAD in decode steps.
            icl_input_embed = text_embed + self._codec_embed(
                torch.tensor(
                    [[self._talker_config.codec_pad_id] * text_lens],
                    device=codec_embed_sum.device,
                    dtype=torch.long,
                )
            )
            icl_input_embed = torch.cat([icl_input_embed, codec_embed_sum + tts_pad_embed], dim=1)
            return icl_input_embed, tts_pad_embed
        if text_lens > codec_lens:
            return text_embed[:, :codec_lens] + codec_embed_sum, text_embed[:, codec_lens:]
        text_embed = torch.cat([text_embed] + [tts_pad_embed] * (codec_lens - text_lens), dim=1)
        return text_embed + codec_embed_sum, tts_pad_embed

    # -------------------- main entry point --------------------

    def build_prompt_embeds(
        self,
        *,
        task_type: str,
        info_dict: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, int | None, torch.Tensor | None]:
        """Build the prefill prompt embeddings for a single TTS request.

        Args:
            task_type: One of ``"CustomVoice"``, ``"VoiceDesign"``, ``"Base"``.
            info_dict: Per-request ``additional_information`` plus any
                precomputed artifacts stashed by the batched preprocess pass
                (``PRECOMPUTED_TEXT_IDS_KEY``, ``PRECOMPUTED_REF_IDS_KEY``,
                ``PRECOMPUTED_REF_CODE_KEY``, ``NORMALIZED_REF_AUDIO_KEY``).
                Consumed keys are popped so the caller doesn't re-feed them
                on subsequent prefill chunks.

        Returns:
            ``(talker_prompt, trailing_text_hidden, ref_code_len, ref_code)``:

            * ``talker_prompt``: ``[prompt_len, hidden]`` prefill embedding.
            * ``trailing_text_hidden``: ``[T, hidden]`` queue of text-side
                    embeddings consumed one-per-decode-step (streaming mode)
                    or a single pad row (non-streaming).
            * ``ref_code_len``: Number of ref codec frames consumed when the
                    Base task ran in in-context mode (``None`` otherwise).
            * ``ref_code``: The ``[T, Q]`` ref codec tensor used for ICL
                    (``None`` outside Base / ICL).
        """
        config = self._config
        talker_config = self._talker_config
        text_embedding = self._text_embedding
        text_projection = self._text_projection
        codec_embed = self._codec_embed

        text = (info_dict.get("text") or [""])[0]
        language = (info_dict.get("language") or ["Auto"])[0]
        non_streaming_mode_val = info_dict.get("non_streaming_mode")
        if isinstance(non_streaming_mode_val, list):
            non_streaming_mode_raw = non_streaming_mode_val[0] if non_streaming_mode_val else None
        else:
            non_streaming_mode_raw = non_streaming_mode_val
        if isinstance(non_streaming_mode_raw, bool):
            non_streaming_mode = non_streaming_mode_raw
        else:
            # Match official inference defaults:
            # - CustomVoice/VoiceDesign: non_streaming_mode=True
            # - Base: non_streaming_mode=False
            non_streaming_mode = task_type in ("CustomVoice", "VoiceDesign")

        # Text ids for assistant template (always).
        tok = self.get_text_tokenizer()
        dev = self._device()
        input_ids = coerce_token_ids(info_dict.pop(PRECOMPUTED_TEXT_IDS_KEY, None), device=dev)
        if input_ids is None:
            input_ids = tok(build_assistant_text(text), return_tensors="pt", padding=False)["input_ids"].to(device=dev)

        # Optional instruct prefix.
        instruct = (info_dict.get("instruct") or [""])[0]
        instruct_embed = None
        if isinstance(instruct, str) and instruct.strip():
            instruct_ids = tok(build_instruct_text(instruct), return_tensors="pt", padding=False)["input_ids"].to(
                device=input_ids.device
            )
            instruct_embed = text_projection(text_embedding(instruct_ids))

        # tts special token embeds (projected into talker hidden).
        # ``tts_pad_embed`` is precomputed (request-independent), so we only
        # need bos/eos here.
        tts_tokens = torch.tensor(
            [[config.tts_bos_token_id, config.tts_eos_token_id]],
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        tts_bos_embed, tts_eos_embed = text_projection(text_embedding(tts_tokens)).chunk(2, dim=1)
        tts_pad_embed = self._pad_embed(input_ids.device, tts_bos_embed.dtype)

        # Codec prefill tags.
        language_id = None
        if isinstance(language, str) and language.lower() != "auto":
            language_id = talker_config.codec_language_id.get(language.lower())
        # Match official dialect override:
        # If language is Chinese/Auto and the selected speaker is a dialect voice,
        # set language_id to that dialect to improve code generation stability.
        if language_id is None and isinstance(language, str) and language.lower() in ("chinese", "auto"):
            speaker_for_dialect = None
            if task_type == "CustomVoice":
                speaker_for_dialect = (info_dict.get("speaker") or [""])[0]
            if isinstance(speaker_for_dialect, str) and speaker_for_dialect.strip():
                spk_is_dialect = getattr(talker_config, "spk_is_dialect", None) or {}
                dialect = spk_is_dialect.get(speaker_for_dialect.lower())
                if isinstance(dialect, str) and dialect:
                    language_id = talker_config.codec_language_id.get(dialect)
        if language_id is None:
            codec_prefill_list = [
                [
                    talker_config.codec_nothink_id,
                    talker_config.codec_think_bos_id,
                    talker_config.codec_think_eos_id,
                ]
            ]
        else:
            codec_prefill_list = [
                [
                    talker_config.codec_think_id,
                    talker_config.codec_think_bos_id,
                    int(language_id),
                    talker_config.codec_think_eos_id,
                ]
            ]

        codec_input_0 = codec_embed(torch.tensor(codec_prefill_list, device=input_ids.device, dtype=torch.long))
        codec_input_1 = codec_embed(
            torch.tensor([[talker_config.codec_pad_id, talker_config.codec_bos_id]], device=input_ids.device)
        )

        # Speaker embedding/token (task-dependent)
        speaker_embed: torch.Tensor | None = None
        ref_code_len: int | None = None
        ref_code_prompt: torch.Tensor | None = None

        def _as_singleton(x: object) -> object:
            if isinstance(x, list):
                return x[0] if x else None
            return x

        def _to_long_tensor(x: object, *, device: torch.device) -> torch.Tensor | None:
            x = _as_singleton(x)
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                t = x
            elif isinstance(x, np.ndarray):
                t = torch.from_numpy(x)
            elif isinstance(x, list) and x and all(isinstance(v, (int, np.integer)) for v in x):
                t = torch.tensor(x, dtype=torch.long)
            else:
                return None
            if t.ndim == 1:
                t = t.unsqueeze(0)
            return t.to(device=device, dtype=torch.long)

        def _normalize_voice_clone_prompt(raw: object) -> dict[str, object] | None:
            raw = _as_singleton(raw)
            if raw is None:
                return None
            if isinstance(raw, dict):
                return raw
            # Some callers may pass list[dict] directly.
            if isinstance(raw, list) and raw and isinstance(raw[0], dict):
                return raw[0]
            return None

        if task_type == "Base":
            # Base supports voice clone prompt with in-context mode.
            xvec_only = bool((info_dict.get("x_vector_only_mode") or [False])[0])
            in_context_mode = not xvec_only
            voice_clone_prompt = _normalize_voice_clone_prompt(info_dict.get("voice_clone_prompt"))
            if voice_clone_prompt is not None and "icl_mode" in voice_clone_prompt:
                icl_flag = _as_singleton(voice_clone_prompt.get("icl_mode"))
                if isinstance(icl_flag, bool):
                    in_context_mode = icl_flag
                    xvec_only = not in_context_mode
            prompt_has_ref_code = voice_clone_prompt is not None and self._has_ref_code_like(
                voice_clone_prompt.get("ref_code")
            )
            prompt_spk = voice_clone_prompt.get("ref_spk_embedding") if voice_clone_prompt is not None else None
            prompt_has_speaker = isinstance(prompt_spk, torch.Tensor) or (
                isinstance(prompt_spk, (list, np.ndarray)) and len(prompt_spk) > 0
            )
            needs_ref_audio_cache_key = not (prompt_has_speaker and (not in_context_mode or prompt_has_ref_code))
            ref_audio_wav: np.ndarray | None = None
            ref_audio_sr: int | None = None

            def _get_ref_audio() -> tuple[np.ndarray, int]:
                nonlocal ref_audio_wav, ref_audio_sr
                if ref_audio_wav is None or ref_audio_sr is None:
                    normalized_ref_audio = info_dict.pop(NORMALIZED_REF_AUDIO_KEY, None)
                    if (
                        isinstance(normalized_ref_audio, tuple)
                        and len(normalized_ref_audio) == 2
                        and isinstance(normalized_ref_audio[0], np.ndarray)
                    ):
                        ref_audio_wav = normalized_ref_audio[0]
                        ref_audio_sr = int(normalized_ref_audio[1])
                        return ref_audio_wav, ref_audio_sr
                    ref_audio_list = info_dict.get("ref_audio")
                    if not isinstance(ref_audio_list, list) or not ref_audio_list:
                        raise ValueError("Base requires `ref_audio`.")
                    ref_audio_wav, ref_audio_sr = self.normalize_ref_audio(ref_audio_list[0])
                return ref_audio_wav, ref_audio_sr

            # Per-ref-audio artifact cache: usually pre-populated by
            # preprocess_batch. Fall back to on-demand key creation for
            # paths it skips but that still extract an ECAPA spk_embed
            # from ref_audio: x_vector_only_mode, precomputed ref_code
            # with no ref_spk_embedding, voice_clone_prompt.icl_mode=False.
            ref_audio_cache_key = info_dict.pop(REF_AUDIO_CACHE_KEY, None)
            ref_audio_cache_key = first_value(ref_audio_cache_key)
            if needs_ref_audio_cache_key and not (isinstance(ref_audio_cache_key, str) and ref_audio_cache_key):
                normalized = info_dict.get(NORMALIZED_REF_AUDIO_KEY)
                wav_peek: np.ndarray | None = None
                sr_peek: int | None = None
                if isinstance(normalized, tuple) and len(normalized) == 2 and isinstance(normalized[0], np.ndarray):
                    wav_peek, sr_peek = normalized[0], int(normalized[1])
                else:
                    ref_audio_list = info_dict.get("ref_audio")
                    if isinstance(ref_audio_list, list) and ref_audio_list:
                        try:
                            wav_peek, sr_peek = self.normalize_ref_audio(ref_audio_list[0])
                            info_dict[NORMALIZED_REF_AUDIO_KEY] = (wav_peek, int(sr_peek))
                        except Exception:
                            wav_peek = None
                if isinstance(wav_peek, np.ndarray) and sr_peek is not None:
                    ref_audio_cache_key = self.make_ref_audio_cache_key(wav_peek, sr_peek)
            cached_artifacts = None
            if isinstance(ref_audio_cache_key, str) and ref_audio_cache_key:
                cached_artifacts = self.get_ref_audio_artifacts(ref_audio_cache_key)
            ref_audio_list = info_dict.get("ref_audio")
            has_ref_audio_payload = isinstance(ref_audio_list, list) and bool(ref_audio_list)
            artifact_only = (
                isinstance(ref_audio_cache_key, str) and bool(ref_audio_cache_key) and not has_ref_audio_payload
            )
            if artifact_only and cached_artifacts is None:
                raise RuntimeError(
                    "Qwen3-TTS ref_audio artifact cache miss for artifact-only request; "
                    "retry with ref_audio or precompute a custom voice profile."
                )

            # Speaker cache: only for uploaded (named) speakers
            _speaker_cache_key = None
            _cache_lookup_voice = None
            if voice_clone_prompt is None and self._speaker_cache is not None:
                _speaker_list = info_dict.get("speaker")
                if isinstance(_speaker_list, list) and _speaker_list:
                    _voice_name = str(_speaker_list[0]).lower()
                    _cache_lookup_voice = _voice_name
                    # Per-mode namespace — xvec and icl produce different artifacts
                    # for the same voice, so they must not share a cache slot.
                    _mode = "xvec" if xvec_only else "icl"
                    _voice_created_at = int((info_dict.get("voice_created_at") or [0])[0])
                    _speaker_cache_key = self._speaker_cache.make_cache_key(
                        _voice_name,
                        model_type=f"qwen3_tts_{_mode}",
                        created_at=_voice_created_at,
                    )
                    _cached = self._speaker_cache.get(_speaker_cache_key)
                    if _cached is not None:
                        ref_code_cached = _cached.get("ref_code")
                        ref_spk_embed_cached = _cached.get("ref_spk_embedding")
                        if isinstance(ref_code_cached, torch.Tensor):
                            ref_code_cached = ref_code_cached.to(device=input_ids.device)
                        if isinstance(ref_spk_embed_cached, torch.Tensor):
                            ref_spk_embed_cached = ref_spk_embed_cached.to(device=input_ids.device)
                        voice_clone_prompt = {
                            "ref_code": ref_code_cached,
                            "ref_spk_embedding": ref_spk_embed_cached,
                            "icl_mode": _cached.get("icl_mode"),
                            "ref_text": _cached.get("ref_text"),
                        }
                        _speaker_cache_key = None  # hit → don't store again

            if voice_clone_prompt is None and _speaker_cache_key is not None:
                ref_audio_list = info_dict.get("ref_audio")
                if not isinstance(ref_audio_list, list) or not ref_audio_list:
                    raise ValueError(
                        f"Qwen3-TTS speaker '{_cache_lookup_voice}' was requested without ref_audio, "
                        "but no precomputed cache entry was loaded"
                    )

            ref_code = None
            if voice_clone_prompt is not None:
                ref_code = _as_singleton(voice_clone_prompt.get("ref_code"))
            ref_code_t = None
            if isinstance(ref_code, torch.Tensor):
                ref_code_t = ref_code
            elif isinstance(ref_code, np.ndarray):
                ref_code_t = torch.from_numpy(ref_code)
            if isinstance(ref_code_t, torch.Tensor):
                if ref_code_t.ndim == 3:
                    ref_code_t = ref_code_t[0]
                ref_code_t = ref_code_t.to(device=input_ids.device, dtype=torch.long)
                ref_code_len = int(ref_code_t.shape[0])
            else:
                codes = info_dict.get("codes")
                precomputed_ref_code = codes.get(PRECOMPUTED_REF_CODE_KEY) if isinstance(codes, dict) else None
                ref_code_t = coerce_ref_code_tensor(precomputed_ref_code, device=input_ids.device)
                if isinstance(ref_code_t, torch.Tensor):
                    ref_code_len = int(ref_code_t.shape[0])
            if ref_code_t is None and in_context_mode and cached_artifacts is not None:
                cached_ref_code = coerce_ref_code_tensor(cached_artifacts.get("ref_code"), device=input_ids.device)
                if isinstance(cached_ref_code, torch.Tensor):
                    ref_code_t = cached_ref_code
                    ref_code_len = int(ref_code_t.shape[0])
            if ref_code_t is None and in_context_mode and artifact_only:
                raise RuntimeError("Qwen3-TTS ref_audio artifact cache entry is missing ref_code.")
            if ref_code_t is None and in_context_mode:
                # Compute ref_code from ref_audio if not provided.
                wav_np, sr = _get_ref_audio()
                ref_code_t = self.encode_ref_audio_to_code(wav_np, sr).to(device=input_ids.device)
                ref_code_len = int(ref_code_t.shape[0])
            if isinstance(ref_code_t, torch.Tensor):
                ref_code_prompt = ref_code_t

            # Speaker embedding: use prompt embed if provided; otherwise extract from audio.
            # NOTE: Do NOT use _as_singleton here — the embedding may be a plain
            # float list (from API via msgspec IPC) that _as_singleton would
            # destructively unwrap to a single scalar.
            speaker_dtype = self._embedding_dtype
            spk = None
            if voice_clone_prompt is not None:
                spk = voice_clone_prompt.get("ref_spk_embedding")
            if isinstance(spk, torch.Tensor):
                speaker_embed = spk.to(device=input_ids.device, dtype=speaker_dtype).view(1, 1, -1)
            elif isinstance(spk, (list, np.ndarray)):
                speaker_embed = torch.tensor(spk, dtype=speaker_dtype, device=input_ids.device).view(1, 1, -1)
            elif cached_artifacts is not None and isinstance(cached_artifacts.get("ref_spk_embedding"), torch.Tensor):
                speaker_embed = (
                    cached_artifacts["ref_spk_embedding"]
                    .to(device=input_ids.device, dtype=speaker_dtype)
                    .view(1, 1, -1)
                )
            elif artifact_only:
                raise RuntimeError("Qwen3-TTS ref_audio artifact cache entry is missing ref_spk_embedding.")
            else:
                wav_np, sr = _get_ref_audio()
                speaker_embed = self.extract_speaker_embedding(wav_np, sr).view(1, 1, -1)

            # Cache miss: store extraction result in the speaker cache.
            if _speaker_cache_key is not None and speaker_embed is not None and self._speaker_cache is not None:
                self._speaker_cache.put(
                    _speaker_cache_key,
                    {
                        "ref_code": ref_code_prompt.detach().cpu()
                        if isinstance(ref_code_prompt, torch.Tensor)
                        else None,
                        "ref_spk_embedding": speaker_embed.detach().cpu().reshape(-1),
                        "icl_mode": in_context_mode,
                    },
                )

            # Cache miss: store extraction result in the per-ref-audio artifact cache.
            if isinstance(ref_audio_cache_key, str) and ref_audio_cache_key:
                self.put_ref_audio_artifacts(
                    ref_audio_cache_key,
                    ref_code=ref_code_prompt if isinstance(ref_code_prompt, torch.Tensor) else None,
                    ref_spk_embedding=speaker_embed.reshape(-1) if isinstance(speaker_embed, torch.Tensor) else None,
                )

            codec_input = torch.cat([codec_input_0, speaker_embed, codec_input_1], dim=1)

            # Role header (<|im_start|>assistant\n) -> projected text embeds.
            role_embed = text_projection(text_embedding(input_ids[:, :3]))

            codec_prefix = torch.cat((tts_pad_embed.expand(-1, codec_input.shape[1] - 2, -1), tts_bos_embed), dim=1)
            codec_prefix = codec_prefix + codec_input[:, :-1]
            talker_prompt = torch.cat((role_embed, codec_prefix), dim=1)

            if in_context_mode:
                # Prefer explicit tokenized `ref_ids` if provided (matches official signature).
                ref_ids = coerce_token_ids(info_dict.pop(PRECOMPUTED_REF_IDS_KEY, None), device=input_ids.device)
                if ref_ids is None:
                    ref_ids = _to_long_tensor(info_dict.get("ref_ids"), device=input_ids.device)
                if ref_ids is None and voice_clone_prompt is not None:
                    ref_ids = _to_long_tensor(
                        voice_clone_prompt.get("ref_ids") or voice_clone_prompt.get("ref_id"),
                        device=input_ids.device,
                    )
                if ref_ids is None:
                    ref_text = _as_singleton(info_dict.get("ref_text"))
                    if (not isinstance(ref_text, str) or not ref_text.strip()) and voice_clone_prompt is not None:
                        ref_text = _as_singleton(voice_clone_prompt.get("ref_text"))
                    if isinstance(ref_text, str) and ref_text.strip():
                        ref_ids = tok(
                            build_ref_text(ref_text),
                            return_tensors="pt",
                            padding=False,
                        )["input_ids"].to(device=input_ids.device)
                    else:
                        logger.warning("Base ICL: ref_text/ref_ids missing, falling back to x-vector-only mode.")
                        in_context_mode = False
            if in_context_mode:
                icl_input_embed, trailing_text_hidden = self._generate_icl_prompt(
                    text_id=input_ids[:, 3:-5],
                    ref_id=ref_ids[:, 3:-2],
                    ref_code=ref_code_t,  # type: ignore[arg-type]
                    tts_pad_embed=tts_pad_embed,
                    tts_eos_embed=tts_eos_embed,
                    non_streaming_mode=non_streaming_mode,
                )
                talker_prompt = torch.cat([talker_prompt, icl_input_embed], dim=1)
            else:
                # First text token (+ codec_bos).
                if non_streaming_mode:
                    # Official non-streaming mode: put the full text into the
                    # prefill prompt and use PAD for decode steps.
                    text_all = text_projection(text_embedding(input_ids[:, 3:-5]))
                    text_all = torch.cat([text_all, tts_eos_embed], dim=1)
                    pad_ids = torch.full(
                        (1, int(text_all.shape[1])),
                        int(talker_config.codec_pad_id),
                        device=input_ids.device,
                        dtype=torch.long,
                    )
                    talker_prompt = torch.cat(
                        [
                            talker_prompt,
                            text_all + codec_embed(pad_ids),
                            tts_pad_embed
                            + codec_embed(torch.tensor([[talker_config.codec_bos_id]], device=input_ids.device)),
                        ],
                        dim=1,
                    )
                    trailing_text_hidden = tts_pad_embed
                else:
                    first_text = text_projection(text_embedding(input_ids[:, 3:4])) + codec_input[:, -1:]
                    talker_prompt = torch.cat([talker_prompt, first_text], dim=1)
                    trailing_text_hidden = torch.cat(
                        (
                            text_projection(text_embedding(input_ids[:, 4:-5])),
                            tts_eos_embed,
                        ),
                        dim=1,
                    )

        elif task_type == "CustomVoice":
            _speaker_raw = info_dict.get("speaker") or [""]
            speaker = (
                ((_speaker_raw[0] if isinstance(_speaker_raw, (list, tuple)) else _speaker_raw) or "").lower().strip()
            )
            if not speaker:
                raise ValueError("CustomVoice requires additional_information.speaker.")
            spk_id_map = {k.lower(): v for k, v in (getattr(talker_config, "spk_id", None) or {}).items()}
            if speaker not in spk_id_map:
                raise ValueError(f"Unsupported speaker: {speaker}")
            spk_id = spk_id_map[speaker]
            # Keep it at least 1D; embedding on a 0-d tensor can return 1D.
            spk_tensor = torch.tensor([spk_id], device=input_ids.device, dtype=torch.long)
            spk_embed = codec_embed(spk_tensor)
            if spk_embed.ndim == 1:
                spk_embed = spk_embed.view(1, 1, -1)
            elif spk_embed.ndim == 2:
                spk_embed = spk_embed.view(1, 1, -1)
            speaker_embed = spk_embed
            codec_input = torch.cat([codec_input_0, speaker_embed, codec_input_1], dim=1)

            role_embed = text_projection(text_embedding(input_ids[:, :3]))
            codec_prefix = torch.cat((tts_pad_embed.expand(-1, codec_input.shape[1] - 2, -1), tts_bos_embed), dim=1)
            codec_prefix = codec_prefix + codec_input[:, :-1]
            talker_prompt = torch.cat((role_embed, codec_prefix), dim=1)

            if non_streaming_mode:
                text_all = text_projection(text_embedding(input_ids[:, 3:-5]))
                text_all = torch.cat([text_all, tts_eos_embed], dim=1)
                pad_ids = torch.full(
                    (1, int(text_all.shape[1])),
                    int(talker_config.codec_pad_id),
                    device=input_ids.device,
                    dtype=torch.long,
                )
                talker_prompt = torch.cat(
                    [
                        talker_prompt,
                        text_all + codec_embed(pad_ids),
                        tts_pad_embed
                        + codec_embed(torch.tensor([[talker_config.codec_bos_id]], device=input_ids.device)),
                    ],
                    dim=1,
                )
                trailing_text_hidden = tts_pad_embed
            else:
                first_text = text_projection(text_embedding(input_ids[:, 3:4])) + codec_input[:, -1:]
                talker_prompt = torch.cat([talker_prompt, first_text], dim=1)
                trailing_text_hidden = torch.cat(
                    (
                        text_projection(text_embedding(input_ids[:, 4:-5])),
                        tts_eos_embed,
                    ),
                    dim=1,
                )

        elif task_type == "VoiceDesign":
            # No known speaker identity; only codec tags + text.
            codec_input = torch.cat([codec_input_0, codec_input_1], dim=1)

            role_embed = text_projection(text_embedding(input_ids[:, :3]))
            codec_prefix = torch.cat((tts_pad_embed.expand(-1, codec_input.shape[1] - 2, -1), tts_bos_embed), dim=1)
            codec_prefix = codec_prefix + codec_input[:, :-1]
            talker_prompt = torch.cat((role_embed, codec_prefix), dim=1)

            if non_streaming_mode:
                text_all = text_projection(text_embedding(input_ids[:, 3:-5]))
                text_all = torch.cat([text_all, tts_eos_embed], dim=1)
                pad_ids = torch.full(
                    (1, int(text_all.shape[1])),
                    int(talker_config.codec_pad_id),
                    device=input_ids.device,
                    dtype=torch.long,
                )
                talker_prompt = torch.cat(
                    [
                        talker_prompt,
                        text_all + codec_embed(pad_ids),
                        tts_pad_embed
                        + codec_embed(torch.tensor([[talker_config.codec_bos_id]], device=input_ids.device)),
                    ],
                    dim=1,
                )
                trailing_text_hidden = tts_pad_embed
            else:
                first_text = text_projection(text_embedding(input_ids[:, 3:4])) + codec_input[:, -1:]
                talker_prompt = torch.cat([talker_prompt, first_text], dim=1)
                trailing_text_hidden = torch.cat(
                    (
                        text_projection(text_embedding(input_ids[:, 4:-5])),
                        tts_eos_embed,
                    ),
                    dim=1,
                )
        else:
            raise ValueError(f"Unsupported task_type={task_type}")

        if instruct_embed is not None:
            talker_prompt = torch.cat([instruct_embed, talker_prompt], dim=1)

        return (
            talker_prompt.squeeze(0),  # [prompt_len, H]
            trailing_text_hidden.squeeze(0),  # [T, H]
            ref_code_len,
            ref_code_prompt.contiguous() if isinstance(ref_code_prompt, torch.Tensor) else None,
        )

    # -------------------- prompt-length estimator --------------------

    @staticmethod
    def estimate_prompt_len_from_additional_information(
        additional_information: dict[str, Any] | None,
        *,
        task_type: str,
        tokenize_prompt: Callable[[str], list[int]],
        codec_language_id: Mapping[str, int] | None,
        spk_is_dialect: Mapping[str, object] | None,
        estimate_ref_code_len: Callable[[object], int | None] | None = None,
    ) -> int:
        """Compute Stage-0 placeholder prompt length (length-only mirror of
        :meth:`build_prompt_embeds`).

        It must match the model-side ``inputs_embeds`` length to avoid extra
        padding and quality drop.
        """

        def _first(x: object, default: object) -> object:
            if isinstance(x, list):
                return x[0] if x else default
            return x if x is not None else default

        info: dict[str, Any] = additional_information or {}
        text = _first(info.get("text"), "")
        language = _first(info.get("language"), "Auto")
        speaker = _first(info.get("speaker"), "").lower().strip()
        instruct = _first(info.get("instruct"), "")
        non_streaming_mode_raw = _first(info.get("non_streaming_mode"), None)

        if isinstance(non_streaming_mode_raw, bool):
            non_streaming_mode = non_streaming_mode_raw
        else:
            # Official defaults: CustomVoice/VoiceDesign -> non_streaming_mode=True; Base -> False.
            non_streaming_mode = task_type in ("CustomVoice", "VoiceDesign")

        if not isinstance(text, str):
            text = ""
        if not isinstance(instruct, str):
            instruct = ""
        if not isinstance(language, str):
            language = "Auto"

        instruct_len = 0
        if instruct.strip():
            instruct_len = len(tokenize_prompt(build_instruct_text(instruct)))

        # ---- codec prefix portion (matches build_prompt_embeds) ----
        language_id = None
        if language.lower() != "auto" and codec_language_id:
            language_id = codec_language_id.get(language.lower())
        if (
            language_id is None
            and codec_language_id
            and spk_is_dialect
            and isinstance(language, str)
            and language.lower() in ("chinese", "auto")
            and isinstance(speaker, str)
            and speaker.strip()
        ):
            dialect = spk_is_dialect.get(speaker.lower())
            if isinstance(dialect, str) and dialect:
                language_id = codec_language_id.get(dialect)
        prefill_len = 3 if language_id is None else 4

        speaker_len = 1 if task_type in ("CustomVoice", "Base") else 0
        codec_input_len = prefill_len + speaker_len + 2  # + [codec_pad, codec_bos]
        codec_prefix_len = codec_input_len - 1  # codec_input[:-1] + tts_bos

        # Role header: input_ids[:, :3] in model.
        role_len = 3
        prompt_len = instruct_len + role_len + codec_prefix_len

        # ---- text conditioning portion (matches build_prompt_embeds) ----
        assistant_len = len(tokenize_prompt(build_assistant_text(text)))
        if assistant_len < 8:
            raise ValueError(f"Unexpected assistant prompt length: {assistant_len}")

        if task_type in ("CustomVoice", "VoiceDesign"):
            if non_streaming_mode:
                # model: full text ids (input_ids[:, 3:-5]) + eos + codec_bos step
                prompt_len += assistant_len - 6
            else:
                # model: only first text token in prefill
                prompt_len += 1

        if task_type == "Base":
            xvec_only = bool(_first(info.get("x_vector_only_mode"), False))
            in_context_mode = not xvec_only

            voice_clone_prompt = _first(info.get("voice_clone_prompt"), None)
            if isinstance(voice_clone_prompt, dict):
                icl_flag = _first(voice_clone_prompt.get("icl_mode"), None)
                if isinstance(icl_flag, bool):
                    in_context_mode = icl_flag

            if in_context_mode:
                ref_code = None
                if isinstance(voice_clone_prompt, dict):
                    # Do NOT apply `_first(...)` here. `_first` unwraps the singleton-list
                    # batching convention used at the top level of `additional_information`;
                    # values inside `voice_clone_prompt` are per-item payloads. `ref_code`
                    # is a `[num_frames, num_codebooks]` 2D list, so `_first` would strip
                    # its outer dim and report `num_codebooks` (~8) instead of `num_frames`
                    # (~hundreds), silently truncating the prefill embeddings downstream.
                    ref_code = voice_clone_prompt.get("ref_code")

                ref_code_len: int | None = None
                if isinstance(ref_code, list):
                    if ref_code and isinstance(ref_code[0], list):
                        ref_code_len = len(ref_code)
                    elif ref_code:
                        ref_code_len = len(ref_code)
                elif hasattr(ref_code, "shape"):
                    try:
                        shape = getattr(ref_code, "shape")
                        if shape and len(shape) >= 1:
                            ref_code_len = int(shape[0])
                    except Exception:
                        ref_code_len = None

                if ref_code_len is None:
                    ref_code_length = _first(info.get("ref_code_length"), None)
                    try:
                        if ref_code_length is not None:
                            ref_code_len = int(ref_code_length)  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        ref_code_len = None
                if ref_code_len is None and estimate_ref_code_len is not None:
                    ref_code_len = estimate_ref_code_len(info.get("ref_audio"))
                if ref_code_len is None:
                    raise ValueError(
                        "Base in-context voice cloning requires either `voice_clone_prompt.ref_code` "
                        "or a readable `ref_audio` that can be mapped to a codec frame length."
                    )

                codec_lens = 1 + int(ref_code_len)  # codec_bos + ref_code
                if non_streaming_mode:
                    # Mirrors _generate_icl_prompt(non_streaming_mode=True):
                    # text_embed = ref_ids + text_ids + eos.
                    ref_ids = _first(info.get("ref_ids"), None)
                    if isinstance(voice_clone_prompt, dict) and ref_ids is None:
                        # Same reasoning as ref_code above: values inside `voice_clone_prompt`
                        # are per-item payloads, not singleton-wrapped batches. Applying
                        # `_first` to a list-of-int `ref_ids` collapses it to an int and the
                        # branches below fall through to `ref_ids_len = 0`.
                        ref_ids = voice_clone_prompt.get("ref_ids") or voice_clone_prompt.get("ref_id")

                    if ref_ids is None:
                        ref_text = _first(info.get("ref_text"), "")
                        if (not isinstance(ref_text, str) or not ref_text.strip()) and isinstance(
                            voice_clone_prompt, dict
                        ):
                            ref_text = _first(voice_clone_prompt.get("ref_text"), "")
                        if not isinstance(ref_text, str) or not ref_text.strip():
                            raise ValueError(
                                "Base in-context non-streaming requires `ref_text` or tokenized `ref_ids`."
                            )
                        ref_ids_len = len(tokenize_prompt(build_ref_text(ref_text)))
                    elif hasattr(ref_ids, "shape"):
                        shape = getattr(ref_ids, "shape", None)
                        ref_ids_len = int(shape[-1]) if shape else 0
                    elif isinstance(ref_ids, list):
                        ref_ids_len = len(ref_ids)
                    else:
                        ref_ids_len = 0

                    # model uses ref_ids[:, 3:-2] (strip 5 tokens) and text_id=input_ids[:, 3:-5] (strip 8).
                    ref_id_len = max(0, int(ref_ids_len) - 5)
                    text_id_len = max(0, int(assistant_len) - 8)
                    text_embed_len = ref_id_len + text_id_len + 1  # + eos
                    prompt_len += text_embed_len + codec_lens
                else:
                    # Mirrors _generate_icl_prompt(non_streaming_mode=False):
                    # aligned to codec_lens.
                    prompt_len += codec_lens
            else:
                # Base without ICL behaves like CustomVoice.
                if non_streaming_mode:
                    prompt_len += assistant_len - 6
                else:
                    prompt_len += 1

        return max(2, int(prompt_len))
