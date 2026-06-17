# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared forced aligner for streaming TTS word timestamps.

Hosts one lazy-loaded, in-process pooling ``vllm.LLM`` shared by the TTS
frontend. ``llm.encode`` is sync/blocking, so :func:`align` runs it in
``asyncio.to_thread``, serialized on a lock since the offline LLM is not
thread-safe. See :func:`align` for the return-value contract.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from vllm_omni.utils import qwen3_force_align_processor as _processor

logger = logging.getLogger(__name__)


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "deploy" / "qwen3_tts_forced_aligner.yaml"


@dataclass(frozen=True, slots=True)
class WordTimestamp:
    """Internal alignment record. Serialized to a plain JSON object
    (``{"word", "start_ms", "end_ms"}``) at the WebSocket boundary.
    """

    word: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True, slots=True)
class ForcedAlignerConfig:
    """Plain-data config for constructing the aligner ``vllm.LLM`` (lazy-loaded)."""

    model: str
    runner: str | None = None
    architecture: str | None = None
    pooling_task: str | None = None
    gpu_memory_utilization: float | None = None
    dtype: str | None = None
    max_model_len: int | None = None
    trust_remote_code: bool | None = None
    extra_llm_kwargs: dict[str, Any] = field(default_factory=dict)


def _load_forced_aligner_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return dict(raw.get("forced_aligner", raw))


def build_forced_aligner_config(args: Any) -> ForcedAlignerConfig | None:
    """Build a config from CLI args, or ``None`` when the feature is off.

    Precedence (lowest to highest): the packaged default YAML
    (:data:`_DEFAULT_CONFIG_PATH`, Qwen deploy defaults) -> a user YAML passed
    via ``--forced-aligner-config`` -> the ``--forced-aligner`` model path. The
    feature is off (returns ``None``) unless a model resolves from this chain.
    Per-field overrides such as ``gpu_memory_utilization`` live in the YAML.
    """
    config_path = getattr(args, "forced_aligner_config", None)
    config_data: dict[str, Any] = {}
    model = getattr(args, "forced_aligner", None)
    if config_path or model:
        config_data.update(_load_forced_aligner_yaml(_DEFAULT_CONFIG_PATH))
    if config_path:
        config_data.update(_load_forced_aligner_yaml(config_path))
    if model:
        config_data["model"] = str(model)
    model = config_data.get("model")
    if not model:
        return None
    allowed = set(ForcedAlignerConfig.__dataclass_fields__)
    return ForcedAlignerConfig(**{k: v for k, v in config_data.items() if k in allowed})


class ForcedAlignerLoadError(RuntimeError):
    """The aligner model/config could not be loaded (permanent until restart).

    Distinguished from a per-request alignment failure so the streaming layer
    can report a clear reason once rather than degrading every request to
    ``timestamps: null``.
    """


# --- Singleton state ---
# A single LLM serves the whole API server. ``_lock`` guards the lazy
# constructor; ``_encode_lock`` serializes ``encode`` because the offline
# ``vllm.LLM`` is a sync batch interface that is not safe to call from multiple
# threads at once (concurrent WebSocket sessions otherwise race the engine).
_lock = threading.Lock()
_encode_lock = asyncio.Lock()
_llm: Any = None
_classify_num: int | None = None
_timestamp_token_id: int | None = None
_timestamp_segment_time_ms: float | None = None
_loaded_config: ForcedAlignerConfig | None = None


async def align(
    *,
    audio: bytes,
    text: str,
    sample_rate: int,
    config: ForcedAlignerConfig,
    language: str | None = None,
) -> list[WordTimestamp] | None:
    """Run one forced-alignment pass.

    Args:
        audio: Signed-int16 little-endian mono PCM bytes.
        text: Ground-truth text whose words to align.
        sample_rate: Sample rate of ``audio`` in Hz.
        config: Aligner config (same instance for every call across the
            server's lifetime; reload requires a server restart).
        language: Optional language hint for word segmentation. ``None`` /
            ``"auto"`` use the space + Chinese-mixed path; ``"japanese"`` /
            ``"korean"`` (or their codes) need ``qwen_asr`` installed for
            faithful tokenisation, else they degrade to whitespace splitting.

    Returns:
        List of :class:`WordTimestamp` on success (possibly empty for
        silence / no aligned tokens), ``None`` on a per-request alignment
        failure (decode error).

    Raises:
        ForcedAlignerLoadError: the model/config could not be loaded. This is
            permanent until restart, so callers surface it once instead of
            degrading every request to ``None``.
    """
    async with _encode_lock:
        try:
            await asyncio.to_thread(_ensure_loaded, config)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Forced aligner failed to load")
            raise ForcedAlignerLoadError(str(exc)) from exc
        try:
            return await asyncio.to_thread(_align_sync, audio, text, sample_rate, config, language)
        except Exception:  # noqa: BLE001
            logger.exception("Forced aligner failed for text=%r", text)
            return None


def _align_sync(
    audio: bytes,
    text: str,
    sample_rate: int,
    config: ForcedAlignerConfig,
    language: str | None = None,
) -> list[WordTimestamp]:
    # Precondition: align() has already run _ensure_loaded(config) under
    # _encode_lock and owns the load-failure -> ForcedAlignerLoadError mapping.
    # Don't reload here, so a load error can't be swallowed as a per-request None.
    audio_arr = _pcm_bytes_to_float32(audio)
    if audio_arr.size == 0:
        return []
    audio_duration_ms = (audio_arr.size / sample_rate) * 1000.0

    # Segment once and reuse for both the prompt and the decode: the word
    # units MUST match between the two or the markers drift out of sync.
    words = _processor.segment_words(text, language)
    prompt = _processor.build_prompt(words)
    request = {
        "prompt": prompt,
        "multi_modal_data": {"audio": (audio_arr, sample_rate)},
    }

    from vllm.pooling_params import PoolingParams

    outputs = _llm.encode(  # type: ignore[union-attr]
        [request],
        pooling_params=PoolingParams(),
        pooling_task=config.pooling_task or "token_classify",
        use_tqdm=False,
    )
    if not outputs:
        return []

    result = outputs[0]
    logits = result.outputs.data  # [n_token, classify_num]
    prompt_token_ids = list(result.prompt_token_ids)
    timestamp_positions = [i for i, tid in enumerate(prompt_token_ids) if tid == _timestamp_token_id]
    if not timestamp_positions:
        logger.warning(
            "No <|timestamp|> tokens found in prompt for text=%r; aligner returned %d rows.",
            text,
            logits.shape[0] if hasattr(logits, "shape") else len(logits),
        )
        return []

    return _decode_timestamps(
        logits=logits,
        words=words,
        timestamp_positions=timestamp_positions,
        classify_num=_classify_num,
        timestamp_segment_time_ms=_timestamp_segment_time_ms,
        audio_duration_ms=audio_duration_ms,
    )


def _ensure_loaded(config: ForcedAlignerConfig) -> None:
    """Lazy-load the singleton ``vllm.LLM`` under lock; idempotent."""
    global _llm, _classify_num, _timestamp_token_id, _timestamp_segment_time_ms, _loaded_config

    if _llm is not None:
        if _loaded_config is not None and _loaded_config.model != config.model:
            # Multiple configs from different requests — refuse rather
            # than swap models silently. A server restart is required.
            raise RuntimeError(
                f"Forced aligner already loaded with config={_loaded_config!r}; "
                f"cannot serve a request that asks for {config!r}. "
                "Restart the server to change the aligner config."
            )
        return

    with _lock:
        if _llm is not None:
            return  # raced; another caller did the load

        # Lazy import: vllm pulls torch + CUDA, which we want to avoid at
        # module import time.
        from vllm import LLM

        logger.info(
            "Loading forced aligner %s (gpu_memory_utilization=%s)",
            config.model,
            config.gpu_memory_utilization if config.gpu_memory_utilization is not None else "default",
        )
        llm_kwargs: dict[str, Any] = {"model": config.model}
        if config.runner is not None:
            llm_kwargs["runner"] = config.runner
        if config.architecture is not None:
            llm_kwargs["hf_overrides"] = {"architectures": [config.architecture]}
        if config.gpu_memory_utilization is not None:
            llm_kwargs["gpu_memory_utilization"] = config.gpu_memory_utilization
        if config.trust_remote_code is not None:
            llm_kwargs["trust_remote_code"] = config.trust_remote_code
        if config.dtype is not None:
            llm_kwargs["dtype"] = config.dtype
        if config.max_model_len is not None:
            llm_kwargs["max_model_len"] = config.max_model_len
        llm_kwargs.update(config.extra_llm_kwargs)

        # Force the aligner engine in-process: in multiprocessing mode the
        # token_classify ``pooling_output`` fails to round-trip over the
        # engine-core IPC on current vLLM. Hard-set (not setdefault) so a
        # globally exported ``=1`` can't push it into a subprocess; save/restore
        # to avoid leaking the override to anything built later in this process.
        prev_mp = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        try:
            llm = LLM(**llm_kwargs)
        finally:
            if prev_mp is None:
                os.environ.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
            else:
                os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = prev_mp

        thinker_config = getattr(llm.llm_engine.model_config.hf_config, "thinker_config", None)
        if thinker_config is None or not hasattr(thinker_config, "classify_num"):
            raise RuntimeError(
                "Loaded aligner has no thinker_config.classify_num; "
                "expected a Qwen3ASRForcedAlignerForTokenClassification checkpoint."
            )
        timestamp_segment_time_ms = getattr(llm.llm_engine.model_config.hf_config, "timestamp_segment_time", None)
        if timestamp_segment_time_ms is None:
            raise RuntimeError(
                "Loaded aligner has no timestamp_segment_time; expected a Qwen3ASR forced aligner checkpoint."
            )

        tokenizer = llm.get_tokenizer()
        timestamp_token_id = _processor.resolve_timestamp_token_id(tokenizer)

        # Publish in this order so a concurrent reader either sees
        # _llm == None (will block on the lock) or sees a fully
        # initialized aligner.
        _classify_num = int(thinker_config.classify_num)
        _timestamp_token_id = timestamp_token_id
        _timestamp_segment_time_ms = float(timestamp_segment_time_ms)
        _loaded_config = config
        _llm = llm

        logger.info(
            "Forced aligner ready: timestamp_token_id=%d, classify_num=%d, timestamp_segment_time_ms=%.1f",
            timestamp_token_id,
            _classify_num,
            _timestamp_segment_time_ms,
        )


# --- pure helpers (testable without a GPU / vllm) ---
#
# Word segmentation, prompt building, timestamp repair and marker-token
# lookup are Qwen-specific and live in
# :mod:`vllm_omni.utils.qwen3_force_align_processor`. This module owns only
# the generic vLLM orchestration.


def _pcm_bytes_to_float32(audio: bytes) -> np.ndarray:
    """Decode signed-int16 mono PCM bytes into a [-1, 1] float32 array."""
    if not audio:
        return np.zeros(0, dtype=np.float32)
    if len(audio) % 2 != 0:
        # Drop a trailing odd byte rather than raise; keeps streaming
        # robust against off-by-one chunk boundaries.
        audio = audio[:-1]
    pcm = np.frombuffer(audio, dtype=np.int16)
    return (pcm.astype(np.float32) / 32768.0).copy()


def _decode_timestamps(
    *,
    logits: Any,
    words: list[str],
    timestamp_positions: list[int],
    classify_num: int,
    audio_duration_ms: float,
    timestamp_segment_time_ms: float | None = None,
) -> list[WordTimestamp]:
    """Translate ``[n_token, classify_num]`` logits into word timestamps.

    ``words`` must be the exact segmentation used to build the prompt (see
    :func:`qwen3_force_align_processor.segment_words`); each word owns two
    consecutive markers.
    """
    arr = logits.detach().cpu().numpy() if hasattr(logits, "detach") else np.asarray(logits)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D logits [n_token, classify_num]; got shape {arr.shape}")
    if arr.shape[1] != classify_num:
        raise ValueError(
            f"Logits last dim {arr.shape[1]} != classify_num {classify_num}; "
            "model config and prompt template may be out of sync."
        )

    expected = len(words) * 2
    if len(timestamp_positions) != expected:
        logger.warning(
            "Got %d timestamp positions but text has %d words (expected %d start/end markers); "
            "returning empty alignment.",
            len(timestamp_positions),
            len(words),
            expected,
        )
        return []

    marker_logits = arr[timestamp_positions, :]
    bin_idx = marker_logits.argmax(axis=-1)
    bin_size_ms = (
        float(timestamp_segment_time_ms)
        if timestamp_segment_time_ms is not None
        else (audio_duration_ms / classify_num if classify_num > 0 else 0.0)
    )

    # Repair non-monotonic bins across the whole start/end sequence before
    # pairing, so each (start, end) pair stays ordered and one bad bin can't
    # corrupt its neighbours.
    marker_ms = _processor.fix_timestamp([int(round(int(b) * bin_size_ms)) for b in bin_idx])

    # Pair markers into per-word intervals, enforcing the wire contract here so
    # callers don't have to: monotonic and non-overlapping across words, with
    # start <= end and nothing past the audio length.
    duration_ms = int(round(audio_duration_ms))
    out: list[WordTimestamp] = []
    prev_end_ms = 0
    for i, word in enumerate(words):
        start_ms = min(max(marker_ms[i * 2], prev_end_ms), duration_ms)
        end_ms = min(max(marker_ms[i * 2 + 1], start_ms), duration_ms)
        out.append(WordTimestamp(word=word, start_ms=start_ms, end_ms=end_ms))
        prev_end_ms = end_ms
    return out


# Test hooks ---------------------------------------------------------------
# Tests need a way to reset module state without restarting Python. Not
# part of the public API; do not call in production code.


def _reset_for_tests() -> None:
    """Drop the cached aligner state so the next call reloads."""
    global _llm, _classify_num, _timestamp_token_id, _timestamp_segment_time_ms, _loaded_config
    with _lock:
        _llm = None
        _classify_num = None
        _timestamp_token_id = None
        _timestamp_segment_time_ms = None
        _loaded_config = None
    _processor._reset_for_tests()
