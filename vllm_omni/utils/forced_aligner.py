# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared forced aligner for streaming TTS word timestamps.

Hosts one in-process ``vllm.LLM(runner="pooling")`` (upstream's
``Qwen3ASRForcedAlignerForTokenClassification``), shared by the whole TTS
frontend and lazy-loaded on first use. ``llm.encode`` is sync/blocking, so
:func:`align` wraps it in ``asyncio.to_thread`` to keep the event loop free.

Public API:
* :func:`build_forced_aligner_config` — CLI args -> ``ForcedAlignerConfig``
  (``None`` means the feature is off).
* :func:`align` — returns ``list[WordTimestamp]`` on success, ``[]`` for
  silence / no aligned tokens, ``None`` on failure. It never raises (any
  load or decode error is caught and returned as ``None``); the streaming
  layer maps ``None`` to JSON ``timestamps: null`` and always keeps audio
  flowing. ``None`` vs ``[]`` lets clients tell "failed" from "no speech".
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
    """Plain-data config built from CLI args at server startup.

    Captures only the fields needed to construct the ``vllm.LLM``
    instance; the LLM itself is lazy-loaded inside :func:`align`.
    Override defaults by setting the matching CLI flag (currently only
    ``--forced-aligner``; the rest follow conservative defaults that
    are enough for Qwen/Qwen3-ForcedAligner-0.6B on a 24GB GPU).
    """

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
    """Build a config from CLI args, or ``None`` when the flag is off.

    Mirrors @Dmaner's #3804 helper of the same name so test fixtures
    that target either implementation port unchanged.
    """
    config_path = getattr(args, "forced_aligner_config", None)
    config_data: dict[str, Any] = {}
    model = getattr(args, "forced_aligner", None)
    if config_path or model:
        # The default YAML owns Qwen-specific deployment defaults. A user YAML
        # can override any subset, and CLI flags override both.
        config_data.update(_load_forced_aligner_yaml(_DEFAULT_CONFIG_PATH))
    if config_path:
        config_data.update(_load_forced_aligner_yaml(config_path))
    if model:
        config_data["model"] = str(model)
    gpu_mem = getattr(args, "forced_aligner_gpu_memory_utilization", None)
    if gpu_mem is not None:
        config_data["gpu_memory_utilization"] = float(gpu_mem)
    model = config_data.get("model")
    if not model:
        return None
    allowed = set(ForcedAlignerConfig.__dataclass_fields__)
    return ForcedAlignerConfig(**{k: v for k, v in config_data.items() if k in allowed})


# --- Singleton state ---
# A single LLM serves the whole API server. The lock guards the lazy
# constructor; once `_llm` is set, callers can read it lock-free.
_lock = threading.Lock()
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
        silence / no aligned tokens), ``None`` if alignment failed.
    """
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
    _ensure_loaded(config)
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

    # Lazy import so ``vllm.pooling_params`` doesn't hit the parent
    # process until alignment is actually invoked.
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
        # module import time. The aligner runs in-process, sharing the API
        # server's visible device with the TTS stages.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
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

        llm = LLM(**llm_kwargs)

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
# :mod:`vllm_omni.utils.qwen3_force_align_processor` (the model-agnostic seam
# the issue asks for). This module owns only the generic vLLM orchestration.


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
    # pairing, mirroring the official decoder. This keeps each (start, end)
    # pair ordered and stops one bad bin from corrupting neighbours.
    marker_ms = _processor.fix_timestamp([int(round(int(b) * bin_size_ms)) for b in bin_idx])

    out: list[WordTimestamp] = []
    for i, word in enumerate(words):
        start_ms = marker_ms[i * 2]
        end_ms = max(marker_ms[i * 2 + 1], start_ms)
        out.append(WordTimestamp(word=word, start_ms=start_ms, end_ms=end_ms))
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
