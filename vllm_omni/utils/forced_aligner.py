# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared forced aligner utility for streaming TTS word timestamps (issue #3631).

Hosts a single in-process ``vllm.LLM(runner="pooling")`` running upstream's
:class:`vllm.model_executor.models.qwen3_asr_forced_aligner.\
Qwen3ASRForcedAlignerForTokenClassification`. The whole TTS frontend
shares one instance — the aligner is always the slowest path in the
audio request, so a single GPU-resident model is enough.

Public API
----------
* :func:`build_forced_aligner_config` — projects CLI args into a
  ``ForcedAlignerConfig | None``. ``None`` means "feature off".
* :func:`align` — async wrapper around ``llm.encode``; lazy-loads the
  underlying ``vllm.LLM`` on first call. Returns ``list[WordTimestamp]``
  on success, ``[]`` for silence/no aligned tokens, ``None`` when
  alignment failed (the streaming layer maps this to JSON
  ``timestamps: null`` and keeps audio flowing).

Why a single shared utility, not a subprocess
---------------------------------------------
* The model card says ``LLM(runner="pooling")`` is the canonical
  interface; we just consume it.
* ``llm.encode`` is sync + blocking. We wrap it in ``asyncio.to_thread``
  so the event loop stays responsive without spawning a process.
* PR-2 (later, optional) can move the aligner into the vllm-omni stage
  pipeline; the public surface here stays the same.

Failure semantics
-----------------
* On startup failure (model not found, OOM): the first call to
  :func:`align` raises; the streaming layer catches and degrades to
  ``timestamps: null`` for that request, then disables alignment for
  the rest of it. Subsequent requests retry from scratch.
* On per-request failure (decoding error, model spit out empty result):
  ``align`` returns ``None`` (failure) or ``[]`` (silence). The two are
  intentionally distinguishable so clients can tell "no speech" from
  "alignment failed".
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

logger = logging.getLogger(__name__)


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "deploy" / "qwen3_tts_forced_aligner.yaml"

# Prompt tokens for the Qwen3 forced aligner. These are baked in rather than
# configurable because the surrounding prompt template (``<|im_start|>`` chat
# wrapping in ``_build_prompt``) and the ``_tokenize_text`` word splitter are
# already Qwen-specific: a different aligner family would need code changes
# here regardless, so exposing just these two strings as config is misleading.
_AUDIO_PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>"
_TIMESTAMP_TOKEN = "<timestamp>"


@dataclass(frozen=True, slots=True)
class WordTimestamp:
    """Internal alignment record. Converted to the pydantic
    :class:`vllm_omni.entrypoints.openai.protocol.audio.WordTimestamp`
    at the HTTP/WebSocket boundary.
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
    # Physical GPU index (as a string, e.g. "7") to pin the aligner LLM to.
    # None shares the server's default visible device (cuda:0).
    device: str | None = None
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
    device = getattr(args, "forced_aligner_device", None)
    if device is not None:
        config_data["device"] = str(device)
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
) -> list[WordTimestamp] | None:
    """Run one forced-alignment pass.

    Args:
        audio: Signed-int16 little-endian mono PCM bytes.
        text: Ground-truth text whose words to align.
        sample_rate: Sample rate of ``audio`` in Hz.
        config: Aligner config (same instance for every call across the
            server's lifetime; reload requires a server restart).

    Returns:
        List of :class:`WordTimestamp` on success (possibly empty for
        silence / no aligned tokens), ``None`` if alignment failed.
    """
    try:
        return await asyncio.to_thread(_align_sync, audio, text, sample_rate, config)
    except Exception:  # noqa: BLE001
        logger.exception("Forced aligner failed for text=%r", text)
        return None


def _align_sync(
    audio: bytes,
    text: str,
    sample_rate: int,
    config: ForcedAlignerConfig,
) -> list[WordTimestamp]:
    _ensure_loaded(config)
    audio_arr = _pcm_bytes_to_float32(audio)
    if audio_arr.size == 0:
        return []
    audio_duration_ms = (audio_arr.size / sample_rate) * 1000.0

    prompt = _build_prompt(text)
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
        text=text,
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

        # Lazy import: vllm pulls torch + CUDA, which we want to avoid
        # at module import time.
        #
        # Multiprocessing mode depends on whether we pin a GPU:
        #   * No device pin: run the engine in-process (lighter, no extra
        #     subprocess) sharing the API server's default visible device.
        #   * Device pin: force a spawned engine subprocess. The API server
        #     process has already initialized its own CUDA context, so an
        #     in-process LLM can no longer be redirected to another card.
        #     A fresh subprocess reads CUDA_VISIBLE_DEVICES at startup and
        #     binds cleanly to the requested GPU.
        if config.device is not None:
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
        else:
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

        # Pin the aligner to a specific GPU by scoping CUDA_VISIBLE_DEVICES
        # around construction. The engine subprocess (forced above) reads this
        # env var once at startup and binds its CUDA context to the requested
        # card. ``torch.cuda.set_device`` does not work for this: vLLM
        # initializes the device itself and ignores it.
        prev_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if config.device is not None:
            pinned = config.device.strip().removeprefix("cuda:")
            os.environ["CUDA_VISIBLE_DEVICES"] = pinned
            logger.info("Pinning forced aligner to CUDA_VISIBLE_DEVICES=%s", pinned)
        try:
            llm = LLM(**llm_kwargs)
        finally:
            if config.device is not None:
                if prev_visible is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = prev_visible

        thinker_config = getattr(llm.llm_engine.model_config.hf_config, "thinker_config", None)
        if thinker_config is None or not hasattr(thinker_config, "classify_num"):
            raise RuntimeError(
                "Loaded aligner has no thinker_config.classify_num; "
                "expected a Qwen3ASRForcedAlignerForTokenClassification checkpoint."
            )
        timestamp_segment_time_ms = getattr(llm.llm_engine.model_config.hf_config, "timestamp_segment_time", None)
        if timestamp_segment_time_ms is None:
            raise RuntimeError(
                "Loaded aligner has no timestamp_segment_time; "
                "expected a Qwen3ASR forced aligner checkpoint."
            )

        tokenizer = llm.get_tokenizer()
        timestamp_token_id = _resolve_timestamp_token_id(tokenizer)

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


def _build_prompt(
    text: str,
    *,
    audio_placeholder: str = _AUDIO_PLACEHOLDER,
    timestamp_token: str = _TIMESTAMP_TOKEN,
) -> str:
    """Construct the prompt with start/end timestamp slots per word."""
    words = _tokenize_text(text)
    if not words:
        # Pad with one timestamp so the decoder always has something to
        # read; an empty result still surfaces as "[]" upstream.
        body = timestamp_token
    else:
        body = f"{timestamp_token}{timestamp_token}".join(words) + f"{timestamp_token}{timestamp_token}"
    return f"<|im_start|>user\n{audio_placeholder}{body}<|im_end|>\n<|im_start|>assistant\n"


def _tokenize_text(text: str) -> list[str]:
    """Approximate Qwen3ForceAlignProcessor tokenization for common languages."""
    words: list[str] = []
    current: list[str] = []

    def flush_current() -> None:
        if current:
            words.append("".join(current))
            current.clear()

    for ch in text:
        code = ord(ch)
        is_cjk = (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0x20000 <= code <= 0x2A6DF
            or 0x2A700 <= code <= 0x2B73F
            or 0x2B740 <= code <= 0x2B81F
            or 0x2B820 <= code <= 0x2CEAF
            or 0xF900 <= code <= 0xFAFF
        )
        if is_cjk:
            flush_current()
            words.append(ch)
        elif ch == "'" or ch.isalnum():
            current.append(ch)
        else:
            flush_current()

    flush_current()
    return words


def _resolve_timestamp_token_id(tokenizer: Any, timestamp_token: str | None = None) -> int:
    """Look up the integer id of the timestamp special token."""
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        raise RuntimeError("Aligner tokenizer has no convert_tokens_to_ids method.")
    tid = convert(timestamp_token)
    if isinstance(tid, list):
        tid = tid[0] if tid else None
    if tid is None or (isinstance(tid, int) and tid < 0):
        raise RuntimeError(
            f"Aligner tokenizer does not recognise {timestamp_token!r} (got id={tid}). "
            "Check the model card; the marker token may use a different name."
        )
    return int(tid)


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
    text: str,
    timestamp_positions: list[int],
    classify_num: int,
    audio_duration_ms: float,
    timestamp_segment_time_ms: float | None = None,
) -> list[WordTimestamp]:
    """Translate ``[n_token, classify_num]`` logits into word timestamps."""
    arr = logits.detach().cpu().numpy() if hasattr(logits, "detach") else np.asarray(logits)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D logits [n_token, classify_num]; got shape {arr.shape}")
    if arr.shape[1] != classify_num:
        raise ValueError(
            f"Logits last dim {arr.shape[1]} != classify_num {classify_num}; "
            "model config and prompt template may be out of sync."
        )

    words = _tokenize_text(text)
    expected = len(words) * 2
    if len(timestamp_positions) != expected:
        logger.warning(
            "Got %d timestamp positions but text has %d words (expected %d start/end markers); returning empty alignment.",
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

    out: list[WordTimestamp] = []
    for i, word in enumerate(words):
        start_bin = int(bin_idx[i * 2])
        end_bin = int(bin_idx[i * 2 + 1])
        if end_bin < start_bin:
            # Pathological output; skip this word rather than crash.
            continue
        out.append(
            WordTimestamp(
                word=word,
                start_ms=int(round(start_bin * bin_size_ms)),
                end_ms=int(round(end_bin * bin_size_ms)),
            )
        )
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
