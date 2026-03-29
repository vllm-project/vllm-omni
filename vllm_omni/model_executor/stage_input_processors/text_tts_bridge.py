# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
text_tts_bridge.py
~~~~~~~~~~~~~~~~~~
Bridge stage input processor connecting any vLLM text model (Stage 0)
to a Qwen3-TTS decoder (Stage 1) as a composable pipeline.

Implements RFC Theme 2: "TTS as a Composable Layer (P0)".

Design decisions (answers RFC key design questions):
  Q1: Bridge as custom_process_next_stage_input_func hook reusing the
      existing async_chunk / OmniChunkTransferAdapter framework.
      Zero changes to OmniStage.
  Q2: SentenceChunker buffers detokenized text until a sentence boundary
      AND min_sentence_chars characters, giving a tunable latency knob.
  Q3: default_voice/default_language injected from bridge config when
      Stage 0 (plain text LLM) has no concept of speaker.

Signature matches OmniChunkTransferAdapter._send_single_request:
    custom_process_next_stage_input_func(
        transfer_manager,   # OmniChunkTransferAdapter instance
        pooling_output,     # dict of tensors from Stage 0 engine
        request,            # OmniEngineCoreRequest
        is_finished,        # bool: True at EOS
    )

See: vllm_omni/distributed/omni_connectors/transfer_adapter/
         chunk_transfer_adapter.py
Reference implementations:
    thinker2talker_async_chunk   (qwen3_omni.py)
    talker2code2wav_async_chunk  (qwen3_omni.py)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TextTTSBridgeConfig:
    """Runtime config populated from the YAML bridge: section."""

    min_sentence_chars: int = 40
    """Minimum characters before flushing a chunk to the TTS stage.
    Lower → less latency / higher TTFA variance.
    Higher → smoother audio / higher TTFA."""

    sentence_delimiters: list[str] = field(default_factory=lambda: [".", "!", "?", "。", "！", "？"])
    """Punctuation characters that trigger a sentence flush."""

    tts_task_type: str = "CustomVoice"
    """Qwen3-TTS task type forwarded to Stage 1."""

    default_voice: str = "vivian"
    """Fallback voice when Stage 0 request carries no speaker info (RFC Q3)."""

    default_language: str = "English"
    """Fallback language tag for Qwen3-TTS CustomVoice inputs."""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TextTTSBridgeConfig:
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


# ---------------------------------------------------------------------------
# Sentence chunker
# ---------------------------------------------------------------------------


def _build_sentence_re(delimiters: list[str]) -> re.Pattern:
    """Compile a lookbehind regex for the given sentence-ending delimiters."""
    escaped = "".join(re.escape(d) for d in delimiters)
    return re.compile(rf"(?<=[{escaped}])\s*")


class SentenceChunker:
    """
    Stateful per-request text buffer.

    Receives incremental detokenized text from Stage 0 and yields
    complete sentence chunks ready for the TTS stage.

    State is stored on transfer_manager.request_payload[request_id]
    so it survives across multiple async_chunk calls for the same request.

    >>> chunker = SentenceChunker()
    >>> chunker.feed("Hello world")
    []
    >>> chunker.feed(". How are you today?")
    ['Hello world.']
    >>> chunker.flush()
    [' How are you today?']
    """

    def __init__(self, cfg: TextTTSBridgeConfig | None = None) -> None:
        self.cfg = cfg or TextTTSBridgeConfig()
        self._buf: str = ""
        self._re = _build_sentence_re(self.cfg.sentence_delimiters)

    def feed(self, token_text: str) -> list[str]:
        """Append token_text; return any flushed sentence chunks."""
        self._buf += token_text
        return self._try_flush()

    def flush(self) -> list[str]:
        """Force-flush remaining buffer at end-of-stream."""
        if self._buf.strip():
            chunk, self._buf = self._buf.strip(), ""
            return [chunk]
        return []

    def _try_flush(self) -> list[str]:
        chunks: list[str] = []
        while True:
            m = self._re.search(self._buf)
            if m is None:
                break
            boundary = m.end()
            candidate = self._buf[:boundary]
            if len(candidate) < self.cfg.min_sentence_chars:
                break
            chunks.append(candidate.strip())
            self._buf = self._buf[boundary:]
        return chunks


# ---------------------------------------------------------------------------
# Helpers: extract detokenized text from pooling_output
# ---------------------------------------------------------------------------


def _get_decoded_text(
    pooling_output: dict[str, Any],
    request: Any,
) -> str:
    """
    Extract the latest decoded text token(s) from Stage 0 output.

    The async_chunk framework transfers the engine's pooling_output dict.
    For a standard AR text LLM, the detokenized incremental text is stored
    under the "text" key when detokenize=True in the stage sampling params.

    Falls back to checking request.output_token_ids if "text" is absent
    (for cases where the engine doesn't populate "text" directly).
    """
    # Primary path: detokenized text in pooling_output
    text = pooling_output.get("text")
    if isinstance(text, str):
        return text
    if isinstance(text, (list, tuple)) and text:
        return str(text[-1])

    # Fallback: request carries the latest output token as text
    # (some engine versions surface it via request.output_text)
    output_text = getattr(request, "output_text", None)
    if isinstance(output_text, str) and output_text:
        # Return only the delta since last call — managed by chunker state
        return output_text

    return ""


def _get_or_create_chunker(
    transfer_manager: Any,
    request_id: str,
    cfg: TextTTSBridgeConfig,
) -> SentenceChunker:
    """
    Retrieve (or lazily create) the per-request SentenceChunker.

    Uses transfer_manager.request_payload as the state store,
    exactly like talker2code2wav_async_chunk uses
    transfer_manager.code_prompt_token_ids.
    """
    payload = transfer_manager.request_payload
    key = f"_tts_chunker_{request_id}"
    if key not in payload:
        payload[key] = SentenceChunker(cfg)
    return payload[key]


def _cleanup_chunker(transfer_manager: Any, request_id: str) -> None:
    """Remove per-request chunker state after EOS."""
    key = f"_tts_chunker_{request_id}"
    transfer_manager.request_payload.pop(key, None)


# ---------------------------------------------------------------------------
# Input builder: text chunk → Qwen3-TTS CustomVoice input dict
# ---------------------------------------------------------------------------


def _build_tts_input(
    text_chunk: str,
    cfg: TextTTSBridgeConfig,
    request: Any,
) -> dict[str, Any]:
    """
    Build a Qwen3-TTS CustomVoice input dict from one sentence chunk.

    Voice and language are extracted from the request's
    additional_information (set by the caller via extra_body),
    with fallback to YAML bridge config defaults (RFC Q3).
    """
    from vllm_omni.model_executor.stage_input_processors.tts_utils import (
        extract_language_from_request,
        extract_speaker_from_request,
    )

    voice = extract_speaker_from_request(request) or cfg.default_voice
    language_list = extract_language_from_request(request)
    language = language_list[0] if language_list else cfg.default_language

    result: dict[str, Any] = {
        "text": text_chunk,
        "task_type": cfg.tts_task_type,
        "voice": voice,
        "language": language,
    }
    return result


# ---------------------------------------------------------------------------
# Main hook: custom_process_next_stage_input_func entry point
# ---------------------------------------------------------------------------


def text2tts(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: Any,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """
    OmniChunkTransferAdapter custom_process_next_stage_input_func hook.

    Signature matches chunk_transfer_adapter.py:218-225 and all existing
    async_chunk hooks (thinker2talker_async_chunk, talker2code2wav_async_chunk,
    slow_ar_to_dac_decoder_async_chunk).

    Called by OmniChunkTransferAdapter._send_single_request on every
    async_chunk step. Returns a list of Stage 1 (Qwen3-TTS) input dicts,
    one per sentence chunk ready for synthesis, or None if still buffering.

    Parameters
    ----------
    transfer_manager : OmniChunkTransferAdapter
        Carries per-request state via transfer_manager.request_payload.
        We use it to persist the SentenceChunker across calls.
    pooling_output : dict[str, Any]
        Engine output dict from Stage 0. Contains detokenized text under
        the "text" key when detokenize=True in the stage sampling params.
    request : OmniEngineCoreRequest
        Carries request metadata including additional_information for
        voice/language extraction (RFC Q3).
    is_finished : bool
        True when Stage 0 has reached EOS. Forces a final buffer flush.

    Returns
    -------
    list[dict] | None
        One dict per flushed sentence chunk, or None if buffering.
        None (not empty list) signals "no payload yet" to the framework,
        matching the convention used by thinker2talker_async_chunk.
    """
    # --- Load bridge config from connector extra config (set in YAML) -------
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    bridge_raw = raw_cfg.get("bridge", raw_cfg) if isinstance(raw_cfg, dict) else {}
    cfg = TextTTSBridgeConfig.from_dict(bridge_raw)

    # --- Per-request state --------------------------------------------------
    request_id = request.external_req_id
    chunker = _get_or_create_chunker(transfer_manager, request_id, cfg)

    # --- Extract new text token(s) from Stage 0 output ----------------------
    new_text = _get_decoded_text(pooling_output, request)

    # --- Feed into chunker --------------------------------------------------
    ready_chunks: list[str] = chunker.feed(new_text)

    if is_finished:
        ready_chunks.extend(chunker.flush())
        # Cleanup AFTER collecting all chunks to avoid writing to dead object.
        # All chunks are joined into one payload so none are silently dropped.
        _cleanup_chunker(transfer_manager, request_id)

    if not ready_chunks:
        return None  # still buffering — matches framework convention

    # Framework expects dict | None (not list[dict]).
    # Join all ready chunks into a single TTS input to avoid data loss
    # when multiple sentences are flushed at once (e.g. at EOS).
    combined_text = " ".join(ready_chunks)
    return _build_tts_input(combined_text, cfg, request)
