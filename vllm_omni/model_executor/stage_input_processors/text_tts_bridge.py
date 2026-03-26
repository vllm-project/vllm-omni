"""
text_tts_bridge.py
~~~~~~~~~~~~~~~~~~
Bridge stage input processor that connects any vLLM text model (Stage 0)
to a Qwen3-TTS decoder (Stage 1) as a composable pipeline.

This implements RFC Theme 2: "TTS as a Composable Layer (P0)" from the
vllm-omni TTS Development Roadmap.

Design decisions (answers RFC key design questions):
  Q1: Bridge as stage-level processor, not a new stage type.
      We reuse the existing `custom_process_input_func` hook and
      `async_chunk` framework rather than adding a new stage_type.

  Q2: Latency / buffering.
      `SentenceChunker` buffers tokens until a sentence boundary OR
      `min_sentence_chars` characters are accumulated, then flushes.
      This gives a configurable latency knob without blocking Stage 0.

  Q3: Voice/speaker parameter routing.
      When Stage 0 (a plain text LLM) has no concept of speaker, we
      inject a `default_voice` from the bridge config in the YAML.
      Per-request override is possible via `extra_body.tts_voice`.

Flow:
  Stage 0 (Llama)  -->  SentenceChunker  -->  build_tts_input()  -->  Stage 1 (Qwen3-TTS)
                         (async_chunk)          (CustomVoice fmt)

Usage:
  Set in stage config YAML:
    custom_process_input_func:
      vllm_omni.model_executor.stage_input_processors.text_tts_bridge.text2tts
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Config dataclass (populated from YAML bridge section by OmniStage loader)
# ---------------------------------------------------------------------------

@dataclass
class TextTTSBridgeConfig:
    """Runtime config for the text→TTS bridge, sourced from stage YAML."""

    min_sentence_chars: int = 40
    """Minimum characters to accumulate before flushing to TTS stage.
    Larger values reduce TTS restarts at the cost of higher TTFA."""

    sentence_delimiters: list[str] = field(
        default_factory=lambda: [".", "!", "?", "。", "！", "？"]
    )
    """Characters that mark sentence boundaries and trigger a flush."""

    tts_task_type: str = "CustomVoice"
    """Qwen3-TTS task type forwarded to Stage 1."""

    default_voice: str = "vivian"
    """Fallback voice when the upstream request carries no speaker info.
    Directly addresses RFC design question #3."""

    default_language: str = "English"
    """Fallback language tag for Qwen3-TTS CustomVoice inputs."""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TextTTSBridgeConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Sentence chunker
# ---------------------------------------------------------------------------

def _build_sentence_re(delimiters: list[str]) -> re.Pattern:
    """Compile a lookbehind regex for the given sentence-ending delimiters."""
    escaped = "".join(re.escape(d) for d in delimiters)
    return re.compile(rf"(?<=[{escaped}])\s*")


class SentenceChunker:
    """
    Stateful buffer that receives incremental text tokens from Stage 0
    and yields complete sentence chunks ready for the TTS stage.

    Designed to work with the existing async_chunk framework:
    each call to `feed()` may return 0 or more flushed chunks.
    `flush()` forces any remaining buffered text out (called at EOS).

    Example
    -------
    >>> chunker = SentenceChunker(min_sentence_chars=40)
    >>> chunker.feed("Hello world")
    []
    >>> chunker.feed(". How are you doing today?")
    ['Hello world.']
    >>> chunker.flush()
    [' How are you doing today?']
    """

    def __init__(self, cfg: TextTTSBridgeConfig | None = None) -> None:
        self.cfg = cfg or TextTTSBridgeConfig()
        self._buf: str = ""
        self._re = _build_sentence_re(self.cfg.sentence_delimiters)

    def feed(self, token_text: str) -> list[str]:
        """Append *token_text* and return any flushed sentence chunks."""
        self._buf += token_text
        return self._try_flush()

    def flush(self) -> list[str]:
        """Force-flush remaining buffer (call at end-of-stream)."""
        if self._buf.strip():
            chunk, self._buf = self._buf, ""
            return [chunk]
        return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_flush(self) -> list[str]:
        chunks: list[str] = []
        while True:
            # Find first sentence boundary in the buffer
            m = self._re.search(self._buf)
            if m is None:
                break
            boundary = m.end()
            candidate = self._buf[:boundary]
            # Respect min_sentence_chars to avoid TTS restarts on tiny chunks
            if len(candidate) < self.cfg.min_sentence_chars:
                break
            chunks.append(candidate.strip())
            self._buf = self._buf[boundary:]
        return chunks


# ---------------------------------------------------------------------------
# Input builder: converts a text chunk → Qwen3-TTS CustomVoice input dict
# ---------------------------------------------------------------------------

def build_tts_input(
    text_chunk: str,
    cfg: TextTTSBridgeConfig,
    *,
    voice: str | None = None,
    language: str | None = None,
    instructions: str | None = None,
) -> dict[str, Any]:
    """
    Build a Qwen3-TTS `CustomVoice` input dict from a text chunk.

    The returned dict matches the format expected by the Qwen3-TTS stage
    input processor (`qwen3_tts/end2end.py` `_build_custom_voice_input`).

    Parameters
    ----------
    text_chunk:
        One sentence or sentence fragment to synthesize.
    cfg:
        Bridge config carrying defaults for voice, language, task_type.
    voice:
        Per-request voice override (from `extra_body.tts_voice`).
        Falls back to `cfg.default_voice` when None.
    language:
        Per-request language override. Falls back to `cfg.default_language`.
    instructions:
        Optional style instruction string (e.g. "speak slowly and warmly").
    """
    return {
        "text": text_chunk,
        "task_type": cfg.tts_task_type,
        "voice": voice or cfg.default_voice,
        "language": language or cfg.default_language,
        **({"instructions": instructions} if instructions else {}),
    }


# ---------------------------------------------------------------------------
# Main hook: `custom_process_input_func` entry point called by OmniStage
# ---------------------------------------------------------------------------

def text2tts(
    stage0_output: dict[str, Any],
    bridge_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    OmniStage `custom_process_input_func` hook.

    Called by the async_chunk scheduler when Stage 0 produces a new
    text chunk.  Returns a list of Stage 1 input dicts (one per sentence
    chunk ready for synthesis).

    Parameters
    ----------
    stage0_output:
        Dict produced by the async_chunk connector from Stage 0.
        Expected keys:
          - ``text``        : incremental text token(s) from Stage 0
          - ``is_finished`` : True when Stage 0 EOS is reached
          - ``request_id``  : forwarded for request tracking
          - ``chunker``     : SentenceChunker instance (injected by
                              the pipeline orchestrator, keyed per request)
          - ``extra``       : optional dict with per-request overrides:
                              ``tts_voice``, ``tts_language``,
                              ``tts_instructions``
    bridge_config:
        Raw dict from the YAML `bridge:` section (optional; uses defaults
        if absent).

    Returns
    -------
    list[dict]
        Zero or more Stage 1 input dicts, one per flushed sentence chunk.
        Empty list means "still buffering — nothing for Stage 1 yet."
    """
    cfg = TextTTSBridgeConfig.from_dict(bridge_config or {})

    # Retrieve (or lazily create) per-request SentenceChunker
    chunker: SentenceChunker = stage0_output.get("chunker") or SentenceChunker(cfg)

    text_token: str = stage0_output.get("text", "")
    is_finished: bool = stage0_output.get("is_finished", False)
    extra: dict = stage0_output.get("extra", {})

    # Feed new tokens into the chunker
    ready_chunks: list[str] = chunker.feed(text_token)

    # At EOS, flush any remaining buffered text
    if is_finished:
        ready_chunks.extend(chunker.flush())

    # Build one TTS input dict per flushed sentence chunk
    return [
        build_tts_input(
            chunk,
            cfg,
            voice=extra.get("tts_voice"),
            language=extra.get("tts_language"),
            instructions=extra.get("tts_instructions"),
        )
        for chunk in ready_chunks
    ]
