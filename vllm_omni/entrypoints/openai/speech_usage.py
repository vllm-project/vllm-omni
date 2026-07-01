# SPDX-License-Identifier: Apache-2.0
"""Token-usage accounting for the Speech (``/v1/audio/speech``) API.

Why this module exists (issue #4646)
------------------------------------
For staged TTS models the engine prompt is a PLACEHOLDER: the serving layer
builds ``prompt_token_ids = [1] * prefill_len`` and lets the model rebuild the
real conditioning (text / ref_audio / ref_text) into ``inputs_embeds`` later.
So ``len(prompt_token_ids)`` is NOT a faithful count of what the caller sent.

For Qwen3-TTS specifically the placeholder length mirrors the model prefill:
  * CustomVoice/VoiceDesign: the full input text is embedded in the prefill, so
    the placeholder length scales with the input text.
  * Base in-context voice cloning: the prefill embeds only ``codec_bos`` + the
    reference-audio codec frames; the input text is consumed incrementally
    during DECODE, not prefill. So the placeholder length tracks the *reference
    audio* and is independent of the input text.

That is why ``usage.prompt_tokens`` (== ``len(prompt_token_ids)``) looked wrong
for Base: it counted reference-audio frames instead of the synthesized text.

This module computes usage from the *semantic* inputs instead:

    input_tokens  = text_tokens + audio_tokens
        text_tokens  -> tokens of ``input`` (+ ``instructions``); the text to speak
        audio_tokens -> reference-audio codec frames used as voice-clone
                        conditioning, counted ONLY when in-context cloning is
                        actually active (see ``gate_audio_tokens``)
    output_tokens = generated codec/audio tokens (stage-0 decode steps)
    total_tokens  = input_tokens + output_tokens

Naming (``input_tokens``/``output_tokens``) follows OpenAI's ``speech.audio.done``
event; the ``input_token_details`` breakdown follows OpenAI's realtime/chat
convention of never folding audio into an opaque text count.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from vllm_omni.entrypoints.openai.protocol.audio import (
    SpeechInputTokenDetails,
    SpeechTokenUsage,
)


def _first(value: Any, default: Any = None) -> Any:
    """Unwrap the singleton-list convention used by ``tts_params``.

    ``tts_params`` wraps scalars in 1-element lists (e.g. ``task_type=["Base"]``)
    because the model side batches per request. This returns the inner scalar.
    """
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value if value is not None else default


def gate_audio_tokens(
    *,
    task_type: str | None,
    x_vector_only_mode: bool,
    icl_mode_override: bool | None,
    ref_code_length: Any,
) -> int:
    """Reference-audio codec frames that actually enter the prefill.

    Mirrors the ``in_context_mode`` decision in
    ``Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information``:
    reference codec frames are prefilled ONLY for Base in-context voice cloning.
    They are NOT prefilled for:
      * CustomVoice / VoiceDesign (no reference audio at all), or
      * Base ``x_vector_only_mode`` (the reference audio is reduced to a single
        speaker embedding vector; no per-frame codec context is inserted).

    Counting ``ref_code_length`` outside the in-context path would re-introduce
    the issue #4646 inconsistency in the other direction, so we gate it here.
    """
    in_context = (task_type == "Base") and not x_vector_only_mode
    # An explicit per-request ``voice_clone_prompt.icl_mode`` wins if present,
    # but ONLY for the Base task -- the estimator consults ``icl_mode`` inside
    # its ``task_type == "Base"`` branch only, so CustomVoice / VoiceDesign must
    # never have audio counted even if such a flag leaks into their params.
    if task_type == "Base" and icl_mode_override is not None:
        in_context = bool(icl_mode_override)
    if not in_context:
        return 0
    try:
        return max(0, int(ref_code_length)) if ref_code_length is not None else 0
    except (TypeError, ValueError):
        return 0


def qwen3_tts_input_token_details(
    *,
    input_text: str,
    instructions: str | None,
    tts_params: dict[str, Any],
    count_text_tokens: Callable[[str], int],
) -> SpeechInputTokenDetails:
    """Compute the input-token breakdown for a Qwen3-TTS request.

    ``count_text_tokens`` tokenizes a string with the model's *text* tokenizer
    and returns the token count. ``tts_params`` is the finalized param dict from
    the adapter build (it already carries ``ref_code_length`` for ICL clones and
    the resolved ``task_type`` / ``x_vector_only_mode``).
    """
    # Text tokens = the text to synthesize plus the style/emotion instructions,
    # because Qwen3-TTS tokenizes and prepends the instruction block too.
    text = input_text or ""
    instr = instructions or ""
    text_tokens = count_text_tokens(text) if text else 0
    if instr.strip():
        text_tokens += count_text_tokens(instr)

    # Audio tokens = reference codec frames, gated to the in-context clone path.
    voice_clone_prompt = _first(tts_params.get("voice_clone_prompt"), None)
    icl_override = None
    if isinstance(voice_clone_prompt, dict):
        icl_flag = voice_clone_prompt.get("icl_mode")
        if isinstance(icl_flag, bool):
            icl_override = icl_flag
    audio_tokens = gate_audio_tokens(
        task_type=_first(tts_params.get("task_type"), "CustomVoice"),
        x_vector_only_mode=bool(_first(tts_params.get("x_vector_only_mode"), False)),
        icl_mode_override=icl_override,
        ref_code_length=_first(tts_params.get("ref_code_length"), None),
    )
    return SpeechInputTokenDetails(text_tokens=int(text_tokens), audio_tokens=int(audio_tokens))


def build_speech_usage(details: SpeechInputTokenDetails, output_tokens: int) -> SpeechTokenUsage:
    """Assemble the final usage object from the input breakdown + output count."""
    input_tokens = int(details.text_tokens) + int(details.audio_tokens)
    out = max(0, int(output_tokens))
    return SpeechTokenUsage(
        input_tokens=input_tokens,
        output_tokens=out,
        total_tokens=input_tokens + out,
        input_token_details=details,
    )


@dataclass
class SpeechOutputTokenCounter:
    """Accumulates the count of generated output (codec) tokens off the stream.

    Why not count ``res.outputs[0].token_ids``? In the staged TTS pipeline the
    token-generating stage (stage 0, the AR talker) is NOT surfaced to the audio
    consumer loop — only the downstream code2wav (audio) outputs reach it, and
    those carry an empty ``token_ids`` (their payload is a waveform, not tokens).

    The generated codec-token count is instead reported by the pipeline itself
    in per-stage metrics on the final (finished) output:
    ``res.metrics['stage_metrics'][<stage>]['num_tokens_out']``. Only the AR
    stage has a non-zero ``num_tokens_out`` (code2wav reports 0), so we take the
    max across stages. Tracking the max is safe because the metrics appear on
    the terminal output (intermediate outputs carry empty metrics).
    """

    output_tokens: int = 0

    def observe(self, res: Any) -> None:
        metrics = getattr(res, "metrics", None)
        if not isinstance(metrics, dict):
            return
        stage_metrics = metrics.get("stage_metrics")
        if not isinstance(stage_metrics, dict):
            return
        for stage in stage_metrics.values():
            if not isinstance(stage, dict):
                continue
            n = stage.get("num_tokens_out")
            if isinstance(n, int) and n > self.output_tokens:
                self.output_tokens = n

    def total(self) -> int:
        """Generated codec/audio token count (max stage ``num_tokens_out``)."""
        return self.output_tokens
