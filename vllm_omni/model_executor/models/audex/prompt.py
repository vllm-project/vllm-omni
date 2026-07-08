# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audex TTS prompt construction.

The Audex thinker consumes the exact ChatML prompt used by the official
inference script (``inference_scripts_vllm/audiogen_scripts/run_audio_gen_vllm.py``
in the nvidia/Nemotron-Labs-Audex-2B repo). The checkpoint's bundled chat
template opens a thinking block (``<think>\\n``) instead of the closed
``<think></think>`` + ``<speechgen_start>`` priming that TTS generation
requires, so the prompt is built from a literal template here; a unit test
pins it byte-for-byte against the official format.

``build_null_prompt`` is the unconditional-prompt counterpart used by
classifier-free guidance. CFG execution is not supported yet, so it is a
guarded stub; the split exists so CFG support can slot in without reshaping
callers.
"""

AUDEX_SYSTEM_PROMPT = "You are a helpful and harmless assistant.\n\nYou are not allowed to use any tools."

_TTS_PROMPT_TEMPLATE = (
    "<|im_start|>system\n{system_prompt}<|im_end|>\n"
    "<|im_start|>user\n<|text to speech|> Generate speech for this transcription. {text}<|im_end|>\n"
    "<|im_start|>assistant\n<think></think><speechgen_start>"
)


def build_cond_prompt(text: str) -> str:
    """Build the conditional TTS prompt for one transcription."""
    text = text.strip()
    if not text:
        raise ValueError("Audex TTS requires non-empty input text")
    return _TTS_PROMPT_TEMPLATE.format(system_prompt=AUDEX_SYSTEM_PROMPT, text=text)


def build_null_prompt(cond_prompt: str, tokenizer) -> str:
    """Unconditional prompt for CFG (length-matched ``<unk>`` padding).

    Not implemented: Audex CFG execution is not supported yet (requests with
    ``cfg_scale > 1.0`` are rejected at the serving layer).
    """
    raise NotImplementedError("Audex classifier-free guidance is not supported yet")
