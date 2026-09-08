# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audex TTS/TTA prompt construction.

The Audex thinker consumes the exact ChatML prompt used by the official
inference script (``inference_scripts_vllm/audiogen_scripts/run_audio_gen_vllm.py``
in the nvidia/Nemotron-Labs-Audex-2B repo). The checkpoint's bundled chat
template opens a thinking block (``<think>\\n``) instead of the closed
``<think></think>`` + ``<speechgen_start>`` priming that TTS generation
requires, so the prompt is built from a literal template here; a unit test
pins it byte-for-byte against the official format.

``build_null_prompt`` is the unconditional-prompt counterpart used by
classifier-free guidance: the transcription is replaced with repeated
``<unk>`` tokens, iteratively adjusted so the tokenized null prompt is
exactly as long as the conditional prompt (the CFG pair must decode the
same positions).
"""

AUDEX_SYSTEM_PROMPT = "You are a helpful and harmless assistant.\n\nYou are not allowed to use any tools."

# ``{text}`` carries its own leading whitespace: the conditional prompt uses
# " {transcription}" (matching the official script byte-for-byte), while the
# CFG null prompt inserts the bare ``<unk>`` run. ``<unk>`` is a special
# token, so a leading space would tokenize as a standalone space token and
# make some conditional lengths unreachable by whole-``<unk>`` steps.
_TTS_PROMPT_TEMPLATE = (
    "<|im_start|>system\n{system_prompt}<|im_end|>\n"
    "<|im_start|>user\n<|text to speech|> Generate speech for this transcription.{text}<|im_end|>\n"
    "<|im_start|>assistant\n<think></think><speechgen_start>"
)

_TTA_PROMPT_TEMPLATE = (
    "<|im_start|>system\n{system_prompt}<|im_end|>\n"
    "<|im_start|>user\n<|text to audio|> Generate audio for this caption.{text}<|im_end|>\n"
    "<|im_start|>assistant\n<think></think><audiogen_start>"
)


def build_cond_prompt(text: str) -> str:
    """Build the conditional TTS prompt for one transcription."""
    text = text.strip()
    if not text:
        raise ValueError("Audex TTS requires non-empty input text")
    return _TTS_PROMPT_TEMPLATE.format(system_prompt=AUDEX_SYSTEM_PROMPT, text=f" {text}")


def build_tta_cond_prompt(caption: str) -> str:
    """Build the conditional TTA prompt for one audio caption."""
    caption = caption.strip()
    if not caption:
        raise ValueError("Audex TTA requires a non-empty caption")
    return _TTA_PROMPT_TEMPLATE.format(system_prompt=AUDEX_SYSTEM_PROMPT, text=f" {caption}")


def _length_matched_null(cond_prompt: str, tokenizer, template: str, max_iters: int) -> str:
    target_len = len(tokenizer.encode(cond_prompt))
    n_unk = 1
    for _ in range(max_iters):
        prompt = template.format(system_prompt=AUDEX_SYSTEM_PROMPT, text="<unk>" * n_unk)
        diff = target_len - len(tokenizer.encode(prompt))
        if diff == 0:
            return prompt
        n_unk = max(1, n_unk + diff)
        if diff < 0 and n_unk == 1:
            break
    raise ValueError(f"Could not length-match the CFG null prompt to {target_len} tokens")


def build_null_prompt(cond_prompt: str, tokenizer, max_iters: int = 64) -> str:
    """Unconditional TTS prompt for CFG (length-matched ``<unk>`` padding).

    Replaces the transcription with ``<unk>`` repeated, adjusting the count
    until ``tokenizer.encode`` yields exactly the conditional prompt's
    length. Raises when no count matches so callers can fail the request
    instead of submitting a misaligned pair.
    """
    return _length_matched_null(cond_prompt, tokenizer, _TTS_PROMPT_TEMPLATE, max_iters)


def build_tta_null_prompt(cond_prompt: str, tokenizer, max_iters: int = 64) -> str:
    """Unconditional TTA prompt for CFG (length-matched ``<unk>`` padding)."""
    return _length_matched_null(cond_prompt, tokenizer, _TTA_PROMPT_TEMPLATE, max_iters)
