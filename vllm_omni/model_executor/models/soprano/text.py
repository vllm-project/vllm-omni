# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Official Soprano cleaning and sentence boundaries, without token truncation."""

from vllm_omni.model_executor.models.soprano.text_normalizer import clean_text
from vllm_omni.model_executor.models.soprano.text_splitter import split_and_recombine_text


def prepare_prompts(text: str) -> list[str]:
    """Match upstream SopranoTTS._preprocess_text for one input, min_length=30.

    Each returned unit needs its own native AR request and decoder invocation.
    Concatenating these prompts into one AR sequence changes the contract.
    """
    sentences = split_and_recombine_text(clean_text(text.strip()))
    merged: list[str] = []
    for i, sentence in enumerate(sentences):
        if len(sentence) < 30 and len(sentences) > 1:
            if merged:
                merged[-1] = f"{merged[-1]} {sentence}".strip()
            elif i + 1 < len(sentences):
                sentences[i + 1] = f"{sentence} {sentences[i + 1]}".strip()
            else:
                merged.append(sentence)
        else:
            merged.append(sentence)
    if not merged:
        raise ValueError("Soprano input has no speakable text after normalization")
    return [f"[STOP][TEXT]{sentence}[START]" for sentence in merged]
