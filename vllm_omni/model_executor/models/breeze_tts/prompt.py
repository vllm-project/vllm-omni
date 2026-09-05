# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Breeze voice-design prompts for both Omni.generate and the speech API."""

from transformers import PreTrainedTokenizerBase
from vllm.inputs import TokensPrompt, tokens_input


def build_breeze_prompt(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    instructions: str = "",
    *,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    repetition_penalty: float = 1.1,
) -> TokensPrompt:
    if not text.strip():
        raise ValueError("Breeze input text cannot be empty")
    if temperature < 0 or top_k < 0 or not 0 < top_p <= 1 or repetition_penalty <= 0:
        raise ValueError("Invalid Breeze sampling parameters")
    instruction = f"<ins_bos>{instructions}<ins_eos>" if instructions else ""
    text_ids = tokenizer.encode(f"[S0]{instruction}{text}", add_special_tokens=True)
    prompt = tokens_input(prompt_token_ids=[0] * len(text_ids))
    prompt["additional_information"] = {
        "ids": {"prompt": text_ids},
        "breeze_sampling": {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        },
    }
    return prompt
