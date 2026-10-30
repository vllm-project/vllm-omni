# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Soprano's HF-offline predictor contract, not a discrete audio codec."""

from collections.abc import Sequence

import torch

HIDDEN_SIZE = 512
EOS_ID = 3
SAMPLE_RATE = 32000
SAMPLES_PER_INTERVAL = 2048


def select_predictors(hidden: torch.Tensor, prompt_length: int, sampled_ids: Sequence[int]) -> torch.Tensor:
    """Keep s_j only when sampled y_j != EOS; s_0 is the prompt's last row.

    The final sampled token has not been consumed. A length stop keeps its
    predictor; an EOS stop excludes its predictor without forwarding EOS.
    """
    if prompt_length < 1 or not sampled_ids:
        raise ValueError("Soprano needs a nonempty prompt and sampled token sequence")
    expected = prompt_length + len(sampled_ids) - 1
    if hidden.ndim != 2 or hidden.shape != (expected, HIDDEN_SIZE):
        raise ValueError(f"Expected consumed trajectory [{expected},512], got {tuple(hidden.shape)}")
    if not hidden.is_floating_point():
        raise ValueError("Soprano handoff requires continuous floating-point hidden states")
    if not torch.isfinite(hidden).all():
        raise ValueError("Soprano hidden trajectory contains non-finite values")
    if EOS_ID in sampled_ids[:-1]:
        raise ValueError("Tokens after Soprano EOS are not valid")
    count = len(sampled_ids) - int(sampled_ids[-1] == EOS_ID)
    return hidden[prompt_length - 1 : prompt_length - 1 + count].contiguous()


def transform_prompt(prompt, sampling_params_list):
    """Normalize a single unit; multi-sentence callers use offline.synthesize."""
    params = sampling_params_list[0]
    if params.max_tokens is None or not 1 <= params.max_tokens <= 512:
        raise ValueError("Soprano offline max_tokens must be in 1..512")
    if params.ignore_eos or params.stop or set(params.stop_token_ids or ()) - {EOS_ID}:
        raise ValueError("Soprano offline supports only EOS=3 and max_tokens stops")
    if isinstance(prompt, dict) and "prompt_token_ids" in prompt:
        if not 1 <= len(prompt["prompt_token_ids"]) <= 256:
            raise ValueError("Soprano prompt must contain 1..256 tokens")
        return prompt
    text = prompt if isinstance(prompt, str) else prompt.get("prompt", "")
    from vllm_omni.model_executor.models.soprano.text import prepare_prompts

    prompts = prepare_prompts(text)
    if len(prompts) != 1:
        raise ValueError("Use prepare_prompts or offline.synthesize for Soprano multi-sentence inputs")
    result = {} if isinstance(prompt, str) else dict(prompt)
    result["prompt"] = prompts[0]
    return result
