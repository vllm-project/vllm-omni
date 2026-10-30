# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""C1 complete-response frontend over an already initialized native Omni engine.

No model loading or inference happens at import. Sentence fanout is explicit
here because the engine prompt-transform hook admits one request at a time.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from vllm_omni.model_executor.models.soprano.contracts import SAMPLE_RATE
from vllm_omni.model_executor.models.soprano.text import prepare_prompts

if TYPE_CHECKING:
    from vllm_omni import Omni
    from vllm_omni.inputs.data import OmniSamplingParams


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool, truncation: bool) -> list[int]: ...


class SopranoSynthesisError(ValueError):
    """A completed request failed synthesis; the engine remains reusable."""


@dataclass
class SopranoAudio:
    audio: torch.Tensor
    sample_rate: int
    request_ids: tuple[str, ...]
    truncated: bool


def prepare_inputs(text: str, tokenizer: Tokenizer) -> list[dict[str, list[int]]]:
    """Validate every official sentence before admitting the first request."""
    inputs = []
    for prompt in prepare_prompts(text):
        ids = tokenizer.encode(prompt, add_special_tokens=True, truncation=False)
        if not 1 <= len(ids) <= 256:
            raise ValueError("Soprano sentence must contain 1..256 prompt tokens; no tokens were truncated")
        inputs.append({"prompt_token_ids": ids})
    return inputs


def _flag(value) -> bool:
    if isinstance(value, list):
        return any(_flag(item) for item in value)
    return bool(value) if value is not None else False


def synthesize(
    engine: "Omni",
    text: str,
    tokenizer: Tokenizer,
    sampling_params_list: list["OmniSamplingParams"] | None = None,
) -> SopranoAudio:
    """Clean/split English, run native requests, decode B1, concatenate audio.

    The caller owns the engine and tokenizer. Sampling settings (including seed)
    apply per sentence; HF batched generation has a different RNG schedule.
    Failed sentences raise without returning a partially synthesized utterance.
    """
    inputs = prepare_inputs(text, tokenizer)
    waves = []
    request_ids = []
    truncated = False
    for prompt in inputs:
        outputs = engine.generate(prompt, sampling_params_list=sampling_params_list, use_tqdm=False)
        audio_outputs = [output for output in outputs if output.final_output_type == "audio"]
        if len(audio_outputs) != 1:
            raise SopranoSynthesisError("Soprano did not return exactly one completed audio request")
        output = audio_outputs[0]
        mm = output.outputs[0].multimodal_output
        if mm is None or _flag(mm.get("audio_synthesis_error")):
            raise SopranoSynthesisError(f"Soprano synthesis failed for request {output.request_id}")
        chunks = mm.get("audio")
        if chunks is None:
            raise SopranoSynthesisError("Soprano request completed without audio")
        chunks = chunks if isinstance(chunks, list) else [chunks]
        wave = torch.cat([chunk.detach().float().cpu().reshape(-1) for chunk in chunks])
        if wave.numel() == 0 or not torch.isfinite(wave).all():
            raise SopranoSynthesisError("Soprano request completed without finite nonempty audio")
        rate = mm.get("sr")
        rate = rate[-1] if isinstance(rate, list) else rate
        if rate is None or int(rate) != SAMPLE_RATE:
            raise SopranoSynthesisError("Soprano returned an unexpected sample rate")
        waves.append(wave)
        request_ids.append(output.request_id)
        truncated |= _flag(mm.get("audio_truncated"))
    return SopranoAudio(torch.cat(waves), SAMPLE_RATE, tuple(request_ids), truncated)
