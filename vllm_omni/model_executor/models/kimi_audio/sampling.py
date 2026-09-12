# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One Kimi-Audio generation step; request history and the loop stay outside.

Preserve KimiASampler's filter order and KimiAudio._generate_loop's token
overrides. The runner adapter must record the returned tokens in each request's
own history, including forced blanks and the first text EOS.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Any, Literal

import torch

from .prompt import KimiAudioSpecialTokens

if TYPE_CHECKING:
    from vllm.v1.sample.metadata import SamplingMetadata


@dataclass(frozen=True)
class KimiAudioSamplingParams:
    audio_top_k: int = 5
    audio_temperature: float = 0.0
    audio_repetition_penalty: float = 1.0
    audio_repetition_window_size: int = 64
    text_top_k: int = 5
    text_temperature: float = 0.0
    text_repetition_penalty: float = 1.0
    text_repetition_window_size: int = 16

    @classmethod
    def from_sampling_metadata(
        cls, metadata: "SamplingMetadata", row: int, extra_args: dict[str, Any]
    ) -> "KimiAudioSamplingParams":
        """Read the native batch row and apply per-request stream overrides.

        vLLM may omit temperature/top-k tensors for an all-greedy or unfiltered
        batch. Repetition still uses Kimi's generated-token window semantics.
        """
        overrides = extra_args.get("kimi_audio", {})
        if not isinstance(overrides, dict):
            raise ValueError("SamplingParams.extra_args.kimi_audio must be a parameter dictionary")
        values = dict(
            text_temperature=(0.0 if metadata.all_greedy else float(metadata.temperature[row])),
            text_top_k=0 if metadata.top_k is None else int(metadata.top_k[row]),
            text_repetition_penalty=(1.0 if metadata.no_penalties else float(metadata.repetition_penalties[row])),
        )
        values.update(overrides)
        return cls(**values)

    def __post_init__(self) -> None:
        for stream in ("text", "audio"):
            for suffix, minimum in (("temperature", 0.0), ("repetition_penalty", 1.0)):
                value = getattr(self, f"{stream}_{suffix}")
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not isfinite(value)
                    or value < minimum
                ):
                    raise ValueError(f"{stream}_{suffix} must be finite and >= {minimum}")
            for suffix, minimum in (("top_k", -1), ("repetition_window_size", 1)):
                value = getattr(self, f"{stream}_{suffix}")
                if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                    raise ValueError(f"{stream}_{suffix} must be an integer >= {minimum}")


@dataclass(frozen=True)
class KimiAudioSample:
    text_token: int
    audio_token: int
    text_finished: bool
    finished: bool


def sample_kimi_audio_step(
    text_logits: torch.Tensor,
    audio_logits: torch.Tensor,
    *,
    text_history: Sequence[int],
    audio_history: Sequence[int],
    text_finished: bool,
    output_type: Literal["text", "both"],
    special_tokens: KimiAudioSpecialTokens,
    audio_delay: int,
    params: KimiAudioSamplingParams = KimiAudioSamplingParams(),
    generator: torch.Generator | None = None,
) -> KimiAudioSample:
    """Sample two full-vocabulary rows for ONE request, then apply stream rules.

    Both logits are [vocab_size], with IDs already in the Kimi vocabulary.
    Histories contain generated tokens only, after previous steps' overrides.
    No history, progress, or RNG is retained here. This is the model-specific
    step called by the AR model's runner sampler adapter, not a generation loop.
    """
    if text_logits.ndim != 1 or audio_logits.shape != text_logits.shape:
        raise ValueError("Kimi-Audio sampling expects two [vocab_size] rows")
    if len(text_history) != len(audio_history):
        raise ValueError("Kimi-Audio text and audio histories must have the same number of steps")
    if output_type not in ("text", "both") or audio_delay < 0:
        raise ValueError("Kimi-Audio requires output_type text/both and a nonnegative audio delay")

    sampled = []
    # Keep the official draw order, even for a branch whose token will be
    # overwritten below. Skipping that draw changes seeded stochastic output.
    for logits, history, top_k, temperature, penalty, window in (
        (
            text_logits,
            text_history,
            params.text_top_k,
            params.text_temperature,
            params.text_repetition_penalty,
            params.text_repetition_window_size,
        ),
        (
            audio_logits,
            audio_history,
            params.audio_top_k,
            params.audio_temperature,
            params.audio_repetition_penalty,
            params.audio_repetition_window_size,
        ),
    ):
        if window <= 0:
            raise ValueError("Kimi-Audio repetition windows must be positive")
        # Official behavior is strictly > window, not >=, and includes
        # generated control tokens/blanks. Leave the caller's logits intact.
        if penalty > 1.0 and len(history) > window:
            logits = logits.clone()
            recent = torch.as_tensor(history[-window:], device=logits.device, dtype=torch.long)
            scores = logits.gather(0, recent)
            scores = torch.where(scores < 0, scores * penalty, scores / penalty)
            logits.scatter_(0, recent, scores)

        logprobs = torch.log_softmax(logits.unsqueeze(0), dim=-1, dtype=torch.float)
        if temperature > 1e-6:
            logprobs = logprobs / temperature
            if top_k > 0:
                top_probs, top_ids = torch.topk(torch.exp(logprobs), top_k, dim=-1)
                selected = torch.multinomial(top_probs, num_samples=1, generator=generator)
                token = top_ids.gather(-1, selected)
            else:
                token = torch.multinomial(torch.exp(logprobs), num_samples=1, generator=generator)
        else:
            token = torch.argmax(logprobs, dim=-1)
        sampled.append(int(token.item()))

    text_token, audio_token = sampled
    if text_finished:
        text_token = special_tokens.kimia_text_blank
    elif text_token == special_tokens.kimia_text_eos:
        text_finished = True

    if len(audio_history) < audio_delay or output_type == "text":
        audio_token = special_tokens.kimia_text_blank

    finished = (
        text_finished if output_type == "text" else audio_token in (special_tokens.msg_end, special_tokens.media_end)
    )
    return KimiAudioSample(text_token, audio_token, text_finished, finished)
