# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Bounded, physical-request-owned inputs for KV preemption recomputation."""

from bisect import bisect_right
from dataclasses import dataclass

import torch

from vllm_omni.model_executor.models.output_templates import ModelInputError


@dataclass(frozen=True)
class _InputSpan:
    start: int
    end: int
    identity: tuple[object, ...]
    embeddings: torch.Tensor
    token_ids: tuple[int, ...]


class DuplexPromptHistory:
    """Retain actual audio/vision embeddings, not scheduler placeholders.

    Gaps between units are computed output tokens and use normal embedding
    lookup. Spans live on CPU, are added once per unit, and are released with
    the physical request. Lookup touches only spans overlapping this forward.
    """

    def __init__(self, *, max_tokens: int, max_bytes: int = 512 * 1024 * 1024) -> None:
        self.max_tokens = max_tokens
        self.max_bytes = max_bytes
        self.num_bytes = 0
        self.spans: list[_InputSpan] = []
        self.starts: list[int] = []

    def append(
        self,
        *,
        prompt_len: int,
        embeddings: torch.Tensor,
        token_ids: list[int],
        identity: tuple[object, ...],
        expected_start: int | None = None,
    ) -> None:
        count = len(token_ids)
        start = prompt_len - count
        if count <= 0 or embeddings.ndim != 2 or embeddings.shape[0] != count or start < 0:
            raise ModelInputError("native_duplex_input_budget_mismatch: invalid prepared unit span")
        if prompt_len > self.max_tokens:
            raise ModelInputError("native_duplex_input_history_capacity_exhausted: context limit")
        if expected_start is not None and start != expected_start:
            raise ModelInputError(
                f"native_duplex_input_budget_mismatch: reserved={prompt_len - expected_start}, prepared={count}"
            )
        if self.spans and self.spans[-1].identity == identity:
            last = self.spans[-1]
            if (last.start, last.end, last.token_ids) != (start, prompt_len, tuple(token_ids)):
                raise ModelInputError("native_duplex_input_budget_mismatch: prepared unit changed on retry")
            return
        if (not self.spans and start != 0) or (self.spans and start < self.spans[-1].end):
            raise ModelInputError("native_duplex_input_history_unavailable: missing prefix or overlapping unit")
        size = embeddings.numel() * embeddings.element_size()
        if self.num_bytes + size > self.max_bytes:
            raise ModelInputError("native_duplex_input_history_capacity_exhausted: embedding byte limit")
        # copy=True also makes CPU tests/CPU producers safe from later mutation.
        stored = embeddings.detach().to(device="cpu", copy=True).contiguous()
        self.spans.append(_InputSpan(start, prompt_len, identity, stored, tuple(token_ids)))
        self.starts.append(start)
        self.num_bytes += size

    def overlay(self, *, offset: int, input_ids: torch.Tensor, embeddings: torch.Tensor) -> None:
        end = offset + input_ids.shape[0]
        index = max(0, bisect_right(self.starts, offset) - 1)
        while index < len(self.spans):
            span = self.spans[index]
            index += 1
            if span.start >= end:
                break
            lo, hi = max(offset, span.start), min(end, span.end)
            if lo >= hi:
                continue
            source = slice(lo - span.start, hi - span.start)
            target = slice(lo - offset, hi - offset)
            embeddings[target] = span.embeddings[source].to(device=embeddings.device, dtype=embeddings.dtype)
            input_ids[target] = torch.tensor(span.token_ids[source], device=input_ids.device, dtype=input_ids.dtype)

    def prompt_token_ids(self, scheduler_ids: list[int]) -> list[int]:
        result = list(scheduler_ids)
        for span in self.spans:
            result[span.start : span.end] = span.token_ids
        return result
