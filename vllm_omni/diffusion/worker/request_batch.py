# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-level batch abstraction for diffusion runner."""

from __future__ import annotations

from dataclasses import dataclass

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType


@dataclass
class DiffusionRequestBatch:
    """Request-level batch wrapping original diffusion requests.

    Each :class:`~vllm_omni.diffusion.request.OmniDiffusionRequest` represents
    one logical diffusion request with one prompt. The scheduler and runner use
    this wrapper to present a compatible request batch to pipeline
    ``forward()`` methods without reintroducing list-shaped request payloads.

    This is distinct from ``InputBatch`` (aliased as ``StepInputBatch``),
    which manages step/tensor-level data for stepwise execution.

    Args:
        requests: Independent diffusion requests scheduled together for
            request-mode execution.

    Attributes:
        requests: Original request objects in scheduler order.
        num_reqs: Number of requests in the batch.
        request_ids: Request IDs in the same order as ``requests``.
        prompts: Prompt list assembled from each request's single ``prompt``.
        sampling_params: Sampling parameters shared by the batch. Scheduler
            compatibility checks ensure batched requests can use this value.
        request_id: First request ID, kept as a compatibility convenience for
            code paths that handle a single-request batch.
        kv_sender_info: KV-transfer metadata from the first request.
    """

    requests: list[OmniDiffusionRequest]

    @property
    def num_reqs(self) -> int:
        return len(self.requests)

    @property
    def request_ids(self) -> list[str]:
        return [req.request_id for req in self.requests]

    @property
    def prompts(self) -> list[OmniPromptType]:
        return [req.prompt for req in self.requests]

    @property
    def sampling_params(self) -> OmniDiffusionSamplingParams:
        return self.requests[0].sampling_params

    @property
    def request_id(self) -> str:
        return self.requests[0].request_id

    @property
    def kv_sender_info(self) -> dict | None:
        return self.requests[0].kv_sender_info

    def get(self, request_id: str) -> OmniDiffusionRequest | None:
        for req in self.requests:
            if req.request_id == request_id:
                return req
        return None
