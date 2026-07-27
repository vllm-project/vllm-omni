# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-level batch abstraction for diffusion runner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType


def _slice_request_output(value: Any, start: int, stop: int) -> Any:
    if isinstance(value, tuple):
        return tuple(_slice_request_output(item, start, stop) for item in value)
    if isinstance(value, list):
        return value[start:stop]
    if isinstance(value, torch.Tensor):
        return value[start:stop]
    return value


def split_diffusion_output_by_request(
    result: DiffusionOutput,
    req: DiffusionRequestBatch,
    *,
    num_outputs_per_prompt: int,
) -> list[DiffusionOutput]:
    """Split a batched DiffusionOutput into one output per request."""
    if num_outputs_per_prompt <= 0:
        raise ValueError(f"num_outputs_per_prompt must be positive, got {num_outputs_per_prompt}.")

    return [
        DiffusionOutput(
            output=_slice_request_output(
                result.output,
                idx * num_outputs_per_prompt,
                (idx + 1) * num_outputs_per_prompt,
            ),
            error=result.error,
            finished=result.finished,
            stage_durations=result.stage_durations,
            peak_memory_mb=result.peak_memory_mb,
            chunk_index=result.chunk_index,
            total_chunks=result.total_chunks,
            started_event_ids=result.started_event_ids,
            active_event_ids=result.active_event_ids,
            completed_event_ids=result.completed_event_ids,
        )
        for idx in range(req.num_reqs)
    ]


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
        sampling_params_list: Per-request sampling parameters in scheduler
            order. Request-batch pipelines read request-local values here.
        sampling_params: Sampling parameters for single-request legacy paths.
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
    def sampling_params_list(self) -> list[OmniDiffusionSamplingParams]:
        return [req.sampling_params for req in self.requests]

    @property
    def sampling_params(self) -> OmniDiffusionSamplingParams:
        # Legacy pipelines do not accept RequestBatch, so they are invoked with
        # one request at a time. In that path, this batch is expected to contain
        # a single request, and we expose its sampling params for compatibility.
        assert len(self.requests) == 1, "RequestBatch with multiple requests does not have a single sampling_params"
        return self.requests[0].sampling_params

    @property
    def request_id(self) -> str:
        return self.requests[0].request_id

    @property
    def kv_sender_info(self) -> dict | None:
        return self.requests[0].kv_sender_info

    def is_dummy_run(self) -> bool:
        return self.requests[0].is_dummy_run_request_id(self.request_id)

    def get(self, request_id: str) -> OmniDiffusionRequest | None:
        for req in self.requests:
            if req.request_id == request_id:
                return req
        return None

    def collate_request_generators(
        self,
        num_outputs_per_prompt: int,
        default_generator: torch.Generator | list[torch.Generator] | None,
    ) -> torch.Generator | list[torch.Generator] | None:
        return self.collate_sampling_param_generators(
            self.sampling_params_list,
            num_outputs_per_prompt,
            default_generator,
        )

    def collate_request_tensors(
        self,
        attr: str,
        default_tensor: torch.Tensor | None,
    ) -> torch.Tensor | None:
        return self.collate_tensors(
            [getattr(sampling, attr) for sampling in self.sampling_params_list],
            attr,
            default_tensor,
        )

    @staticmethod
    def collate_tensors(
        tensors: list[torch.Tensor | None],
        name: str,
        default_tensor: torch.Tensor | None,
    ) -> torch.Tensor | None:
        validated_tensors = DiffusionRequestBatch._validate_tensor_sequence(tensors, name)
        if validated_tensors is None:
            return default_tensor
        return torch.cat(validated_tensors, dim=0)

    @staticmethod
    def collate_prompt_tensors(
        tensors: list[torch.Tensor | None],
        name: str,
        default_tensor: torch.Tensor | None,
    ) -> torch.Tensor | None:
        validated_tensors = DiffusionRequestBatch._validate_tensor_sequence(tensors, name)
        if validated_tensors is None:
            return default_tensor
        return torch.stack(validated_tensors, dim=0)

    @staticmethod
    def get_prompt_field(prompt: OmniPromptType, name: str) -> Any:
        if isinstance(prompt, str):
            return None
        value = prompt.get(name)
        if value is None:
            additional = prompt.get("additional_information")
            if isinstance(additional, dict):
                value = additional.get(name)
        if isinstance(value, list):
            return value[0] if value else None
        return value

    @staticmethod
    def collate_prompt_fields(
        prompts: list[OmniPromptType],
        name: str,
        default_tensor: torch.Tensor | None,
    ) -> torch.Tensor | None:
        return DiffusionRequestBatch.collate_prompt_tensors(
            [DiffusionRequestBatch.get_prompt_field(prompt, name) for prompt in prompts],
            name,
            default_tensor,
        )

    @staticmethod
    def get_prompt_field_with_aliases(prompt: OmniPromptType, names: Sequence[str]) -> Any:
        for name in names:
            value = DiffusionRequestBatch.get_prompt_field(prompt, name)
            if value is not None:
                return value
        return None

    @staticmethod
    def collate_prompt_field_map(
        prompts: list[OmniPromptType],
        field_defaults: Mapping[str, torch.Tensor | None],
        field_aliases: Mapping[str, Sequence[str]] | None = None,
    ) -> dict[str, torch.Tensor | None]:
        collated_fields: dict[str, torch.Tensor | None] = {}
        for name, default_tensor in field_defaults.items():
            aliases = field_aliases.get(name, (name,)) if field_aliases is not None else (name,)
            collated_fields[name] = DiffusionRequestBatch.collate_prompt_tensors(
                [DiffusionRequestBatch.get_prompt_field_with_aliases(prompt, aliases) for prompt in prompts],
                name,
                default_tensor,
            )
        return collated_fields

    @staticmethod
    def _validate_tensor_sequence(
        tensors: list[torch.Tensor | None],
        name: str,
    ) -> list[torch.Tensor] | None:
        if not any(tensor is not None for tensor in tensors):
            return None
        if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise ValueError(f"Cannot batch requests with a mix of provided and missing {name}.")

        first = tensors[0]
        assert isinstance(first, torch.Tensor)
        for tensor in tensors[1:]:
            assert isinstance(tensor, torch.Tensor)
            if tensor.shape != first.shape or tensor.dtype != first.dtype or tensor.device != first.device:
                raise ValueError(
                    f"Batched request {name} must have matching shape, dtype, and device; "
                    f"got {tensor.shape}/{tensor.dtype}/{tensor.device} and "
                    f"{first.shape}/{first.dtype}/{first.device}."
                )
        return tensors

    @staticmethod
    def collate_sampling_param_generators(
        sampling_params_list: list[Any],
        num_outputs_per_prompt: int,
        default_generator: torch.Generator | list[torch.Generator] | None,
    ) -> torch.Generator | list[torch.Generator] | None:
        request_generators = [sampling.generator for sampling in sampling_params_list]
        if not any(generator is not None for generator in request_generators):
            return default_generator
        if not all(generator is not None for generator in request_generators):
            raise ValueError("Cannot batch requests with a mix of provided and missing generators.")

        generators: list[torch.Generator] = []
        for generator in request_generators:
            if isinstance(generator, list):
                if len(generator) != num_outputs_per_prompt:
                    raise ValueError(
                        "Per-request generator lists must match num_outputs_per_prompt, "
                        f"got {len(generator)} and {num_outputs_per_prompt}."
                    )
                generators.extend(generator)
            else:
                generators.extend([generator] * num_outputs_per_prompt)
        return generators
