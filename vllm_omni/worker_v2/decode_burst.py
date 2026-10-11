# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Several one-token decode steps of a FULL-graph batch per scheduler step.

A small autoregressive stage (e.g. a codec Talker) decodes in well under a
millisecond on the GPU, but every scheduler step costs a host round trip:
schedule, prepare inputs, launch, sample, copy and publish the output. A
model can opt a uniform decode batch into a *burst*: after the scheduled
step, the runner replays the same FULL graph for further one-token steps.
Each step reads its inputs from the device state the previous step's
postprocess left (position, sampled token, the KV slots the scheduler
reserved as lookahead) and launches without a host sync, so the host
prepares step ``k + 1`` while the GPU runs step ``k``.

Every step is the scheduled step's exact computation for the rows that
continue. A row the model ends (e.g. a sampled stop token) stops advancing:
its later steps neither append a token nor advance its computed length, and
their outputs are dropped. The scheduler accepts the burst's tokens as one
step's output and advances the request past the KV the burst computed.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def decode_burst_steps(vllm_config: VllmConfig) -> int:
    """The stage processor's decode limit; ordinary stages reserve no lookahead.

    The scheduler and model must agree on the extra KV slots. Resolve the
    model-owned limit through the existing stage processor, before admission.
    """
    config = vllm_config.model_config
    if not getattr(config, "use_v2_model_runner", False) or getattr(config, "session_mode", "turn") != "duplex":
        return 1
    processor = getattr(config, "custom_process_next_stage_input_func", None)
    if not processor:
        return 1
    module = importlib.import_module(processor.rsplit(".", 1)[0])
    return int(getattr(module, "MRV2_DECODE_BURST_STEPS", 1))


class DecodeBurst(Protocol):
    """A model's plan for one burst, created after the scheduled step sampled."""

    #: Decode steps in this burst, the scheduled step included.
    steps: int

    def continues(self, sampled_token_ids: torch.Tensor) -> torch.Tensor:
        """[num_reqs] bool: rows that decode another step after this sample."""
        ...

    def record_step(self) -> None:
        """Called after each step's sampling and postprocess."""
        ...

    def merge_outputs(self, outputs: list[dict[str, Any]], live: list[torch.Tensor]) -> dict[str, Any]:
        """One multimodal payload with each request's steps contiguous on the token axis."""
        ...

    def finalizer(self, batch: BurstOutputBatch) -> Callable[..., Any] | None:
        """The output finalizer for the merged payload, or ``None``."""
        ...

    def finish(self, num_sampled_np: np.ndarray, copy_event: torch.cuda.Event) -> None:
        """Called once the output copy is enqueued; host counts are valid after ``copy_event``."""
        ...


@dataclass(frozen=True)
class BurstOutputBatch:
    """The batch layout ``OmniAsyncOutput`` and output finalizers slice a burst by."""

    num_reqs: int
    query_start_loc_np: np.ndarray
    num_scheduled_tokens: np.ndarray
    num_tokens_after_padding: int
    is_prefilling_np: np.ndarray

    @classmethod
    def uniform(cls, num_reqs: int, steps: int) -> BurstOutputBatch:
        return cls(
            num_reqs=num_reqs,
            query_start_loc_np=np.arange(num_reqs + 1, dtype=np.int32) * steps,
            num_scheduled_tokens=np.full(num_reqs, steps, dtype=np.int32),
            num_tokens_after_padding=num_reqs * steps,
            is_prefilling_np=np.zeros(num_reqs, dtype=bool),
        )


def burst_step_batch(input_batch: Any, step: int) -> Any:
    """The scheduled one-token batch, ``step`` tokens further along.

    Device positions, sequence lengths and input ids are rebuilt from runner
    state before each step; only the host-side lengths move here. They stay
    exact for continuing rows, so attention metadata matches a scheduled step.
    """
    num_reqs = input_batch.num_reqs
    upper_bound = input_batch.seq_lens_cpu_upper_bound.clone()
    upper_bound[:num_reqs] += step
    return replace(
        input_batch,
        num_computed_tokens_np=input_batch.num_computed_tokens_np + step,
        seq_lens_cpu_upper_bound=upper_bound,
    )
