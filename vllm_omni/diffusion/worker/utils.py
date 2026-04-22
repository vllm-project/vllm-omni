# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request mutable state for step-wise diffusion execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
import contextlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType


@dataclass
class DiffusionRequestState:
    """Per-request mutable state across all pipeline stages.

    Owned by Runner and passed through all step-execution stages:
    ``prepare_encode()`` initializes/updates fields, ``denoise_step()`` and
    ``step_scheduler()`` mutate per-step fields, and ``post_decode()``
    consumes final latents. This state object is also the cache unit for
    future continuous batching.

    This dataclass keeps only the minimal cross-model state required by the
    step-execution contract. Pipeline-specific state should be stored in
    ``extra`` and promoted here only when it becomes shared across models.

    Examples:
    - Wan-style pipelines may keep ``condition``, ``first_frame_mask``, or
      ``image_embeds`` in ``extra``.
    - Bagel-style pipelines may keep ``gen_context``,
      ``cfg_text_context``, ``cfg_img_context``, or ``image_shape`` in
      ``extra``.
    """

    # ── Identity / request-level inputs ──
    req_id: str
    sampling: OmniDiffusionSamplingParams
    prompts: list[OmniPromptType] | None = None

    # ── Encoded prompts (set once by prepare_encode) ──
    prompt_embeds: torch.Tensor | None = None
    prompt_embeds_mask: torch.Tensor | None = None
    negative_prompt_embeds: torch.Tensor | None = None
    negative_prompt_embeds_mask: torch.Tensor | None = None

    # ── Latent state (mutated every step by step_scheduler) ──
    latents: torch.Tensor | None = None

    # ── Timestep schedule (set once by prepare_encode) ──
    timesteps: torch.Tensor | list[torch.Tensor] | None = None
    step_index: int = 0

    # ── Per-request scheduler instance (set once by prepare_encode) ──
    scheduler: Any | None = None

    # Active chunk index while inside a ``use_chunk`` context (None otherwise).
    current_chunk_idx: int | None = None

    # ── CFG config (set once by prepare_encode) ──
    do_true_cfg: bool = False
    guidance: torch.Tensor | None = None

    # ── Spatial / sequence metadata (set once by prepare_encode) ──
    img_shapes: list | None = None
    txt_seq_lens: list[int] | None = None
    negative_txt_seq_lens: list[int] | None = None

    # Pipeline-specific extras. Keep model-private fields here unless they
    # become part of the shared step-execution contract.
    # For example: Wan condition tensors / masks, or Bagel KV contexts.
    extra: dict[str, Any] = field(default_factory=dict)

    # ── Properties ──

    @property
    def current_timestep(self) -> torch.Tensor | None:
        if self.timesteps is None:
            return None
        if self.step_index >= self.total_steps:
            return None
        if isinstance(self.timesteps, torch.Tensor):
            if self.timesteps.ndim == 0:
                return self.timesteps
            return self.timesteps[self.step_index]
        return self.timesteps[self.step_index]

    @property
    def total_steps(self) -> int:
        if self.timesteps is None:
            return 0
        if isinstance(self.timesteps, torch.Tensor):
            if self.timesteps.ndim == 0:
                return 1
            return int(self.timesteps.shape[0])
        return len(self.timesteps)

    @property
    def denoise_completed(self) -> bool:
        total_steps = self.total_steps
        if total_steps == 0:
            return False
        return self.step_index >= total_steps

    @property
    def new_request(self) -> bool:
        # TODO: this is only an approximation for current stepwise mode.
        # A real "new request" signal should eventually come from scheduler/runner state transitions.
        return self.step_index == 0 or self.timesteps is None

    @contextlib.contextmanager
    def use_chunk(self, chunk: ChunkState) -> Iterator[None]:
        """Temporarily alias per-chunk fields on ``self`` to a ``ChunkState``'s view.

        Swapped fields: ``latents``, ``step_index``, ``scheduler``.

        Lets ``prepare_encode`` / ``denoise_step`` / ``step_scheduler`` operate
        per-chunk without any pipeline-side changes. Updates made inside the
        context are written back to the chunk on exit; the request-level fields
        are restored.
        """
        saved_latents = self.latents
        saved_step_index = self.step_index
        saved_scheduler = self.scheduler
        saved_chunk_idx = self.current_chunk_idx
        self.latents = chunk.latents
        self.step_index = chunk.step_index
        self.scheduler = chunk.scheduler
        self.current_chunk_idx = chunk.idx
        try:
            yield
        finally:
            chunk.latents = self.latents
            chunk.step_index = self.step_index
            chunk.scheduler = self.scheduler
            self.latents = saved_latents
            self.step_index = saved_step_index
            self.scheduler = saved_scheduler
            self.current_chunk_idx = saved_chunk_idx


@dataclass
class ChunkState:
    """Per-chunk state for one in-flight chunk of a streaming request.

    Lives inside ``DiffusionRequestState.extra["chunks"]`` (keyed by
    ``chunk_idx``). The runner swaps a chunk into the request state via
    ``state.use_chunk(chunk)`` for the duration of one micro-step's
    ``denoise_step + step_scheduler`` calls.

    Each chunk owns its own ``scheduler`` instance (deepcopied from the
    pipeline's scheduler by ``prepare_encode``) because multi-step ODE solvers
    (e.g. ``FlowUniPCMultistepScheduler``) are stateful — they track per-step
    ``model_outputs`` that must not leak between chunks.
    """

    idx: int
    latents: torch.Tensor | None = None
    step_index: int = 0
    scheduler: Any | None = None


class BaseRunnerOutput(ABC):
    @abstractmethod
    def get_req_output(self, sched_req_id: str) -> RunnerOutput | None:
        pass


@dataclass
class RunnerOutput:
    """Output of a single execution step for a request.

    Each scheduler reads the fields it needs:

    - ``StepScheduler`` reads ``step_index`` / ``finished``.
    - ``StreamBatchScheduler`` reads ``chunk_idx`` / ``step_index`` /
      ``chunk_completed`` / ``finished``.

    Fields not relevant to an execution path are left as ``None`` / ``False``.
    """

    req_id: str
    step_index: int | None = None
    finished: bool = False
    result: DiffusionOutput | None = None

    # ── Temporal-PP micro-step fields ──
    chunk_idx: int | None = None
    chunk_completed: bool = False

    def get_req_output(self, sched_req_id: str) -> RunnerOutput | None:
        return self if self.req_id == sched_req_id else None


@dataclass
class BatchRunnerOutput(BaseRunnerOutput):
    runner_outputs: list[RunnerOutput]
    _id_to_idx: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._id_to_idx = {out.req_id: i for i, out in enumerate(self.runner_outputs)}

    def __getitem__(self, req_id: str) -> RunnerOutput | None:
        """access single RunnerOutput by req_id"""
        idx = self._id_to_idx.get(req_id)
        return self.runner_outputs[idx] if idx is not None else None

    def get_req_output(self, sched_req_id: str) -> RunnerOutput | None:
        return self[sched_req_id]

    @property
    def req_ids(self) -> list[str]:
        return list(self._id_to_idx.keys())

    def __len__(self) -> int:
        return len(self.runner_outputs)

    @classmethod
    def from_list(cls, runner_output_list: list[RunnerOutput]) -> BatchRunnerOutput:
        return cls(runner_outputs=runner_output_list)