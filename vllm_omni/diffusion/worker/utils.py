# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request mutable state for step-wise diffusion execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
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


class BaseRunnerOutput(ABC):
    @abstractmethod
    def get_req_output(self, sched_req_id: str) -> RunnerOutput | None:
        pass


@dataclass
class RunnerOutput(BaseRunnerOutput):
    """Output of a single denoising step for a request.

    NOTE: `latents` may be None when returned through IPC to avoid
    serialization overhead. The actual latents are kept in Worker's
    _request_state_cache.
    """

    req_id: str
    step_index: int | None = None
    finished: bool = False
    result: DiffusionOutput | None = None

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
