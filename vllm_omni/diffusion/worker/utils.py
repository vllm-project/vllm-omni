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


def clear_pipeline_stage_durations(pipeline: Any) -> None:
    clear_records = getattr(pipeline, "clear_profiler_records", None)
    if getattr(pipeline, "enable_diffusion_pipeline_profiler", False) and callable(clear_records):
        clear_records()


def consume_pipeline_stage_durations(pipeline: Any) -> dict[str, float]:
    if not getattr(pipeline, "enable_diffusion_pipeline_profiler", False):
        return {}
    stage_durations = getattr(pipeline, "stage_durations", None)
    if not isinstance(stage_durations, dict):
        return {}
    result = {stage: float(duration) for stage, duration in stage_durations.items()}
    clear_pipeline_stage_durations(pipeline)
    return result


def merge_stage_durations(
    state: DiffusionRequestState,
    stage_durations: dict[str, float],
) -> None:
    if not stage_durations:
        return
    for stage, duration in stage_durations.items():
        state.stage_durations[stage] = float(state.stage_durations.get(stage, 0.0)) + float(duration)


def attach_stage_durations(
    state: DiffusionRequestState,
    output: DiffusionOutput,
) -> None:
    if state.stage_durations:
        output.stage_durations = dict(state.stage_durations)


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
    request_id: str
    sampling: OmniDiffusionSamplingParams
    prompt: OmniPromptType | None = None
    kv_sender_info: dict | None = None

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

    # ── Optional chunked streaming progress ──
    chunk_index: int = 0
    step_in_chunk: int = 0
    total_chunks: int = 1
    chunk_num_steps: int | None = None

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

    # ── Runner-owned profiling metadata ──
    stage_durations: dict[str, float] = field(default_factory=dict)

    # Peak device memory observed while this request is active in step mode.
    peak_memory_mb: float = 0.0

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
    def chunk_denoise_completed(self) -> bool:
        if self.chunk_num_steps is None:
            return False
        return self.step_in_chunk >= self.chunk_num_steps

    @property
    def request_denoise_completed(self) -> bool:
        if self.chunk_num_steps is None:
            return self.denoise_completed
        return self.chunk_index >= self.total_chunks


class BaseRunnerOutput(ABC):
    @abstractmethod
    def get_request_output(self, request_id: str) -> RunnerOutput | None:
        pass


@dataclass
class RunnerOutput(BaseRunnerOutput):
    """Output of a single denoising step for a request.

    NOTE: `latents` may be None when returned through IPC to avoid
    serialization overhead. The actual latents are kept in Worker's
    _request_state_cache.
    """

    request_id: str
    step_index: int | None = None
    finished: bool = False
    result: DiffusionOutput | None = None

    def get_request_output(self, request_id: str) -> RunnerOutput | None:
        return self if self.request_id == request_id else None


@dataclass
class BatchRunnerOutput(BaseRunnerOutput):
    runner_outputs: list[RunnerOutput]
    _id_to_idx: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._id_to_idx = {out.request_id: i for i, out in enumerate(self.runner_outputs)}

    def __getitem__(self, request_id: str) -> RunnerOutput | None:
        """access single RunnerOutput by request_id"""
        idx = self._id_to_idx.get(request_id)
        return self.runner_outputs[idx] if idx is not None else None

    def get_request_output(self, request_id: str) -> RunnerOutput | None:
        return self[request_id]

    @property
    def request_ids(self) -> list[str]:
        return list(self._id_to_idx.keys())

    def __len__(self) -> int:
        return len(self.runner_outputs)

    @classmethod
    def from_list(cls, runner_output_list: list[RunnerOutput]) -> BatchRunnerOutput:
        return cls(runner_outputs=runner_output_list)
