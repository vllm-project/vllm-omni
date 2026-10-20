# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Alpamayo 2 Super reasoning stage: Qwen3-VL chain-of-thought up to the trajectory.

The Qwen3-VL decoder runs natively in vLLM and generates the driving
reasoning. When it emits ``<|traj_future_start|>``, the next forward writes
that token's KV and ``compute_logits`` forces ``<|traj_future_end|>``, so the
request stops with its prefix complete. The request's KV is then handed to the
expert stage (``Alpamayo2SuperPipeline``) with the prefix length and the
trigger token's M-RoPE position as KV-transfer metadata.
"""

from __future__ import annotations

import itertools
from collections import deque
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
    Qwen3VLProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

# Steps kept per request: the trigger step plus any async-scheduled steps after it.
_RECENT_STEPS = 4
# A request not scheduled for this many steps is forgotten (aborted or preempted);
# a finished request is read well before that.
_STALE_STEPS = 64


class Alpamayo2SuperProcessingInfo(Qwen3VLProcessingInfo):
    """Resize camera frames to the pixel budget the checkpoint was trained with."""

    def get_hf_processor(self, **kwargs: object) -> Qwen3VLProcessor:
        config = self.get_hf_config()
        return self.ctx.get_hf_processor(
            Qwen3VLProcessor,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            **kwargs,
        )


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Alpamayo2SuperProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Alpamayo2SuperForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Reasoning stage of the Alpamayo 2 Super policy.

    Every request is a trajectory request: it must end with
    ``<|traj_future_start|>``, so text EOS and the trajectory vocabulary are
    never sampled. The request stops on ``<|traj_future_end|>``.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        self.future_start_id = int(config.traj_ids["future_start"])
        self.future_end_id = int(config.traj_ids["future_end"])
        self.traj_vocab_start = min(int(config.traj_ids["history_id0"]), int(config.traj_ids["future_id0"]))
        self.traj_vocab_size = int(config.traj_vocab_size)

        eos_ids = vllm_config.model_config.try_get_generation_config().get("eos_token_id")
        if eos_ids is None:
            eos_ids = config.text_config.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        self.text_eos_ids = torch.tensor(
            [int(token) for token in eos_ids or () if int(token) != self.future_start_id], dtype=torch.long
        )

        # Whether each row of this step fed back future_start, on the GPU; compute_logits ends those rows.
        self._step_triggers: torch.Tensor | None = None
        # req_id -> the request's last few steps, newest last: (step, first input ids of the
        # step, their (3, rows) positions, row, prefix length). Read when the request
        # finishes, so detecting the trigger never waits for the GPU.
        self._recent_steps: dict[str, deque[tuple[int, torch.Tensor, torch.Tensor, int, int]]] = {}
        self._step = 0

    def prepare_runner_inputs(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        req_ids: Sequence[str],
        num_computed_tokens: Sequence[int],
        num_scheduled_tokens: Sequence[int],
        input_ids_buffer: torch.Tensor,
        **_: object,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        # future_start is sampled at the end of one step and fed as the first
        # input of the next, whose forward writes its KV.
        token_ids = input_ids if input_ids is not None else input_ids_buffer
        counts = [int(n) for n in num_scheduled_tokens]
        starts = list(itertools.accumulate(counts, initial=0))[:-1]
        index = torch.tensor(starts, dtype=torch.long).to(token_ids.device, non_blocking=True)
        first_ids = token_ids[index]
        first_positions = positions[:, index] if positions.ndim == 2 else positions[index].expand(3, -1)
        self._step_triggers = first_ids == self.future_start_id
        self._step += 1
        for row, req_id in enumerate(req_ids):
            if counts[row] > 0:
                steps = self._recent_steps.setdefault(req_id, deque(maxlen=_RECENT_STEPS))
                steps.append((self._step, first_ids, first_positions, row, int(num_computed_tokens[row]) + counts[row]))
        for req_id in [r for r, steps in self._recent_steps.items() if steps[-1][0] < self._step - _STALE_STEPS]:
            del self._recent_steps[req_id]
        return input_ids, positions

    def get_kv_transfer_metadata(self, req_id: str, *, num_computed_tokens: int | None = None) -> dict[str, Any] | None:
        """Prefix length and ``<|traj_future_start|>`` position, sent with the request's KV.

        ``None`` when the request ended without the trigger (for example at
        ``max_tokens``); the expert stage then reports an error for it.
        """
        # Async scheduling may run one more step after the trigger before the
        # request is known to be finished, so search back from the newest step.
        for _, first_ids, first_positions, row, prefix_len in reversed(self._recent_steps.pop(req_id, ())):
            if int(first_ids[row]) == self.future_start_id:
                return {
                    "prefix_len": prefix_len,
                    "trigger_position": [int(p) for p in first_positions[:, row].tolist()],
                }
        return None

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = super().compute_logits(hidden_states)
        if logits is None:
            return None
        # Trajectory tokens are produced by the expert stage, never sampled,
        # and reasoning must end with future_start rather than a text EOS.
        logits[..., self.traj_vocab_start : self.traj_vocab_start + self.traj_vocab_size] = -torch.inf
        if self.text_eos_ids.numel():
            logits[..., self.text_eos_ids.to(logits.device, non_blocking=True)] = -torch.inf
        triggers, self._step_triggers = self._step_triggers, None
        if triggers is not None and triggers.shape[0] == logits.shape[0]:
            # Rows that fed back future_start end with future_end.
            forced = torch.full_like(logits[:1], -torch.inf)
            forced[:, self.future_end_id] = 0
            logits = torch.where(triggers[:, None], forced, logits)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        vlm_weights: list[tuple[str, torch.Tensor]] = []
        for name, weight in weights:
            if name.startswith("vlm."):
                vlm_weights.append((name.removeprefix("vlm."), weight))
            elif not name.startswith("expert."):  # the expert stage loads expert.*
                raise ValueError(f"Unexpected Alpamayo 2 Super weight {name!r}: expected a 'vlm.' or 'expert.' prefix")
        return super().load_weights(vlm_weights)
