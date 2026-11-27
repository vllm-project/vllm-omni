# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Voxtral request state that survives MRV2 slot preemption."""

from typing import Any

import torch
from vllm.v1.core.sched.output import NewRequestData

from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState


class VoxtralTTSModelState(OmniModelState):
    """Keep request-local acoustic feedback needed to replay a preempted request."""

    def __init__(self, vllm_config: Any, model: Any, encoder_cache: Any, device: torch.device) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        self._suspended_feedback_history: dict[str, torch.Tensor] = {}

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        if getattr(self.model, "model_stage", None) != "audio_generation":
            return

        history = self._suspended_feedback_history.pop(new_req_data.req_id, None)
        generated_count = self._generated_token_count(new_req_data)
        if history is None:
            if generated_count:
                raise RuntimeError(
                    f"Voxtral request {new_req_data.req_id} resumes with {generated_count} generated tokens "
                    "but no acoustic feedback history was saved"
                )
            return
        if generated_count == 0:
            return
        if history.shape[0] < generated_count:
            raise RuntimeError(
                f"Voxtral request {new_req_data.req_id} resumes with {generated_count} generated tokens "
                f"but only {history.shape[0]} acoustic feedback frames were saved"
            )

        # The scheduler may have sampled a token whose payload was not accepted
        # before preemption. Only restore frames paired with replayed token IDs.
        history = history[:generated_count]
        request_info = self.intermediate_buffer.buffers[req_index]
        codes = request_info.setdefault("codes", {})
        if not isinstance(codes, dict):
            raise TypeError(f"Voxtral codes state must be a dict, got {type(codes).__name__}")
        codes["audio_history"] = list(history.unbind(0))
        codes["audio"] = history[-1].reshape(1, 1, -1)

    @staticmethod
    def _generated_token_count(new_req_data: NewRequestData) -> int:
        prompt_token_ids = new_req_data.prompt_token_ids or []
        prefill_token_ids = new_req_data.prefill_token_ids
        if prefill_token_ids is None:
            prefill_token_ids = prompt_token_ids
        count = len(prefill_token_ids) - len(prompt_token_ids)
        if count < 0:
            raise ValueError(
                f"Voxtral request {new_req_data.req_id} has {len(prefill_token_ids)} prefill tokens "
                f"but {len(prompt_token_ids)} prompt tokens"
            )
        return count

    def on_request_preempted(self, req_id: str, req_index: int) -> None:
        super().on_request_preempted(req_id, req_index)
        if getattr(self.model, "model_stage", None) != "audio_generation":
            return
        if req_index >= len(self.intermediate_buffer.buffers):
            return

        request_info = self.intermediate_buffer.buffers[req_index]
        if request_info.get("req_id") != req_id:
            return
        codes = request_info.get("codes")
        if not isinstance(codes, dict):
            self._suspended_feedback_history.pop(req_id, None)
            return

        frames = codes.get("audio_history")
        if isinstance(frames, torch.Tensor):
            if frames.numel() == 0:
                self._suspended_feedback_history.pop(req_id, None)
                return
            if frames.ndim == 1:
                frames = frames.unsqueeze(0)
            else:
                frames = frames.reshape(frames.shape[0], -1)
            history = frames.detach().to(device="cpu", copy=True)
        elif isinstance(frames, (list, tuple)):
            rows = [frame.detach().reshape(-1) for frame in frames if isinstance(frame, torch.Tensor)]
            if not rows:
                self._suspended_feedback_history.pop(req_id, None)
                return
            # Batch host transfers by device so a long request does not issue
            # one synchronous D2H copy for every tiny acoustic frame.
            history = torch.empty(
                (len(rows), rows[0].numel()),
                dtype=rows[0].dtype,
                device="cpu",
            )
            groups: dict[torch.device, list[tuple[int, torch.Tensor]]] = {}
            for row_index, frame in enumerate(rows):
                groups.setdefault(frame.device, []).append((row_index, frame))
            for source_device, group in groups.items():
                row_indices = [row for row, _ in group]
                batch = torch.stack([frame for _, frame in group])
                if source_device.type != "cpu":
                    batch = batch.to(device="cpu", copy=True)
                history.index_copy_(
                    0,
                    torch.tensor(row_indices, dtype=torch.long, device="cpu"),
                    batch,
                )
        else:
            # Keep a one-frame fallback for buffers created before history was
            # initialized; resume will fail loudly if token and frame counts do
            # not line up.
            frame = codes.get("audio")
            if not isinstance(frame, torch.Tensor):
                self._suspended_feedback_history.pop(req_id, None)
                return
            history = frame.detach().reshape(1, -1).to(device="cpu", copy=True)

        # The owned CPU copy outlives remove_request clearing this runner slot.
        self._suspended_feedback_history[req_id] = history

    def on_requests_finished(self, req_ids: set[str]) -> None:
        super().on_requests_finished(req_ids)
        for req_id in req_ids:
            self._suspended_feedback_history.pop(req_id, None)
