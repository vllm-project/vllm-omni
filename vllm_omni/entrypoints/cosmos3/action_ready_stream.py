# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Publish completed action positions without changing the model's denoising state."""

from typing import Any
from urllib.parse import urlsplit

import requests
import torch

POLICY_PROFILE = "libero10_native_nonstream_v1"
BASELINE_PROFILE = "libero10_native_constant_v1"


class ActionReadyPublisher:
    """Observe the latent *after* each previous Euler step, before the next forward."""

    def __init__(self, callback_url: str, total_steps: int, action_count: int = 16) -> None:
        url = urlsplit(callback_url)
        if url.scheme not in ("http", "https") or not url.hostname or "/internal/action_events/" not in url.path:
            raise ValueError("Invalid policy action callback URL")
        self.callback_url = callback_url
        self.total_steps = total_steps
        self.action_count = action_count
        self.seen = torch.zeros(16, dtype=torch.bool)
        self.forward_count = 0
        self.session = requests.Session()
        self.session.trust_env = False

    def _publish(self, actions: torch.Tensor, ready: torch.Tensor, completed_steps: int) -> None:
        if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[1] != 16 or actions.shape[2] < 10:
            raise ValueError("Streaming requires [1,16,D>=10] native action latents")
        newly_ready = ready.detach().to(device="cpu") & ~self.seen
        newly_ready[self.action_count :] = False
        indices = torch.nonzero(newly_ready, as_tuple=False).flatten()
        if indices.numel() == 0:
            return
        values = actions[0].index_select(0, indices.to(actions.device))[:, :10].detach().float().cpu()
        response = self.session.post(
            self.callback_url,
            json={
                "type": "partial",
                "step": completed_steps,
                "total_steps": self.total_steps,
                "action_indices": indices.tolist(),
                "actions": values.tolist(),
                "policy_profile": POLICY_PROFILE,
                "action_stream_version": 1,
            },
            timeout=10,
        )
        response.raise_for_status()
        self.seen |= newly_ready

    def before_forward(self, _module: Any, _args: Any, kwargs: dict[str, Any]) -> None:
        times = kwargs.get("action_timesteps")
        actions = kwargs.get("action_latents")
        if times is None or times.shape != (1, 16) or actions is None:
            raise ValueError("Action streaming requires the aligned HAS transformer inputs")
        self._publish(actions, times[0] <= 1e-3, self.forward_count)
        self.forward_count += 1

    def after_step(self, step, total, chunks, newly_ready, cumulative_ready) -> None:
        """Publish native sampler callbacks immediately after the Euler update."""
        if total != self.total_steps or len(chunks) != 1:
            raise ValueError("Streaming requires one native action chunk")
        self._publish(chunks[0].unsqueeze(0), cumulative_ready[0], step + 1)

    def verify_complete(self) -> None:
        """A successful full-chunk stream must finalize every position in-loop."""
        if not bool(self.seen[: self.action_count].all()):
            raise RuntimeError("HAS denoising did not publish every completed action")

    def finish(self, actions: torch.Tensor) -> None:
        if self.forward_count != self.total_steps or not bool(self.seen.any()):
            raise RuntimeError("The backend did not publish actions during HAS denoising")
        self._publish(actions, torch.ones(16, dtype=torch.bool), self.total_steps)

    def close(self) -> None:
        self.session.close()
