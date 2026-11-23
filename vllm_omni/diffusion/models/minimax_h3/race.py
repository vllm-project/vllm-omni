# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-local spectral refresh, SGVE, and RAIC for MiniMax H3 T2VA."""

from collections import deque

import torch
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger

from vllm_omni.diffusion.cache.seacache.sea_filter import apply_sea_filter
from vllm_omni.diffusion.layers.indexed_modulation import rms_norm_indexed_scale_shift

from .minimax_h3_blocks import MINIMAX_H3_ADALN_MODALITY_NUM

logger = init_logger(__name__)


class RACETrajectory:
    """Keep the two most recent true FP32 state and velocity pairs."""

    def __init__(self) -> None:
        self.history: deque[tuple[torch.Tensor, torch.Tensor]] = deque(maxlen=2)
        self.secant: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self.anchor_time = 0.0
        self.sum_h = 0.0
        self.sum_time_h = 0.0
        self.reused = False

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        anchor_state, anchor_velocity = self.history[-1]
        if len(self.history) == 1:
            return anchor_velocity
        if self.secant is None:
            old_state, old_velocity = self.history[0]
            delta_state = old_state - anchor_state
            delta_velocity = old_velocity - anchor_velocity
            self.secant = (delta_state, delta_velocity, delta_state.square().sum())
        delta_state, delta_velocity, gram = self.secant
        rhs = (delta_state * (state - anchor_state)).sum()
        coefficient = torch.where(gram != 0, rhs / gram, torch.zeros_like(rhs))
        return anchor_velocity + coefficient * delta_velocity

    def step(
        self, state: torch.Tensor, velocity: torch.Tensor | None, time: float, next_time: float
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = next_time - time
        correction = None
        if velocity is None:
            velocity = self.predict(state)
            self.reused = True
        else:
            if self.reused:
                weight = (self.sum_time_h - self.anchor_time * self.sum_h) / (time - self.anchor_time)
                correction = (velocity - self.predict(state)) * weight
            # The sampler assigns new row tensors on every step. Keep snapshots
            # here so future changes to its update strategy cannot alter history.
            self.history.append((state.clone(), velocity.clone()))
            self.secant = None
            self.anchor_time = time
            self.sum_h = 0.0
            self.sum_time_h = 0.0
            self.reused = False
        self.sum_h += h
        self.sum_time_h += time * h
        return velocity, correction


class MiniMaxH3RACE:
    threshold = 0.3

    def __init__(self, *, model, branch, num_steps: int) -> None:
        self.model = model
        self.num_steps = num_steps
        self.video = RACETrajectory()
        self.audio = RACETrajectory()
        self.previous = None
        self.accumulated = 0.0
        self.refresh_steps: list[int] = []
        self.tp_group = get_tp_group()
        self.is_gate_rank = self.tp_group.rank_in_group == 0
        self.flags = torch.empty(1, dtype=torch.int32, device=branch.device) if self.tp_group.world_size > 1 else None

        layout = branch.static_kwargs["video_token_layout"]
        if layout.video_spans:
            (target,) = (span for span in layout.video_spans if span.role == "target")
            self.grid = target.latent_grid
        else:
            self.grid = layout.latent_grid
        self.video_positions = branch.img_pos_dev[branch.update_mask_dev]

    def should_refresh(self, step: int, sigma: float, forward_kwargs: dict) -> bool:
        probe = self.probe(forward_kwargs)
        refresh = True
        if self.is_gate_rank:
            assert probe is not None
            refresh = self.decide(probe, step, sigma)
        if self.flags is not None and step not in (0, self.num_steps - 1):
            # Keep every rank on the same path through the DiT collectives.
            if self.is_gate_rank:
                self.flags.fill_(int(refresh))
            refresh = bool(self.tp_group.broadcast(self.flags).item())
        if refresh:
            self.refresh_steps.append(step)
        return refresh

    def decide(self, probe: torch.Tensor, step: int, sigma: float) -> bool:
        """Match the H3 reference's raw endpoints and W, H, T FFT order."""
        if step in (0, self.num_steps - 1):
            self.previous = probe.detach()
            self.accumulated = 0.0
            return True
        previous = self.previous
        assert previous is not None
        metric = apply_sea_filter(probe, sigma, dimensions=(2, 1, 0))
        distance = (metric - previous).abs().mean() / (previous.abs().mean() + 1e-16)
        self.accumulated += float(distance)
        refresh = not self.accumulated < self.threshold
        if refresh:
            self.accumulated = 0.0
        self.previous = metric.detach()
        return refresh

    def probe(self, forward_kwargs: dict) -> torch.Tensor | None:
        model = self.model
        video = forward_kwargs["x"][0].index_select(0, self.video_positions)
        # These projections all participate in TP collectives.
        video_embed, _ = model.video_patch_proj(video)
        time_embed = model.time_embedder(forward_kwargs["unique_timesteps"])
        block = model.blocks[0]
        shift, scale, *_ = block.adaln_proj(time_embed)
        if not self.is_gate_rank:
            return None
        time_indices = forward_kwargs["inverse_indices"].index_select(0, self.video_positions)
        tags = forward_kwargs["token_tags"].index_select(0, self.video_positions)
        indices = time_indices * MINIMAX_H3_ADALN_MODALITY_NUM + tags
        norm = block.norm1
        return rms_norm_indexed_scale_shift(
            video_embed.to(torch.bfloat16), norm.weight, shift, scale, indices, norm.variance_epsilon
        ).view(*self.grid, model.hidden_size)

    def log_result(self) -> None:
        if self.is_gate_rank:
            logger.info(
                "MiniMax H3 RACE threshold=%.2f full_steps=%d/%d refresh_steps=%s",
                self.threshold,
                len(self.refresh_steps),
                self.num_steps,
                self.refresh_steps,
            )
