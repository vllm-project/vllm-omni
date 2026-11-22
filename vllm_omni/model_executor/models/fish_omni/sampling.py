# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Drama 3 frame sampling: top-k / top-p heads and repetition-aware semantic draws.

Every op here is a fixed-shape tensor op without host synchronization, so a
decode step that samples a whole frame can be captured in one CUDA graph.
Randomness comes from caller-owned noise (see ``FusedFrameSampler.noise``).
"""

from __future__ import annotations

import torch

from vllm_omni.model_executor.models.fish_omni.kernels import draw_top_k_top_p

# RAS settings are fixed for Drama 3, not request parameters: when a semantic
# first draw repeats a main token of the last 16 frames, the frame takes a
# second draw with these parameters instead.
RAS_WINDOW_FRAMES = 16
RAS_TEMPERATURE = 1.0
RAS_TOP_K = 30
RAS_TOP_P = 0.95
CODEBOOK_TOP_K = 30
# Pads the window of a request with fewer frames; it matches no token id.
EMPTY_RAS_SLOT = -1
DEFAULT_REFERENCE_TOP_P = 0.9


def parse_reference_top_p(value: object) -> float | None:
    """Return a reference nucleus probability for a JSON number in (0, 1]."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < value <= 1:
        return float(value)
    return None


class FusedFrameSampler:
    """The main token uses the request's top-k; codebook heads keep at most 30.

    The two semantic draws share their candidates. Randomness is one
    ``torch.rand`` per frame, so CUDA graph replays draw fresh noise.
    """

    def __init__(self, *, max_top_k: int, num_codebooks: int, max_rows: int, device: torch.device) -> None:
        self.max_top_k = max_top_k
        # Each row contains the main draw, the RAS redraw, then the codebook draws.
        self.noise_width = max_top_k + RAS_TOP_K + (num_codebooks - 1) * CODEBOOK_TOP_K
        self.ras_temperature = torch.full((max_rows,), RAS_TEMPERATURE, dtype=torch.float32, device=device)
        self.ras_top_k = torch.full((max_rows,), RAS_TOP_K, dtype=torch.int64, device=device)
        self.ras_top_p = torch.full((max_rows,), RAS_TOP_P, dtype=torch.float32, device=device)

    def noise(self, rows: int, device: torch.device) -> torch.Tensor:
        return torch.rand((rows, self.noise_width), dtype=torch.float32, device=device)

    def main_tokens(
        self,
        candidate_logits: torch.Tensor,
        candidate_ids: torch.Tensor,
        ras_window: torch.Tensor,
        temperature: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        noise: torch.Tensor,
        semantic_begin_id: int,
        codebook_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = candidate_logits.shape[0]
        top = torch.topk(candidate_logits, self.max_top_k, dim=-1)
        first = draw_top_k_top_p(
            candidate_logits,
            temperature,
            top_k,
            top_p,
            noise[:, : self.max_top_k],
            self.max_top_k,
            candidate_ids,
            top,
            keep_top_p_crossing=True,
        )
        ras = draw_top_k_top_p(
            candidate_logits,
            self.ras_temperature[:rows],
            self.ras_top_k[:rows],
            self.ras_top_p[:rows],
            noise[:, self.max_top_k : self.max_top_k + RAS_TOP_K],
            RAS_TOP_K,
            candidate_ids,
            (top.values[:, :RAS_TOP_K], top.indices[:, :RAS_TOP_K]),
            keep_top_p_crossing=True,
        )
        is_repeat = (ras_window == first.unsqueeze(-1)).any(dim=-1)
        is_semantic = (first >= semantic_begin_id) & (first < semantic_begin_id + codebook_size)
        main_tokens = torch.where(is_repeat & is_semantic, ras, first)
        code_zero = (main_tokens - semantic_begin_id).clamp(0, codebook_size - 1)
        return main_tokens, code_zero

    def code(
        self,
        logits: torch.Tensor,
        temperature: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        noise: torch.Tensor,
        head_index: int,
    ) -> torch.Tensor:
        start = self.max_top_k + RAS_TOP_K + head_index * CODEBOOK_TOP_K
        return draw_top_k_top_p(
            logits, temperature, top_k, top_p, noise[:, start : start + CODEBOOK_TOP_K], CODEBOOK_TOP_K
        )


__all__ = ["EMPTY_RAS_SLOT", "RAS_TOP_K", "RAS_WINDOW_FRAMES", "FusedFrameSampler"]
