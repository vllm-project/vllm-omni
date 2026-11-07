# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fish Audio's reference sampling for Drama 3, in plain PyTorch.

Opt in on stage 0 with ``hf_overrides`` ``fish_omni_reference_sampling: true``.
The main token's first draw uses epsilon sampling (Hewitt et al., 2022): with
``r`` a candidate's rank, ``p[r]`` its probability and ``cum[r]`` the inclusive
cumulative sum, rank ``r`` is kept when ``r < top_k and (r == 0 or (cum[r] <=
top_p and p[r] >= epsilon_cutoff))``; ``top_k < 1``, ``top_p >= 1`` and
``epsilon_cutoff <= 0`` turn their rule off. A first draw that repeats one of
the last 16 main tokens becomes the most probable candidate, or with
``fish_omni_reference_sampling_fallback: ras`` a draw at temperature 1.0,
top-p 0.95 and top-k 30; neither applies the cutoff. The codebook heads use the
request's temperature, top-k and top-p without the cutoff, on the whole row,
then keep only codes the codec's residual codebooks hold.

Every draw is an exponential race on the caller-owned uniform noise the model
state redraws before each step, so a seeded request stays reproducible and no
graph holds an RNG op.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_omni.model_executor.models.fish_omni.sampling import RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P
from vllm_omni.model_executor.models.fish_speech.dac_utils import DAC_RESIDUAL_CODEBOOK_SIZE

# Temperatures are clamped to this; 0 therefore samples greedily.
MIN_TEMPERATURE = 1e-5
# A request's epsilon_cutoff when it sets none; 0 turns the cutoff off.
DEFAULT_EPSILON_CUTOFF = 1e-3
FALLBACKS = ("argmax", "ras")


def reference_sampling_enabled(config: Any) -> bool:
    """``hf_overrides`` fish_omni_reference_sampling of stage 0 (default false)."""
    value = getattr(config, "fish_omni_reference_sampling", False)
    if not isinstance(value, bool):
        raise ValueError(f"fish_omni_reference_sampling={value!r}: needs true or false")
    return value


def reference_sampling_fallback(config: Any) -> str:
    """``hf_overrides`` fish_omni_reference_sampling_fallback (default argmax)."""
    value = getattr(config, "fish_omni_reference_sampling_fallback", "argmax")
    if value not in FALLBACKS:
        raise ValueError(f"fish_omni_reference_sampling_fallback={value!r}: needs one of {FALLBACKS}")
    return value


def resolve_epsilon_cutoff(value: Any, reference_sampling: bool) -> float:
    """A request's ``epsilon_cutoff`` (``extra_params``): in [0, 1), 1e-3 when unset under
    reference sampling, and refused when non-zero without it."""
    if value is None:
        return DEFAULT_EPSILON_CUTOFF if reference_sampling else 0.0
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= value < 1.0:
        raise ValueError(f"epsilon_cutoff must be in [0, 1), got {value!r}.")
    if value and not reference_sampling:
        raise ValueError(
            "epsilon_cutoff needs Drama 3 reference sampling: start stage 0 with "
            "hf_overrides fish_omni_reference_sampling: true"
        )
    return float(value)


def _per_row(value: float | torch.Tensor, rows: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.full((rows,), value, dtype=dtype, device=device)


def truncate(
    logits: torch.Tensor,
    temperature: float | torch.Tensor,
    top_k: int | torch.Tensor,
    top_p: float | torch.Tensor,
    epsilon_cutoff: float | torch.Tensor,
    legal_ids: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scaled logits sorted in descending order, the sort order, and the kept mask in that order.

    ``legal_ids``: only indices below it may be drawn. The rules run on the whole
    row first; then illegal indices drop and the most probable legal index stays.
    """
    rows, n = logits.shape
    device = logits.device
    temperature = _per_row(temperature, rows, torch.float32, device)
    top_k = _per_row(top_k, rows, torch.int64, device)
    top_p = _per_row(top_p, rows, torch.float32, device)
    epsilon_cutoff = _per_row(epsilon_cutoff, rows, torch.float32, device)

    scaled = logits.float() / temperature.clamp(min=MIN_TEMPERATURE).unsqueeze(-1)
    sorted_logits, order = torch.sort(scaled, dim=-1, descending=True, stable=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    rank = torch.arange(n, device=device).unsqueeze(0)

    within_top_k = (rank < top_k.unsqueeze(-1)) | (top_k < 1).unsqueeze(-1)
    within_top_p = (cumulative <= top_p.unsqueeze(-1)) | (top_p >= 1.0).unsqueeze(-1)
    above_cutoff = probs >= epsilon_cutoff.unsqueeze(-1)
    keep = within_top_k & within_top_p & above_cutoff
    if legal_ids is None:
        keep[:, 0] = True
    else:
        legal = order < legal_ids
        keep &= legal
        keep.scatter_(-1, legal.int().argmax(dim=-1, keepdim=True), True)
    return sorted_logits, order, keep


def truncated_sample(
    logits: torch.Tensor,
    temperature: float | torch.Tensor,
    top_k: int | torch.Tensor,
    top_p: float | torch.Tensor,
    epsilon_cutoff: float | torch.Tensor,
    uniform: torch.Tensor,
    legal_ids: int | None = None,
) -> torch.Tensor:
    """One index per row of ``logits`` ``[rows, n]``, drawn from the kept tokens in
    proportion to ``p``; ``uniform`` is ``[rows, >= n]`` U(0, 1) noise."""
    sorted_logits, order, keep = truncate(logits, temperature, top_k, top_p, epsilon_cutoff, legal_ids)
    kept = torch.softmax(sorted_logits.masked_fill(~keep, float("-inf")), dim=-1)
    # Exponential race: argmax of p_i / E_i with E_i = -log(U_i) ~ Exp(1) is a draw from p.
    noise = -torch.log(uniform[:, : kept.shape[-1]].clamp_min(1e-30))
    choice = torch.argmax(kept / noise, dim=-1, keepdim=True)
    return torch.gather(order, -1, choice).squeeze(-1)


class ReferenceFrameSampler:
    """``FusedFrameSampler``'s interface over the reference rules, with no top-k cap:
    the main token's draws span every candidate and the heads every code."""

    NUM_DRAWS_EXTRA = 2  # first and fallback main-token draws

    def __init__(self, *, fallback: str, num_codebooks: int, width: int) -> None:
        self.fallback = fallback
        self.num_draws = num_codebooks - 1 + self.NUM_DRAWS_EXTRA
        self.width = width

    def noise(self, rows: int, device: torch.device) -> torch.Tensor:
        return torch.rand((rows, self.num_draws, self.width), dtype=torch.float32, device=device)

    def main_tokens(
        self,
        candidate_logits: torch.Tensor,
        candidate_ids: torch.Tensor,
        ras_window: torch.Tensor,
        temperature: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        epsilon_cutoff: torch.Tensor,
        noise: torch.Tensor,
        semantic_begin_id: int,
        codebook_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        draw = candidate_ids[truncated_sample(candidate_logits, temperature, top_k, top_p, epsilon_cutoff, noise[:, 0])]
        if self.fallback == "argmax":
            alternative = candidate_ids[candidate_logits.float().argmax(dim=-1)]
        else:
            alternative = candidate_ids[
                truncated_sample(candidate_logits, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P, 0.0, noise[:, 1])
            ]
        repeated = (ras_window == draw.unsqueeze(-1)).any(dim=-1)
        main_tokens = torch.where(repeated, alternative, draw)
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
        return truncated_sample(
            logits,
            temperature,
            top_k,
            top_p,
            0.0,
            noise[:, self.NUM_DRAWS_EXTRA + head_index],
            legal_ids=DAC_RESIDUAL_CODEBOOK_SIZE,
        )


__all__ = [
    "DEFAULT_EPSILON_CUTOFF",
    "ReferenceFrameSampler",
    "reference_sampling_enabled",
    "reference_sampling_fallback",
    "resolve_epsilon_cutoff",
    "truncate",
    "truncated_sample",
]
