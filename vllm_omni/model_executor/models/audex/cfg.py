# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# Adapted from nvidia/Nemotron-Labs-Audex-2B (Apache-2.0):
#   inference_scripts_vllm/audiogen_scripts/cfg_logits_processor.py
# Divergences from the official file are commented at the divergence site.
"""Classifier-free guidance for the Audex thinker stage.

CFG pairs a conditional request with an unconditional (null-prompt)
companion in the same engine and blends their logits every step:

    blended = uncond + cfg_scale * (cond - uncond)

Both rows receive the blended logits and the sampled token is copied from
the cond row to the uncond row, so the two sequences stay token-identical.
Requests opt in via ``SamplingParams.extra_args``:

    cond:   {"cfg_scale": 1.5, "cfg_role": "cond",   "cfg_pair_id": <id>}
    uncond: {"cfg_scale": 1.5, "cfg_role": "uncond", "cfg_pair_id": <id>}

Blending is only correct when both pair members decode the same position in
the same engine step. Keeping the pair step-locked is model-neutral and lives
in :mod:`vllm_omni.model_executor.models.common.cfg_pairing`; the scheduler
patches are re-exported here so existing imports keep working.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.logits_processor import LogitsContext, LogitsProcessor, LogitsProcRequestState

from vllm_omni.model_executor.models.common.cfg_pairing import (
    _COND,
    _UNCOND,
    apply_cfg_patches,
    cfg_patches_applied,
)
from vllm_omni.worker_v2.omni_sampler import OmniSampler

logger = init_logger(__name__)

__all__ = [
    "AudexCFGLogitsProcessor",
    "apply_cfg_patches",
    "cfg_patches_applied",
]


class AudexCFGLogitsProcessor(LogitsProcessor):
    """Blend only the active cond/uncond rows, keyed by persistent slots."""

    @classmethod
    def validate_params(cls, params: SamplingParams) -> None:
        extra_args = params.extra_args or {}
        role = extra_args.get("cfg_role")
        if role is not None and role not in (_COND, _UNCOND):
            raise ValueError(f"cfg_role must be 'cond' or 'uncond', got '{role}'")
        scale = extra_args.get("cfg_scale")
        if scale is not None and (not isinstance(scale, int | float) or scale < 1.0):
            raise ValueError(f"cfg_scale must be >= 1.0, got {scale}")

    def __init__(self, vllm_config: VllmConfig, req_states: LogitsProcRequestState) -> None:
        self._info: dict[int, dict[str, Any]] = {}
        self._speculative = vllm_config.speculative_config is not None

    def add_request(self, req_idx: int, params: SamplingParams) -> bool:
        self._info.pop(req_idx, None)
        extra = params.extra_args or {}
        if extra.get("cfg_role") not in (_COND, _UNCOND):
            return False
        if self._speculative:
            raise ValueError("Audex CFG requires non-speculative decoding")
        self._info[req_idx] = {
            "role": extra["cfg_role"],
            "cfg_scale": float(extra.get("cfg_scale", 1.0)),
            "pair_id": extra.get("cfg_pair_id"),
        }
        return True

    def pairs(self, idx_mapping_np) -> list[tuple[int, int, float]]:
        by_pair: dict[str, dict[str, tuple[int, float]]] = {}
        for row, slot in enumerate(idx_mapping_np):
            info = self._info.get(int(slot))
            if info is not None and info["pair_id"] is not None:
                by_pair.setdefault(info["pair_id"], {})[info["role"]] = (row, info["cfg_scale"])
        return [
            (roles[_COND][0], roles[_UNCOND][0], roles[_COND][1])
            for roles in by_pair.values()
            if _COND in roles and _UNCOND in roles
        ]

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        for cond_row, uncond_row, scale in self.pairs(ctx.idx_mapping_np):
            allowed = torch.isfinite(logits[cond_row]) & torch.isfinite(logits[uncond_row])
            blended = logits[uncond_row] + scale * (logits[cond_row] - logits[uncond_row])
            blended.masked_fill_(~allowed, -float("inf"))
            logits[cond_row] = blended
            logits[uncond_row] = blended
        return logits


class AudexCFGSampler(OmniSampler):
    """Synchronize sampled pair tokens through the existing model sampler seam."""

    def __call__(self, logits, input_batch):
        output = self.base_sampler(logits, input_batch)
        for processor in self.base_sampler.logits_processors:
            if isinstance(processor, AudexCFGLogitsProcessor):
                for cond_row, uncond_row, _ in processor.pairs(input_batch.idx_mapping_np):
                    output.sampled_token_ids[uncond_row] = output.sampled_token_ids[cond_row]
        return output
