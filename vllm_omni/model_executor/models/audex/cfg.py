# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

from vllm_omni.model_executor.models.common.cfg_pairing import (
    _COND,
    _UNCOND,
    apply_cfg_patches,
    cfg_patches_applied,
)

logger = init_logger(__name__)

__all__ = [
    "AudexCFGLogitsProcessor",
    "apply_cfg_patches",
    "cfg_patches_applied",
]


class AudexCFGLogitsProcessor(LogitsProcessor):
    """Blend cond/uncond logits for classifier-free guidance.

    Pairs are matched by ``cfg_pair_id``. For each pair the blended logits
    are written to *both* rows so the sampler picks the same token; the
    post-sampling copy of the cond token into the uncond slot is installed
    by :meth:`_ensure_sample_patched` (which must run inside each worker
    process, hence from ``__init__`` rather than a main-process patch).
    """

    _sample_patched = False

    @classmethod
    def validate_params(cls, params: SamplingParams) -> None:
        extra_args = params.extra_args
        if not extra_args:
            return
        role = extra_args.get("cfg_role")
        if role is not None and role not in (_COND, _UNCOND):
            raise ValueError(f"cfg_role must be 'cond' or 'uncond', got '{role}'")
        scale = extra_args.get("cfg_scale")
        if scale is not None and (not isinstance(scale, int | float) or scale < 1.0):
            raise ValueError(f"cfg_scale must be >= 1.0, got {scale}")

    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool) -> None:
        self._info: dict[int, dict[str, Any]] = {}
        self._output_tokens: dict[int, list[int]] = {}
        self._pairs: list[tuple[int, int, float]] = []
        self._dirty = True
        self._ensure_sample_patched()

    @classmethod
    def _ensure_sample_patched(cls) -> None:
        """Install the post-sampling cond→uncond token copy (once per process)."""
        if cls._sample_patched:
            return
        cls._sample_patched = True

        # Divergence from the official file: it patches vLLM's
        # ``GPUModelRunner._sample``, but vllm-omni's AR runner overrides
        # ``_sample``, so the base-class patch would never run for the Audex
        # thinker stage. Patch the omni AR runner instead.
        from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

        orig_sample = GPUARModelRunner._sample

        def _sample_with_cfg_sync(self, logits, spec_decode_metadata):
            sampler_output = orig_sample(self, logits, spec_decode_metadata)
            for proc in self.input_batch.logitsprocs.all:
                if isinstance(proc, AudexCFGLogitsProcessor) and proc._pairs:
                    sampled = sampler_output.sampled_token_ids
                    num_rows = sampled.shape[0] if hasattr(sampled, "shape") else len(sampled)
                    for cond_idx, uncond_idx, _ in proc._pairs:
                        # Steps that sample fewer rows than the persistent
                        # batch (pure-prefill / partially scheduled steps)
                        # have nothing to sync for this pair; the scheduler
                        # pair patches keep both members step-locked on the
                        # decode steps that matter.
                        if cond_idx < num_rows and uncond_idx < num_rows:
                            sampled[uncond_idx] = sampled[cond_idx]
                    break
            return sampler_output

        _sample_with_cfg_sync._audex_cfg_sync = True  # type: ignore[attr-defined]
        GPUARModelRunner._sample = _sample_with_cfg_sync
        logger.info("AudexCFGLogitsProcessor: patched GPUARModelRunner._sample for CFG token sync")

    def is_argmax_invariant(self) -> bool:
        return False

    def _reset(self) -> None:
        self._info.clear()
        self._output_tokens.clear()
        self._pairs.clear()
        self._dirty = True

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if batch_update is None:
            return

        for idx in batch_update.removed:
            self._info.pop(idx, None)
            self._output_tokens.pop(idx, None)

        if not self._info and batch_update.added:
            self._reset()

        for idx, params, _, output_token_ids in batch_update.added:
            extra_args = params.extra_args if params else None
            if extra_args and extra_args.get("cfg_role") in (_COND, _UNCOND):
                self._info[idx] = {
                    "role": extra_args["cfg_role"],
                    "cfg_scale": float(extra_args.get("cfg_scale", 1.0)),
                    "pair_id": extra_args.get("cfg_pair_id"),
                }
                self._output_tokens[idx] = output_token_ids
            else:
                self._info.pop(idx, None)
                self._output_tokens.pop(idx, None)
        self._dirty = True

        if self._info:
            for src, dst, direction in batch_update.moved:
                src_info = self._info.pop(src, None)
                dst_info = self._info.pop(dst, None)
                src_tokens = self._output_tokens.pop(src, None)
                dst_tokens = self._output_tokens.pop(dst, None)
                if src_info is not None:
                    self._info[dst] = src_info
                if src_tokens is not None:
                    self._output_tokens[dst] = src_tokens
                if direction == MoveDirectionality.SWAP:
                    if dst_info is not None:
                        self._info[src] = dst_info
                    if dst_tokens is not None:
                        self._output_tokens[src] = dst_tokens
            self._dirty = True

    def _rebuild_pairs(self) -> None:
        by_pair: dict[str, dict[str, tuple[int, float]]] = {}
        for idx, info in self._info.items():
            pair_id = info.get("pair_id")
            if pair_id is None:
                continue
            by_pair.setdefault(pair_id, {})[info["role"]] = (idx, info["cfg_scale"])

        self._pairs = [
            (roles[_COND][0], roles[_UNCOND][0], roles[_COND][1])
            for roles in by_pair.values()
            if _COND in roles and _UNCOND in roles
        ]
        self._dirty = False

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._info:
            return logits

        if self._dirty:
            self._rebuild_pairs()

        num_rows = logits.shape[0]
        for cond_idx, uncond_idx, cfg_scale in self._pairs:
            # Rows beyond this step's logits (pure-prefill / partially
            # scheduled steps) have nothing to blend; decode steps carry
            # both pair rows thanks to the scheduler pair patches.
            if cond_idx >= num_rows or uncond_idx >= num_rows:
                continue
            blended = logits[uncond_idx] + cfg_scale * (logits[cond_idx] - logits[uncond_idx])
            logits[cond_idx] = blended
            logits[uncond_idx] = blended

        return logits
