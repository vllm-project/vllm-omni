# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Original Chatterbox guidance before the upstream sampling pipeline."""

from collections.abc import MutableMapping, Sequence
from typing import Any

import torch

from vllm_omni.worker_v2.omni_sampler import OmniSampler

DEFAULT_CFG_WEIGHT = 0.5
SPEECH_START_TOKEN = 6561
SPEECH_STOP_TOKEN = 6562
# Request ID -> (pair key, role, weight). The model owns registration/cleanup.
CFGPairRegistry = MutableMapping[str, tuple[str, str, float]]


class ChatterboxCFGSampler:
    """Blend paired rows, delegate sampling, and synchronize companion tokens.

    Registry roles are ``cond`` and ``uncond``. Both requests
    must be scheduled atomically and have identical speech token histories.
    A conditional request with zero weight needs no companion. Synthetic
    MRv2 warmup rows bypass pairing and still receive the speech token mask.
    Input logits must be unmasked: masking before subtraction creates NaNs.
    The wrapper deliberately owns no nn.Module state or request lifecycle.
    """

    def __init__(self, base_sampler: Any, pairs: CFGPairRegistry):
        self.base_sampler = base_sampler
        self.pairs = pairs

    def prepare_logits(
        self, logits: torch.Tensor, req_ids: Sequence[str]
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """Resolve current batch rows by request identity and apply guidance."""
        if logits.ndim != 2 or logits.shape[0] != len(req_ids):
            raise ValueError("Chatterbox CFG requires one logits row per request")
        if logits.shape[1] <= SPEECH_STOP_TOKEN:
            raise ValueError("Chatterbox CFG requires the Original speech vocabulary")
        groups: dict[str, dict[str, tuple[int, float]]] = {}
        for row, req_id in enumerate(req_ids):
            if req_id not in self.pairs and req_id.startswith("_warmup_"):
                continue
            if req_id not in self.pairs:
                raise ValueError(f"Missing Chatterbox CFG registration: {req_id}")
            key, role, weight = self.pairs[req_id]
            if role not in ("cond", "uncond"):
                raise ValueError(f"Unsupported Chatterbox CFG role: {role}")
            group = groups.setdefault(key, {})
            if role in group:
                raise ValueError(f"Duplicate Chatterbox CFG role in pair: {key}")
            group[role] = (row, weight)
        row_pairs = []
        guided = logits.float().clone()
        for key, group in groups.items():
            if len(group) == 1 and "cond" in group and group["cond"][1] == 0.0:
                continue
            if len(group) != 2:
                raise ValueError(f"Missing Chatterbox CFG companion in batch: {key}")
            cond, weight = group["cond"]
            uncond, companion_weight = group["uncond"]
            if weight != companion_weight:
                raise ValueError(f"Mismatched Chatterbox CFG weights: {key}")
            blended = logits[cond].float() + weight * (logits[cond].float() - logits[uncond].float())
            guided[cond] = blended
            guided[uncond] = blended
            row_pairs.append((cond, uncond))
        guided[:, SPEECH_START_TOKEN] = -torch.inf
        guided[:, SPEECH_STOP_TOKEN + 1 :] = -torch.inf
        return guided, row_pairs

    def sample(self, logits: torch.Tensor, sampling_metadata: Any, input_batch: Any, requests: Any = None) -> Any:
        """V1 model sampler entry point; preserve upstream sampling metadata."""
        guided, row_pairs = self.prepare_logits(logits, input_batch.req_ids)
        output = self.base_sampler(logits=guided, sampling_metadata=sampling_metadata)
        for cond, uncond in row_pairs:
            output.sampled_token_ids[uncond].copy_(output.sampled_token_ids[cond])
        return output


class ChatterboxMRv2CFGSampler(OmniSampler):
    """MRv2 adapter with upstream request lifecycle and logits processing."""

    def __init__(self, base_sampler: Any, pairs: CFGPairRegistry):
        super().__init__(base_sampler)
        self.cfg_sampler = ChatterboxCFGSampler(base_sampler, pairs)

    def __call__(self, logits: torch.Tensor, input_batch: Any) -> Any:
        guided, row_pairs = self.cfg_sampler.prepare_logits(logits, input_batch.req_ids)
        output = self.base_sampler(guided, input_batch)
        for cond, uncond in row_pairs:
            output.sampled_token_ids[uncond].copy_(output.sampled_token_ids[cond])
        return output
