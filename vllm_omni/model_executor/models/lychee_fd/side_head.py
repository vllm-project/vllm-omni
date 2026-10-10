# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Released side-vocabulary projection without projecting unused text rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod


@dataclass
class _Projection:
    source: torch.Tensor
    ids: torch.Tensor
    weight: torch.Tensor


class LycheeSideHead:
    """Two bounded, model-owned subsets with global output token coordinates.

    Native parameter storage and the text head are unchanged. Unsupported
    quantized, parallel or head-dtype configurations use the native processor.
    """

    def __init__(self, config: Any, vocab_size: int) -> None:
        speech_extras = (
            config.stoken_delay_token_id,
            config.stoken_pad_token_id,
            config.start_speaking_token_id,
            config.start_listening_token_id,
            config.keep_speaking_token_id,
            config.keep_listening_token_id,
            config.sleep_token_id,
            config.detect_token_id,
        )
        control_extras = (
            config.start_speaking_token_id,
            config.start_listening_token_id,
            config.keep_listening_token_id,
            config.keep_speaking_token_id,
            config.start_bc_token_id,
            config.sleep_token_id,
            config.detect_token_id,
        )
        self.vocab_size = vocab_size
        self.token_ids = {
            "speech": self._ids(config.stoken_token_ids_min, config.stoken_token_ids_max, speech_extras),
            "control": self._ids(config.control_token_ids_min, config.control_token_ids_max, control_extras),
        }
        self._cache: dict[str, _Projection] = {}

    def _ids(self, lower: int, upper: int, extras: tuple[int, ...]) -> torch.Tensor:
        ids = set(range(max(0, lower), min(self.vocab_size, upper)))
        ids.update(token for token in extras if 0 <= token < self.vocab_size)
        if not ids:
            raise ValueError("Lychee side projection requires a nonempty token subset")
        return torch.tensor(sorted(ids), dtype=torch.long, device="cpu")

    def clear(self) -> None:
        self._cache.clear()

    def project(self, branch: str, hidden: torch.Tensor, head: Any, processor: Any) -> torch.Tensor:
        if branch not in self.token_ids:
            raise ValueError(f"Unknown Lychee side projection: {branch}")
        if (
            head.tp_size != 1
            or not isinstance(head.quant_method, (UnquantizedEmbeddingMethod, UnquantizedLinearMethod))
            or processor.head_dtype not in (None, hidden.dtype)
            or processor.logits_as_input
        ):
            return processor(head, hidden)
        weight = head.weight
        cached = self._cache.get(branch)
        if (
            cached is None
            or cached.source is not weight
            or cached.weight.device != hidden.device
            or cached.weight.dtype != hidden.dtype
        ):
            ids = self.token_ids[branch].to(device=weight.device)
            subset = weight.detach().index_select(0, ids).to(device=hidden.device, dtype=hidden.dtype)
            cached = _Projection(weight, ids.to(hidden.device), subset)
            self._cache[branch] = cached
        values = nn.functional.linear(hidden, cached.weight)
        if processor.soft_cap is not None:
            values = torch.tanh(values / processor.soft_cap) * processor.soft_cap
        if processor.scale != 1.0:
            values *= processor.scale
        result = torch.full(
            (*hidden.shape[:-1], self.vocab_size), float("-inf"), dtype=values.dtype, device=hidden.device
        )
        result.index_copy_(-1, cached.ids, values)
        return result
