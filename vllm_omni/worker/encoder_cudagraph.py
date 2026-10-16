# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Encoder CUDA graph manager that replays a group only when one graph holds it."""

from typing import Any

import torch
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager


class SingleReplayEncoderCudaGraphManager(EncoderCudaGraphManager):
    """Replay an encoder group only when one captured graph holds all of it.

    The upstream manager splits a group across several replays and gives each
    item above the largest budget its own eager call. For a launch-bound vision
    tower that is usually slower than one eager call over the whole group, so a
    group that does not fit one graph goes back to the runner, which then
    encodes it with a single ``embed_multimodal`` call. Models opt in by
    setting ``encoder_cudagraph_single_replay``.
    """

    def execute(self, mm_kwargs: dict[str, Any]) -> list[torch.Tensor] | None:  # type: ignore[override]
        if not self.use_dp:
            specs = self._get_item_specs(mm_kwargs)
            fits = len(specs) <= self.max_batch_size and all(
                sum(spec.get_path_output_tokens(path) for spec in specs) <= max(budgets)
                for path, budgets in self.path_token_budgets.items()
            )
            if not fits:
                self.graph_misses += len(specs)
                return None
        return super().execute(mm_kwargs)
