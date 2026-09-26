# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Encoder CUDA graph manager that runs eager when a batch exceeds the capture.

Packing bounds a batch by output-token count and item count, never by frame
count, so a batch can reach replay with more rows than the captured buffers
hold. Two configurations break the frame bound: an explicit
``encoder_cudagraph_max_frames_per_batch`` below what the batch carries, and an
item whose own frame count exceeds the model's ``max_frames_per_video`` (the
default ``max_frames_per_batch`` is ``max_batch_size`` times that value, which
a batch of at most ``max_batch_size`` conforming items cannot exceed). Upstream
then copies into a shorter destination and raises. The subclass below answers
the same call with the eager result, so the request completes and the batch is
counted as a graph miss.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.v1.worker.encoder_cudagraph import BudgetGraphMetadata, EncoderCudaGraphManager

logger = init_logger(__name__)

# WHY: the override reproduces upstream's copy-and-replay sequence so a batch
# that fits is prepared exactly once, as upstream prepares it. A batch that does
# not fit pays for the prepared buffers it then discards; the fallback is the
# slow path already.
# FRAGILITY: that makes it depend on the inputs of the upstream method it
# replaces, and on the metadata fields it dereferences.
# SCOPE: checked when a manager is actually created (see
# `OmniGPUModelRunner._create_encoder_cudagraph_manager`), not at import, so a
# stage that never captures encoder graphs is unaffected by a mismatch it would
# never reach. The caller logs and keeps upstream's manager.
_EXPECTED_PARAMETERS = ("self", "mm_kwargs", "token_budget", "path")
_REQUIRED_METADATA_FIELDS = ("input_buffers", "output_buffer", "graph", "max_batch_size", "max_frames_per_batch")
# Everything the override calls or reads on the upstream manager.
_REQUIRED_MANAGER_ATTRIBUTES = (
    "_get_item_specs",
    "_get_graph_set",
    "_copy_padded_buffer",
    "max_batch_size",
    "max_frames_per_batch",
    "graph_hits",
    "graph_misses",
)


def upstream_contract_mismatch() -> str | None:
    """Describe how upstream diverged from what the override needs, or None."""
    actual = tuple(inspect.signature(EncoderCudaGraphManager._run_budget_graph).parameters)
    if actual != _EXPECTED_PARAMETERS:
        return f"EncoderCudaGraphManager._run_budget_graph takes {actual}, expected {_EXPECTED_PARAMETERS}"
    missing_fields = tuple(
        name for name in _REQUIRED_METADATA_FIELDS if name not in BudgetGraphMetadata.__dataclass_fields__
    )
    if missing_fields:
        return f"BudgetGraphMetadata is missing {missing_fields}"
    missing_attrs = tuple(
        name
        for name in _REQUIRED_MANAGER_ATTRIBUTES
        if not hasattr(EncoderCudaGraphManager, name) and name not in EncoderCudaGraphManager.__init__.__code__.co_names
    )
    if missing_attrs:
        return f"EncoderCudaGraphManager is missing {missing_attrs}"
    return None


class OmniEncoderCudaGraphManager(EncoderCudaGraphManager):
    """Replay when the batch fits the captured buffers, otherwise run eager."""

    @staticmethod
    def _exceeds_buffer(destination: torch.Tensor, source: torch.Tensor) -> bool:
        return source.ndim > 0 and source.shape[0] > destination.shape[0]

    def _eager_forward(self, mm_kwargs: dict[str, Any], path: str) -> torch.Tensor:
        with torch.inference_mode():
            return self.model.encoder_eager_forward(mm_kwargs, path=path)

    def _run_budget_graph(
        self,
        mm_kwargs: dict[str, Any],
        token_budget: int,
        path: str = "default",
    ) -> torch.Tensor:
        """Execute the budget graph, or the eager encoder when it does not fit.

        Upstream returns ``None`` when the budget has no captured graph and its
        caller asserts on that. Returning the eager output keeps the batch on
        the same postprocessing path the all-eager branch already uses.
        """
        num_items = len(self._get_item_specs(mm_kwargs))
        graph_meta = self._get_graph_set(path).get(token_budget)
        if graph_meta is None:
            # `_execute_local` skips a path contributing zero tokens before it
            # selects a budget, and every other budget it can select was
            # captured, so this is unreachable today; upstream's own caller
            # asserts on the None it would otherwise return, which is why the
            # branch is kept.
            logger.warning_once(
                "Encoder CUDA graph replay skipped on path %s: token budget %d has no captured graph. Running eager.",
                path,
                token_budget,
            )
            self.graph_misses += num_items
            return self._eager_forward(mm_kwargs, path)

        replay = self.model.prepare_encoder_cudagraph_replay_buffers(
            mm_kwargs,
            self.max_batch_size,
            self.max_frames_per_batch,
            path,
        )

        # One pass: whether a source fits is decided by the routine that will
        # copy it. `_copy_padded_buffer` needs the whole source to fit, while a
        # model's own logic can accept a taller source on a different layout
        # (Qwen2.5-VL's FlashInfer `cu_seqlens` bounds half the rows), so that
        # one answers by raising.
        oversized: str | None = None
        for key, buffer in graph_meta.input_buffers.items():
            source = replay.values.get(key)
            if source is None:
                continue
            if source.ndim == 0:
                buffer.copy_(source)
                continue
            padding_logic = self.config.padding_logics.get(key)
            if padding_logic is None:
                if self._exceeds_buffer(buffer, source):
                    oversized = f"{key} ({source.shape[0]} > {buffer.shape[0]})"
                    break
                self._copy_padded_buffer(buffer, source)
                continue
            try:
                padding_logic(buffer, source)
            except AssertionError as exc:
                oversized = f"{key} ({exc})"
                break

        if oversized is not None:
            # Which knob raises the capacity depends on the buffer: the frame
            # axis follows max_frames_per_batch, the token axis the budget the
            # batch was packed into. Report both rather than name one.
            logger.warning_once(
                "Encoder CUDA graph replay skipped on path %s: %s exceeds the captured capacity "
                "(token_budget=%d, max_batch_size=%d, max_frames_per_batch=%d). Running eager.",
                path,
                oversized,
                token_budget,
                graph_meta.max_batch_size,
                graph_meta.max_frames_per_batch,
            )
            self.graph_misses += num_items
            return self._eager_forward(mm_kwargs, path)

        graph_meta.graph.replay()

        self.graph_hits += num_items
        return graph_meta.output_buffer
