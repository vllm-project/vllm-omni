# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graph capture for the RVQ depth loop.

The depth loop is seven sequential forwards of a four-layer block over a
sequence that grows from two to eight positions, plus a guided draw after each
one. At the batch sizes this stage runs, none of those kernels is anywhere near
large enough to hide its own launch: measured in isolation on one L20X, the
loop costs 8.4 ms per frame when driven from Python and 4.0 ms when replayed
from a graph, and that figure barely moves between one and eight guided pairs.
It is launch-bound, so it is worth recording.

Capture is lazy and keyed on the exact pair count, never padded. That keeps the
recorded kernels the same ones the eager loop selected for that shape, which is
what makes a replay bit-identical to the loop it replaces rather than merely
close to it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

__all__ = ["DepthGraphCache"]

# Inputs, in the order the eager loop takes them.
_DepthInputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
_DepthOutputs = tuple[torch.Tensor, torch.Tensor]


@dataclass
class _CapturedDepth:
    """One recorded depth loop and the buffers it reads and writes."""

    graph: torch.cuda.CUDAGraph
    inputs: _DepthInputs
    codes: torch.Tensor
    depth_hidden: torch.Tensor


class DepthGraphCache:
    """Replay the depth loop from a graph once its shape has settled.

    Args:
        run_eager: The loop itself, taking ``(cond_hidden, uncond_hidden, c0,
            seeds, positions)`` and returning ``(codes, depth_hidden)``.
        warmup_steps: How many times a pair count runs eagerly before it is
            recorded.
    """

    def __init__(
        self,
        run_eager: Callable[..., _DepthOutputs],
        *,
        warmup_steps: int = 2,
    ) -> None:
        self._run_eager = run_eager
        self._warmup_steps = warmup_steps
        self._captured: dict[int, _CapturedDepth] = {}
        self._eager_steps: dict[int, int] = {}
        self._disabled: set[int] = set()

    def run(self, *inputs: torch.Tensor) -> _DepthOutputs:
        """Run one depth loop, recording or replaying it where possible."""
        pairs = int(inputs[2].shape[0])
        entry = self._captured.get(pairs)
        if entry is None:
            if not self._should_capture(pairs, inputs[0]):
                return self._run_eager(*inputs)
            entry = self._capture(pairs, inputs)
            if entry is None:
                return self._run_eager(*inputs)

        for static, live in zip(entry.inputs, inputs, strict=True):
            static.copy_(live)
        entry.graph.replay()
        # The codes are fed back as the next frame's input and appended to the
        # request's trajectory, so they outlive the step; the graph overwrites
        # its output buffer on the next replay. The depth hidden states are
        # consumed inside this step by the concatenation that builds the
        # conditioning frame, which allocates, so that one can be handed back
        # as is.
        return entry.codes.clone(), entry.depth_hidden

    def _should_capture(self, pairs: int, sample: torch.Tensor) -> bool:
        if pairs in self._disabled or not sample.is_cuda:
            return False
        if torch.cuda.is_current_stream_capturing():
            # Already inside somebody else's capture; nested graphs are not a
            # thing and the outer recording covers this work anyway.
            return False
        count = self._eager_steps.get(pairs, 0) + 1
        self._eager_steps[pairs] = count
        # Let a new pair count run eagerly a few times first. Capture records
        # whatever the kernels do at that moment, and the first call into a
        # cuBLAS or SDPA path for an unseen shape can allocate a workspace or
        # pick an algorithm, neither of which belongs inside the recording.
        return count > self._warmup_steps

    def _capture(self, pairs: int, inputs: _DepthInputs) -> _CapturedDepth | None:
        try:
            static = tuple(torch.empty_like(tensor) for tensor in inputs)
            for target, live in zip(static, inputs, strict=True):
                target.copy_(live)
            graph = torch.cuda.CUDAGraph()
            with (
                torch.inference_mode(),
                torch.cuda.graph(
                    graph,
                    pool=current_platform.get_global_graph_pool(),
                    # The engine's connector threads keep their own streams
                    # busy while this runs. They never touch the depth loop's
                    # buffers, so restrict the legality check to this thread
                    # rather than failing the capture on their activity.
                    capture_error_mode="thread_local",
                ),
            ):
                codes, depth_hidden = self._run_eager(*static)
            entry = _CapturedDepth(graph=graph, inputs=static, codes=codes, depth_hidden=depth_hidden)
            self._captured[pairs] = entry
            logger.info("MiniMax Music 3 captured the RVQ depth CUDA graph for %d guided pair(s)", pairs)
            return entry
        except Exception as exc:  # noqa: BLE001 - any capture failure must fall back, not crash
            self._disabled.add(pairs)
            logger.warning(
                "MiniMax Music 3 RVQ depth CUDA graph disabled for %d guided pair(s), "
                "falling back to the eager loop: %s",
                pairs,
                exc,
            )
            return None
