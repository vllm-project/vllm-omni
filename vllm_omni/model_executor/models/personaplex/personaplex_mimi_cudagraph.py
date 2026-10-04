# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Replay the PersonaPlex streaming Mimi frame steps from CUDA graphs.

One 80 ms Mimi encode or decode frame is a few hundred small kernels (SEANet
convs, an 8-layer streaming transformer and the RVQ), so eager execution is
launch bound. Every streaming state tensor of ``PersonaPlexMimiCodec`` (conv
carries, fresh-row flags, ring KV caches and offsets) is allocated once by
``streaming_init`` and only updated in place, and row resets are in place too,
so a graph recorded at the streaming batch size stays valid across frames,
session slot recycling and full stream resets.

The ``active`` row mask is a graph input: one graph at the shared batch size
serves any subset of live rows, and inactive rows keep their state exactly as
in eager execution.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm_omni.model_executor.models.personaplex.personaplex_mimi import PersonaPlexMimiCodec

logger = init_logger(__name__)

__all__ = ["MimiFrameGraph", "capture_mimi_frame_graphs"]


class MimiFrameGraph:
    """A recorded ``(frame, active) -> output`` codec step with static buffers."""

    def __init__(
        self,
        graph: torch.cuda.CUDAGraph,
        static_input: torch.Tensor,
        static_active: torch.Tensor,
        static_output: torch.Tensor,
        eager: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        self.graph = graph
        self.static_input = static_input
        self.static_active = static_active
        self.static_output = static_output
        self.eager = eager
        self.replays = 0
        self._warned = False

    def replay(self, frame: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        if frame.shape != self.static_input.shape or active.shape != self.static_active.shape:
            if not self._warned:
                self._warned = True
                logger.warning(
                    "PersonaPlex Mimi graph input shape %s does not match capture shape %s; running eagerly",
                    tuple(frame.shape),
                    tuple(self.static_input.shape),
                )
            return self.eager(frame, active)
        self.static_input.copy_(frame)
        self.static_active.copy_(active)
        self.graph.replay()
        self.replays += 1
        # The static output is overwritten by the next replay.
        return self.static_output.clone()


def capture_mimi_frame_graphs(
    codec: PersonaPlexMimiCodec,
    *,
    encode: bool,
    decode_frame_counts: tuple[int, ...],
    warmup_iters: int = 2,
    pool: tuple[int, int] | None = None,
) -> dict[str, MimiFrameGraph]:
    """Capture the codec's encode and ``F``-frame decode steps.

    Returns an empty dict off CUDA. A failed capture logs a warning with the
    traceback and also returns an empty dict, so the codec keeps running eagerly.
    """
    from vllm_omni.model_executor.models.personaplex.personaplex_mimi import CODEBOOKS, FRAME_SIZE

    device = torch.device(codec.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        logger.info("PersonaPlex Mimi CUDA graphs skipped on device %s", device)
        return {}
    batch_size = codec._batch_size
    specs: list[tuple[str, Callable, torch.Tensor]] = []
    if encode:
        specs.append(
            (
                "encode",
                codec._encode_frame_eager,
                torch.zeros(batch_size, FRAME_SIZE, device=device, dtype=codec.dtype),
            )
        )
    for frames in sorted(set(decode_frame_counts)):
        specs.append(
            (
                f"decode_f{frames}",
                codec._decode_frames_eager,
                torch.zeros(batch_size, CODEBOOKS, frames, device=device, dtype=torch.long),
            )
        )
    # A private pool by default: these graphs are captured outside the model
    # runner's own capture sequence, so they must not reuse its pool blocks.
    if pool is None:
        pool = torch.cuda.graph_pool_handle()
    graphs: dict[str, MimiFrameGraph] = {}
    try:
        with torch.no_grad():
            for name, eager, static_input in specs:
                static_active = torch.ones(batch_size, dtype=torch.bool, device=device)
                for _ in range(max(warmup_iters, 1)):
                    eager(static_input, static_active)
                torch.accelerator.synchronize(device)
                graph = torch.cuda.CUDAGraph()
                # Other worker threads may issue CUDA calls while this records.
                with torch.cuda.graph(graph, pool=pool, capture_error_mode="thread_local"):
                    static_output = eager(static_input, static_active)
                torch.accelerator.synchronize(device)
                graphs[name] = MimiFrameGraph(graph, static_input, static_active, static_output, eager)
    except RuntimeError:
        # CUDA capture errors (including out of memory) are RuntimeErrors. The
        # codec is still correct without graphs, so keep serving eagerly, and
        # log the traceback because eager frames are launch bound.
        logger.warning(
            "PersonaPlex Mimi CUDA graphs were requested (%s at batch size %d) but capture failed; "
            "the codec runs eagerly, which lowers realtime session capacity",
            "/".join(name for name, _, _ in specs),
            batch_size,
            exc_info=True,
        )
        graphs = {}
    finally:
        # Warmup frames advanced the real streaming state; restore a fresh
        # stream without reallocating the tensors the graphs point at.
        codec.reset_streaming()
    if graphs:
        logger.debug("Captured PersonaPlex Mimi %s graph(s) at batch size %d", "/".join(graphs), batch_size)
    return graphs
