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

import traceback
from collections.abc import Callable
from contextlib import ExitStack
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm_omni.model_executor.models.personaplex.personaplex_mimi import PersonaPlexMimiCodec

logger = init_logger(__name__)

__all__ = ["MimiFrameGraph", "capture_mimi_frame_graphs"]


def _is_recoverable_capture_error(error: RuntimeError) -> bool:
    """Allow only known capture limitations, never an unexplained CUDA error."""
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    known_failure = False
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        message = str(current).split("\n", 1)[0]
        if isinstance(current, torch.cuda.OutOfMemoryError) or (
            isinstance(current, RuntimeError)
            and (
                message.startswith("CUDA out of memory.")
                or message
                in {
                    "CUDA error: out of memory",
                    "CUDA error: operation not permitted when stream is capturing",
                }
            )
        ):
            known_failure = True
        elif not (
            isinstance(current, RuntimeError)
            and message == "CUDA error: operation failed due to a previous error during capture"
        ):
            return False
        # capture_end can replace a body exception with capture-invalidated.
        # The original failure must also be recoverable, even if suppressed.
        pending.extend(exc for exc in (current.__cause__, current.__context__) if exc is not None)
    return known_failure


def _capture_error_details(error: RuntimeError) -> str:
    """Preserve diagnostics without retaining capture frames or graph pools."""
    details = "".join(traceback.format_exception(error))
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if current.__traceback__ is not None:
            # ExitStack frames can retain their own traceback in exception tuples.
            traceback.clear_frames(current.__traceback__)
        current.__traceback__ = None
        pending.extend(exc for exc in (current.__cause__, current.__context__) if exc is not None)
    return details


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

    Returns an empty dict off CUDA or after a recoverable capture failure.
    Warmup, device execution and stream-reset failures always propagate.
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
    warmed_specs: list[tuple[str, Callable, torch.Tensor, torch.Tensor]] = []
    capture_error_details: str | None = None
    try:
        with torch.no_grad():
            # Validate every eager step before capture can select the fallback.
            for name, eager, static_input in specs:
                static_active = torch.ones(batch_size, dtype=torch.bool, device=device)
                for _ in range(max(warmup_iters, 1)):
                    eager(static_input, static_active)
                torch.accelerator.synchronize(device)
                warmed_specs.append((name, eager, static_input, static_active))
            for name, eager, static_input, static_active in warmed_specs:
                # capture_end may raise before torch.cuda.graph restores its
                # stream. Always restore our caller's stream on that path.
                with torch.cuda.stream(torch.cuda.current_stream(device)):
                    graph = torch.cuda.CUDAGraph()
                    # __enter__ synchronizes and clears the CUDA cache. A
                    # failure there is a setup failure, not a capture fallback.
                    capture = torch.cuda.graph(graph, pool=pool, capture_error_mode="thread_local")
                    capture.__enter__()
                    try:
                        # Other worker threads may issue CUDA calls while this records.
                        with ExitStack() as stack:
                            stack.push(capture)
                            static_output = eager(static_input, static_active)
                    except RuntimeError as error:
                        if not _is_recoverable_capture_error(error):
                            raise
                        capture_error_details = _capture_error_details(error)
                        graphs.clear()
                        # Release partial graphs and graph-owned output tensors
                        # before reset, including the last loop iteration.
                        graph = static_output = capture = None
                        break
                # A device execution failure is not an optional capture failure.
                torch.accelerator.synchronize(device)
                graphs[name] = MimiFrameGraph(graph, static_input, static_active, static_output, eager)
    finally:
        # Warmup frames advanced the real streaming state; restore a fresh
        # stream without reallocating the tensors the graphs point at.
        codec.reset_streaming()
    # Reset enqueues device work too. Do not report usable graphs or an eager
    # fallback unless that work completed successfully.
    torch.accelerator.synchronize(device)
    if capture_error_details is not None:
        logger.warning(
            "PersonaPlex Mimi CUDA graphs were requested (%s at batch size %d) but capture failed; "
            "the codec runs eagerly, which lowers realtime session capacity\n%s",
            "/".join(name for name, _, _ in specs),
            batch_size,
            capture_error_details,
        )
    if graphs:
        logger.debug("Captured PersonaPlex Mimi %s graph(s) at batch size %d", "/".join(graphs), batch_size)
    return graphs
