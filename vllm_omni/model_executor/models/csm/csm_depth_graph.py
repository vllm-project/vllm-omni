# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA graphs for the CSM depth decoder's frame step."""

from __future__ import annotations

from typing import Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger

from vllm_omni.model_executor.models.csm.csm_depth_decoder import CsmDepthDecoder

logger = init_logger(__name__)

_GRAPH_WARMUP_ITERS = 2


def depth_graph_buckets(vllm_config: Any) -> list[int]:
    """Batch sizes whose frame step is captured; none for a stage that runs eagerly.

    These are vLLM's capture sizes below ``max_num_seqs`` (powers of two when
    it has none) and ``max_num_seqs`` itself, so every batch has a bucket.
    """
    compilation = vllm_config.compilation_config
    if vllm_config.model_config.enforce_eager or compilation.cudagraph_mode == CUDAGraphMode.NONE:
        return []
    max_batch = int(vllm_config.scheduler_config.max_num_seqs)
    sizes = {int(size) for size in compilation.cudagraph_capture_sizes or () if 0 < int(size) < max_batch}
    if not sizes:
        sizes = {1 << shift for shift in range(max_batch.bit_length()) if (1 << shift) < max_batch}
    return sorted(sizes | {max_batch})


def _sampling_key(do_sample: bool, temperature: float, top_k: int, top_p: float) -> tuple[bool, float, int, float]:
    if not do_sample:
        return (False, 1.0, 0, 1.0)
    return (True, float(temperature), int(top_k), float(top_p))


class CsmDepthDecoderGraphs:
    """Runs ``CsmDepthDecoder.generate_frame`` from one CUDA graph per batch-size bucket.

    A batch is padded to its bucket in fixed input buffers. The graphs record
    one sampling setting; a frame that asks for another, or has more rows than
    the largest bucket, runs on the decoder directly, as does a bucket without
    a graph (off CUDA, or after a failed capture) on the padded buffers.
    """

    def __init__(
        self,
        decoder: CsmDepthDecoder,
        *,
        buckets: list[int],
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> None:
        self.decoder = decoder
        self.buckets = sorted({int(bucket) for bucket in buckets})
        self.sampling = _sampling_key(do_sample, temperature, top_k, top_p)
        projector = decoder.model.inputs_embeds_projector.weight
        max_rows = self.buckets[-1]
        self._hidden = projector.new_zeros((max_rows, projector.shape[1]))
        self._first_codes = torch.zeros(max_rows, dtype=torch.long, device=projector.device)
        # float32 whatever the default dtype is (vLLM loads weights under the stage's): bf16 rounds noise to 1.
        self._uniforms = (
            torch.full(
                (max_rows, decoder.num_steps, decoder.vocab_size), 0.5, dtype=torch.float32, device=projector.device
            )
            if self.sampling[0]
            else None
        )
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._outputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._kv_scratch: torch.Tensor | None = None

    def generate_frame(
        self,
        backbone_hidden: torch.Tensor,
        first_codes: torch.Tensor,
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        uniforms: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Complete one frame for every row; same contract as the decoder's ``generate_frame``."""
        rows = first_codes.shape[0]
        bucket = next((bucket for bucket in self.buckets if bucket >= rows), None)
        recorded = _sampling_key(do_sample, temperature, top_k, top_p) == self.sampling
        if bucket is None or rows == 0 or not recorded or (do_sample and uniforms is None):
            return self.decoder.generate_frame(
                backbone_hidden,
                first_codes,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                uniforms=uniforms,
            )
        # Padding rows keep whatever an earlier frame left there; rows do not interact.
        self._hidden[:rows].copy_(backbone_hidden)
        self._first_codes[:rows].copy_(first_codes)
        if self._uniforms is not None:
            self._uniforms[:rows].copy_(uniforms)
        graph = self._graphs.get(bucket)
        if graph is not None and not torch.cuda.is_current_stream_capturing():
            graph.replay()
            codes, frame_embeds = self._outputs[bucket]
        else:
            codes, frame_embeds = self._step(bucket)
        # The next replay overwrites the graph's outputs, so hand out copies.
        return codes[:rows].clone(), frame_embeds[:rows].clone()

    def _step(self, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        do_sample, temperature, top_k, top_p = self.sampling
        return self.decoder.generate_frame(
            self._hidden[:rows],
            self._first_codes[:rows],
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            uniforms=None if self._uniforms is None else self._uniforms[:rows],
        )

    @torch.inference_mode()
    def capture(self) -> bool:
        """Capture one graph per bucket, largest first; returns whether every bucket has one.

        The buckets never replay concurrently and every frame copies its result
        out, so they share one pool. It is not vLLM's: the backbone's hidden
        states live there and are read again after the frame step.
        """
        if self._hidden.device.type != "cuda":
            return False
        # The graphs record the scratch's address; hold it so it outlives them.
        self.decoder.reserve(self.buckets[-1])
        self._kv_scratch = self.decoder._kv_scratch
        pool = torch.cuda.graph_pool_handle()
        for rows in reversed(self.buckets):
            if rows in self._graphs:
                continue
            try:
                for _ in range(_GRAPH_WARMUP_ITERS):
                    self._step(rows)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, capture_error_mode="thread_local"):
                    outputs = self._step(rows)
            except Exception:
                logger.warning(
                    "CSM depth decoder CUDA graph capture failed at %d rows; those frames run eagerly",
                    rows,
                    exc_info=True,
                )
                continue
            self._graphs[rows] = graph
            self._outputs[rows] = outputs
        logger.info("Captured CSM depth decoder CUDA graphs for batch sizes %s", sorted(self._graphs))
        return len(self._graphs) == len(self.buckets)
