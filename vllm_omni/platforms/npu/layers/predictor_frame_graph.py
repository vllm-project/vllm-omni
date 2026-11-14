# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One NPU graph for all residual codebooks of one Qwen3-TTS frame.

The per-step graphs replay the predictor once per codebook and return to
Python to sample and write the next embedding. This graph runs all 15 steps
in one replay, with top-k Gumbel sampling and optional frame-local KV reuse.
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON

logger = init_logger(__name__)


class PredictorFrameGraph:
    """Fixed batch, top-k and temperature. The caller owns the real rows."""

    def __init__(self, predictor, batch: int, top_k: int, inv_temperature: float) -> None:
        self.predictor = predictor
        self.batch = batch
        self.top_k = int(top_k)
        self.inv_temperature = float(inv_temperature)
        self.steps = int(predictor._num_groups) - 1
        weight = next(predictor.model.parameters())
        self.device = weight.device
        vocab = int(predictor.config.vocab_size)
        self.sampler = None
        if predictor._fused_sampling and HAS_TRITON and 0 < vocab <= 4096 and 0 <= top_k <= vocab:
            from vllm_omni.platforms.npu.layers.code_predictor_sampling import sample_topk_gumbel

            self.sampler = sample_topk_gumbel
        self.uniforms = torch.empty(batch, self.steps, vocab, device=self.device, dtype=torch.float32)
        self.codes = torch.empty(batch, predictor._num_groups, device=self.device, dtype=torch.long)
        self.graph: torch.npu.NPUGraph | None = None
        self.kv_caches = None
        self.cached_steps = []
        if predictor._kv_requested:
            model = predictor.model
            capacity = self.steps + 1
            self.kv_caches = [
                tuple(
                    torch.zeros(
                        batch,
                        layer.self_attn.num_kv_heads,
                        capacity,
                        layer.self_attn.head_dim,
                        device=self.device,
                        dtype=weight.dtype,
                    )
                    for _ in range(2)
                )
                for layer in model.layers
            ]
            for step in range(1, self.steps + 1):
                begin = 0 if step == 1 else step
                positions = torch.arange(begin, step + 1, device=self.device)
                mask = torch.arange(capacity, device=self.device)[None, :] <= positions[:, None]
                self.cached_steps.append(
                    (begin, positions, positions[None, :].expand(batch, -1), mask[None, None, :, :])
                )

    def compute(self) -> torch.Tensor:
        predictor = self.predictor
        proj = predictor._proj_buf
        pos = predictor._bucket_pos_ids[self.batch]
        forward = predictor._compiled_model_fwd
        heads = predictor._lm_heads_list
        embeds = predictor._codec_embeds_list
        project = predictor.small_to_mtp_projection
        codes = self.codes
        for step in range(1, self.steps + 1):
            if self.kv_caches is None:
                hidden = forward(proj[: self.batch], pos)
                current = hidden[:, step, :]
            else:
                begin, writes, positions, mask = self.cached_steps[step - 1]
                hidden = predictor.model.forward_cached(
                    proj[: self.batch, begin : step + 1, :],
                    positions,
                    self.kv_caches,
                    writes,
                    mask,
                )
                current = hidden[:, -1, :]
            logits = heads[step - 1](current)
            draw = self.uniforms[:, step - 1, :]
            if self.sampler is not None:
                code = self.sampler(logits, draw, self.top_k, self.inv_temperature).squeeze(-1)
            else:
                scaled = logits * self.inv_temperature
                if self.top_k > 0:
                    threshold = scaled.topk(self.top_k, dim=-1).values[:, -1:]
                    scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
                code = (scaled.float() - torch.log(-torch.log(draw))).argmax(dim=-1)
            codes[:, step] = code
            if step < self.steps:
                embedded = embeds[step - 1](code.unsqueeze(1))
                proj[: self.batch, step + 1, :] = project(embedded.reshape(self.batch, 1, -1)).reshape(self.batch, -1)
        return codes

    @torch.inference_mode()
    def capture(self) -> None:
        stream = torch.npu.Stream(device=self.device)
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(2):
                self.compute()
        torch.npu.current_stream().wait_stream(stream)
        torch.npu.synchronize()
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph, stream=stream):
            self.compute()

    @torch.inference_mode()
    def run(self, uniforms: torch.Tensor) -> torch.Tensor:
        self.uniforms[: uniforms.shape[0]].copy_(uniforms)
        assert self.graph is not None
        self.graph.replay()
        return self.codes


def capture_predictor_frame_graphs(predictor, top_k: int, temperature: float) -> dict[int, PredictorFrameGraph]:
    """Capture one graph per already-warmed batch bucket. Empty on failure."""
    if temperature <= 0 or top_k < 0:
        return {}
    graphs: dict[int, PredictorFrameGraph] = {}
    inv_temperature = 1.0 / temperature
    try:
        for batch in predictor._bucket_sizes:
            graph = PredictorFrameGraph(predictor, batch, top_k, inv_temperature)
            graph.capture()
            graphs[batch] = graph
    except Exception:
        logger.warning("Qwen3-TTS predictor frame graph capture failed; keeping per-step graphs", exc_info=True)
        for graph in graphs.values():
            if graph.graph is not None:
                graph.graph.reset()
        return {}
    logger.info(
        "Qwen3-TTS predictor frame graphs captured for batches %s (top_k=%d, temperature=%s)",
        sorted(graphs),
        top_k,
        temperature,
    )
    return graphs
