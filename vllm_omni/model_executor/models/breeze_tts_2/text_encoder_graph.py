# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Padded T5Gemma2 prefill graphs with explicit bidirectional masks."""

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.t5gemma2.modeling_t5gemma2 import (
    T5Gemma2EncoderLayer,
    T5Gemma2TextEncoder,
    sliding_window_mask_function,
)
from vllm.platforms import current_platform

from vllm_omni.platforms import current_omni_platform


class BreezeTextEncoderCompiled:
    """Compiled prefill layers without retaining large activation graph pools."""

    def __init__(self, encoder: T5Gemma2TextEncoder, projection: nn.Linear) -> None:
        self.encoder = encoder
        self.projection = projection
        self._layer = torch.compile(
            T5Gemma2EncoderLayer.forward,
            fullgraph=True,
            dynamic=True,
            options={"triton.cudagraphs": False, "epilogue_fusion": False},
        )

    def run_batch(self, prompts: list[torch.Tensor]) -> list[torch.Tensor]:
        encoder = self.encoder
        device = self.projection.weight.device
        dtype = self.projection.weight.dtype
        lengths = [prompt.numel() for prompt in prompts]
        ids = nn.utils.rnn.pad_sequence(prompts, batch_first=True).to(device)
        positions = torch.arange(ids.shape[1], device=device)[None]
        invalid = positions >= torch.tensor(lengths, device=device)[:, None]
        mask = torch.zeros((len(prompts), 1, ids.shape[1], ids.shape[1]), device=device, dtype=dtype)
        mask.masked_fill_(invalid[:, None, None, :], torch.finfo(dtype).min)
        mask_function = sliding_window_mask_function(encoder.config.sliding_window, is_causal=False)
        allowed = mask_function(0, 0, positions[:, :, None], positions[:, None, :])
        local_mask = mask.masked_fill(~allowed[:, None], torch.finfo(dtype).min)
        masks = {"full_attention": mask, "sliding_attention": local_mask}
        embedding = encoder.embed_tokens
        hidden = F.embedding(ids, embedding.weight) * embedding.embed_scale.to(dtype)
        hidden = torch.where(
            (ids == embedding.eoi_token_index).unsqueeze(-1), embedding.eoi_embedding.to(dtype), hidden
        )
        position_embeddings = {
            kind: encoder.rotary_emb(hidden, positions, kind) for kind in set(encoder.config.layer_types)
        }
        for layer in encoder.layers:
            kind = layer.attention_type
            hidden = self._layer(layer, hidden, position_embeddings[kind], masks[kind], positions)
        output = self.projection(encoder.norm(hidden))
        return [output[row, :length].clone() for row, length in enumerate(lengths)]

    def warmup(self, max_batch_size: int) -> None:
        # Dynamo specializes size-one dimensions; a second batch exercises
        # the general batch dimension used by paired and reference prompts.
        for batch in sorted({1, max_batch_size}):
            self.run_batch([torch.zeros(256, dtype=torch.long) for _ in range(batch)])
        torch.accelerator.empty_cache()


class BreezeTextEncoderGraph:
    def __init__(self, encoder: T5Gemma2TextEncoder, projection: nn.Linear, size: int, batch_size: int = 1) -> None:
        device = projection.weight.device
        dtype = projection.weight.dtype
        self.ids = torch.zeros((batch_size, size), device=device, dtype=torch.long)
        self.positions = torch.arange(size, device=device).unsqueeze(0)
        self.full_mask = torch.zeros((batch_size, 1, size, size), device=device, dtype=dtype)
        mask_function = sliding_window_mask_function(encoder.config.sliding_window, is_causal=False)
        allowed = mask_function(0, 0, self.positions[:, :, None], self.positions[:, None, :])
        self.local_mask_base = torch.zeros_like(self.full_mask).masked_fill_(~allowed[:, None], torch.finfo(dtype).min)
        self.local_mask = self.local_mask_base.clone()
        self.masks = {"full_attention": self.full_mask, "sliding_attention": self.local_mask}
        for _ in range(3):
            self._encode(encoder, projection)
        current_omni_platform.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            self.graph, pool=current_platform.get_global_graph_pool(), capture_error_mode="thread_local"
        ):
            self.output = self._encode(encoder, projection)

    def _encode(self, encoder: T5Gemma2TextEncoder, projection: nn.Linear) -> torch.Tensor:
        embedding = encoder.embed_tokens
        hidden = F.embedding(self.ids, embedding.weight) * embedding.embed_scale.to(embedding.weight.dtype)
        # HF's boolean-index assignment materializes a data-dependent index
        # list. Express the same EOI substitution with a fixed-shape select.
        hidden = torch.where(
            (self.ids == embedding.eoi_token_index).unsqueeze(-1),
            embedding.eoi_embedding.to(hidden.dtype),
            hidden,
        )
        encoded = encoder(inputs_embeds=hidden, attention_mask=self.masks, position_ids=self.positions)
        return projection(encoded.last_hidden_state)

    def run(self, prompt: torch.Tensor) -> torch.Tensor:
        return self.run_batch([prompt[0]])[0]

    def run_batch(self, prompts: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(prompts) != self.ids.shape[0]:
            raise ValueError("Breeze text batch differs from the captured graph size")
        lengths = [prompt.numel() for prompt in prompts]
        if not all(0 < length <= self.ids.shape[1] for length in lengths):
            raise ValueError("Breeze text segment does not fit its graph bucket")
        self.ids.zero_()
        for row, (prompt, length) in enumerate(zip(prompts, lengths, strict=True)):
            self.ids[row, :length].copy_(prompt)
        invalid = self.positions >= torch.tensor(lengths, device=self.ids.device)[:, None]
        invalid = invalid[:, None, None, :]
        self.full_mask.zero_()
        self.full_mask.masked_fill_(invalid, torch.finfo(self.full_mask.dtype).min)
        self.local_mask.copy_(self.local_mask_base)
        self.local_mask.masked_fill_(invalid, torch.finfo(self.local_mask.dtype).min)
        self.graph.replay()
        # Subsequent prefills and backbone/depth graphs reuse graph storage.
        # The runner must receive independently owned prompt embeddings.
        return [self.output[row, :length].clone() for row, length in enumerate(lengths)]
