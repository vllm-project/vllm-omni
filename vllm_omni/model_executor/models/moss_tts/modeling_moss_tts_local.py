# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Local depth transformer for MossTTSRealtime.

A small (4-layer) Qwen3-style decoder that generates the rvq=16 RVQ codebook
codes for one audio frame, autoregressively over codebooks. It runs inside the
talker's per-step ``make_omni_output``, independent from vLLM's main scheduler.

The transformer body is shared with Qwen3-TTS and Qwen3-Omni via
``common.qwen3_code_predictor.CodePredictorBaseModel`` (re-prefill, no KV cache,
HF-compatible numerics). MossTTSRealtime differs in two ways:

  * codebook 0 is generated here from ``backbone_last_hidden`` (the other models
    receive it from the talker's main LM head), so we run one extra step and own
    all ``rvq`` LM heads;
  * sampling adds top-p and a windowed repetition penalty on top of
    temperature + top-k, matching upstream ``MossTTSRealtimeInference.generate``.

Re-prefilling the (<=rvq) frame each step is numerically identical to the
previous KV-cache loop -- causal attention, only the last position is read.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni.model_executor.models.common.qwen3_code_predictor import (
    CodePredictorBaseModel,
)
from vllm_omni.model_executor.models.moss_tts.configuration_moss_tts import (
    MossTTSLocalTransformerConfig,
)


class MossTTSRealtimeLocalTransformer(nn.Module):
    """Per-frame depth transformer. Mirrors upstream ``...LocalTransformer``.

    State per audio frame:
      - The first input token uses ``backbone_last_hidden_state`` as the
        embedding (codebook 0's "input" is the backbone hidden, not a token).
      - Subsequent tokens (1..rvq-1) embed via ``model.codec_embedding[idx-1]``.

    Outputs:
      - One logit row per codebook position, projected through
        ``local_lm_heads[codebook_idx]`` (passed in by the talker).
    """

    def __init__(self, cfg: MossTTSLocalTransformerConfig) -> None:
        super().__init__()
        self.config = cfg
        # Shared body. Its codec_embedding holds rvq-1 embeddings -- upstream's
        # embed_tokens for codebooks 1..rvq-1 (codebook 0 uses the backbone
        # hidden). embedding_dim == hidden_size, so no projection is needed.
        self.model = CodePredictorBaseModel(
            cfg,
            embedding_dim=cfg.hidden_size,
            use_parallel_embedding=False,
            prefix="model",
        )

    @torch.no_grad()
    def generate_frame(
        self,
        backbone_last_hidden: torch.Tensor,  # (B, H)
        lm_heads: nn.ModuleList,  # ModuleList of rvq Linear(H -> audio_vocab_size)
        *,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        history_per_codebook: list[list[int]] | None = None,
    ) -> torch.Tensor:
        """Generate one audio frame (rvq codebook tokens) for batch B.

        Returns a ``(B, rvq)`` LongTensor.

        ``history_per_codebook[i]`` is a list of recently-emitted token ids for
        codebook ``i``; when ``repetition_penalty != 1.0`` those tokens get
        their logits scaled down (mirrors upstream's rep-penalty behaviour).
        """
        device = backbone_last_hidden.device
        B = backbone_last_hidden.shape[0]
        rvq = self.config.rvq
        hidden_size = self.config.hidden_size

        codec_embeds = self.model.codec_embedding

        # Re-prefill buffer: position 0 = backbone hidden, position s = embed of
        # code_{s-1}. At step s we forward positions [0..s] and read the last.
        embeds = backbone_last_hidden.new_zeros((B, rvq, hidden_size))
        embeds[:, 0, :] = backbone_last_hidden.to(embeds.dtype)

        pos_ids_full = torch.arange(rvq, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)

        codes = backbone_last_hidden.new_zeros((B, rvq), dtype=torch.long)

        for step in range(rvq):
            seq_len = step + 1
            hidden = self.model(embeds[:, :seq_len, :], pos_ids_full[:, :seq_len])
            logits = lm_heads[step](hidden[:, step, :]).float()

            if repetition_penalty != 1.0 and history_per_codebook is not None and step < len(history_per_codebook):
                hist = history_per_codebook[step]
                if hist:
                    hist_t = torch.tensor(hist, dtype=torch.long, device=logits.device)
                    sel = logits.index_select(-1, hist_t)
                    pos = sel > 0
                    sel = torch.where(pos, sel / repetition_penalty, sel * repetition_penalty)
                    logits.index_copy_(-1, hist_t, sel)

            codes[:, step] = _sample_token(logits, temperature, top_k, top_p, do_sample)

            if step + 1 < rvq:
                embeds[:, step + 1, :] = codec_embeds[step](codes[:, step].view(B, 1)).view(B, hidden_size)

        return codes


def _sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    do_sample: bool,
) -> torch.Tensor:
    """Top-k + top-p sampling (matches upstream's ``sample_token`` for the
    inference branch).
    """
    if not do_sample or temperature <= 0:
        return logits.argmax(dim=-1)

    logits = logits / max(temperature, 1e-6)
    if top_k and top_k > 0 and top_k < logits.shape[-1]:
        top_vals, _ = torch.topk(logits, top_k, dim=-1)
        thresh = top_vals[..., -1:].expand_as(logits)
        logits = torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        # Drop tail beyond top_p (keep at least one token).
        drop = cum > top_p
        drop[..., 1:] = drop[..., :-1].clone()
        drop[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    flat = probs.reshape(-1, probs.shape[-1])
    sampled = torch.multinomial(flat, num_samples=1).reshape(probs.shape[:-1])
    return sampled


__all__ = ["MossTTSRealtimeLocalTransformer"]
