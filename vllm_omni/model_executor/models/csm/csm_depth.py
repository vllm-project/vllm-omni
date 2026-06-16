# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CSM-1B depth decoder: the 31-step inner AR loop (Stage-0 side computation).

This wraps the HF ``CsmDepthDecoderForCausalLM`` 31-step autoregressive loop as
a single ``run(...)`` call so the Stage-0 backbone model can invoke it once per
frame inside its own ``forward()`` (mirroring MiMo-Audio's ``base_local_forward``
inner multi-step decode, ``mimo_audio_llm.py:807-859``). The body is the proven
``_run_depth_loop`` from the single-stage CSM worktree (csm.py:647 @ d92e35f7),
which already passed bit-exact-vs-HF on the clean req-0 run.

Why this lives in Stage 0 (A3 design §2): the backbone's input at frame t+1 is
the sum of the embeddings of all 32 codebooks of frame t (cb0 from the backbone,
cb1..cb31 from this depth decoder). Keeping the depth loop inside Stage 0 makes
the entire per-frame coupling a side-computation of Stage 0's own AR step, so the
inter-stage boundary stays clean (only finished 32-code frames cross to Stage 1).

The depth decoder runs as custom dense torch (its own ``DynamicCache``, reset
every frame, 33 positions) OUTSIDE vLLM PagedAttention. It never advances a
backbone KV position, so it cannot corrupt the paged KV heap (A3 design §1).

Hot-loop discipline (I3): one ``(B, 32)`` device staging tensor filled in place;
no per-step ``.item()`` / ``.cpu()`` host syncs. ``_sample_logits`` casts to fp32
+ ``nan_to_num`` before every sample (the b3 numerics fix).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def sample_logits(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    """Sample one token id per row from logits (REUSE: csm.py:89 ``_sample_logits``).

    Logits are cast to fp32 + ``nan_to_num`` before any softmax/argmax (the b3
    numerics fix: the bf16 backbone can emit +/-inf or NaN that would trip
    ``torch.multinomial``'s device-side assert). ``temperature<=0`` is greedy.
    Op shapes are kept fixed (top-k via masked fill) so a future CUDA-graph
    capture sees a stable reduction order. Returns a ``(B,)`` LongTensor.
    """
    logits = logits.float()
    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
    if temperature is None or temperature <= 0.0:
        return torch.argmax(logits, dim=-1)

    logits = logits / temperature
    if top_k and top_k > 0 and top_k < logits.shape[-1]:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


class CsmDepthDecoder(nn.Module):
    """Single-call wrapper around the CSM depth decoder's 31-step AR loop.

    Holds the HF ``CsmDepthDecoderForCausalLM`` module (built + tied + weight-
    loaded by ``CsmBackboneForConditionalGeneration.load_weights``). ``run()`` is
    the de-risked eager path (A3 design §6 "risk call", option (b)): the 31 steps
    run in plain torch inside one Stage-0 ``forward()``; it is NOT captured as a
    single CUDA-graph primitive (the deploy yaml sets ``enforce_eager`` for the
    first GPU pass).
    """

    def __init__(self, *, num_codebooks: int, hidden_size: int, aux_dtype: torch.dtype) -> None:
        super().__init__()
        # The HF depth module is assigned by the parent after construction so the
        # parent owns the load_weights / tie path (B2 weight-name contract). Kept
        # as a plain attribute (NOT a submodule) so the parent's registered-name
        # bookkeeping stays under ``_depth_decoder`` exactly as the reuse loader
        # credits it.
        self._module = None  # CsmDepthDecoderForCausalLM
        self.num_codebooks = int(num_codebooks)
        self.hidden_size = int(hidden_size)
        self._aux_dtype = aux_dtype

    def set_module(self, module: nn.Module) -> None:
        self._module = module

    @torch.inference_mode()
    def run(
        self,
        *,
        cb0: torch.Tensor,
        backbone_last_hidden_state: torch.Tensor,
        temperature: float,
        top_k: int,
    ) -> torch.Tensor:
        """31-step inner depth-decoder AR loop -> a 32-code frame.

        REUSE VERBATIM: csm.py:647 ``_run_depth_loop`` (d92e35f7). The depth
        decoder (4 layers / d1024 / head_dim128) runs with its own
        ``DynamicCache`` reset every frame (33 positions: backbone hidden at
        position 0, then cb0..cb31 at positions 1..32).

        Args are kept ``(B, *)`` (per-lane B=1 is the correctness anchor) so a
        future across-lane depth-batch can be flipped on by stacking lanes into
        the batch dim without changing the loop body.

        Returns ``(B, 32)`` Long: cb0 from the backbone plus cb1..cb31.
        """
        from transformers.cache_utils import DynamicCache

        depth = self._module
        device = cb0.device
        bsz = cb0.shape[0]
        n_steps = self.num_codebooks - 1  # 31

        # Packed staging (I3): one (B, 32) tensor filled on device; NO per-step
        # host sync inside the loop.
        codes = torch.empty((bsz, self.num_codebooks), dtype=torch.long, device=device)
        codes[:, 0] = cb0

        past = DynamicCache(config=depth.config)
        cur_input = cb0.view(bsz, 1)  # (B, 1) = cb0
        backbone_hs = backbone_last_hidden_state.view(bsz, self.hidden_size).to(self._aux_dtype)

        for step in range(n_steps):
            depth_ids = cur_input
            if step == 0:
                # Seed position 0 from the backbone hidden state. Sequence is
                # [hidden(pos0), cb0]; the placeholder at position 0 is
                # overwritten internally by ``backbone_last_hidden_state``.
                depth_ids = torch.nn.functional.pad(cur_input, (1, 0), value=0)  # (B, 2)
                out = depth(
                    input_ids=depth_ids,
                    backbone_last_hidden_state=backbone_hs,
                    past_key_values=past,
                    use_cache=True,
                    logits_to_keep=1,
                )
            else:
                out = depth(
                    input_ids=depth_ids,
                    past_key_values=past,
                    use_cache=True,
                    logits_to_keep=1,
                )
            past = out.past_key_values
            step_logits = out.logits[:, -1, :]  # (B, vocab)
            next_code = sample_logits(step_logits, temperature, top_k)  # (B,)
            codes[:, step + 1] = next_code
            cur_input = next_code.view(bsz, 1)

        return codes
