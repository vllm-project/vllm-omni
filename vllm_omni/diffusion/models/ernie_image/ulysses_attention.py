"""
Optimized Ulysses SP processor for NPU.

Uses vLLM's SeqAllToAll4D for efficient communication (AllToAll over heads),
but applies RoPE AFTER AllToAll to ensure correct positions.

Strategy:
1. to_q/k/v: [B, S_local, C] → [B, S_local, H, D]
2. AllToAll over heads: [B, S_local, H, D] → [B, S_global, H/P, D]
3. Apply RoPE (GLOBAL sequence)
4. Compute attention on GLOBAL sequence
5. AllToAll output: [B, S_global, H, D] → [B, S_local, H/P, D]
6. to_out: [B, S_local, H/P, D] → [B, S_local, C]
"""

import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D
from vllm_omni.diffusion.distributed.sp_sharding import sp_gather


class ErnieImageUlyssesAttnProcessorV2:
    """
    Optimized Ulysses SP processor using AllToAll over heads.

    Applies RoPE AFTER AllToAll to ensure correct positions.
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        **kwargs,
    ):
        """AllToAll -> RoPE -> Attention -> AllToAll."""
        from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
        from vllm_omni.diffusion.models.ernie_image.ernie_image_transformer import _apply_rotary_emb

        # hidden_states is [B, S_local, C] from ErnieImageSharedAdaLNBlock

        # Step 1: to_q/k/v
        query, _ = attn.to_q(hidden_states)
        key, _ = attn.to_k(hidden_states)
        value, _ = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        if hasattr(attn, "norm_q") and attn.norm_q is not None:
            query = attn.norm_q(query)
            key = attn.norm_k(key)

        # Step 2: AllToAll over heads
        # [B, S_local, H, D] → [B, S_global, H/P, D]
        sp_group = get_sp_group()
        ulysses_pg = sp_group.ulysses_group

        query = SeqAllToAll4D.apply(ulysses_pg, query, 2, 1, False)
        key = SeqAllToAll4D.apply(ulysses_pg, key, 2, 1, False)
        value = SeqAllToAll4D.apply(ulysses_pg, value, 2, 1, False)

        S_global = query.shape[1]

        # Step 3: Apply RoPE (GLOBAL sequence)
        if image_rotary_emb is not None and isinstance(image_rotary_emb, tuple):
            freqs_cos, freqs_sin = image_rotary_emb
            if freqs_cos.shape[1] != S_global:
                freqs_cos = sp_gather(freqs_cos.transpose(0, 1), dim=0).transpose(0, 1)
                freqs_sin = sp_gather(freqs_sin.transpose(0, 1), dim=0).transpose(0, 1)

            query = _apply_rotary_emb(query, freqs_cos, freqs_sin)
            key = _apply_rotary_emb(key, freqs_cos, freqs_sin)

        # Step 4: Compute attention on GLOBAL sequence
        attn_metadata = None
        if attention_mask is not None and attention_mask.ndim == 4:
            mask_1d_local = attention_mask[:, 0, 0, :].transpose(0, 1)
            mask_1d_full = sp_gather(mask_1d_local, dim=0).transpose(0, 1)
            mask_full = mask_1d_full.unsqueeze(1).unsqueeze(-1).expand(attention_mask.shape[0], 1, S_global, S_global)
            attn_metadata = AttentionMetadata(attn_mask=mask_full)

        output_global = attn.attn(query, key, value, attn_metadata)

        # Step 5: AllToAll output back
        # [B, S_global, H/P, D] → [B, S_local, H, D]
        output_local = SeqAllToAll4D.apply(ulysses_pg, output_global, 1, 2, False)

        # Step 6: Apply to_out
        output_local = output_local.flatten(2, 3)
        output_local = attn.to_out[0](output_local.contiguous())
        output_local = attn.to_out[1](output_local)

        return output_local
