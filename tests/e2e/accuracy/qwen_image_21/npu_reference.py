# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""NPU compatibility for the pinned reference, using only upstream computation."""

from contextlib import contextmanager


@contextmanager
def real_rope_on_npu(transformer):
    """Avoid unsupported complex indexing/multiplication in the reference worker.

    Keep upstream position construction and complex frequency lookup on CPU;
    use its existing real-valued rotation on the accelerator. No native model
    code or tensors participate. The patch is confined to this worker process.
    """
    import torch
    from diffusers.models.transformers import transformer_qwenimage21 as upstream

    position = transformer.pos_embed
    original_position = position.forward
    original_rotation = upstream.apply_rotary_emb_qwen

    def cpu_position(img_shapes, image_pad_mask, device):
        return original_position(img_shapes, image_pad_mask.cpu(), device=torch.device("cpu"))

    def real_rotation(x, freqs_cis, use_real=True, use_real_unbind_dim=-1):
        if use_real:
            return original_rotation(x, freqs_cis, use_real, use_real_unbind_dim)
        cos = freqs_cis.real.repeat_interleave(2, dim=-1)
        sin = freqs_cis.imag.repeat_interleave(2, dim=-1)
        # Upstream's real branch expects [B, H, S, D]; its complex branch uses
        # [B, S, H, D]. Preserve the caller's layout and dtype at the boundary.
        return original_rotation(x.transpose(1, 2), (cos, sin), use_real=True).transpose(1, 2)

    try:
        position.forward = cpu_position
        upstream.apply_rotary_emb_qwen = real_rotation
        yield
    finally:
        position.forward = original_position
        upstream.apply_rotary_emb_qwen = original_rotation
