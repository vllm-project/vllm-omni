# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Qwen3-TTS helpers for the 310P NPU path."""

from __future__ import annotations

import torch


class ModuleTorchDtypeProxy:
    """Module-local torch proxy used by 310P patches to replace bf16 constants."""

    def __init__(self, torch_mod, *, bfloat16_replacement: torch.dtype) -> None:
        self._torch_mod = torch_mod
        self._bfloat16_replacement = bfloat16_replacement

    def __getattr__(self, name: str):
        if name == "bfloat16":
            return self._bfloat16_replacement
        return getattr(self._torch_mod, name)


def patch_module_bfloat16(module) -> None:
    """Replace one target module's bf16 constants without changing global torch."""
    if isinstance(module.torch, ModuleTorchDtypeProxy):
        return
    module.torch = ModuleTorchDtypeProxy(module.torch, bfloat16_replacement=torch.float16)


def runtime_dtype(_device: torch.device) -> torch.dtype:
    """Use fp16 for Qwen3-TTS tensors on the validated 310P path."""
    return torch.float16


def audio_frontend_runtime(_device: torch.device) -> tuple[torch.device, torch.dtype]:
    """Run unsupported 310P audio frontend/encoder paths on CPU fp32."""
    return torch.device("cpu"), torch.float32


def aligned_code_predictor_seq_len(num_code_groups: int) -> int:
    """Include the text step, then align the flash-attention token axis to 16."""
    return _align_up(int(num_code_groups) + 1, 16)


def build_code_predictor_attention_mask(device: torch.device, max_seq_len: int):
    """Reuse vLLM-Ascend's 310P mask builder for code predictor attention."""
    from vllm_ascend._310p.attention.attention_mask import AttentionMaskBuilder310

    original_max_seqlen = AttentionMaskBuilder310.max_seqlen
    builder = AttentionMaskBuilder310(device, int(max_seq_len))
    builder.max_seqlen = int(max_seq_len)
    AttentionMaskBuilder310.max_seqlen = original_max_seqlen
    return builder


def forward_code_predictor_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    mask_builder,
) -> torch.Tensor:
    """Run code predictor attention through the 310P flash attention kernel."""
    if mask_builder is None or mask_builder.device != q.device:
        raise RuntimeError("310P code predictor attention requires a shared attention mask builder.")

    import torch_npu
    from vllm_ascend.utils import aligned_16

    real_tokens = int(batch_size) * int(seq_len)
    output_dtype = q.dtype
    q_f = aligned_16(
        _reshape_flash_tensor(
            q,
            real_tokens=real_tokens,
            num_heads=num_heads,
            head_dim=head_dim,
        )
    )
    k_f = aligned_16(
        _reshape_flash_tensor(
            k,
            real_tokens=real_tokens,
            num_heads=num_kv_heads,
            head_dim=head_dim,
        )
    )
    v_f = aligned_16(
        _reshape_flash_tensor(
            v,
            real_tokens=real_tokens,
            num_heads=num_kv_heads,
            head_dim=head_dim,
        )
    )
    aligned_tokens = int(q_f.shape[0])
    pad_tokens = aligned_tokens - real_tokens
    seq_lens = torch.full((int(batch_size),), int(seq_len), dtype=torch.int32, device="cpu")
    if pad_tokens:
        seq_lens[-1] += pad_tokens

    out = torch.empty(
        (aligned_tokens, int(num_heads), int(head_dim)),
        dtype=torch.float16,
        device=q.device,
    )
    torch_npu._npu_flash_attention(
        query=q_f.contiguous(),
        key=k_f.contiguous(),
        value=v_f.contiguous(),
        mask=mask_builder._get_causal_mask(mask_builder.max_seqlen),
        seq_len=seq_lens,
        scale_value=float(scale),
        num_heads=int(num_heads),
        num_kv_heads=int(num_kv_heads),
        out=out,
    )
    return (
        out[:real_tokens]
        .reshape(int(batch_size), int(seq_len), int(num_heads), int(head_dim))
        .transpose(1, 2)
        .to(output_dtype)
    )


def _align_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _reshape_flash_tensor(
    tensor: torch.Tensor,
    *,
    real_tokens: int,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    return tensor.to(torch.float16).transpose(1, 2).reshape(int(real_tokens), int(num_heads), int(head_dim))
