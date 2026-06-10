# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Qwen3-TTS helpers for the 310P NPU path."""

from __future__ import annotations

from typing import Any

import torch

_QWEN3_TTS_TALKER_ARCH = "Qwen3TTSTalkerForConditionalGeneration"


def runtime_dtype(device: torch.device, *, default: torch.dtype = torch.bfloat16) -> torch.dtype:
    """Return Qwen3-TTS runtime dtype; only 310P NPU uses fp16."""
    from vllm_omni.platforms.npu._310p import is_310p

    return torch.float16 if is_310p(device) else default


def audio_frontend_runtime(
    device: torch.device,
    *,
    default: torch.dtype = torch.bfloat16,
) -> tuple[torch.device, torch.dtype]:
    """Return where Qwen3-TTS audio frontend/encoder work can execute."""
    from vllm_omni.platforms.npu._310p import is_310p

    # 310P does not support this Qwen3-TTS audio frontend path; run it on CPU.
    if is_310p(device):
        return torch.device("cpu"), torch.float32
    return device, runtime_dtype(device, default=default)


def is_qwen3_tts_talker_model(model_config: Any) -> bool:
    return getattr(model_config, "model_arch", None) == _QWEN3_TTS_TALKER_ARCH


def use_qwen3_tts_talker_310p_path(model_config: Any) -> bool:
    from vllm_omni.platforms.npu._310p import is_310p

    return is_310p() and is_qwen3_tts_talker_model(model_config)


def aligned_code_predictor_seq_len(num_code_groups: int) -> int:
    """Include the text step, then align the flash-attention token axis to 16."""
    return _align_up(int(num_code_groups) + 1, 16)


class CodePredictorAttentionMask:
    """Small 310P causal mask cache for the Qwen3-TTS code predictor."""

    def __init__(self, device: torch.device, max_seq_len: int) -> None:
        self.device = device
        self.max_seq_len = int(max_seq_len)
        self._causal_mask: torch.Tensor | None = None

    def causal_mask(self) -> torch.Tensor:
        if self._causal_mask is None:
            import torch_npu
            from vllm_ascend._310p.attention.attention_mask import AttentionMaskBuilder310
            from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, nd_to_nz_2d

            mask = AttentionMaskBuilder310.gen_causal_additive_mask(self.max_seq_len, self.device)
            self._causal_mask = torch_npu.npu_format_cast(nd_to_nz_2d(mask), ACL_FORMAT_FRACTAL_NZ)
        return self._causal_mask


def build_code_predictor_attention_mask(device: torch.device, max_seq_len: int) -> CodePredictorAttentionMask:
    return CodePredictorAttentionMask(device, max_seq_len)


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
        mask=mask_builder.causal_mask(),
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
