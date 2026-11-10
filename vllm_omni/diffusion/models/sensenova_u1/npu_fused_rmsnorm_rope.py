# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NPU-native fused QK RMS norm + 3D RoPE for SenseNova-U1.

Eager-path counterpart of ``fused_rmsnorm_rope.py`` (CUDA/Triton). Instead of
expressing each head-vector norm and pairwise rotation with a dozen PyTorch
elementwise kernels, the whole q/k norm+rope chain is collapsed onto two
native Ascend ops:

* ``torch_npu.npu_rms_norm``   fused positional RMS norm (fp32 compute)
* ``torch_npu.npu_rotary_mul`` fused Neox-style RoPE application

Both ops are the ones vllm-ascend uses for compiled LLM inference
(``AscendRMSNorm``/``AscendApplyRotaryEmb``), so they are also safe under
ACL graph capture. The interface is intentionally identical to
``triton_qk_norm_rope`` so callers just select a backend.
"""

from __future__ import annotations

import torch

__all__ = ["npu_qk_norm_rope"]


def _rotary_mul(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply Neox-style RoPE to an ``[B, H, S, D]`` tensor.

    Reference semantics (``apply_rotary_pos_emb`` / ``rotate_half`` in the
    model): with ``x1 = x[..., :D//2]`` and ``x2 = x[..., D//2:]``,
    ``out = x * cos + cat(-x2, x1) * sin``, i.e. pairs ``(i, i + D//2)`` and
    the same ``cos``/``sin`` applied to both halves.  ``npu_rotary_mul`` is
    exactly this Neox pairing. ``cos``/``sin`` are ``[B, S, D]``; reshape to
    ``[B, 1, S, D]`` so they broadcast over the head dim the same way the
    reference does via ``unsqueeze(1)``.
    """
    import torch_npu  # noqa: F401

    cos = cos.reshape(cos.shape[0], 1, cos.shape[1], cos.shape[2])
    sin = sin.reshape(sin.shape[0], 1, sin.shape[1], sin.shape[2])
    return torch_npu.npu_rotary_mul(x, cos, sin)


def _verify_available() -> None:
    """Raise ImportError if the required native NPU ops are missing."""
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:  # pragma: no cover - NPU-only module
        raise ImportError("torch_npu is not installed") from exc
    missing = [name for name in ("npu_rms_norm", "npu_rotary_mul") if not hasattr(torch_npu, name)]
    if missing:
        raise ImportError(f"torch_npu missing required ops: {', '.join(missing)}")


def npu_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    q_norm_hw_weight: torch.Tensor,
    k_norm_hw_weight: torch.Tensor,
    cos_t: torch.Tensor,
    sin_t: torch.Tensor,
    cos_h: torch.Tensor,
    sin_h: torch.Tensor,
    cos_w: torch.Tensor,
    sin_w: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply QK RMS norms + 3D (t/h/w) RoPE with native Ascend ops.

    ``head_dim`` is split into a t half and an h/w half; the h/w half is
    further split into h and w quarters. Each band gets its own RMS norm
    and Neox-style rope.

    Args:
        q: query states, shape ``[B, S, H, head_dim]``.
        k: key states, shape ``[B, S, H, head_dim]``.
        q_norm_weight: RMS norm weight for the t band of q, ``[head_dim/2]``.
        k_norm_weight: RMS norm weight for the t band of k, ``[head_dim/2]``.
        q_norm_hw_weight: RMS norm weight for the h/w band of q,
            ``[head_dim/2]``.
        k_norm_hw_weight: RMS norm weight for the h/w band of k,
            ``[head_dim/2]``.
        cos_t: t-band cos table, ``[B, S, head_dim/2]`` (frequencies already
            duplicated full-width, matching Neox-style applied cos).
        sin_t: t-band sin table, ``[B, S, head_dim/2]``.
        cos_h: h-band cos table, ``[B, S, head_dim/4]``.
        sin_h: h-band sin table, ``[B, S, head_dim/4]``.
        cos_w: w-band cos table, ``[B, S, head_dim/4]``.
        sin_w: w-band sin table, ``[B, S, head_dim/4]``.
        eps: RMS epsilon.

    Returns:
        Query/key states in ``[B, H, S, head_dim]`` layout, ready for SDPA.
    """
    _verify_available()
    import torch_npu  # noqa: F401

    hd = q.shape[-1]
    t_half = hd // 2

    # --- split head_dim into t / hw halves: [B, S, H, D/2] ---
    q_t = q[..., :t_half].contiguous()
    q_hw = q[..., t_half:].contiguous()
    k_t = k[..., :t_half].contiguous()
    k_hw = k[..., t_half:].contiguous()

    # --- fused positional RMS norm per half (fp32 compute on NPU) ---
    q_t, _ = torch_npu.npu_rms_norm(q_t, q_norm_weight, eps)
    q_hw, _ = torch_npu.npu_rms_norm(q_hw, q_norm_hw_weight, eps)
    k_t, _ = torch_npu.npu_rms_norm(k_t, k_norm_weight, eps)
    k_hw, _ = torch_npu.npu_rms_norm(k_hw, k_norm_hw_weight, eps)

    # --- [B, S, H, D/2] -> [B, H, S, D/2] ---
    q_t = q_t.transpose(1, 2)
    k_t = k_t.transpose(1, 2)
    q_hw = q_hw.transpose(1, 2)
    k_hw = k_hw.transpose(1, 2)

    # h / w axis quarters
    q_h, q_w = q_hw.chunk(2, dim=-1)
    k_h, k_w = k_hw.chunk(2, dim=-1)

    # --- fused Neox-style RoPE applied per axis ---
    q_t = _rotary_mul(q_t, cos_t, sin_t)
    k_t = _rotary_mul(k_t, cos_t, sin_t)
    q_h = _rotary_mul(q_h, cos_h, sin_h)
    k_h = _rotary_mul(k_h, cos_h, sin_h)
    q_w = _rotary_mul(q_w, cos_w, sin_w)
    k_w = _rotary_mul(k_w, cos_w, sin_w)

    query_states = torch.cat([q_t, q_h, q_w], dim=-1)  # [B, H, S, hd]
    key_states = torch.cat([k_t, k_h, k_w], dim=-1)
    return query_states, key_states
