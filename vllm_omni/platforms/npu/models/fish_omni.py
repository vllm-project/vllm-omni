# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend-specific Drama 3 attention compatibility."""

from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


def _install_npu_eager_qk_rope() -> None:
    """Keep Qwen3.5 attention off the fused Triton kernel on Ascend.

    The codebook tower selects the torch implementation itself. The backbone
    takes upstream's non-fused branch: forcing the flag off in ``__init__``
    means forward never calls ``fused_qk_rmsnorm_rope_gate``. The NPU worker
    reports a CUDA-like platform, so the stock constructor would otherwise
    enable that kernel. The patch is process-wide on ``Qwen3NextAttention``;
    this stage process builds the Drama 3 backbone, and the Triton kernel is
    not valid on Ascend.
    """
    import vllm.model_executor.models.qwen3_next as qwen_next

    attention = qwen_next.Qwen3NextAttention
    if getattr(attention.__init__, "_fish_omni_npu_eager", False):
        return
    original_init = attention.__init__

    def _eager_qk_rope_init(attn, *args, **kwargs):
        original_init(attn, *args, **kwargs)
        attn.use_fused_qk_norm_rope_gate = False

    _eager_qk_rope_init._fish_omni_npu_eager = True  # type: ignore[attr-defined]
    attention.__init__ = _eager_qk_rope_init


def _require_npu_eager_qk_rope(model: Any) -> None:
    """Fail if any module still takes the fused Triton QK-norm path.

    The ``__init__`` patch covers ``Qwen3NextAttention``. A subclass that sets
    the flag again after ``super().__init__``, or a different class that owns
    the same flag, would otherwise reach the Triton kernel on Ascend. One
    check after construction keeps that guarantee without a forward hook
    inside the compiled region. No matching modules means the upstream flag
    may have changed, so eager execution cannot be verified either.
    """
    enabled: list[str] = []
    armed = 0
    for module in model.modules():
        if not hasattr(module, "use_fused_qk_norm_rope_gate"):
            continue
        armed += 1
        if module.use_fused_qk_norm_rope_gate:
            enabled.append(type(module).__name__)
    if enabled:
        names = ", ".join(sorted(set(enabled)))
        raise RuntimeError(
            f"fish_omni: fused QK-norm RoPE stayed enabled on NPU ({names}). The Triton kernel is not valid on Ascend."
        )
    if armed == 0:
        raise RuntimeError(
            "fish_omni: no attention modules expose use_fused_qk_norm_rope_gate on NPU. "
            "Cannot verify eager QK-norm RoPE; check the upstream attention implementation."
        )
    logger.info("fish_omni: eager QK-norm RoPE on NPU (%d attention modules)", armed)
