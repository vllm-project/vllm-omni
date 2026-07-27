# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import dataclass, replace

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)

logger = init_logger(__name__)


def _validate_control(value, name: str, lo: float, hi: float | None) -> float | None:
    if value is None:
        return None
    v = float(value)
    if not math.isfinite(v) or v < lo or (hi is not None and v > hi):
        rng = f"in [{lo}, {hi}]" if hi is not None else f">= {lo}"
        raise ValueError(f"{name} must be finite and {rng}; got {value!r}.")
    return v


@dataclass(frozen=True)
class SkipSoftmaxConfig:
    threshold: float | None = None
    target_sparsity: float | None = None
    disabled_until_timestep: float = 0.0
    a: float | None = None
    b: float | None = None

    @classmethod
    def from_backend_kwargs(cls, backend_kwargs: dict | None) -> "SkipSoftmaxConfig":
        bk = backend_kwargs or {}
        return cls(
            threshold=_validate_control(bk.get("skip_softmax_threshold"), "skip_softmax_threshold", 0.0, None),
            target_sparsity=_validate_control(bk.get("target_sparsity"), "target_sparsity", 0.0, 1.0),
            disabled_until_timestep=_validate_control(
                bk.get("disabled_until_timestep", 0.0), "disabled_until_timestep", 0.0, 1.0
            ),
        )

    @property
    def enabled(self) -> bool:
        return self.threshold is not None or (
            self.target_sparsity is not None and self.a is not None and self.b is not None
        )

    @property
    def configured(self) -> bool:
        return self.threshold is not None or self.target_sparsity is not None

    @property
    def gated(self) -> bool:
        return self.disabled_until_timestep > 0.0

    def resolve_factor(self, seqlen: int, timestep: float | None) -> float | None:
        if self.threshold is not None:
            factor = self.threshold * seqlen
        elif self.target_sparsity is not None and self.a is not None and self.b is not None:
            factor = self.a * math.exp(self.b * self.target_sparsity)
        else:
            return None
        if self.gated and timestep is not None and timestep > self.disabled_until_timestep:
            return None
        return factor


try:
    from flashinfer.prefill import trtllm_ragged_attention_deepseek

    HAS_FLASHINFER = True
except Exception as e:  # pragma: no cover - import guard
    HAS_FLASHINFER = False
    logger.warning(
        "FlashInfer is unavailable; TRTLLM_ATTN backend will not work. Reason: %s",
        e,
    )


def _workspace_bytes() -> int:
    import vllm.envs as envs

    return getattr(envs, "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE", 394 * 1024 * 1024)


class TrtllmAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [128]

    @staticmethod
    def get_name() -> str:
        return "TRTLLM_ATTN"

    @staticmethod
    def get_impl_cls() -> type["TrtllmAttentionImpl"]:
        return TrtllmAttentionImpl


class TrtllmAttentionImpl(AttentionImpl):
    _workspace: torch.Tensor | None = None

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        qkv_layout: str | None = None,
        backend_kwargs: dict | None = None,
        **extra_impl_args,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads

        self.skip = SkipSoftmaxConfig.from_backend_kwargs(backend_kwargs)
        self._warned_missing_timestep = False

    def set_layer_calibration(self, a: float, b: float) -> None:
        self.skip = replace(self.skip, a=a, b=b)

    def _resolve_skip_factor(self, seqlen: int) -> float | None:
        if not self.skip.enabled:
            return None

        timestep = None
        if self.skip.gated:
            from vllm_omni.diffusion.forward_context import get_forward_context

            timestep = getattr(get_forward_context(), "denoise_timestep", None)
            if timestep is None:
                if not self._warned_missing_timestep:
                    logger.warning(
                        "TRTLLM skip: disabled_until_timestep=%s set but this pipeline does not "
                        "publish denoise_timestep; staying dense. Have the pipeline call "
                        "DenoiseProgressMixin.record_denoise_step to enable timestep gating.",
                        self.skip.disabled_until_timestep,
                    )
                    self._warned_missing_timestep = True
                return None
        return self.skip.resolve_factor(seqlen, timestep)

    @classmethod
    def _get_workspace(cls, device: torch.device) -> torch.Tensor:
        nbytes = _workspace_bytes()
        ws = cls._workspace
        if ws is None or ws.device != device or ws.numel() < nbytes:
            ws = torch.zeros(nbytes, dtype=torch.uint8, device=device)
            cls._workspace = ws
        return ws

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        attn_mask = getattr(attn_metadata, "attn_mask", None) if attn_metadata is not None else None
        if attn_mask is not None:
            raise ValueError(
                "TRTLLM_ATTN does not support attn_mask (Wan sets one only under SP with "
                "mask_sp_padding=True and a non-divisible seq len). Either set "
                "parallel_config.mask_sp_padding=False, or select a mask-capable backend "
                "(e.g. CUDNN_ATTN / TORCH_SDPA)."
            )

        if not HAS_FLASHINFER:
            raise ImportError(
                "TRTLLM_ATTN backend requires flashinfer. Install it or select "
                "another backend via --diffusion-attention-backend."
            )

        batch, q_len, num_q_heads, head_dim = query.shape
        kv_len, num_kv_heads = key.shape[1], key.shape[2]

        device = query.device

        q = query.reshape(batch * q_len, num_q_heads, head_dim).contiguous()
        k = key.reshape(batch * kv_len, num_kv_heads, head_dim).contiguous()
        v = value.reshape(batch * kv_len, num_kv_heads, head_dim).contiguous()

        seq_lens = torch.full((batch,), kv_len, dtype=torch.int32, device=device)
        cu_seq_lens_q = torch.arange(0, (batch + 1) * q_len, step=q_len, dtype=torch.int32, device=device)
        cu_seq_lens_kv = torch.arange(0, (batch + 1) * kv_len, step=kv_len, dtype=torch.int32, device=device)
        workspace = self._get_workspace(device)

        bmm1_scale = self.softmax_scale
        bmm2_scale = 1.0

        _skip_factor = self._resolve_skip_factor(kv_len)

        out = trtllm_ragged_attention_deepseek(
            query=q,
            key=k,
            value=v,
            workspace_buffer=workspace,
            seq_lens=seq_lens,
            max_q_len=q_len,
            max_kv_len=kv_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            o_sf_scale=-1.0,
            batch_size=batch,
            window_left=-1,
            cum_seq_lens_q=cu_seq_lens_q,
            cum_seq_lens_kv=cu_seq_lens_kv,
            enable_pdl=False,
            is_causal=self.causal,
            return_lse=False,
            skip_softmax_threshold_scale_factor=_skip_factor,
        )
        return out.reshape(batch, q_len, num_q_heads, head_dim)
