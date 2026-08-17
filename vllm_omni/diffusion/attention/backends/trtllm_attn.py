# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import inspect
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


@functools.lru_cache(maxsize=1)
def _sage_kernel_available() -> bool:
    if not HAS_FLASHINFER:
        return False
    try:
        return "sage_attn_sfs" in inspect.signature(trtllm_ragged_attention_deepseek).parameters
    except (TypeError, ValueError):
        return False


@functools.lru_cache(maxsize=1)
def _sage_quantize_fn():
    try:
        from flashinfer import trtllm_sage_attention_quantize

        return trtllm_sage_attention_quantize
    except Exception:  # pragma: no cover
        return None


_QK_QUANT_DTYPES = {
    "int8": torch.int8,
    "fp8_e4m3": torch.float8_e4m3fn,
}


@dataclass(frozen=True)
class QuantConfig:
    dtype_qk: str | None = None
    q_block_size: int = 1
    k_block_size: int = 16

    @classmethod
    def from_backend_kwargs(cls, backend_kwargs: dict | None) -> "QuantConfig":
        q = (backend_kwargs or {}).get("quant") or {}
        return cls(
            dtype_qk=q.get("dtype_qk"),
            q_block_size=int(q.get("q_block_size", 1) or 1),
            k_block_size=int(q.get("k_block_size", 16) or 16),
        )

    @property
    def enabled(self) -> bool:
        return self.dtype_qk is not None

    def quantize(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, quantize_fn):
        qk_quant_dtype = _QK_QUANT_DTYPES[self.dtype_qk]
        q_q, k_q, v_q, q_sfs, k_sfs, v_sfs = quantize_fn(
            q,
            k,
            v,
            q_block_size=self.q_block_size,
            k_block_size=self.k_block_size,
            qk_quant_dtype=qk_quant_dtype,
        )
        sage_attn_sfs = (q_sfs, k_sfs, None, v_sfs)
        num_elts_per_sage_attn_blk = (self.q_block_size, self.k_block_size, 0, 1)
        return q_q, k_q, v_q, sage_attn_sfs, num_elts_per_sage_attn_blk


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
        role: str = "self",
        **extra_impl_args,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.role = role

        self.skip = SkipSoftmaxConfig.from_backend_kwargs(backend_kwargs)
        self._warned_missing_timestep = False

        self.quant = QuantConfig.from_backend_kwargs(backend_kwargs)
        # Resolve the SAGE quantize fn once at init so the compiled forward path never calls the
        # lru_cache-wrapped getter (which triggers a Dynamo graph break every step).
        self._sage_quantize_fn = None
        if self.quant.enabled:
            if self.quant.dtype_qk not in _QK_QUANT_DTYPES:
                raise RuntimeError(
                    f"TRTLLM_ATTN quant (SAGE) supports dtype_qk in {sorted(_QK_QUANT_DTYPES)}, got "
                    f"{self.quant.dtype_qk!r}. FLASHINFER_ATTN dtypes (float16/bfloat16) are not SAGE."
                )
            if not _sage_kernel_available():
                raise RuntimeError(
                    "TRTLLM_ATTN quant (SAGE) was requested but this FlashInfer build does not "
                    "expose the trtllm-gen sage_attn_sfs kernel path. Install a FlashInfer build "
                    "that provides it, or remove the quant config."
                )
            self._sage_quantize_fn = _sage_quantize_fn()
            if self._sage_quantize_fn is None:
                raise RuntimeError(
                    "TRTLLM_ATTN quant (SAGE) was requested but this FlashInfer build lacks "
                    "trtllm_sage_attention_quantize (added in flashinfer >= 0.6.16rc1). Upgrade "
                    "FlashInfer, or remove the quant config."
                )

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
        extra = getattr(attn_metadata, "extra", {}) if attn_metadata is not None else {}
        packed_keys = ("cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k")
        present_packed_keys = [key for key in packed_keys if key in extra]
        if present_packed_keys and len(present_packed_keys) != len(packed_keys):
            missing = sorted(set(packed_keys) - set(present_packed_keys))
            raise ValueError(f"Incomplete packed TRTLLM attention metadata; missing {missing}")
        has_packed_metadata = len(present_packed_keys) == len(packed_keys)

        attn_mask = getattr(attn_metadata, "attn_mask", None) if attn_metadata is not None else None
        if attn_mask is not None and not has_packed_metadata:
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

        physical_batch, q_len, num_q_heads, head_dim = query.shape
        kv_len, num_kv_heads = key.shape[1], key.shape[2]

        device = query.device

        q = query.reshape(physical_batch * q_len, num_q_heads, head_dim).contiguous()
        k = key.reshape(physical_batch * kv_len, num_kv_heads, head_dim).contiguous()
        v = value.reshape(physical_batch * kv_len, num_kv_heads, head_dim).contiguous()
        output_tokens = q.shape[0]

        if has_packed_metadata:
            cu_seq_lens_q = extra["cu_seqlens_q"]
            cu_seq_lens_kv = extra["cu_seqlens_k"]
            if cu_seq_lens_q.ndim != 1 or cu_seq_lens_kv.ndim != 1:
                raise ValueError("Packed TRTLLM attention cu_seqlens tensors must be one-dimensional")
            if cu_seq_lens_q.numel() != cu_seq_lens_kv.numel() or cu_seq_lens_q.numel() < 2:
                raise ValueError("Packed TRTLLM attention requires matching non-empty Q and KV sequence batches")
            if int(cu_seq_lens_q[-1].item()) != q.shape[0] or int(cu_seq_lens_kv[-1].item()) != k.shape[0]:
                raise ValueError("Packed TRTLLM attention cu_seqlens must cover all Q/K/V tokens")

            # Drop trailing empty sequences retained by packed producers when
            # alignment adds no padding tokens.
            while (
                cu_seq_lens_q.numel() > 2
                and int(cu_seq_lens_q[-1].item()) == int(cu_seq_lens_q[-2].item())
                and int(cu_seq_lens_kv[-1].item()) == int(cu_seq_lens_kv[-2].item())
            ):
                cu_seq_lens_q = cu_seq_lens_q[:-1]
                cu_seq_lens_kv = cu_seq_lens_kv[:-1]

            # A packed prefix mask denotes structural suffix padding, not another
            # document. Exclude it before quantization and attention; in particular,
            # SAGE block quantization must not span the valid/padding boundary.
            if attn_mask is not None:
                if q.shape[0] != k.shape[0] or attn_mask.numel() != q.shape[0]:
                    raise ValueError("Packed TRTLLM prefix masks require equal Q/K lengths")
                flat_mask = attn_mask.reshape(-1).to(dtype=torch.bool)
                valid_tokens = int(flat_mask.sum().item())
                expected_mask = torch.arange(flat_mask.numel(), device=device) < valid_tokens
                if not torch.equal(flat_mask, expected_mask):
                    raise ValueError("TRTLLM_ATTN only supports prefix-valid packed attention masks")

                q_boundary = (cu_seq_lens_q == valid_tokens).nonzero(as_tuple=False)
                kv_boundary = (cu_seq_lens_kv == valid_tokens).nonzero(as_tuple=False)
                if q_boundary.numel() != 1 or kv_boundary.numel() != 1:
                    raise ValueError("Packed TRTLLM prefix length must be a Q/K sequence boundary")
                q_end = int(q_boundary.item()) + 1
                kv_end = int(kv_boundary.item()) + 1
                if q_end != kv_end:
                    raise ValueError("Packed TRTLLM prefix masks require matching Q/K sequence batches")

                q = q[:valid_tokens]
                k = k[:valid_tokens]
                v = v[:valid_tokens]
                cu_seq_lens_q = cu_seq_lens_q[:q_end].contiguous()
                cu_seq_lens_kv = cu_seq_lens_kv[:kv_end].contiguous()

            batch = cu_seq_lens_q.numel() - 1
            seq_lens = (cu_seq_lens_kv[1:] - cu_seq_lens_kv[:-1]).to(dtype=torch.int32).contiguous()
            max_q_len = int(extra["max_seqlen_q"])
            max_kv_len = int(extra["max_seqlen_k"])
        else:
            batch = physical_batch
            seq_lens = torch.full((batch,), kv_len, dtype=torch.int32, device=device)
            cu_seq_lens_q = torch.arange(0, (batch + 1) * q_len, step=q_len, dtype=torch.int32, device=device)
            cu_seq_lens_kv = torch.arange(0, (batch + 1) * kv_len, step=kv_len, dtype=torch.int32, device=device)
            max_q_len = q_len
            max_kv_len = kv_len
        workspace = self._get_workspace(device)

        bmm1_scale = self.softmax_scale
        bmm2_scale = 1.0

        _skip_factor = self._resolve_skip_factor(max_kv_len)

        # SAGE kwargs are only understood by newer FlashInfer builds; pass them exclusively when
        # SAGE quant is active (which already requires the kernel, checked at init) so the dense
        # path stays compatible with older builds that lack these parameters.
        sage_kwargs: dict = {}
        # The SAGE kernel requires every KV sequence to contain at least one full
        # quantization block. Small auxiliary attention sites use the dense kernel.
        use_sage = self.quant.enabled and bool(torch.all(seq_lens >= self.quant.k_block_size).item())
        if self.quant.enabled and not use_sage:
            message = (
                f"TRTLLM_ATTN SAGE quantization is configured for attention role {self.role!r}, but at least one "
                f"KV sequence is shorter than k_block_size={self.quant.k_block_size}. Falling back to dense "
                "attention for this input."
            )
            logger.warning_once(message)
        if use_sage:
            q, k, v, sage_attn_sfs, sage_block_sizes = self.quant.quantize(q, k, v, self._sage_quantize_fn)
            sage_kwargs["sage_attn_sfs"] = sage_attn_sfs
            sage_kwargs["num_elts_per_sage_attn_blk"] = sage_block_sizes

        out = trtllm_ragged_attention_deepseek(
            query=q,
            key=k,
            value=v,
            workspace_buffer=workspace,
            seq_lens=seq_lens,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
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
            **sage_kwargs,
        )
        if out.shape[0] != output_tokens:
            padded_out = torch.zeros(
                (output_tokens, num_q_heads, head_dim),
                dtype=out.dtype,
                device=out.device,
            )
            padded_out[: out.shape[0]] = out
            out = padded_out
        return out.reshape(physical_batch, q_len, num_q_heads, head_dim)
