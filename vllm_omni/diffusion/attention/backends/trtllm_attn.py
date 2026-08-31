# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import functools
import inspect
import math
from dataclasses import dataclass, replace
from typing import NamedTuple, cast

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    PackedPaddingMetadata,
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
            disabled_until_timestep=cast(
                float,
                _validate_control(bk.get("disabled_until_timestep", 0.0), "disabled_until_timestep", 0.0, 1.0),
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
        qk_quant_dtype = _QK_QUANT_DTYPES[cast(str, self.dtype_qk)]
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


class _PackedLayout(NamedTuple):
    """Active token ranges and kernel metadata for a flattened ragged batch."""

    q_tokens: int
    kv_tokens: int
    cu_seqlens_q: torch.Tensor
    cu_seqlens_kv: torch.Tensor
    batch_size: int
    seq_lens: torch.Tensor
    known_min_kv_length: int | None = None


def _workspace_bytes() -> int:
    import vllm.envs as envs

    return getattr(envs, "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE", 394 * 1024 * 1024)


class TrtllmAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @classmethod
    def supports_packed_mask_free(cls) -> bool:
        return True

    @classmethod
    def supports_multi_doc_packed_varlen(cls) -> bool:
        return True

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

    @staticmethod
    def _prepare_packed_padding_layout(
        physical_batch: int,
        q: torch.Tensor,
        k: torch.Tensor,
        packed_padding: PackedPaddingMetadata,
        extra: dict,
    ) -> _PackedLayout:
        """Prepare a [real, pad] layout from an explicit host-side boundary."""
        valid_q_tokens = packed_padding.q_length
        valid_kv_tokens = packed_padding.kv_length
        max_q_len = extra["max_seqlen_q"]
        max_kv_len = extra["max_seqlen_k"]
        lengths = (valid_q_tokens, valid_kv_tokens, max_q_len, max_kv_len)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in lengths):
            raise ValueError("Mask-free packed TRTLLM attention lengths must be Python integers")
        if physical_batch != 1:
            raise ValueError("Mask-free packed TRTLLM attention requires a single packed batch")

        total_q_tokens = q.shape[0]
        total_kv_tokens = k.shape[0]
        if not 0 < valid_q_tokens <= total_q_tokens:
            raise ValueError(
                "PackedPaddingMetadata.q_length must be within the packed Q sequence, "
                f"got {valid_q_tokens} for length {total_q_tokens}"
            )
        if not 0 < valid_kv_tokens <= total_kv_tokens:
            raise ValueError(
                "PackedPaddingMetadata.kv_length must be within the packed K/V sequence, "
                f"got {valid_kv_tokens} for length {total_kv_tokens}"
            )
        if max_q_len != valid_q_tokens or max_kv_len != valid_kv_tokens:
            raise ValueError("Mask-free packed TRTLLM attention lengths must match max_seqlen_q/k")
        published_kv_length = extra.get("valid_kv_length")
        if published_kv_length is not None:
            if isinstance(published_kv_length, bool) or not isinstance(published_kv_length, int):
                raise ValueError("valid_kv_length must be a Python integer")
            if published_kv_length != valid_kv_tokens:
                raise ValueError("PackedPaddingMetadata.kv_length must match valid_kv_length")

        cu_seq_lens_q = packed_padding.cu_seqlens_q
        cu_seq_lens_kv = packed_padding.cu_seqlens_k
        if cu_seq_lens_q.dtype != torch.int32 or cu_seq_lens_kv.dtype != torch.int32:
            raise ValueError("Mask-free packed TRTLLM attention requires int32 cu_seqlens")
        if cu_seq_lens_q.device != q.device or cu_seq_lens_kv.device != k.device:
            raise ValueError("Mask-free packed TRTLLM attention metadata must be on the Q/K device")
        if cu_seq_lens_q.shape != (2,) or cu_seq_lens_kv.shape != (2,):
            raise ValueError("Mask-free packed TRTLLM attention requires canonical two-element cu_seqlens")
        return _PackedLayout(
            q_tokens=valid_q_tokens,
            kv_tokens=valid_kv_tokens,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_kv=cu_seq_lens_kv,
            batch_size=1,
            # Canonical packed-padding metadata is [0, valid_kv_tokens], so this slice
            # is the one-element sequence-length view expected by the kernel.
            seq_lens=cu_seq_lens_kv[1:],
            known_min_kv_length=valid_kv_tokens,
        )

    @staticmethod
    def _prepare_generic_packed_layout(
        q: torch.Tensor,
        k: torch.Tensor,
        cu_seq_lens_q: torch.Tensor,
        cu_seq_lens_kv: torch.Tensor,
    ) -> _PackedLayout:
        """Validate and normalize a general ragged batch."""
        if int(cu_seq_lens_q[-1].item()) != q.shape[0] or int(cu_seq_lens_kv[-1].item()) != k.shape[0]:
            raise ValueError("Packed TRTLLM attention cu_seqlens must cover all Q/K/V tokens")

        # Packed producers may retain a zero-length trailing segment when the
        # input is already aligned. It is not a real sequence and is removed
        # before deriving the batch size and per-sequence lengths.
        while (
            cu_seq_lens_q.numel() > 2
            and int(cu_seq_lens_q[-1].item()) == int(cu_seq_lens_q[-2].item())
            and int(cu_seq_lens_kv[-1].item()) == int(cu_seq_lens_kv[-2].item())
        ):
            cu_seq_lens_q = cu_seq_lens_q[:-1]
            cu_seq_lens_kv = cu_seq_lens_kv[:-1]

        return _PackedLayout(
            q_tokens=q.shape[0],
            kv_tokens=k.shape[0],
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_kv=cu_seq_lens_kv,
            batch_size=cu_seq_lens_q.numel() - 1,
            seq_lens=(cu_seq_lens_kv[1:] - cu_seq_lens_kv[:-1]).to(dtype=torch.int32).contiguous(),
        )

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
        packed_padding = getattr(attn_metadata, "packed_padding", None) if attn_metadata is not None else None
        if packed_padding is not None and not isinstance(packed_padding, PackedPaddingMetadata):
            raise ValueError("packed_padding must be PackedPaddingMetadata")
        if packed_padding is not None and not has_packed_metadata:
            raise ValueError("PackedPaddingMetadata requires complete packed TRTLLM attention metadata")
        if attn_mask is not None:
            raise ValueError(
                "TRTLLM_ATTN does not support attn_mask. Represent structural suffix padding "
                "with packed-padding metadata, or select a mask-capable backend such as "
                "CUDNN_ATTN or TORCH_SDPA."
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

        known_min_kv_length: int | None = kv_len
        if has_packed_metadata:
            cu_seq_lens_q = extra["cu_seqlens_q"]
            cu_seq_lens_kv = extra["cu_seqlens_k"]
            if cu_seq_lens_q.ndim != 1 or cu_seq_lens_kv.ndim != 1:
                raise ValueError("Packed TRTLLM attention cu_seqlens tensors must be one-dimensional")
            if cu_seq_lens_q.numel() != cu_seq_lens_kv.numel() or cu_seq_lens_q.numel() < 2:
                raise ValueError("Packed TRTLLM attention requires matching non-empty Q and KV sequence batches")

            # PackedPaddingMetadata is an optional producer-published shortcut
            # for structural suffix padding. Inputs without it retain the
            # general cu_seqlens path for arbitrary ragged Q/KV batches.
            if packed_padding is not None:
                packed_layout = self._prepare_packed_padding_layout(
                    physical_batch,
                    q,
                    k,
                    packed_padding,
                    extra,
                )
            else:
                # Generic metadata can describe any number of real sequences,
                # not only a single [real, pad] pair.
                packed_layout = self._prepare_generic_packed_layout(
                    q,
                    k,
                    cu_seq_lens_q,
                    cu_seq_lens_kv,
                )

            q = q[: packed_layout.q_tokens]
            k = k[: packed_layout.kv_tokens]
            v = v[: packed_layout.kv_tokens]
            cu_seq_lens_q = packed_layout.cu_seqlens_q
            cu_seq_lens_kv = packed_layout.cu_seqlens_kv
            batch = packed_layout.batch_size
            seq_lens = packed_layout.seq_lens
            known_min_kv_length = packed_layout.known_min_kv_length
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
        use_sage = False
        if self.quant.enabled:
            if known_min_kv_length is not None:
                sage_lengths_supported = known_min_kv_length >= self.quant.k_block_size
            else:
                sage_lengths_supported = bool(torch.all(seq_lens >= self.quant.k_block_size).item())
            use_sage = sage_lengths_supported
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
