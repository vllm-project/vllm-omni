# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)

logger = init_logger(__name__)

try:
    from flashinfer.prefill import (
        # Templated single prefill
        single_prefill_with_kv_cache,
    )
    from flashinfer.prefill import (
        # TRTLLM-optimized kernels for DeepSeek and DiT models
        trtllm_ragged_attention_deepseek as _trtllm_ragged_attention_deepseek,
    )
    from flashinfer.prefill import (
        trtllm_sage_attention_quantize as _trtllm_sage_attention_quantize,
    )

    trtllm_ragged_attention_deepseek = torch.compiler.disable(_trtllm_ragged_attention_deepseek)
    trtllm_sage_attention_quantize = torch.compiler.disable(_trtllm_sage_attention_quantize)

    HAS_FLASHINFER = True
except Exception as e:
    HAS_FLASHINFER = False
    trtllm_ragged_attention_deepseek, trtllm_sage_attention_quantize = None, None
    logger.warning(
        "FlashInfer is unavailable; FLASHINFER_ATTN backend will not work. Reason: %s",
        e,
    )


class FlashInferAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @classmethod
    def supports_attention_mask(cls) -> bool:
        # The single-prefill path accepts a boolean custom mask; the direct
        # quantized path falls back to SDPA when a nontrivial mask is present.
        return True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        # FlashInfer dense prefill is well-tested for these head_dims on
        # Ampere/Hopper/Blackwell. Covers the dominant diffusion DiT shapes
        # (SD3 = 64, Flux/HV/Wan = 128, joint-attn = 256).
        return [64, 128, 256]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_ATTN"

    @staticmethod
    def get_impl_cls() -> type["FlashInferAttentionImpl"]:
        return FlashInferAttentionImpl


class FlashInferAttentionImpl(AttentionImpl):
    _QK_DTYPES = {torch.float16, torch.bfloat16, torch.int8}
    _VO_DTYPES = {torch.float16, torch.bfloat16, torch.float8_e4m3fn}

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        backend_kwargs: dict | None = None,
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.device = torch.device("cuda", torch.accelerator.current_device_index())
        backend_kwargs = backend_kwargs or {}
        self.dtype_qk = self._check_dtype(backend_kwargs.get("dtype_qk"), "dtype_qk", self._QK_DTYPES)
        self.dtype_vo = self._check_dtype(backend_kwargs.get("dtype_vo"), "dtype_vo", self._VO_DTYPES)
        self.sage_q_block_size = backend_kwargs.get("sage_q_block_size")
        self.sage_k_block_size = backend_kwargs.get("sage_k_block_size")
        self.is_sage = self.sage_q_block_size or self.sage_k_block_size
        requested_backend = backend_kwargs.get("flashinfer_backend", "auto")

        if not HAS_FLASHINFER:
            raise ImportError("FLASHINFER_ATTN backend requires flashinfer")

        sm_major, _ = torch.cuda.get_device_capability(self.device)
        self.use_trtllm_ragged = head_size == 128 and sm_major == 10
        self._workspace: torch.Tensor | None = None
        if self.use_trtllm_ragged:
            if requested_backend == "auto":
                self.flashinfer_backend = "trtllm-gen" if self.is_sage else "cute-dsl"
            else:
                self.flashinfer_backend = requested_backend
            if self.flashinfer_backend == "trtllm-gen":
                self._workspace = torch.zeros(256 * 1024 * 1024, device=self.device, dtype=torch.uint8)

            if self.is_sage and self.dtype_vo != torch.float8_e4m3fn:
                raise ValueError("SageAttention requires QK in {FP8, INT8}, V in {FP8}.")
            if self.dtype_qk or self.dtype_vo:
                logger.info_once(
                    "FLASHINFER_ATTN dtype override: Q/K=%s, V=%s.",
                    self.dtype_qk,
                    self.dtype_vo,
                )
        else:
            self.flashinfer_backend = requested_backend
            if self.dtype_qk or self.dtype_vo or self.is_sage:
                raise ValueError(f"Quantization options are not supported for {head_size=} on cc={sm_major}")

        logger.info_once(
            "FLASHINFER_ATTN initialized path=%s backend=%s on %s.",
            "trtllm-ragged" if self.use_trtllm_ragged else "single-prefill",
            self.flashinfer_backend,
            self.device,
        )

    @classmethod
    def _check_dtype(
        cls,
        dtype: torch.dtype | str | None,
        option_name: str,
        allowed: set[torch.dtype],
    ) -> torch.dtype | None:
        if dtype is None:
            return None
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype) # "bfloat16" -> torch.bfloat16
        if dtype not in allowed:
            choices = ", ".join(sorted(str(item) for item in allowed))
            raise ValueError(f"Unsupported {option_name}={dtype}; expected one of: {choices}")
        return dtype

    @staticmethod
    def _pack_mask_for_flashinfer(
        attn_mask: torch.Tensor, batch_size: int, qo_len: int, kv_len: int
    ) -> torch.Tensor | None:
        """Convert a diffusion-style attn_mask into the boolean form
        FlashInfer's ``custom_mask`` expects (``True`` = keep).

        Returns either ``(qo_len, kv_len)`` (shared across the batch) or
        ``(batch_size, qo_len, kv_len)`` (per-sample), or ``None`` when the
        mask is all-keep (elide). Only boolean masks are handled here;
        additive/float masks raise ``ValueError`` so the caller falls back to
        SDPA, which applies them with the correct softmax semantics. Shape
        mismatches also raise ``ValueError``.
        """
        mask = attn_mask
        if mask.dtype != torch.bool:
            # Additive masks (0 / -inf / -1e4 / finfo.min) cannot be faithfully
            # reduced to a boolean keep-mask here; SDPA handles them correctly.
            raise ValueError(
                f"non-boolean attn_mask (dtype={mask.dtype}); FlashInfer custom_mask "
                "is boolean-only — deferring to SDPA"
            )
        # Diffusion masks arrive as (qo,kv), (1,1,kv), (B,1,1,kv), (B,1,qo,kv)
        # or (B,H,qo,kv). The mask is identical across heads, so collapse the
        # head dim, but keep a real batch dim — indexing mask[0] would reuse
        # sample 0's padding for every sample (wrong under CFG / mixed lengths).
        if mask.dim() == 4:
            mask = mask[:, 0]  # (B, qo|1, kv)
        if mask.dim() == 3 and mask.shape[0] == 1:
            mask = mask[0]  # (qo|1, kv) — shared across the batch
        try:
            if mask.dim() >= 3:
                mask = mask.broadcast_to((batch_size, qo_len, kv_len))
            else:
                mask = mask.broadcast_to((qo_len, kv_len))
        except RuntimeError as e:
            raise ValueError(
                f"attn_mask shape {tuple(attn_mask.shape)} cannot broadcast to "
                f"(batch={batch_size}, qo_len={qo_len}, kv_len={kv_len})"
            ) from e
        if mask.all():
            return None
        # ``broadcast_to`` returns a non-contiguous view; materialize for the
        # kernel, which reads from GPU memory directly.
        return mask.contiguous()

    def _sdpa_fallback(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> torch.Tensor:
        from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl

        impl = SDPAImpl(
            num_heads=query.shape[2],
            head_size=query.shape[3],
            softmax_scale=self.softmax_scale,
            causal=self.causal,
        )
        return impl.forward_cuda(query, key, value, attn_metadata)

    def _run_single_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        custom_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        outputs = []
        for batch_idx in range(query.shape[0]):
            kwargs = {
                "causal": self.causal,
                "sm_scale": self.softmax_scale,
                "backend": self.flashinfer_backend,
            }
            if custom_mask is not None:
                kwargs["custom_mask"] = custom_mask if custom_mask.dim() == 2 else custom_mask[batch_idx]
            outputs.append(
                single_prefill_with_kv_cache(
                    query[batch_idx],
                    key[batch_idx],
                    value[batch_idx],
                    **kwargs,
                )
            )
        return torch.stack(outputs, dim=0)

    def _run_trtllm_ragged(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, qo_len, num_q_heads, head_dim = query.shape
        kv_len = key.shape[1]
        num_kv_heads = key.shape[2]
        q = query.reshape(batch_size * qo_len, num_q_heads, head_dim)
        k = key.reshape(batch_size * kv_len, num_kv_heads, key.shape[3])
        v = value.reshape(batch_size * kv_len, num_kv_heads, value.shape[3])

        sage_attn_sfs = (None, None, None, None)
        sage_block_sizes = (0, 0, 0, 0)
        if self.is_sage:
            q, k, v, q_sf, k_sf, v_sf = trtllm_sage_attention_quantize(
                q,
                k,
                v,
                q_block_size=self.sage_q_block_size,
                k_block_size=self.sage_k_block_size,
                qk_quant_dtype=self.dtype_qk,
            )
            sage_attn_sfs = (q_sf, k_sf, None, v_sf)
            sage_block_sizes = (
                self.sage_q_block_size,
                self.sage_k_block_size,
                0,
                1,
            )
        elif self.dtype_qk is not None:
            q = q.to(self.dtype_qk)
            k = k.to(self.dtype_qk)
            v = v.to(self.dtype_vo)

        qo_indptr = torch.arange(
            0,
            (batch_size + 1) * qo_len,
            qo_len,
            device=query.device,
            dtype=torch.int32,
        )
        kv_indptr = torch.arange(
            0,
            (batch_size + 1) * kv_len,
            kv_len,
            device=query.device,
            dtype=torch.int32,
        )
        seq_lens = torch.full((batch_size,), kv_len, device=query.device, dtype=torch.int32)
        out = trtllm_ragged_attention_deepseek(
            query=q,
            key=k,
            value=v,
            workspace_buffer=self._workspace,
            seq_lens=seq_lens,
            max_q_len=qo_len,
            max_kv_len=kv_len,
            bmm1_scale=self.softmax_scale,
            bmm2_scale=1.0,
            o_sf_scale=-1.0,
            batch_size=batch_size,
            window_left=-1,
            cum_seq_lens_q=qo_indptr,
            cum_seq_lens_kv=kv_indptr,
            enable_pdl=False,
            is_causal=self.causal,
            return_lse=False,
            sage_attn_sfs=sage_attn_sfs,
            num_elts_per_sage_attn_blk=sage_block_sizes,
            backend=self.flashinfer_backend,
        )
        out = out.reshape(batch_size, qo_len, num_q_heads, value.shape[3])
        return out.to(query.dtype) if out.dtype != query.dtype else out

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        batch_size = query.shape[0]
        custom_mask = None
        if attn_metadata is not None and attn_metadata.attn_mask is not None:
            try:
                custom_mask = self._pack_mask_for_flashinfer(
                    attn_metadata.attn_mask,
                    batch_size=batch_size,
                    qo_len=query.shape[1],
                    kv_len=key.shape[1],
                )
            except ValueError as error:
                logger.debug("Falling back to SDPA for mask path: %s", error)
                return self._sdpa_fallback(query, key, value, attn_metadata)
            if custom_mask is not None and self.causal:
                return self._sdpa_fallback(query, key, value, attn_metadata)

        if self.use_trtllm_ragged:
            if custom_mask is not None:
                logger.debug("TRT-LLM ragged attention has no custom-mask input; deferring to SDPA")
                return self._sdpa_fallback(query, key, value, attn_metadata)
            return self._run_trtllm_ragged(query, key, value)

        return self._run_single_prefill(query, key, value, custom_mask)
