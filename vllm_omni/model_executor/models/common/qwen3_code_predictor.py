"""Qwen3 Code Predictor -- optimized re-prefill, no KV cache.

Shared by Qwen3-Omni and Qwen3-TTS talker models.

* SDPA attention (F.scaled_dot_product_attention) with native GQA support
* HF-compatible numerics (float32 RMSNorm, float32 RoPE, separate linear layers)
* Per-call embedding buffer to avoid cross-request aliasing
* Pre-allocated position_ids (read-only, safe to persist)
* torch.compile (epilogue_fusion=False) on inner transformer by default
* Optional manual CUDA graph capture per batch-size bucket
* Inline sampling (top-k + top-p) -- no custom op overhead
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


# ===================================================================
# HF-numerics-compatible layers for code predictor
# ===================================================================
#
# These use plain PyTorch ops (nn.Linear, manual RMSNorm in float32,
# rotate_half RoPE) to produce outputs numerically identical to the
# HuggingFace reference. vLLM's fused kernels (RMSNorm, QKVParallel,
# get_rope) introduce small precision differences that compound across
# the autoregressive steps of the code predictor, causing severe
# audio quality degradation.
#
# See: https://github.com/vllm-project/vllm-omni/issues/2274


class _RMSNorm(nn.Module):
    """RMSNorm matching HuggingFace's implementation exactly.

    Computes variance in float32 to avoid bfloat16 precision loss.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class _RotaryEmbedding(nn.Module):
    """RoPE matching HuggingFace's implementation exactly.

    Forces float32 computation for cos/sin, matching HF's torch.autocast(enabled=False).
    """

    def __init__(self, config) -> None:
        super().__init__()
        head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        rope_theta = getattr(config, "rope_theta", 10000.0)
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [batch, seq_len]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 (matching HF)
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ===================================================================
#  Attention
# ===================================================================


class CodePredictorAttention(nn.Module):
    """Multi-head self-attention for code predictor.

    Uses ``F.scaled_dot_product_attention`` with HF-compatible RoPE and RMSNorm.
    No KV cache -- the code predictor always re-prefills the full (short)
    sequence each AR step.

    Input : [B, seq_len, hidden_size]
    Output: [B, seq_len, hidden_size]
    """

    def __init__(self, config, *, prefix: str = "") -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        assert self.num_heads % self.num_kv_heads == 0
        self.is_gqa = self.num_kv_heads != self.num_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        self.hidden_size = config.hidden_size
        self.scaling = self.head_dim**-0.5
        self.max_seq = int(config.num_code_groups) + 1

        # Separate q/k/v projections matching HF (no fused packing)
        bias = getattr(config, "attention_bias", False)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_norm = _RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = _RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        if current_omni_platform.is_npu():
            if self.max_seq > 2048:
                raise ValueError(
                    "Qwen3-TTS code predictor NPU fusion attention uses a fixed 2048x2048 "
                    f"causal mask, but max_seq={self.max_seq} exceeds the mask size."
                )
            # Ascend SDPA is_causal migration example uses a fixed 2048x2048
            # compressed causal mask with sparse_mode=2.
            fusion_mask = torch.triu(
                torch.ones(2048, 2048, dtype=torch.bool),
                diagonal=1,
            )
            self.register_buffer("_fusion_causal_mask", fusion_mask, persistent=False)

    def _forward_npu_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bsz: int,
        seq_len: int,
    ) -> torch.Tensor:
        import torch_npu

        q_f, k_f, v_f = q, k, v
        if self.is_gqa:
            k_f = (
                k[:, :, None, :, :]
                .expand(bsz, self.num_kv_heads, self.num_queries_per_kv, seq_len, self.head_dim)
                .reshape(bsz, self.num_heads, seq_len, self.head_dim)
            )
            v_f = (
                v[:, :, None, :, :]
                .expand(bsz, self.num_kv_heads, self.num_queries_per_kv, seq_len, self.head_dim)
                .reshape(bsz, self.num_heads, seq_len, self.head_dim)
            )

        mask = self._fusion_causal_mask
        mask = mask.contiguous()
        q_f = q_f.contiguous()
        k_f = k_f.contiguous()
        v_f = v_f.contiguous()
        return torch_npu.npu_fusion_attention(
            q_f,
            k_f,
            v_f,
            self.num_heads,
            "BNSD",
            pse=None,
            padding_mask=None,
            atten_mask=mask,
            scale=float(self.scaling),
            keep_prob=1.0,
            # Keep torch_npu's API spelling.
            pre_tockens=2147483647,
            next_tockens=2147483647,
            inner_precise=0,
            prefix=None,
            actual_seq_qlen=None,
            actual_seq_kvlen=None,
            # Ascend SDPA is_causal migration example uses sparse_mode=2.
            sparse_mode=2,
            gen_mask_parallel=True,
            # Keep sync=True for the NPU fused attention path.
            sync=True,
        )[0]

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        hidden_shape_q = (bsz, seq_len, self.num_heads, self.head_dim)
        hidden_shape_kv = (bsz, seq_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape_q)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape_kv)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape_kv).transpose(1, 2)

        cos, sin = position_embeddings
        # cos/sin are [batch, seq_len, head_dim], need unsqueeze at dim=1 for heads
        cos = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
        sin = sin.unsqueeze(1)
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)

        if not current_omni_platform.is_npu():
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=self.scaling,
                is_causal=True,
                enable_gqa=self.is_gqa,
            )
        else:
            attn_out = self._forward_npu_attention(q, k, v, bsz, seq_len)

        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj(attn_out)


# ===================================================================
#  MLP
# ===================================================================


class CodePredictorMLP(nn.Module):
    """SiLU-gated MLP for code predictor, matching HF's implementation."""

    def __init__(self, config, *, prefix: str = "") -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


# ===================================================================
#  Decoder Layer
# ===================================================================


class CodePredictorDecoderLayer(nn.Module):
    """Transformer decoder layer (SDPA, no KV cache)."""

    def __init__(self, config, *, prefix: str = "") -> None:
        super().__init__()
        self.self_attn = CodePredictorAttention(config, prefix=f"{prefix}.self_attn")
        self.mlp = CodePredictorMLP(config, prefix=f"{prefix}.mlp")
        self.input_layernorm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ===================================================================
#  Base Transformer Model (re-prefill, no KV cache)
# ===================================================================


class CodePredictorBaseModel(nn.Module):
    """Inner transformer for code predictor.

    Signature: ``forward(inputs_embeds, position_ids) -> hidden_states``
    """

    def __init__(
        self,
        config,
        *,
        embedding_dim: int | None = None,
        use_parallel_embedding: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        emb_dim = int(embedding_dim) if embedding_dim is not None else int(config.hidden_size)
        if use_parallel_embedding:
            self.codec_embedding = nn.ModuleList(
                [VocabParallelEmbedding(config.vocab_size, emb_dim) for _ in range(config.num_code_groups - 1)]
            )
        else:
            self.codec_embedding = nn.ModuleList(
                [nn.Embedding(config.vocab_size, emb_dim) for _ in range(config.num_code_groups - 1)]
            )

        self.layers = nn.ModuleList(
            [
                CodePredictorDecoderLayer(config, prefix=f"{prefix}.layers.{idx}")
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = _RotaryEmbedding(config)

    def get_input_embeddings(self) -> nn.ModuleList:
        return self.codec_embedding

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Run the transformer body in float32 when the model is in fp16.
        # fp16 lacks the dynamic range for stable attention scores and
        # SiLU-gated MLP intermediates, producing NaN on GPUs without
        # native bf16 support (Turing, Volta).  The RMSNorm and RoPE
        # layers already upcast internally; this extends the same
        # treatment to attention and MLP.
        input_dtype = inputs_embeds.dtype
        use_fp32 = input_dtype == torch.float16
        if use_fp32:
            inputs_embeds = inputs_embeds.float()
        hidden_states = inputs_embeds
        with torch.amp.autocast(inputs_embeds.device.type, enabled=use_fp32, dtype=torch.float32):
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            for layer in self.layers:
                hidden_states = layer(hidden_states, position_embeddings)
            hidden_states = self.norm(hidden_states)
        return hidden_states.to(input_dtype)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            param = params_dict.get(name)
            if param is None:
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


# ===================================================================
#  Wrapper Configuration
# ===================================================================


@dataclasses.dataclass
class CodePredictorWrapperConfig:
    """Controls behavioral differences between model-specific code predictors."""

    use_cuda_graphs: bool = False
    use_parallel_embedding: bool = False
    use_projection: bool = False
    return_proj_buf: bool = False
    sampling_mode: str = "stored"


# ===================================================================
#  Code Predictor Wrapper (optimized re-prefill, persistent buffers)
# ===================================================================


class CodePredictorWrapper(nn.Module):
    """Optimized code predictor -- re-prefill approach, no KV cache.

    Each AR step forwards the full growing sequence (len 2 -> num_code_groups+1)
    through the transformer.  The extra O(T^2) FLOPs are negligible for
    short sequences, and this avoids all KV-cache management overhead.

    Optimizations:
      1. Per-call embedding buffer -- avoids cross-request aliasing.
      2. Pre-allocated position_ids -- no torch.arange per step.
      3. Cached module references -- bypass ModuleList indexing.
      4. torch.compile on inner transformer.
      5. Inline sampling (top-k + top-p) -- no custom op overhead.
      6. Optional manual CUDA graph capture per batch-size bucket.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        cp_config,
        wrapper_config: CodePredictorWrapperConfig,
        talker_hidden_size: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self._vllm_config = vllm_config
        self.config = cp_config
        self._wrapper_config = wrapper_config
        self.prefix = prefix

        self._num_groups = int(cp_config.num_code_groups)
        self._cp_hidden = int(cp_config.hidden_size)

        # For Omni backward compat (accessed by the talker)
        self.num_code_groups = self._num_groups

        # Determine embedding dimension
        _talker_hidden = int(talker_hidden_size) if talker_hidden_size is not None else self._cp_hidden

        self.model = CodePredictorBaseModel(
            cp_config,
            embedding_dim=_talker_hidden,
            use_parallel_embedding=wrapper_config.use_parallel_embedding,
            prefix=f"{prefix}.model" if prefix else "model",
        )

        self.lm_head = nn.ModuleList(
            [nn.Linear(cp_config.hidden_size, cp_config.vocab_size, bias=False) for _ in range(self._num_groups - 1)]
        )

        # Projection: Identity when hidden sizes match or not needed
        if wrapper_config.use_projection and _talker_hidden != self._cp_hidden:
            self.small_to_mtp_projection = nn.Linear(_talker_hidden, self._cp_hidden, bias=True)
        else:
            self.small_to_mtp_projection = nn.Identity()

        # Sampling defaults for "stored" mode
        self._top_k: int = 50
        self._top_p: float = 0.8

        # Lazily initialised state
        self._proj_buf: torch.Tensor | None = None
        self._model_dtype: torch.dtype | None = None
        self._compiled_model_fwd = None
        self._bucket_sizes: list[int] = []
        self._bucket_pos_ids: dict[int, torch.Tensor] = {}
        self._lm_heads_list: list[nn.Module] | None = None
        self._codec_embeds_list: list[nn.Module] | None = None
        self._device_graphs: dict[int, tuple] = {}  # (graph, static_output) per bucket

    def get_input_embeddings(self) -> nn.ModuleList:
        return self.model.get_input_embeddings()

    def set_sampling_params(self, top_k: int = 50, top_p: float = 0.8) -> None:
        """Configure sampling parameters to maintain consistency with previous implementation."""
        self._top_k = top_k
        self._top_p = top_p
        logger.debug("Sampling parameters updated: top_k=%d, top_p=%.2f", top_k, top_p)

    # ------------------------------------------------------------------
    #  Lazy-init helpers
    # ------------------------------------------------------------------

    def _ensure_buffers(self, device: torch.device, dtype: torch.dtype, bsz: int) -> None:
        """Ensure the projection buffer can hold at least *bsz* rows."""
        max_seq = self._num_groups + 1
        if (
            self._proj_buf is not None
            and self._proj_buf.device == device
            and self._proj_buf.dtype == dtype
            and self._proj_buf.shape[0] >= bsz
        ):
            return
        self._proj_buf = torch.zeros(bsz, max_seq, self._cp_hidden, dtype=dtype, device=device)

    def _setup_compile(self) -> None:
        """Lazily set up torch.compile with optional device graph capture."""
        if self._compiled_model_fwd is not None:
            return

        # Cache model parameter dtype so forward() doesn't need to query it
        # on every call.  Also ensures warmup buffers match model precision
        # even when upstream modules produce a different dtype (#2385).
        self._model_dtype = next(self.model.parameters()).dtype
        self._lm_heads_list = list(self.lm_head)
        self._codec_embeds_list = list(self.model.codec_embedding)

        if not current_omni_platform.supports_torch_inductor():
            # NPU or other platforms without Inductor support
            self._compiled_model_fwd = self.model.forward

            if current_omni_platform.is_npu() and self._wrapper_config.use_cuda_graphs:
                # For NPU, use eager + NPU graphs (no torch.compile)
                self._warmup_buckets()
                self._capture_npu_graphs()
                logger.info("code_predictor: eager mode + NPU graphs")
            else:
                logger.warning_once("code_predictor: torch.compile disabled")
            return

        # torch.compile fuses RMSNorm/RoPE in ways that lose float32
        # precision, compounding across AR steps. Use epilogue_fusion=False
        # to disable the problematic fusions while still getting kernel
        # fusion benefits for the linear layers and SDPA.
        self._compiled_model_fwd = torch.compile(
            self.model.forward,
            dynamic=False,
            options={"epilogue_fusion": False},
        )
        self._warmup_buckets()

        if self._wrapper_config.use_cuda_graphs:
            self._capture_cuda_graphs()
            logger.info("code_predictor: torch.compile (no epilogue fusion) + CUDA graphs")
        else:
            logger.info("code_predictor: torch.compile (dynamic=False, no epilogue fusion)")

    def _padded_bsz(self, bsz: int) -> int:
        """Round batch size up to nearest power-of-2 bucket."""
        for bucket in self._bucket_sizes:
            if bsz <= bucket:
                return bucket
        return bsz

    def _warmup_buckets(self) -> None:
        """Warmup power-of-2 batch-size buckets to front-load Inductor compilation."""
        max_bsz = self._vllm_config.scheduler_config.max_num_seqs
        bucket_sizes = [1 << i for i in range(max_bsz.bit_length()) if (1 << i) <= max_bsz]
        if max_bsz not in bucket_sizes:
            bucket_sizes.append(max_bsz)
        self._bucket_sizes = sorted(bucket_sizes)

        max_seq = self._num_groups + 1
        device = next(self.model.parameters()).device

        # Ensure proj_buf matches model parameter dtype to avoid dtype
        # mismatch during warmup compilation (see #2385).
        self._ensure_buffers(device, self._model_dtype, max(self._bucket_sizes))
        proj_buf = self._proj_buf

        for bsz in self._bucket_sizes:
            pos_ids = torch.arange(max_seq, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1).contiguous()
            self._bucket_pos_ids[bsz] = pos_ids
            for _ in range(3):
                self._compiled_model_fwd(proj_buf[:bsz, :max_seq, :], pos_ids)
        logger.info("code_predictor: warmup done for buckets %s", self._bucket_sizes)

    def _capture_cuda_graphs(self) -> None:
        """Capture a CUDA graph per bucket using vLLM's global graph pool."""
        from vllm.platforms import current_platform

        pool = current_platform.get_global_graph_pool()
        max_seq = self._num_groups + 1
        proj_buf = self._proj_buf

        for bsz in self._bucket_sizes:
            static_input = proj_buf[:bsz, :max_seq, :]
            pos_ids = self._bucket_pos_ids[bsz]

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                static_output = self._compiled_model_fwd(static_input, pos_ids)

            self._device_graphs[bsz] = (g, static_output)

        logger.info("code_predictor: captured CUDA graphs for buckets %s", self._bucket_sizes)

    def _capture_npu_graphs(self) -> None:
        """Capture an NPU graph per bucket using torch_npu's NPUGraph."""
        max_seq = self._num_groups + 1
        proj_buf = self._proj_buf
        pool = torch.npu.graph_pool_handle()

        for bsz in self._bucket_sizes:
            static_input = proj_buf[:bsz, :max_seq, :]
            pos_ids = self._bucket_pos_ids[bsz]

            g = torch.npu.NPUGraph()
            with torch.npu.graph(g, pool=pool):
                static_output = self._compiled_model_fwd(static_input, pos_ids)

            self._device_graphs[bsz] = (g, static_output)

        logger.info("code_predictor: captured NPU graphs for buckets %s", self._bucket_sizes)

    # ------------------------------------------------------------------
    #  Forward -- re-prefill + inline sampling
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def forward(
        self,
        layer0_code: torch.Tensor,
        layer0_embed: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Predict residual codebooks 1..G-1 autoregressively via re-prefill."""
        bsz = int(layer0_code.shape[0])
        num_groups = self._num_groups
        device = layer0_code.device

        # _setup_compile caches _model_dtype on first call; use it for buffers
        # so they always match model weight precision (#2385).
        self._setup_compile()
        dtype = self._model_dtype

        padded_bsz = self._padded_bsz(bsz)
        self._ensure_buffers(device, dtype, padded_bsz)

        proj_buf = self._proj_buf
        max_seq = num_groups + 1
        projection = self.small_to_mtp_projection
        model_fwd = self._compiled_model_fwd
        lm_heads = self._lm_heads_list
        codec_embeds = self._codec_embeds_list

        # Zero the padded region of the buffer
        proj_buf[:padded_bsz].zero_()

        # Fill buffer positions 0 (talker hidden) & 1 (layer0 embed)
        proj_buf[:bsz, 0, :] = projection(last_talker_hidden.reshape(bsz, 1, -1).to(dtype)).reshape(bsz, -1)
        proj_buf[:bsz, 1, :] = projection(layer0_embed.reshape(bsz, 1, -1).to(dtype)).reshape(bsz, -1)

        # Get pre-computed pos_ids for this bucket
        full_pos_ids = self._bucket_pos_ids.get(padded_bsz)
        if full_pos_ids is None:
            full_pos_ids = (
                torch.arange(max_seq, device=device, dtype=torch.long).unsqueeze(0).expand(padded_bsz, -1).contiguous()
            )

        # Use captured device graph if available, otherwise call compiled fn.
        device_graph_entry = self._device_graphs.get(padded_bsz)

        # Prepare sampling parameters
        stored_mode = self._wrapper_config.sampling_mode == "stored"
        if stored_mode:
            s_top_k = self._top_k
            s_top_p = self._top_p
        else:
            use_sampling = do_sample and temperature > 0
            inv_temperature = 1.0 / max(temperature, 1e-6) if use_sampling else 0.0
            if use_sampling and top_p != 1.0:
                raise NotImplementedError(
                    "top_p sampling is not implemented for the vLLM-native code predictor; please set top_p=1.0."
                )

        # Output codes -- shape depends on return mode
        if self._wrapper_config.return_proj_buf:
            all_codes = torch.empty(bsz, num_groups, 1, dtype=torch.int64, device=device)
            all_codes[:, 0] = layer0_code.reshape(bsz, -1)[:, :1]
        else:
            all_codes = torch.empty(bsz, num_groups, dtype=torch.long, device=device)
            all_codes[:, 0] = layer0_code.reshape(bsz)

        # Autoregressive loop: predict layers 1..G-1
        for step in range(1, num_groups):
            # Run transformer (device graph replay or compiled forward)
            if device_graph_entry is not None:
                device_graph_entry[0].replay()
                hidden_out = device_graph_entry[1]
            else:
                hidden_out = model_fwd(proj_buf[:padded_bsz, :max_seq, :], full_pos_ids)

            logits = lm_heads[step - 1](hidden_out[:bsz, step, :])

            # Sample next code
            if stored_mode:
                # "stored" mode: top-k -> top-p -> softmax -> multinomial
                if s_top_k > 0:
                    topk_vals, _ = logits.topk(s_top_k, dim=-1)
                    logits = logits.masked_fill(logits < topk_vals[:, -1:], float("-inf"))
                if s_top_p < 1.0:
                    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
                    sorted_probs = F.softmax(sorted_logits, dim=-1, dtype=torch.float32)
                    cumulative_probs = sorted_probs.cumsum(dim=-1)
                    remove_mask = (cumulative_probs - sorted_probs) >= s_top_p
                    sorted_logits[remove_mask] = float("-inf")
                    logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
                probs = F.softmax(logits, dim=-1, dtype=torch.float32)
                code = torch.multinomial(probs, num_samples=1)
            else:
                # "per_call" mode: temperature-scaled + top-k
                if use_sampling:
                    scaled = logits * inv_temperature
                    if top_k > 0:
                        topk_vals, _ = scaled.topk(top_k, dim=-1)
                        scaled = scaled.masked_fill(scaled < topk_vals[:, -1:], float("-inf"))
                    probs = F.softmax(scaled, dim=-1, dtype=torch.float32)
                    code = torch.multinomial(probs, num_samples=1)
                else:
                    code = logits.argmax(dim=-1, keepdim=True)

            # Store code
            if self._wrapper_config.return_proj_buf:
                all_codes[:, step] = code
            else:
                all_codes[:, step] = code.reshape(bsz)

            # Embed predicted code -> project -> next buffer position
            if step < num_groups - 1 or self._wrapper_config.return_proj_buf:
                new_embed = codec_embeds[step - 1](code)
                proj_buf[:bsz, step + 1, :] = projection(new_embed.reshape(bsz, 1, -1)).reshape(bsz, -1)

        if self._wrapper_config.return_proj_buf:
            return all_codes, proj_buf[:bsz].clone()
        return all_codes

    # ------------------------------------------------------------------
    #  Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights directly (no fused projection remapping needed)."""
        loaded: set[str] = set()
        model_weights: list[tuple[str, torch.Tensor]] = []
        other_weights: list[tuple[str, torch.Tensor]] = []

        for name, w in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if name.startswith("model."):
                model_weights.append((name[len("model.") :], w))
            else:
                other_weights.append((name, w))

        loaded_model = self.model.load_weights(model_weights)
        loaded |= {f"model.{n}" for n in loaded_model}

        params = dict(self.named_parameters(remove_duplicate=False))
        for name, w in other_weights:
            param = params.get(name)
            if param is None:
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, w)
            loaded.add(name)

        return loaded
