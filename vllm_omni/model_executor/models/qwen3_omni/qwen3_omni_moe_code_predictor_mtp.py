"""Qwen3-Omni Code Predictor -- optimized re-prefill, no KV cache.

* SDPA attention (F.scaled_dot_product_attention) with native GQA support
* HF-compatible numerics (float32 RMSNorm, float32 RoPE, separate linear layers)
* Per-call embedding buffer to avoid cross-request aliasing
* Pre-allocated position_ids (read-only, safe to persist)
* torch.compile (epilogue_fusion=False) on inner transformer by default
* Inline sampling (top-k + top-p) -- no custom op overhead
"""

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


class Qwen3OmniCodePredictorAttention(nn.Module):
    """Multi-head self-attention for code predictor.

    Uses ``F.scaled_dot_product_attention`` with HF-compatible RoPE and RMSNorm.
    No KV cache -- the code predictor always re-prefills the full (short)
    sequence each AR step.

    Input : [B, seq_len, hidden_size]
    Output: [B, seq_len, hidden_size]
    """

    def __init__(
        self,
        config,
        prefix: str = "",
    ):
        super().__init__()
        cp_cfg = config.code_predictor_config
        self.num_heads = cp_cfg.num_attention_heads
        self.num_kv_heads = cp_cfg.num_key_value_heads
        self.head_dim = getattr(
            cp_cfg,
            "head_dim",
            cp_cfg.hidden_size // cp_cfg.num_attention_heads,
        )
        self.hidden_size = cp_cfg.hidden_size
        self.scaling = self.head_dim**-0.5
        self._use_gqa = self.num_kv_heads != self.num_heads

        # Separate q/k/v projections matching HF (no fused packing)
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )
        self.q_norm = _RMSNorm(self.head_dim, eps=cp_cfg.rms_norm_eps)
        self.k_norm = _RMSNorm(self.head_dim, eps=cp_cfg.rms_norm_eps)

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

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scaling,
            is_causal=True,
            enable_gqa=self._use_gqa,
        )

        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.o_proj(attn_out)
        return output


# ===================================================================
#  MLP
# ===================================================================


class Qwen3OmniCodePredictorMLP(nn.Module):
    """SiLU-gated MLP for code predictor, matching HF's implementation."""

    def __init__(
        self,
        config,
        prefix: str = "",
    ):
        super().__init__()
        hidden_size = config.code_predictor_config.hidden_size
        intermediate_size = config.code_predictor_config.intermediate_size

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


# ===================================================================
#  Decoder Layer
# ===================================================================


class Qwen3OmniCodePredictorDecoderLayer(nn.Module):
    """Transformer decoder layer (SDPA, no KV cache)."""

    def __init__(
        self,
        config,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3OmniCodePredictorAttention(
            config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Qwen3OmniCodePredictorMLP(
            config,
            prefix=f"{prefix}.mlp",
        )
        cp_cfg = config.code_predictor_config
        self.input_layernorm = _RMSNorm(cp_cfg.hidden_size, eps=cp_cfg.rms_norm_eps)
        self.post_attention_layernorm = _RMSNorm(cp_cfg.hidden_size, eps=cp_cfg.rms_norm_eps)

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


class Qwen3OmniCodePredictorBaseModel(nn.Module):
    """Inner transformer for code predictor.

    Signature: ``forward(inputs_embeds, position_ids) -> hidden_states``
    -- plain Tensor in, plain Tensor out (no namedtuple).
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config.code_predictor_config
        self.config = config

        self.codec_embedding = nn.ModuleList(
            [VocabParallelEmbedding(config.vocab_size, config.hidden_size) for _ in range(config.num_code_groups - 1)]
        )

        self.layers = nn.ModuleList(
            [
                Qwen3OmniCodePredictorDecoderLayer(
                    vllm_config.model_config.hf_config,
                    prefix=f"{prefix}.layers.{idx}",
                )
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = _RotaryEmbedding(config)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings)
        hidden_states = self.norm(hidden_states)
        return hidden_states


# ===================================================================
#  Code Predictor Wrapper (optimized re-prefill, persistent buffers)
# ===================================================================


class Qwen3OmniMoeTalkerCodePredictor(nn.Module):
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
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.prefix = prefix

        self.num_code_groups = config.code_predictor_config.num_code_groups
        self._hidden_size = config.code_predictor_config.hidden_size

        self.model = Qwen3OmniCodePredictorBaseModel(
            vllm_config=vllm_config,
            prefix=prefix,
        )

        # One lm_head per residual layer (layers 1 .. G-1)
        self.lm_head = nn.ModuleList(
            [
                nn.Linear(
                    config.code_predictor_config.hidden_size,
                    config.code_predictor_config.vocab_size,
                    bias=False,
                )
                for _ in range(self.num_code_groups - 1)
            ]
        )

        self.set_sampling_params()

        # Lazily initialised position ids (read-only, safe to persist)
        self._pos_ids: torch.Tensor | None = None

        # Cached plain-list refs (set once)
        self._lm_heads: list | None = None
        self._codec_embeds: list | None = None

        # Model forward (optionally compiled)
        self._model_fwd: object | None = None

    def set_sampling_params(self, top_k: int = 50, top_p: float = 0.8):
        """Configure sampling parameters to maintain consistency with previous implementation."""
        self._top_k = top_k
        self._top_p = top_p
        logger.debug(f"Sampling parameters updated: top_k={top_k}, top_p={top_p}s")

    # ------------------------------------------------------------------
    #  Lazy-init helpers
    # ------------------------------------------------------------------

    def _ensure_pos_ids(self, device: torch.device) -> None:
        if self._pos_ids is not None and self._pos_ids.device == device:
            return
        max_seq = self.num_code_groups + 1
        # [1, max_seq] for HF-style RoPE (will be expanded to [bsz, seq_len] at use)
        self._pos_ids = torch.arange(max_seq, dtype=torch.long, device=device).unsqueeze(0)

    def _ensure_cached_refs(self) -> None:
        if self._lm_heads is not None:
            return
        self._lm_heads = list(self.lm_head)
        self._codec_embeds = list(self.model.codec_embedding)

    def _ensure_model_fwd(self) -> None:
        if self._model_fwd is not None:
            return
        if current_omni_platform.supports_torch_inductor():
            # torch.compile fuses RMSNorm/RoPE in ways that lose float32
            # precision, compounding across AR steps. Use epilogue_fusion=False
            # to disable the problematic fusions while still getting kernel
            # fusion benefits for the linear layers and SDPA.
            self._model_fwd = torch.compile(
                self.model.forward,
                dynamic=True,
                options={
                    "epilogue_fusion": False,
                },
            )
            logger.info("code_predictor: torch.compile enabled (no epilogue fusion)")
        else:
            self._model_fwd = self.model.forward
            logger.info("code_predictor: using eager mode (no torch.compile)")

    # ------------------------------------------------------------------
    #  Forward -- re-prefill + inline sampling
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def forward(
        self,
        layer0_code: torch.Tensor,
        layer0_embed: torch.Tensor,
        last_talker_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict residual codebooks 1..G-1 autoregressively via re-prefill.

        Args:
            layer0_code:        [bsz, 1]  int64
            layer0_embed:       [bsz, 1, hidden_size]
            last_talker_hidden: [bsz, 1, hidden_size]

        Returns:
            all_codes: [bsz, num_code_groups, 1]
            proj_buf:  [bsz, num_code_groups + 1, hidden_size]
                pos 0   = last_talker_hidden (NOT a codec embed)
                pos 1   = layer0_embed
                pos 2.. = `codec_embedding[i](predicted_code_i)`
        """
        bsz = int(layer0_code.shape[0])
        device = layer0_code.device
        dtype = last_talker_hidden.dtype
        num_groups = self.num_code_groups

        # Lazy init (read-only caches only)
        self._ensure_pos_ids(device)
        self._ensure_model_fwd()
        self._ensure_cached_refs()

        # Allocate proj_buf locally each call to avoid cross-call aliasing
        max_seq = num_groups + 1
        proj_buf = torch.zeros(bsz, max_seq, self._hidden_size, dtype=dtype, device=device)
        pos_ids = self._pos_ids
        model_fwd = self._model_fwd
        lm_heads = self._lm_heads
        codec_embeds = self._codec_embeds

        # Output codes
        all_codes = torch.empty(bsz, num_groups, 1, dtype=torch.int64, device=device)
        all_codes[:, 0] = layer0_code

        # Fill buffer positions 0 & 1
        proj_buf[:bsz, 0:1, :] = last_talker_hidden
        proj_buf[:bsz, 1:2, :] = layer0_embed

        # Autoregressive loop: predict layers 1..G-1
        for step in range(1, num_groups):
            seq_len = step + 1
            projected = proj_buf[:bsz, :seq_len, :]
            # position_ids: [batch, seq_len] for HF-style RoPE
            step_pos_ids = pos_ids[:, :seq_len].expand(bsz, -1)

            hidden_out = model_fwd(projected, step_pos_ids)

            # Inline sampling: top-k -> top-p -> softmax -> multinomial
            logits = lm_heads[step - 1](hidden_out[:, -1, :])  # [bsz, vocab]
            if self._top_k > 0:
                topk_vals, _ = logits.topk(self._top_k, dim=-1)
                logits = logits.masked_fill(logits < topk_vals[:, -1:], float("-inf"))
            if self._top_p < 1.0:
                sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
                cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                # Remove tokens with cumulative probability above top_p
                remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= self._top_p
                sorted_logits[remove_mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            code = torch.multinomial(probs, num_samples=1)  # [bsz, 1]

            all_codes[:, step] = code

            # Embed predicted code -> next buffer position
            new_embed = codec_embeds[step - 1](code)  # [batch, 1, hidden_size]
            proj_buf[:bsz, step + 1 : step + 2, :] = new_embed

        return all_codes, proj_buf[:bsz]

    # ------------------------------------------------------------------
    #  Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights directly (no fused projection remapping needed).

        Since we use separate nn.Linear for q/k/v/o and gate/up/down,
        weight names match the HF checkpoint directly.
        """
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            # Skip rotary embeddings
            if "rotary_emb.inv_freq" in name:
                continue

            param = params_dict.get(name)
            if param is None:
                continue

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params
