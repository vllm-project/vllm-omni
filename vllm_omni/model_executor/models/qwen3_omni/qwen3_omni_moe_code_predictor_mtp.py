"""Qwen3-Omni Code Predictor -- optimized re-prefill, no KV cache.

* SDPA attention (F.scaled_dot_product_attention) -- no HF backend fallback
* Persistent pre-allocated buffers (_proj_buf, _pos_ids) -- zero per-call alloc
* Inline top-k sampling -- no LogitsProcessorList / custom-op overhead
* torch.compile on inner transformer by default
* No @support_torch_compile / static_forward_context / namedtuple
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


# ===================================================================
#  Standalone Attention (SDPA, no KV cache, no HF backend fallback)
# ===================================================================


class Qwen3OmniCodePredictorAttention(nn.Module):
    """Multi-head self-attention for code predictor.

    Uses ``F.scaled_dot_product_attention`` directly.  No KV cache -- the code
    predictor always re-prefills the full (short) sequence each AR step.

    Input : [B, seq_len, hidden_size]
    Output: [B, seq_len, hidden_size]
    """

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
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
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self._use_gqa = self.num_kv_heads != self.num_heads

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            disable_tp=True,
        )
        self.o_proj = RowParallelLinear(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            disable_tp=True,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=cp_cfg.max_position_embeddings,
            rope_parameters=None,
            dual_chunk_attention_config=None,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=cp_cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=cp_cfg.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        # Flatten to 2-D so vLLM rotary_emb gets [num_tokens, size]
        qkv, _ = self.qkv_proj(hidden_states.reshape(bsz * seq_len, -1))
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # QK-norm -> RoPE (both 2-D)
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(q.shape)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(k.shape)
        q, k = self.rotary_emb(position_ids, q, k)

        # [B, heads, seq, head_dim] for SDPA
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scaling,
            is_causal=True,
            enable_gqa=self._use_gqa,
        )

        attn_out = attn_out.transpose(1, 2).reshape(bsz * seq_len, -1)
        output, _ = self.o_proj(attn_out)
        return output.view(bsz, seq_len, -1)


# ===================================================================
#  MLP
# ===================================================================


class Qwen3OmniCodePredictorMLP(nn.Module):
    """SiLU-gated MLP for code predictor."""

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        hidden_size = config.code_predictor_config.hidden_size
        intermediate_size = config.code_predictor_config.intermediate_size

        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size, intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            disable_tp=True,
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            disable_tp=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        down, _ = self.down_proj(F.silu(gate) * up)
        return down


# ===================================================================
#  Decoder Layer
# ===================================================================


class Qwen3OmniCodePredictorDecoderLayer(nn.Module):
    """Transformer decoder layer (SDPA, no KV cache)."""

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3OmniCodePredictorAttention(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Qwen3OmniCodePredictorMLP(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        cp_cfg = config.code_predictor_config
        self.input_layernorm = RMSNorm(cp_cfg.hidden_size, eps=cp_cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cp_cfg.hidden_size, eps=cp_cfg.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids)
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
                    quant_config=vllm_config.quant_config,
                    prefix=f"{prefix}.layers.{idx}",
                )
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids)
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
      1. Pre-allocated embedding buffer -- no torch.cat per step.
      2. Pre-allocated position_ids -- no torch.arange per step.
      3. Inline top-k sampling -- no LogitsProcessorList / custom op.
      4. Cached module references -- bypass ModuleList indexing.
      5. torch.compile on inner transformer.
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

        # Sampling hyperparams (inlined)
        self._top_k = 50

        # Persistent buffers (lazily initialised on first forward)
        self._proj_buf: torch.Tensor | None = None
        self._pos_ids: torch.Tensor | None = None

        # Cached plain-list refs (set once)
        self._lm_heads: list | None = None
        self._codec_embeds: list | None = None

        # Model forward (optionally compiled)
        self._model_fwd: object | None = None

    # ------------------------------------------------------------------
    #  Lazy-init helpers
    # ------------------------------------------------------------------

    def _ensure_buffers(self, bsz: int, device: torch.device, dtype: torch.dtype) -> None:
        max_seq = self.num_code_groups + 1
        if (
            self._proj_buf is not None
            and self._proj_buf.shape[0] >= bsz
            and self._proj_buf.device == device
            and self._proj_buf.dtype == dtype
        ):
            return
        self._proj_buf = torch.zeros(bsz, max_seq, self._hidden_size, dtype=dtype, device=device)
        self._pos_ids = torch.arange(max_seq, dtype=torch.long, device=device)

    def _ensure_cached_refs(self) -> None:
        if self._lm_heads is not None:
            return
        self._lm_heads = list(self.lm_head)
        self._codec_embeds = list(self.model.codec_embedding)

    def _ensure_model_fwd(self) -> None:
        if self._model_fwd is not None:
            return
        if not current_omni_platform.supports_torch_inductor():
            logger.warning_once("code_predictor: torch.compile disabled")
            self._model_fwd = self.model.forward
            return
        self._model_fwd = torch.compile(
            self.model.forward,
            mode="default",
            dynamic=True,
        )
        logger.info("code_predictor: torch.compile enabled")

    # ------------------------------------------------------------------
    #  Forward -- re-prefill + persistent buffers + inline sampling
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
                pos 2.. = codec_embedding[i](predicted_code_i)
        """
        bsz = int(layer0_code.shape[0])
        device = layer0_code.device
        dtype = last_talker_hidden.dtype
        num_groups = self.num_code_groups
        top_k = self._top_k

        # Lazy init
        self._ensure_buffers(bsz, device, dtype)
        self._ensure_model_fwd()
        self._ensure_cached_refs()

        proj_buf = self._proj_buf
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
            step_pos_ids = pos_ids[:seq_len] if bsz == 1 else pos_ids[:seq_len].repeat(bsz)

            hidden_out = model_fwd(projected, step_pos_ids)

            # Inline top-k sampling
            logits = lm_heads[step - 1](hidden_out[:, -1, :])
            if top_k > 0:
                topk_vals, _ = logits.topk(top_k, dim=-1)
                logits = logits.masked_fill(logits < topk_vals[:, -1:], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            code = torch.multinomial(probs, num_samples=1)  # [bsz, 1]

            all_codes[:, step] = code

            # Embed predicted code -> next buffer position
            new_embed = codec_embeds[step - 1](code)
            proj_buf[:bsz, step + 1 : step + 2, :] = new_embed

        return all_codes, proj_buf[:bsz]

    # ------------------------------------------------------------------
    #  Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with mapping for fused QKV and gate_up projections.

        Maps original HF weights (q_proj, k_proj, v_proj, gate_proj, up_proj)
        to fused vLLM weights (qkv_proj, gate_up_proj).
        """
        # Mapping for fused projections
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            # Skip rotary embeddings
            if "rotary_emb.inv_freq" in name:
                continue

            # Handle stacked/fused parameters
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip if parameter doesn't exist (e.g., bias)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Non-stacked parameters - use default loading
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", None)
                if weight_loader is not None:
                    weight_loader(param, loaded_weight)
                else:
                    param.data.copy_(loaded_weight)

            loaded_params.add(name)

        return loaded_params
