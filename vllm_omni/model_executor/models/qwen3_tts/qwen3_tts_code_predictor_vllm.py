from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.config.vllm import set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import is_pp_missing_parameter

from vllm_omni.platforms import current_omni_platform

from .configuration_qwen3_tts import Qwen3TTSTalkerCodePredictorConfig, Qwen3TTSTalkerConfig

logger = init_logger(__name__)


# ===================================================================
#  Standalone Code Predictor Layers (no vLLM paged attention)
# ===================================================================
#
# These replace vLLM's Qwen3DecoderLayer for the code predictor.
# Input is batch-major [B, seq_len, H], attention uses F.scaled_dot_product_attention.
# Weight names match the checkpoint (self_attn.qkv_proj, mlp.gate_up_proj, etc.)
# so load_weights works unchanged.


class _CodePredictorAttention(nn.Module):
    """Standalone multi-head attention for code predictor.

    Uses F.scaled_dot_product_attention (SDPA) instead of vLLM's paged Attention.
    Supports fused QKV, RoPE, q/k normalization, and native GQA via enable_gqa.
    Input: [B, seq_len, hidden_size], output: [B, seq_len, hidden_size].
    """

    def __init__(
        self,
        config: Qwen3TTSTalkerCodePredictorConfig,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self._use_gqa = self.num_kv_heads != self.num_heads

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_kv_heads,
            bias=getattr(config, "attention_bias", False),
            prefix=f"{prefix}.qkv_proj",
            disable_tp=True,
        )
        self.o_proj = RowParallelLinear(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=False,
            prefix=f"{prefix}.o_proj",
            disable_tp=True,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=getattr(config, "rope_parameters", None),
            dual_chunk_attention_config=None,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        qkv, _ = self.qkv_proj(hidden_states.reshape(bsz * seq_len, -1))
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(q.shape)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(k.shape)

        q, k = self.rotary_emb(position_ids, q, k)

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


class _CodePredictorMLP(nn.Module):
    """SiLU-gated MLP for code predictor, matching Qwen3MLP structure."""

    def __init__(
        self,
        config: Qwen3TTSTalkerCodePredictorConfig,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.intermediate_size] * 2,
            bias=False,
            prefix=f"{prefix}.gate_up_proj",
            disable_tp=True,
        )
        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=False,
            prefix=f"{prefix}.down_proj",
            disable_tp=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = F.silu(gate) * up
        x, _ = self.down_proj(x)
        return x


class _CodePredictorDecoderLayer(nn.Module):
    """Transformer decoder layer for code predictor (SDPA, no KV cache)."""

    def __init__(
        self,
        config: Qwen3TTSTalkerCodePredictorConfig,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = _CodePredictorAttention(config, prefix=f"{prefix}.self_attn")
        self.mlp = _CodePredictorMLP(config, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
#  Code Predictor Transformer Model
# ===================================================================


class Qwen3TTSTalkerCodePredictorModelVLLM(nn.Module):
    """Transformer model for the code predictor (re-prefill, no KV cache)."""

    def __init__(
        self,
        config: Qwen3TTSTalkerCodePredictorConfig,
        *,
        talker_hidden_size: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(
            [_CodePredictorDecoderLayer(config, prefix=f"{prefix}.layers.{i}") for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Codec embeddings: one per residual group. Stored in talker hidden dim
        # (some checkpoints use talker_hidden_size != code_predictor hidden_size).
        emb_dim = int(talker_hidden_size) if talker_hidden_size is not None else int(config.hidden_size)
        self.codec_embedding = nn.ModuleList(
            [nn.Embedding(config.vocab_size, emb_dim) for _ in range(config.num_code_groups - 1)]
        )

    def get_input_embeddings(self) -> nn.ModuleList:
        return self.codec_embedding

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

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped.endswith(".bias") and mapped not in params_dict:
                    continue
                if is_pp_missing_parameter(mapped, self):
                    continue
                if mapped.endswith("scale"):
                    mapped = maybe_remap_kv_scale_name(mapped, params_dict)
                    if mapped is None:
                        continue
                param = params_dict.get(mapped)
                if param is None:
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped)
                break
            else:
                mapped = maybe_remap_kv_scale_name(name, params_dict)
                if mapped is None:
                    continue
                if name.endswith(".bias") and mapped not in params_dict:
                    continue
                if is_pp_missing_parameter(mapped, self):
                    continue
                param = params_dict.get(mapped)
                if param is None:
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(mapped)
        return loaded_params


# ===================================================================
#  Code Predictor Wrapper (optimized re-prefill + torch.compile)
# ===================================================================


class Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM(nn.Module):
    """vLLM-native code_predictor for the AR talker (residual codebooks).

    Re-prefill approach: each AR step forwards the full growing sequence
    through the 5-layer transformer. No KV cache needed. This trades
    ~O(T^2) extra attention FLOPs (negligible for T=16, 5 layers) for
    zero KV cache management overhead and a simpler execution model.

    Optimizations over baseline:
      1. torch.compile on model forward -- fuses 60+ small kernel launches per step
         into fewer fused kernels (4x speedup on model_fwd, ~75% of total time).
      2. Pre-allocated embedding buffer [B, max_seq, H] -- no torch.cat per step.
      3. Projection caching -- each token projected once and cached, avoids O(T^2)
         redundant projections.
      4. Pre-allocated position_ids [max_seq] -- no torch.arange per step.
      5. Inline sampling -- no custom op / forward_context overhead.
      6. No context managers in forward().
      7. Cached module references -- bypass nn.Module.__call__ and ModuleList indexing.
      8. Pre-allocated output tensor.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen3TTSTalkerCodePredictorConfig,
        talker_config: Qwen3TTSTalkerConfig,
        prefix: str = "code_predictor",
    ) -> None:
        super().__init__()
        self._vllm_config = vllm_config
        self.config = config
        self.talker_config = talker_config

        self.model = Qwen3TTSTalkerCodePredictorModelVLLM(
            config,
            talker_hidden_size=int(talker_config.hidden_size),
            prefix=f"{prefix}.model",
        )

        self.lm_head = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_code_groups - 1)]
        )

        if config.hidden_size != talker_config.hidden_size:
            self.small_to_mtp_projection = nn.Linear(talker_config.hidden_size, config.hidden_size, bias=True)
        else:
            self.small_to_mtp_projection = nn.Identity()

        self._num_groups = int(config.num_code_groups)
        self._talker_hidden = int(talker_config.hidden_size)
        self._cp_hidden = int(config.hidden_size)

        # Pre-allocated buffers (lazily initialized on first forward).
        self._proj_buf: torch.Tensor | None = None

        # torch.compile + warmup state (lazily initialized in _setup_compile).
        self._compiled_model_fwd = None
        self._bucket_sizes: list[int] = []
        self._bucket_pos_ids: dict[int, torch.Tensor] = {}
        self._lm_heads_list: list[nn.Module] | None = None
        self._codec_embeds_list: list[nn.Module] | None = None

    def get_input_embeddings(self) -> nn.ModuleList:
        return self.model.get_input_embeddings()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        with set_current_vllm_config(self._vllm_config):
            loaded: set[str] = set()
            model_weights: list[tuple[str, torch.Tensor]] = []
            other_weights: list[tuple[str, torch.Tensor]] = []
            for name, w in weights:
                if name.startswith("model."):
                    model_weights.append((name[len("model.") :], w))
                else:
                    other_weights.append((name, w))

            loaded_model = self.model.load_weights(model_weights)
            loaded |= {f"model.{n}" for n in loaded_model}

            params = dict(self.named_parameters(remove_duplicate=False))
            for name, w in other_weights:
                if name not in params:
                    continue
                default_weight_loader(params[name], w)
                loaded.add(name)

            return loaded

    # ------------------------------------------------------------------
    #  Pre-allocated buffer management
    # ------------------------------------------------------------------

    def _ensure_buffers(self, device: torch.device, dtype: torch.dtype) -> None:
        max_seq = self._num_groups + 1
        if self._proj_buf is not None and self._proj_buf.device == device and self._proj_buf.dtype == dtype:
            return
        max_bsz = self._vllm_config.scheduler_config.max_num_seqs
        self._proj_buf = torch.zeros(
            max_bsz,
            max_seq,
            self._cp_hidden,
            dtype=dtype,
            device=device,
        )

    def _setup_compile(self) -> None:
        """Lazily set up torch.compiled model forward for kernel fusion.

        Uses ``mode="reduce-overhead"`` with ``dynamic=False`` so Inductor
        captures internal CUDA graphs for fixed shapes, eliminating kernel
        launch overhead entirely.
        """
        if self._compiled_model_fwd is not None:
            return
        self._lm_heads_list = list(self.lm_head)
        self._codec_embeds_list = list(self.model.codec_embedding)
        if not current_omni_platform.supports_torch_inductor():
            logger.warning_once("code_predictor: torch.compile disabled")
            self._compiled_model_fwd = self.model.forward
            return
        self._compiled_model_fwd = torch.compile(
            self.model.forward,
            mode="reduce-overhead",
            dynamic=False,
        )
        logger.info("code_predictor: torch.compile enabled (mode=reduce-overhead, dynamic=False)")
        self._warmup_compile()

    def _padded_bsz(self, bsz: int) -> int:
        for bucket in self._bucket_sizes:
            if bsz <= bucket:
                return bucket
        return bsz

    def _warmup_compile(self) -> None:
        """Warmup power-of-2 batch-size buckets to front-load compilation."""
        max_bsz = self._vllm_config.scheduler_config.max_num_seqs
        bucket_sizes = [1 << i for i in range(max_bsz.bit_length()) if (1 << i) <= max_bsz]
        if max_bsz not in bucket_sizes:
            bucket_sizes.append(max_bsz)
        self._bucket_sizes = sorted(bucket_sizes)

        max_seq = self._num_groups + 1
        device = next(self.model.parameters()).device
        base_pos = torch.arange(max_seq, device=device, dtype=torch.long)

        proj_buf = self._proj_buf
        for bsz in self._bucket_sizes:
            pos_ids = base_pos if bsz == 1 else base_pos.repeat(bsz)
            self._bucket_pos_ids[bsz] = pos_ids
            for _ in range(3):
                self._compiled_model_fwd(proj_buf[:bsz, :max_seq, :], pos_ids)
        logger.info("code_predictor: warmup done for bucket sizes %s", self._bucket_sizes)

    # ------------------------------------------------------------------
    #  Optimized forward: re-prefill + torch.compile + projection cache
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
    ) -> torch.Tensor:
        """Predict residual codebooks 1..Q-1 autoregressively via re-prefill.

        torch.compile fuses the ~60 small kernel launches per step into fewer
        fused kernels, reducing kernel launch overhead by ~75%.

        Projection caching: each token is projected once via small_to_mtp_projection
        and cached in _proj_buf, avoiding redundant re-projection of past tokens.
        """
        bsz = int(layer0_code.shape[0])
        num_groups = self._num_groups
        device = layer0_code.device
        dtype = layer0_embed.dtype

        all_codes = torch.empty(bsz, num_groups, dtype=torch.long, device=device)
        all_codes[:, 0] = layer0_code.reshape(bsz)

        self._ensure_buffers(device, dtype)
        self._setup_compile()

        proj_buf = self._proj_buf
        max_seq = self._num_groups + 1

        projection = self.small_to_mtp_projection
        model_fwd = self._compiled_model_fwd
        lm_heads = self._lm_heads_list
        codec_embeds = self._codec_embeds_list

        use_sampling = do_sample and temperature > 0
        inv_temperature = 1.0 / max(temperature, 1e-6) if use_sampling else 0.0
        if use_sampling and top_p != 1.0:
            raise NotImplementedError(
                "top_p sampling is not implemented for the vLLM-native code predictor; please set top_p=1.0."
            )

        padded_bsz = self._padded_bsz(bsz)
        proj_buf[:padded_bsz].zero_()

        proj_buf[:bsz, 0, :] = projection(last_talker_hidden.reshape(bsz, 1, -1)).reshape(bsz, -1)
        proj_buf[:bsz, 1, :] = projection(layer0_embed.reshape(bsz, 1, -1)).reshape(bsz, -1)
        full_pos_ids = self._bucket_pos_ids.get(padded_bsz)
        if full_pos_ids is None:
            base_pos = torch.arange(max_seq, device=device, dtype=torch.long)
            full_pos_ids = base_pos if padded_bsz == 1 else base_pos.repeat(padded_bsz)

        for step in range(1, num_groups):
            projected = proj_buf[:padded_bsz, :max_seq, :]

            hidden_out = model_fwd(projected, full_pos_ids)
            logits = lm_heads[step - 1](hidden_out[:bsz, step, :])

            if use_sampling:
                scaled = logits * inv_temperature
                if top_k > 0:
                    topk_vals, _ = scaled.topk(top_k, dim=-1)
                    scaled = scaled.masked_fill(scaled < topk_vals[:, -1:], float("-inf"))
                probs = F.softmax(scaled, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1)
            else:
                next_ids = logits.argmax(dim=-1, keepdim=True)

            all_codes[:, step] = next_ids.reshape(bsz)

            if step < num_groups - 1:
                new_embed = codec_embeds[step - 1](next_ids)
                proj_buf[:bsz, step + 1, :] = projection(new_embed.reshape(bsz, 1, -1)).reshape(bsz, -1)

        return all_codes
