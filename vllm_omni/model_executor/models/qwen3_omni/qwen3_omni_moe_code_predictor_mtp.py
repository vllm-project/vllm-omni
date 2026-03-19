"""Qwen3-Omni Code Predictor with MTP (Multi-Token Prediction) support.

This module implements the code predictor component for Qwen3-Omni talker models.

The code predictor generates residual RVQ (Residual Vector Quantization) codes
autoregressively, predicting layers 1 to N based on layer-0 codes from the talker.
"""

from collections import namedtuple
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Cache, PretrainedConfig
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
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
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

# ============================================================================
# Code Predictor Attention Layer
# ============================================================================


class Qwen3OmniCodePredictorAttention(nn.Module):
    """Multi-head self-attention for code predictor with vLLM optimization."""

    def __init__(
        self,
        config,
        layer_idx: int,
        vllm_config: VllmConfig = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.num_heads = config.code_predictor_config.num_attention_heads
        self.num_key_value_heads = config.code_predictor_config.num_key_value_heads
        self.head_dim = getattr(
            config.code_predictor_config,
            "head_dim",
            config.code_predictor_config.hidden_size // config.code_predictor_config.num_attention_heads,
        )
        self.hidden_size = config.code_predictor_config.hidden_size

        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_key_value_heads,
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
            max_position=config.code_predictor_config.max_position_embeddings,
            rope_parameters=None,
            dual_chunk_attention_config=None,
        )

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim

        # Query/Key normalization
        self.q_norm = RMSNorm(self.head_dim, eps=config.code_predictor_config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.code_predictor_config.rms_norm_eps)
        self.is_causal = True
        self.config = config

        self.attention_backends = ["flash_attention_2", "xformers", "eager", "sdpa"]
        cudagraph_mode = get_current_vllm_config().compilation_config.cudagraph_mode
        if "flash_attention_2" in ALL_ATTENTION_FUNCTIONS and cudagraph_mode.has_full_cudagraphs():
            logger.warning(
                f"CUDAGraphMode.{cudagraph_mode.name} is currently not supported "
                f"with flash attention for Qwen3-Omni talker MTP."
                f"removing flash attention from attention_backends"
            )
            self.attention_backends.remove("flash_attention_2")

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool = False,
        position_ids: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Reshape for attention
        q = q.reshape(bsz, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.reshape(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        # Apply normalization
        q = self.q_norm(q).contiguous()
        k = self.k_norm(k).contiguous()
        q = q.reshape(-1, self.q_size)
        k = k.reshape(-1, self.kv_size)

        # Apply RoPE
        q, k = self.rotary_emb(position_ids, q, k)

        # Reshape for attention
        q = q.reshape(bsz, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        v_heads = v.transpose(1, 2).contiguous()
        q_heads = q.transpose(1, 2).contiguous()
        k_heads = k.transpose(1, 2).contiguous()

        if past_key_values is not None:
            sin, cos = self.rotary_emb.get_cos_sin(seq_len)
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k_heads, v_heads = past_key_values.update(k_heads, v_heads, self.layer_idx, cache_kwargs)

        # Try attention backends in order of preference, with runtime error handling
        # This handles cases where the backend is registered but not actually available
        attn_output = None
        last_error = None

        for backend_name in self.attention_backends:
            if backend_name not in ALL_ATTENTION_FUNCTIONS:
                continue

            try:
                attention_interface = ALL_ATTENTION_FUNCTIONS[backend_name]
                attn_output, _ = attention_interface(
                    self,
                    q_heads,
                    k_heads,
                    v_heads,
                    None,
                    dropout=0.0 if not self.training else getattr(self, "attention_dropout", 0.0),
                    scaling=self.head_dim**-0.5,
                    sliding_window=None,
                    use_cache=use_cache,
                    position_ids=position_ids[:seq_len].unsqueeze(0),
                    output_hidden_states=True,
                    output_attentions=False,
                )
                break
            except (ValueError, ImportError, RuntimeError, AttributeError) as e:
                # Store error and try next backend
                last_error = e
                continue

        if attn_output is None:
            raise RuntimeError(
                f"All attention backends failed. Last error: {last_error}. "
                "Please install flash-attn, or ensure PyTorch's scaled_dot_product_attention is available."
            )
        attn_output = attn_output.reshape(*(hidden_states.shape[:-1]), -1).contiguous()

        attn_output, _ = self.o_proj(attn_output)
        return attn_output


# ============================================================================
# Code Predictor MLP Layer
# ============================================================================


class Qwen3OmniCodePredictorMLP(nn.Module):
    """Feed-forward network for code predictor with fused gate/up projection."""

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


# ============================================================================
# MTP Layer (Multi-Token Prediction Layer)
# ============================================================================


class Qwen3OmniCodePredictorMTPLayer(nn.Module):
    """MTP layer for speculative decoding - predicts next residual code layer."""

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        model_config: ModelConfig,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        self.self_attn = Qwen3OmniCodePredictorAttention(
            config,
            layer_idx,
            vllm_config=type(
                "VllmConfig",
                (),
                {"cache_config": cache_config, "quant_config": quant_config, "model_config": model_config},
            )(),
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Qwen3OmniCodePredictorMLP(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(
            config.code_predictor_config.hidden_size, eps=config.code_predictor_config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            config.code_predictor_config.hidden_size, eps=config.code_predictor_config.rms_norm_eps
        )

    def mtp_block(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool = False,
        position_ids: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, past_key_values, cache_position, use_cache, position_ids)
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3OmniCodePredictorBaseModel(nn.Module):
    """
    Base model for code predictor - matches HF Qwen3OmniMoeTalkerCodePredictorModel structure.

    This is a simple transformer that processes inputs_embeds and outputs hidden states.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config.code_predictor_config

        self.config = config
        self.vocab_size = config.vocab_size
        self.num_code_groups = config.num_code_groups

        # Codec embeddings (for layers 1-num_code_groups-1)
        self.codec_embedding = nn.ModuleList(
            [
                VocabParallelEmbedding(
                    config.vocab_size,
                    config.hidden_size,
                )
                for _ in range(config.num_code_groups - 1)
            ]
        )

        # Decoder layers
        self.layers = nn.ModuleList(
            [
                Qwen3OmniCodePredictorMTPLayer(
                    vllm_config.model_config.hf_config,
                    f"{prefix}.layers.{idx}",
                    model_config=vllm_config.model_config,
                    layer_idx=idx,
                    cache_config=vllm_config.cache_config,
                    quant_config=vllm_config.quant_config,
                )
                for idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Forward pass matching HF structure.

        Args:
            inputs_embeds: [batch, seq_len, hidden_size]
            position_ids: Optional position IDs tensor
            past_key_values: Optional cached key-value pairs
            use_cache: Whether to use cache
            cache_position: Optional cache position tensor
            **kwargs: Additional keyword arguments

        Returns:
            Named tuple with .last_hidden_state and .past_key_values attributes
        """
        batch_size, seq_len, _ = inputs_embeds.shape
        # Forward through decoder layers
        hidden_states = inputs_embeds

        for layer in self.layers:
            hidden_states = layer.mtp_block(hidden_states, past_key_values, cache_position, use_cache, position_ids)

        # Final norm
        hidden_states = self.norm(hidden_states)

        # Return in HF-compatible format
        Output = namedtuple("Output", ["last_hidden_state", "past_key_values"])
        return Output(last_hidden_state=hidden_states, past_key_values=None)  # [batch, num_code_groups-1, hidden_size]

    def get_input_embeddings(self):
        """Return codec embeddings for HF compatibility."""
        return self.codec_embedding


def code_predictor_sample(
    logits: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    logits = self.logits_processors(None, logits[:, -1])
    probs = F.softmax(logits, dim=-1)
    code = torch.multinomial(probs.squeeze(1), num_samples=1)  # [batch, 1]
    return code


def code_predictor_sample_fake(
    logits: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty((logits.shape[0], 1), dtype=torch.int64, device=logits.device)


direct_register_custom_op(
    op_name="qwen3_omni_code_predictor_sample",
    op_func=code_predictor_sample,
    fake_impl=code_predictor_sample_fake,
)


@support_torch_compile
class Qwen3OmniMoeTalkerCodePredictor(nn.Module):
    """
    Code predictor wrapper matching HF structure.

    Structure:
    - self.model: Qwen3OmniCodePredictorBaseModel (transformer)
    - self.lm_head: ModuleList of output heads
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        talker_code_predictor_config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.prefix = prefix

        self.config = talker_code_predictor_config
        self.vocab_size = self.config.code_predictor_config.vocab_size
        self.num_code_groups = self.config.code_predictor_config.num_code_groups

        # Base transformer model (matches HF structure)
        self.model = Qwen3OmniCodePredictorBaseModel(vllm_config=vllm_config, prefix=prefix)

        # Output heads for each residual layer (1-num_layers-1)
        self.lm_head = nn.ModuleList(
            [
                nn.Linear(
                    self.config.code_predictor_config.hidden_size,
                    self.config.code_predictor_config.vocab_size,
                    bias=False,
                )
                for _ in range(self.num_code_groups - 1)
            ]
        )
        self.logits_processors = LogitsProcessorList(
            [
                TopKLogitsWarper(top_k=50),
                TopPLogitsWarper(top_p=0.8),
            ]
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.layer_name = prefix

    def forward(
        self,
        layer0_code: torch.Tensor,
        layer0_embed: torch.Tensor,
        last_talker_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for code predictor.

        Args:
            layer0_code:
                Code index for code-group (layer) 0.
                Shape: [batch_size, 1], dtype typically int64.

            last_talker_hidden:

                Shape: [batch_size, hidden_size].

        Returns:
            pos_all_layers:
                Predicted codes for all code groups, including `layer0_code`.
                Shape: [batch_size, num_code_groups, 1].

            current_input:
                The final input embedding sequence after appending embeddings of all
                predicted codes (one token per predicted layer).
                Shape: [batch_size, num_code_groups + 2, hidden_size].
        """
        pos_codes = [layer0_code]  # Start with layer 0: [batch, 1]
        try:
            current_input = torch.cat([last_talker_hidden, layer0_embed], dim=1)  # [batch, 2, hidden_size]
        except Exception as e:
            print(f"Error in current_input: {e}")
            print(f"last_talker_hidden shape: {last_talker_hidden.shape}")
            print(f"prev_embed shape: {layer0_embed.shape}")
            raise e
        batch_size = current_input.shape[0]

        # Predict all residual layers (layers 1 to num_code_groups-1) autoregressively
        for layer_idx in range(self.num_code_groups - 1):
            seq_len = layer_idx + 2
            # Compute position_ids dynamically to avoid torch.compile specializing batch_size
            position_ids = torch.arange(seq_len, device=current_input.device, dtype=torch.int64).repeat(batch_size)
            # Forward through code_predictor model
            outputs = self.model(
                inputs_embeds=current_input,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=None,
            )
            hidden_state = outputs.last_hidden_state  # [batch, 2, hidden_size]

            # Use the corresponding lm_head for this layer
            logits = self.lm_head[layer_idx](hidden_state[:, -1:, :])
            code = torch.ops.vllm.qwen3_omni_code_predictor_sample(logits, self.layer_name)
            pos_codes.append(code)
            # Update prev_embed for next layer (if not last layer)
            # layer_idx=0 predicts layer 1, embed with codec_embedding[1]
            new_embed = self.model.codec_embedding[layer_idx](code)  # [batch, 1, hidden_size]
            current_input = torch.cat([current_input, new_embed], dim=1)  # [batch, 3~n, hidden_size]
        pos_all_layers = torch.stack(pos_codes, dim=1)  # [batch, num_code_groups, 1]
        return pos_all_layers, current_input

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
