from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeMLP,
    Qwen3MoeModel,  # as _BaseQwen3MoeModel,
)
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeForCausalLM as _BaseQwen3MoeForCausalLM,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    maybe_prefix,
)

logger = init_logger(__name__)


# Individual expert MoE block using Qwen3MoeMLP instead of FusedMoE
class Qwen3OmniMoeSparseMoeBlock(nn.Module):
    """Sparse MoE block using individual Qwen3MoeMLP experts instead of FusedMoE."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size

        # Create individual expert MLPs
        self.experts = nn.ModuleList(
            [
                Qwen3MoeMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.moe_intermediate_size,
                    hidden_act=config.hidden_act,
                    quant_config=quant_config,
                    reduce_results=False,
                    prefix=f"{prefix}.experts.{i}",
                )
                for i in range(self.num_experts)
            ]
        )

        # Router for expert selection
        from vllm.model_executor.layers.linear import ReplicatedLinear

        self.gate = ReplicatedLinear(
            config.hidden_size, config.num_experts, bias=False, quant_config=quant_config, prefix=f"{prefix}.gate"
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass using individual experts."""
        # Handle 3D inputs (batch, seq_len, hidden_size) by reshaping to 2D
        orig_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            batch_size, seq_len, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)
        elif hidden_states.dim() == 2:
            num_tokens, hidden_dim = hidden_states.shape
        elif hidden_states.dim() == 1:
            hidden_states = hidden_states.unsqueeze(0)
            num_tokens, hidden_dim = hidden_states.shape
        else:
            raise ValueError(
                f"Qwen3OmniMoeSparseMoeBlock only supports 1D, 2D, or 3D inputs, got {hidden_states.dim()}D"
            )

        is_input_1d = len(orig_shape) == 1
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Get router logits and select experts (matching transformers)
        router_logits, _ = self.gate(hidden_states)
        selected_experts, routing_weights = self._route_tokens(router_logits)

        # Forward through individual experts
        final_hidden_states = self._forward_experts(hidden_states, selected_experts, routing_weights)

        # Reshape back to original shape
        if is_input_1d:
            return final_hidden_states.squeeze(0)
        elif len(orig_shape) == 3:
            # Reshape back to 3D (batch, seq_len, hidden_dim)
            return final_hidden_states.view(orig_shape)
        else:
            return final_hidden_states

    def _route_tokens(self, router_logits: torch.Tensor):
        """Route tokens to experts using top-k selection (matching transformers)."""
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(router_logits.dtype)
        return selected_experts, routing_weights

    def _forward_experts(
        self, hidden_states: torch.Tensor, selected_experts: torch.Tensor, routing_weights: torch.Tensor
    ):
        """Forward through individual experts (matching transformers implementation)."""
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states[None, top_x].reshape(-1, hidden_states.shape[-1])
            current_hidden_states = self.experts[expert_idx](current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        return final_hidden_states


class Qwen3MoeForCausalLM(_BaseQwen3MoeForCausalLM):
    """Thin wrapper to swap in the patched `Qwen3MoeModel`."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Don't call super().__init__() to avoid duplicate layer registration.
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen3MoeModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size, quant_config=quant_config, prefix=maybe_prefix(prefix, "lm_head")
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        # Set MoE hyperparameters for individual experts
        self.expert_weights = []

        self.moe_layers: list[FusedMoE] = []
        example_layer = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, Qwen3MoeDecoderLayer)
            if isinstance(layer.mlp, FusedMoE):
                example_layer = layer.mlp
                self.moe_layers.append(layer.mlp)

        if example_layer is None:
            raise RuntimeError("No Qwen3OmniMoe layer found in the model.layers.")

        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_layer.n_logical_experts
        self.num_physical_experts = example_layer.n_physical_experts
        self.num_local_physical_experts = example_layer.n_local_physical_experts
        self.num_routed_experts = example_layer.n_routed_experts
        self.num_redundant_experts = example_layer.n_redundant_experts
