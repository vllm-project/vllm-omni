# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM-native PersonaPlex Helium temporal transformer.

The module maps the Moshi temporal LM backbone onto vLLM's Llama-style decoder
components. It intentionally does not include Moshi's text/audio input
embeddings or depformer; PersonaPlex feeds temporal `inputs_embeds` directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import islice

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType

from vllm_omni.model_executor.models.personaplex.configuration_helium import (
    HeliumConfig,
)

__all__ = ["HeliumForCausalLM", "HeliumModel"]


class HeliumRMSNorm(RMSNorm):
    """Moshi-compatible RMSNorm.

    Moshi stores the parameter as `alpha` with shape `[1, 1, hidden]` and
    computes variance and the alpha multiply in fp32 before casting back to the
    input dtype. vLLM's parameter is named `weight`; the loader squeezes alpha.
    """

    def _forward_f32(self, x: torch.Tensor) -> torch.Tensor:
        output_dtype = x.dtype
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
        x_f32 = x_f32 * torch.rsqrt(variance + self.variance_epsilon)
        if self.has_weight:
            x_f32 = x_f32 * self.weight.float()
        return x_f32.to(output_dtype)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self._forward_f32(x)
        residual = residual + x
        return self._forward_f32(residual), residual


class HeliumMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. Only silu is supported.")
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


def _get_helium_rope(config: HeliumConfig):
    rope_kwargs = {
        "head_size": config.head_dim,
        "max_position": config.max_position_embeddings,
        "is_neox_style": False,
        "rope_parameters": {"base": config.rope_theta},
    }
    try:
        return get_rope(**rope_kwargs)
    except TypeError:
        # vLLM 0.23 uses `rope_parameters`; this fallback is for older local
        # source trees whose get_rope still takes rotary_dim/base/rope_scaling.
        return get_rope(
            head_size=config.head_dim,
            rotary_dim=config.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            is_neox_style=False,
        )


class HeliumAttention(nn.Module):
    def __init__(
        self,
        config: HeliumConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = _get_helium_rope(config)
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=config.sliding_window,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class HeliumDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        config: HeliumConfig | None = None,
    ) -> None:
        super().__init__()
        config = config or vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.hidden_size = config.hidden_size

        self.self_attn = HeliumAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            attn_type=AttentionType.DECODER,
        )
        self.mlp = HeliumMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = HeliumRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = HeliumRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states,
            residual,
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


def _flatten_inputs(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int] | None]:
    if hidden_states.dim() != 3:
        return hidden_states, positions.reshape(-1), None

    batch, seq_len, hidden = hidden_states.shape
    flat_hidden = hidden_states.reshape(batch * seq_len, hidden)
    if positions.dim() == 1:
        if positions.numel() == seq_len:
            flat_positions = positions.unsqueeze(0).expand(batch, seq_len).reshape(-1)
        elif positions.numel() == batch * seq_len:
            flat_positions = positions.reshape(-1)
        else:
            raise ValueError(
                "1D positions must have length S or B*S when inputs_embeds is "
                f"3D, got {positions.numel()} for B={batch}, S={seq_len}."
            )
    else:
        flat_positions = positions.reshape(-1)
    return flat_hidden, flat_positions, (batch, seq_len)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class HeliumModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = HeliumDecoderLayer,
        config: HeliumConfig | None = None,
    ) -> None:
        super().__init__()
        # ``config`` lets a composite model (e.g. the PersonaPlex talker) build the
        # Helium backbone from its temporal sub-config while the engine's top-level
        # hf_config is the composite PersonaPlexConfig. Defaults to the engine's.
        config = config or vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.vocab_size = config.vocab_size

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: layer_type(vllm_config=vllm_config, prefix=prefix, config=config),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = HeliumRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"],
            config.hidden_size,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise ValueError(
            "Helium temporal forward requires inputs_embeds; Moshi token "
            "embeddings are intentionally not part of this vLLM module."
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        unflatten_shape: tuple[int, int] | None = None
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                raise ValueError("HeliumModel.forward requires inputs_embeds.")
            hidden_states, positions, unflatten_shape = _flatten_inputs(
                inputs_embeds,
                positions,
            )
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
            positions = positions.reshape(-1)

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

        hidden_states, _ = self.norm(hidden_states, residual)
        if unflatten_shape is not None:
            batch, seq_len = unflatten_shape
            hidden_states = hidden_states.reshape(batch, seq_len, -1)
        return hidden_states


class HeliumForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    embedding_modules = {
        "lm_head": "output_embeddings",
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = HeliumDecoderLayer,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.model = HeliumModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            layer_type=layer_type,
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if hidden_states.dim() == 3:
            batch, seq_len, hidden = hidden_states.shape
            flat_hidden = hidden_states.reshape(batch * seq_len, hidden)
            logits = self.logits_processor(self.lm_head, flat_hidden)
            if logits is None:
                return None
            return logits.reshape(batch, seq_len, -1)
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]] | Mapping[str, torch.Tensor],
    ) -> set[str]:
        if isinstance(weights, Mapping):
            weights = weights.items()

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            mapped = self._load_moshi_weight(name, loaded_weight, params_dict)
            loaded_params.update(mapped)

        return loaded_params

    def _load_moshi_weight(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict[str, nn.Parameter],
    ) -> set[str]:
        if name == "out_norm.alpha":
            return self._load_direct(
                "model.norm.weight",
                loaded_weight.squeeze(),
                params_dict,
            )
        if name == "text_linear.weight":
            return self._load_direct("lm_head.weight", loaded_weight, params_dict)

        prefix = "transformer.layers."
        if not name.startswith(prefix):
            return set()

        rest = name.removeprefix(prefix)
        layer_index, _, suffix = rest.partition(".")
        if not layer_index.isdigit() or not suffix:
            return set()

        base = f"model.layers.{layer_index}"
        if suffix == "self_attn.in_proj_weight":
            q_weight, k_weight, v_weight = loaded_weight.chunk(3, dim=0)
            param_name = f"{base}.self_attn.qkv_proj.weight"
            self._load_shard(param_name, q_weight, "q", params_dict)
            self._load_shard(param_name, k_weight, "k", params_dict)
            self._load_shard(param_name, v_weight, "v", params_dict)
            return {param_name}
        if suffix == "self_attn.out_proj.weight":
            return self._load_direct(
                f"{base}.self_attn.o_proj.weight",
                loaded_weight,
                params_dict,
            )
        if suffix == "gating.linear_in.weight":
            gate_weight, up_weight = loaded_weight.chunk(2, dim=0)
            param_name = f"{base}.mlp.gate_up_proj.weight"
            self._load_shard(param_name, gate_weight, 0, params_dict)
            self._load_shard(param_name, up_weight, 1, params_dict)
            return {param_name}
        if suffix == "gating.linear_out.weight":
            return self._load_direct(
                f"{base}.mlp.down_proj.weight",
                loaded_weight,
                params_dict,
            )
        if suffix == "norm1.alpha":
            return self._load_direct(
                f"{base}.input_layernorm.weight",
                loaded_weight.squeeze(),
                params_dict,
            )
        if suffix == "norm2.alpha":
            return self._load_direct(
                f"{base}.post_attention_layernorm.weight",
                loaded_weight.squeeze(),
                params_dict,
            )
        return set()

    def _load_direct(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict[str, nn.Parameter],
    ) -> set[str]:
        if name not in params_dict or is_pp_missing_parameter(name, self):
            return set()
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)
        return {name}

    def _load_shard(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        shard_id: str | int,
        params_dict: dict[str, nn.Parameter],
    ) -> None:
        if name not in params_dict or is_pp_missing_parameter(name, self):
            return
        param = params_dict[name]
        weight_loader = param.weight_loader
        weight_loader(param, loaded_weight, shard_id)
