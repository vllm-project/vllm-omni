from collections.abc import Iterable
from itertools import islice
from typing import Any, Callable, Optional, Union

import torch
from torch import nn
from transformers import Qwen2Config

from vllm.attention import AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.models.qwen2 import Qwen2Attention, Qwen2MLP
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import is_interleaved

from vllm.model_executor.models.utils import (
    PPMissingLayer,
    make_empty_intermediate_tensors_factory, make_layers
)

from vllm_omni.model_executor.models.output_templates import OmniOutput
def moe_forward(
    hidden_states: torch.Tensor,
    und_expert: Callable[[torch.Tensor], torch.Tensor],
    gen_expert: Callable[[torch.Tensor], torch.Tensor],
    gen_token_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    多专家前向（按布尔 mask 分流）.

    Parameters
    ----------
    hidden_states : Tensor           # (B, L, D)
    und_expert    : nn.Module / fn   # False 区域用的专家
    gen_expert    : nn.Module / fn   # True  区域用的专家
    gen_token_mask: BoolTensor|(B,L) # True  → gen_expert；若 None 则全部走 und_expert

    Returns
    -------
    merged : Tensor                  # (B, L, D) 与输入顺序一致
    """
    # 若没有提供 mask，直接全量走 und_expert
    if gen_token_mask is None or not gen_token_mask.any():
        return und_expert(hidden_states)
    if gen_token_mask.all():
        return gen_expert(hidden_states)

    B, L, D = hidden_states.shape
    flat_hid = hidden_states.reshape(-1, D)  # (B*L, D)

    flat_mask = gen_token_mask.reshape(-1)  # (B*L,)
    gen_pos = torch.where(flat_mask)[0]
    und_pos = torch.where(~flat_mask)[0]
    permute_order = torch.cat([gen_pos, und_pos], dim=0)
    inverse_order = torch.argsort(permute_order)
    gen_token_num = gen_token_mask.sum()
    gen_hid, und_hid = flat_hid[permute_order].split([gen_token_num, B * L - gen_token_num], dim=0)
    # gen_hid = flat_hid[flat_mask]
    # und_hid = flat_hid[~flat_mask]

    # -------- 1) 两路前向 --------
    # 1.1 gen-token（True）
    gen_out = gen_expert(gen_hid)  # (N_gen, D)

    # 1.2 普通 token（False）
    und_out = und_expert(und_hid)  # (N_und, D)
    out_dim = und_out.shape[-1]

    # -------- 2) 合并结果 --------
    merged = torch.cat([gen_out, und_out], dim=0)
    merged = merged[inverse_order]
    # merged = torch.empty((B * L, out_dim), dtype=und_out.dtype, device=und_out.device)  # (B*L, D)
    # merged[flat_mask] = gen_out
    # merged[~flat_mask] = und_out

    # -------- 3) 恢复形状 --------
    return merged.view(B, L, out_dim).contiguous()


class Mammoth2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        dual_chunk_attention_config = getattr(config,
                                              "dual_chunk_attention_config",
                                              None)

        # By default, Qwen2 uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. Alibaba-NLP/gte-Qwen2-7B-instruct)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = Qwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        self.gen_mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        gen_token_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        # hidden_states = self.mlp(hidden_states)
        hidden_states = moe_forward(hidden_states, self.mlp, self.gen_mlp, gen_token_mask)
        return hidden_states, residual


class MammothModa2ARForConditionalGeneration(nn.Module):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 decoder_layer_type: type[nn.Module] = Mammoth2DecoderLayer):
        super().__init__()

        config = vllm_config.model_config.hf_config.get_text_config()
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        # TODO (@robertgshaw2): see if this can be moved out
        if is_interleaved(vllm_config.model_config.hf_text_config):
            assert config.max_window_layers == config.num_hidden_layers, (
                "Sliding window for some but all layers is not supported. "
                "This model uses sliding window but `max_window_layers` = {} "
                "is less than `num_hidden_layers` = {}. Please open an issue "
                "to discuss this feature.".format(
                    config.max_window_layers,
                    config.num_hidden_layers,
                ))

        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size
        # 生成 token 的起始下标（用于 gen_token_mask）
        self.gen_vocab_start_index = getattr(
            vllm_config.model_config.hf_config, "gen_vocab_start_index", None
        ) or getattr(config, "gen_vocab_start_index", None)

        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Use the provided decoder layer type or default to Qwen2DecoderLayer
        decoder_layer_type = decoder_layer_type or Mammoth2DecoderLayer
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: decoder_layer_type(config=config,
                                              cache_config=cache_config,
                                              quant_config=quant_config,
                                              prefix=prefix),
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            # gen token mask: True 表示生成图像 token，走 gen_mlp
            if self.gen_vocab_start_index is None:
                gen_token_mask = None
            else:
                gen_token_mask = input_ids >= self.gen_vocab_start_index
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
            gen_token_mask = None

        for idx, layer in enumerate(
                islice(self.layers, self.start_layer, self.end_layer)):
            hidden_states, residual = layer(positions, hidden_states, residual, gen_token_mask)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)

        return OmniOutput(text_hidden_states=hidden_states)
