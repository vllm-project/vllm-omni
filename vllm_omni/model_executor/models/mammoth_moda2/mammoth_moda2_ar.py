from collections.abc import Iterable, Mapping
from itertools import islice
from typing import Any, Callable, Optional, Union
from copy import deepcopy
import json
from pathlib import Path

import torch
from torch import nn
from transformers import Qwen2Config

from vllm.attention.backends.abstract import AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.qwen2 import Qwen2Attention, Qwen2MLP
from vllm.model_executor.models.qwen2_vl import Qwen2VLMultiModalDataParser
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import is_interleaved
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    init_vllm_registered_model,
)

from vllm.transformers_utils.config import (
    is_interleaved,
    patch_rope_parameters,
    set_default_rope_theta,
)

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessingInfo,
    Qwen2_5_VLMultiModalProcessor,
    Qwen2_5_VLDummyInputsBuilder,
)
from vllm_omni.model_executor.models.mammoth_moda2.configuration_mammothmoda2 import (
    Mammothmoda2Config,
)
from vllm_omni.model_executor.models.mammoth_moda2.processing_mammothmoda2 import Mammothmoda2Processor


def moe_forward(
    hidden_states: torch.Tensor,
    und_expert: Callable[[torch.Tensor], torch.Tensor],
    gen_expert: Callable[[torch.Tensor], torch.Tensor] | None,
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
    # 若该层不启用 MoE（gen_expert=None），直接全量走 und_expert
    if gen_expert is None:
        return und_expert(hidden_states)
    # 若没有提供 mask，直接全量走 und_expert
    if gen_token_mask is None or not gen_token_mask.any():
        return und_expert(hidden_states)
    if gen_token_mask.all():
        return gen_expert(hidden_states)

    # vLLM 在 forward 中通常用 2D token-major layout: [num_tokens, hidden_size]；
    # 但在一些旧路径/外部调用中也可能是 3D: [B, L, D]。这里两者都支持。
    if hidden_states.ndim == 2:
        flat_hid = hidden_states
        d_model = hidden_states.shape[-1]
        total_tokens = hidden_states.shape[0]
    elif hidden_states.ndim == 3:
        d_model = hidden_states.shape[-1]
        flat_hid = hidden_states.reshape(-1, d_model)  # (B*L, D)
        total_tokens = flat_hid.shape[0]
    else:
        raise ValueError(f"Unexpected hidden_states shape: {tuple(hidden_states.shape)}")

    # mask: [num_tokens] or [B, L] -> flatten to [total_tokens]
    flat_mask = gen_token_mask.reshape(-1)  # type: ignore[union-attr]
    if flat_mask.numel() != total_tokens:
        raise ValueError(
            "gen_token_mask shape mismatch: "
            f"mask={tuple(gen_token_mask.shape)}, hidden_states={tuple(hidden_states.shape)}"
        )
    gen_pos = torch.where(flat_mask)[0]
    und_pos = torch.where(~flat_mask)[0]
    permute_order = torch.cat([gen_pos, und_pos], dim=0)
    inverse_order = torch.argsort(permute_order)
    gen_token_num = int(flat_mask.sum().item())
    gen_hid, und_hid = flat_hid[permute_order].split([gen_token_num, total_tokens - gen_token_num], dim=0)
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
    if hidden_states.ndim == 2:
        return merged.view(total_tokens, out_dim).contiguous()
    return merged.view(*hidden_states.shape[:-1], out_dim).contiguous()


class MammothModa2ARProcessingInfo(Qwen2_5_VLProcessingInfo):
    """处理 Mammoth Moda2 AR 的多模态信息，返回子配置的 VL config。"""

    def get_hf_config(self):
        mammoth_cfg: Mammothmoda2Config = self.ctx.get_hf_config(Mammothmoda2Config)
        llm_cfg = getattr(mammoth_cfg, "llm_config", None)
        if llm_cfg is not None:
            return llm_cfg
        # 兜底：若配置已是子配置则直接返回
        return getattr(mammoth_cfg, "text_config", mammoth_cfg)

    def get_hf_processor(self, **kwargs: object) -> Mammothmoda2Processor:
        return self.ctx.get_hf_processor(
            Mammothmoda2Processor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # MammothModa2 当前仅支持图像，不支持视频输入。
        return {"image": None}


class MammothModa2ARDummyInputsBuilder(Qwen2_5_VLDummyInputsBuilder):
    """复用 Qwen2.5-VL 的 dummy 输入生成逻辑。"""


class MammothModa2ARMultiModalProcessor(Qwen2_5_VLMultiModalProcessor):
    """复用 Qwen2.5-VL 的多模态处理，只调整 parser 初始化入口。"""

    def _get_data_parser(self) -> Qwen2VLMultiModalDataParser:
        return Qwen2VLMultiModalDataParser(
            spatial_merge_size=self.info.get_hf_config().vision_config.spatial_merge_size
        )


class Mammoth2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        patch_rope_parameters(config)
        set_default_rope_theta(config, default_theta=1000000)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )

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
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
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

        if 14 <= layer_idx < 28:
            self.gen_mlp = Qwen2MLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.gen_mlp",
            )
        else:
            self.gen_mlp = None
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

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


class MammothModa2Qwen2ForCausalLM(nn.Module):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 decoder_layer_type: type[nn.Module] = Mammoth2DecoderLayer):
        super().__init__()

        hf_config = vllm_config.model_config.hf_config
        if hasattr(hf_config, "get_text_config"):
            config = hf_config.get_text_config()
        elif hasattr(hf_config, "text_config"):
            config = hf_config.text_config
        else:
            config = hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.prefix = prefix

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
        # NOTE: MammothModa2 支持 extra gen vocab（用于图像 token）。
        # token id 范围为 [gen_vocab_start_index, gen_vocab_start_index + gen_vocab_size)。
        # vLLM 的采样器/processor 期望 “logits 的最后一维 == model_config.get_vocab_size()”，因此我们需要在
        # compute_logits 中输出 base+gen 的 logits，同时 embedding 也要能接收这些 token id。
        self.extra_gen_vocab = bool(getattr(config, "extra_gen_vocab", False))
        # 生成 token 的起始下标（用于 gen_token_mask）
        self.gen_vocab_start_index = getattr(
            hf_config, "gen_vocab_start_index", None
        ) or getattr(config, "gen_vocab_start_index", None)
        self.gen_vocab_size = int(getattr(config, "gen_vocab_size", 0) or 0)

        self.base_vocab_size = int(self.gen_vocab_start_index) if self.extra_gen_vocab else int(config.vocab_size)
        # 配置层面（hf_text_config.vocab_size）已经被上游配置类扩展为 base+gen，
        # 这里用 config.vocab_size 作为“总 vocab size”。
        self.total_vocab_size = int(getattr(config, "vocab_size", self.base_vocab_size))

        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                self.base_vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        if self.extra_gen_vocab:
            if get_pp_group().is_first_rank or (config.tie_word_embeddings and get_pp_group().is_last_rank):
                self.gen_embed_tokens = VocabParallelEmbedding(
                    self.gen_vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=f"{prefix}.gen_embed_tokens",
                )
            else:
                self.gen_embed_tokens = PPMissingLayer()
        else:
            self.gen_embed_tokens = None

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.base_vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.lm_head",
            )
        else:
            self.lm_head = PPMissingLayer()

        if self.extra_gen_vocab:
            if get_pp_group().is_last_rank:
                self.gen_head = ParallelLMHead(
                    self.gen_vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=f"{prefix}.gen_head",
                )
            else:
                self.gen_head = PPMissingLayer()
        else:
            self.gen_head = None

        # vLLM 的 logits 计算不能直接调用 ParallelLMHead.forward（会抛异常），
        # 需要通过 LogitsProcessor 使用其权重矩阵。
        self.logits_processor = LogitsProcessor(self.base_vocab_size)
        self.gen_logits_processor = LogitsProcessor(self.gen_vocab_size) if self.extra_gen_vocab else None

        # Use the provided decoder layer type or default to Qwen2DecoderLayer
        decoder_layer_type = decoder_layer_type or Mammoth2DecoderLayer

        def _make_decoder_layer(*, prefix: str) -> nn.Module:
            # vLLM make_layers 只会传 prefix，形如 "{prefix}.layers.{idx}"。
            # 这里从 prefix 提取 idx，确保 layer_idx 能用于按层启用 MoE。
            try:
                layer_idx = int(prefix.rsplit(".", 1)[-1])
            except Exception:
                layer_idx = 0
            return decoder_layer_type(
                config=config,
                layer_idx=layer_idx,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            _make_decoder_layer,
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    @property
    def model(self) -> "MammothModa2Qwen2ForCausalLM":
        # vLLM 的 Qwen2.5-VL 路径会调用 `language_model.model(...)` 来拿到 hidden states。
        # 这里用 property 避免把 self 注册成子模块从而形成 named_modules 的递归环。
        return self

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not self.extra_gen_vocab or self.gen_embed_tokens is None:
            return self.embed_tokens(input_ids)

        # input_ids 可能同时包含 base token 与 gen token；分别走不同 embedding 再合并。
        gen_mask = input_ids >= int(self.gen_vocab_start_index)
        if not gen_mask.any():
            return self.embed_tokens(input_ids)
        if gen_mask.all():
            gen_ids = input_ids - int(self.gen_vocab_start_index)
            return self.gen_embed_tokens(gen_ids)

        flat_ids = input_ids.reshape(-1)
        flat_mask = gen_mask.reshape(-1)
        out = torch.empty(
            (flat_ids.shape[0], self.config.hidden_size),
            dtype=self.embed_tokens.weight.dtype,  # type: ignore[attr-defined]
            device=flat_ids.device,
        )

        base_pos = torch.where(~flat_mask)[0]
        gen_pos = torch.where(flat_mask)[0]
        if base_pos.numel() > 0:
            out[base_pos] = self.embed_tokens(flat_ids[base_pos])
        if gen_pos.numel() > 0:
            gen_ids = flat_ids[gen_pos] - int(self.gen_vocab_start_index)
            out[gen_pos] = self.gen_embed_tokens(gen_ids)
        return out.view(*input_ids.shape, -1).contiguous()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                assert input_ids is not None
                hidden_states = self.get_input_embeddings(input_ids)
            # gen token mask: True 表示生成图像 token，走 gen_mlp
            # vLLM v1 路径下可能只提供 inputs_embeds，并将 input_ids 置为 None；
            # 此时无法按 token id 区分 gen token，退化为全量走 und_expert。
            if self.gen_vocab_start_index is None or input_ids is None:
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

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if isinstance(self.lm_head, PPMissingLayer):
            return None
        base_logits = self.logits_processor(self.lm_head, hidden_states)
        if not self.extra_gen_vocab:
            return base_logits
        if self.gen_head is None or isinstance(self.gen_head, PPMissingLayer):
            return base_logits
        assert self.gen_logits_processor is not None
        gen_logits = self.gen_logits_processor(self.gen_head, hidden_states)
        if base_logits is None or gen_logits is None:
            return None
        return torch.cat([base_logits, gen_logits], dim=-1)

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

            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue

                name = name.replace(weight_name, param_name)

                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name.endswith("scale"):
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(name)

        return loaded_params


@MULTIMODAL_REGISTRY.register_processor(
    MammothModa2ARMultiModalProcessor,
    info=MammothModa2ARProcessingInfo,
    dummy_inputs=MammothModa2ARDummyInputsBuilder,
)
class MammothModa2ARForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """在 Qwen2_5_VLForConditionalGeneration 的多模态框架下替换语言骨干为 MoE。"""

    # 兼容 gpu_model_runner 的 OmniOutput 解包逻辑。
    have_multimodal_outputs = True

    # 复用原有权重映射，再加一层 model. -> language_model.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # 生成侧（DiT/VAE）权重不属于 AR stage，跳过
            "gen_image_condition_refiner.": None,
            "gen_transformer.": None,
            "gen_vae.": None,

            # LLM 主体：checkpoint 使用 llm_model.* 前缀
            # 额外 gen vocab（图像 token）权重：需要单独映射到 vLLM 的 language_model 子模块
            "llm_model.model.language_model.gen_embed_tokens.": "language_model.gen_embed_tokens.",
            "llm_model.gen_head.": "language_model.gen_head.",

            "llm_model.model.language_model.": "language_model.",
            "llm_model.model.visual.": "visual.",
            "llm_model.lm_head.": "language_model.lm_head.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # 将 hf_config 切换为 AR 子配置，保证 Qwen2.5-VL 路径拿到正确类型
        mammoth_cfg = vllm_config.model_config.hf_config
        ar_hf_config = getattr(mammoth_cfg, "llm_config", mammoth_cfg)
        ar_vllm_config = vllm_config.with_hf_config(
            ar_hf_config, architectures=vllm_config.model_config.architectures
        )
        # 先初始化视觉塔等多模态组件
        super().__init__(vllm_config=ar_vllm_config, prefix=prefix)
        # 用自定义 MoE 语言模型替换
        lm_hf_config = getattr(
            ar_vllm_config.model_config.hf_config, "text_config", ar_vllm_config.model_config.hf_config
        )
        self.language_model = init_vllm_registered_model(
            vllm_config=ar_vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            hf_config=lm_hf_config,
            architectures=["MammothModa2Qwen2ForCausalLM"],
        )
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

        # -------- t2i (AR grid) token constraints --------
        # 约束逻辑依赖 per-step 的 sampling_metadata + runtime_additional_information。
        # 这些都由 vllm-omni runner 每步通过 kwargs 传入，因此只需在模型侧缓存即可。
        self._last_sampling_metadata = None
        self._last_runtime_additional_information: list[dict[str, Any]] | None = None
        self._t2i_defaults: dict[str, int] = {}
        self._try_load_t2i_generation_defaults(vllm_config)

    def _try_load_t2i_generation_defaults(self, vllm_config: VllmConfig) -> None:
        """Best-effort 从模型目录读取 t2i_generation_config.json 的默认 token 范围。"""
        try:
            model_dir = getattr(vllm_config.model_config, "model", None)
            if not isinstance(model_dir, str) or not model_dir:
                return
            cfg_path = Path(model_dir) / "t2i_generation_config.json"
            if not cfg_path.is_file():
                return
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            eol = int(cfg.get("eol_token_id"))
            visual_start = int(cfg.get("visual_token_start_id"))
            visual_end = int(cfg.get("visual_token_end_id"))
            self._t2i_defaults = {
                "eol_token_id": eol,
                "visual_token_start_id": visual_start,
                "visual_token_end_id": visual_end,
            }
        except Exception:
            # 不要因默认配置读取失败而影响模型初始化；请求侧可通过 additional_information 传入。
            return

    @staticmethod
    def _is_t2i_request(runtime_info: dict[str, Any]) -> bool:
        v = runtime_info.get("omni_task", runtime_info.get("task"))
        return v[0].lower() in {"t2i", "text_image", "text2image"}

    def _get_t2i_runtime_params(self, runtime_info: dict[str, Any]) -> dict[str, int] | None:
        """从 runtime_info / defaults 获取 t2i 约束参数。"""
        def _get_int(key: str) -> int | None:
            val = runtime_info.get(key, self._t2i_defaults.get(key))
            if val is None:
                return None
            try:
                if isinstance(val, list) and val:
                    val = val[0]
                if isinstance(val, torch.Tensor):
                    if val.numel() == 0:
                        return None
                    val = val.flatten()[0].item()
                return int(val)
            except Exception:
                return None

        ar_width = _get_int("ar_width")
        ar_height = _get_int("ar_height")
        eol = _get_int("eol_token_id")
        visual_start = _get_int("visual_token_start_id")
        visual_end = _get_int("visual_token_end_id")
        if ar_width is None or eol is None or visual_start is None or visual_end is None:
            return None
        params = {
            "ar_width": ar_width,
            "eol_token_id": eol,
            "visual_token_start_id": visual_start,
            "visual_token_end_id": visual_end,
        }
        if ar_height is not None:
            params["ar_height"] = ar_height
        return params

    def _apply_t2i_token_constraints(self, logits: torch.Tensor) -> torch.Tensor:
        """对 logits 做 t2i 动态约束：每行末尾 token 强制只能采样 eol_token_id。"""
        if logits is None or not isinstance(logits, torch.Tensor):
            return logits
        sampling_metadata = getattr(self, "_last_sampling_metadata", None)
        runtime_infos = getattr(self, "_last_runtime_additional_information", None)
        if sampling_metadata is None or runtime_infos is None:
            return logits
        output_token_ids = getattr(sampling_metadata, "output_token_ids", None)
        if not isinstance(output_token_ids, list):
            return logits
        if logits.ndim != 2:
            return logits
        num_reqs = logits.shape[0]
        if num_reqs == 0:
            return logits
        if len(output_token_ids) < num_reqs or len(runtime_infos) < num_reqs:
            return logits

        neg_inf = -float("inf")
        vocab_size = logits.shape[-1]
        for i in range(num_reqs):
            runtime_info = runtime_infos[i] if isinstance(runtime_infos[i], dict) else {}
            if not self._is_t2i_request(runtime_info):
                continue
            params = self._get_t2i_runtime_params(runtime_info)
            if params is None:
                continue

            ar_width = int(params["ar_width"])
            ar_height = int(params["ar_height"])
            eol_token_id = int(params["eol_token_id"])
            visual_start = int(params["visual_token_start_id"])
            visual_end = int(params["visual_token_end_id"])
            expected_token_num = (ar_width + 1) * ar_height

            if ar_width < 0:
                continue
            if not (0 <= eol_token_id < vocab_size):
                continue
            # 允许范围裁剪到 vocab 内，避免越界
            visual_start_c = max(0, min(visual_start, vocab_size))
            visual_end_c = max(0, min(visual_end, vocab_size))
            if visual_end_c <= visual_start_c:
                continue

            generated_len = len(output_token_ids[i]) if isinstance(output_token_ids[i], list) else 0
            w = generated_len % (ar_width + 1)

            row = logits[i]
            if w == ar_width:
                # 行末 token：只允许 eol
                eol_logit = row[eol_token_id].clone()
                row.fill_(neg_inf)
                row[eol_token_id] = eol_logit
            else:
                # 行内 token：只允许视觉 token（并显式禁止 eol）
                row[:visual_start_c] = neg_inf
                row[visual_end_c:] = neg_inf
                row[eol_token_id] = neg_inf
            
            if generated_len >= expected_token_num:
                row.fill_(neg_inf)
                end_of_image_id = 152071
                row[end_of_image_id] = 1.0  # Allow only end_of_image_id after expected tokens

        return logits

    def _build_dummy_mm_embeddings(
        self,
        grid_thw: torch.Tensor,
    ) -> list[torch.Tensor]:
        """在 profiling 路径兜底生成零向量 embedding，避免 mm sanity check 失败。"""
        if not isinstance(grid_thw, torch.Tensor) or grid_thw.numel() == 0:
            return []
        if grid_thw.ndim != 2 or grid_thw.shape[-1] != 3:
            return []

        merge_size = getattr(self.visual, "spatial_merge_size", 1)  # type: ignore[union-attr]
        hidden_size = int(self.language_model.config.hidden_size)
        device = self.visual.device if self.visual is not None else grid_thw.device
        dtype = self.visual.dtype if self.visual is not None else torch.float16

        embeds: list[torch.Tensor] = []
        for t, h, w in grid_thw.tolist():
            tokens = int(t * h * w) // int(merge_size) // int(merge_size)
            embeds.append(
                torch.zeros((tokens, hidden_size), device=device, dtype=dtype)
            )
        return embeds

    def _normalize_mm_kwargs(
        self,
        kwargs: dict[str, object],
    ) -> dict[str, object]:
        """将 batch 维展开为 vLLM 期望的扁平形状。"""
        out = dict(kwargs)
        pv = out.get("pixel_values")
        if isinstance(pv, torch.Tensor) and pv.ndim == 3:
            out["pixel_values"] = pv.reshape(-1, pv.shape[-1])
        pv_v = out.get("pixel_values_videos")
        if isinstance(pv_v, torch.Tensor) and pv_v.ndim == 3:
            out["pixel_values_videos"] = pv_v.reshape(-1, pv_v.shape[-1])
        ig = out.get("image_grid_thw")
        if isinstance(ig, torch.Tensor) and ig.ndim == 3 and ig.shape[1] == 1:
            out["image_grid_thw"] = ig.squeeze(1)
        vg = out.get("video_grid_thw")
        if isinstance(vg, torch.Tensor) and vg.ndim == 3 and vg.shape[1] == 1:
            out["video_grid_thw"] = vg.squeeze(1)
        ie = out.get("image_embeds")
        if isinstance(ie, torch.Tensor) and ie.ndim == 3:
            out["image_embeds"] = ie.reshape(-1, ie.shape[-1])
        ve = out.get("video_embeds")
        if isinstance(ve, torch.Tensor) and ve.ndim == 3:
            out["video_embeds"] = ve.reshape(-1, ve.shape[-1])
        return out

    def embed_multimodal(self, **kwargs: object) -> list[torch.Tensor]:
        # 先走原生实现（支持 pixel_values / image_embeds / video_embeds）。
        mm_kwargs = self._normalize_mm_kwargs(kwargs)
        mm_embeddings = super().embed_multimodal(**mm_kwargs)
        if mm_embeddings:
            return list(mm_embeddings)

        # vLLM profiling 的 dummy mm inputs 在某些路径下只带 grid 元信息，
        # 此时需要构造占位 embedding 以通过 sanity check。
        embeds: list[torch.Tensor] = []
        image_grid_thw = mm_kwargs.get("image_grid_thw")
        video_grid_thw = mm_kwargs.get("video_grid_thw")
        if isinstance(image_grid_thw, torch.Tensor):
            embeds.extend(self._build_dummy_mm_embeddings(image_grid_thw))
        if isinstance(video_grid_thw, torch.Tensor):
            embeds.extend(self._build_dummy_mm_embeddings(video_grid_thw))
        return embeds


    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        # vllm-omni runner 会在每步 forward 传入 sampling_metadata 与 runtime_additional_information。
        # compute_logits 在 forward 之后被立刻调用，因此这里缓存即可实现“随步动态”的 token 约束。
        self._last_sampling_metadata = kwargs.get("sampling_metadata")
        runtime_infos = kwargs.get("runtime_additional_information")
        self._last_runtime_additional_information = (
            runtime_infos if isinstance(runtime_infos, list) else None
        )
        hidden_states = super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        # NOTE:
        # `gpu_model_runner._dummy_run` 会在 forward 之后对 hidden_states 做
        # `hidden_states[logit_indices]` 抽样，因此这里必须保证
        # `text_hidden_states` 是 `torch.Tensor`。
        if isinstance(hidden_states, IntermediateTensors):
            text_hidden_states = hidden_states["hidden_states"]
            out_intermediate_tensors = hidden_states
        elif isinstance(hidden_states, list):
            text_hidden_states = hidden_states[0]
            out_intermediate_tensors = None
        else:
            text_hidden_states = hidden_states
            out_intermediate_tensors = None

        return OmniOutput(
            text_hidden_states=text_hidden_states,
            multimodal_outputs={},
            intermediate_tensors=out_intermediate_tensors,
        )

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput):
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        logits = super().compute_logits(hidden_states)
        if isinstance(logits, torch.Tensor):
            logits = self._apply_t2i_token_constraints(logits)
        return logits
