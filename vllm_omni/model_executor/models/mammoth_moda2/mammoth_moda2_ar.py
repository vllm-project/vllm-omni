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

    if gen_expert is None:
        return und_expert(hidden_states)
    
    if gen_token_mask is None or not gen_token_mask.any():
        return und_expert(hidden_states)
    if gen_token_mask.all():
        return gen_expert(hidden_states)

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

    # 1.1 Generation tokens (True)
    gen_out = gen_expert(gen_hid)  # (N_gen, D)

    # 1.2 Understanding tokens (False)
    und_out = und_expert(und_hid)  # (N_und, D)
    out_dim = und_out.shape[-1]

    merged = torch.cat([gen_out, und_out], dim=0)
    merged = merged[inverse_order]

    if hidden_states.ndim == 2:
        return merged.view(total_tokens, out_dim).contiguous()
    return merged.view(*hidden_states.shape[:-1], out_dim).contiguous()


class MammothModa2ARProcessingInfo(Qwen2_5_VLProcessingInfo):
    """Processes multi-modal information for MammothModa2 AR, returning the VL sub-configuration."""

    def get_hf_config(self):
        mammoth_cfg: Mammothmoda2Config = self.ctx.get_hf_config(Mammothmoda2Config)
        llm_cfg = getattr(mammoth_cfg, "llm_config", None)
        if llm_cfg is not None:
            return llm_cfg
        # Fallback: return directly if the config is already a sub-config.
        return getattr(mammoth_cfg, "text_config", mammoth_cfg)

    def get_hf_processor(self, **kwargs: object) -> Mammothmoda2Processor:
        return self.ctx.get_hf_processor(
            Mammothmoda2Processor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # MammothModa2 currently supports only image input, not video.
        return {"image": None}


class MammothModa2ARDummyInputsBuilder(Qwen2_5_VLDummyInputsBuilder):
    """ Reuse Qwen2.5-VL's dummy input generation logic. """


class MammothModa2ARMultiModalProcessor(Qwen2_5_VLMultiModalProcessor):
    """ Reuse Qwen2.5-VL's multi-modal processing, """
    """only adjusting parser initialization entry. """

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
        # NOTE: MammothModa2 supports extra generation vocabulary (for image tokens).
        # Token ID range: [gen_vocab_start_index, gen_vocab_start_index + gen_vocab_size).
        # vLLM sampler/processor expects "last dimension of logits == model_config.get_vocab_size()",
        # so we output base+gen logits in compute_logits, and embeddings must accept these IDs.
        self.extra_gen_vocab = bool(getattr(config, "extra_gen_vocab", False))
        # Starting index for generation tokens (used for gen_token_mask).
        self.gen_vocab_start_index = getattr(
            hf_config, "gen_vocab_start_index", None
        ) or getattr(config, "gen_vocab_start_index", None)
        self.gen_vocab_size = int(getattr(config, "gen_vocab_size", 0) or 0)

        self.base_vocab_size = int(self.gen_vocab_start_index) if self.extra_gen_vocab else int(config.vocab_size)
        # The configuration level (hf_text_config.vocab_size) has been extended to base+gen 
        # by the upstream config class. Use config.vocab_size as the total vocab size.
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

        # vLLM logits computation cannot directly call ParallelLMHead.forward (throws exception);
        # it must use the weight matrix via LogitsProcessor.
        self.logits_processor = LogitsProcessor(self.base_vocab_size)
        self.gen_logits_processor = LogitsProcessor(self.gen_vocab_size) if self.extra_gen_vocab else None

        # Use the provided decoder layer type or default to Qwen2DecoderLayer
        decoder_layer_type = decoder_layer_type or Mammoth2DecoderLayer

        def _make_decoder_layer(*, prefix: str) -> nn.Module:
            # vLLM make_layers only passes prefix, e.g., "{prefix}.layers.{idx}".
            # Extract idx here to ensure layer_idx can be used for layer-wise MoE activation.
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
            # gen_token_mask: True indicates image generation tokens, which use gen_mlp.
            # In vLLM v1 path, only inputs_embeds might be provided, with input_ids set to None.
            # In this case, gen tokens cannot be distinguished by ID, falling back to und_expert.
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
    """Replaces the language backbone with MoE within the Qwen2_5_VLForConditionalGeneration multi-modal framework."""

    have_multimodal_outputs = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # Skip generation-side (DiT/VAE) weights as they do not belong to the AR stage.
            "gen_image_condition_refiner.": None,
            "gen_transformer.": None,
            "gen_vae.": None,

            # LLM backbone: checkpoint uses the llm_model.* prefix.
            # Extra generation vocab (image tokens) weights: mapped separately to the vLLM language_model submodule.
            "llm_model.model.language_model.gen_embed_tokens.": "language_model.gen_embed_tokens.",
            "llm_model.gen_head.": "language_model.gen_head.",

            "llm_model.model.language_model.": "language_model.",
            "llm_model.model.visual.": "visual.",
            "llm_model.lm_head.": "language_model.lm_head.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Switch hf_config to the AR sub-config to ensure the Qwen2.5-VL path receives the correct type.
        mammoth_cfg = vllm_config.model_config.hf_config
        ar_hf_config = getattr(mammoth_cfg, "llm_config", mammoth_cfg)
        ar_vllm_config = vllm_config.with_hf_config(
            ar_hf_config, architectures=vllm_config.model_config.architectures
        )
        # Initialize multi-modal components like the vision tower first.
        super().__init__(vllm_config=ar_vllm_config, prefix=prefix)
        # Replace with the custom MoE language model.
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
        # Constraint logic depends on per-step sampling_metadata + runtime_additional_information.
        # These are passed by the vllm-omni runner via kwargs, so caching them in the model is sufficient.
        self._last_runtime_additional_information: list[dict[str, Any]] | None = None

    def _apply_t2i_token_constraints(self, logits: torch.Tensor) -> torch.Tensor:
        """Applies per-request token constraints.

        - For T2I requests: constrain AR grid tokens (force EOL at row end and
          restrict intra-row sampling to visual token range).
        - For non-T2I (text/understanding/chat) requests: disallow sampling
          from the extra generation vocabulary (image tokens) to prevent
          accidentally emitting visual-token sequences.
        """
        if logits is None or not isinstance(logits, torch.Tensor):
            return logits
        
        runtime_infos = self._last_runtime_additional_information

        if runtime_infos is None:
            # There is no runtime info in dummy/profile run
            return logits

        neg_inf = -float("inf")
        num_reqs = int(logits.shape[0])
        for i in range(num_reqs):
            runtime_info = runtime_infos[i] if isinstance(runtime_infos[i], dict) else {}
            if runtime_info["omni_task"][0] != "t2i":
                # Text/understanding/chat: forbid sampling from the extra gen vocab.
                logits[i, self.language_model.base_vocab_size:] = neg_inf
                continue
            
            ar_width = runtime_info["ar_width"][0]
            ar_height = runtime_info["ar_height"][0]
            eol_token_id = runtime_info["eol_token_id"][0]
            visual_start = runtime_info["visual_token_start_id"][0]
            visual_end = runtime_info["visual_token_end_id"][0]
            generated_len = runtime_info["generated_len"]

            expected_token_num = (ar_width + 1) * ar_height

            row = logits[i]
            column_id = generated_len % (ar_width + 1)
            if column_id == ar_width:
                # End-of-row token: only allow eol.
                eol_logit = row[eol_token_id].clone()
                row.fill_(neg_inf)
                row[eol_token_id] = eol_logit
            else:
                # Intra-row tokens: only allow visual tokens (explicitly forbid eol).
                row[:visual_start] = neg_inf
                row[visual_end:] = neg_inf
                row[eol_token_id] = neg_inf
            
            if generated_len >= expected_token_num:
                row.fill_(neg_inf)
                end_of_image_id = 152071
                row[end_of_image_id] = 1.0  # Allow only end_of_image_id after expected tokens num

        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        # vllm-omni runner passes sampling_metadata and runtime_additional_information in each forward step.
        # compute_logits is called immediately after forward, so caching here enables step-by-step dynamic token constraints.
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
        # NOTE: gpu_model_runner._dummy_run performs hidden_states[logit_indices] after forward.
        # We must ensure text_hidden_states is a torch.Tensor to avoid errors when 
        # indexing (which happens if it's a list/tuple).
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
