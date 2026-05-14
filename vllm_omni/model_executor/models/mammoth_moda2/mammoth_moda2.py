from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from itertools import islice
from typing import Any

import torch
from torch import nn
from transformers import Qwen2Config
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
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
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer, Qwen2MLP
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLDummyInputsBuilder,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLMultiModalProcessor,
    Qwen2_5_VLProcessingInfo,
)
from vllm.model_executor.models.qwen2_vl import Qwen2VLMultiModalDataParser
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    WeightsMapper,
    init_vllm_registered_model,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import (
    is_interleaved,
    patch_rope_parameters,
    set_default_rope_theta,
)

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.utils import add_prefix_to_loaded_weights
from vllm_omni.transformers_utils.configs.mammoth_moda2 import Mammothmoda2Config


def moe_enable(moe_type, layer_type, layer_idx) -> bool:
    """Determine if MoE should be enabled for a specific layer type and index.

    Args:
        moe_type (str): The MoE configuration string (e.g., "ffn", "ffn_attention-14:28").
        layer_type (str): The type of layer being checked (e.g., "ffn", "attention").
        layer_idx (int): The index of the current layer.

    Returns:
        bool: True if MoE should be enabled for this layer, False otherwise.
    """
    if ":" in moe_type:
        # moe_type like ffn_attention-14:28
        moe_type, layers = moe_type.split("-")
        start, end = [int(n) for n in layers.split(":")]
    else:
        start, end = 0, float("inf")
    assert moe_type in ["none", "attention", "ffn", "ffn_attention"]
    return layer_type in moe_type and start <= layer_idx < end


def moe_forward(
    hidden_states: torch.Tensor,
    und_expert: Callable[[torch.Tensor], torch.Tensor],
    gen_expert: Callable[[torch.Tensor], torch.Tensor] | None,
    gen_token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Perform Mixture-of-Experts (MoE) routing and forward pass.

    This function routes tokens to either the understanding expert (`und_expert`) or the
    generation expert (`gen_expert`) based on the `gen_token_mask`.

    Routing Logic:
    - If `gen_expert` is None: All tokens go to `und_expert`.
    - If `gen_token_mask` is None or all False: All tokens go to `und_expert`.
    - If `gen_token_mask` is all True: All tokens go to `gen_expert`.
    - Otherwise (mixed batch):
        - Tokens where `gen_token_mask` is True go to `gen_expert`.
        - Tokens where `gen_token_mask` is False go to `und_expert`.
        - Results are concatenated and reordered to match the original input order.

    Args:
        hidden_states (torch.Tensor): Input hidden states. Shape: `(batch_size, seq_len, hidden_size)`
            or `(num_tokens, hidden_size)`.
        und_expert (Callable): The expert module for understanding/text tokens.
            Takes `(N, D)` tensor, returns `(N, D_out)`.
        gen_expert (Callable | None): The expert module for generation/image tokens, or None.
            If provided, takes `(N, D)` tensor, returns `(N, D_out)`.
        gen_token_mask (torch.Tensor | None): Boolean mask indicating generation tokens.
            Shape matches `hidden_states` (excluding feature dim). True for generation tokens.

    Returns:
        torch.Tensor: The processed hidden states with the same shape as input (except potentially
        different feature dimension if `D_out != D`).
    """
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

    # Validate before reshape to catch shape mismatches where numel() would
    # coincidentally match after flattening dimensions of different sizes.
    if gen_token_mask.numel() != total_tokens:  # type: ignore[union-attr]
        raise ValueError(
            "gen_token_mask shape mismatch: "
            f"mask={tuple(gen_token_mask.shape)}, hidden_states={tuple(hidden_states.shape)}"
        )
    # mask: [num_tokens] or [B, L] -> flatten to [total_tokens]
    flat_mask = gen_token_mask.reshape(-1)  # type: ignore[union-attr]
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


class Mammothmoda2Processor(Qwen2_5_VLProcessor):
    """Qwen2.5-VL Processor with MammothU tokenizer."""

    tokenizer_class = ("MammothUTokenizer", None)


class MammothModa2ARProcessingInfo(Qwen2_5_VLProcessingInfo):
    """Processes multi-modal information for MammothModa2 AR, returning the VL sub-configuration."""

    def get_hf_config(self):
        mammoth_cfg: Mammothmoda2Config = self.ctx.get_hf_config(Mammothmoda2Config)
        llm_cfg = getattr(mammoth_cfg, "llm_config", None)
        return llm_cfg

    def get_hf_processor(self, **kwargs: object) -> Mammothmoda2Processor:
        return self.ctx.get_hf_processor(
            Mammothmoda2Processor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # MammothModa2 currently supports only image input, not video.
        return {"image": None}

    def get_data_parser(self) -> Qwen2VLMultiModalDataParser:
        # vLLM >=0.16 expects the parser to be provided by ProcessingInfo,
        # not by BaseMultiModalProcessor._get_data_parser.
        return Qwen2VLMultiModalDataParser(
            spatial_merge_size=self.get_hf_config().vision_config.spatial_merge_size,
            expected_hidden_size=self._get_expected_hidden_size(),
        )


class MammothModa2ARDummyInputsBuilder(Qwen2_5_VLDummyInputsBuilder):
    """Reuse Qwen2.5-VL's dummy input generation logic."""


class MammothModa2ARMultiModalProcessor(Qwen2_5_VLMultiModalProcessor):
    """Reuse Qwen2.5-VL's multi-modal processing,"""


class Mammoth2DecoderLayer(Qwen2DecoderLayer):
    def __init__(
        self,
        config: Qwen2Config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        # Must patch rope parameters on config BEFORE calling super().__init__,
        # because Qwen2DecoderLayer.__init__ creates Qwen2Attention using the
        # config's rope settings.  Patching afterwards would leave the attention
        # module initialised with wrong parameters, causing shape mismatches in
        # rotary_embedding at runtime.
        patch_rope_parameters(config)
        set_default_rope_theta(config, default_theta=1000000)
        super().__init__(config, cache_config, quant_config, prefix)

        self.moe_enable = moe_enable(config.moe_type, "ffn", layer_idx)
        if self.moe_enable:
            self.gen_mlp = Qwen2MLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.gen_mlp",
            )
        else:
            self.gen_mlp = None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        gen_token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = moe_forward(hidden_states, self.mlp, self.gen_mlp, gen_token_mask)
        return hidden_states, residual


class MammothModa2Qwen2ForCausalLM(nn.Module, SupportsPP):
    def __init__(
        self, *, vllm_config: VllmConfig, prefix: str = "", decoder_layer_type: type[nn.Module] = Mammoth2DecoderLayer
    ):
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

        if is_interleaved(vllm_config.model_config.hf_text_config):
            assert config.max_window_layers == config.num_hidden_layers, (
                "Sliding window for some but all layers is not supported. "
                f"This model uses sliding window but `max_window_layers` = {config.max_window_layers} "
                f"is less than `num_hidden_layers` = {config.num_hidden_layers}. Please open an issue "
                "to discuss this feature."
            )

        self.config = config
        self.quant_config = quant_config
        # NOTE: MammothModa2 supports extra generation vocabulary (for image tokens).
        # Token ID range: [gen_vocab_start_index, gen_vocab_start_index + gen_vocab_size).
        # vLLM sampler/processor expects "last dimension of logits == model_config.get_vocab_size()",
        # so we output base+gen logits in compute_logits, and embeddings must accept these IDs.
        self.extra_gen_vocab = bool(getattr(config, "extra_gen_vocab", False))
        # Starting index for generation tokens (used for gen_token_mask).
        self.gen_vocab_start_index = getattr(hf_config, "gen_vocab_start_index", None) or getattr(
            config, "gen_vocab_start_index", None
        )
        self.gen_vocab_size = int(getattr(config, "gen_vocab_size", 0) or 0)

        self.base_vocab_size = int(self.gen_vocab_start_index) if self.extra_gen_vocab else int(config.vocab_size)
        # The configuration level (hf_text_config.vocab_size) has been extended to base+gen
        # by the upstream config class. Use config.vocab_size as the total vocab size.
        self.total_vocab_size = int(getattr(config, "vocab_size", self.base_vocab_size))

        if get_pp_group().is_first_rank or (config.tie_word_embeddings and get_pp_group().is_last_rank):
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

        self.logits_processor = LogitsProcessor(self.base_vocab_size)
        self.gen_logits_processor = LogitsProcessor(self.gen_vocab_size) if self.extra_gen_vocab else None

        decoder_layer_type = decoder_layer_type or Mammoth2DecoderLayer

        def _make_decoder_layer(*, prefix: str) -> nn.Module:
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

        def _make_empty_intermediate_tensors(
            batch_size: int,
            dtype: torch.dtype,
            device: torch.device,
        ) -> IntermediateTensors:
            return IntermediateTensors(
                {
                    "hidden_states": torch.zeros((batch_size, config.hidden_size), dtype=dtype, device=device),
                    "residual": torch.zeros((batch_size, config.hidden_size), dtype=dtype, device=device),
                    "gen_token_mask": torch.zeros((batch_size,), dtype=torch.bool, device=device),
                }
            )

        self.make_empty_intermediate_tensors = _make_empty_intermediate_tensors
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    @property
    def model(self) -> MammothModa2Qwen2ForCausalLM:
        return self

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not self.extra_gen_vocab or self.gen_embed_tokens is None:
            return self.embed_tokens(input_ids)

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
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
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
            gen_token_mask = intermediate_tensors.tensors.get("gen_token_mask")

        for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
            hidden_states, residual = layer(positions, hidden_states, residual, gen_token_mask)

        if not get_pp_group().is_last_rank:
            tensors = {"hidden_states": hidden_states, "residual": residual}
            if gen_token_mask is not None:
                tensors["gen_token_mask"] = gen_token_mask
            return IntermediateTensors(tensors)

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

            if self.quant_config is not None and (scale_name := self.quant_config.get_cache_scale(name)):
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
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
        ar_vllm_config = vllm_config.with_hf_config(ar_hf_config, architectures=vllm_config.model_config.architectures)
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
            meta = runtime_info.get("meta", {})
            omni_task = meta.get("omni_task")
            if not isinstance(omni_task, list) or not omni_task or omni_task[0] != "t2i":
                # Text/understanding/chat: forbid sampling from the extra gen vocab.
                logits[i, self.language_model.base_vocab_size :] = neg_inf
                continue

            ar_width = meta["ar_width"][0]
            eol_token_id = meta["eol_token_id"][0]
            visual_start = meta["visual_token_start_id"][0]
            visual_end = meta["visual_token_end_id"][0]
            generated_len = runtime_info["generated_len"]

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
                row[visual_end + 1 :] = neg_inf
                row[eol_token_id] = neg_inf

        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        # vllm-omni runner passes sampling_metadata and runtime_additional_information
        # in each forward step. compute_logits is called immediately after
        # forward, so caching here enables step-by-step dynamic token constraints.
        runtime_infos = kwargs.get("runtime_additional_information")
        self._last_runtime_additional_information = runtime_infos if isinstance(runtime_infos, list) else None
        hidden_states = super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if isinstance(hidden_states, IntermediateTensors) and not get_pp_group().is_last_rank:
            return hidden_states
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


@MULTIMODAL_REGISTRY.register_processor(
    MammothModa2ARMultiModalProcessor,
    info=MammothModa2ARProcessingInfo,
    dummy_inputs=MammothModa2ARDummyInputsBuilder,
)
class MammothModa2ForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    # Ensure vllm_omni/worker/gpu_model_runner.py's `extract_multimodal_outputs` follows
    # the OmniOutput branch to retrieve text_hidden_states as a pure torch.Tensor,
    # preventing errors in `hidden_states[logit_indices]` due to type mismatch (list/tuple).
    have_multimodal_outputs = True

    multimodal_cpu_fields = {"image_grid_thw", "video_grid_thw"}
    merge_by_field_config = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        # Consistent with Qwen2_5OmniForConditionalGeneration: instance-level flag.
        self.have_multimodal_outputs = True
        self.vllm_config = vllm_config
        cfg = vllm_config.model_config.hf_config
        self.model_stage = vllm_config.model_config.model_stage
        self.multimodal_config = vllm_config.model_config.multimodal_config

        # For debugging/alignment with qwen2.5-omni: explicitly nullify unused stages.
        self.ar = None
        self.dit = None
        self.vae = None

        if self.model_stage == "ar":
            # AR stage: multi-modal + MoE text.
            self.ar = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "ar"),
                hf_config=cfg.llm_config if hasattr(cfg, "llm_config") else cfg.text_config,
                architectures=["MammothModa2ARForConditionalGeneration"],
            )
            self.model = self.ar
        elif self.model_stage == "dit":
            self.dit = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "dit"),
                # NOTE: init_vllm_registered_model -> VllmConfig.with_hf_config requires a
                # transformers.PretrainedConfig; however, Mammothmoda2Config.gen_dit_config
                # is a dict (diffusers config). The DiT stage hf_config still uses the
                # top-level Mammothmoda2Config, and the DiT module reads its own
                # gen_dit_config / gen_vae_config dicts.
                hf_config=cfg,
                architectures=["MammothModa2DiTPipeline"],
            )
            self.model = self.dit
        elif self.model_stage == "vae":
            # Reserved: VAEs not implemented yet; raise explicit error.
            raise NotImplementedError("MammothModa2 VAE stage not implemented yet.")
        else:
            raise ValueError(f"Unsupported model_stage: {self.model_stage}")

        # Expose intermediate tensor factory for PP if provided by the submodule.
        self.make_empty_intermediate_tensors = getattr(self.model, "make_empty_intermediate_tensors", lambda: None)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int):  # noqa: ARG003
        return Qwen2_5_VLForConditionalGeneration.get_placeholder_str(modality, i)

    def get_language_model(self) -> nn.Module:
        if hasattr(self.model, "get_language_model"):
            return self.model.get_language_model()
        return self.model

    def get_multimodal_embeddings(self, **kwargs: object):
        # Backward compatibility: route through embed_multimodal.
        return self.embed_multimodal(**kwargs)

    def embed_multimodal(self, **kwargs: object):
        if hasattr(self.model, "embed_multimodal"):
            return self.model.embed_multimodal(**kwargs)
        if hasattr(self.model, "get_multimodal_embeddings"):
            return self.model.get_multimodal_embeddings(**kwargs)
        return []

    def get_input_embeddings(self, input_ids: torch.Tensor, multimodal_embeddings=None) -> torch.Tensor:
        if hasattr(self.model, "get_input_embeddings"):
            return self.model.get_input_embeddings(input_ids, multimodal_embeddings=multimodal_embeddings)
        # DiT stage does not consume token embeddings from `input_ids`; it uses
        # condition embeddings passed via additional_information.
        # However, vLLM's generation runner may still request token embeddings
        # to populate `inputs_embeds` buffers, so we provide a dummy tensor.
        if self.model_stage == "dit":
            hidden_size = int(self.vllm_config.model_config.get_hidden_size())
            try:
                target_dtype = next(self.model.parameters()).dtype
            except StopIteration:
                target_dtype = self.vllm_config.model_config.dtype
            return torch.zeros(
                (input_ids.numel(), hidden_size),
                device=input_ids.device,
                dtype=target_dtype,
            )
        raise NotImplementedError("Underlying model does not implement get_input_embeddings")

    def forward(self, *args, **kwargs) -> OmniOutput | torch.Tensor:
        out = self.model(*args, **kwargs)
        if isinstance(out, OmniOutput):
            return out
        if isinstance(out, list):
            out = out[0]
        return OmniOutput(text_hidden_states=out, multimodal_outputs={}, intermediate_tensors=None)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, *args, **kwargs):
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hasattr(self.model, "compute_logits"):
            return self.model.compute_logits(hidden_states)
        return None

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict[str, object]]:
        if self.model_stage != "dit":
            raise RuntimeError(
                f"get_dummy_runtime_additional_information only valid for dit stage, got {self.model_stage}"
            )
        if self.dit is None:
            raise RuntimeError("dit stage model is not initialized")
        if not hasattr(self.dit, "get_dummy_runtime_additional_information"):
            raise AttributeError("dit model missing get_dummy_runtime_additional_information")
        return self.dit.get_dummy_runtime_additional_information(num_reqs)

    def load_weights(self, weights):
        if self.model_stage == "ar":
            if self.ar is None or not hasattr(self.ar, "load_weights"):
                return set()
            loaded = self.ar.load_weights(weights)
            return add_prefix_to_loaded_weights(loaded, "ar")
        if self.model_stage == "dit":
            if self.dit is None or not hasattr(self.dit, "load_weights"):
                return set()
            loaded = self.dit.load_weights(weights)
            return add_prefix_to_loaded_weights(loaded, "dit")
        if self.model_stage == "vae":
            return set()
        raise ValueError(f"Unsupported model_stage: {self.model_stage}")
