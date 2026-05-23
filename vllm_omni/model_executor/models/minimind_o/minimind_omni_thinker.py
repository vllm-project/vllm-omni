# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.

import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import BatchFeature, PreTrainedModel, SiglipImageProcessor, SiglipVisionModel
from vllm.config import VllmConfig
from vllm.config.cache import CacheConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
)
from vllm.model_executor.models.utils import (
    WeightsMapper,
    _merge_multimodal_embeddings,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.parse import (
    AudioProcessorItems,
    ImageProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.model_executor.models.minimind_o.minimind_omni_config import (
    MiniMindConfig,
    MiniMindOmniConfig,
)
from vllm_omni.model_executor.models.minimind_o.resource_utils import resolve_model_dir

CapturedHiddenStates = dict[str, dict[str, dict[int, torch.Tensor]]]


class MiniMindOmniAttention(nn.Module):
    def __init__(
        self,
        config: MiniMindConfig,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_key_value_heads,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "qkv_proj"),
        )
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.head_dim**-0.5,
            num_kv_heads=self.num_key_value_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "attn"),
        )
        self.o_proj = RowParallelLinear(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "o_proj"),
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            is_neox_style=True,
            rope_parameters={
                "rope_theta": config.rope_theta,
                "rope_type": "default",
            },
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        num_tokens = q.shape[0]
        q = self.q_norm(q.view(num_tokens, self.num_heads, self.head_dim)).view(num_tokens, self.q_size)
        k = self.k_norm(k.view(num_tokens, self.num_key_value_heads, self.head_dim)).view(num_tokens, self.kv_size)

        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, quant_config: QuantizationConfig | None = None, prefix: str = ""):
        super().__init__()
        intermediate_size = config.intermediate_size

        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[intermediate_size, intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "gate_up_proj"),
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "down_proj"),
        )

        if config.hidden_act != "silu":
            raise NotImplementedError(f"MiniMind vLLM MLP only supports silu, got {config.hidden_act}")
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, quant_config: QuantizationConfig | None = None, prefix: str = ""):
        super().__init__()
        self.config = config

        self.gate = ReplicatedLinear(
            input_size=config.hidden_size,
            output_size=config.num_experts,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "gate"),
        )

        self.experts = FusedMoE(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "experts"),
        )

        self.experts.expert_mapping = FusedMoE.make_expert_params_mapping(
            self.experts,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=config.num_experts,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        hidden_dim = x.shape[-1]
        x_flat = x.view(-1, hidden_dim)

        router_logits, _ = self.gate(x_flat)
        y = self.experts(x_flat, router_logits)
        return y.view(*original_shape)


class MiniMindBlock(nn.Module):
    def __init__(
        self,
        config: MiniMindConfig,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.self_attn = MiniMindOmniAttention(
            config=config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, f"layers.{layer_idx}.self_attn"),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = (
            FeedForward(config, quant_config=quant_config, prefix=maybe_prefix(prefix, f"layers.{layer_idx}.mlp"))
            if not config.use_moe
            else MOEFeedForward(
                config, quant_config=quant_config, prefix=maybe_prefix(prefix, f"layers.{layer_idx}.mlp")
            )
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(positions, self.input_layernorm(hidden_states))
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class MiniMindModel(nn.Module):
    def __init__(
        self,
        config: MiniMindConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [
                MiniMindBlock(
                    config,
                    layer_idx=l_idx,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=prefix,
                )
                for l_idx in range(self.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        capture_layer_indices: Sequence[int] | None = None,
        return_hidden_states: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, CapturedHiddenStates | None]:
        # TODO: support PP once MiniMind has a vLLM paged-attention path.
        if intermediate_tensors is not None:
            raise NotImplementedError("MiniMindModel does not support intermediate tensors.")

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            if input_ids is None:
                raise ValueError("input_ids must be provided when inputs_embeds is None.")
            hidden_states = self.embed_tokens(input_ids)

        capture_set = set(capture_layer_indices) if capture_layer_indices else None
        captured_hidden_states: CapturedHiddenStates | None = {} if return_hidden_states else None

        for layer_idx, layer in enumerate(self.layers):
            if capture_set is not None and captured_hidden_states is not None:
                if layer_idx in capture_set:
                    hs: dict[str, dict[int, torch.Tensor]] = captured_hidden_states.setdefault("hidden_states", {})
                    layers: dict[int, torch.Tensor] = hs.setdefault("layers", {})
                    layers[layer_idx] = hidden_states.clone().view(-1, hidden_states.shape[-1])

            hidden_states = layer(positions, hidden_states)
            if captured_hidden_states is not None and layer_idx == getattr(self.config, "bridge_layer", -1):
                hs = captured_hidden_states.setdefault("hidden_states", {})
                hs["bridge"] = hidden_states.clone().view(-1, hidden_states.shape[-1])

        hidden_states = self.norm(hidden_states)
        return hidden_states, captured_hidden_states


class MiniMindForCausalLM(PreTrainedModel):
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(
        self,
        config: MiniMindConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(
            self.config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.post_init()

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)


_PROJECTOR_ACTIVATION_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _is_projector_activation_dtype(dtype: object) -> bool:
    return isinstance(dtype, torch.dtype) and dtype in _PROJECTOR_ACTIVATION_DTYPES


def _select_projector_activation_dtype(
    supported_dtypes: Sequence[torch.dtype],
    fallback: torch.dtype,
) -> torch.dtype:
    supported = [dtype for dtype in supported_dtypes if _is_projector_activation_dtype(dtype)]
    if not supported:
        return fallback
    if fallback in supported:
        return fallback
    for dtype in _PROJECTOR_ACTIVATION_DTYPES:
        if dtype in supported:
            return dtype
    return supported[0]


def _linear_activation_dtype(linear: nn.Module, fallback: torch.dtype) -> torch.dtype:
    quant_method = getattr(linear, "quant_method", None)
    for attr in ("input_dtype", "act_dtype"):
        dtype = getattr(quant_method, attr, None)
        if _is_projector_activation_dtype(dtype):
            return dtype

    weight = getattr(linear, "weight", None)
    if isinstance(weight, torch.Tensor) and _is_projector_activation_dtype(weight.dtype):
        return weight.dtype

    quant_config = getattr(linear, "quant_config", None)
    if quant_config is not None:
        try:
            return _select_projector_activation_dtype(
                quant_config.get_supported_act_dtypes(),
                fallback,
            )
        except Exception:
            pass

    params_dtype = getattr(linear, "params_dtype", None)
    if _is_projector_activation_dtype(params_dtype):
        return params_dtype

    return fallback


class _MiniMindProjectorMixin:
    _activation_dtype_fallback: torch.dtype

    def set_activation_dtype_fallback(self, dtype: torch.dtype) -> None:
        if _is_projector_activation_dtype(dtype):
            self._activation_dtype_fallback = dtype

    @property
    def input_dtype(self) -> torch.dtype:
        return _linear_activation_dtype(self.mlp[1], self._activation_dtype_fallback)

    def _linear_dtype(self, linear_idx: int) -> torch.dtype:
        return _linear_activation_dtype(self.mlp[linear_idx], self._activation_dtype_fallback)

    def _cast_for_linear(self, x: torch.Tensor, linear_idx: int) -> torch.Tensor:
        target_dtype = self._linear_dtype(linear_idx)
        if x.dtype == target_dtype:
            return x
        return x.to(dtype=target_dtype)


class MMAudioProjector(_MiniMindProjectorMixin, nn.Module):
    def __init__(self, config: MiniMindOmniConfig, quant_config: QuantizationConfig | None = None, prefix: str = ""):
        super().__init__()
        self._activation_dtype_fallback = torch.get_default_dtype()
        self.mlp = nn.ModuleList(
            [
                LayerNorm(config.audio_hidden_size),
                ReplicatedLinear(
                    config.audio_hidden_size,
                    config.hidden_size,
                    bias=True,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "mlp.1"),
                ),
                nn.GELU(),
                ReplicatedLinear(
                    config.hidden_size,
                    config.hidden_size,
                    bias=True,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "mlp.3"),
                ),
            ]
        )

    def forward(self, x):
        x = self.mlp[0](x)
        x, _ = self.mlp[1](self._cast_for_linear(x, 1))
        x = self.mlp[2](x)
        x, _ = self.mlp[3](self._cast_for_linear(x, 3))
        return x


class MMVisionProjector(_MiniMindProjectorMixin, nn.Module):
    def __init__(self, config: MiniMindOmniConfig, quant_config: QuantizationConfig | None = None, prefix: str = ""):
        super().__init__()
        self._activation_dtype_fallback = torch.get_default_dtype()
        self.mlp = nn.ModuleList(
            [
                LayerNorm(config.image_hidden_size),
                ReplicatedLinear(
                    config.image_hidden_size,
                    config.hidden_size,
                    bias=True,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "mlp.1"),
                ),
                nn.GELU(),
                ReplicatedLinear(
                    config.hidden_size,
                    config.hidden_size,
                    bias=True,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "mlp.3"),
                ),
            ]
        )

    def forward(self, x):
        x = self.mlp[0](x)
        x, _ = self.mlp[1](self._cast_for_linear(x, 1))
        x = self.mlp[2](x)
        x, _ = self.mlp[3](self._cast_for_linear(x, 3))
        return x


def _load_sensevoice_model(path: str, *, device: str = "cpu"):
    if not path:
        raise FileNotFoundError("SenseVoice path or repo id is empty.")
    path = resolve_model_dir(path, "SenseVoice encoder")

    try:
        from funasr import AutoModel

        return AutoModel(
            model=path,
            trust_remote_code=True,
            disable_update=True,
            device=device,
        )
    except ImportError as exc:
        raise ImportError(f"funasr is required to load SenseVoice model: {exc}") from exc


class MiniMindOmniProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> MiniMindOmniConfig:
        return self.ctx.get_hf_config(MiniMindOmniConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "audio": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        del seq_len
        config = self.get_hf_config()
        vision_config = config.vision_config
        audio_config = config.audio_config
        limits: dict[str, int] = {}
        if mm_counts.get("image", 0) > 0:
            limits["image"] = int(vision_config.image_token_len)
        if mm_counts.get("audio", 0) > 0:
            limits["audio"] = int(audio_config.max_audio_tokens)
        return limits

    def get_data_parser(self):
        audio_config = self.get_hf_config().audio_config
        return MultiModalDataParser(
            target_sr=audio_config.audio_sample_rate,
            target_channels=audio_config.audio_target_channels,
            expected_hidden_size=self._get_expected_hidden_size(),
        )


class MiniMindOmniDummyInputsBuilder(BaseDummyInputsBuilder[MiniMindOmniProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        config = self.info.get_hf_config()
        vision_config = config.vision_config
        audio_config = config.audio_config
        num_images = mm_counts.get("image", 0)
        num_audios = mm_counts.get("audio", 0)
        return (vision_config.image_special_token * num_images) + (audio_config.audio_special_token * num_audios)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        del seq_len
        num_images = mm_counts.get("image", 0)
        num_audios = mm_counts.get("audio", 0)
        mm_data: MultiModalDataDict = {}
        if num_images:
            image_overrides = mm_options.get("image") if mm_options else None
            mm_data["image"] = self._get_dummy_images(
                width=256,
                height=256,
                num_images=num_images,
                overrides=image_overrides,
            )
        if num_audios:
            audio_config = self.info.get_hf_config().audio_config
            audio_overrides = mm_options.get("audio") if mm_options else None
            mm_data["audio"] = self._get_dummy_audios(
                length=int(audio_config.audio_sample_rate),
                num_audios=num_audios,
                overrides=audio_overrides,
            )
        return mm_data


class MiniMindOmniMultiModalProcessor(BaseMultiModalProcessor[MiniMindOmniProcessingInfo]):
    def _get_sensevoice_frontend(self):
        if not hasattr(self, "_sensevoice_frontend"):
            audio_config = self.info.get_hf_config().audio_config
            model = _load_sensevoice_model(audio_config.audio_encoder_path, device="cpu")
            self._sensevoice_frontend = model.kwargs["frontend"].eval()
        return self._sensevoice_frontend

    def _get_image_processor(self):
        if not hasattr(self, "_image_processor"):
            vision_config = self.info.get_hf_config().vision_config
            vision_dir = resolve_model_dir(vision_config.vision_model_path, "vision encoder")
            self._image_processor = SiglipImageProcessor.from_pretrained(
                vision_dir,
                local_files_only=True,
            )
        return self._image_processor

    @staticmethod
    def _normalise_audio_item(audio_item: object) -> torch.Tensor:
        if isinstance(audio_item, tuple):
            audio_item = audio_item[0]
        wav = torch.as_tensor(audio_item, dtype=torch.float32)
        if wav.dim() > 1:
            wav = wav.mean(dim=0)
        return wav

    def _process_audios(self, audios: object) -> dict[str, torch.Tensor]:
        if audios is None:
            return {}
        if isinstance(audios, (np.ndarray, torch.Tensor)) or (isinstance(audios, tuple) and len(audios) == 2):
            audio_items = [audios]
        else:
            audio_items = list(audios)
        if not audio_items:
            return {}

        frontend = self._get_sensevoice_frontend()
        fbanks: list[torch.Tensor] = []
        lens: list[int] = []
        for item in audio_items:
            wav = self._normalise_audio_item(item)
            fbank, flen = frontend(wav.unsqueeze(0), torch.tensor([wav.numel()]))
            fbanks.append(fbank.squeeze(0))
            lens.append(int(flen[0].item()))

        max_len = max(fbank.shape[0] for fbank in fbanks)
        feat_size = fbanks[0].shape[-1]
        audio_inputs = fbanks[0].new_zeros((len(fbanks), max_len, feat_size))
        for i, fbank in enumerate(fbanks):
            audio_inputs[i, : fbank.shape[0]] = fbank
        return {
            "audio_inputs": audio_inputs,
            "audio_lens": torch.tensor(lens, dtype=torch.long),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs,
    ) -> Sequence[PromptUpdate]:
        del hf_processor_mm_kwargs
        config = self.info.get_hf_config()
        vision_config = config.vision_config
        audio_config = config.audio_config
        out_mm_data = out_mm_kwargs.get_data()
        audio_token_id = int(audio_config.audio_ids[0])
        image_token_id = int(vision_config.image_ids[0])

        def get_replacement_audio(item_idx: int) -> PromptUpdateDetails:
            audio_lens = out_mm_data.get("audio_lens")
            num_tokens = 1
            if audio_lens is not None:
                num_tokens = max(1, int(audio_lens[item_idx].item()))
            return PromptUpdateDetails.select_token_id(
                [audio_token_id] * num_tokens,
                embed_token_id=audio_token_id,
            )

        def get_replacement_image(item_idx: int) -> PromptUpdateDetails:
            del item_idx
            num_tokens = int(getattr(vision_config, "image_token_len", 64))
            return PromptUpdateDetails.select_token_id(
                [image_token_id] * num_tokens,
                embed_token_id=image_token_id,
            )

        updates: list[PromptUpdate] = []
        if "image" in mm_items and mm_items.get_items("image", ImageProcessorItems):
            updates.append(
                PromptReplacement(
                    modality="image",
                    target=vision_config.image_special_token,
                    replacement=get_replacement_image,
                )
            )
        if "audio" in mm_items and mm_items.get_items("audio", AudioProcessorItems):
            updates.append(
                PromptReplacement(
                    modality="audio",
                    target=audio_config.audio_special_token,
                    replacement=get_replacement_audio,
                )
            )
        return updates

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        del hf_processor_mm_kwargs
        config: dict[str, MultiModalFieldConfig] = {}
        if "pixel_values" in hf_inputs:
            config["pixel_values"] = MultiModalFieldConfig.batched("image")
        if "audio_inputs" in hf_inputs:
            config["audio_inputs"] = MultiModalFieldConfig.batched("audio")
        if "audio_lens" in hf_inputs:
            config["audio_lens"] = MultiModalFieldConfig.batched("audio")
        return config

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        del mm_kwargs
        data: dict[str, object] = {}

        images = mm_data.get("images", mm_data.get("image"))
        if images is not None:
            data.update(self._get_image_processor()(images=images, return_tensors="pt"))

        audios = mm_data.get("audios", mm_data.get("audio"))
        data.update(self._process_audios(audios))

        tokenizer = self.info.get_tokenizer()
        data.update(tokenizer(prompt, return_tensors="pt", **tok_kwargs))
        return BatchFeature(data=data, tensor_type=None)


@MULTIMODAL_REGISTRY.register_processor(
    MiniMindOmniMultiModalProcessor,
    info=MiniMindOmniProcessingInfo,
    dummy_inputs=MiniMindOmniDummyInputsBuilder,
)
class MiniMindOmniThinkerForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "thinker.lm_head.": "language_model.lm_head.",
            "thinker.model.": "language_model.model.",
            "thinker.": "",
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        config: MiniMindOmniConfig = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.quant_config = quant_config
        self.multimodal_config = multimodal_config
        self.device = get_local_device()

        self.text_config = config.text_config
        self.text_config.bridge_layer = config.bridge_layer
        with self._mark_language_model(vllm_config):
            self.language_model = MiniMindForCausalLM(
                self.text_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )
        self.language_model_dtype = self._module_dtype(self.language_model)

        self.vision_config = config.vision_config
        with self._mark_tower_model(vllm_config, "image"):
            if multimodal_config.get_limit_per_prompt("image"):
                self.vision_proj = MMVisionProjector(
                    self.vision_config,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "vision_proj"),
                )
                vision_dir = (
                    resolve_model_dir(self.vision_config.vision_model_path, "vision encoder")
                    if self.vision_config.vision_model_path
                    else None
                )
                self.vision_encoder = (
                    SiglipVisionModel.from_pretrained(
                        vision_dir,
                        local_files_only=True,
                    ).to(self.device)
                    if vision_dir
                    else None
                )
                self.vision_proj.set_activation_dtype_fallback(self.language_model_dtype)
                self.vision_encoder_dtype = self._module_input_dtype(self.vision_encoder)
                self.vision_proj_dtype = self.vision_proj.input_dtype
            else:
                self.vision_encoder = None
                self.vision_proj = None
                self.vision_encoder_dtype = torch.float32
                self.vision_proj_dtype = torch.float32

        self.audio_config = config.audio_config
        with self._mark_tower_model(vllm_config, "audio"):
            if multimodal_config.get_limit_per_prompt("audio"):
                self.audio_proj = MMAudioProjector(
                    self.audio_config,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "audio_proj"),
                )
                self.audio_encoder = (
                    self._init_audio_encoder(self.audio_config.audio_encoder_path)
                    if self.audio_config.audio_encoder_path
                    else None
                )
                self.audio_proj.set_activation_dtype_fallback(self.language_model_dtype)
                self.audio_encoder_dtype = self._module_input_dtype(self.audio_encoder)
                self.audio_proj_dtype = self.audio_proj.input_dtype
            else:
                self.audio_proj = None
                self.audio_encoder = None
                self.audio_encoder_dtype = torch.float32
                self.audio_proj_dtype = torch.float32

    def _init_audio_encoder(self, path: str | None) -> nn.Module | None:
        if not path:
            warnings.warn("[MiniMindOmni] SenseVoice path or repo id is empty.")
            return None
        try:
            model = _load_sensevoice_model(path, device=str(self.device))
        except (ImportError, RuntimeError, FileNotFoundError) as exc:
            warnings.warn(f"[MiniMindOmni] {exc}")
            return None
        return model.model.encoder.to(device=self.device, dtype=torch.float32)

    def encode_audio_inputs(
        self,
        audio_inputs: torch.Tensor | None,
        audio_lens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if audio_inputs is None:
            return ()
        if self.audio_encoder is None:
            raise RuntimeError("audio_inputs were provided but audio_encoder is not initialized.")
        if audio_inputs.numel() == 0 or audio_inputs.size(0) == 0:
            return ()

        valid_fbank = audio_inputs.to(device=self.device, dtype=self.audio_encoder_dtype)
        if audio_lens is not None:
            valid_lens = audio_lens.reshape(-1).to(device=valid_fbank.device, dtype=torch.long)
        else:
            valid_lens = torch.full(
                (valid_fbank.size(0),),
                valid_fbank.size(1),
                device=valid_fbank.device,
                dtype=torch.long,
            )

        with torch.autocast(device_type=valid_fbank.device.type, enabled=False):
            audio_hidden, _ = self.audio_encoder(valid_fbank, valid_lens)

        audio_embeddings = []
        for idx in range(audio_hidden.size(0)):
            feature_len = max(1, min(int(valid_lens[idx].item()), audio_hidden.size(1)))
            projected = self.audio_proj(audio_hidden[idx, :feature_len].unsqueeze(0).to(device=self.device)).squeeze(0)
            audio_embeddings.append(projected)
        return tuple(audio_embeddings)

    def get_language_model(self) -> nn.Module:
        return self.language_model

    @staticmethod
    def _as_embedding_tuple(embeddings: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if embeddings.dim() == 2:
            return (embeddings,)
        return tuple(embeddings.unbind(0))

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        pixel_values = kwargs.get("pixel_values")
        image_embeds = kwargs.get("image_embeds")
        audio_features = kwargs.get("audio_features")
        audio_inputs = kwargs.get("audio_inputs")
        audio_lens = kwargs.get("audio_lens")
        audio_embeds = kwargs.get("audio_embeds")
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        if image_embeds is not None:
            multimodal_embeddings += self._as_embedding_tuple(image_embeds)  # type: ignore[arg-type]
        elif pixel_values is not None:
            if self.vision_encoder is None:
                raise RuntimeError("pixel_values were provided but vision_encoder is not initialized.")
            pixel_values = pixel_values.to(device=self.device, dtype=self.vision_encoder_dtype)  # type: ignore[union-attr]
            image_hidden = self.vision_encoder(pixel_values=pixel_values).last_hidden_state
            multimodal_embeddings += self._as_embedding_tuple(self.vision_proj(image_hidden.to(device=self.device)))

        if audio_embeds is not None:
            multimodal_embeddings += self._as_embedding_tuple(audio_embeds)  # type: ignore[arg-type]
        elif audio_features is not None:
            audio_features = audio_features.to(device=self.device)  # type: ignore[union-attr]
            multimodal_embeddings += self._as_embedding_tuple(self.audio_proj(audio_features))
        elif audio_inputs is not None:
            multimodal_embeddings += self.encode_audio_inputs(
                audio_inputs,  # type: ignore[arg-type]
                audio_lens if isinstance(audio_lens, torch.Tensor) else None,
            )

        return multimodal_embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        if not multimodal_embeddings:
            return inputs_embeds

        multimodal_embeddings = tuple(
            embedding.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype) for embedding in multimodal_embeddings
        )

        if is_multimodal is not None:
            return _merge_multimodal_embeddings(
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                is_multimodal=is_multimodal,
            )

        image_embeds = multimodal_embeddings[0]
        image_token_id = self.vision_config.image_ids[0]
        image_mask = input_ids == image_token_id

        flat_embeds = inputs_embeds.view(-1, inputs_embeds.shape[-1])
        flat_mask = image_mask.view(-1)

        flat_image_embeds = image_embeds.reshape(-1, image_embeds.shape[-1])
        if flat_mask.sum().item() != flat_image_embeds.shape[0]:
            raise ValueError(
                f"Image token count {flat_mask.sum().item()} != image embedding count {flat_image_embeds.shape[0]}"
            )

        flat_embeds[flat_mask] = flat_image_embeds.to(flat_embeds.dtype)
        return inputs_embeds

    def compute_logits(self, hidden_states: torch.Tensor, **kwargs):
        return self.language_model.compute_logits(hidden_states)

    @staticmethod
    def _module_dtype(module: nn.Module | None, default: torch.dtype = torch.float32) -> torch.dtype:
        if module is None:
            return default
        for p in module.parameters(recurse=True):
            if p.is_floating_point():
                return p.dtype
        for b in module.buffers(recurse=True):
            if b.is_floating_point():
                return b.dtype
        return default

    @staticmethod
    def _module_input_dtype(module: nn.Module | None, default: torch.dtype = torch.float32) -> torch.dtype:
        if module is None:
            return default
        input_module_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)
        for child in module.modules():
            if isinstance(child, input_module_types):
                weight = getattr(child, "weight", None)
                if isinstance(weight, torch.Tensor) and weight.is_floating_point():
                    return weight.dtype
        return MiniMindOmniThinkerForConditionalGeneration._module_dtype(module, default)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_weights: set[str] = set()
        use_moe = self.text_config.use_moe

        self.expert_params_mapping = []
        if use_moe:
            for layer in self.language_model.model.layers:
                mlp = getattr(layer, "mlp", None)
                experts = getattr(mlp, "experts", None)
                if experts is not None and hasattr(experts, "expert_mapping"):
                    self.expert_params_mapping = experts.expert_mapping
                    break
            if not self.expert_params_mapping:
                raise RuntimeError("MiniMind MoE is enabled but no FusedMoE expert_mapping was found.")

        for name, loaded_weight in self.hf_to_vllm_mapper.apply(weights):
            if name.startswith("talker."):
                continue
            if "rotary_emb.inv_freq" in name:
                continue

            # MoE expert weights:
            # layers.x.mlp.experts.i.gate_proj.weight
            # layers.x.mlp.experts.i.up_proj.weight
            # layers.x.mlp.experts.i.down_proj.weight
            if use_moe and ".mlp.experts." in name:
                for param_name, weight_name, expert_id, shard_id in self.expert_params_mapping:
                    if weight_name not in name:
                        continue

                    mapped_name = name.replace(weight_name, param_name)
                    param = params_dict.get(mapped_name)
                    if param is None:
                        break

                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        mapped_name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    loaded_weights.add(mapped_name)
                    break

                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                mapped_name = name.replace(weight_name, param_name)
                if mapped_name not in params_dict:
                    break

                param = params_dict[mapped_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_weights.add(mapped_name)
                break
            else:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_weights.add(name)

        # Vision/audio encoders are initialized from their own external
        # checkpoints during __init__, not from the MiniMind thinker checkpoint.
        # Mark them as loaded so vLLM's post-load validation does not report
        # already-initialized tower parameters as missing.
        for external_prefix in ("vision_encoder.", "audio_encoder."):
            loaded_weights.update(name for name in params_dict if name.startswith(external_prefix))

        return loaded_weights

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        capture_layer_indices: Sequence[int] | None = None,
        return_hidden_states: bool = False,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states, captured_hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            capture_layer_indices=capture_layer_indices,
            return_hidden_states=return_hidden_states,
            **kwargs,
        )
        return hidden_states, captured_hidden_states
