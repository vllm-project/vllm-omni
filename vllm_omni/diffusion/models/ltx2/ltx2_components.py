# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared component construction helpers for the LTX model family."""

from __future__ import annotations

import inspect
import json
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from diffusers import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx2 import LTX2TextConnectors
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
from diffusers.video_processor import VideoProcessor
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention as OmniAttention
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_ltx2 import DistributedAutoencoderKLLTX2Video
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.offloader.module_collector import ModuleDiscovery

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from .ltx2_transformer import (
    LTX2VideoTransformer3DModel,
    apply_interleaved_rotary_emb,
    apply_split_rotary_emb,
    to_ltx_padding_mask,
)

try:
    from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
except ImportError:
    LTX2VocoderWithBWE = None


_LTX_COMPONENT_SUBFOLDERS = (
    "tokenizer",
    "text_encoder",
    "connectors",
    "vae",
    "audio_vae",
    "vocoder",
    "scheduler",
    "latent_upsampler",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LTXComponentProfile:
    """Component construction and discovery contract for an LTX variant."""

    name: str
    dit_modules: tuple[str, ...]
    encoder_modules: tuple[str, ...]
    vae_modules: tuple[str, ...]
    resident_modules: tuple[str, ...] = ()
    video_vae_cls: type = AutoencoderKLLTX2Video
    vocoder_cls: type = LTX2Vocoder
    vocoder_fallback_cls: type | None = None


LTX2_COMPONENT_PROFILE = LTXComponentProfile(
    name="ltx2",
    dit_modules=("transformer",),
    encoder_modules=("text_encoder", "connectors"),
    vae_modules=("vae", "audio_vae"),
    resident_modules=("vocoder",),
    video_vae_cls=DistributedAutoencoderKLLTX2Video,
)

LTX23_COMPONENT_PROFILE = LTXComponentProfile(
    name="ltx2_3",
    dit_modules=("transformer",),
    encoder_modules=("text_encoder", "connectors"),
    vae_modules=("vae", "audio_vae"),
    resident_modules=("vocoder",),
    video_vae_cls=DistributedAutoencoderKLLTX2Video,
    vocoder_cls=LTX2VocoderWithBWE or LTX2Vocoder,
    vocoder_fallback_cls=LTX2Vocoder,
)

LTX2_DISTILLED_COMPONENT_PROFILE = LTXComponentProfile(
    name="ltx2_distilled",
    dit_modules=("transformer",),
    encoder_modules=("text_encoder", "connectors"),
    vae_modules=("vae", "audio_vae"),
    resident_modules=("vocoder", "latent_upsampler"),
    video_vae_cls=DistributedAutoencoderKLLTX2Video,
)


_COMPONENT_PROFILES: dict[tuple[str, str], LTXComponentProfile] = {
    ("one_stage", "2"): LTX2_COMPONENT_PROFILE,
    ("one_stage", "2.3"): LTX23_COMPONENT_PROFILE,
    ("distilled_two_stage", "2"): LTX2_DISTILLED_COMPONENT_PROFILE,
    ("dmd2", "2"): LTX2_COMPONENT_PROFILE,
    ("dmd2", "2.3"): LTX23_COMPONENT_PROFILE,
}


def resolve_ltx_component_profile(pipeline_kind: str, model_version: str) -> LTXComponentProfile:
    """Resolve component construction independently from execution recipes."""
    try:
        return _COMPONENT_PROFILES[(pipeline_kind, model_version)]
    except KeyError as exc:
        raise ValueError(f"Unsupported LTX component kind/version: {pipeline_kind!r}/{model_version!r}.") from exc


def _load_ltx_metadata_json(model: str, filename: str) -> dict[str, Any]:
    """Load small checkpoint metadata without relying on repository names."""
    if os.path.isdir(model):
        path = os.path.join(model, filename)
        if not os.path.isfile(path):
            return {}
    else:
        try:
            path = hf_hub_download(repo_id=model, filename=filename)
        except Exception:
            return {}
    try:
        with open(path) as config_file:
            value = json.load(config_file)
    except (OSError, TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def detect_ltx_model_version(model: str) -> str:
    """Detect LTX-2 versus LTX-2.3 from checkpoint component metadata.

    Official checkpoints use ``model_version`` metadata. Diffusers repositories
    expose the equivalent distinction through the BWE vocoder introduced by
    LTX-2.3. Unknown conversions retain the official LTX-2 fallback.
    """
    model_index = _load_ltx_metadata_json(model, "model_index.json")
    if str(model_index.get("model_version", "")).startswith("2.3"):
        return "2.3"

    vocoder_entry = model_index.get("vocoder")
    if isinstance(vocoder_entry, (list, tuple)) and vocoder_entry:
        vocoder_class = str(vocoder_entry[-1])
    elif isinstance(vocoder_entry, dict):
        vocoder_class = str(vocoder_entry.get("_class_name", ""))
    else:
        vocoder_class = ""
    if vocoder_class == "LTX2VocoderWithBWE":
        return "2.3"

    vocoder_config = _load_ltx_metadata_json(model, "vocoder/config.json")
    if str(vocoder_config.get("model_version", "")).startswith("2.3"):
        return "2.3"
    if vocoder_config.get("_class_name") == "LTX2VocoderWithBWE":
        return "2.3"

    logger.info("Using LTX-2 defaults for checkpoint %s", model)
    return "2"


class _LTXConnectorAttnProcessor:
    """Preserve official connector math around Omni attention dispatch."""

    def __init__(self, *, has_learned_registers: bool) -> None:
        self.has_learned_registers = has_learned_registers

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        query_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        gate_logits = attn.to_gate_logits(hidden_states) if attn.to_gate_logits is not None else None

        query = attn.norm_q(attn.to_q(hidden_states))
        key = attn.norm_k(attn.to_k(encoder_hidden_states))
        value = attn.to_v(encoder_hidden_states)

        if query_rotary_emb is not None:
            key_rotary_emb = key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
            if attn.rope_type == "interleaved":
                query = apply_interleaved_rotary_emb(query, query_rotary_emb)
                key = apply_interleaved_rotary_emb(key, key_rotary_emb)
            elif attn.rope_type == "split":
                query = apply_split_rotary_emb(query, query_rotary_emb, head_dim=attn.head_dim)
                key = apply_split_rotary_emb(key, key_rotary_emb, head_dim=attn.head_dim)
            else:
                raise ValueError(f"Unsupported LTX connector RoPE type: {attn.rope_type}")

        batch_size, _, inner_dim = query.shape
        head_dim = inner_dim // attn.heads
        kv_heads = attn.inner_kv_dim // attn.head_dim
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        # The connector replaces padding tokens with learned registers before
        # entering its blocks, so every key is valid and the old padding mask
        # becomes an all-keep no-op.
        if self.has_learned_registers:
            attention_mask = None
        elif attention_mask is not None and attn.omni_attention.attn_backend.get_name().upper() == "FLASH_ATTN":
            attention_mask = to_ltx_padding_mask(attention_mask)
        attn_metadata = AttentionMetadata(attn_mask=attention_mask) if attention_mask is not None else None
        hidden_states = attn.omni_attention(query, key, value, attn_metadata)
        hidden_states = hidden_states.reshape(batch_size, -1, inner_dim)

        if gate_logits is not None:
            hidden_states = hidden_states.unflatten(2, (attn.heads, -1))
            hidden_states = hidden_states * (2.0 * torch.sigmoid(gate_logits)).unsqueeze(-1)
            hidden_states = hidden_states.flatten(2, 3)

        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


def _install_connector_attention(connectors: LTX2TextConnectors) -> None:
    for connector_name in ("video_connector", "audio_connector"):
        connector = getattr(connectors, connector_name, None)
        has_learned_registers = getattr(connector, "learnable_registers", None) is not None
        for block_index, block in enumerate(getattr(connector, "transformer_blocks", ())):
            attention = getattr(block, "attn1", None)
            if attention is not None:
                attention.omni_attention = OmniAttention(
                    num_heads=attention.heads,
                    head_size=attention.head_dim,
                    num_kv_heads=attention.inner_kv_dim // attention.head_dim,
                    softmax_scale=1.0 / (attention.head_dim**0.5),
                    causal=False,
                    prefix=f"connectors.{connector_name}.transformer_blocks.{block_index}.attn1",
                    role="ltx2.connector",
                    role_category="self",
                    skip_sequence_parallel=True,
                    disable_kv_quant=True,
                )
                attention.set_processor(_LTXConnectorAttnProcessor(has_learned_registers=has_learned_registers))


def _detect_vocoder_output_sample_rate(model: str) -> int | None:
    """Read the generated waveform sample rate from the vocoder config."""
    vocoder_config_path = os.path.join(model, "vocoder", "config.json")
    if not os.path.exists(vocoder_config_path):
        try:
            vocoder_config_path = hf_hub_download(model, "vocoder/config.json")
        except Exception:
            return None
    try:
        with open(vocoder_config_path) as config_file:
            return json.load(config_file).get("output_sampling_rate")
    except Exception:
        return None


def get_ltx2_post_process_func(od_config: Any):
    """Build the common LTX engine-output adapter."""
    output_sample_rate = _detect_vocoder_output_sample_rate(od_config.model)

    def post_process_func(output: tuple[torch.Tensor, torch.Tensor] | torch.Tensor):
        if not (isinstance(output, tuple) and len(output) == 2):
            return output
        video, audio = output
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu()
        result: dict[str, Any] = {"video": video, "audio": audio}
        if output_sample_rate is not None:
            result["audio_sample_rate"] = output_sample_rate
        return result

    return post_process_func


def _load_component(
    component_cls: type,
    model: str,
    subfolder: str,
    *,
    local_files_only: bool,
    dtype: torch.dtype,
) -> Any:
    return from_pretrained_with_prefetch(
        component_cls.from_pretrained,
        model,
        subfolder=subfolder,
        prefetch_list=_LTX_COMPONENT_SUBFOLDERS,
        local_files_only=local_files_only,
        torch_dtype=dtype,
    )


def _place_aux_components(pipeline: Any) -> None:
    parallel_config = getattr(pipeline.od_config, "parallel_config", None)
    use_managed_placement = bool(
        getattr(pipeline.od_config, "enable_cpu_offload", False)
        or getattr(pipeline.od_config, "enable_layerwise_offload", False)
        or getattr(parallel_config, "use_hsdp", False)
    )
    if use_managed_placement:
        return

    modules = ModuleDiscovery.discover(pipeline)
    for module in (*modules.encoders, *modules.vaes, *modules.resident_modules):
        module.to(pipeline.device)


def initialize_pipeline_components(pipeline: Any, od_config: Any) -> None:
    """Build the common LTX component graph selected by ``component_profile``."""
    profile: LTXComponentProfile = pipeline.component_profile
    pipeline.od_config = od_config
    pipeline.device = get_local_device()
    dtype = getattr(od_config, "dtype", torch.bfloat16)
    model = od_config.model
    local_files_only = os.path.exists(model)

    pipeline.weights_sources = [
        DiffusersPipelineLoader.ComponentSource(
            model_or_path=model,
            subfolder="transformer",
            revision=None,
            prefix="transformer.",
            fall_back_to_pt=True,
        ),
    ]
    prefetch_subfolders(model, _LTX_COMPONENT_SUBFOLDERS, local_files_only=local_files_only)

    pipeline.tokenizer = AutoTokenizer.from_pretrained(
        model,
        subfolder="tokenizer",
        local_files_only=local_files_only,
    )
    with torch.device("cpu"):
        pipeline.text_encoder = _load_component(
            Gemma3ForConditionalGeneration,
            model,
            "text_encoder",
            local_files_only=local_files_only,
            dtype=dtype,
        )
    pipeline.connectors = _load_component(
        LTX2TextConnectors,
        model,
        "connectors",
        local_files_only=local_files_only,
        dtype=dtype,
    )
    _install_connector_attention(pipeline.connectors)
    pipeline.vae = _load_component(
        profile.video_vae_cls,
        model,
        "vae",
        local_files_only=local_files_only,
        dtype=dtype,
    )
    pipeline.audio_vae = _load_component(
        AutoencoderKLLTX2Audio,
        model,
        "audio_vae",
        local_files_only=local_files_only,
        dtype=dtype,
    )
    try:
        pipeline.vocoder = _load_component(
            profile.vocoder_cls,
            model,
            "vocoder",
            local_files_only=local_files_only,
            dtype=dtype,
        )
    except (TypeError, OSError, ValueError):
        if profile.vocoder_fallback_cls is None or profile.vocoder_fallback_cls is profile.vocoder_cls:
            raise
        pipeline.vocoder = _load_component(
            profile.vocoder_fallback_cls,
            model,
            "vocoder",
            local_files_only=local_files_only,
            dtype=dtype,
        )

    if "latent_upsampler" in profile.resident_modules:
        # BlurDownsample constructs an integer kernel that must be initialized
        # on CPU; component placement is handled uniformly after construction.
        with torch.device("cpu"):
            pipeline.latent_upsampler = _load_component(
                LTX2LatentUpsamplerModel,
                model,
                "latent_upsampler",
                local_files_only=local_files_only,
                dtype=dtype,
            )

    transformer_config = load_transformer_config(model, "transformer", local_files_only)
    quant_config = getattr(od_config, "quantization_config", None)
    pipeline.transformer = create_transformer_from_config(transformer_config, quant_config=quant_config)
    _place_aux_components(pipeline)
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model,
        subfolder="scheduler",
        local_files_only=local_files_only,
    )

    pipeline.vae_spatial_compression_ratio = pipeline.vae.spatial_compression_ratio
    pipeline.vae_temporal_compression_ratio = pipeline.vae.temporal_compression_ratio
    pipeline.audio_vae_mel_compression_ratio = pipeline.audio_vae.mel_compression_ratio
    pipeline.audio_vae_temporal_compression_ratio = pipeline.audio_vae.temporal_compression_ratio
    pipeline.transformer_spatial_patch_size = pipeline.transformer.config.patch_size
    pipeline.transformer_temporal_patch_size = pipeline.transformer.config.patch_size_t
    pipeline.audio_sampling_rate = pipeline.audio_vae.config.sample_rate
    pipeline.audio_hop_length = pipeline.audio_vae.config.mel_hop_length
    pipeline.video_processor = VideoProcessor(vae_scale_factor=pipeline.vae_spatial_compression_ratio)

    tokenizer_max_length = pipeline.tokenizer.model_max_length
    if tokenizer_max_length is None or tokenizer_max_length > 100000:
        encoder_config = getattr(pipeline.text_encoder, "config", None)
        tokenizer_max_length = getattr(encoder_config, "max_position_embeddings", None)
        if tokenizer_max_length is None:
            tokenizer_max_length = getattr(encoder_config, "max_seq_len", None)
    pipeline.tokenizer_max_length = int(tokenizer_max_length or 1024)

    pipeline._interrupt = False


def load_transformer_config(
    model_path: str,
    subfolder: str = "transformer",
    local_files_only: bool = True,
) -> dict:
    """Load an LTX transformer config from a local model or the HF Hub."""
    if local_files_only:
        config_path = os.path.join(model_path, subfolder, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"LTX transformer config not found: {config_path}")
    else:
        config_path = hf_hub_download(
            repo_id=model_path,
            filename=f"{subfolder}/config.json",
        )
    with open(config_path) as config_file:
        return json.load(config_file)


def create_transformer_from_config(
    config: dict,
    quant_config: QuantizationConfig | None = None,
) -> LTX2VideoTransformer3DModel:
    """Construct the shared LTX transformer from a Diffusers config."""
    if not config and quant_config is None:
        return LTX2VideoTransformer3DModel()

    signature = inspect.signature(LTX2VideoTransformer3DModel.__init__)
    allowed_keys = set(signature.parameters)
    kwargs = {key: value for key, value in config.items() if key in allowed_keys}
    if quant_config is not None:
        kwargs["quant_config"] = quant_config

    return LTX2VideoTransformer3DModel(**kwargs)
