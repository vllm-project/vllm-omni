from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch._dynamo as dynamo
import torch.nn.functional as F
from einops import rearrange
from safetensors import safe_open
from torch import Tensor, nn
from transformers import Qwen3VLProcessor
from transformers.models.auto import CONFIG_MAPPING
from vllm.model_executor.models.utils import AutoWeightsLoader

from .adapter_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLTextModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from .config import (
    DEFAULT_COSMOS_DIR,
    DEFAULT_COSMOS_REPO,
    DEFAULT_QWEN3_VL_MODEL,
    OBS_IMAGES,
    OBS_PREFIX,
    OBS_STATE,
    OBS_TASK,
    OPENPI_ATTENTION_MASK_VALUE,
    InternVLAA1Config,
)
from .model_cosmos import ImageTokenizer


def get_safe_dtype(target_dtype: torch.dtype, device_type: str) -> torch.dtype:
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu" and target_dtype == torch.bfloat16:
        return torch.float32
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: torch.device,
) -> Tensor:
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector: torch.Tensor, new_dim: int) -> torch.Tensor:
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def resize_with_pad(images: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    target_h, target_w = size
    if images.ndim != 4:
        raise ValueError(f"Expected [T, C, H, W], got {tuple(images.shape)}")
    _, _, src_h, src_w = images.shape
    scale = min(target_h / src_h, target_w / src_w)
    resized_h = max(1, int(round(src_h * scale)))
    resized_w = max(1, int(round(src_w * scale)))
    resized = F.interpolate(images, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
    pad_h = target_h - resized_h
    pad_w = target_w - resized_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom))


class Qwen3VLInputBuilder:
    def __init__(
        self,
        processor_model_name: str = DEFAULT_QWEN3_VL_MODEL,
        max_length: int = 48,
        spatial_merge_size: int = 2,
    ) -> None:
        processor_model_name = os.getenv("INTERNVLA_A1_PROCESSOR_DIR", processor_model_name)
        local_path = Path(processor_model_name).expanduser()
        if local_path.exists():
            self.processor = Qwen3VLProcessor.from_pretrained(str(local_path), local_files_only=True)
        else:
            if os.getenv("INTERNVLA_A1_PROCESSOR_DIR"):
                raise FileNotFoundError(f"INTERNVLA_A1_PROCESSOR_DIR points to a missing path: {local_path}")
            self.processor = Qwen3VLProcessor.from_pretrained(processor_model_name)
        self.max_length = max_length
        self.spatial_merge_size = spatial_merge_size

    def build(self, camera_images: list[torch.Tensor], task: str) -> dict[str, torch.Tensor]:
        input_ids: list[int] = []
        attention_mask: list[int] = []
        pixel_values: list[torch.Tensor] = []
        image_grid_thw: list[torch.Tensor] = []

        for image_history in camera_images:
            current_image = image_history[1]
            image_inputs = self.processor.image_processor(current_image, do_rescale=False)
            pixel_values.append(image_inputs.pixel_values)
            image_grid_thw.append(image_inputs.image_grid_thw)
            num_img_token = int(torch.prod(image_inputs.image_grid_thw[0]).item()) // (self.spatial_merge_size**2)
            input_ids.extend(
                [self.processor.vision_start_token_id]
                + [self.processor.image_token_id] * num_img_token
                + [self.processor.vision_end_token_id]
            )
            attention_mask.extend([1] * (num_img_token + 2))

        text_inputs = self.processor.tokenizer(
            task,
            max_length=self.max_length,
            padding="max_length",
            padding_side="right",
            truncation=True,
        )
        input_ids.extend(text_inputs.input_ids)
        attention_mask.extend(text_inputs.attention_mask)

        return {
            f"{OBS_PREFIX}pixel_values": torch.cat(pixel_values, dim=0),
            f"{OBS_PREFIX}image_grid_thw": torch.cat(image_grid_thw, dim=0),
            f"{OBS_PREFIX}input_ids": torch.tensor(input_ids, dtype=torch.long),
            f"{OBS_PREFIX}attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


@dataclass
class SuffixStaticContext:
    state_emb: torch.Tensor
    full_att_2d_masks_4d: torch.Tensor
    position_ids: torch.Tensor


def compute_layer_complete(
    layer_idx: int,
    inputs_embeds: list[torch.Tensor],
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    und_expert: nn.Module,
    gen_expert: nn.Module,
    act_expert: nn.Module,
) -> list[torch.Tensor]:
    models = [und_expert.language_model, gen_expert, act_expert]
    query_states = []
    key_states = []
    value_states = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states = layer.input_layernorm(hidden_states)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        if layer.self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype=torch.bfloat16)
        query_state = layer.self_attn.q_norm(layer.self_attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_state = layer.self_attn.k_norm(layer.self_attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)

    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = und_expert.model.language_model.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = apply_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        unsqueeze_dim=1,
    )
    scaling = und_expert.language_model.layers[layer_idx].self_attn.scaling
    att_output, _ = eager_attention_forward(
        und_expert.language_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )

    head_dim = und_expert.language_model.layers[layer_idx].self_attn.head_dim
    num_attention_heads = und_expert.language_model.layers[layer_idx].self_attn.config.num_attention_heads
    batch_size = query_states.shape[0]
    att_output = att_output.reshape(batch_size, -1, num_attention_heads * head_dim)

    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        att_chunk = att_output[:, start_pos:end_pos]
        if att_chunk.dtype != layer.self_attn.o_proj.weight.dtype:
            att_chunk = att_chunk.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_chunk)
        out_emb = out_emb + hidden_states
        residual = out_emb
        out_emb = layer.post_attention_layernorm(out_emb)
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        out_emb = out_emb + residual
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


class QwenConfig:
    def __init__(
        self,
        head_dim: int,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        num_key_value_heads: int,
    ) -> None:
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads


def get_qwen_config(variant: str) -> QwenConfig:
    num_hidden_layers = int(variant.split("_")[-1][:-1])
    if variant.startswith("qwen3_vl"):
        return QwenConfig(128, 2048, 6144, 16, num_hidden_layers, 8)
    if variant.startswith("qwen3"):
        return QwenConfig(128, 1024, 3072, 16, num_hidden_layers, 8)
    raise ValueError(f"Unknown variant: {variant}")


def resolve_cosmos_checkpoint_paths(
    *,
    encoder_path: str | Path | None = None,
    decoder_path: str | Path | None = None,
) -> tuple[Path, Path]:
    encoder_override = encoder_path or os.getenv("INTERNVLA_A1_COSMOS_ENCODER_PATH")
    decoder_override = decoder_path or os.getenv("INTERNVLA_A1_COSMOS_DECODER_PATH")
    default_encoder_name = "encoder.safetensors"
    default_decoder_name = "decoder.safetensors"
    checkpoint_enc = (
        Path(encoder_override).expanduser()
        if encoder_override is not None
        else DEFAULT_COSMOS_DIR / default_encoder_name
    )
    checkpoint_dec = (
        Path(decoder_override).expanduser()
        if decoder_override is not None
        else DEFAULT_COSMOS_DIR / default_decoder_name
    )

    missing = [str(path) for path in (checkpoint_enc, checkpoint_dec) if not path.exists()]
    if missing:
        missing_display = ", ".join(missing)
        raise FileNotFoundError(
            "InternVLA-A1 requires local Cosmos tokenizer checkpoints. "
            f"Missing: {missing_display}. "
            f"Download {DEFAULT_COSMOS_REPO} and set INTERNVLA_A1_COSMOS_DIR to that directory, "
            "or point INTERNVLA_A1_COSMOS_ENCODER_PATH / INTERNVLA_A1_COSMOS_DECODER_PATH "
            "to explicit checkpoint files."
        )

    return checkpoint_enc, checkpoint_dec


class Qwen3VLWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config: QwenConfig,
        action_expert_config: QwenConfig,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ) -> None:
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["qwen3_vl"]()
        vlm_config_hf.text_config.hidden_size = vlm_config.hidden_size
        vlm_config_hf.text_config.intermediate_size = vlm_config.intermediate_size
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_attention_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.num_hidden_layers
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_key_value_heads
        vlm_config_hf.text_config.max_position_embeddings = 262144
        vlm_config_hf.text_config.rope_scaling = {
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
            "rope_type": "default",
        }
        vlm_config_hf.text_config.tie_word_embeddings = True
        vlm_config_hf.tie_word_embeddings = True
        vlm_config_hf.vision_config.deepstack_visual_indexes = [5, 11, 17]
        vlm_config_hf.vision_config.depth = 24
        vlm_config_hf.vision_config.hidden_size = 1024
        vlm_config_hf.vision_config.intermediate_size = 4096
        vlm_config_hf.vision_config.out_hidden_size = 2048

        self.und_expert = Qwen3VLForConditionalGeneration(config=vlm_config_hf)

        gen_expert_config_hf = CONFIG_MAPPING["qwen3_vl_text"]()
        gen_expert_config_hf.head_dim = action_expert_config.head_dim
        gen_expert_config_hf.hidden_size = action_expert_config.hidden_size
        gen_expert_config_hf.intermediate_size = action_expert_config.intermediate_size
        gen_expert_config_hf.num_attention_heads = action_expert_config.num_attention_heads
        gen_expert_config_hf.num_hidden_layers = action_expert_config.num_hidden_layers
        gen_expert_config_hf.num_key_value_heads = action_expert_config.num_key_value_heads
        gen_expert_config_hf.max_position_embeddings = self.und_expert.config.text_config.max_position_embeddings
        gen_expert_config_hf.rope_scaling = self.und_expert.config.text_config.rope_scaling
        self.gen_expert = Qwen3VLTextModel(config=gen_expert_config_hf)
        self.gen_expert.embed_tokens = None
        self.gen_expert.lm_head = None

        act_expert_config_hf = CONFIG_MAPPING["qwen3_vl_text"]()
        act_expert_config_hf.head_dim = action_expert_config.head_dim
        act_expert_config_hf.hidden_size = action_expert_config.hidden_size
        act_expert_config_hf.intermediate_size = action_expert_config.intermediate_size
        act_expert_config_hf.num_attention_heads = action_expert_config.num_attention_heads
        act_expert_config_hf.num_hidden_layers = action_expert_config.num_hidden_layers
        act_expert_config_hf.num_key_value_heads = action_expert_config.num_key_value_heads
        act_expert_config_hf.max_position_embeddings = self.und_expert.config.text_config.max_position_embeddings
        act_expert_config_hf.rope_scaling = self.und_expert.config.text_config.rope_scaling
        self.act_expert = Qwen3VLTextModel(config=act_expert_config_hf)
        self.act_expert.embed_tokens = None
        self.act_expert.lm_head = None

        self.to_selected_precision(precision)

    def to_selected_precision(self, precision: Literal["bfloat16", "float32"]) -> None:
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            # The fp32 path keeps the whole module in full precision, so no
            # selective post-conversion fixups are required.
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Unsupported precision: {precision}")

        # Match the original runtime numerics by keeping normalization layers
        # in fp32 even when the rest of the module runs in bf16.
        keep_fp32 = ["input_layernorm", "post_attention_layernorm", "model.norm"]
        for name, param in self.named_parameters():
            if any(key in name for key in keep_fp32):
                param.data = param.data.to(dtype=torch.float32)

    def forward(
        self,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None,
        past_key_values: Any,
        inputs_embeds: list[torch.Tensor | None],
        use_cache: bool,
    ) -> tuple[list[torch.Tensor | None], Any]:
        if inputs_embeds[1] is None and inputs_embeds[2] is None:
            prefix_output = self.und_expert.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            return [prefix_output.last_hidden_state, None, None], prefix_output.past_key_values

        if inputs_embeds[0] is None and inputs_embeds[2] is None:
            middle_output = self.gen_expert.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            return [None, middle_output.last_hidden_state, None], middle_output.past_key_values

        if inputs_embeds[0] is None and inputs_embeds[1] is None:
            suffix_output = self.act_expert.forward(
                inputs_embeds=inputs_embeds[2],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            return [None, None, suffix_output.last_hidden_state], None

        models = [self.und_expert.language_model, self.gen_expert, self.act_expert]
        stacked_inputs = [inputs_embeds[0], inputs_embeds[1], inputs_embeds[2]]
        for layer_idx in range(self.und_expert.config.text_config.num_hidden_layers):
            stacked_inputs = compute_layer_complete(
                layer_idx,
                stacked_inputs,
                attention_mask,
                position_ids,
                und_expert=self.und_expert,
                gen_expert=self.gen_expert,
                act_expert=self.act_expert,
            )
        outputs = [models[i].norm(stacked_inputs[i]) for i in range(3)]
        return outputs, None


class InternVLAA1(nn.Module):
    def __init__(
        self,
        config: InternVLAA1Config,
        *,
        cosmos_encoder_path: str | Path | None = None,
        cosmos_decoder_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.config = config

        vlm_config = get_qwen_config(config.qwen3_vl_variant)
        action_expert_config = get_qwen_config(config.action_expert_variant)
        self.qwen3_vl_with_expert = Qwen3VLWithExpertModel(vlm_config, action_expert_config, precision=config.dtype)

        cosmos_encoder_path, cosmos_decoder_path = resolve_cosmos_checkpoint_paths(
            encoder_path=cosmos_encoder_path,
            decoder_path=cosmos_decoder_path,
        )
        self.cosmos = ImageTokenizer(
            checkpoint_enc=str(cosmos_encoder_path),
            checkpoint_dec=str(cosmos_decoder_path),
            device=config.device,
        )

        hidden_size = action_expert_config.hidden_size
        vae_dim = 16
        ds = config.scale_factor
        self.cosmos_in_proj = nn.Conv2d(vae_dim, hidden_size, kernel_size=1, stride=1, padding=0)
        self.downsample_conv = nn.Conv2d(hidden_size, hidden_size, kernel_size=ds, stride=ds, padding=0)
        self.upsample_conv = nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=ds, stride=ds, padding=0)
        self.cosmos_out_proj = nn.Linear(hidden_size, vae_dim)
        self.cosmos_out_layer_norm = nn.LayerNorm(hidden_size)

        self.action_in_proj = nn.Linear(config.max_action_dim, hidden_size)
        self.action_out_proj = nn.Linear(hidden_size, config.max_action_dim)
        self.state_proj = nn.Linear(config.max_state_dim, hidden_size)
        self.action_time_mlp_in = nn.Linear(2 * hidden_size, hidden_size)
        self.action_time_mlp_out = nn.Linear(hidden_size, hidden_size)

        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)

        self.cosmos.eval()
        for param in self.cosmos.parameters():
            param.requires_grad = False

        self.set_attention_implementation(config.attn_implementation)

    def set_attention_implementation(self, attn_implementation: str) -> None:
        self.config.attn_implementation = attn_implementation
        self.qwen3_vl_with_expert.und_expert.config.text_config._attn_implementation = attn_implementation
        self.qwen3_vl_with_expert.und_expert.language_model.config._attn_implementation = attn_implementation
        self.qwen3_vl_with_expert.gen_expert.config._attn_implementation = attn_implementation
        self.qwen3_vl_with_expert.act_expert.config._attn_implementation = attn_implementation

    def _prepare_attention_masks_4d(
        self,
        att_2d_masks: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        bias_dtype = torch.float32 if self.config.attn_implementation == "eager" else dtype
        valid_bias = torch.zeros((), dtype=bias_dtype, device=att_2d_masks.device)
        invalid_bias = torch.tensor(
            OPENPI_ATTENTION_MASK_VALUE,
            dtype=bias_dtype,
            device=att_2d_masks.device,
        )
        return torch.where(att_2d_masks_4d, valid_bias, invalid_bias)

    def sample_noise(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        return torch.normal(0.0, 1.0, shape, dtype=torch.float32, device=device)

    @dynamo.disable
    def embed_prefix(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_token_id = self.qwen3_vl_with_expert.und_expert.config.image_token_id
        pixel_values = pixel_values.view(-1, pixel_values.shape[-1])
        image_grid_thw = image_grid_thw.view(-1, 3)
        image_embs, _ = self.qwen3_vl_with_expert.und_expert.visual(pixel_values, image_grid_thw)

        embs = self.qwen3_vl_with_expert.und_expert.get_input_embeddings()(lang_tokens)
        batch_size, seq_len, hidden_dim = embs.shape
        embs = embs.view(-1, hidden_dim)
        lang_tokens_flat = lang_tokens.view(-1)
        embs[lang_tokens_flat == image_token_id] = image_embs
        embs = embs.view(batch_size, seq_len, hidden_dim)

        pad_masks = lang_masks.to(torch.bool)
        att_masks = torch.zeros_like(pad_masks, dtype=torch.bool, device=pad_masks.device)
        return embs, pad_masks, att_masks

    def get_cosmos_features(self, images: torch.Tensor) -> torch.Tensor:
        shape = images.shape[:-3]
        channels, height, width = images.shape[-3:]
        images = images.reshape(-1, channels, height, width)
        if (height, width) != (256, 256):
            images = F.interpolate(images, size=(256, 256), mode="bilinear", align_corners=False)
        images = images * 2 - 1
        features = self.cosmos.encode(images)
        channels, height, width = features.shape[-3:]
        return features.view(*shape, channels, height, width)

    def embed_middle(
        self,
        images: torch.Tensor,
        img_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = images.device
        batch_size, n_view, timesteps = images.shape[:3]
        features = self.get_cosmos_features(images)
        features = rearrange(features, "b n t c h w -> (b n t) c h w")
        features = self.cosmos_in_proj(features)
        features = self.downsample_conv(features)
        features = rearrange(features, "(b n t) c h w -> b n t c h w", b=batch_size, n=n_view, t=timesteps)

        _, _, _, _, height, width = features.shape
        embs = rearrange(features, "b n t c h w -> b (n t h w) c")
        pad_masks = torch.zeros((batch_size, n_view, timesteps, height, width), dtype=torch.bool, device=device)
        pad_masks[img_masks] = True
        pad_masks = rearrange(pad_masks, "b n t h w -> b (n t h w)")
        att_masks = torch.tensor([1] + [0] * (embs.shape[1] - 1), dtype=torch.bool, device=device)
        att_masks = att_masks[None, :].expand(batch_size, att_masks.shape[0])
        return embs, pad_masks, att_masks

    def prepare_suffix_static_context(
        self,
        state: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        max_prefix_position_ids: torch.Tensor,
    ) -> SuffixStaticContext:
        if self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)

        state_emb = self.state_proj(state)
        device = state_emb.device
        batch_size = state_emb.shape[0]
        suffix_len = self.config.chunk_size + 1
        prefix_len = prefix_pad_masks.shape[1]

        suffix_pad_masks = torch.ones(batch_size, suffix_len, dtype=torch.bool, device=device)
        suffix_att_masks = torch.tensor([1] + [1] + [0] * (self.config.chunk_size - 1), dtype=torch.bool, device=device)
        suffix_att_masks = suffix_att_masks[None, :].expand(batch_size, suffix_len)
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(
            torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2),
            dtype=state_emb.dtype,
        )
        position_ids = (
            torch.arange(1, suffix_len + 1, device=max_prefix_position_ids.device)
            .repeat(3, 1, 1)
            .to(max_prefix_position_ids)
            + max_prefix_position_ids
        )
        return SuffixStaticContext(
            state_emb=state_emb,
            full_att_2d_masks_4d=full_att_2d_masks_4d,
            position_ids=position_ids,
        )

    def embed_suffix(
        self,
        state: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)

        state_emb = self.state_proj(state)
        batch_size = state_emb.shape[0]
        device = state_emb.device
        state_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        ).to(dtype=timestep.dtype)
        action_emb = self.action_in_proj(noisy_actions)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(action_time_emb)))
        action_mask = torch.ones(batch_size, action_time_emb.shape[1], dtype=torch.bool, device=device)

        embs = torch.cat([state_emb[:, None, :], action_time_emb], dim=1)
        pad_masks = torch.cat([state_mask, action_mask], dim=1)
        att_masks = torch.tensor([1] + [1] + [0] * (self.config.chunk_size - 1), dtype=embs.dtype, device=device)
        att_masks = att_masks[None, :].expand(batch_size, self.config.chunk_size + 1)
        return embs, pad_masks, att_masks

    def get_position_ids(
        self,
        lang_tokens: torch.Tensor,
        image_grid_thw: torch.Tensor | None,
        pad_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, Any]:
        seq_len = lang_tokens.shape[1]
        padded_lang_tokens = torch.ones_like(pad_masks).to(lang_tokens) * 777
        padded_lang_tokens[:, :seq_len] = lang_tokens
        attention_mask = pad_masks.to(lang_tokens)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.view(-1, 3)
        return self.qwen3_vl_with_expert.und_expert.model.get_rope_index(
            padded_lang_tokens,
            image_grid_thw,
            attention_mask=attention_mask,
        )

    @torch.no_grad()
    def sample_actions(
        self,
        images: torch.Tensor,
        img_masks: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
        decode_image: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        batch_size = state.shape[0]
        device = state.device
        dtype = state.dtype
        if noise is None:
            noise = self.sample_noise((batch_size, self.config.chunk_size, self.config.max_action_dim), device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            pixel_values,
            image_grid_thw,
            lang_tokens,
            lang_masks,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids, _ = self.get_position_ids(lang_tokens, image_grid_thw, prefix_pad_masks)
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(
            prefix_att_2d_masks,
            dtype=prefix_embs.dtype,
        )
        _, past_key_values = self.qwen3_vl_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None, None],
            use_cache=True,
        )
        max_prefix_position_ids = prefix_position_ids.max(dim=-1, keepdim=True).values

        middle_embs, middle_pad_masks, middle_att_masks = self.embed_middle(images[:, :, :2], img_masks)
        middle_len = middle_pad_masks.shape[1]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, middle_len, prefix_len)
        middle_att_2d_masks = make_att_2d_masks(middle_pad_masks, middle_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, middle_att_2d_masks], dim=2)
        middle_position_ids = (
            torch.arange(1, middle_len + 1, device=max_prefix_position_ids.device)
            .repeat(3, 1, 1)
            .to(max_prefix_position_ids)
            + max_prefix_position_ids
        )
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(
            full_att_2d_masks,
            dtype=middle_embs.dtype,
        )
        (_, middle_out, _), past_key_values = self.qwen3_vl_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=middle_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, middle_embs, None],
            use_cache=True,
        )

        max_position_ids = middle_position_ids.max(dim=-1, keepdim=True).values
        curr_pad_masks = torch.cat([prefix_pad_masks, middle_pad_masks], dim=1)
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        suffix_static = None
        if self.config.enable_suffix_static_context_optimization:
            suffix_static = self.prepare_suffix_static_context(state, curr_pad_masks, max_position_ids)
        while time >= -dt / 2:
            expanded_time = time.expand(batch_size)
            if suffix_static is None:
                v_t = self.denoise_step(
                    state,
                    curr_pad_masks,
                    past_key_values,
                    max_position_ids,
                    x_t.to(dtype),
                    expanded_time.to(dtype),
                )
            else:
                v_t = self.denoise_step_optimized(
                    suffix_static, past_key_values, x_t.to(dtype), expanded_time.to(dtype)
                )
            x_t = x_t + dt * v_t
            time += dt

        return x_t, None if not decode_image else middle_out

    def denoise_step(
        self,
        state: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values: Any,
        max_prefix_position_ids: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        position_ids = (
            torch.arange(1, suffix_len + 1, device=max_prefix_position_ids.device)
            .repeat(3, 1, 1)
            .to(max_prefix_position_ids)
            + max_prefix_position_ids
        )
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(
            full_att_2d_masks,
            dtype=suffix_embs.dtype,
        )
        outputs_embeds, _ = self.qwen3_vl_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, None, suffix_embs],
            use_cache=False,
        )
        suffix_out = outputs_embeds[2][:, -self.config.chunk_size :].to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    def denoise_step_optimized(
        self,
        suffix_static: SuffixStaticContext,
        past_key_values: Any,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        state_emb = suffix_static.state_emb
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        ).to(dtype=timestep.dtype)
        action_emb = self.action_in_proj(x_t)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(action_time_emb)))
        suffix_embs = torch.cat([state_emb[:, None, :], action_time_emb], dim=1)
        outputs_embeds, _ = self.qwen3_vl_with_expert.forward(
            attention_mask=suffix_static.full_att_2d_masks_4d,
            position_ids=suffix_static.position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, None, suffix_embs],
            use_cache=False,
        )
        suffix_out = outputs_embeds[2][:, -self.config.chunk_size :].to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


class InternVLAA1Policy(nn.Module):
    _AUTO_WEIGHTS_IGNORE_PREFIXES = ["model.cosmos."]

    def __init__(
        self,
        config: InternVLAA1Config,
        *,
        processor_model_name: str = DEFAULT_QWEN3_VL_MODEL,
    ) -> None:
        super().__init__()
        self.config = config
        self.input_builder = Qwen3VLInputBuilder(
            processor_model_name=processor_model_name,
            max_length=config.tokenizer_max_length,
        )
        cosmos_encoder_path, cosmos_decoder_path = resolve_cosmos_checkpoint_paths()
        self.model = InternVLAA1(
            config,
            cosmos_encoder_path=cosmos_encoder_path,
            cosmos_decoder_path=cosmos_decoder_path,
        )
        self.model.to(config.device)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str | Path,
        *,
        config: InternVLAA1Config | None = None,
        processor_model_name: str = DEFAULT_QWEN3_VL_MODEL,
        strict: bool = False,
    ) -> InternVLAA1Policy:
        checkpoint_dir = Path(checkpoint_dir)
        if config is None:
            config = InternVLAA1Config.from_pretrained(checkpoint_dir)
        instance = cls(config, processor_model_name=processor_model_name)
        checkpoint_path = checkpoint_dir / "model.safetensors"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"InternVLAA1 checkpoint not found: {checkpoint_path}")

        loader = AutoWeightsLoader(
            instance,
            ignore_unexpected_prefixes=cls._AUTO_WEIGHTS_IGNORE_PREFIXES,
        )
        with safe_open(str(checkpoint_path), framework="pt", device=str(config.device)) as f:
            file_keys = list(f.keys())
            loaded = loader.load_weights((name, f.get_tensor(name)) for name in file_keys)

        if strict:
            expected = {
                name
                for name, _ in instance.state_dict().items()
                if not any(name.startswith(prefix) for prefix in cls._AUTO_WEIGHTS_IGNORE_PREFIXES)
            }
            missing = sorted(expected - loaded)
            if missing:
                preview = ", ".join(missing[:10])
                suffix = " ..." if len(missing) > 10 else ""
                raise RuntimeError(f"Missing weights after AutoWeightsLoader load: {preview}{suffix}")

        instance.eval()
        return instance

    def to(self, *args, **kwargs):  # type: ignore[override]
        super().to(*args, **kwargs)
        self.model.cosmos.to(torch.bfloat16)
        self.model.action_out_proj.to(torch.float32)
        return self

    def _prepare_resized_histories(self, batch: dict[str, Any]) -> list[torch.Tensor]:
        resized_images: list[torch.Tensor] = []
        for i in range(3):
            image_history = batch[f"{OBS_IMAGES}.image{i}"]
            if image_history.shape[0] != 1:
                raise ValueError(f"InternVLAA1Policy only supports bs=1, got image batch {tuple(image_history.shape)}")
            resized_images.append(resize_with_pad(image_history[0], self.config.image_resolution))
        return resized_images

    def _preprocess_images(
        self,
        batch: dict[str, torch.Tensor],
        *,
        resized_histories: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if resized_histories is None:
            resized_histories = self._prepare_resized_histories(batch)
        images = torch.stack(resized_histories, dim=0).unsqueeze(0)
        img_masks = torch.stack([batch[f"{OBS_IMAGES}.image{i}_mask"] for i in range(3)], dim=1)
        return images, img_masks

    def prepare_state(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return pad_vector(batch[OBS_STATE], self.config.max_state_dim)

    def _get_task(self, batch: dict[str, Any]) -> str:
        task = batch[OBS_TASK]
        if isinstance(task, str):
            return task
        if isinstance(task, (list, tuple)):
            if len(task) != 1:
                raise ValueError(f"InternVLAA1Policy only supports bs=1, got {len(task)} tasks")
            item = task[0]
            if not isinstance(item, str):
                raise TypeError(f"Expected task string, got {type(item)!r}")
            return item
        raise TypeError(f"Unsupported task payload type: {type(task)!r}")

    def _prepare_qwen_prefix_inputs(
        self,
        resized_histories: list[torch.Tensor],
        task: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qwen_inputs = self.input_builder.build(resized_histories, task)
        device = self.model.action_in_proj.weight.device
        return (
            qwen_inputs[f"{OBS_PREFIX}pixel_values"]
            .unsqueeze(0)
            .to(device=device, dtype=self.model.action_in_proj.weight.dtype),
            qwen_inputs[f"{OBS_PREFIX}image_grid_thw"].unsqueeze(0).to(device=device),
            qwen_inputs[f"{OBS_PREFIX}input_ids"].unsqueeze(0).to(device=device),
            qwen_inputs[f"{OBS_PREFIX}attention_mask"].unsqueeze(0).to(device=device),
        )

    @torch.inference_mode()
    def forward(
        self,
        batch: dict[str, Any],
        *,
        noise: torch.Tensor | None = None,
        decode_image: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        task = self._get_task(batch)
        resized_histories = self._prepare_resized_histories(batch)
        pixel_values, image_grid_thw, lang_tokens, lang_masks = self._prepare_qwen_prefix_inputs(
            resized_histories,
            task,
        )
        state = self.prepare_state(batch)
        images, img_masks = self._preprocess_images(batch, resized_histories=resized_histories)
        return self.model.sample_actions(
            images,
            img_masks,
            pixel_values,
            image_grid_thw,
            lang_tokens,
            lang_masks,
            state,
            noise=noise,
            decode_image=decode_image,
        )
