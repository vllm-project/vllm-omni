# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from Tencent-Hunyuan/Hy-Nomas@911cea6458 (Copyright 2026 Tencent).

import os
from collections.abc import Iterable
from functools import partial
from types import SimpleNamespace

import regex as re
import torch
from omegaconf import DictConfig
from torch import nn
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import WeightsMapper

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention
from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
from vllm_omni.diffusion.models.hy_nomas.attention import RangeAttention, action_ranges, assemble_kv
from vllm_omni.diffusion.models.hy_nomas.layers import (
    MLP,
    FinalLayer,
    ModulateDiT,
    PatchEmbed,
    RMSNorm,
    SingleTokenRefiner,
    TimestepEmbedder,
    apply_rotary,
    modulate,
    rotary_tables,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.segmented import ARDiffusionSegmentedKVState

_DIT_NAMES = WeightsMapper(
    orig_to_new_regex={
        re.compile(r"^(double_blocks\.\d+)\.attn\.((?:img|txt)_attn_.+)$"): r"\1.\2",
        re.compile(r"^(txt_in\.individual_token_refiner\.blocks\.\d+)\.attn\.(self_attn_.+)$"): r"\1.\2",
        re.compile(r"^double_blocks\.(\d+)\.action_module\.(.+)$"): r"action_modules.\1.\2",
        re.compile(r"\._checkpoint_wrapped_module\."): ".",
        re.compile(r"_checkpoint_wrapped_module\."): "",
    },
)


def build_transformer(config: DictConfig, device: torch.device, model_config: dict) -> nn.Module:
    transformer = HyNomasTransformer(
        model_config, sp_size=config.parallelism.sp_size, device=device, dtype=torch.bfloat16
    )
    return transformer.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)


def load_transformer_weights(transformer: nn.Module, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    return load_component_weights(transformer, _DIT_NAMES.apply(weights))


def load_component_weights(module: nn.Module, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    parameters = dict(module.named_parameters())
    buffers = dict(module.named_buffers())
    loaded = set()
    for name, tensor in weights:
        if name in loaded:
            raise ValueError(f"Duplicate Hy-Nomas parameter after normalization: {name}")
        target = parameters.get(name, buffers.get(name))
        if target is None:
            raise ValueError(f"Unexpected Hy-Nomas weight: {name}")
        default_weight_loader(target, tensor)
        loaded.add(name)
    missing = parameters.keys() - loaded
    if missing:
        raise ValueError(f"Hy-Nomas checkpoint is missing {len(missing)} parameters: {sorted(missing)[:10]}")
    return loaded


def validate_sequence_parallelism(sp_size: int) -> None:
    if sp_size > 1:
        group = get_sp_group()
        if group.ulysses_world_size != sp_size or group.world_size != sp_size:
            raise ValueError("Hy-Nomas requires the initialized SP group to match ulysses_degree.")


class MMDoubleStreamBlock(nn.Module):
    def __init__(self, hidden, heads, ratio, **kwargs):
        super().__init__()
        self.heads_num = heads
        for stream in ("img", "txt"):
            setattr(self, stream + "_mod", ModulateDiT(hidden, **kwargs))
            for index in (1, 2):
                setattr(
                    self, f"{stream}_norm{index}", nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6, **kwargs)
                )
            for projection in ("q", "k", "v", "proj"):
                setattr(self, f"{stream}_attn_{projection}", nn.Linear(hidden, hidden, **kwargs))
            for projection in ("q", "k"):
                setattr(self, f"{stream}_attn_{projection}_norm", RMSNorm(hidden // heads, **kwargs))
            setattr(
                self,
                stream + "_mlp",
                MLP(hidden, int(hidden * ratio), act=partial(nn.GELU, approximate="tanh"), **kwargs),
            )

    def forward(
        self,
        img,
        txt,
        vec,
        tables,
        text_mask,
        cache,
        history,
        update,
        cache_txt,
        cache_mask,
        layer_idx,
        attention,
        sp_size,
        offsets,
        windows,
        tokens_per_frame,
        forward_masks=None,
        forward_visibility=None,
    ):
        streams = []
        for name, x in (("img", img), ("txt", txt)):
            mods = getattr(self, name + "_mod")(vec).chunk(6, dim=-1)
            normalized = modulate(getattr(self, name + "_norm1")(x), mods[0], mods[1])
            q, k, v = [
                getattr(self, f"{name}_attn_{projection}")(normalized).view(x.shape[0], x.shape[1], self.heads_num, -1)
                for projection in ("q", "k", "v")
            ]
            q = getattr(self, name + "_attn_q_norm")(q).to(v)
            k = getattr(self, name + "_attn_k_norm")(k).to(v)
            if name == "img":
                q, k = apply_rotary(q, tables), apply_rotary(k, tables)
            streams.append((q, k, v, mods))
        iq, ik, iv, im = streams[0]
        tq, tk, tv, tm = streams[1]
        dtype = iq.dtype
        cache_dtype = cache.cache.dtype if isinstance(cache, ARDiffusionSegmentedKVState) else cache["k"].dtype
        iq, ik, iv, tq, tk, tv = [value.to(cache_dtype) for value in (iq, ik, iv, tq, tk, tv)]
        image_tokens = img.shape[1] * sp_size
        if forward_visibility is not None and "inputs" in forward_visibility:
            current_mask, key_mask, ranges = forward_visibility["inputs"]
        else:
            image_mask = torch.ones((img.shape[0], image_tokens), dtype=torch.bool, device=ik.device)
            current_mask = torch.cat((image_mask, text_mask.bool()), 1)
            history_mask = cache_mask[:, :history]
            if update:
                count = image_tokens + (text_mask.shape[1] if cache_txt else 0)
                cache_mask[:, history : history + count] = current_mask if cache_txt else image_mask
            key_mask = torch.cat((history_mask, current_mask if cache_txt or update else image_mask), 1)
            ranges = None
            if cache_txt and offsets is not None and windows:
                ranges = action_ranges(history_mask, text_mask.bool(), image_tokens, offsets, windows, tokens_per_frame)
            if forward_visibility is not None:
                forward_visibility["inputs"] = (current_mask, key_mask, ranges)
        metadata = AttentionMetadata(
            joint_query=tq if sp_size > 1 else None,
            joint_key=tk if sp_size > 1 else None,
            joint_value=tv if sp_size > 1 else None,
            joint_strategy="rear",
        )
        q, k, v = (
            (iq, ik, iv) if sp_size > 1 else (torch.cat((iq, tq), 1), torch.cat((ik, tk), 1), torch.cat((iv, tv), 1))
        )
        strategy = (
            UlyssesParallelAttention(get_sp_group(), scatter_idx=2, gather_idx=1, use_sync=False)
            if sp_size > 1
            else NoParallelAttention()
        )
        q, k, v, _, context = strategy.pre_attention(q, k, v, metadata)
        k, v = assemble_kv(
            cache, layer_idx, k, v, history=history, media_tokens=image_tokens, cache_text=cache_txt, update=update
        )
        hidden = strategy.post_attention(
            attention(q, k, v, ranges=ranges, query_mask=current_mask, key_mask=key_mask, _forward_masks=forward_masks),
            context,
        )
        image_attention, text_attention = hidden[:, : img.shape[1]], hidden[:, img.shape[1] :]
        results = []
        for name, x, attn, mods in (("img", img, image_attention, im), ("txt", txt, text_attention, tm)):
            attn = attn.to(dtype).reshape(x.shape)
            x = x + getattr(self, name + "_attn_proj")(attn) * mods[2].unsqueeze(1)
            x = x + getattr(self, name + "_mlp")(modulate(getattr(self, name + "_norm2")(x), mods[3], mods[4])) * mods[
                5
            ].unsqueeze(1)
            results.append(x)
        return tuple(results)


class ActionModule(nn.Module):
    def __init__(self, config, hidden, **kwargs):
        super().__init__()
        channels, intermediate = config["camera_embed_dim"], config["camera_attention_dim"]
        self.heads_num = config["camera_num_attention_heads"]
        self.head_dim = intermediate // self.heads_num
        self.keyboard_embed = nn.Sequential(
            nn.Linear(64, channels, **kwargs), nn.SiLU(), nn.Linear(channels, channels, **kwargs)
        )
        self.img_attn_q = nn.Linear(hidden, intermediate, bias=False, **kwargs)
        self.keyboard_attn_kv = nn.Linear(channels, 2 * intermediate, bias=False, **kwargs)
        self.proj_keyboard = nn.Linear(intermediate, hidden, bias=False, **kwargs)
        self.key_attn_q_norm = RMSNorm(self.head_dim, **kwargs)
        self.key_attn_k_norm = RMSNorm(self.head_dim, **kwargs)
        cos, sin = rotary_tables((7500, 1, 1), config["camera_rope_axes_dim"], config["rope_theta"])
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

    def forward(self, x, frames, height, width, camera, attention, sp_size, camera_mask):
        batch, length, _ = x.shape
        q = self.img_attn_q(x).view(batch, length, self.heads_num, -1)
        k, v = self.keyboard_attn_kv(self.keyboard_embed(camera)).view(batch, frames, 2, self.heads_num, -1).unbind(2)
        q, k = self.key_attn_q_norm(q).to(v), self.key_attn_k_norm(k).to(v)
        first = 0 if sp_size == 1 else get_sp_group().rank_in_group * length
        frame_ids = (torch.arange(length, device=q.device) + first) // (height * width)
        q = apply_rotary(q, (self.freqs_cos[frame_ids], self.freqs_sin[frame_ids]))
        k = apply_rotary(k, (self.freqs_cos[:frames], self.freqs_sin[:frames]))
        rectangles = [
            (max(0, f * height * width - first), min(length, (f + 1) * height * width - first), f, f + 1)
            for f in range(frames)
            if f * height * width < first + length and (f + 1) * height * width > first
        ]
        # Preserve the reference D=128 padding and the original camera scale.
        if self.head_dim == 64:
            q, k, v = [nn.functional.pad(value, (0, 64)) for value in (q, k, v)]
        out = attention(q, k, v, ranges=[rectangles] * batch)[..., : self.head_dim]
        out = self.proj_keyboard(out.reshape(batch, length, -1))
        if camera_mask is not None:
            out = out * camera_mask.float()[:, None, None]
        return x + out


class HyNomasTransformer(nn.Module):
    def __init__(self, model_config, *, sp_size=1, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.config = SimpleNamespace(**model_config)
        self.in_channels, self.out_channels = self.config.in_channels, self.config.out_channels
        self.heads_num = self.config.num_attention_heads
        positive_dimensions = (
            "in_channels",
            "out_channels",
            "num_layers",
            "num_attention_heads",
            "attention_head_dim",
            "text_embed_dim",
            "camera_embed_dim",
            "camera_attention_dim",
            "camera_num_attention_heads",
        )
        if any(type(model_config[name]) is not int or model_config[name] <= 0 for name in positive_dimensions):
            raise ValueError("Hy-Nomas model dimensions must be positive integers.")
        self.hidden_size = self.heads_num * self.config.attention_head_dim
        for axes, dimension in (
            (self.config.rope_axes_dim, self.config.attention_head_dim),
            (
                self.config.camera_rope_axes_dim,
                self.config.camera_attention_dim // self.config.camera_num_attention_heads,
            ),
        ):
            if (
                len(axes) != 3
                or any(type(value) is not int or value <= 0 or value % 2 for value in axes)
                or sum(axes) != dimension
            ):
                raise ValueError(
                    "Hy-Nomas RoPE requires three positive even axes summing to the attention head dimension."
                )
        if self.config.camera_attention_dim % self.config.camera_num_attention_heads:
            raise ValueError("Hy-Nomas camera attention dimension must be divisible by its head count.")
        self.patch_size = (1, 1, 1)
        self.camera_injection_mode = "extrinsics_attn"
        self.sp_size = sp_size
        kwargs = {"device": device, "dtype": dtype}
        self.img_in = PatchEmbed(self.in_channels, self.hidden_size, **kwargs)
        self.txt_in = SingleTokenRefiner(self.config.text_embed_dim, self.hidden_size, self.heads_num, **kwargs)
        self.time_in = TimestepEmbedder(self.hidden_size, **kwargs)
        self.double_blocks = nn.ModuleList(
            MMDoubleStreamBlock(self.hidden_size, self.heads_num, self.config.mlp_ratio, **kwargs)
            for _ in range(self.config.num_layers)
        )
        self.final_layer = FinalLayer(self.hidden_size, self.out_channels, **kwargs)
        self.attention = RangeAttention(self.config.attention_head_dim**-0.5)
        camera_dim = self.config.camera_attention_dim // self.config.camera_num_attention_heads
        self.camera_attention = RangeAttention(camera_dim**-0.5)
        self.action_modules = nn.ModuleList(
            ActionModule(model_config, self.hidden_size, **kwargs) for _ in range(self.config.num_layers)
        )
        self.extrinsics_embedding_camctrl = nn.Linear(64, 64, **kwargs)
        self.c2ws_hidden_states_layer1 = nn.Linear(64, 64, **kwargs)
        self.c2ws_hidden_states_layer2 = nn.Linear(64, 64, **kwargs)

    def _preprocess_camera(self, *, norm_extrinsics, norm_intrinsics, action_onehot, oh, ow, img_dtype, device):
        values = norm_extrinsics.to(img_dtype)
        padding = (4 - values.shape[1] % 4) % 4
        if padding:
            values = torch.cat((values[:, :1].repeat(1, padding, 1, 1), values), 1)
        values = values.contiguous().view(values.shape[0], values.shape[1] // 4, 64)
        values = self.extrinsics_embedding_camctrl(values)
        return values + self.c2ws_hidden_states_layer2(nn.functional.silu(self.c2ws_hidden_states_layer1(values)))

    def forward_with_cache(
        self,
        hidden_states,
        timestep,
        text_states,
        text_mask,
        kv_cache_list=None,
        cache_valid_len=0,
        update_cache=False,
        global_frame_counter=0,
        guidance=None,
        cache_txt=True,
        cache_mask_list=None,
        camera_embedding=None,
        camera_has_mask=None,
        intra_seg_txt_offsets=None,
        intra_seg_vid_windows=None,
        camera_rope_time_offset=0,
        kv_cache_state=None,
    ):
        if camera_rope_time_offset != 0:
            raise ValueError("Hy-Nomas uses local camera RoPE.")
        if kv_cache_state is None and kv_cache_list is None:
            raise ValueError("Hy-Nomas forward requires KV storage.")
        batch, _, frames, height, width = hidden_states.shape
        tables = rotary_tables(
            (frames, height, width),
            self.config.rope_axes_dim,
            self.config.rope_theta,
            time_offset=global_frame_counter,
            device=hidden_states.device if global_frame_counter == 0 else None,
        )
        img = self.img_in(hidden_states)
        if self.sp_size > 1:
            rank = get_sp_group().rank_in_group
            img = torch.chunk(img, self.sp_size, 1)[rank]
            tables = tuple(torch.chunk(value, self.sp_size, 0)[rank] for value in tables)
        if os.environ.get("VLLM_OMNI_HY_NOMAS_B3_ROPE_UPLOAD_REUSE") == "1":
            # Keep CPU-generated offset tables bitwise unchanged, but upload
            # each rank-local table once instead of once per Q/K and layer.
            tables = tuple(value.to(hidden_states.device) for value in tables)
        vec = self.time_in(timestep)
        txt = self.txt_in(text_states, timestep, text_mask)
        # All layers of the segmented session share the same history visibility.
        # The memo lives for one forward only; denoise/commit and later prompts
        # always prepare their own masks. Separate per-layer masks keep the old path.
        forward_masks = None
        if (
            os.environ.get("VLLM_OMNI_HY_NOMAS_B3_MASK_REUSE") == "1"
            and cache_mask_list
            and all(mask is cache_mask_list[0] for mask in cache_mask_list)
        ):
            forward_masks = {}
        forward_visibility = (
            {}
            if forward_masks is not None and os.environ.get("VLLM_OMNI_HY_NOMAS_B3_VISIBILITY_REUSE") == "1"
            else None
        )
        for index, block in enumerate(self.double_blocks):
            img, txt = block(
                img,
                txt,
                vec,
                tables,
                text_mask,
                kv_cache_state if kv_cache_state is not None else kv_cache_list[index],
                cache_valid_len,
                update_cache,
                cache_txt,
                cache_mask_list[index],
                index,
                self.attention,
                self.sp_size,
                intra_seg_txt_offsets,
                intra_seg_vid_windows,
                height * width,
                forward_masks,
                forward_visibility,
            )
            if camera_embedding is not None:
                img = self.action_modules[index](
                    img, frames, height, width, camera_embedding, self.camera_attention, self.sp_size, camera_has_mask
                )
        if self.sp_size > 1:
            group = get_sp_group()
            img = group.all_gather(img, dim=1, group=group.ulysses_group)
        img = self.final_layer(img, vec)
        img = img.reshape(batch, frames, height, width, self.out_channels).permute(0, 4, 1, 2, 3)
        written = frames * height * width + (txt.shape[1] if cache_txt else 0)
        return img, written if update_cache else 0

    def clear_attention_cache(self):
        for attention in (self.attention, self.camera_attention):
            attention.clear()
