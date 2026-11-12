# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from Tencent-Hunyuan/Hy-Nomas@911cea6458 (Copyright 2026 Tencent).

from collections import OrderedDict
from collections.abc import Iterable
from functools import partial
from types import SimpleNamespace
from typing import ClassVar

import regex as re
import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn.attention.flex_attention import BlockMask
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
        *,
        block_masks: tuple[BlockMask | None, ...] | None = None,
    ):
        img_mod, txt_mod = self.img_mod(vec), self.txt_mod(vec)
        qkv = []
        for name, x, mod in (("img", img, img_mod), ("txt", txt, txt_mod)):
            shift, scale = mod.chunk(6, dim=-1)[:2]
            normalized = modulate(getattr(self, name + "_norm1")(x), shift, scale)
            q, k, v = [
                getattr(self, f"{name}_attn_{projection}")(normalized).view(x.shape[0], x.shape[1], self.heads_num, -1)
                for projection in ("q", "k", "v")
            ]
            q = getattr(self, name + "_attn_q_norm")(q).to(v)
            k = getattr(self, name + "_attn_k_norm")(k).to(v)
            if name == "img":
                q, k = apply_rotary(q, tables), apply_rotary(k, tables)
            qkv.extend((q, k, v))
        dtype = qkv[0].dtype
        image_attention, text_attention = self._attend_double_stream(
            tuple(qkv),
            (text_mask, cache, history, update, cache_txt, cache_mask, layer_idx, attention, sp_size),
            (offsets, windows, tokens_per_frame, block_masks),
        )
        results = []
        for name, x, attn, mod in (("img", img, image_attention, img_mod), ("txt", txt, text_attention, txt_mod)):
            gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)[2:]
            attn = attn.to(dtype).reshape(x.shape)
            x = x + getattr(self, name + "_attn_proj")(attn) * gate_msa.unsqueeze(1)
            x = x + getattr(self, name + "_mlp")(
                modulate(getattr(self, name + "_norm2")(x), shift_mlp, scale_mlp)
            ) * gate_mlp.unsqueeze(1)
            results.append(x)
        return tuple(results)

    @torch._dynamo.disable
    def _attend_double_stream(self, qkv, ctx, mask_args) -> tuple[torch.Tensor, torch.Tensor]:
        text_mask, cache, history, update, cache_txt, cache_mask, layer_idx, attention, sp_size = ctx
        offsets, windows, tokens_per_frame, block_masks = mask_args
        cache_dtype = cache.cache.dtype if isinstance(cache, ARDiffusionSegmentedKVState) else cache["k"].dtype
        iq, ik, iv, tq, tk, tv = [value.to(cache_dtype) for value in qkv]
        batch, img_len = iq.shape[:2]
        image_tokens = img_len * sp_size
        current_mask = key_mask = ranges = None
        if block_masks is None:
            image_mask = torch.ones((batch, image_tokens), dtype=torch.bool, device=ik.device)
            current_mask = torch.cat((image_mask, text_mask.bool()), 1)
            history_mask = cache_mask[:, :history]
            if update:
                count = image_tokens + (text_mask.shape[1] if cache_txt else 0)
                cache_mask[:, history : history + count] = current_mask if cache_txt else image_mask
            key_mask = torch.cat((history_mask, current_mask if cache_txt or update else image_mask), 1)
            if cache_txt and offsets is not None and windows:
                ranges = action_ranges(history_mask, text_mask.bool(), image_tokens, offsets, windows, tokens_per_frame)
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
        attn_out = (
            attention(q, k, v, block_masks=block_masks)
            if block_masks is not None
            else attention(q, k, v, ranges=ranges, query_mask=current_mask, key_mask=key_mask)
        )
        hidden = strategy.post_attention(attn_out, context)
        fmt = torch.contiguous_format  # Fresh storage prevents cross-break stride/offset guards.
        return hidden[:, :img_len].clone(memory_format=fmt), hidden[:, img_len:].clone(memory_format=fmt)


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
        axes, theta, dev = config["camera_rope_axes_dim"], config["rope_theta"], self.img_attn_q.weight.device
        cos, sin = (t.to(device=dev) for t in rotary_tables((7500, 1, 1), axes, theta))
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

    def forward(
        self,
        x,
        frames,
        height,
        width,
        camera,
        attention,
        sp_size,
        camera_mask,
        *,
        q_rotary=None,
        k_rotary=None,
        block_masks=None,
    ):
        batch, length, _ = x.shape
        q = self.img_attn_q(x).view(batch, length, self.heads_num, -1)
        k, v = self.keyboard_attn_kv(self.keyboard_embed(camera)).view(batch, frames, 2, self.heads_num, -1).unbind(2)
        q, k = self.key_attn_q_norm(q).to(v), self.key_attn_k_norm(k).to(v)
        if q_rotary is None or k_rotary is None or block_masks is None:
            q_rotary, k_rotary, block_masks = self._fallback_camera_plan(
                batch, length, frames, height, width, sp_size, attention, x.device, q_rotary, k_rotary, block_masks
            )
        q = apply_rotary(q, q_rotary)
        k = apply_rotary(k, k_rotary)
        # Preserve the reference D=128 padding and the original camera scale.
        if self.head_dim == 64:
            q, k, v = [nn.functional.pad(value, (0, 64)) for value in (q, k, v)]
        out = attention(q, k, v, block_masks=block_masks)[..., : self.head_dim]
        out = self.proj_keyboard(out.reshape(batch, length, -1))
        if camera_mask is not None:
            out = out * camera_mask.float()[:, None, None]
        return x + out

    @torch._dynamo.disable
    def _fallback_camera_plan(
        self, batch, length, frames, height, width, sp_size, attention, device, q_rotary, k_rotary, block_masks
    ):
        first = 0 if sp_size == 1 else get_sp_group().rank_in_group * length
        if q_rotary is None:
            frame_ids = (torch.arange(length, device=self.freqs_cos.device) + first) // (height * width)
            q_rotary = (self.freqs_cos[frame_ids].to(device=device), self.freqs_sin[frame_ids].to(device=device))
        if k_rotary is None:
            k_rotary = (
                self.freqs_cos[:frames].to(device=device).clone(),
                self.freqs_sin[:frames].to(device=device).clone(),
            )
        if block_masks is None:
            rectangles = [
                (max(0, f * height * width - first), min(length, (f + 1) * height * width - first), f, f + 1)
                for f in range(frames)
                if f * height * width < first + length and (f + 1) * height * width > first
            ]
            block_masks = attention.resolve_masks(length, frames, device, batch_size=batch, ranges=[rectangles] * batch)
        return q_rotary, k_rotary, block_masks


class HyNomasTransformer(nn.Module):
    _repeated_blocks: ClassVar[list[str]] = ["MMDoubleStreamBlock", "ActionModule"]

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
        self._step_rope_cache: OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._step_attention_plan_cache: OrderedDict[tuple, tuple] = OrderedDict()
        self._step_camera_plan_cache: OrderedDict[tuple, tuple] = OrderedDict()

    @staticmethod
    def _tensor_meta(t: torch.Tensor) -> tuple:
        return (id(t), t.data_ptr(), t._version, tuple(t.shape), t.device, t.dtype)

    @staticmethod
    def _lru_put(cache: OrderedDict, key: tuple, value: tuple) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > 8:
            cache.popitem(last=False)

    def _resolve_step_rope_tables(
        self, frames: int, height: int, width: int, gfc: int, rank: int, src_device, target_device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        compute_device = src_device if gfc == 0 else None
        rope_key = (frames, height, width, int(gfc), self.sp_size, rank, compute_device, target_device)
        if rope_key in self._step_rope_cache:
            self._step_rope_cache.move_to_end(rope_key)
            return self._step_rope_cache[rope_key]
        axes, theta = self.config.rope_axes_dim, self.config.rope_theta
        tables = rotary_tables((frames, height, width), axes, theta, time_offset=gfc, device=compute_device)
        if self.sp_size > 1:
            tables = tuple(torch.chunk(value, self.sp_size, 0)[rank].clone() for value in tables)
        tables = (tables[0].to(device=target_device), tables[1].to(device=target_device))
        self._lru_put(self._step_rope_cache, rope_key, tables)
        return tables

    @staticmethod
    def _normalize_nested(value):
        if value is None:
            return None
        return tuple(
            None if row is None else (tuple(int(item) for item in row) if isinstance(row, Iterable) else int(row))
            for row in value
        )

    def _make_step_attention_key(self, kw: tuple, update_cache: bool) -> tuple:
        shape, text_mask, unique_masks, valid_len, cache_txt, offsets, windows, device = kw
        has_act = bool(cache_txt and offsets is not None and windows)
        cache_meta = (int(valid_len), tuple(self._tensor_meta(m) for m in unique_masks)) if valid_len > 0 else (0,)
        act_meta = (self._normalize_nested(offsets), self._normalize_nested(windows)) if has_act else (None, None)
        flags = (bool(cache_txt), bool(update_cache), bool(cache_txt or update_cache))
        return (shape, self.sp_size, device, *flags, self._tensor_meta(text_mask), cache_meta, *act_meta)

    def _cache_attention_plan_variants(self, entry: tuple, kw: tuple, update_cache: bool) -> None:
        cache_txt = kw[4]
        for cand in (False, True):
            if bool(cache_txt or cand) == bool(cache_txt or update_cache):
                self._lru_put(self._step_attention_plan_cache, self._make_step_attention_key(kw, cand), entry)

    def _resolve_step_attention_plan(self, shape, img, text_mask, cache_masks, valid_len, update, cache_txt, act_args):
        batch, _, height, width = shape
        img_local, device = img.shape[1], img.device
        offsets, windows = act_args
        unique_masks = tuple({id(mask): mask for mask in cache_masks}.values())
        kw = (shape, text_mask, unique_masks, valid_len, cache_txt, offsets, windows, device)
        plan_key = self._make_step_attention_key(kw, update)
        if plan_key in self._step_attention_plan_cache:
            self._step_attention_plan_cache.move_to_end(plan_key)
            current_mask, resolved, _ = self._step_attention_plan_cache[plan_key]
        else:
            image_tokens = img_local * self.sp_size
            image_mask = torch.ones((batch, image_tokens), dtype=torch.bool, device=device)
            text_mask_bool = text_mask.bool()
            current_mask = torch.cat((image_mask, text_mask_bool), 1)
            first_hist = unique_masks[0][:, :valid_len]
            hetero = (
                len(unique_masks) > 1
                and valid_len > 0
                and any(not torch.equal(m[:, :valid_len], first_hist) for m in unique_masks[1:])
            )
            histories = [m[:, :valid_len] for m in unique_masks] if hetero else [first_hist]
            resolved_list = []
            for history_mask in histories:
                key_mask = torch.cat((history_mask, current_mask if cache_txt or update else image_mask), 1)
                ranges = (
                    action_ranges(history_mask, text_mask_bool, image_tokens, offsets, windows, height * width)
                    if cache_txt and offsets is not None and windows
                    else None
                )
                q_len, k_len = current_mask.shape[1], key_mask.shape[1]
                bm = self.attention.resolve_masks(
                    q_len, k_len, device, batch_size=batch, ranges=ranges, query_mask=current_mask, key_mask=key_mask
                )
                resolved_list.append(bm)
            resolved = tuple(resolved_list)
            self._cache_attention_plan_variants((current_mask, resolved, (text_mask, unique_masks)), kw, update)
        if update:
            count = img_local * self.sp_size + (text_mask.shape[1] if cache_txt else 0)
            written_slice = current_mask[:, :count]
            for mask in unique_masks:
                mask[:, valid_len : valid_len + count] = written_slice
            self._cache_attention_plan_variants((current_mask, resolved, (text_mask, unique_masks)), kw, update)
        by_id = {id(m): bm for m, bm in zip(unique_masks, resolved)}
        return resolved * len(cache_masks) if len(resolved) == 1 else tuple(by_id[id(m)] for m in cache_masks)

    def _resolve_step_camera_plan(
        self, *, batch: int, length: int, frames: int, height: int, width: int, device: torch.device
    ):
        first = 0 if self.sp_size == 1 else get_sp_group().rank_in_group * length
        freqs_cos, freqs_sin = self.action_modules[0].freqs_cos, self.action_modules[0].freqs_sin
        meta = (self._tensor_meta(freqs_cos), self._tensor_meta(freqs_sin))
        cam_key = (batch, length, first, frames, height, width, device, *meta)
        if cam_key in self._step_camera_plan_cache:
            self._step_camera_plan_cache.move_to_end(cam_key)
            q_rotary, k_rotary, cam_block_masks, _ = self._step_camera_plan_cache[cam_key]
            return q_rotary, k_rotary, cam_block_masks
        frame_ids = (torch.arange(length, device=freqs_cos.device) + first) // (height * width)
        q_rotary = (freqs_cos[frame_ids].to(device=device), freqs_sin[frame_ids].to(device=device))
        k_rotary = (freqs_cos[:frames].to(device=device).clone(), freqs_sin[:frames].to(device=device).clone())
        rectangles = [
            (max(0, f * height * width - first), min(length, (f + 1) * height * width - first), f, f + 1)
            for f in range(frames)
            if f * height * width < first + length and (f + 1) * height * width > first
        ]
        cam_block_masks = self.camera_attention.resolve_masks(
            length, frames, device, batch_size=batch, ranges=[rectangles] * batch
        )
        self._lru_put(
            self._step_camera_plan_cache, cam_key, (q_rotary, k_rotary, cam_block_masks, (freqs_cos, freqs_sin))
        )
        return q_rotary, k_rotary, cam_block_masks

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
        img = self.img_in(hidden_states)
        rank = get_sp_group().rank_in_group if self.sp_size > 1 else 0
        if self.sp_size > 1:
            img = torch.chunk(img, self.sp_size, 1)[rank]
        img = img.clone(memory_format=torch.contiguous_format)  # Normalize PatchEmbed/chunk view stride.
        tables = self._resolve_step_rope_tables(
            frames, height, width, global_frame_counter, rank, hidden_states.device, img.device
        )
        vec = self.time_in(timestep)
        txt = self.txt_in(text_states, timestep, text_mask)
        shape, act_args = (batch, frames, height, width), (intra_seg_txt_offsets, intra_seg_vid_windows)
        layer_block_masks = self._resolve_step_attention_plan(
            shape, img, text_mask, cache_mask_list, cache_valid_len, update_cache, cache_txt, act_args
        )
        cam_q = cam_k = cam_masks = None
        if camera_embedding is not None and len(self.action_modules) > 0:
            cam_q, cam_k, cam_masks = self._resolve_step_camera_plan(
                batch=batch, length=img.shape[1], frames=frames, height=height, width=width, device=img.device
            )
        cam_args = (frames, height, width, camera_embedding, self.camera_attention, self.sp_size, camera_has_mask)
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
                None,
                None,
                height * width,
                block_masks=layer_block_masks[index],
            )
            if camera_embedding is not None:
                img = self.action_modules[index](img, *cam_args, q_rotary=cam_q, k_rotary=cam_k, block_masks=cam_masks)
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
        self._step_rope_cache.clear()
        self._step_attention_plan_cache.clear()
        self._step_camera_plan_cache.clear()
