# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted for inference from huggingface/lerobot at f58f7fb9dbb241304df96e967aea0f5b42ff9a4d.
from __future__ import annotations

import logging
from collections.abc import Iterable

import einops
import torch
from torch import Tensor, nn
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

from .modeling_lingbot_vla_v2_base import FlowMatching, replace_lnorm_with_adanorm
from .qwen2_action_expert import Qwen2ForCausalLM, Qwen2TokenMoeBlock
from .qwen3vl_in_vla import Qwen3VLForConditionalGeneration
from .utils import make_att_2d_masks, our_eager_attention_forward, our_sdpa_attention_forward

logger = logging.getLogger(__name__)


class QwenvlWithExpertV2Config(PretrainedConfig):
    model_type = "QwenvlWithExpertV2Model"

    def __init__(
        self,
        vocab_size: int = 0,
        use_lm_head: bool = False,
        attention_implementation: str = "sdpa",
        use_cache: bool = False,
        expert_hidden_size: int = 768,
        expert_intermediate_size: int = 2752,
        action_num_attention_heads: int = 32,
        action_num_key_value_heads: int = 8,
        action_head_dim: int = 128,
        num_layers: int = 36,
        **kwargs,
    ):
        self.attention_implementation = attention_implementation
        self.vocab_size = vocab_size
        self.use_lm_head = use_lm_head
        self.action_num_attention_heads = action_num_attention_heads
        self.action_num_key_value_heads = action_num_key_value_heads
        self.action_head_dim = action_head_dim

        self.qwen_expert_config = CONFIG_MAPPING["qwen2"](
            attention_dropout=0.0,
            bos_token_id=151643,
            eos_token_id=151645,
            hidden_act="silu",
            hidden_size=expert_hidden_size,
            head_dim=action_head_dim,
            initializer_range=0.02,
            intermediate_size=expert_intermediate_size,
            max_position_embeddings=32768,
            max_window_layers=21,
            model_type="qwen2",
            num_attention_heads=action_num_attention_heads,
            num_hidden_layers=num_layers,
            num_key_value_heads=action_num_key_value_heads,
            rms_norm_eps=1e-06,
            rope_theta=1000000.0,
            sliding_window=32768,
            tie_word_embeddings=True,
            torch_dtype="bfloat16",
            transformers_version="4.57.3",
            use_cache=use_cache,
            use_sliding_window=False,
            vocab_size=151936,
        )
        super().__init__(**kwargs)


class QwenvlWithExpertV2Model(PreTrainedModel):
    config_class = QwenvlWithExpertV2Config

    def __init__(self, config: QwenvlWithExpertV2Config, vlm_config: PretrainedConfig):
        super().__init__(config=config)
        self.config = config
        # The LLM attention is computed by the custom dual-stream forward, so the HF
        # model is built with "eager"; the vision tower reads its value straight through.
        hf_attn = "eager"
        hf_vit_attn = self.config.vit_attn_implementation
        if self.config.vocab_size:
            vlm_config.text_config.vocab_size = self.config.vocab_size
        vlm_config._attn_implementation = hf_attn
        vlm_config.text_config._attn_implementation = hf_attn
        vlm_config.vision_config._attn_implementation = hf_vit_attn
        self.qwenvl = Qwen3VLForConditionalGeneration._from_config(vlm_config)
        if self.config.use_lm_head:
            self.qwenvl.tie_weights()

        self.config.qwen_expert_config._attn_implementation = hf_attn
        self.qwen_expert = Qwen2ForCausalLM._from_config(self.config.qwen_expert_config)

        if self.config.adanorm_time:
            replace_lnorm_with_adanorm(
                self.qwen_expert,
                self.config.qwen_expert_config.hidden_size,
                self.config.qwen_expert_config.hidden_size,
                config.final_norm_adanorm,
            )

        self._install_moe_blocks()
        del self.qwen_expert.model.embed_tokens
        self.attention_interface = self.get_attention_interface()

    def _install_moe_blocks(self):
        if not self.config.use_moe:
            return
        layers = self.qwen_expert.model.layers
        # Layers past the expert depth (e.g. the 36-layer default on a smaller VLM) are skipped.
        token_moe_layers = [idx for idx in self.config.token_moe_layers if idx < len(layers)]

        if token_moe_layers:
            token_config = CONFIG_MAPPING["qwen2_moe"](
                num_experts=self.config.token_num_experts,
                num_experts_per_tok=self.config.token_top_k,
                norm_topk_prob=True,
                hidden_size=self.config.qwen_expert_config.hidden_size,
                moe_intermediate_size=self.config.token_moe_intermediate_size,
                shared_expert_intermediate_size=self.config.token_shared_intermediate_size,
                output_router_logits=False,
            )
            token_config.router_activation = self.config.router_activation
            token_config.routed_scaling_factor = self.config.routed_scaling_factor
            token_config.use_shared_expert_gate = self.config.use_shared_expert_gate
            for idx in token_moe_layers:
                layers[idx].mlp = Qwen2TokenMoeBlock(token_config)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
    ):
        image_embeds, deepstack_image_embeds = self.qwenvl.model.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.qwenvl.model.visual.spatial_merge_size**2).tolist()
        image_chunks = list(torch.split(image_embeds, split_sizes))
        deepstack_chunks = [
            list(torch.split(deepstack_embeds, split_sizes)) for deepstack_embeds in deepstack_image_embeds
        ]
        image_embeds = torch.stack(image_chunks, dim=0)
        deepstack_image_embeds = [torch.stack(chunks, dim=0) for chunks in deepstack_chunks]
        return image_embeds, deepstack_image_embeds

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.qwenvl.model.language_model.embed_tokens(tokens)

    def embed_special_token(self, token_id: int, batch: int, count: int, device, dtype):
        token = torch.tensor([token_id], device=device, dtype=torch.long)
        emb = self.embed_language_tokens(token).to(dtype=dtype)
        return emb.view(1, 1, 1, -1).expand(batch, count, 1, -1)

    def build_prefix_position_ids(self, input_ids, attention_mask, image_grid_thw=None, video_grid_thw=None):
        # transformers>=5.5 externalized modality detection: get_rope_index now takes an
        # explicit ``mm_token_type_ids`` (0=text, 1=image, 2=video) instead of matching
        # the placeholder token ids internally. Reconstruct it from the vision token ids.
        vlm_cfg = self.qwenvl.config
        image_token_id = getattr(vlm_cfg, "image_token_id", None)
        video_token_id = getattr(vlm_cfg, "video_token_id", None)
        mm_token_type_ids = torch.zeros_like(input_ids)
        if image_token_id is not None:
            mm_token_type_ids[input_ids == image_token_id] = 1
        if video_token_id is not None:
            mm_token_type_ids[input_ids == video_token_id] = 2
        position_ids, _ = self.qwenvl.model.get_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        # transformers 4.57 (which the 6B checkpoint was trained under) filled masked
        # (padding) rope positions with 1; transformers 5.5 fills them with 0. Restore the
        # 4.57 convention so cached-prefix rope matches the trained weights exactly.
        if attention_mask is not None:
            pad = attention_mask == 0
            while pad.dim() < position_ids.dim():
                pad = pad.unsqueeze(0)
            position_ids = position_ids.masked_fill(pad.expand_as(position_ids), 1)
        return position_ids

    def apply_mrope(self, query_states, key_states, position_ids=None, position_embeddings=None):
        if position_embeddings is None:
            position_embeddings = self.qwenvl.model.language_model.rotary_emb(query_states, position_ids)
        return apply_rotary_pos_emb(query_states, key_states, *position_embeddings, unsqueeze_dim=2)

    def handle_kv_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        past_key_values: dict[int, dict[str, Tensor]] | None = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
    ):
        if use_cache:
            if past_key_values is None:
                past_key_values = {}
            if fill_kv_cache:
                past_key_values[layer_idx] = {"key_states": key_states, "value_states": value_states}
            else:
                key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                value_states = torch.cat([past_key_values[layer_idx]["value_states"], value_states], dim=1)
        return key_states, value_states, past_key_values

    def _apply_deepstack(self, hidden_states, layer_idx, visual_pos_masks, deepstack_visual_embeds):
        """Add the level's dense deepstack delta."""
        if (
            deepstack_visual_embeds is not None
            and visual_pos_masks is not None
            and layer_idx < len(deepstack_visual_embeds)
        ):
            hidden_states = hidden_states + deepstack_visual_embeds[layer_idx].to(hidden_states.dtype)
        return hidden_states

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: dict[int, dict[str, Tensor]] | None = None,
        inputs_embeds: list[Tensor | None] | None = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
        ada_cond: Tensor | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        # Every call site passes inputs_embeds explicitly (list of per-stream
        # embeds, entries possibly None); the default only satisfies the signature.
        assert inputs_embeds is not None
        models = [self.qwenvl.model.language_model, self.qwen_expert.model]
        num_layers = self.qwenvl.config.text_config.num_hidden_layers

        # mrope cos/sin depend only on position_ids (and dtype/device) — compute once
        # per forward instead of once per layer. Callers with a loop-invariant
        # position_ids (e.g. the flow-matching denoise loop) can pass a precomputed
        # ``position_embeddings`` to skip this entirely.
        if position_embeddings is None:
            rep = next(h for h in inputs_embeds if h is not None)
            position_embeddings = self.qwenvl.model.language_model.rotary_emb(rep, position_ids)

        for layer_idx in range(num_layers):
            inputs_embeds, past_key_values = self._layer_forward(
                layer_idx,
                inputs_embeds,
                attention_mask,
                position_embeddings,
                past_key_values,
                use_cache,
                fill_kv_cache,
                ada_cond,
                visual_pos_masks,
                deepstack_visual_embeds,
            )

        outputs_embeds: list[torch.FloatTensor | None] = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is None:
                outputs_embeds.append(None)
            elif self.config.final_norm_adanorm and i == 1:
                out_emb = models[i].norm(hidden_states, ada_cond)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(models[i].norm(hidden_states))
        return outputs_embeds, past_key_values

    def _layer_forward(
        self,
        layer_idx,
        inputs_embeds,
        attention_mask,
        position_embeddings,
        past_key_values,
        use_cache,
        fill_kv_cache,
        ada_cond,
        visual_pos_masks,
        deepstack_visual_embeds,
    ):
        """One dual-stream layer: per-stream QKV -> joint attention -> per-stream out/MLP."""
        models = [self.qwenvl.model.language_model, self.qwen_expert.model]
        query_states = []
        key_states = []
        value_states = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is None:
                continue
            if i == 1:
                q, k, v = models[i].layers[layer_idx](hidden_states, compute_kqv=True, ada_cond=ada_cond)
            else:
                q, k, v = models[i].layers[layer_idx](hidden_states, compute_kqv=True)
            query_states.append(q)
            key_states.append(k)
            value_states.append(v)

        query_states = torch.cat(query_states, dim=1)
        key_states = torch.cat(key_states, dim=1)
        value_states = torch.cat(value_states, dim=1)
        query_states, key_states = self.apply_mrope(query_states, key_states, position_embeddings=position_embeddings)
        key_states, value_states, past_key_values = self.handle_kv_cache(
            key_states,
            value_states,
            layer_idx,
            past_key_values=past_key_values,
            use_cache=use_cache,
            fill_kv_cache=fill_kv_cache,
        )
        att_output = self.attention_interface(query_states, key_states, value_states, attention_mask)

        outputs_embeds = []
        start = 0
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is None:
                outputs_embeds.append(None)
                continue
            end = start + hidden_states.shape[1]
            if i == 1:
                out_emb = models[i].layers[layer_idx](
                    hidden_states,
                    att_output,
                    start,
                    end,
                    output_atten=True,
                    ada_cond=ada_cond,
                )
            else:
                out_emb = models[i].layers[layer_idx](hidden_states, att_output, start, end, output_atten=True)
                out_emb = self._apply_deepstack(out_emb, layer_idx, visual_pos_masks, deepstack_visual_embeds)
            outputs_embeds.append(out_emb)
            start = end
        return outputs_embeds, past_key_values

    def get_attention_interface(self):
        if self.config.attention_implementation == "sdpa":
            return our_sdpa_attention_forward
        if self.config.attention_implementation == "eager":
            logger.debug("Using Eager attention")
            return our_eager_attention_forward
        raise ValueError(f"Invalid attention implementation: {self.config.attention_implementation}")


class LingBotVLAV2ForActionPrediction(FlowMatching):
    def __init__(self, config, vlm_config: PretrainedConfig | None = None):
        super().__init__()
        self.config = config
        if vlm_config is None:
            vlm_config = AutoConfig.from_pretrained(config.tokenizer_path, revision=config.tokenizer_revision)
        qwenvl_with_export_config = QwenvlWithExpertV2Config(
            vocab_size=config.vocab_size,
            use_lm_head=config.use_lm_head,
            attention_implementation=config.attention_implementation,
            use_cache=config.use_cache,
            expert_hidden_size=config.expert_hidden_size,
            expert_intermediate_size=config.expert_intermediate_size,
            action_num_attention_heads=config.action_num_attention_heads,
            action_num_key_value_heads=config.action_num_key_value_heads,
            action_head_dim=config.action_head_dim,
            num_layers=vlm_config.text_config.num_hidden_layers,
        )
        for name in [
            "adanorm_time",
            "final_norm_adanorm",
            "vit_attn_implementation",
            "use_moe",
            "token_moe_layers",
            "token_num_experts",
            "token_top_k",
            "token_moe_intermediate_size",
            "token_shared_intermediate_size",
            "router_activation",
            "routed_scaling_factor",
            "use_shared_expert_gate",
        ]:
            setattr(qwenvl_with_export_config, name, getattr(config, name))
        self.qwenvl_with_expert = QwenvlWithExpertV2Model(qwenvl_with_export_config, vlm_config)
        self.proj_width = width = config.expert_hidden_size

        self.state_proj = nn.Linear(config.max_state_dim, width)
        self.action_in_proj = nn.Linear(config.max_action_dim, width)
        self.action_out_proj = nn.Linear(width, config.max_action_dim)
        self.action_time_mlp_in = nn.Linear(width * 2, width)
        self.action_time_mlp_out = nn.Linear(width, width)

        self._init_align(config.align_params, vlm_config.text_config.hidden_size)
        del self.qwenvl_with_expert.qwenvl.lm_head
        del self.qwenvl_with_expert.qwen_expert.lm_head

    def _init_align(self, params: dict, llm_hidden_size: int):
        """Build the learned query tokens and projections used during action inference."""
        self.use_align = bool(params)
        self.num_task_tokens = 0
        self.use_future_depth = self.use_future_video = self.use_future_video_patch = False
        self.use_current_shared_task_proj = self.use_current_video_patch = self.use_future_video_cls = False
        self.future_video_share_future_depth_query = self.use_shared_future_task_proj = False
        self.block_future_depth_to_action = self.block_suffix_to_future_video = False
        if not params:
            return
        if params.get("mode") != "query":
            raise ValueError(f"align_params.mode must be 'query', got {params.get('mode')!r}.")
        dim = params["llm"]["dim_out"]
        if dim != llm_hidden_size:
            raise ValueError(f"align_params.llm.dim_out ({dim}) must equal the VLM hidden size ({llm_hidden_size}).")
        depth = params["depth"]
        self.num_task_tokens = params["num_task_tokens"]
        if depth["num_backbone_tokens"] % self.num_task_tokens:
            raise ValueError("align_params.depth.num_backbone_tokens must be divisible by num_task_tokens.")
        self.use_future_depth = bool(depth.get("use_future_depth", False))
        self.block_future_depth_to_action = bool(depth.get("block_future_depth_to_action", False))
        self.depth_align_embs = nn.Parameter(torch.randn(depth["num_backbone_tokens"], dim))
        if self.use_future_depth:
            self.future_depth_align_embs = nn.Parameter(torch.randn(depth["num_backbone_tokens"], dim))

        self.use_future_video = bool(params.get("use_future_video", False))
        if not self.use_future_video:
            return
        video = {**depth, **params.get("video", {})}
        self.use_future_video_patch = bool(video.get("use_patch_loss", True))
        self.use_current_video_patch = use_current_patch = bool(video.get("use_current_patch_loss", False))
        self.use_current_shared_task_proj = bool(video.get("use_current_shared_task_proj", use_current_patch))
        self.use_future_video_cls = bool(video.get("use_cls_loss", False))
        self.future_video_share_future_depth_query = bool(video.get("share_future_depth_query", False))
        self.use_shared_future_task_proj = bool(video.get("use_shared_future_task_proj", False))
        self.block_suffix_to_future_video = bool(video.get("block_suffix_to_future_video", False))
        if use_current_patch and not self.use_future_video_patch:
            raise ValueError("align_params.video.use_current_patch_loss requires use_patch_loss.")
        if self.use_current_shared_task_proj and not use_current_patch:
            raise ValueError("align_params.video.use_current_shared_task_proj requires use_current_patch_loss.")
        if self.use_shared_future_task_proj and not (
            self.use_future_video_patch and self.future_video_share_future_depth_query
        ):
            raise ValueError(
                "align_params.video.use_shared_future_task_proj requires use_patch_loss and share_future_depth_query."
            )
        if self.future_video_share_future_depth_query and (
            not self.use_future_depth or video["num_backbone_tokens"] != depth["num_backbone_tokens"]
        ):
            raise ValueError(
                "align_params.video.share_future_depth_query requires depth.use_future_depth and equal "
                "num_backbone_tokens."
            )
        if self.use_future_video_patch:
            if use_current_patch:
                self.current_video_align_embs = nn.Parameter(torch.randn(video["num_backbone_tokens"], dim))
                if self.use_current_shared_task_proj:
                    self.current_shared_task_proj = nn.Linear(dim * 2, dim)
            if not self.future_video_share_future_depth_query or self.use_shared_future_task_proj:
                self.future_video_align_embs = nn.Parameter(torch.randn(video["num_backbone_tokens"], dim))
            if self.use_shared_future_task_proj:
                self.future_shared_task_proj = nn.Linear(dim * 2, dim)
        if self.use_future_video_cls:
            self.future_video_cls_align_emb = nn.Embedding(1, dim)

    def _align_query_tokens(self):
        """[n, D] query tokens appended after the language tokens, in upstream ``prefix_query_segments``
        order: current (depth, mixed with current video), future-video cls, future video, future depth
        (mixed with future video when the future query is shared)."""

        def pool(embs):  # group consecutive rows into num_task_tokens tokens
            return embs.view(self.num_task_tokens, -1, embs.shape[-1]).mean(dim=1)

        current = pool(self.depth_align_embs)
        if self.use_current_shared_task_proj:
            current = self.current_shared_task_proj(torch.cat([current, pool(self.current_video_align_embs)], -1))
        tokens = [current]
        if self.use_future_video_cls:
            tokens.append(self.future_video_cls_align_emb.weight)
        if self.use_future_video_patch and not self.future_video_share_future_depth_query:
            tokens.append(pool(self.future_video_align_embs))
        if self.use_future_depth:
            future = pool(self.future_depth_align_embs)
            if self.use_shared_future_task_proj:
                future = self.future_shared_task_proj(torch.cat([future, pool(self.future_video_align_embs)], -1))
            tokens.append(future)
        return torch.cat(tokens, dim=0)

    def _block_suffix_to_queries_(self, att_2d_masks, suffix_row_start, prefix_len):
        """Optional upstream masks hiding the future query tokens from the suffix (off in released recipes)."""
        n = self.num_task_tokens
        if self.block_future_depth_to_action:  # upstream block_suffix_to_fv_: the last n prefix tokens
            att_2d_masks[:, suffix_row_start:, prefix_len - n : prefix_len] = False
        if self.block_suffix_to_future_video:
            own = int(self.use_future_video_cls) + (
                n if self.use_future_video_patch and not self.future_video_share_future_depth_query else 0
            )
            end = prefix_len - (n if self.use_future_depth else 0)
            att_2d_masks[:, suffix_row_start:, end - own : end] = False
        return att_2d_masks

    def embed_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        image_grid_thw=None,
    ):
        if image_grid_thw is None:
            raise ValueError("LingbotVLAV2Policy requires image_grid_thw from the Qwen3-VL image processor.")
        bsize = images.shape[0]
        device = images.device
        if images.ndim == 3:
            bsize = 1
            num_images = images.shape[0]
        else:
            num_images = images.shape[1] if images.ndim >= 4 else 1
        if images.ndim == 4:
            images = einops.rearrange(images, "b n l d -> (b n) l d")
        elif images.ndim == 5:
            images = einops.rearrange(images, "b n c h w -> (b n) c h w")
        if image_grid_thw.ndim == 3:
            flat_grid_thw = einops.rearrange(image_grid_thw, "b n d -> (b n) d")
        else:
            flat_grid_thw = image_grid_thw

        img_emb, deepstack_embs = self.qwenvl_with_expert.get_image_features(images, flat_grid_thw)
        embed_dtype = img_emb.dtype
        num_patch = img_emb.shape[1]
        img_emb = einops.rearrange(img_emb, "(b n) l d -> b n l d", b=bsize, n=num_images)
        deepstack_embs = [einops.rearrange(x, "(b n) l d -> b n l d", b=bsize, n=num_images) for x in deepstack_embs]
        if img_masks.ndim == 1:
            img_masks = img_masks.unsqueeze(0)

        cfg = self.qwenvl_with_expert.qwenvl.config
        visual_token_id = cfg.image_token_id

        start_emb = self.qwenvl_with_expert.embed_special_token(
            cfg.vision_start_token_id, bsize, num_images, device, embed_dtype
        )
        end_emb = self.qwenvl_with_expert.embed_special_token(
            cfg.vision_end_token_id, bsize, num_images, device, embed_dtype
        )
        img_chunks = torch.cat([start_emb, img_emb, end_emb], dim=2)
        image_token_len = num_patch + 2
        image_pad_masks = einops.repeat(img_masks, "b n -> b n l", l=image_token_len)
        image_visual_masks = torch.zeros_like(image_pad_masks)
        image_visual_masks[:, :, 1 : 1 + num_patch] = einops.repeat(img_masks, "b n -> b n l", l=num_patch)
        fake_image_ids = torch.full(
            (bsize, num_images, image_token_len),
            visual_token_id,
            dtype=torch.long,
            device=device,
        )
        fake_image_ids[:, :, 0] = cfg.vision_start_token_id
        fake_image_ids[:, :, -1] = cfg.vision_end_token_id
        # note (0xtoward): missing cameras take no MRoPE span (as in training), so image spans match rope_grid_thw.
        fake_image_ids = fake_image_ids.masked_fill(~img_masks[..., None], cfg.text_config.eos_token_id)

        img_emb = einops.rearrange(img_chunks, "b n l d -> b (n l) d")
        image_pad_masks = einops.rearrange(image_pad_masks, "b n l -> b (n l)")
        visual_pos_masks = einops.rearrange(image_visual_masks, "b n l -> b (n l)")
        fake_image_ids = einops.rearrange(fake_image_ids, "b n l -> b (n l)")

        lang_emb = self.qwenvl_with_expert.embed_language_tokens(lang_tokens).to(dtype=embed_dtype)

        embs = [img_emb, lang_emb]
        pad_masks = [image_pad_masks, lang_masks]
        prefix_input_ids = [fake_image_ids, lang_tokens.to(device)]
        full_visual_pos_masks = [visual_pos_masks, torch.zeros_like(lang_masks)]
        if self.use_align:  # query tokens after the language: text tokens with eos ids for mrope
            queries = self._align_query_tokens().to(embed_dtype).expand(bsize, -1, -1)
            query_masks = lang_masks.new_ones(bsize, queries.shape[1])
            embs.append(queries)
            pad_masks.append(query_masks)
            prefix_input_ids.append(torch.full_like(query_masks, cfg.text_config.eos_token_id, dtype=torch.long))
            full_visual_pos_masks.append(torch.zeros_like(query_masks))
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        prefix_input_ids = torch.cat(prefix_input_ids, dim=1)
        full_visual_pos_masks = torch.cat(full_visual_pos_masks, dim=1)

        if self.config.vlm_causal:
            att_masks = torch.ones((bsize, embs.shape[1]), device=device, dtype=torch.bool)
        else:
            att_masks = torch.zeros((bsize, embs.shape[1]), device=device, dtype=torch.bool)

        rope_grid_thw = flat_grid_thw[einops.rearrange(img_masks, "b n -> (b n)")]
        prefix_position_ids = self.qwenvl_with_expert.build_prefix_position_ids(
            prefix_input_ids,
            pad_masks.long(),
            image_grid_thw=rope_grid_thw,
            video_grid_thw=None,
        )
        # Dense (capture-safe) deepstack layout: zero-masked per-camera embeds
        # laid out over the full prefix length instead of bool-filtered rows.
        # _apply_deepstack then does a plain shape-static add; the old filtered
        # form fed transformers' _deepstack_process, whose bool indexing
        # (nonzero -> device/host sync) breaks CUDA-graph capture and graph-
        # breaks torch.compile. x + 0.0 == x keeps the injection numerically
        # identical (only -0.0 can flip to +0.0), verified bitwise in
        # bench/deepstack_dense_parity.py.
        img_visual_only = einops.repeat(img_masks, "b n -> b n l", l=num_patch)
        img_len = img_emb.shape[1]  # boundary-wrapped image part, [b, n*l, d]
        tail_len = embs.shape[1] - img_len
        dense_deepstack = []
        for deepstack in deepstack_embs:
            level = deepstack * img_visual_only.unsqueeze(-1).to(deepstack.dtype)
            wrapped = level.new_zeros(bsize, num_images, image_token_len, level.shape[-1])
            wrapped[:, :, 1 : 1 + num_patch] = level  # patches sit between the vision boundaries
            level = einops.rearrange(wrapped, "b n l d -> b (n l) d")
            if tail_len > 0:  # language (and query) tail stays zero
                level = torch.cat([level, level.new_zeros(bsize, tail_len, level.shape[-1])], dim=1)
            dense_deepstack.append(level.to(embed_dtype))

        result = (
            embs,
            pad_masks,
            att_masks,
            prefix_position_ids,
            full_visual_pos_masks,
            dense_deepstack,
        )
        return result

    def _build_full_position_ids(self, prefix_position_ids, prefix_pad_masks, suffix_pad_masks):
        valid_prefix_pos = prefix_position_ids.masked_fill(~prefix_pad_masks.unsqueeze(0), 0)
        prefix_offsets = valid_prefix_pos.amax(dim=(0, 2)) + 1
        suffix_1d = prefix_offsets[:, None] + torch.cumsum(suffix_pad_masks.long(), dim=1) - 1
        suffix_1d = suffix_1d.masked_fill(~suffix_pad_masks, 1)
        suffix_position_ids = suffix_1d.unsqueeze(0).expand(3, -1, -1)
        return torch.cat([prefix_position_ids, suffix_position_ids], dim=-1)

    def _embed_and_fill_prefix(self, images, img_masks, lang_tokens, lang_masks, image_grid_thw):
        """Prefix half of sample_actions: embed_prefix
        (vision tower + language embedding + mrope position ids) followed by the
        36-layer KV fill. Returns exactly what the denoise loop consumes."""
        (
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            prefix_position_ids,
            visual_pos_masks,
            deepstack_visual_embeds,
        ) = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            image_grid_thw=image_grid_thw,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        _, past_key_values = self.qwenvl_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        return prefix_pad_masks, prefix_position_ids, past_key_values

    def predict_velocity(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        prefix_position_ids=None,
        _denoise_cache: dict | None = None,
    ):
        """Predict the flow velocity using the observation prefix and request-local masks."""
        if prefix_position_ids is None:
            raise ValueError("LingBotVLAV2ForActionPrediction.predict_velocity requires Qwen3-VL prefix_position_ids.")

        time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state,
            x_t,
            timestep,
        )

        suffix_len = suffix_pad_masks.shape[1]
        prefix_len = prefix_pad_masks.shape[1]
        cache = _denoise_cache if _denoise_cache is not None else {}
        if "full_att_2d_masks" not in cache:
            batch_size = prefix_pad_masks.shape[0]
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
                batch_size,
                suffix_len,
                prefix_len,
            )
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
            full_att_2d_masks = self._block_suffix_to_queries_(full_att_2d_masks, 0, prefix_len)

            full_position_ids = self._build_full_position_ids(
                prefix_position_ids,
                prefix_pad_masks,
                suffix_pad_masks,
            )
            position_ids = full_position_ids[:, :, -suffix_len:]
            core = self.qwenvl_with_expert
            position_embeddings = core.qwenvl.model.language_model.rotary_emb(suffix_embs, position_ids)
            cache["full_att_2d_masks"] = full_att_2d_masks
            cache["position_ids"] = position_ids
            cache["position_embeddings"] = position_embeddings

        outputs_embeds, _ = self.qwenvl_with_expert.forward(
            attention_mask=cache["full_att_2d_masks"],
            position_ids=cache["position_ids"],
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
            ada_cond=time_embs if self.config.adanorm_time else None,
            position_embeddings=cache["position_embeddings"],
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        if self.config.action_fp32:
            v_t = self._fp32_linear(self.action_out_proj, suffix_out)
        else:
            if suffix_out.dtype != self.action_out_proj.weight.dtype:
                suffix_out = suffix_out.to(self.action_out_proj.weight.dtype)
            v_t = self.action_out_proj(suffix_out)
        return v_t

    @torch.no_grad()
    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        image_grid_thw,
        noise=None,
        num_steps=None,
        generator=None,
    ):
        steps = self.config.num_steps if num_steps is None else num_steps
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("num_inference_steps must be a positive integer")
        shape = (state.shape[0], self.config.chunk_size, self.config.max_action_dim)
        if noise is None:
            x_t = torch.randn(shape, device=state.device, dtype=state.dtype, generator=generator)
        else:
            if tuple(noise.shape) != shape:
                raise ValueError(f"Expected noise shape {shape}, got {tuple(noise.shape)}")
            x_t = noise.to(device=state.device, dtype=state.dtype).clone()
        masks, positions, kv = self._embed_and_fill_prefix(images, img_masks, lang_tokens, lang_masks, image_grid_thw)
        dt = torch.tensor(-1.0 / steps, device=state.device, dtype=state.dtype)
        time = torch.tensor(1.0, device=state.device, dtype=state.dtype)
        cache = {}
        for _ in range(steps):
            velocity = self.predict_velocity(state, masks, kv, x_t, time.expand(state.shape[0]), positions, cache)
            x_t = x_t + dt * velocity
            time = time + dt
        return x_t

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        expected = self.state_dict()
        loaded = set()
        skipped_heads = (
            "depth_align_head.",
            "future_depth_align_head.",
            "future_video_align_head.",
            "current_video_align_head.",
            "future_video_cls_head.",
        )
        for original_name, value in weights:
            name = original_name.removeprefix("model.")
            if name.startswith(skipped_heads):
                continue
            if name not in expected:
                raise ValueError(f"Unexpected LingBot inference weight: {original_name}")
            if name in loaded or expected[name].shape != value.shape:
                raise ValueError(f"Duplicate or incompatible LingBot weight: {original_name}")
            with torch.no_grad():
                expected[name].copy_(value.to(expected[name].device, expected[name].dtype))
            loaded.add(name)
        missing = set(expected) - loaded
        if missing:
            raise ValueError(f"Missing LingBot inference weights: {sorted(missing)[:20]}")
        return loaded
