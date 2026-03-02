"""DiNa-LRM model code — fully self-contained, no diffusion_rm dependency.

This module is a faithful copy of the model components from the diffusion-rm
repository: https://github.com/HKUST-C4G/diffusion-rm
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.layer import Attention

logger = init_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Config helper
# ══════════════════════════════════════════════════════════════════════════════


def load_rm_config(config_path: str):
    """Load a YAML/JSON config file and return an OmegaConf DictConfig."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return OmegaConf.create(raw)


# ══════════════════════════════════════════════════════════════════════════════
# Text-encoding utilities  (from diffusion_rm/models/sd3_rm.py)
# ══════════════════════════════════════════════════════════════════════════════


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length: int,
    prompt=None,
    num_images_per_prompt: int = 1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    elif text_input_ids is None:
        raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    elif text_input_ids is None:
        raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    return prompt_embeds, pooled_prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt,
    max_sequence_length: int,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    """Encode a prompt with SD3's three text encoders (CLIP ×2 + T5)."""
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for i, (tokenizer, text_encoder) in enumerate(zip(clip_tokenizers, clip_text_encoders)):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=text_input_ids_list[i] if text_input_ids_list else None,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[-1] if text_input_ids_list else None,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = F.pad(
        clip_prompt_embeds,
        (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
    return prompt_embeds, pooled_prompt_embeds


# ══════════════════════════════════════════════════════════════════════════════
# Reward Head  (from diffusion_rm/models/reward_head.py)
# ══════════════════════════════════════════════════════════════════════════════


class FiLMLayerAdapter(nn.Module):
    """FiLM modulation adapter: projects a feature token conditioned on t_emb."""

    def __init__(
        self,
        in_dim: int,
        emb_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_proj_in: bool = True,
    ):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim) if (in_dim != hidden_dim or use_proj_in) else nn.Identity()
        self.layer_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.cond_mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
        self.proj = nn.Linear(hidden_dim, output_dim)
        nn.init.zeros_(self.cond_mlp[-1].weight)
        nn.init.zeros_(self.cond_mlp[-1].bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        style = self.cond_mlp(t_emb)
        gamma, beta = style.chunk(2, dim=-1)
        x = x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        x = x + self.layer_embed
        return self.proj(x)


class CrossAttnBlockVGated(nn.Module):
    """Cross-attention block with token-wise V-gating on value projections."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        use_norm: bool = True,
        use_text: bool = True,
        dropout: float = 0.0,
        use_v_gating: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_text = use_text
        self.use_v_gating = use_v_gating

        self.norm_q = nn.RMSNorm(dim) if use_norm else None
        self.norm_v = nn.RMSNorm(dim) if use_norm else None
        self.norm_t = nn.RMSNorm(dim) if (use_norm and use_text) else None

        self.to_q = nn.Linear(dim, dim)
        self.to_k_vis = nn.Linear(dim, dim)
        self.to_v_vis = nn.Linear(dim, dim)

        if use_text:
            self.to_k_text = nn.Linear(dim, dim)
            self.to_v_text = nn.Linear(dim, dim)
        else:
            self.to_k_text = None
            self.to_v_text = None

        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

        if self.use_v_gating:
            self.gate_vis = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))
            nn.init.zeros_(self.gate_vis[-1].weight)
            nn.init.zeros_(self.gate_vis[-1].bias)

            if self.use_text:
                self.gate_text = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))
                nn.init.zeros_(self.gate_text[-1].weight)
                nn.init.zeros_(self.gate_text[-1].bias)
            else:
                self.gate_text = None
        else:
            self.gate_vis = None
            self.gate_text = None

        softmax_scale = 1.0 / (self.head_dim**0.5)
        self.attn_vis = Attention(
            num_heads=num_heads,
            head_size=self.head_dim,
            softmax_scale=softmax_scale,
            causal=False,
            num_kv_heads=num_heads,
        )
        self.attn_text = (
            Attention(
                num_heads=num_heads,
                head_size=self.head_dim,
                softmax_scale=softmax_scale,
                causal=False,
                num_kv_heads=num_heads,
            )
            if use_text
            else None
        )

    def _to_multihead(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape [B, seq, dim] → [B, seq, num_heads, head_dim] for Attention."""
        return x.unflatten(-1, (self.num_heads, self.head_dim))

    def forward(
        self,
        queries: torch.Tensor,
        context_visual: torch.Tensor,
        context_text: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = queries
        if self.norm_q is not None:
            queries = self.norm_q(queries)
        if self.norm_v is not None:
            context_visual = self.norm_v(context_visual)
        if self.norm_t is not None and context_text is not None:
            context_text = self.norm_t(context_text)

        q = self._to_multihead(self.to_q(queries))
        k_vis = self._to_multihead(self.to_k_vis(context_visual))
        v_vis = self._to_multihead(self.to_v_vis(context_visual))
        if self.use_v_gating and self.gate_vis is not None:
            v_vis = v_vis * torch.sigmoid(self.gate_vis(context_visual)).unsqueeze(-1)
        h = self.attn_vis(q, k_vis, v_vis).flatten(-2)

        if self.use_text and context_text is not None and self.to_k_text is not None:
            k_text = self._to_multihead(self.to_k_text(context_text))
            v_text = self._to_multihead(self.to_v_text(context_text))
            if self.use_v_gating and self.gate_text is not None:
                v_text = v_text * torch.sigmoid(self.gate_text(context_text)).unsqueeze(-1)
            h = h + self.attn_text(q, k_text, v_text).flatten(-2)

        return residual + self.to_out(h)


class RewardHead(nn.Module):
    """Cross-attention reward head.

    Aggregates multi-scale visual and text features from the backbone into a
    single scalar reward value using FiLM adapters, two cross-attention blocks,
    and a small FFN.
    """

    def __init__(
        self,
        token_dim: int,
        width: int = -1,
        out_dim: int = 1,
        n_visual_heads: int = 1,
        n_text_heads: int = 1,
        num_queries: int = 4,
        num_attn_heads: int = 8,
        dropout: float = 0.0,
        t_embed_dim: int = -1,
        use_proj_in: bool = False,
        **kwargs,
    ):
        super().__init__()
        if width == -1:
            width = token_dim
        feature_out_dim = width // 4

        self.layer_adapters_visual = nn.ModuleList(
            [
                FiLMLayerAdapter(
                    in_dim=token_dim,
                    emb_dim=t_embed_dim if t_embed_dim > 0 else width,
                    hidden_dim=width,
                    output_dim=feature_out_dim,
                    use_proj_in=use_proj_in,
                )
                for _ in range(n_visual_heads)
            ]
        )
        self.layer_adapters_text = nn.ModuleList(
            [
                FiLMLayerAdapter(
                    in_dim=token_dim,
                    emb_dim=t_embed_dim if t_embed_dim > 0 else width,
                    hidden_dim=width,
                    output_dim=feature_out_dim,
                    use_proj_in=use_proj_in,
                )
                for _ in range(n_text_heads)
            ]
        )

        self.agg_visual = nn.Linear(n_visual_heads * feature_out_dim, width) if n_visual_heads > 0 else None
        self.agg_text = nn.Linear(n_text_heads * feature_out_dim, width) if n_text_heads > 0 else None

        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, width) * 0.02)

        self.attn1 = CrossAttnBlockVGated(
            dim=width,
            num_heads=num_attn_heads,
            use_norm=True,
            use_text=(n_text_heads > 0),
            dropout=dropout,
            use_v_gating=True,
        )
        self.attn2 = CrossAttnBlockVGated(
            dim=width,
            num_heads=num_attn_heads,
            use_norm=True,
            use_text=False,
            dropout=dropout,
            use_v_gating=False,
        )

        self.norm_ff = nn.RMSNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(width, out_dim)

    def _build_view_tokens(
        self,
        visual_features: list[torch.Tensor],
        text_features: list[torch.Tensor] | None,
        t_emb: torch.Tensor,
    ):
        assert len(visual_features) == len(self.layer_adapters_visual)
        out_v = torch.cat(
            [a(vf, t_emb) for a, vf in zip(self.layer_adapters_visual, visual_features)],
            dim=-1,
        )
        visual_out = self.agg_visual(out_v)

        if not text_features or len(self.layer_adapters_text) == 0:
            return visual_out, None

        assert len(text_features) == len(self.layer_adapters_text)
        out_t = torch.cat(
            [a(tf, t_emb) for a, tf in zip(self.layer_adapters_text, text_features)],
            dim=-1,
        )
        text_out = self.agg_text(out_t)
        return visual_out, text_out

    def forward(
        self,
        visual_features: list[torch.Tensor],
        text_features: list[torch.Tensor] | None,
        t_embed: torch.Tensor | None = None,
        hw=None,
    ) -> torch.Tensor:
        assert t_embed is not None
        visual_out, text_out = self._build_view_tokens(visual_features, text_features, t_embed)
        B = visual_out.size(0)
        queries = self.query_tokens.expand(B, -1, -1)
        queries = self.attn1(queries, visual_out, text_out)
        queries = self.attn2(queries, visual_out, None)
        queries = queries + self.ff(self.norm_ff(queries))
        return self.head(queries).mean(dim=1)  # (B, out_dim)

    def forward_ensemble(
        self,
        visual_features_per_t: list[list[torch.Tensor]],
        text_features_per_t: list[list[torch.Tensor]] | None,
        t_embed_per_t: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-timestep ensemble: concat tokens across timesteps."""
        B, K, _ = t_embed_per_t.shape
        visual_all, text_all = [], []
        for i in range(K):
            v_out, t_out = self._build_view_tokens(
                visual_features_per_t[i],
                text_features_per_t[i] if text_features_per_t else None,
                t_embed_per_t[:, i, :],
            )
            visual_all.append(v_out)
            if t_out is not None:
                text_all.append(t_out)

        visual_cat = torch.cat(visual_all, dim=1)
        text_cat = torch.cat(text_all, dim=1) if text_all else None
        queries = self.query_tokens.expand(B, -1, -1)
        queries = self.attn1(queries, visual_cat, text_cat)
        queries = self.attn2(queries, visual_cat, None)
        queries = queries + self.ff(self.norm_ff(queries))
        return self.head(queries).mean(dim=1)


# ══════════════════════════════════════════════════════════════════════════════
# SD3 Backbone + Reward Model  (from diffusion_rm/models/sd3_rm.py)
# ══════════════════════════════════════════════════════════════════════════════


class SD3Backbone(nn.Module):
    """First N transformer layers of the SD3 transformer, used as feature extractor."""

    def __init__(self, transformer, config_model):
        super().__init__()
        self.pos_embed = transformer.pos_embed
        self.time_text_embed = transformer.time_text_embed
        self.context_embedder = transformer.context_embedder
        self.transformer_blocks = nn.ModuleList(transformer.transformer_blocks[: config_model.num_transformer_layers])
        self.visual_head_idx = config_model.visual_head_idx
        self.text_head_idx = config_model.text_head_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        unpatched: bool = False,
    ):
        hidden_states = self.pos_embed(hidden_states)
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        hidden_states_list = [hidden_states] if self.visual_head_idx[0] == 0 else []
        encoder_hidden_states_list = [encoder_hidden_states] if self.text_head_idx[0] == 0 else []

        for idx, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
            )
            if idx + 1 in self.visual_head_idx:
                hidden_states_list.append(hidden_states)
            if idx + 1 in self.text_head_idx:
                encoder_hidden_states_list.append(encoder_hidden_states)

        return temb, hidden_states_list, encoder_hidden_states_list


class SD3RewardModel(nn.Module):
    """SD3.5-M backbone (first N layers) + RewardHead → scalar reward."""

    def __init__(self, pipeline, config_model, device, dtype):
        super().__init__()
        from peft import LoraConfig, get_peft_model

        text_encoder_1 = pipeline.text_encoder
        text_encoder_2 = pipeline.text_encoder_2
        text_encoder_3 = pipeline.text_encoder_3
        for enc in [text_encoder_1, text_encoder_2, text_encoder_3]:
            enc.requires_grad_(False)

        self.text_encoders = [text_encoder_1, text_encoder_2, text_encoder_3]
        self.tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

        self.backbone = SD3Backbone(
            transformer=pipeline.transformer,
            config_model=config_model,
        )

        if config_model.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif config_model.use_lora and config_model.lora_config is not None:
            target_modules = [
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "add_q_proj",
                "add_k_proj",
                "add_v_proj",
                "to_add_out",
            ]
            n_layers = config_model.num_transformer_layers
            exclude_modules = [
                f"transformer_blocks.{n_layers - 1}.attn.add_q_proj",
                f"transformer_blocks.{n_layers - 1}.attn.add_k_proj",
                f"transformer_blocks.{n_layers - 1}.attn.add_v_proj",
                f"transformer_blocks.{n_layers - 1}.attn.to_add_out",
            ]
            if config_model.use_text_features and config_model.text_head_idx[-1] == n_layers:
                exclude_modules = None

            lora_cfg = LoraConfig(
                r=config_model.lora_config.r,
                lora_alpha=config_model.lora_config.lora_alpha,
                init_lora_weights=config_model.lora_config.init_lora_weights,
                target_modules=target_modules,
                exclude_modules=exclude_modules,
            )
            self.backbone = get_peft_model(self.backbone, lora_cfg)
            self.backbone.to(device, dtype=dtype)

        backbone_dim = pipeline.transformer.inner_dim
        self.reward_head = RewardHead(
            token_dim=backbone_dim,
            n_visual_heads=len(config_model.visual_head_idx),
            n_text_heads=len(config_model.text_head_idx),
            t_embed_dim=backbone_dim,
            **config_model.reward_head,
        )
        self.reward_head.to(device, dtype=dtype)

        self.use_logistic = getattr(config_model, "use_logistic", False)
        if self.use_logistic:
            self.eta1 = 2.0
            self.eta2 = -2.0
            self.eta3 = nn.Parameter(torch.tensor(0.0))
            self.eta4 = nn.Parameter(torch.tensor(0.15))

    def _logistic(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_logistic:
            return x
        exp_pow = -(x - self.eta3) / (torch.abs(self.eta4) + 1e-6)
        return (self.eta1 - self.eta2) / (1 + torch.exp(exp_pow)) + self.eta2

    def encode_prompt(self, prompts):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                self.text_encoders,
                self.tokenizers,
                prompts,
                max_sequence_length=128,
            )
        return {
            "encoder_hidden_states": prompt_embeds.to(self.text_encoders[0].device),
            "pooled_projections": pooled_prompt_embeds.to(self.text_encoders[0].device),
        }

    def forward(
        self,
        latents: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor | None,
        timesteps: torch.LongTensor,
        **kwargs,
    ) -> torch.Tensor:
        temb, hidden_states_list, enc_hidden_states_list = self.backbone(
            hidden_states=latents,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timesteps,
        )
        reward = self.reward_head(
            visual_features=hidden_states_list,
            text_features=enc_hidden_states_list,
            t_embed=temb,
            hw=latents.shape[-2:],
        )
        return self._logistic(reward)
