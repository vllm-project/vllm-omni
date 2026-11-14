# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Architecture adapted from https://github.com/runwayml/praxis-1-oss (Praxis-1).
"""Action expert: a small transformer that predicts the action velocity from backbone taps."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from vllm.model_executor.layers.activation import GeluAndMul
from vllm.model_executor.layers.linear import ReplicatedLinear

from vllm_omni.diffusion.models.praxis1.layers import (
    ActionAdaRMSNorm,
    MaskCache,
    PraxisAttention,
    RotaryTable,
    SparseMask,
    TimestepEmbedder,
    action_rms_norm,
    apply_rope,
    pad_tags,
)


def _chunk_of_query(num_action_tokens: int, num_future_per_chunk: int, chunk_size: int, device) -> Tensor:
    """Chunk index of each expert token, in the order ``[proprio, future, actions]``."""
    num_chunks = max(1, num_action_tokens // chunk_size)
    chunks = torch.arange(num_chunks, device=device)
    future = chunks.repeat_interleave(num_future_per_chunk)
    actions = torch.arange(num_action_tokens, device=device) // chunk_size
    return torch.cat([chunks, future, actions])


def trunk_query_rope_ids(
    kv_rope_ids: Tensor,
    *,
    num_global_tokens: int,
    num_clean_frames: int,
    num_noised_frames: int,
    tokens_per_frame: int,
    num_action_tokens: int,
    num_future_per_chunk: int,
    chunk_size: int,
) -> Tensor:
    """``(B, Q)`` trunk rope ids of the expert tokens: the first position of their frame."""
    b = kv_rope_ids.shape[0]
    total_frames = num_clean_frames + num_noised_frames
    video = kv_rope_ids[:, num_global_tokens:]
    if video.shape[1] != total_frames * tokens_per_frame:
        raise ValueError(f"{video.shape[1]} video rope ids for {total_frames} frames of {tokens_per_frame} tokens")
    if num_action_tokens != num_noised_frames:
        raise ValueError(f"one action token per noised frame: {num_action_tokens} tokens, {num_noised_frames} frames")
    frame_ids = video.view(b, total_frames, tokens_per_frame).amin(dim=-1)
    noised_ids = frame_ids[:, num_clean_frames:]
    q_chunk = _chunk_of_query(num_action_tokens, num_future_per_chunk, chunk_size, kv_rope_ids.device)
    n_prefix = q_chunk.numel() - num_action_tokens
    q_frame = torch.cat(
        [(q_chunk[:n_prefix] * chunk_size).clamp(min=0), torch.arange(num_action_tokens, device=kv_rope_ids.device)]
    )
    return noised_ids.gather(1, q_frame[None].expand(b, -1))


def trunk_masks(
    *,
    num_action_tokens: int,
    num_future_per_chunk: int,
    chunk_size: int,
    num_clean_frames: int,
    num_noised_frames: int,
    tokens_per_frame: int,
    window_size: int,
    num_global_tokens: int,
    device: torch.device,
    include_noised: bool = True,
) -> tuple[SparseMask, SparseMask]:
    """Cross mask (expert tokens over global + video tokens) and self mask (within a chunk).

    An expert token of chunk ``c`` sees every global token, the clean frames
    within ``window_size`` frames before its chunk start and, unless
    ``clean_only_conditioning``, the noised tokens of its own chunk.
    """
    if num_clean_frames != num_noised_frames - chunk_size:
        raise ValueError("Praxis requires complete observed chunks before the current chunk")
    chunks = _chunk_of_query(num_action_tokens, num_future_per_chunk, chunk_size, device)
    q_len = chunks.numel()
    k_len = num_global_tokens + (num_clean_frames + num_noised_frames) * tokens_per_frame
    q_chunk = pad_tags(chunks, q_len)
    q_start = pad_tags(chunks * chunk_size, q_len)
    clean_frames = torch.arange(num_clean_frames, device=device).repeat_interleave(tokens_per_frame)
    noised_frames = torch.arange(num_noised_frames, device=device).repeat_interleave(tokens_per_frame)
    globals_ = torch.zeros(num_global_tokens, device=device, dtype=torch.int32)
    k_kind = pad_tags(torch.cat([globals_, torch.ones_like(clean_frames), torch.full_like(noised_frames, 2)]), k_len)
    k_frame = pad_tags(torch.cat([globals_, clean_frames, noised_frames]), k_len)
    k_chunk = pad_tags(
        torch.cat([torch.full_like(globals_, -1), torch.full_like(clean_frames, -1), noised_frames // chunk_size]),
        k_len,
    )

    def cross(b: Tensor | None, h: Tensor | None, qi: Tensor, ki: Tensor) -> Tensor:
        kind, frame, start = k_kind[ki], k_frame[ki], q_start[qi]
        clean = (kind == 1) & (frame < start) & (frame >= start - window_size)
        rule = (kind == 0) | clean
        if include_noised:
            rule = rule | ((kind == 2) & (k_chunk[ki] == q_chunk[qi]))
        return (q_chunk[qi] >= 0) & rule

    def self_(b: Tensor | None, h: Tensor | None, qi: Tensor, ki: Tensor) -> Tensor:
        return (q_chunk[qi] >= 0) & (q_chunk[qi] == q_chunk[ki])

    return SparseMask(cross, q_len, k_len, device), SparseMask(self_, q_len, q_len, device)


class ActionTauEncoder(nn.Module):
    """Action embedding with the flow time at the input."""

    def __init__(self, action_patch_dim: int, dim: int) -> None:
        super().__init__()
        self.w1 = ReplicatedLinear(action_patch_dim, dim, return_bias=False)
        self.tau = TimestepEmbedder(dim)
        self.w2 = ReplicatedLinear(2 * dim, dim, return_bias=False)
        self.w3 = ReplicatedLinear(dim, dim, return_bias=False)

    def forward(self, actions: Tensor, tau: Tensor) -> Tensor:
        """``actions (B, T, action_patch_dim)``, ``tau (B, T)`` -> ``(B, T, dim)``."""
        a = self.w1(actions)
        t = self.tau(tau).to(a.dtype)
        if t.shape[:2] != a.shape[:2]:
            raise ValueError(f"tau shape {tuple(t.shape)} != {tuple(a.shape)}")
        return self.w3(F.silu(self.w2(torch.cat([a, t], dim=-1))))


class ActionExpertBlock(nn.Module):
    """One expert layer: attention (cross or self), then a GEGLU MLP."""

    def __init__(
        self, *, dim: int, num_heads: int, head_dim: int, is_cross: bool, kv_dim: int, attention_impl: str
    ) -> None:
        super().__init__()
        self.is_cross = is_cross
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner = num_heads * head_dim
        self.adaln_attn = ActionAdaRMSNorm(dim)
        self.to_q = ReplicatedLinear(dim, inner, bias=False, return_bias=False)
        self.to_kv = ReplicatedLinear(kv_dim if is_cross else dim, 2 * inner, bias=False, return_bias=False)
        self.to_out = ReplicatedLinear(inner, dim, bias=False, return_bias=False)
        self.adaln_mlp = ActionAdaRMSNorm(dim)
        hidden = 4 * dim
        self.mlp_in = ReplicatedLinear(dim, 2 * hidden, return_bias=False)
        self.mlp_out = ReplicatedLinear(hidden, dim, return_bias=False)
        self.activation = GeluAndMul(approximate="tanh")
        self.attention = PraxisAttention(num_heads, num_heads, head_dim, cross=is_cross, attention_impl=attention_impl)

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        *,
        kv_source: Tensor | None,
        mask: SparseMask,
        q_table: Tensor,
        k_table: Tensor,
    ) -> Tensor:
        b, q_len, _ = x.shape
        h = self.adaln_attn(x, cond)
        q = self.to_q(h)
        kv_in = kv_source if self.is_cross else h
        if kv_in is None:
            raise ValueError("a cross-attention block needs kv_source")
        k, v = self.to_kv(kv_in).chunk(2, dim=-1)

        def split(t: Tensor) -> Tensor:
            return t.view(b, t.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        q = action_rms_norm(apply_rope(split(q), q_table))
        k = action_rms_norm(apply_rope(split(k), k_table))
        attn = self.attention(q, k, split(v), mask)
        x = x + self.to_out(attn.transpose(1, 2).reshape(b, q_len, -1))
        return x + self.mlp_out(self.activation(self.mlp_in(self.adaln_mlp(x, cond))))


class ActionExpert(nn.Module):
    """Action transformer that reads video backbone taps."""

    def __init__(self, cfg: dict, attention_impl: str = "flex") -> None:
        super().__init__()
        dim, num_layers = cfg["dim"], cfg["num_layers"]
        proprio_dim, trunk_dim = cfg["proprio_patch_dim"], cfg["trunk_dim"]
        rope = cfg["rope"]
        if num_layers % 2:
            raise ValueError(f"num_layers must be even, got {num_layers}")
        self.rope_table = RotaryTable(
            cfg["head_dim"], int(rope["effective_dim"]), int(rope["base"]), int(rope["table_size"])
        )
        self.dim = dim
        self.num_cross_layers = num_layers // 2
        self.num_future_per_chunk = cfg["num_future_per_chunk"]
        self.action_patch_dim = cfg["action_patch_dim"]
        self.clean_only_conditioning = bool(cfg["clean_only_conditioning"])
        self.action_encoder = ActionTauEncoder(self.action_patch_dim, dim)
        self.timestep_embedder = TimestepEmbedder(dim)
        self.future_tokens = nn.Parameter(torch.zeros(self.num_future_per_chunk, dim))
        self.future_chunk_embed = nn.Embedding(cfg["max_chunks"], dim)
        self.proprio_token = nn.Parameter(torch.zeros(1, dim))
        self.embed_proprio = ReplicatedLinear(proprio_dim, dim, return_bias=False) if proprio_dim > 0 else None
        self.tap_proj = ReplicatedLinear(trunk_dim, dim, bias=False, return_bias=False)
        self.text_proj = ReplicatedLinear(trunk_dim, dim, bias=False, return_bias=False)
        self.tap_norm_scale: nn.Parameter | None = nn.Parameter(torch.ones(trunk_dim)) if cfg["tap_norm"] else None
        self.blocks = nn.ModuleList(
            [
                ActionExpertBlock(
                    dim=dim,
                    num_heads=cfg["num_heads"],
                    head_dim=cfg["head_dim"],
                    is_cross=i % 2 == 0,
                    kv_dim=dim,
                    attention_impl=attention_impl,
                )
                for i in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-06)
        self.head = ReplicatedLinear(dim, self.action_patch_dim, return_bias=False)
        self._masks = MaskCache()

    def _prepare_queries(
        self,
        noised_action_latents: Tensor,
        action_timesteps: Tensor,
        proprio_latents: Tensor | None,
        chunk_size: int,
    ) -> tuple[Tensor, Tensor]:
        """Build proprio/future/action query tokens and their time conditioning."""
        b, t_lat, _ = noised_action_latents.shape
        if b != 1:
            raise ValueError(f"the trunk-rope layout runs one sample, got {b}")
        device = noised_action_latents.device
        num_chunks = max(1, t_lat // chunk_size)
        if action_timesteps.ndim == 1:
            a_t = action_timesteps[:, None].expand(-1, t_lat)
        else:
            if action_timesteps.shape[1] != t_lat:
                raise ValueError(f"action_timesteps has {action_timesteps.shape[1]} entries for {t_lat} action tokens")
            a_t = action_timesteps
        act = self.action_encoder(noised_action_latents, a_t)
        if num_chunks > self.future_chunk_embed.num_embeddings:
            raise ValueError(
                f"{num_chunks} chunks exceed the {self.future_chunk_embed.num_embeddings} chunk embeddings"
            )
        fut = self.future_tokens[None, None].expand(b, num_chunks, -1, -1)
        fut = fut + self.future_chunk_embed.weight[:num_chunks][None, :, None, :].to(fut.dtype)
        fut = fut.reshape(b, num_chunks * self.num_future_per_chunk, self.dim)
        prop = self.proprio_token[None].expand(b, num_chunks, -1).to(act.dtype)
        if self.embed_proprio is not None and proprio_latents is not None:
            idx = torch.arange(num_chunks, device=device) * chunk_size
            prop = prop + self.embed_proprio(proprio_latents[:, idx].to(prop.dtype))
        x = torch.cat([prop, fut.to(act.dtype), act], dim=1)
        q_chunk = _chunk_of_query(t_lat, self.num_future_per_chunk, chunk_size, device)
        tok_frame = (q_chunk * chunk_size).clamp(min=0, max=t_lat - 1)
        cond = self.timestep_embedder(a_t.gather(1, tok_frame[None].expand(b, -1))).to(x.dtype)
        return x, cond

    def _project_trunk_tap(self, features: Tensor, text: Tensor | None) -> Tensor:
        """Normalize/project one video tap and prepend optional global conditioning."""
        if self.tap_norm_scale is not None:
            features = (action_rms_norm(features) * self.tap_norm_scale).to(features.dtype)
        features = self.tap_proj(features)
        return features if text is None else torch.cat([text.to(features.dtype), features], dim=1)

    def forward_trunk_rope(
        self,
        *,
        noised_action_latents: Tensor,
        action_timesteps: Tensor,
        trunk_features: Sequence[Tensor],
        global_tokens: Tensor | None,
        proprio_latents: Tensor | None,
        kv_rope_ids: Tensor,
        chunk_size: int,
        num_clean_frames: int,
        num_noised_frames: int,
        tokens_per_frame: int,
        window_size: int,
    ) -> Tensor:
        """Predict the action velocity ``(1, T, action_patch_dim)``."""
        if len(trunk_features) != self.num_cross_layers:
            raise ValueError(f"expected {self.num_cross_layers} taps, got {len(trunk_features)}")
        b, t_lat, _ = noised_action_latents.shape
        device = noised_action_latents.device
        x, cond = self._prepare_queries(noised_action_latents, action_timesteps, proprio_latents, chunk_size)
        num_global = 0 if global_tokens is None else global_tokens.shape[1]
        key = (
            t_lat,
            chunk_size,
            num_clean_frames,
            num_noised_frames,
            tokens_per_frame,
            window_size,
            num_global,
            str(device),
        )
        masks = self._masks.get(
            key,
            lambda: trunk_masks(
                num_action_tokens=t_lat,
                num_future_per_chunk=self.num_future_per_chunk,
                chunk_size=chunk_size,
                num_clean_frames=num_clean_frames,
                num_noised_frames=num_noised_frames,
                tokens_per_frame=tokens_per_frame,
                window_size=window_size,
                num_global_tokens=num_global,
                device=device,
                include_noised=not self.clean_only_conditioning,
            ),
        )
        assert isinstance(masks, tuple)
        cross_mask, self_mask = masks
        kv_ids = kv_rope_ids.to(device).long()
        expected_kv = num_global + (num_clean_frames + num_noised_frames) * tokens_per_frame
        if tuple(kv_ids.shape) != (b, expected_kv):
            raise ValueError(f"kv_rope_ids {tuple(kv_ids.shape)} != {(b, expected_kv)}")
        q_ids = trunk_query_rope_ids(
            kv_ids,
            num_global_tokens=num_global,
            num_clean_frames=num_clean_frames,
            num_noised_frames=num_noised_frames,
            tokens_per_frame=tokens_per_frame,
            num_action_tokens=t_lat,
            num_future_per_chunk=self.num_future_per_chunk,
            chunk_size=chunk_size,
        )
        q_table, k_table = self.rope_table(q_ids), self.rope_table(kv_ids)
        text_kv = None if global_tokens is None else self.text_proj(global_tokens)
        cross_idx = 0
        for block in self.blocks:
            assert isinstance(block, ActionExpertBlock)
            if block.is_cross:
                kv = self._project_trunk_tap(trunk_features[cross_idx], text_kv)
                x = block(x, cond, kv_source=kv, mask=cross_mask, q_table=q_table, k_table=k_table)
                cross_idx += 1
            else:
                x = block(x, cond, kv_source=None, mask=self_mask, q_table=q_table, k_table=q_table)
        return self.head(self.norm_out(x[:, -t_lat:])).view(b, t_lat, self.action_patch_dim)


def run_expert_on_taps(
    expert: ActionExpert,
    taps: Sequence[Tensor],
    embeddings: Tensor,
    rope_ids: Tensor,
    *,
    noised_span: tuple[int, int],
    clean_span: tuple[int, int] | None,
    n_chunks: int,
    frames_per_chunk: int,
    tokens_per_frame: int,
    noised_actions: Tensor,
    action_t_per_chunk: Tensor,
    proprio_per_chunk: Tensor,
) -> Tensor:
    """Run the expert on the taps of one chunk-causal encoder forward."""
    device, dtype = embeddings.device, embeddings.dtype
    n_noised = frames_per_chunk * n_chunks
    n_clean = frames_per_chunk * (n_chunks - 1)
    # The encoder packs [registers, anchor, text] contiguously before the noised span.
    n_global = noised_span[0]

    def video_tokens(x: Tensor) -> Tensor:
        noised = x.narrow(1, *noised_span)
        return noised if clean_span is None else torch.cat([x.narrow(1, *clean_span), noised], dim=1)

    return expert.forward_trunk_rope(
        noised_action_latents=noised_actions.to(device, dtype),
        action_timesteps=action_t_per_chunk.to(device).repeat_interleave(frames_per_chunk, dim=1),
        trunk_features=[video_tokens(f) for f in taps],
        global_tokens=embeddings[:, :n_global],
        proprio_latents=proprio_per_chunk.to(device, dtype).repeat_interleave(frames_per_chunk, dim=1),
        kv_rope_ids=torch.cat([rope_ids[:, :n_global], video_tokens(rope_ids)], dim=1),
        chunk_size=frames_per_chunk,
        num_clean_frames=n_clean,
        num_noised_frames=n_noised,
        tokens_per_frame=tokens_per_frame,
        window_size=n_clean + n_noised,
    )
