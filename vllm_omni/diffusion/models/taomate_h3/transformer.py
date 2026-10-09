# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax-H3 DiT with TaoMate streaming attention in the 50 main blocks."""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel

from .attention import TaoMateH3StreamingAttention

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)

_EMBED_PARAMETERS = frozenset(
    {
        "x",
        "audio_x",
        "text_embeddings_selected",
        "unique_timesteps",
        "img_pos",
        "audio_pos",
        "text_pos",
        "refiner_cu_seqlens",
        "refiner_max_seqlen",
        "seq_len",
        "device",
        "local_span",
        "num_requests",
    }
)


@dataclass(frozen=True)
class LocalEmbedPlan:
    """Precomputed row selections of the sequence-parallel embedding step.

    The upstream embedding narrows the packed rows to this rank's slice with
    boolean masks, which is a host synchronization per forward and cannot be
    captured in a CUDA graph. The plan holds the same selections as fixed
    index tensors for one document layout (positions and local span), so a
    forward that runs with the plan installed performs no data-dependent
    indexing. ``fingerprint`` identifies the layout the plan was built for.
    """

    seq_len: int
    local_span: tuple[int, int]
    fingerprint: tuple[int, ...]
    img_global_pos: torch.Tensor
    audio_global_pos: torch.Tensor
    img_local_pos: torch.Tensor
    audio_local_pos: torch.Tensor
    text_local_pos: torch.Tensor
    text_local_indices: torch.Tensor | None

    @staticmethod
    def build(
        *,
        img_pos: torch.Tensor,
        audio_pos: torch.Tensor,
        text_pos: torch.Tensor,
        seq_len: int,
        local_span: tuple[int, int],
        device: torch.device,
    ) -> LocalEmbedPlan:
        """Mirror the upstream mask logic once, on the host, for one layout."""
        img = img_pos.detach().to("cpu", torch.long).view(-1)
        audio = audio_pos.detach().to("cpu", torch.long).view(-1)
        text = text_pos.detach().to("cpu", torch.long).view(-1)
        local_start, local_len = int(local_span[0]), int(local_span[1])
        local_end = local_start + local_len
        fingerprint = (int(seq_len), local_start, local_len, int(img.numel()), int(audio.numel()), int(text.numel()))
        if local_len != int(seq_len):
            img_mask = (img >= local_start) & (img < local_end)
            audio_mask = (audio >= local_start) & (audio < local_end)
            text_mask = (text >= local_start) & (text < local_end)
            img_global = img[img_mask]
            audio_global = audio[audio_mask]
            img_local = img_global - local_start
            audio_local = audio_global - local_start
            text_local = text[text_mask] - local_start
            text_indices: torch.Tensor | None = torch.nonzero(text_mask, as_tuple=False).view(-1)
        else:
            img_global = img_local = img
            audio_global = audio_local = audio
            text_local = text
            text_indices = None
        return LocalEmbedPlan(
            seq_len=int(seq_len),
            local_span=(local_start, local_len),
            fingerprint=fingerprint,
            img_global_pos=img_global.to(device),
            audio_global_pos=audio_global.to(device),
            img_local_pos=img_local.to(device),
            audio_local_pos=audio_local.to(device),
            text_local_pos=text_local.to(device),
            text_local_indices=None if text_indices is None else text_indices.to(device),
        )


_LOCAL_EMBED_PLAN: ContextVar[LocalEmbedPlan | None] = ContextVar("taomate_h3_local_embed_plan", default=None)


@contextmanager
def local_embed_plan(plan: LocalEmbedPlan | None) -> Iterator[None]:
    """Install ``plan`` for the forwards run inside the block."""
    token = _LOCAL_EMBED_PLAN.set(plan)
    try:
        yield
    finally:
        _LOCAL_EMBED_PLAN.reset(token)


class TaoMateH3DiTModel(MiniMaxH3DiTModel):
    """Checkpoint-compatible H3 DiT whose block attention can stream over clean KV.

    The token refiner keeps the upstream attention: TaoMate never caches text
    K/V, and prompt refinement runs on replicated rows before sequence
    parallel sharding.
    """

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        diffusers_weights: bool | None = None,
    ) -> None:
        super().__init__(
            od_config,
            quant_config,
            diffusers_weights=diffusers_weights,
            attention_cls=TaoMateH3StreamingAttention,
        )
        for index, block in enumerate(self.blocks):
            block.attn.layer_name = f"blocks.{index}.attn"
        # The graph-safe embedding below re-implements the upstream method; if
        # upstream changes its contract the plan path is switched off rather
        # than trusted.
        # (A distinct name: the API docs render this assignment and cross-link
        # bare identifiers; "parameters" is an ambiguous anchor there.)
        embed_signature = set(inspect.signature(MiniMaxH3DiTModel._embed).parameters) - {"self"}
        self.local_embed_plan_supported = embed_signature == _EMBED_PARAMETERS
        if not self.local_embed_plan_supported:
            logger.warning(
                "TaoMate-H3: MiniMaxH3DiTModel._embed signature changed; graph-safe embedding plans are disabled"
            )

    @property
    def streaming_local_heads(self) -> int:
        """Heads per rank after the Ulysses all-to-all (the persistent KV head count)."""
        return int(self.blocks[0].attn.num_sp_heads)

    def plan_local_embed(
        self,
        *,
        img_pos: torch.Tensor,
        audio_pos: torch.Tensor,
        text_pos: torch.Tensor,
        seq_len: int,
        device: torch.device,
    ) -> LocalEmbedPlan | None:
        """Build the embedding plan of one layout for this rank, or ``None`` if unsupported."""
        if not self.local_embed_plan_supported:
            return None
        return LocalEmbedPlan.build(
            img_pos=img_pos,
            audio_pos=audio_pos,
            text_pos=text_pos,
            seq_len=int(seq_len),
            local_span=self._rope_local_span(int(seq_len)),
            device=device,
        )

    def _embed(  # type: ignore[override]
        self,
        *,
        x: torch.Tensor,
        audio_x: torch.Tensor,
        text_embeddings_selected: torch.Tensor,
        unique_timesteps: torch.Tensor,
        img_pos: torch.Tensor,
        audio_pos: torch.Tensor,
        text_pos: torch.Tensor,
        refiner_cu_seqlens: torch.Tensor,
        refiner_max_seqlen: int,
        seq_len: int,
        device: torch.device,
        local_span: tuple[int, int],
        num_requests: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        plan = _LOCAL_EMBED_PLAN.get()
        if plan is None or not self.local_embed_plan_supported:
            return super()._embed(
                x=x,
                audio_x=audio_x,
                text_embeddings_selected=text_embeddings_selected,
                unique_timesteps=unique_timesteps,
                img_pos=img_pos,
                audio_pos=audio_pos,
                text_pos=text_pos,
                refiner_cu_seqlens=refiner_cu_seqlens,
                refiner_max_seqlen=refiner_max_seqlen,
                seq_len=seq_len,
                device=device,
                local_span=local_span,
                num_requests=num_requests,
            )
        local_start, local_len = int(local_span[0]), int(local_span[1])
        expected = (
            int(seq_len),
            local_start,
            local_len,
            int(img_pos.numel()),
            int(audio_pos.numel()),
            int(text_pos.numel()),
        )
        if plan.fingerprint[3:] != expected[3:]:
            raise RuntimeError(
                f"TaoMate-H3 local embedding plan {plan.fingerprint} does not describe this document {expected}"
            )
        if plan.fingerprint[:3] != expected[:3]:
            # The rank's row span is resolved from the forward context; a plan
            # built under another context is not wrong, only unusable here.
            # Take the upstream path (graph capture then fails loudly and the
            # teacher falls back to eager execution).
            if not getattr(self, "_taomate_span_warned", False):
                self._taomate_span_warned = True
                logger.warning(
                    "TaoMate-H3 local embedding plan span %s differs from the forward span %s; using the upstream "
                    "embedding",
                    plan.fingerprint[:3],
                    expected[:3],
                )
            return super()._embed(
                x=x,
                audio_x=audio_x,
                text_embeddings_selected=text_embeddings_selected,
                unique_timesteps=unique_timesteps,
                img_pos=img_pos,
                audio_pos=audio_pos,
                text_pos=text_pos,
                refiner_cu_seqlens=refiner_cu_seqlens,
                refiner_max_seqlen=refiner_max_seqlen,
                seq_len=seq_len,
                device=device,
                local_span=local_span,
                num_requests=num_requests,
            )
        # Same arithmetic as the upstream method, with the plan's fixed
        # selections in place of boolean-mask indexing.
        x_rows = x.view(-1, x.shape[-1]).index_select(0, plan.img_global_pos).to(torch.float32)
        video_embed, _ = self.video_patch_proj(x_rows)
        audio_rows = audio_x.view(-1, audio_x.shape[-1]).index_select(0, plan.audio_global_pos).to(torch.float32)
        audio_embed, _ = self.audio_patch_proj(audio_rows)

        text_rows = text_embeddings_selected.to(device=device, dtype=torch.bfloat16)
        text_embed, _ = self.condition_proj(text_rows)
        text_embed = self.token_refiner(
            text_embed,
            cu_seqlens=refiner_cu_seqlens,
            max_seqlen=refiner_max_seqlen,
            num_requests=num_requests,
        )
        if plan.text_local_indices is not None:
            text_embed = text_embed.index_select(0, plan.text_local_indices)

        embeddings = torch.zeros((local_len, self.hidden_size), device=device, dtype=torch.bfloat16)
        embeddings.index_add_(0, plan.text_local_pos, text_embed.to(torch.bfloat16)[: plan.text_local_pos.shape[0]])
        embeddings.index_add_(0, plan.img_local_pos, video_embed.to(torch.bfloat16)[: plan.img_local_pos.shape[0]])
        embeddings.index_add_(0, plan.audio_local_pos, audio_embed.to(torch.bfloat16)[: plan.audio_local_pos.shape[0]])
        t_emb = self.time_embedder(unique_timesteps)
        return embeddings, t_emb


EntryClass = TaoMateH3DiTModel

__all__ = ["LocalEmbedPlan", "TaoMateH3DiTModel", "local_embed_plan"]
