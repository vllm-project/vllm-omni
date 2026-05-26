# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stable Audio 3 DiT (Diffusion Transformer) for vLLM-Omni.

Ported from: https://github.com/Stability-AI/stable-audio-3 (MIT)

Architecture (per Stability-AI release + issue #3787):
  - 1.4B param DiT (Medium variant)
  - Stereo 44.1kHz audio, 256-dim SAME latents
  - Variable-length generation: sample_size adapts to requested duration
  - Requires Flash Attention 2 (Medium variant)
  - LoRA adapters stackable and runtime-adjustable

PORT STATUS: skeleton only — implement by porting block-by-block from upstream
and adapting to vllm-omni primitives (see TODOs below).

Adaptation checklist (per add-diffusion-model SKILL.md A3):
  [ ] Remove diffusers mixins (ModelMixin, ConfigMixin, AttentionModuleMixin)
  [ ] Replace attention with vllm_omni.diffusion.attention.layer.Attention
      QKV shape: [B, seq, heads, head_dim]
  [ ] Add od_config: OmniDiffusionConfig | None = None to __init__
  [ ] Add load_weights() method mapping upstream weight names → vllm-omni names
  [ ] Add _repeated_blocks, _layerwise_offload_blocks_attr
  [ ] (later) Add _sp_plan for sequence parallelism (Phase 10b)
  [ ] (later) Add _hsdp_shard_conditions for HSDP (Phase 10d)
  [ ] (later) Replace nn.Linear → ColumnParallelLinear/RowParallelLinear (TP, Phase 10a)
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

import torch
from torch import nn
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.data import OmniDiffusionConfig


# ---------------------------------------------------------------------------
# USER DECISION #2 — Variable-length latent sizing
# ---------------------------------------------------------------------------
# Per issue #3787: "The model sizes latents to the requested duration instead
# of a fixed window, which affects how the scheduler batches requests."
#
# Two strategies for handling this:
#
# Strategy A — Resize at pipeline level (simple, recommended for v1):
#   The transformer treats `sample_size` (token count along time axis) as a
#   runtime parameter, not a config constant. The pipeline computes it from
#   `audio_end_in_s - audio_start_in_s` and passes that latent shape directly.
#   Scheduler.timesteps stays fixed.
#
# Strategy B — Adapt timestep schedule to duration (better quality long-form):
#   Longer audio gets more denoising steps (or different sigma schedule).
#   Requires changes to scheduler_step. Defer to v2.
#
# This scaffold assumes Strategy A. Switch to B by adding a hook that
# the pipeline can call: `transformer.suggest_timesteps(duration_s)`.
# ---------------------------------------------------------------------------


# Helper from upstream's model.py — rotary embedding application.
# TODO(stable-audio-3): port apply_rotary_emb_stable_audio_3 from upstream.
def apply_rotary_emb_stable_audio_3(
    x: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embeddings to Q/K. Same axis layout as SA Open 1.0."""
    raise NotImplementedError


class StableAudio3SchedulerWrapper(nn.Module):
    """Wraps the underlying scheduler with SA3-specific noise sampling.

    Mirrors `StableAudioSchedulerWrapper` from SA Open 1.0. SA3 may ship its
    own scheduler (per upstream); if so, wrap it here so the pipeline can use
    the standard `step()` / `set_timesteps()` / `scale_model_input()` API.
    """

    def __init__(self, scheduler) -> None:  # noqa: ANN001 - upstream type
        super().__init__()
        self.scheduler = scheduler
        # Expose attributes the pipeline reads directly:
        self.init_noise_sigma = getattr(scheduler, "init_noise_sigma", 1.0)
        self.timesteps = getattr(scheduler, "timesteps", None)

    def set_timesteps(self, *args, **kwargs) -> None:
        self.scheduler.set_timesteps(*args, **kwargs)
        self.timesteps = self.scheduler.timesteps

    def scale_model_input(self, *args, **kwargs):
        return self.scheduler.scale_model_input(*args, **kwargs)

    def step(self, *args, **kwargs):
        return self.scheduler.step(*args, **kwargs)


# ---------------------------------------------------------------------------
# DiT Blocks — port from upstream model.py
# ---------------------------------------------------------------------------


class StableAudio3SelfAttention(nn.Module):
    """Self-attention with rotary embeddings, using vllm-omni's Attention layer."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        # TODO(stable-audio-3): port from upstream. Use vllm_omni.diffusion.attention.layer.Attention.
        # Skeleton sketch:
        #
        #   from vllm_omni.diffusion.attention.layer import Attention
        #   self.to_qkv = nn.Linear(dim, dim * 3)
        #   self.attn = Attention(num_heads=num_heads, head_size=dim // num_heads)
        #   self.to_out = nn.Linear(dim, dim)
        raise NotImplementedError


class StableAudio3CrossAttention(nn.Module):
    """Cross-attention conditioning audio latents on text/duration embeds."""

    def __init__(self, dim: int, num_heads: int, ctx_dim: int) -> None:
        super().__init__()
        # TODO(stable-audio-3): port
        raise NotImplementedError


class StableAudio3DiTBlock(nn.Module):
    """One DiT block: self-attn → cross-attn → FFN with residuals + AdaLN.

    Marked as repeated for `_repeated_blocks` so the loader handles per-block
    weight loading correctly.
    """

    def __init__(self, dim: int, num_heads: int, ctx_dim: int) -> None:
        super().__init__()
        # TODO(stable-audio-3): port
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Top-level DiT
# ---------------------------------------------------------------------------


class StableAudio3DiTModel(nn.Module):
    """Stable Audio 3 DiT — top-level denoiser.

    Inputs (per upstream `forward`):
      - hidden_states: [B, latent_channels=256, T_latent] noisy latents
      - timestep: [B] diffusion timestep
      - encoder_hidden_states: [B, S_text, D_text] projected text + duration conditioning
      - global_hidden_states: [B, ..., D_global] duration-only conditioning (for AdaLN)
      - rotary_embedding: (cos, sin) tuple

    Output:
      - noise prediction: [B, 256, T_latent]
    """

    # Class attrs picked up by the diffusion engine / model loader:
    _repeated_blocks: ClassVar[list[str]] = ["StableAudio3DiTBlock"]
    _layerwise_offload_blocks_attr: ClassVar[str] = "transformer_blocks"

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig | None = None,
        # ---- architecture hyperparams (TODO: read from upstream config) ----
        in_channels: int = 256,
        out_channels: int = 256,
        attention_head_dim: int = 64,
        num_layers: int = 24,  # TODO(stable-audio-3): confirm Medium variant depth
        num_attention_heads: int = 24,  # TODO(stable-audio-3): confirm
        ffn_mult: int = 4,
        text_embed_dim: int = 1024,  # depends on text encoder choice (USER DECISION #3)
        global_embed_dim: int = 1024,
        sample_size: int | None = None,  # variable-length — see USER DECISION #2
    ) -> None:
        super().__init__()
        self.od_config = od_config

        # Diffusers-compatible config object — pipeline reads
        # `self.transformer.config.attention_head_dim`, etc.
        class _Cfg:
            pass

        self.config = _Cfg()
        self.config.in_channels = in_channels
        self.config.out_channels = out_channels
        self.config.attention_head_dim = attention_head_dim
        self.config.num_layers = num_layers
        self.config.num_attention_heads = num_attention_heads
        self.config.text_embed_dim = text_embed_dim
        self.config.global_embed_dim = global_embed_dim
        self.config.sample_size = sample_size  # None signals variable-length

        # TODO(stable-audio-3): instantiate:
        #   - input projection from [B, 256, T] → [B, T, dim]
        #   - timestep embedding (likely GaussianFourierProjection like SA Open)
        #   - global conditioning embedder (duration)
        #   - text/global projection-in
        #   - self.transformer_blocks = nn.ModuleList([StableAudio3DiTBlock(...) for _ in range(num_layers)])
        #   - output norm + projection
        raise NotImplementedError(
            "StableAudio3DiTModel is a scaffold. Port the architecture from "
            "https://github.com/Stability-AI/stable-audio-3 (MIT) block by block, "
            "replacing each torch attention with vllm_omni.diffusion.attention.layer.Attention. "
            "See the file header for the full adaptation checklist."
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        global_hidden_states: torch.Tensor,
        rotary_embedding: tuple[torch.Tensor, torch.Tensor],
        return_dict: bool = True,
    ):
        # TODO(stable-audio-3): port forward pass
        raise NotImplementedError

    # ------------------------------------------------------------------ weights

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Pattern 2 (BAGEL-style): standard loader + custom name remapping.

        Upstream Stability-AI repo stores weights with names like
        `model.transformer_blocks.<i>.attn.to_q.weight`. vllm-omni's standard
        loader expects names that match `dict(self.named_parameters())` keys.

        TODO(stable-audio-3): fill in the name-mapping table below by diffing
        a checkpoint against `dict(self.named_parameters()).keys()`.
        """
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, tensor in weights:
            mapped = self._remap_weight_name(name)
            if mapped is None:
                continue  # silently skip non-DiT weights (handled by other components)
            if mapped in params:
                default_weight_loader(params[mapped], tensor)
                loaded.add(mapped)
        return loaded

    @staticmethod
    def _remap_weight_name(name: str) -> str | None:
        """Map upstream weight name → vllm-omni parameter name.

        Returns None if `name` doesn't belong to the DiT (e.g. VAE or text
        encoder weights bundled in the same checkpoint).
        """
        # TODO(stable-audio-3): build this table after inspecting a real checkpoint.
        # Common patterns to handle:
        #   1. Strip a leading prefix like "model." or "diffusion_model."
        #   2. Map upstream "to_qkv" → vllm-omni "to_qkv" (or split q/k/v)
        #   3. Drop entries belonging to the VAE / text encoder
        # Example:
        #     if name.startswith("vae.") or name.startswith("text_encoder."):
        #         return None
        #     return name.removeprefix("model.")
        return name
