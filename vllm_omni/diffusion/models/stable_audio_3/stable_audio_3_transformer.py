# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stable Audio 3 DiT (Diffusion Transformer) for vLLM-Omni.

PORT_FROM: stable_audio_3/models/dit.py (642 lines)
           stable_audio_3/models/transformer.py (1272 lines — port only used pieces)
           stable_audio_3/models/blocks.py (84 lines — FourierFeatures helpers)

Two-stage adaptation:

  Stage 1 (THIS PR): port upstream's Attention class verbatim so the model
                     runs correctly. Use raw torch / F.scaled_dot_product_attention.
  Stage 2 (LATER):  replace upstream's Attention with vllm_omni.diffusion.
                     attention.layer.Attention so we get FA2 + sequence parallel
                     + per-role backend selection. Also wire ColumnParallelLinear
                     etc. for tensor parallelism.

Stage 1 keeps vllm-omni adaptation minimal: only `od_config` + `load_weights()`
+ class attrs for layerwise offload / repeated blocks. Sequence parallel /
HSDP / TP added in follow-up PRs.

The DiT architecture (per model_config.json for `medium`):
  - patch_size:   1 (no patching beyond the autoencoder's PatchedPretransform)
  - depth:        ~24 layers
  - embed_dim:    ~1024 (1.4B params total)
  - num_heads:    ~16
  - cond_token_dim: 768 (T5Gemma)
  - global_cond_dim: depends on NumberConditioner output
  - diffusion_objective: "v"
  - transformer_type: "continuous_transformer"
  - global_cond_type: "prepend" or "adaLN"
  - timestep_cond_type: "global"
  - Conformer / sliding_window / rotary_pos_emb: TBD from config

(Exact values come from the downloaded model_config.json — these are
the kwargs that get passed via factory.py's diffusion_model_config.)
"""

from __future__ import annotations

import math
import typing as tp
from collections.abc import Iterable
from typing import ClassVar

import torch
from torch import nn
from torch.nn import functional as F
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.data import OmniDiffusionConfig


# ---------------------------------------------------------------------------
# Fourier features (PORT_FROM: models/blocks.py:42-84)
# ---------------------------------------------------------------------------


class FourierFeatures(nn.Module):
    """PORT_FROM: blocks.py:42-50"""

    def __init__(self, in_features: int, out_features: int, std: float = 16.0) -> None:
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer("weight", torch.randn(out_features // 2, in_features) * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PORT_FROM: blocks.py:47-50
        raise NotImplementedError


class ExpoFourierFeatures(nn.Module):
    """PORT_FROM: blocks.py:52-84"""

    def __init__(self, dim: int, min_freq: float = 0.5, max_freq: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.min_freq = min_freq
        self.max_freq = max_freq

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Norms / position embeddings (PORT_FROM: transformer.py)
# Only the variants actually used by SA3 medium config are needed.
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """PORT_FROM: transformer.py:392-410"""

    def __init__(self, dim: int, fix_scale: bool = False, force_fp32: bool = False, eps: float = 1e-5) -> None:
        super().__init__()
        # PORT_FROM
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LayerNorm(nn.Module):
    """PORT_FROM: transformer.py:363-390"""

    def __init__(self, dim: int, bias: bool = False, fix_scale: bool = False, force_fp32: bool = False, eps: float = 1e-5) -> None:
        super().__init__()
        # PORT_FROM
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RotaryEmbedding(nn.Module):
    """PORT_FROM: transformer.py:239-323"""

    def __init__(self, dim: int, *args, **kwargs) -> None:
        super().__init__()
        # PORT_FROM
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Attention (PORT_FROM: transformer.py:523-820 — large class)
#
# Stage 1: port the upstream Attention verbatim (uses F.scaled_dot_product_attention).
# Stage 2: replace with vllm_omni.diffusion.attention.layer.Attention.
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """SA3 attention block — port upstream as-is for Stage 1.

    PORT_FROM: transformer.py:523-820
    """

    def __init__(
        self,
        dim: int,
        dim_heads: int = 64,
        dim_context: int | None = None,
        causal: bool = False,
        zero_init_output: bool = True,
        qk_norm: bool = False,
        natten_kernel_size: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # PORT_FROM: transformer.py:523-700 (Attention.__init__)
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        rotary_pos_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # PORT_FROM: transformer.py:700-820 (Attention.forward)
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Feed-forward + Transformer block (PORT_FROM: transformer.py)
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    """PORT_FROM: transformer.py:453-521"""

    def __init__(self, dim: int, dim_out: int | None = None, mult: int = 4, **kwargs) -> None:
        super().__init__()
        # PORT_FROM
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerBlock(nn.Module):
    """One DiT block: self-attn + cross-attn + FFN with adaptive normalization.

    PORT_FROM: transformer.py:859-1068
    """

    def __init__(
        self,
        dim: int,
        dim_heads: int = 64,
        cross_attend: bool = False,
        dim_context: int | None = None,
        global_cond_dim: int | None = None,
        causal: bool = False,
        zero_init_branch_outputs: bool = True,
        conformer: bool = False,
        layer_ix: int = 0,
        remove_norms: bool = False,
        attn_kwargs: dict | None = None,
        ff_kwargs: dict | None = None,
        norm_kwargs: dict | None = None,
        local_add_cond_dim: int | None = None,
        modular_local_cond_configs: list | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # PORT_FROM: transformer.py:859-960 (TransformerBlock.__init__)
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        global_cond: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        rotary_pos_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        local_add_cond: torch.Tensor | None = None,
        modular_local_cond: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # PORT_FROM: transformer.py:960-1068 (TransformerBlock.forward)
        raise NotImplementedError


# ---------------------------------------------------------------------------
# ContinuousTransformer (PORT_FROM: transformer.py:1070-1272)
# Stack of TransformerBlocks. The DiT's `transformer_type=continuous_transformer`
# selects this one.
# ---------------------------------------------------------------------------


class ContinuousTransformer(nn.Module):
    """Stack of TransformerBlocks operating on continuous (audio-latent) tokens.

    PORT_FROM: transformer.py:1070-1272
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        *,
        dim_in: int | None = None,
        dim_out: int | None = None,
        dim_heads: int = 64,
        cross_attend: bool = False,
        cond_token_dim: int | None = None,
        final_cross_attn_ix: int = -1,
        global_cond_dim: int | None = None,
        local_add_cond_dim: int | None = None,
        modular_local_cond_configs: list | None = None,
        causal: bool = False,
        rotary_pos_emb: bool = True,
        cross_attn_rotary_pos_emb: bool = False,
        zero_init_branch_outputs: bool = True,
        conformer: bool = False,
        use_sinusoidal_emb: bool = False,
        use_abs_pos_emb: bool = False,
        abs_pos_emb_max_length: int = 10000,
        num_memory_tokens: int = 0,
        sliding_window: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.causal = causal
        self.layers = nn.ModuleList()

        self.project_in = nn.Linear(dim_in, dim, bias=False) if dim_in is not None else nn.Identity()
        self.project_out = nn.Linear(dim, dim_out, bias=False) if dim_out is not None else nn.Identity()

        if rotary_pos_emb:
            self.rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32))

        # PORT_FROM: transformer.py:1098-1272 — instantiate `depth` blocks +
        # absolute/sinusoidal position embeddings + memory tokens.
        # TODO(stable-audio-3): port body.
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        global_cond: torch.Tensor | None = None,
        local_add_cond: torch.Tensor | None = None,
        modular_local_cond: dict | None = None,
        prepend_embeds: torch.Tensor | None = None,
        prepend_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # PORT_FROM: transformer.py ContinuousTransformer.forward
        raise NotImplementedError


# ---------------------------------------------------------------------------
# DiffusionTransformer — top-level DiT
# PORT_FROM: dit.py:13-345 (__init__) + dit.py:345-642 (forward)
# ---------------------------------------------------------------------------


class DiffusionTransformer(nn.Module):
    """Stable Audio 3 top-level DiT. Built around ContinuousTransformer.

    vLLM-omni adaptation:
      - Adds od_config to __init__
      - Adds class attrs for layerwise offload and torch.compile
      - Adds load_weights() with weight name remapping
      - (Later) adds _sp_plan / _hsdp_shard_conditions / TP linear layers
    """

    # vLLM-omni class attrs
    _repeated_blocks: ClassVar[list[str]] = ["TransformerBlock"]
    _layerwise_offload_blocks_attr: ClassVar[str] = "transformer.layers"

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig | None = None,
        io_channels: int = 32,
        patch_size: int = 1,
        embed_dim: int = 768,
        cond_token_dim: int = 0,
        project_cond_tokens: bool = True,
        global_cond_dim: int = 0,
        project_global_cond: bool = True,
        input_concat_dim: int = 0,
        prepend_cond_dim: int = 0,
        depth: int = 12,
        num_heads: int = 8,
        transformer_type: tp.Literal["continuous_transformer", "mm_transformer"] = "continuous_transformer",
        global_cond_type: tp.Literal["prepend", "adaLN"] = "prepend",
        timestep_cond_type: tp.Literal["global", "input_concat"] = "global",
        timestep_embed_dim: int | None = None,
        diffusion_objective: tp.Literal["v", "rectified_flow", "rf_denoiser"] = "v",
        timestep_features_type: tp.Literal["learned", "expo"] = "learned",
        timestep_features_dim: int = 256,
        timestep_features_logsnr: bool = False,
        modular_local_cond_configs: list | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.od_config = od_config
        self.cond_token_dim = cond_token_dim
        self.diffusion_objective = diffusion_objective
        self.timestep_cond_type = timestep_cond_type
        self.timestep_features_logsnr = timestep_features_logsnr
        self.patch_size = patch_size

        # PORT_FROM: dit.py:47-345 — full __init__ body.
        # Roughly:
        #   - timestep_features (Fourier or Expo)
        #   - to_timestep_embed MLP
        #   - to_cond_embed (project text conditioning)
        #   - to_global_embed (project global conditioning)
        #   - input projection
        #   - transformer = ContinuousTransformer(...)
        #   - output projection
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cross_attn_cond: torch.Tensor | None = None,
        cross_attn_cond_mask: torch.Tensor | None = None,
        negative_cross_attn_cond: torch.Tensor | None = None,
        negative_cross_attn_mask: torch.Tensor | None = None,
        input_concat_cond: torch.Tensor | None = None,
        prepend_cond: torch.Tensor | None = None,
        prepend_cond_mask: torch.Tensor | None = None,
        cfg_scale: float = 1.0,
        cfg_dropout_prob: float = 0.0,
        scale_phi: float = 0.0,
        global_embed: torch.Tensor | None = None,
        local_add_cond: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # PORT_FROM: dit.py:345-642 — full forward including CFG batched eval
        raise NotImplementedError

    # ---- weight loading ----

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Pattern 2 (BAGEL-style): standard loader + custom name remap.

        Upstream weight names (after stripping "diffusion." prefix in SA3
        checkpoints) match this class's named_parameters() reasonably well
        IF we keep the same submodule attribute names. We may still need to
        handle:
          - upstream's torch.nn.utils.parametrize wrapping of LoRA-enabled
            layers (which inserts .parametrizations. in the name)
          - any QKV fusion done by future TP work

        TODO(stable-audio-3): build the actual remap table after diffing a
        real checkpoint against `dict(self.named_parameters()).keys()`.
        """
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, tensor in weights:
            mapped = self._remap_weight_name(name)
            if mapped is None:
                continue
            if mapped in params:
                default_weight_loader(params[mapped], tensor)
                loaded.add(mapped)
        return loaded

    @staticmethod
    def _remap_weight_name(name: str) -> str | None:
        """Map upstream weight name → vllm-omni param name. Returns None to skip."""
        # TODO(stable-audio-3): inspect a real checkpoint and fill this.
        # Common patterns:
        #   - Strip prefix "diffusion." or "model." that upstream adds
        #   - Skip "conditioner.*" weights (loaded separately, not in DiT)
        #   - Skip "pretransform.*" / "autoencoder.*" weights
        if name.startswith(("conditioner.", "pretransform.", "autoencoder.")):
            return None
        return name


# Alias used by older scaffold imports
StableAudio3DiTModel = DiffusionTransformer
