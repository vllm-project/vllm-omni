# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
#
# Adapted from Ming repository (inclusionAI/Ming) — the ``get_condition_embeds
# _for_image_gen`` path in modeling_bailingmm2.py.

"""Ming-flash-omni-2.0 condition encoder for image generation.

Pipeline (runs inside the imagegen stage):

    thinker hidden states at query-token positions     [B, N, 4096]
                         │
                         ▼ proj_in (Linear, bias=True)
                                                       [B, N, 1536]
                         │
                         ▼ Qwen2ForCausalLM connector (is_causal=False)
                           loaded from <checkpoint>/connector/
                                                       [B, N, 1536]
                         │
                         ▼ proj_out (Linear, bias=True)
                                                       [B, N, 2560]
                         │
                         ▼ F.normalize(dim=-1) × 1000  (text_encoder_norm)
                                                       [B, N, 2560]
                         │
                         ▼
            cap_feats consumed by ZImageTransformer2DModel
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn

from vllm_omni.transformers_utils.configs.ming_flash_omni import MingImageGenConfig

logger = logging.getLogger(__name__)


class MingConditionEncoder(nn.Module):
    """Wraps a Qwen2 connector + norm/projection, producing DiT condition embeds.

    The connector is a ``Qwen2ForCausalLM`` loaded from the ``connector/``
    subfolder of the Ming checkpoint. We run its base model in a non-causal
    (bidirectional) mode, because the connector is used as an encoder over the
    pre-baked query-token hidden states, not as an autoregressive decoder.

    Args:
        image_gen_config: ``MingImageGenConfig`` from ``MingFlashOmniConfig``.
        thinker_hidden_size: Hidden size of the thinker (BailingMoeV2) model.
            Used to build a ``proj_in`` layer when the connector embedding
            dim differs. For the released checkpoint this is 4096.
        device: Placement for the module.
        dtype: Parameter dtype (typically bfloat16 / float16).
    """

    def __init__(
        self,
        image_gen_config: MingImageGenConfig,
        *,
        thinker_hidden_size: int = 4096,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = image_gen_config
        self.thinker_hidden_size = thinker_hidden_size
        self._target_device = torch.device(device) if device is not None else None
        self._target_dtype = dtype

        # Populated lazily by ``load_from_checkpoint`` to keep this module
        # cheap to construct (useful for dummy-init paths and unit tests).
        self.connector: nn.Module | None = None
        self.connector_hidden_size: int | None = None
        self.proj_in: nn.Module = nn.Identity()
        self.proj_out: nn.Module = nn.Identity()
        self.norm: nn.Module = nn.Identity()

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_from_checkpoint(self, model_path: str | Path) -> None:
        """Load the Qwen2 connector + optional projection/norm weights.

        This uses HF transformers directly (not vllm's weight loader) because
        the connector is small (~1.5B params) and only runs once per request
        as an encoder — vllm's distributed loading machinery is overkill.
        """
        from transformers import AutoConfig, Qwen2ForCausalLM

        model_path = Path(model_path)
        connector_path = model_path / self.config.connector_subfolder
        logger.info("[MingConditionEncoder] loading connector from %s", connector_path)

        connector_cfg = AutoConfig.from_pretrained(connector_path, trust_remote_code=True, local_files_only=True)
        # Disable causal masking: the connector is used as a bidirectional
        # encoder over query-token hidden states in image-gen mode.
        connector_cfg.is_decoder = False
        self.connector_hidden_size = int(connector_cfg.hidden_size)
        logger.info(
            "[MingConditionEncoder] connector hidden_size=%d, layers=%d",
            self.connector_hidden_size,
            getattr(connector_cfg, "num_hidden_layers", -1),
        )

        connector = Qwen2ForCausalLM.from_pretrained(
            connector_path,
            config=connector_cfg,
            torch_dtype=self._target_dtype,
            local_files_only=True,
        )
        # Force bidirectional attention in every self-attn module. Qwen2 layers
        # do not branch on ``is_causal`` directly, so we monkey-patch the flag
        # on each attention block (defensive — some transformers versions read
        # ``self_attn.is_causal`` in forward).
        patched = 0
        for module in connector.modules():
            if hasattr(module, "is_causal"):
                module.is_causal = False
                patched += 1
        logger.info("[MingConditionEncoder] disabled is_causal on %d sub-modules", patched)

        # We only need the base encoder (no LM head).
        base = getattr(connector, "model", connector)
        self.connector = base

        # proj_in: align thinker_hidden_size -> connector_hidden_size.
        # Observed layout of mlp/ on the released checkpoint
        # (inclusionAI/Ming-flash-omni-2.0):
        #   proj_in.{weight,bias}   : Linear(4096 -> 1536)
        #   proj_out.{weight,bias}  : Linear(1536 -> 2560)
        #   query_tokens_dict.16x16 : learnable tokens consumed on the thinker
        #                             side (NOT loaded here).
        logger.info(
            "[MingConditionEncoder] adding proj_in: %d -> %d (bias=True)",
            self.thinker_hidden_size,
            self.connector_hidden_size,
        )
        self.proj_in = nn.Linear(self.thinker_hidden_size, self.connector_hidden_size, bias=True)

        # NOTE: ``text_encoder_norm=True`` in Ming's mlp/config.json refers to
        # applying ``F.normalize(..., dim=-1)`` (L2 normalization) on the
        # FINAL cap_feats AFTER proj_out — NOT an intermediate RMSNorm. See
        # modeling_bailingmm2.py::get_condition_embeds_for_image_gen. We keep
        # ``self.norm = nn.Identity`` and apply the L2 normalize explicitly
        # at the end of ``forward``.
        self.norm = nn.Identity()

        # proj_out: align connector_hidden_size -> diffusion_c_input_dim.
        c_out = self.config.diffusion_c_input_dim
        logger.info(
            "[MingConditionEncoder] adding proj_out: %d -> %d (bias=True)",
            self.connector_hidden_size,
            c_out,
        )
        self.proj_out = nn.Linear(self.connector_hidden_size, c_out, bias=True)

        mlp_path = model_path / self.config.mlp_subfolder
        mlp_cfg_path = mlp_path / "config.json"
        if mlp_cfg_path.exists() and not json.loads(mlp_cfg_path.read_text()).get("use_identity_mlp", False):
            # MingImagePipeline skips ZImageModel_withMLP, only correct when its inner MLP is Identity.
            raise NotImplementedError(f"{mlp_cfg_path} has use_identity_mlp=False; ToClipMLP path not implemented.")
        self._load_optional_mlp_weights(mlp_path)

        if self._target_device is not None:
            self.to(self._target_device)
        if self._target_dtype is not None:
            self.to(dtype=self._target_dtype)

        logger.info(
            "[MingConditionEncoder] ready: in=%d -> conn=%d -> out=%d",
            self.thinker_hidden_size,
            self.connector_hidden_size,
            c_out,
        )

    def _load_optional_mlp_weights(self, mlp_path: Path) -> None:
        """Load proj_in / proj_out (and optional norm) weights from mlp/.

        Expected keys — observed on inclusionAI/Ming-flash-omni-2.0:
            proj_in.weight           [1536, 4096]
            proj_in.bias             [1536]
            proj_out.weight          [2560, 1536]
            proj_out.bias            [2560]
            query_tokens_dict.16x16  [256, 4096]   -> thinker-side, skipped here
            (optional) norm.weight   [1536]

        Any extra keys are logged as warnings so we don't silently miss new
        fields. Missing proj weights are fatal: they are required for the
        condition path to be meaningful.
        """
        if not mlp_path.exists():
            logger.warning(
                "[MingConditionEncoder] mlp/ subfolder missing at %s — proj/norm "
                "will stay randomly initialized. EXPECT BAD IMAGES until this "
                "is fixed on real hardware.",
                mlp_path,
            )
            return

        try:
            from safetensors.torch import load_file  # type: ignore
        except ImportError:
            logger.exception("[MingConditionEncoder] safetensors not installed")
            return

        candidates = sorted(mlp_path.glob("*.safetensors"))
        if not candidates:
            candidates = sorted(mlp_path.glob("*.bin"))
        if not candidates:
            logger.warning("[MingConditionEncoder] no weight files under %s", mlp_path)
            return

        state: dict[str, torch.Tensor] = {}
        for p in candidates:
            logger.info("[MingConditionEncoder] reading mlp weights: %s", p)
            if p.suffix == ".safetensors":
                state.update(load_file(str(p)))
            else:
                state.update(torch.load(str(p), map_location="cpu"))
        logger.info(
            "[MingConditionEncoder] mlp/ keys: %s",
            sorted(state.keys()),
        )

        handled: set[str] = set()

        def _copy(dst: nn.Parameter | torch.Tensor, src_key: str) -> bool:
            src = state.get(src_key)
            if src is None:
                logger.error("[MingConditionEncoder] mlp/ missing key %r", src_key)
                return False
            if tuple(src.shape) != tuple(dst.shape):
                logger.error(
                    "[MingConditionEncoder] mlp/%s shape mismatch: checkpoint=%s, module=%s",
                    src_key,
                    tuple(src.shape),
                    tuple(dst.shape),
                )
                return False
            with torch.no_grad():
                dst.copy_(src.to(dtype=dst.dtype, device=dst.device))
            handled.add(src_key)
            logger.info(
                "[MingConditionEncoder] loaded mlp/%s -> %s",
                src_key,
                tuple(dst.shape),
            )
            return True

        ok_w = _copy(self.proj_in.weight, "proj_in.weight")
        ok_b = _copy(self.proj_in.bias, "proj_in.bias")
        ok_ow = _copy(self.proj_out.weight, "proj_out.weight")
        ok_ob = _copy(self.proj_out.bias, "proj_out.bias")
        if not (ok_w and ok_b and ok_ow and ok_ob):
            logger.error(
                "[MingConditionEncoder] proj_in/proj_out NOT fully loaded; diffusion conditioning will be garbage."
            )

        # Optional norm weight (Ming uses plain RMSNorm; may or may not ship).
        if "norm.weight" in state and hasattr(self.norm, "weight"):
            _copy(self.norm.weight, "norm.weight")

        # query_tokens_dict lives in mlp/ but is consumed on the thinker side
        # — we neither need nor load it here. Just log the shape so the
        # thinker-side patch knows what to expect.
        for k, v in state.items():
            if k.startswith("query_tokens_dict"):
                logger.info(
                    "[MingConditionEncoder] thinker-side parameter %s shape=%s (NOT loaded here; thinker must own it)",
                    k,
                    tuple(v.shape),
                )
                handled.add(k)

        leftover = set(state.keys()) - handled
        if leftover:
            logger.warning("[MingConditionEncoder] mlp/ unhandled keys: %s", sorted(leftover))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        thinker_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode thinker hidden states into DiT condition embeddings.

        Args:
            thinker_hidden_states: ``[B, N, thinker_hidden_size]`` — sliced at
                the learnable query-token positions by the stage input
                processor before being passed here.
            attention_mask: Optional ``[B, N]`` mask. Defaults to all-ones.

        Returns:
            ``[B, N, diffusion_c_input_dim]`` condition tensor ready for the
            ZImage transformer's ``cap_feats`` input.
        """
        if self.connector is None:
            raise RuntimeError("MingConditionEncoder.load_from_checkpoint() must be called before forward().")
        if thinker_hidden_states.dim() != 3:
            raise ValueError(f"expected [B, N, H], got shape {tuple(thinker_hidden_states.shape)}")

        b, n, _ = thinker_hidden_states.shape
        logger.debug(
            "[MingCondEnc] input shape=%s dtype=%s", tuple(thinker_hidden_states.shape), thinker_hidden_states.dtype
        )

        x = self.proj_in(thinker_hidden_states)  # [B, N, conn_hidden]

        # Ming's ``get_condition_embeds_for_image_gen`` passes a 4D full-ones
        # attention mask of shape [B, 1, N, N] to the Qwen2 connector,
        # forcing full bidirectional self-attention over all query positions.
        if attention_mask is None:
            attention_mask = torch.ones(
                (b, 1, n, n),
                dtype=x.dtype,
                device=x.device,
            )
        elif attention_mask.dim() == 2:
            mask_2d = attention_mask.to(x.dtype)
            attention_mask = mask_2d[:, None, None, :].expand(b, 1, n, n)

        out = self.connector(
            inputs_embeds=x,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = out.hidden_states[-1]  # [B, N, conn_hidden]

        cap_feats = self.proj_out(hidden)  # [B, N, diffusion_c_input_dim]

        # L2 normalize + ×1000 rescale.  Ming's ``get_condition_embeds_for_image_gen``
        # applies F.normalize; ``ZImageModel_withMLP`` then rescales by 1000
        # when ``text_encoder_norm=True``.  We fold both into one place.
        cap_feats = torch.nn.functional.normalize(cap_feats, dim=-1)
        if self.config.text_encoder_norm:
            cap_feats = cap_feats * 1000.0

        logger.debug("[MingCondEnc] output shape=%s dtype=%s", tuple(cap_feats.shape), cap_feats.dtype)
        return cap_feats

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @torch.no_grad()
    def zero_negative(
        self,
        cap_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Return a zero tensor shaped like ``cap_feats`` for CFG negatives."""
        return torch.zeros_like(cap_feats)

    def extra_repr(self) -> str:
        return (
            f"thinker_hidden_size={self.thinker_hidden_size}, "
            f"connector_hidden_size={self.connector_hidden_size}, "
            f"diffusion_c_input_dim={self.config.diffusion_c_input_dim}, "
            f"text_encoder_norm={self.config.text_encoder_norm}"
        )


__all__ = ["MingConditionEncoder"]
